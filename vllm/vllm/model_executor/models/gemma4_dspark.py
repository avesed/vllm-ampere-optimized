# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 DSpark draft model for speculative decoding.

Mirrors the structure of qwen3_dflash.DFlashQwen3{Model,ForCausalLM} (the
DFlash cross-attention drafter: context K/V is pre-inserted from target
hidden states, queries are bonus + mask tokens) but is built from Gemma4
building blocks (scaling=1.0, q/k norm with weight, v norm without weight,
4 sandwich norms + layer_scalar, gelu-tanh MLP, sqrt(hidden) embed scale,
final-logit soft cap). DSpark adds a vanilla Markov head on top.
"""

import re
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.v1.attention.backend import AttentionType

from .gemma4 import Gemma4MLP
from .qwen3_dflash import DFlashAttention
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_DSPARK_VALID_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def _get_dspark_layer_types(config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return ("full_attention",) * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DSpark layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    invalid = set(layer_types) - _DSPARK_VALID_LAYER_TYPES
    if invalid:
        raise ValueError(f"Invalid DSpark layer_type(s): {sorted(invalid)}.")
    if "sliding_attention" in layer_types and not getattr(
        config, "sliding_window", None
    ):
        raise ValueError(
            "DSpark sliding_attention layers require `sliding_window` in config."
        )
    return tuple(layer_types)


def _resolve_rope_parameters(config, layer_type: str) -> dict:
    """Per-layer-type RoPE parameters, mirroring Gemma4Attention."""
    rope_parameters = config.rope_parameters
    if isinstance(rope_parameters, dict) and layer_type in rope_parameters:
        # Per-layer-type rope config (dict keyed by layer type).
        return dict(rope_parameters[layer_type])
    # Legacy / flat config format.
    resolved = dict(rope_parameters)
    if layer_type == "sliding_attention":
        resolved["rope_theta"] = getattr(config, "rope_local_base_freq", 10000.0)
    return resolved


class Gemma4DSparkAttention(nn.Module):
    """Gemma4 attention for DSpark speculative decoding.

    Context K/V are pre-inserted into the KV cache before the forward pass;
    this layer handles only the query tokens. Adapted from Gemma4Attention
    (scaling=1.0, q/k norm with weight, v norm without weight) and the
    DFlash cross-attention KV-precompute path.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        layer_type: str,
        use_k_eq_v: bool = False,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        attn_logits_soft_cap: float | None = None,
        sliding_window: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.config = config
        self.hidden_size = hidden_size
        self.use_k_eq_v = use_k_eq_v

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma4 uses scaling=1.0; q/k norms with learnable weight scale.
        self.scaling = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=_resolve_rope_parameters(config, layer_type),
            is_neox_style=True,
        )
        # head_dim 512 (full_attention) -> FA2 (cap 256) rejects it -> FLEX backend.
        # FLEX's hd512 kernel autotunes to BM64/BN128 -> ~200KB smem >> sm86's 99KB -> OOM.
        # Cap tiles to (16,16) -> 36864 B, fits with 2.75x margin. The draft block is q=7,
        # so tiny tiles cost ~nothing. Gate on head_dim>256: the FA2 (sliding/hd<=256) impl
        # takes no block_m/block_n kwargs and would raise TypeError.
        _flex_tile = {"block_m": 16, "block_n": 16} if self.head_dim > 256 else {}
        self.attn = DFlashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **_flex_tile,
        )
        # Gemma4 norm conventions: q/k norm have learnable weight,
        # v norm is pure normalization (no learnable scale).
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, has_weight=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Attention over query tokens; context K/V already in the cache.

        Numerically identical to Gemma4Attention's non-KV-shared forward
        (q/k norm with weight, v norm weightless, RoPE on q/k only). Under
        k_eq_v the V slot of qkv_proj holds the K weight, so V == K before
        norms. See Gemma4DSparkModel.precompute_and_store_context_kv.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)

        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_type: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        self.is_full_attention = layer_type == "full_attention"

        # Gemma4 uses different head dims for full vs sliding attention.
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        # k_eq_v full-attention layers reuse K as V (no v_proj).
        use_k_eq_v = self.is_full_attention and getattr(
            config, "attention_k_eq_v", False
        )
        if use_k_eq_v:
            num_kv_heads = getattr(
                config, "num_global_key_value_heads", config.num_key_value_heads
            )
        else:
            num_kv_heads = config.num_key_value_heads

        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.self_attn = Gemma4DSparkAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            layer_type=layer_type,
            use_k_eq_v=use_k_eq_v,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            sliding_window=sliding_window,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Gemma4 sandwich norms: input / post-attn / pre-ff / post-ff.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-layer scalar (loaded from checkpoint) applied to all text layers.
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Gemma4 residual pattern: residual folds into hidden_states inside
        # the layer (so the carried `residual` stays None across the stack).
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None


@support_torch_compile
class Gemma4DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = dict(getattr(self.config, "eagle_config", {}) or {})
        drafter_config.update(getattr(self.config, "dflash_config", {}) or {})

        if "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Gemma4 embedding scale = sqrt(hidden_size), cast to model dtype to
        # avoid mixed-precision drift from bf16 * fp32 across the stack.
        self.register_buffer(
            "normalizer",
            torch.tensor(
                self.config.hidden_size**0.5,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )

        self.layer_types = _get_dspark_layer_types(self.config)
        self.layers = nn.ModuleList(
            [
                Gemma4DSparkDecoderLayer(
                    config=self.config,
                    layer_type=self.layer_types[layer_idx],
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.sliding_attention_layer_names = {
            layer.self_attn.attn.layer_name
            for layer in self.layers
            if layer.layer_type == "sliding_attention"
        }
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K/V-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused GEMM
        for all layers at once. Also aliases the hidden_norm weight.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size].
        # Under k_eq_v the V slot of qkv_proj holds the K weight, so the
        # fused [K;V] stack is [k_proj; k_proj] for those layers.
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights (learnable), one [head_dim] tensor per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]
        # V-norm has no learnable weight (has_weight=False) => pure
        # normalization. Use an all-ones weight on the correct device/dtype so
        # ops.rms_norm reproduces the module's weightless normalization.
        self._v_norm_weights = [
            torch.ones_like(a.k_norm.weight.data) for a in layers_attn
        ]

        # RoPE parameters.
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must share RoPE params for DSpark precomputation"

        # Layer metadata.
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must share attn config for DSpark precomputation"

        # References to inner Attention layers for direct cache writes.
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute K/V for context states and write into each layer's cache.

        Context states are projected to K/V, normed (K-norm on the K half,
        V-norm on the V half), and have RoPE applied to K only. Mirrors the
        DFlash precompute, extended with Gemma4's V-norm.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DSpark buffer initialization was skipped. If dummy weights are "
                "not in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # [2, L, num_ctx, nkv, hd] contiguous; dim-0 indexing splits K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K and V ---
        all_k_normed = torch.empty_like(all_k)
        all_v_normed = torch.empty_like(all_v)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )
            ops.rms_norm(
                all_v_normed[i],
                all_v[i],
                self._v_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers (K only) ---
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        all_v_final = all_v_normed
        for i in range(L):
            attn = self._attn_layers[i]
            layer_slot_mapping = (
                context_slot_mapping[attn.layer_name]
                if isinstance(context_slot_mapping, Mapping)
                else context_slot_mapping
            )
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v_final[i],
                kv_cache,
                layer_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        # Gemma4 folds the residual inside each layer, so the final norm runs
        # without residual fusion.
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _maybe_duplicate_k_eq_v(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Fill the qkv_proj V slot for k_eq_v full-attention layers.

        The checkpoint stores only k_proj (no v_proj) for k_eq_v full layers,
        so the packed qkv_proj V slot would be empty. Mirror Gemma4ForCausalLM:
        duplicate k_proj into v_proj so V loads identical weights to K, which
        makes the precompute's fused [K;V] == [k_proj; k_proj] hold.
        """
        k_eq_v_layer_indices: set[int] = set()
        if getattr(self.config, "attention_k_eq_v", False):
            for idx, lt in enumerate(self.layer_types):
                if lt == "full_attention":
                    k_eq_v_layer_indices.add(idx)
        for name, weight in weights:
            if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                m = re.search(r"layers\.(\d+)\.", name)
                if m and int(m.group(1)) in k_eq_v_layer_indices:
                    yield name, weight
                    yield name.replace("k_proj", "v_proj"), weight.clone()
                    continue
            yield name, weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        # Include buffers (e.g. layer_scalar) so they can be loaded too.
        params_dict.update(dict(self.named_buffers()))
        loaded_params: set[str] = set()
        # Checkpoint uses local `layers.N` naming (no midlayer prefix).
        for name, loaded_weight in self._maybe_duplicate_k_eq_v(weights):
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Gemma4DSparkForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Gemma4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # Gemma4 final-logit soft cap (e.g. 30.0) applied in compute_logits.
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
        )
        # Gemma vocab == target vocab => no draft->target remap.
        self.draft_id_to_target_id = None

        # DSpark Markov head: rank-r token->logit bias re-coupling adjacent
        # draft-block positions. markov_rank=0 disables it (no params).
        dflash_cfg = getattr(self.config, "dflash_config", None) or {}
        self.markov_rank = int(dflash_cfg.get("markov_rank", 0) or 0)
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.markov_rank > 0:
            self.markov_w1 = nn.Embedding(target_vocab_size, self.markov_rank)
            self.markov_w2 = nn.Linear(self.markov_rank, target_vocab_size, bias=False)
        else:
            self.markov_w1 = None
            self.markov_w2 = None

    def markov_step_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        # rank-r token->logit bias for the previous draft token.
        return self.markov_w2(self.markov_w1(prev_token_ids.long()))

    def markov_sample_block(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Sequential per-position markov-bias sampling over the draft block.
        # base_logits is already soft-capped (compute_logits applied the cap);
        # the markov bias is added AFTER the soft cap. base_logits
        # [batch, block, vocab]; first_prev_token_ids [batch] = anchor token.
        block = base_logits.shape[1]
        prev = first_prev_token_ids.long()
        sampled = []
        for k in range(block):
            step_logits = base_logits[:, k, :] + self.markov_step_bias(prev)
            tok = step_logits.argmax(dim=-1)
            sampled.append(tok)
            prev = tok
        return torch.stack(sampled, dim=1)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Soft cap is applied by the LogitsProcessor. Gemma vocab == target
        # vocab, so there is no draft->target id remap.
        return self.logits_processor(self.lm_head, hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return self.model.sliding_attention_layer_names

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DSpark should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "confidence_head" in name:
                # Adaptive-verify head not modeled. Drop here so it never
                # reaches the delegated Gemma4DSparkModel.load_weights (which
                # would KeyError on the unknown name).
                continue
            if "d2t" in name:
                # Gemma vocab == target vocab => no draft->target remap.
                continue
            elif "markov" in name:
                # DSpark Markov head lives on the ForCausalLM (self.markov_w1/w2).
                name = name.replace("markov_head.", "")
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["draft_id_to_target_id"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if self.markov_rank <= 0:
            # DSpark Markov head weights present in ckpt but head disabled.
            skip_substrs.append("markov")
        # confidence_head drives adaptive-verify (not yet modeled); skip it.
        skip_substrs.append("confidence_head")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
