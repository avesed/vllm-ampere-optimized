"""SageAttention v1 (INT8-QK + fp16-PV) attention backend for Ampere (sm_80 / sm_86).

vLLM keeps INT8 attention research-state (closed PR #10532); Ampere has a real tuned kernel
(thu-ml/SageAttention v1). This backend un-gates it — the W4A8-style play applied to attention:
on the QK^T matmul, Q/K are quantized to INT8 tensor cores (~4x fp16); P (softmax) and PV stay
fp16 (P's dynamic range can't be int8, and Ampere has no fp8 TC for v2's fp8-PV). Prefill/TTFT
win for STANDARD full-attention transformers (Llama/Mistral/Qwen-dense/Gemma). Decode is unchanged.

Design — wrap + delegate (no fork of FA's complex forward):
  * SageAttentionImpl subclasses FlashAttentionImpl. It uses sageattn_varlen ONLY for a
    "pure fresh prefill" step (no cached prefix, no decode rows — so the packed K/V hold each
    sequence's full KV and no paged gather is needed). It still writes K/V to the paged cache
    (reshape_and_cache_flash) so later decode steps work. EVERYTHING else — decode, chunked
    prefill with a cached prefix, cascade, DCP, unsupported head sizes / quantized KV / alibi /
    sliding-window / softcap / non-DECODER — falls through to the parent FlashAttention forward.
  * The "pure fresh prefill" flag is computed ONCE per step in the builder from CPU metadata
    (zero GPU sync): common_prefix_len == 0 AND num_computed_tokens == 0 for every request.

Shipped default-OFF. Enable by registering against CUSTOM + VLLM_ATTENTION_BACKEND=CUSTOM
(see register_sage_attention() / the patch wiring). Accuracy gate (smooth_k) must be validated
per model class before relying on it.
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
from vllm.v1.attention.backends import flash_attn as _fa
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)

# SageAttention head-dim support. The varlen Triton path covers 64 / 96 / 128. The DENSE CUDA
# kernel (sageattn_qk_int8_pv_fp16_cuda, thu-ml PR #329) adds 256 — needed for hd256 hybrids
# (Qwen3.x full-attn layers). hd256 routes to the CUDA variant per-sequence: varlen Triton has
# no hd256, and the auto sageattn() dispatcher picks a Triton kernel that needs ~192KB smem and
# OOMs sm_86's 100KB limit, so we MUST call the CUDA variant explicitly.
_SAGE_HEAD_SIZES = (64, 96, 128, 256)

try:
    from sageattention import sageattn_varlen

    try:  # dense CUDA int8-QK kernel (provides hd256 on sm_80/86); may be absent on older builds
        from sageattention import sageattn_qk_int8_pv_fp16_cuda
    except Exception:
        sageattn_qk_int8_pv_fp16_cuda = None
    _HAS_SAGE = True
except Exception as e:  # pragma: no cover - import guard
    sageattn_varlen = None
    sageattn_qk_int8_pv_fp16_cuda = None
    _HAS_SAGE = False
    logger.warning("SageAttention not importable (%s); SAGE_ATTN backend will delegate to FA.", e)

# Reuse whatever symbols FlashAttention itself imports, so we never drift on their paths.
reshape_and_cache_flash = _fa.reshape_and_cache_flash
is_quantized_kv_cache = _fa.is_quantized_kv_cache


def _is_pure_fresh_prefill(common_prefix_len: int, m: CommonAttentionMetadata) -> bool:
    """True iff every request is a from-scratch full prefill (no cached prefix, no decode rows),
    so the packed key/value contain each sequence's complete KV. CPU-only, no GPU sync."""
    if common_prefix_len != 0:
        return False
    nct = getattr(m, "_num_computed_tokens_cpu", None)
    if nct is None:
        return False
    try:
        return int(nct.max().item()) == 0
    except Exception:
        return False


class SageAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        pure = _is_pure_fresh_prefill(common_prefix_len, common_attn_metadata)
        try:
            md.sage_pure_prefill = pure  # dataclass instance attr; read in forward (no sync)
        except Exception:
            pass
        return md


class SageAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sw = getattr(self, "sliding_window", None)
        no_sw = sw is None or tuple(sw) == (-1, -1)
        soft_cap = getattr(self, "logits_soft_cap", None)
        no_soft_cap = soft_cap is None or soft_cap == 0
        self._sage_ok = (
            _HAS_SAGE
            and self.head_size in _SAGE_HEAD_SIZES
            and getattr(self, "attn_type", AttentionType.DECODER) == AttentionType.DECODER
            and no_sw
            and getattr(self, "alibi_slopes", None) is None
            and no_soft_cap
            and getattr(self, "sinks", None) is None
        )
        if _HAS_SAGE and not self._sage_ok:
            logger.debug(
                "SAGE_ATTN: layer not eligible (head=%s sw=%s alibi=%s softcap=%s) -> FA fallback",
                self.head_size, sw, getattr(self, "alibi_slopes", None), soft_cap,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Delegate to FlashAttention unless this is an eligible pure-fresh-prefill step.
        if (
            not self._sage_ok
            or attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
            or getattr(attn_metadata, "use_cascade", False)
            or getattr(self, "dcp_world_size", 1) > 1
            or is_quantized_kv_cache(self.kv_cache_dtype)
            or not getattr(attn_metadata, "sage_pure_prefill", False)
            or (self.head_size == 256 and sageattn_qk_int8_pv_fp16_cuda is None)
        ):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

        n = attn_metadata.num_actual_tokens

        # Write K/V into the paged cache so subsequent decode steps read them (same as FA).
        key_cache, value_cache = kv_cache.unbind(1)
        reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            attn_metadata.slot_mapping, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

        # Pure fresh prefill: cu_seqlens_k == cu_seqlens_q, full KV is in the packed key/value.
        cu = attn_metadata.query_start_loc
        mql = attn_metadata.max_query_len

        if self.head_size == 256:
            # hd256 has no varlen Triton kernel — use the dense CUDA int8-QK variant per sequence.
            # query/key/value arrive packed/flattened as [num_tokens, H*head_dim]; reshape each
            # sequence to a [1, L, H, head_dim] NHD batch for the dense kernel (Q has num_heads,
            # K/V have num_kv_heads for GQA). One CPU sync on cu (prefill only).
            H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_size
            cu_list = cu.tolist()
            for i in range(len(cu_list) - 1):
                s, e = int(cu_list[i]), int(cu_list[i + 1])
                if e <= s:
                    continue
                q_seg = query[s:e].reshape(e - s, H, D).unsqueeze(0)
                k_seg = key[s:e].reshape(e - s, Hkv, D).unsqueeze(0)
                v_seg = value[s:e].reshape(e - s, Hkv, D).unsqueeze(0)
                o = sageattn_qk_int8_pv_fp16_cuda(
                    q_seg, k_seg, v_seg,
                    tensor_layout="NHD", is_causal=attn_metadata.causal,
                    sm_scale=self.scale, smooth_k=True, pv_accum_dtype="fp32",
                )
                output[s:e] = o.squeeze(0).reshape(output[s:e].shape).to(output.dtype)
            return output

        out = sageattn_varlen(
            query[:n], key[:n], value[:n],
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=mql, max_seqlen_k=mql,
            is_causal=attn_metadata.causal,
            sm_scale=self.scale,
            smooth_k=True,
        )
        output[:n] = out.reshape(n, -1).to(output.dtype)
        return output


class SageAttentionBackend(FlashAttentionBackend):
    """Inherits FA's metadata/KV-cache-shape/get_kv_cache_stride_order; swaps impl + builder."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SageAttentionImpl]:
        return SageAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[SageAttentionMetadataBuilder]:
        return SageAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # Sage only accelerates DECODER self-attn; other types delegate to FA inside the impl.
        return FlashAttentionBackend.supports_attn_type(attn_type)


_SAGE_CLASS_PATH = "vllm.v1.attention.backends.sage_attn.SageAttentionBackend"


def register_sage_attention() -> None:
    """Map AttentionBackendEnum.CUSTOM -> SageAttentionBackend. Call before engine init, then set
    VLLM_ATTENTION_BACKEND=CUSTOM. Idempotent / best-effort."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM, _SAGE_CLASS_PATH)(SageAttentionBackend)
