"""int8-QK FlashInfer prefill attention backend for Ampere (sm_80 / sm_86) — I-4b + chunked.

Sibling of patches/sage_attn/sage_attn.py. Same wrap+delegate design (SageAttentionImpl-style:
subclass FlashAttentionImpl, intercept the eligible prefill step(s), delegate everything else to
the parent FA forward). The ONE difference is the hd256 prefill compute: instead of calling thu-ml
SageAttention's CUDA kernel, this backend does on-the-fly per-token symmetric int8 quantization
(+ smooth_k) of Q/K, per-tensor int8 of V, and calls the validated int8-QK FlashInfer
`single_prefill_with_kv_cache` (scale_q/scale_k as fa2 additional tensors), keeping PV fp16.
See patches/flashinfer_int8/NOTES.md + i4_test.py for the kernel API/recipe.

CHUNKED PREFILL (cached-prefix steps) — the I-4b-paged extension:
  The original I-4b backend only fired on PURE-FRESH prefill (whole prompt in one step). With
  chunked prefill (enable_chunked_prefill + small max_num_batched_tokens) a long prompt is split
  into chunks; chunk-0 is fresh but chunks 1..N have a CACHED PREFIX (num_computed_tokens > 0).
  This backend now fires int8 on those cached-prefix chunks too:
    1. reshape_and_cache_flash writes the NEW chunk's K/V into the paged cache (as FA does).
    2. For each prefill request we read the FULL context K/V (= prefix + the just-written chunk,
       i.e. the first seq_len tokens) from the paged cache via the block_table.
    3. Per-token int8-quant Q (the new chunk) + the full K (+smooth_k), per-tensor int8 V.
    4. single_prefill_with_kv_cache(q[chunk], k[full], v[full], causal=True). qo_len <= kv_len +
       causal aligns the chunk's queries to the END of the kv (q token i attends to
       kv[0 .. kv_len-qo_len+i]) — exactly chunked-prefix semantics. VERIFIED in isolation
       (i4b_align_test.py: cos 1.00000 vs the last-C rows of the full prefill, fp16 AND int8).
  Pure-fresh chunks skip the gather (use the in-hand K/V — cheaper, bit-identical path).

Decode rows / non-hd256 layers / quantized-KV / cascade / DCP / everything unsupported -> FA
fallback (bit-faithful to the stock baseline). Default-OFF; enabled by the harness via the
registry override + VLLM_ENABLE_V1_MULTIPROCESSING=0 (single-card in-process), exactly like Step A.
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

# This backend only accelerates head_dim=256 (Qwen3.x full-attn) prefill via the int8-QK
# FlashInfer kernel. Other head dims fall through to FA (the FlashInfer int8 module is built
# per (head_dim) so we keep the PoC scoped to the real Qwen3.5 hd; extend the tuple to add more).
_INT8QK_HEAD_SIZES = (256,)

# Global fire counter (read by the harness). "calls" = total int8 single_prefill calls (==
# #full-attn layers * #prefill-requests that used int8, summed over steps). "fresh"/"cached"
# split lets the harness PROVE int8 fired on cached-prefix chunks, not just chunk-0.
INT8QK_FIRE = {"calls": 0, "seqs": 0, "fresh": 0, "cached": 0}

try:
    import flashinfer  # noqa: F401

    _HAS_FI = True
except Exception as e:  # pragma: no cover - import guard
    flashinfer = None
    _HAS_FI = False
    logger.warning("flashinfer not importable (%s); INT8QK backend will delegate to FA.", e)

reshape_and_cache_flash = _fa.reshape_and_cache_flash
is_quantized_kv_cache = _fa.is_quantized_kv_cache


def _quant_qk_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quant. x: [L, Hx, D] -> (int8 [L,Hx,D], fp32 scale [L]).
    One scalar scale per token row (across heads*dim) — matches i4_test.py q_pertoken().
    Memory-lean: compute the scale from |x| in fp32 row-reduced (no full fp32 copy persists),
    then a SINGLE fused fp32 divide->round->clamp->int8. (numerics identical to i4_test)."""
    s = (x.abs().amax(dim=(1, 2)).float()) / 127.0        # [L] (reduce first -> tiny fp32)
    s = torch.clamp(s, min=1e-8)
    # one fp32 temporary (x/s), consumed immediately into int8
    xi = torch.clamp(torch.round(x.float() / s[:, None, None]), -127, 127).to(torch.int8)
    return xi, s.to(torch.float32)


class Int8QKMetadataBuilder(FlashAttentionMetadataBuilder):
    """Pass-through builder; the int8 vs FA decision is made per-request in forward() from the
    standard FA metadata (seq_lens / query_start_loc / block_table). No extra fields needed."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        return super().build(common_prefix_len, common_attn_metadata, fast_build)


class Int8QKAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sw = getattr(self, "sliding_window", None)
        no_sw = sw is None or tuple(sw) == (-1, -1)
        soft_cap = getattr(self, "logits_soft_cap", None)
        no_soft_cap = soft_cap is None or soft_cap == 0
        self._int8qk_ok = (
            _HAS_FI
            and self.head_size in _INT8QK_HEAD_SIZES
            and getattr(self, "attn_type", AttentionType.DECODER) == AttentionType.DECODER
            and no_sw
            and getattr(self, "alibi_slopes", None) is None
            and no_soft_cap
            and getattr(self, "sinks", None) is None
        )

    @staticmethod
    def _gather_one(cache: torch.Tensor, blk: torch.Tensor, n_blocks: int,
                    seq_len: int) -> torch.Tensor:
        """Gather the first `seq_len` logical tokens of ONE request's K or V from the paged cache.
        cache: [num_blocks, block_size, Hkv, D]; blk: [n_blocks] physical block ids. Returns a
        fresh contiguous [seq_len, Hkv, D] in the cache dtype. Caller frees it ASAP."""
        bs, Hkv, D = cache.shape[1], cache.shape[2], cache.shape[3]
        blocks = cache.index_select(0, blk)                          # [n_blocks, bs, Hkv, D]
        return blocks.reshape(n_blocks * bs, Hkv, D)[:seq_len]

    # Stream the per-token int8 quant over the sequence in slices so the fp32 working buffer is
    # bounded by SLICE rows (~one chunk), NOT the full context. At 128k the naive full-context
    # fp32 copy is ~1GB per K and per V (the single biggest int8 transient -> the OOM); streaming
    # caps it to ~SLICE/L_kv of that while the unavoidable int8 outputs (~268MB) are tiny. Numerics
    # are identical (per-token scale is a row reduction; smooth_k mean is computed full-context
    # first, then subtracted slice-wise).
    _QUANT_SLICE = 16384

    def _quant_k_smooth(self, k_full: torch.Tensor):
        """smooth_k (per-(head,channel) full-context mean) + per-token int8 quant of K, STREAMED."""
        L, Hkv, D = k_full.shape
        k_mean = k_full.float().mean(dim=0, keepdim=True)            # [1,Hkv,D]  (transient, freed)
        k_i8 = torch.empty((L, Hkv, D), dtype=torch.int8, device=k_full.device)
        sk = torch.empty((L,), dtype=torch.float32, device=k_full.device)
        for a in range(0, L, self._QUANT_SLICE):
            b = min(a + self._QUANT_SLICE, L)
            sl = k_full[a:b].float() - k_mean                        # [<=SLICE,Hkv,D] fp32 (bounded)
            s = torch.clamp(sl.abs().amax(dim=(1, 2)) / 127.0, min=1e-8)
            k_i8[a:b] = torch.clamp(torch.round(sl / s[:, None, None]), -127, 127).to(torch.int8)
            sk[a:b] = s
            del sl
        del k_mean
        return k_i8, sk

    def _quant_v_pertensor(self, v_full: torch.Tensor):
        """Per-tensor int8 quant of V (uniform scale; cosine-invariant, mag restored by *sv), STREAMED."""
        L, Hkv, D = v_full.shape
        sv = float(torch.clamp(v_full.abs().amax().float() / 127.0, min=1e-8))
        v_i8 = torch.empty((L, Hkv, D), dtype=torch.int8, device=v_full.device)
        for a in range(0, L, self._QUANT_SLICE):
            b = min(a + self._QUANT_SLICE, L)
            v_i8[a:b] = torch.clamp(torch.round(v_full[a:b].float() / sv), -127, 127).to(torch.int8)
        return v_i8, sv

    def _int8_call(self, q_i8, sq, k_i8, sk, v_i8, sv, sm, causal):
        o = flashinfer.single_prefill_with_kv_cache(
            q_i8, k_i8, v_i8, scale_q=sq, scale_k=sk,
            causal=causal, backend="fa2", o_dtype=torch.float16,
            pos_encoding_mode="NONE", sm_scale=sm,
        )
        return o * sv  # dequant V -> restore output magnitude

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
        # Hard fallbacks: anything the int8 path doesn't support -> stock FA (bit-faithful).
        if (
            not self._int8qk_ok
            or attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
            or getattr(attn_metadata, "use_cascade", False)
            or getattr(self, "dcp_world_size", 1) > 1
            or is_quantized_kv_cache(self.kv_cache_dtype)
        ):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )

        cu = attn_metadata.query_start_loc
        cu_list = cu.tolist()                                  # [batch+1] (CPU sync, small)
        n_req = len(cu_list) - 1
        # Per-request query lengths and total context (seq) lengths.
        qlens = [int(cu_list[i + 1] - cu_list[i]) for i in range(n_req)]
        seq_lens_cpu = attn_metadata.seq_lens.tolist()        # [batch] total ctx per req (CPU sync)

        # The int8 path targets PREFILL chunks (q_len >= 1 with the whole new chunk). A row is a
        # "decode" row when q_len==1 AND it has a cached prefix (seq_len>1): bandwidth-bound, route
        # to FA. If ANY row is decode, fall back to FA for the whole batch (mixing is rare in the
        # chunked single-request prefill we target; FA is correct for the mixed case).
        for i in range(n_req):
            if qlens[i] <= 0:
                continue
            if qlens[i] == 1 and seq_lens_cpu[i] > 1:
                return super().forward(
                    layer, query, key, value, kv_cache, attn_metadata, output,
                    output_scale, output_block_scale,
                )

        # Write the NEW chunk's K/V into the paged cache so the gather (and later decode) see them.
        key_cache, value_cache = kv_cache.unbind(1)
        reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            attn_metadata.slot_mapping, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_size
        sm = self.scale  # 1/sqrt(d)
        causal = attn_metadata.causal
        block_table = attn_metadata.block_table               # [batch, max_blocks]
        bs = key_cache.shape[1]

        for i in range(n_req):
            s, e = int(cu_list[i]), int(cu_list[i + 1])
            if e <= s:
                continue
            Lq = e - s
            seq_len = int(seq_lens_cpu[i])
            ctx = seq_len - Lq                                 # cached-prefix length for this req

            # Q (small new chunk) -> int8.
            q_i8, sq = _quant_qk_per_token(query[s:e].reshape(Lq, H, D))

            # K then V, each gathered+quantized+freed SEPARATELY so the large full-context
            # fp32 temporary never coexists with the OTHER tensor's bf16 gather (the 128k OOM:
            # holding k_bf16 + k_fp32 + v_bf16 simultaneously). Peak full-ctx tensors now <= one
            # bf16 gather + one fp32 quant temporary at a time.
            if ctx <= 0:
                # PURE-FRESH chunk: in-hand K/V (no gather; bit-identical to I-4b numerics).
                k_i8, sk = self._quant_k_smooth(key[s:e].reshape(Lq, Hkv, D))
                v_i8, sv = self._quant_v_pertensor(value[s:e].reshape(Lq, Hkv, D))
                INT8QK_FIRE["fresh"] += 1
            else:
                # CACHED-PREFIX chunk: gather the FULL context (prefix + just-written chunk)
                # from the paged cache, one tensor at a time, freeing the bf16 gather ASAP.
                n_blocks = (seq_len + bs - 1) // bs
                blk = block_table[i][:n_blocks]
                k_full = self._gather_one(key_cache, blk, n_blocks, seq_len)
                k_i8, sk = self._quant_k_smooth(k_full)
                del k_full                                     # free K bf16 gather before V
                v_full = self._gather_one(value_cache, blk, n_blocks, seq_len)
                v_i8, sv = self._quant_v_pertensor(v_full)
                del v_full
                INT8QK_FIRE["cached"] += 1

            o = self._int8_call(q_i8, sq, k_i8, sk, v_i8, sv, sm, causal)
            del k_i8, v_i8
            output[s:e] = o.reshape(output[s:e].shape).to(output.dtype)
            INT8QK_FIRE["calls"] += 1
            INT8QK_FIRE["seqs"] += 1
        return output


class Int8QKAttentionBackend(FlashAttentionBackend):
    """Inherits FA's metadata/KV-cache-shape; swaps impl + builder for the int8-QK prefill path."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "INT8QK"

    @staticmethod
    def get_impl_cls() -> type[Int8QKAttentionImpl]:
        return Int8QKAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Int8QKMetadataBuilder]:
        return Int8QKMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return FlashAttentionBackend.supports_attn_type(attn_type)
