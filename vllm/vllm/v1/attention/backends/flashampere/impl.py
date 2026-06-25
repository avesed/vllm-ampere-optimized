# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAmpereImpl — the one Ampere attention impl. forward() is a thin guard + per-call phase
classify + dispatch-table resolve, with stock FlashAttention (super().forward) as the universal,
bit-faithful sink. All accelerated work lives in kernels.py; routing/gating in dispatch.py +
capability.py. The base FA forward already carries the MTP-verify fwd_kvcache fix
(VLLM_FA2_KVCACHE_VERIFY), so VERIFY and DECODE sink to it unchanged."""
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from . import capability, kernels
from .dispatch import DispatchKey, KernelDecline, Phase, QSrc, resolve

logger = init_logger(__name__)

_CAPS: capability.FlashAmpereCaps | None = None


def _caps() -> capability.FlashAmpereCaps:
    # Process-global: one GPU per process, so detect once and reuse across all layers.
    global _CAPS
    if _CAPS is None:
        _CAPS = capability.detect(**capability.gather_inputs())
        logger.info("flashampere caps: %s", _CAPS)
    return _CAPS


def _verify_upper_bound() -> int:
    """Max query length treated as spec-decode verify (= reorder_batch_threshold = 1 + K). A
    prefill chunk has q_len above this; a verify step at or below it. 1 means no spec -> no verify."""
    try:
        from vllm.config import get_current_vllm_config

        sc = get_current_vllm_config().speculative_config
        if sc is not None and sc.num_speculative_tokens is not None:
            mult = 2 if getattr(sc, "parallel_drafting", False) else 1
            return 1 + mult * sc.num_speculative_tokens
    except Exception:
        pass
    return 1


class FlashAmpereImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sw = getattr(self, "sliding_window", None)
        no_sw = sw is None or tuple(sw) == (-1, -1)
        soft_cap = getattr(self, "logits_soft_cap", None)
        no_soft_cap = soft_cap is None or soft_cap == 0
        # Per-layer static eligibility for the fp16-PV legs: a plain causal decoder attention
        # (no SWA/alibi/softcap/sinks); head_size lives in the DispatchKey, the library probe in caps.
        self._plain_decoder = (
            getattr(self, "attn_type", AttentionType.DECODER) == AttentionType.DECODER
            and no_sw
            and getattr(self, "alibi_slopes", None) is None
            and no_soft_cap
            and getattr(self, "sinks", None) is None
        )
        self._caps = _caps()
        self._verify_ub = _verify_upper_bound()

    def _classify(self, m) -> Phase:
        """Per-call phase from CPU-only metadata scalars (capture-safe: no .item()/.tolist())."""
        if getattr(m, "use_cascade", False):
            return Phase.OTHER
        mql = getattr(m, "max_query_len", None)
        if mql is None or mql <= 1:
            return Phase.DECODE
        # Uniform small-query batch == spec-decode verify -> sink to base FA fwd_kvcache.
        num_reqs = m.seq_lens.shape[0]
        if mql <= self._verify_ub and m.num_actual_tokens == num_reqs * mql:
            return Phase.VERIFY
        return Phase.PREFILL

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = attn_metadata
        # Universal hard fallbacks -> stock FA (cannot be expressed in the structural key).
        if (
            m is None
            or output_scale is not None
            or output_block_scale is not None
            or getattr(self, "dcp_world_size", 1) > 1
        ):
            return super().forward(
                layer, query, key, value, kv_cache, m, output,
                output_scale, output_block_scale,
            )

        # Hot-path early-out: DECODE (q=1, bandwidth-bound) + OTHER (encoder/cascade) have no fast
        # row -> sink straight to stock FA (keeps the decode path lean). PREFILL (fp16-PV legs) and
        # VERIFY (xqa_verify) go through resolve below; when their leg is disabled they sink too
        # (VERIFY -> the base-FA fwd_kvcache verify fix).
        phase = self._classify(m)
        if phase is Phase.DECODE or phase is Phase.OTHER:
            return super().forward(
                layer, query, key, value, kv_cache, m, output,
                output_scale, output_block_scale,
            )

        key_t = DispatchKey(
            phase=phase,
            head_dim=self.head_size,
            q_src=(
                QSrc.HALF if query.dtype == torch.float16
                else QSrc.BF16 if query.dtype == torch.bfloat16
                else QSrc.OTHER
            ),
            kv_quantized=kernels.is_quantized_kv_cache(self.kv_cache_dtype),
            cap=self._caps.cap,
            plain_decoder=self._plain_decoder,
        )

        # Sync-bearing prefill legs (.tolist() gather) are illegal during cudagraph capture;
        # capture_safe=False legs route to FA while capturing. Prefill is eager so this only
        # ever fires defensively (incl. spec-decode capture, where verify already sank above).
        capturing = torch.cuda.is_current_stream_capturing()

        for entry in resolve(key_t):
            if capturing and not entry.capture_safe:
                continue
            if not self._caps.enabled(entry.name):
                continue
            try:
                return self._run(entry.name, layer, query, key, value, kv_cache, m, output)
            except KernelDecline:
                continue  # try the next candidate, else the sink below

        return super().forward(
            layer, query, key, value, kv_cache, m, output,
            output_scale, output_block_scale,
        )

    def _run(self, name, layer, query, key, value, kv_cache, m, output):
        # fp16pv (fp16-served) and bf16cvt (bf16-served) share the SAME kernel: it casts Q/K/V
        # to fp16 internally (lossless from bf16) and writes back at output.dtype. The leg name
        # only differs for routing/gating + telemetry. (int8-QK was removed: net-negative.)
        if name in ("fp16pv", "bf16cvt"):
            return kernels.fp16pv_prefill(
                self, layer, query, key, value, kv_cache, m, output, leg=name
            )
        if name == "xqa_verify":
            from . import xqa_verify as _xqa_verify
            return _xqa_verify.xqa_verify(self, layer, query, key, value, kv_cache, m, output)
        # "sage" lands here in a future phase; until implemented, decline -> sink.
        raise KernelDecline
