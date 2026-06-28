# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere — the unified Ampere (sm_80/sm_86) attention backend.

ONE FlashAttention subclass that the deployment actually selects (via Backend.CUSTOM), whose
forward() dispatches per call to the best Ampere kernel and sinks everything else to stock FA:

  prefill hd256, fp16 query (plain dec, non-quant KV) -> fp16-PV FlashInfer prefill (fp16pv)
  prefill hd256, bf16 query                            -> fp16-PV via runtime bf16->fp16 (bf16cvt)
  prefill hd64/96/128                                  -> SageAttn (int8-QK; research-grade, opt-in)
  decode / MTP-verify / encoder / cascade /            -> super().forward() (stock FA; verify already
    quantized-KV / non-plain-decoder / non-GA10x          uses the fwd_kvcache split-KV fix in base FA)

(int8-QK was removed: a sweep found it net-negative in every scenario — fresh 16/32/64k + cached-prefix.)

Opt-in: master env VLLM_FLASHAMPERE=1 (+ per-leg sub-toggles). Registered in every TP/PP worker
via the vllm.general_plugins entry-point register_flashampere (backend.py).

dispatch.py and capability.py are deliberately import-light (no torch/CUDA, no vllm runtime) so
the routing + gating policy is unit-testable on CPU; impl.py / backend.py carry the GPU code.
"""

# Lazily re-export the entry-point so importing this package stays cheap (the entry-point path
# in pyproject points straight at backend:register_flashampere; this is just for convenience).
__all__ = ["register_flashampere"]


def __getattr__(name: str):
    if name == "register_flashampere":
        from .backend import register_flashampere

        return register_flashampere
    raise AttributeError(name)
