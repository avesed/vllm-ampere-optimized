"""famp-owned FA2 single prefill (fp16-PV) — drop-in for the fp16pv leg's
flashinfer.single_prefill_with_kv_cache call, but OWNS the run() marshalling so it does not depend
on a patched flashinfer (the patched single_prefill marshals int8-QK args we don't carry). Validated
cos=1.000000 vs flashinfer single_prefill(use_fp16_pv_reduction=True) on sm86.
"""
import functools
import torch
from ._jit_prefill import get_famp_prefill_module

try:
    from flashinfer.utils import MaskMode, TensorLayout
    _CAUSAL = MaskMode.CAUSAL.value
    _NONCAUSAL = MaskMode.NON_CAUSAL.value
    _NHD = TensorLayout["NHD"].value
except Exception:  # values are stable in the flashinfer ABI
    _CAUSAL, _NONCAUSAL, _NHD = 1, 0, 0


@functools.cache
def _tmp_buf(device: torch.device, nbytes: int = 64 * 1024 * 1024) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


def single_prefill(q, k, v, *, causal, sm_scale, o_dtype=None, use_fp16_pv=True):
    """q/k/v: [L, H, D] (NHD layout). pos_encoding NONE, no logits soft cap, no sliding window —
    matches the famp fp16pv prefill leg. Returns out [Lq, Hq, Dvo] in o_dtype."""
    Dqk = q.shape[-1]
    Dvo = v.shape[-1]
    if o_dtype is None:
        o_dtype = q.dtype
    mod = get_famp_prefill_module(
        q.dtype, k.dtype, o_dtype, Dqk, Dvo,
        0, False, False, False, use_fp16_pv,
    )
    out = torch.empty(q.shape[:-1] + (Dvo,), dtype=o_dtype, device=q.device)
    tmp = _tmp_buf(q.device)
    head = (q, k, v, tmp, out, None,
            _CAUSAL if causal else _NONCAUSAL, _NHD, -1,
            None, None)  # ..., packed_custom_mask, alibi_slopes
    # FlashInfer's run() ABI moved between the versions we support, and the two differ in ARITY
    # (17 vs 20), so a wrong guess raises instead of silently shuffling scalars into each other's
    # slots (that failure mode reads as cos~0.25 output, not an error).
    #   >=0.6.13: ..., logits_soft_cap, sm_scale, scale_q, scale_k, scale_v, rope_scale, rope_theta,
    #             k_sf, v_sf
    #    0.6.12 : ..., k_sf, v_sf, logits_soft_cap, sm_scale, rope_rcp_scale, rope_rcp_theta
    try:
        mod.run(*head, 0.0, float(sm_scale), None, None, None, 1.0, 1e4, None, None)
    except TypeError:
        mod.run(*head, None, None, 0.0, float(sm_scale), 1.0, 1e4)
    return out
