"""HALF-A: fold MEASURED hardware ceilings (actual bandwidth + actual compute) into the tuner.

Spec peaks lie: a 3090's 936 GB/s spec is really ~838 achievable (~89.5%), ~888 with mem-OC. And
single-stream decode time splits into a BANDWIDTH part (streaming weights, ∝ 1/bw) and a FIXED part
(compute/launch/overhead that does NOT scale with bandwidth):

    TPOT(bw) = bw_coef / bw + fixed_t        tok/s = 1 / TPOT

Knowing the SPLIT is the actionable bit: it says exactly how much a fewer-bytes lever (mem-OC,
fp8/int4 KV, a bigger/faster card) can move decode — and when you're already compute/overhead-bound
so it CAN'T. The bw number comes from the bw_verify instrument (a plain CUDA read kernel, NO
privilege), so this lives in HALF-A. All pure; validated against the real 9B-w4a8 measurement
(838 GB/s→85 tok/s, 888→88.2 → 64% bandwidth-bound, compute ceiling ~239 tok/s, +3.8% from mem-OC).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from .analytical import _linreg


@dataclass
class DecodeModel:
    """Single-stream decode split. bandwidth-time(per tok) = bw_coef / bw_gbs; fixed_t = the rest."""
    bw_coef: float       # GB  (so bw_coef/bw_gbs = seconds/token)
    fixed_t: float       # seconds/token that do NOT scale with bandwidth (compute + launch overhead)

    def tpot_s(self, bw_gbs: float) -> float:
        return self.bw_coef / bw_gbs + self.fixed_t

    def toks(self, bw_gbs: float) -> float:
        t = self.tpot_s(bw_gbs)
        return (1.0 / t) if t > 0 else 0.0

    def bw_bound_frac(self, bw_gbs: float) -> float:
        """0..1: how bandwidth-bound decode is at this bw. ~1 = pure weight-streaming (mem-OC /
        fewer-bytes help most); ->0 = compute/overhead-bound (those levers do ~nothing)."""
        t = self.tpot_s(bw_gbs)
        return (self.bw_coef / bw_gbs) / t if t > 0 else 0.0

    def compute_ceiling_toks(self) -> float:
        """tok/s even at INFINITE bandwidth (the fixed/compute wall) — the hard cap mem-OC can't pass."""
        return (1.0 / self.fixed_t) if self.fixed_t > 0 else float("inf")

    def to_dict(self) -> dict:
        return asdict(self)


def fit_decode_two_points(points: List[Tuple[float, float]]) -> Optional[DecodeModel]:
    """[(bw_gbs, toks), ...] at >=2 DISTINCT bandwidths (e.g. stock clock + a mem-OC clock) ->
    solve 1/toks = bw_coef*(1/bw) + fixed_t. This is the model-internal-free path."""
    if len({round(b, 1) for b, _ in points}) < 2 or any(t <= 0 for _, t in points):
        return None
    slope, intercept = _linreg([1.0 / b for b, _ in points], [1.0 / t for _, t in points])
    if slope <= 0 or intercept < 0:                       # non-physical fit (e.g. noise) -> reject
        return None
    return DecodeModel(bw_coef=slope, fixed_t=intercept)


def decode_from_one_point(toks: float, bw_gbs: float, weight_bytes: float) -> Optional[DecodeModel]:
    """ONE measured (toks @ bw) + the model's per-token WEIGHT bytes (from config) -> the split.
    bandwidth-time = weight_bytes/bw; fixed = 1/toks - that. Returns None if the bytes estimate is
    inconsistent (weight-read alone exceeds the measured TPOT)."""
    if toks <= 0 or bw_gbs <= 0 or weight_bytes <= 0:
        return None
    bw_t = weight_bytes / (bw_gbs * 1e9)
    fixed = 1.0 / toks - bw_t
    if fixed < 0:
        return None
    return DecodeModel(bw_coef=weight_bytes / 1e9, fixed_t=fixed)


def memoc_decode_gain_pct(model: DecodeModel, cur_bw: float, new_bw: float) -> float:
    """Predicted decode tok/s change (%) from raising achievable bandwidth cur->new (e.g. a HALF-B
    mem-OC). Sub-proportional to the bw gain by exactly the bw-bound fraction."""
    c, n = model.toks(cur_bw), model.toks(new_bw)
    return (n / c - 1.0) * 100.0 if c > 0 else 0.0


def measure_bw_gbs(bw_bin: str, *, size_gib: int = 8, iters: int = 200,
                   cuda_visible: Optional[str] = None) -> Optional[float]:  # pragma: no cover - GPU
    """Run the bw_verify CUDA kernel (NO privilege) -> achievable read GB/s, or None on failure."""
    import json
    import os
    import subprocess
    env = dict(os.environ)
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    try:
        p = subprocess.run([bw_bin, str(size_gib), str(iters)], env=env,
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reversed(p.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                return float(json.loads(line)["read_GB_s"])
            except (ValueError, KeyError):
                continue
    return None


def render(model: Optional[DecodeModel], cur_bw: float, *, memoc_bw: Optional[float] = None) -> str:
    if model is None:
        return ("hw-factors: need 2 decode points at distinct bandwidths (stock + a mem-OC clock) "
                "OR one point + the model's per-token weight bytes.")
    frac = model.bw_bound_frac(cur_bw)
    lines = [
        "ampere-autotune — measured HW factors (actual bandwidth + compute in the decode model)",
        f"  at {cur_bw:.0f} GB/s: {model.toks(cur_bw):.0f} tok/s "
        f"(TPOT {model.tpot_s(cur_bw) * 1000:.1f} ms) = {frac:.0%} bandwidth-bound + "
        f"{1 - frac:.0%} fixed-compute",
        f"  fixed-compute ceiling ~{model.compute_ceiling_toks():.0f} tok/s (mem-OC/fewer-bytes can NEVER beat this)",
    ]
    if frac < 0.4:
        lines.append("  -> COMPUTE/overhead-bound: mem-OC, fp8/int4-KV won't move decode much; "
                     "spec-decode (MTP) / a faster card is the lever.")
    else:
        lines.append("  -> BANDWIDTH-bound: a fewer-bytes lever pays. "
                     + (f"mem-OC {cur_bw:.0f}->{memoc_bw:.0f} GB/s -> "
                        f"{memoc_decode_gain_pct(model, cur_bw, memoc_bw):+.1f}% decode."
                        if memoc_bw else "run bw_verify at a mem-OC clock to size the gain."))
    return "\n".join(lines)
