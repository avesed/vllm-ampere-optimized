"""HALF-A ANALYTICAL predictor — compute the optimum, use vLLM only to VERIFY.

Restarting vLLM per config is ~minutes; most of the throughput/capacity frontier is a roofline +
a linear KV/state-capacity model, so we PREDICT the optimum from the model/HW (a priori) or from a
couple of cheap probe points, then spend ONE vLLM restart verifying the prediction.

Two models (both pure, unit-tested):
  - KV/state capacity: kv_usage(seqs) is ~linear (per-seq footprint) -> the feasible-seqs wall, and
    fp8 (half the KV bytes) ~doubles it.
  - Decode throughput roofline: aggregate tok/s(B) = B / (c1 + c2*B)  (weight read amortized over
    the batch, KV read grows with it) -> 1/tps = c1/B + c2; ceiling = 1/c2; the knee is where it
    flattens. Optimum batch = min(feasible wall, throughput knee).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class Sample:
    seqs: int
    kv: float        # gpu_cache_usage 0..1 at this max-num-seqs (oversubscribed)
    tps: float       # aggregate decode tok/s


@dataclass
class Prediction:
    feasible_seqs: int          # max max-num-seqs before kv_target (current kv dtype)
    feasible_seqs_fp8: int      # ... with fp8 (KV bytes halved)
    tps_ceiling: float          # roofline aggregate ceiling (B -> inf)
    knee_seqs: int              # batch where throughput ~flattens
    rec_seqs: int               # recommended max-num-seqs (no fp8)
    rec_fp8: bool               # recommend fp8 (only if feasibility-bound BELOW the throughput knee)
    rec_seqs_fp8: int
    pred_tps: float             # predicted aggregate tok/s at the recommendation
    capacity_bound: bool        # the wall, not the roofline, is what limits us
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Least-squares slope, intercept (exact for 2 points)."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-18:
        return 0.0, (sy / n if n else 0.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def fit_kv_linear(samples: List[Sample]) -> Tuple[float, float]:
    """kv ~= a*seqs + b -> (a per-seq footprint, b overhead)."""
    return _linreg([float(s.seqs) for s in samples], [s.kv for s in samples])


def fit_roofline(samples: List[Sample]) -> Tuple[float, float]:
    """1/tps = c1*(1/B) + c2 -> (c1 weight term, c2 = 1/ceiling)."""
    return _linreg([1.0 / s.seqs for s in samples], [1.0 / s.tps for s in samples])


def predict_tps(c1: float, c2: float, seqs: int) -> float:
    denom = c1 / seqs + c2
    return (1.0 / denom) if denom > 0 else 0.0


def predict(samples: List[Sample], kv_target: float = 0.95, plateau_frac: float = 0.95) -> Optional[Prediction]:
    """Predict the throughput-optimal max-num-seqs (+ whether fp8 unlocks more) from >=2 probe
    points. The expensive search collapses to ONE verification run at rec_seqs."""
    if len({s.seqs for s in samples}) < 2:
        return None
    a, b = fit_kv_linear(samples)
    c1, c2 = fit_roofline(samples)
    if a <= 0 or c2 <= 0:
        return None

    feasible = max(1, int((kv_target - b) / a))
    feasible_fp8 = max(1, int((kv_target - b) / (a / 2.0)))   # fp8 halves per-seq KV
    ceiling = 1.0 / c2
    knee = max(1, int(plateau_frac * c1 / ((1.0 - plateau_frac) * c2)))

    rec = min(feasible, knee)
    capacity_bound = feasible < knee                          # the wall bites before the roofline does
    rec_fp8 = capacity_bound                                  # only then does halving KV bytes buy throughput
    rec_fp8_seqs = min(feasible_fp8, knee)

    if capacity_bound:
        reason = (f"capacity-bound: feasible ~{feasible} seqs (KV {kv_target:.0%}) is below the throughput "
                  f"knee ~{knee}, and throughput is still climbing -> push to ~{rec}; fp8 ~doubles KV room "
                  f"-> ~{rec_fp8_seqs} seqs (~{predict_tps(c1, c2, rec_fp8_seqs):.0f} tok/s).")
    else:
        reason = (f"roofline-bound: throughput knee ~{knee} seqs is reached before the capacity wall "
                  f"~{feasible} -> ~{rec} seqs captures it; more concurrency / fp8 won't add throughput.")

    return Prediction(
        feasible_seqs=feasible, feasible_seqs_fp8=feasible_fp8, tps_ceiling=ceiling, knee_seqs=knee,
        rec_seqs=rec, rec_fp8=rec_fp8, rec_seqs_fp8=rec_fp8_seqs,
        pred_tps=predict_tps(c1, c2, rec_fp8_seqs if rec_fp8 else rec),
        capacity_bound=capacity_bound, reason=reason)


@dataclass
class BaselinePrediction:
    seqs0: int
    kv0: float
    feasible_seqs: int          # workload-anchored wall before kv_target (current kv dtype)
    feasible_seqs_fp8: int      # fp8 ~halves KV bytes -> ~doubles the wall
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def predict_from_baseline(baseline: Sample, kv_target: float = 0.95, fp8_ceiling: int = 1024) -> Optional[BaselinePrediction]:
    """ONE real vLLM baseline -> the feasible-seqs wall. Per-TOKEN KV bytes are exact from config,
    but realized usage depends on the workload (PagedAttention allocates on demand), so the SCALE
    must be anchored on a real run, not guessed. KV ~ linear in seqs (b~=0): wall = target*seqs0/kv0;
    fp8 ~doubles it. Throughput-at-the-wall is then confirmed by ONE verify run."""
    if baseline.kv <= 0 or baseline.seqs <= 0:
        return None
    a = baseline.kv / baseline.seqs                       # per-seq KV footprint at this workload
    feasible = max(1, int(kv_target / a))
    feasible_fp8 = min(fp8_ceiling, max(1, int(kv_target / (a / 2.0))))
    return BaselinePrediction(
        seqs0=baseline.seqs, kv0=baseline.kv, feasible_seqs=feasible, feasible_seqs_fp8=feasible_fp8,
        reason=(f"1 baseline ({baseline.seqs} seqs -> KV {baseline.kv:.0%}): KV scales ~linearly, so the "
                f"feasible wall is ~{feasible} seqs (KV {kv_target:.0%}); fp8 ~doubles it -> ~{feasible_fp8}. "
                f"Push to ~{feasible} (or fp8 -> ~{feasible_fp8}) and VERIFY throughput with one restart."))


def render(p: Optional[Prediction], samples: List[Sample]) -> str:
    if p is None:
        return "analytical: need >=2 probe points at distinct max-num-seqs to fit the model."
    pts = ", ".join(f"({s.seqs}: {s.tps:.0f} tok/s, KV {s.kv:.0%})" for s in samples)
    best_seqs = p.rec_seqs_fp8 if p.rec_fp8 else p.rec_seqs
    sweep = f"--max-num-seqs={best_seqs}" + (";--kv-cache-dtype=fp8" if p.rec_fp8 else "")
    flags = sweep.replace(";", " ").replace("=", " ")
    return "\n".join([
        "ampere-autotune — HALF-A ANALYTICAL prediction (compute, then verify)",
        f"  probed: {pts}",
        f"  roofline ceiling ~{p.tps_ceiling:.0f} tok/s; throughput knee ~{p.knee_seqs} seqs; "
        f"feasible wall ~{p.feasible_seqs} seqs (fp8 ~{p.feasible_seqs_fp8}).",
        f"  -> {p.reason}",
        f"  PREDICT BEST: {flags}  (~{p.pred_tps:.0f} tok/s aggregate)",
        f"  VERIFY (one restart): cotune --sweep \"{sweep}\"",
    ])
