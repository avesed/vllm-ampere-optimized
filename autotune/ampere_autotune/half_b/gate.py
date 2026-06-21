"""HALF-B correctness gate — the DECISION logic is pure and GPU-free (so it is fully
unit-testable); only the measurement COLLECTION touches the GPU.

On no-ECC GDDR6X, corruption is SILENT. The real gate is the exact golden token-id compare
(under VLLM_BATCH_INVARIANT=1); BW/EDR-knee, mismatch_count, junction temp are coverage/
health signals. ``evaluate`` implements the climb-stop = FIRST-of ordering, hard-safety first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..preflight import sku as _sku  # gate-family constants

# verdict statuses
PASS = "PASS"        # this offset is correct AND still gaining bandwidth -> keep climbing
KNEE = "KNEE"        # correct, but bandwidth stopped rising (EDR replay) -> stop climbing
REJECT = "REJECT"    # corrupting clock -> back off below it
ABORT = "ABORT"      # hard fault (junction/Xid) -> zero offset immediately


@dataclass
class Thresholds:
    junction_abort_c: float = 95.0
    junction_warn_c: float = 90.0
    knee_rise_pct: float = 1.5     # min % read_GB/s rise vs prev to count as still climbing
    knee_regress_pct: float = 3.0  # % read_GB/s drop that means we're clearly past the knee
    # layered guard band, in 15 MHz ticks (NOT a single hysteresis tick — see RESEARCH §4)
    thermal_guard_ticks: int = 1
    coverage_guard_ticks: int = 1
    hysteresis_ticks: int = 1

    def __post_init__(self) -> None:
        # C3: a negative/None guard tick would make accepted > boundary (deploy a clock
        # higher than anything confirmed correct). Reject at construction.
        for name in ("thermal_guard_ticks", "coverage_guard_ticks", "hysteresis_ticks"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"{name} must be a non-negative int, got {v!r}")
        if (self.thermal_guard_ticks + self.coverage_guard_ticks
                + self.hysteresis_ticks) < 1:
            raise ValueError("guard band must total >= 1 tick")

    def guard_band_mhz(self, tick: int = 15) -> int:
        return (self.thermal_guard_ticks + self.coverage_guard_ticks
                + self.hysteresis_ticks) * tick


@dataclass
class Measurement:
    offset_mhz: int
    golden_ok: bool                  # exact token-id match under VLLM_BATCH_INVARIANT=1 (THE gate)
    mismatch_count: int = 0          # BW+verify cell mismatches; >0 = silent corruption
    read_gbs: Optional[float] = None  # achieved bandwidth (EDR-knee signal; GDDR6X only)
    junction_c: Optional[float] = None
    xid: bool = False
    throttle: bool = False


@dataclass
class GateVerdict:
    status: str
    reason: str

    @property
    def is_safe(self) -> bool:
        """True if this clock is correct (PASS or KNEE), i.e. not a corruption/fault."""
        return self.status in (PASS, KNEE)


def evaluate(cur: Measurement, prev: Optional[Measurement], th: Thresholds,
             gate_family: str) -> GateVerdict:
    """Pure verdict. FIRST-of ordering: hard faults, then correctness, then the (soft) knee.

    ``prev`` is only used for the EDR-knee bandwidth-rise check; pass None during a
    fine-DESCENT (then only the safety checks apply — descent seeks a correct clock, not a
    throughput peak).
    """
    # 1. hard faults -> ABORT (zero the offset now)
    if cur.xid:
        return GateVerdict(ABORT, "Xid fault")
    if cur.junction_c is not None and cur.junction_c >= th.junction_abort_c:
        return GateVerdict(ABORT, f"junction {cur.junction_c:.0f}C >= {th.junction_abort_c:.0f}C")

    # 2. correctness -> REJECT (back off below this clock). Golden is THE gate.
    if not cur.golden_ok:
        return GateVerdict(REJECT, "golden token-id mismatch")
    if cur.mismatch_count > 0:
        return GateVerdict(REJECT, f"BW+verify mismatch_count={cur.mismatch_count}")

    # 3. EDR knee (GDDR6X only, climb phase only): correct but no bandwidth gain -> stop.
    # H1: on GDDR6X the bandwidth reading is the throughput objective AND the main
    # below-corruption stop signal. A missing/zero reading is inconclusive (often itself an
    # instability signal) -> STOP (KNEE), NEVER fall through to PASS and keep climbing.
    # Use explicit None/<=0 checks, not truthiness (0.0 is falsy).
    if gate_family == _sku.GATE_GOLDEN_EDR_JUNCTION and prev is not None:
        if cur.read_gbs is None or prev.read_gbs is None or prev.read_gbs <= 0:
            return GateVerdict(KNEE, "EDR knee: bandwidth reading inconclusive -> stop (no PASS)")
        rise_pct = (cur.read_gbs - prev.read_gbs) / prev.read_gbs * 100.0
        if rise_pct < th.knee_rise_pct or rise_pct <= -th.knee_regress_pct:
            return GateVerdict(KNEE, f"EDR knee: read_GB/s rise {rise_pct:.1f}% < {th.knee_rise_pct}%")

    return GateVerdict(PASS, "ok")


def collect(handle, offset_mhz: int) -> Measurement:  # pragma: no cover - GPU-bound
    """Apply ``offset_mhz`` (via silicon.py) and gather the gate measurements.

    GPU-BOUND — not exercised in no-GPU CI. Wires: golden token-id under
    VLLM_BATCH_INVARIANT=1 + bw_verify kernel (read_GB/s + mismatch_count) + gputemps
    junction + NVML Xid/throttle. STATUS: stub.
    """
    raise NotImplementedError(
        f"gate.collect(offset_mhz={offset_mhz}, handle={handle!r}) requires a GPU + a running "
        "vLLM (golden), the bw_verify kernel, and gputemps (junction). See DESIGN.md / "
        "instruments/. The search state machine is validated GPU-free by injecting a "
        "measurement function (see tests/test_search.py)."
    )


# re-export for callers that want the gate-family vocab without importing preflight.sku
GATE_GOLDEN_EDR_JUNCTION = _sku.GATE_GOLDEN_EDR_JUNCTION
GATE_ECC_GOLDEN = _sku.GATE_ECC_GOLDEN
GATE_ECC = _sku.GATE_ECC
__all__ = [
    "PASS", "KNEE", "REJECT", "ABORT", "Thresholds", "Measurement", "GateVerdict",
    "evaluate", "collect", "GATE_GOLDEN_EDR_JUNCTION", "GATE_ECC_GOLDEN", "GATE_ECC",
]
