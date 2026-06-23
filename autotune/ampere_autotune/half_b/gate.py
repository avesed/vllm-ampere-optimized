"""HALF-B correctness gate — the DECISION logic is pure and GPU-free (so it is fully
unit-testable); only the measurement COLLECTION touches the GPU.

On no-ECC GDDR6X, corruption is SILENT. The real gate is the exact golden token-id compare
(under VLLM_BATCH_INVARIANT=1); BW/EDR-knee, mismatch_count, junction temp are coverage/
health signals. ``evaluate`` implements the climb-stop = FIRST-of ordering, hard-safety first.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from ..preflight import sku as _sku  # gate-family constants
from . import silicon

# bw_verify lives in the package's instruments/ dir (built native); overridable via env.
DEFAULT_BW_BIN = os.environ.get(
    "AMPERE_AUTOTUNE_BW_BIN",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "instruments", "bw_verify", "bw_verify"),
)

# verdict statuses
PASS = "PASS"        # this offset is correct AND still gaining bandwidth -> keep climbing
KNEE = "KNEE"        # correct, but bandwidth stopped rising (EDR replay) -> stop climbing
REJECT = "REJECT"    # corrupting clock -> back off below it
ABORT = "ABORT"      # hard fault (junction/Xid) -> zero offset immediately


@dataclass
class Thresholds:
    junction_abort_c: float = 95.0
    junction_warn_c: float = 90.0
    # min % read_GB/s rise vs prev to count as "still climbing". CALIBRATED to real GDDR6X
    # (2026-06-23): a 3090 rises only ~0.5%/105MHz coarse step (838->917 GB/s over +2000), so the
    # old 1.5% default false-KNEE'd at the FIRST step on a card with +2000 headroom. Set below the
    # climbing slope. CAVEAT: ~0.5%/step is near the bw_verify noise floor -> the bw-knee is a WEAK
    # secondary stop; the PRIMARY stop on GDDR6X is mismatch onset (REJECT), and _combine() medians
    # N samples to fight noise. With a strong sample the climb hits max_offset (MAXED) before a knee.
    knee_rise_pct: float = 0.3
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
    golden_ok: Optional[bool]        # True=match / False=mismatch / None=not run (VLLM_BATCH_INVARIANT)
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
    # tri-state: False = ran and mismatched (REJECT); None = NOT RUN (no model up) -> fall back to
    # the bw_verify integrity check below (weaker — never claim correctness we didn't measure).
    if cur.golden_ok is False:
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


def parse_bw_verify(stdout: str):
    """bw_verify emits one JSON line {read_GB_s, write_GB_s, mismatch_count, ...}. Return
    (read_gbs, mismatch_count); (None, -1) if no parsable line (crash/hang = instability signal)."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
                return float(d["read_GB_s"]), int(d["mismatch_count"])
            except (ValueError, KeyError):
                continue
    return None, -1


def assemble(offset_mhz: int, bw: tuple, *, golden_ok: Optional[bool] = True,
             junction_c: Optional[float] = None, throttle: bool = False,
             set_rc: int = 0) -> Measurement:
    """Pure: fold a (read_gbs, mismatch_count) bw result + side signals into a Measurement.
    set_rc!=0 (couldn't even apply the offset) or a bw crash (mismatch<0) => hard-unsafe.
    golden_ok is tri-state and preserved verbatim (None = golden not run -> see evaluate)."""
    read_gbs, mismatch = bw
    if set_rc != 0:                                   # the SET itself failed -> can't trust anything
        return Measurement(offset_mhz, golden_ok=False, mismatch_count=0, read_gbs=None, xid=True)
    if mismatch is not None and mismatch < 0:         # bw_verify crashed/hung -> instability -> ABORT
        return Measurement(offset_mhz, golden_ok=False, mismatch_count=0, read_gbs=None, xid=True)
    return Measurement(offset_mhz, golden_ok=golden_ok,
                       mismatch_count=max(0, mismatch or 0), read_gbs=read_gbs,
                       junction_c=junction_c, xid=False, throttle=throttle)


def run_bw_verify(uuid: str, size_gib: int, iters: int = 200,
                  bw_bin: str = DEFAULT_BW_BIN):  # pragma: no cover - GPU-bound
    env = {"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": uuid}
    try:
        p = subprocess.run([bw_bin, str(size_gib), str(iters)], env=env,
                           capture_output=True, text=True, timeout=900)
        return parse_bw_verify(p.stdout)
    except (subprocess.TimeoutExpired, OSError):
        return None, -1                              # hang/missing-binary -> unsafe


def collect(handle, offset_mhz: int, *, bw_bin: str = DEFAULT_BW_BIN, iters: int = 200,
            fill_headroom_gib: float = silicon._FILL_HEADROOM_GIB,
            golden_fn: Optional[Callable[[str], bool]] = None,
            junction_fn: Optional[Callable[[str], Optional[float]]] = None,
            set_fn: Optional[Callable] = None, free_fn: Optional[Callable] = None,
            bw_fn: Optional[Callable] = None) -> Measurement:  # pragma: no cover - GPU-bound
    """Apply ``offset_mhz`` on this GPU, FILL its VRAM, soak it, gather the gate signals.

    Per-GPU by construction (``handle`` = the GPU UUID; bw_verify pinned via CUDA_VISIBLE_DEVICES).
    VRAM is auto-detected and (near) FILLED so the back-side GDDR6X chips are actually exercised
    (silicon.fill_gib). golden_fn (exact token-id under VLLM_BATCH_INVARIANT) is THE correctness
    gate when a model is up; without it the verdict leans on mismatch/junction/knee (weaker —
    document it). All side effects are injectable so the assembly is unit-tested GPU-free.
    """
    uuid = handle if isinstance(handle, str) else (getattr(handle, "uuid", None) or str(handle))
    set_fn = set_fn or silicon.set_mem_offset_mhz
    free_fn = free_fn or silicon.mem_free_bytes
    bw_fn = bw_fn or (lambda u, g: run_bw_verify(u, g, iters, bw_bin))
    set_rc, _ = set_fn(uuid, offset_mhz)
    size_gib = silicon.fill_gib(free_fn(uuid), fill_headroom_gib)
    bw = bw_fn(uuid, size_gib) if set_rc == 0 else (None, 0)
    junction = junction_fn(uuid) if junction_fn else None
    # tri-state: None when no golden harness is wired (do NOT claim correctness we didn't measure;
    # evaluate() then leans on bw_verify mismatch). On GDDR6X the caller SHOULD wire golden_fn.
    golden_ok = bool(golden_fn(uuid)) if golden_fn else None
    return assemble(offset_mhz, bw, golden_ok=golden_ok, junction_c=junction, set_rc=set_rc)


# re-export for callers that want the gate-family vocab without importing preflight.sku
GATE_GOLDEN_EDR_JUNCTION = _sku.GATE_GOLDEN_EDR_JUNCTION
GATE_ECC_GOLDEN = _sku.GATE_ECC_GOLDEN
GATE_ECC = _sku.GATE_ECC
__all__ = [
    "PASS", "KNEE", "REJECT", "ABORT", "Thresholds", "Measurement", "GateVerdict",
    "evaluate", "collect", "GATE_GOLDEN_EDR_JUNCTION", "GATE_ECC_GOLDEN", "GATE_ECC",
]
