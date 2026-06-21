"""HALF-B adaptive clock search — coarse-up / fine-up + layered guard band.

PURE control logic: the GPU lives entirely behind an injected ``measure_fn(offset_mhz) ->
Measurement`` callback, so the whole state machine is unit-testable with NO GPU (feed it a
synthetic measurement function; see tests/test_search.py). In production the callback is
``functools.partial(gate.collect, handle)`` and SHOULD itself soak each offset.

Algorithm (objective = max decode tok/s at ZERO sustained errors = the EDR knee, BELOW
corruption; all offsets snap to the 15 MHz KMD tick):
  0. EVERY offset is probed ``samples`` times and combined WORST-of-N (any golden fail /
     mismatch / Xid / hottest junction wins; bandwidth = median, robust vs a single dip).
     A single passing probe is never trusted (C1).
  1. baseline at 0 must be clean — else abort.
  2. COARSE climb (+105 MHz) while each step PASSes; stop on the FIRST of {KNEE|REJECT|ABORT}.
  3. CONTIGUOUS fine ascent (+15 MHz, every settable clock) from the last coarse PASS up to
     the stop point; boundary = the highest offset with NO reject anywhere at-or-below it
     (so a lower non-monotonic corruption can never be bypassed — C2).
  4. RE-SOAK the boundary; back off a tick on failure (C1).
  5. accepted = clamp(boundary - guard_band, 0, boundary)   (never > boundary — C3).
ABORT at any point -> return to stock (0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import gate as _gate
from .gate import Measurement, Thresholds, GateVerdict, PASS, REJECT, ABORT

TICK = 15
COARSE = 7 * TICK   # 105 MHz climb step
FINE = TICK         # 15 MHz: the fine phase probes EVERY settable clock (contiguity, C2)
DEFAULT_SAMPLES = 3

MeasureFn = Callable[[int], Measurement]


def snap(mhz: int, tick: int = TICK) -> int:
    """Round to the nearest KMD tick multiple (open-kernel-module snaps offsets to 15 MHz)."""
    return int(round(mhz / tick)) * tick


def _combine(measure_fn: MeasureFn, off: int, samples: int) -> Measurement:
    """Probe ``off`` ``samples`` times -> a WORST-of-N Measurement (C1).

    Safety fields take the worst sample (any golden fail / max mismatch / any Xid / hottest
    junction); bandwidth takes the MEDIAN (robust against a single dip false-knee).
    """
    n = max(1, samples)
    ms = [measure_fn(off) for _ in range(n)]
    bws = sorted(m.read_gbs for m in ms if m.read_gbs is not None)
    juncs = [m.junction_c for m in ms if m.junction_c is not None]
    return Measurement(
        offset_mhz=off,
        golden_ok=all(m.golden_ok for m in ms),
        mismatch_count=max(m.mismatch_count for m in ms),
        read_gbs=(bws[len(bws) // 2] if bws else None),
        junction_c=(max(juncs) if juncs else None),
        xid=any(m.xid for m in ms),
        throttle=any(m.throttle for m in ms),
    )


@dataclass
class SearchResult:
    accepted_offset_mhz: int          # the offset to deploy (boundary - guard band)
    boundary_offset_mhz: int          # highest offset confirmed correct (contiguous below)
    guard_band_mhz: int
    stop_kind: str                    # KNEE | REJECT | ABORT | MAXED
    stop_offset_mhz: Optional[int]
    ceiling_limited: bool = False     # MAXED == hit user max_offset (gain is a LOWER bound), not silicon
    aborted: bool = False
    abort_reason: str = ""
    trace: List[str] = field(default_factory=list)


def characterize(measure_fn: MeasureFn, *, gate_family: str, max_offset_mhz: int = 1500,
                 th: Optional[Thresholds] = None, samples: int = DEFAULT_SAMPLES) -> SearchResult:
    th = th or Thresholds()
    max_offset_mhz = snap(max_offset_mhz)
    guard = th.guard_band_mhz(TICK)
    trace: List[str] = []

    def meas(off: int) -> Measurement:
        return _combine(measure_fn, off, samples)

    def log(off: int, v: GateVerdict, phase: str) -> None:
        trace.append(f"{phase} off=+{off} -> {v.status}: {v.reason}")

    def _abort(off, v) -> SearchResult:
        return SearchResult(0, 0, guard, ABORT, off, aborted=True, abort_reason=v.reason, trace=trace)

    # --- 1. baseline must be clean ---
    base = meas(0)
    base_v = _gate.evaluate(base, None, th, gate_family)
    log(0, base_v, "BASE")
    if not base_v.is_safe:
        r = _abort(0, base_v)
        r.abort_reason = f"stock not clean: {base_v.reason}"
        return r

    # --- 2. COARSE climb ---
    prev = base
    last_pass = 0
    off = 0
    stop_kind = "MAXED"
    stop_off: Optional[int] = None
    stopped = False
    while off + COARSE <= max_offset_mhz:
        cand = off + COARSE
        m = meas(cand)
        v = _gate.evaluate(m, prev, th, gate_family)
        log(cand, v, "CLIMB")
        if v.status == PASS:
            last_pass, prev, off = cand, m, cand
            continue
        stop_kind, stop_off, stopped = v.status, cand, True
        if v.status == ABORT:
            r = _abort(cand, v)
            r.boundary_offset_mhz = last_pass
            return r
        break
    ceiling_limited = not stopped  # never stopped -> we hit the user max_offset, not silicon

    # --- 3. CONTIGUOUS fine ascent (only after a REJECT bracket) ---
    boundary = last_pass
    if stopped and stop_kind == REJECT and stop_off is not None:
        cur = snap(last_pass + FINE)
        while cur < stop_off:
            m = meas(cur)
            v = _gate.evaluate(m, None, th, gate_family)  # safety-only (knee needs prev)
            log(cur, v, "FINE")
            if v.status == ABORT:
                r = _abort(cur, v)
                r.boundary_offset_mhz = last_pass
                return r
            if not v.is_safe:
                break  # first unsafe -> everything strictly below is confirmed clean
            boundary = cur
            cur = snap(cur + FINE)

    # --- 4. re-soak the boundary; back off a tick on failure (intermittency insurance) ---
    while boundary > 0:
        m = meas(boundary)
        v = _gate.evaluate(m, None, th, gate_family)
        if v.status == ABORT:
            r = _abort(boundary, v)
            r.boundary_offset_mhz = last_pass
            return r
        if v.is_safe:
            break
        log(boundary, v, "RESOAK-FAIL")
        boundary = snap(boundary - FINE)

    # --- 5. accepted, clamped so it can never exceed the confirmed boundary (C3) ---
    accepted = max(0, min(boundary, snap(boundary - guard)))
    return SearchResult(accepted, boundary, guard, stop_kind, stop_off,
                        ceiling_limited=ceiling_limited, trace=trace)
