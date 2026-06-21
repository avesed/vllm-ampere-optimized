"""Adaptive search state machine — pure, NO GPU. Drives the search with synthetic
measurement functions and asserts it converges to the right accepted offset + guard band."""
import pytest

from ampere_autotune.half_b import search, gate
from ampere_autotune.half_b.gate import Measurement, Thresholds
from ampere_autotune.half_b.search import characterize, COARSE, FINE, snap
from ampere_autotune.preflight import sku

GX = sku.GATE_GOLDEN_EDR_JUNCTION
G6 = sku.GATE_ECC_GOLDEN
GUARD = 45  # default 3 ticks


def gddr6x(knee_off=10_000, corrupt_off=None, junc_coeff=0.01):
    """BW rises +3% per coarse step until knee then flat; junction = 70 + coeff*off."""
    def f(off):
        steps_below_knee = min(off, knee_off) / COARSE
        gbs = 936.0 * (1 + 0.03 * steps_below_knee)
        golden = corrupt_off is None or off < corrupt_off
        return Measurement(off, golden_ok=golden, mismatch_count=0,
                           read_gbs=gbs, junction_c=70.0 + junc_coeff * off)
    return f


def test_snap_to_15():
    assert snap(0) == 0 and snap(7) == 0 and snap(8) == 15 and snap(105) == 105 and snap(100) == 105
    assert COARSE == 105 and FINE == 15   # fine phase probes EVERY 15 MHz tick (contiguity)


def test_knee_no_corruption():
    # BW knees at 630; golden always ok -> stop at KNEE, accept = last_pass(630) - guard
    r = characterize(gddr6x(knee_off=630), gate_family=GX, max_offset_mhz=1500)
    assert r.stop_kind == gate.KNEE
    assert r.boundary_offset_mhz == 630
    assert r.accepted_offset_mhz == 630 - GUARD == 585
    assert not r.aborted


def test_corruption_before_knee_fine_ascent():
    # golden fails at >=420; BW never knees -> REJECT at 420, contiguous fine ascent to 405
    r = characterize(gddr6x(knee_off=10_000, corrupt_off=420), gate_family=GX, max_offset_mhz=1500)
    assert r.stop_kind == gate.REJECT
    assert r.stop_offset_mhz == 420
    assert r.boundary_offset_mhz == 405       # highest 15 MHz tick strictly below corruption
    assert r.accepted_offset_mhz == 405 - GUARD == 360


def test_junction_abort_zeros_offset():
    # junction crosses 95 at +525 (70 + 0.05*525 = 96.25) before any corruption
    r = characterize(gddr6x(knee_off=10_000, junc_coeff=0.05), gate_family=GX, max_offset_mhz=1500)
    assert r.aborted and r.stop_kind == gate.ABORT
    assert r.accepted_offset_mhz == 0


def test_stock_not_clean_aborts():
    def dirty(off):
        return Measurement(off, golden_ok=(off != 0), mismatch_count=(1 if off == 0 else 0),
                           read_gbs=936.0, junction_c=70.0)
    r = characterize(dirty, gate_family=GX, max_offset_mhz=1500)
    assert r.aborted and "stock not clean" in r.abort_reason
    assert r.accepted_offset_mhz == 0


def test_gddr6_no_knee_climbs_to_corruption():
    # GDDR6 workstation: read_gbs=None, no EDR knee; climb until golden fails at 315
    def g6(off):
        return Measurement(off, golden_ok=(off < 315), mismatch_count=0,
                           read_gbs=None, junction_c=70.0)
    r = characterize(g6, gate_family=G6, max_offset_mhz=1500)
    assert r.stop_kind == gate.REJECT
    assert r.boundary_offset_mhz == 300       # highest tick below the 315 corruption
    assert r.accepted_offset_mhz == 300 - GUARD == 255


def test_accepted_never_negative():
    # corruption very low so boundary - guard would go negative -> clamp to 0
    def low(off):
        return Measurement(off, golden_ok=(off < COARSE), mismatch_count=0,
                           read_gbs=936.0 * (1 + 0.03 * off / COARSE), junction_c=70.0)
    r = characterize(low, gate_family=GX, max_offset_mhz=1500)
    assert r.accepted_offset_mhz >= 0


def test_monotone_offsets_are_tick_multiples():
    r = characterize(gddr6x(knee_off=630), gate_family=GX, max_offset_mhz=1500)
    assert r.accepted_offset_mhz % search.TICK == 0
    assert r.boundary_offset_mhz % search.TICK == 0


def test_intermittent_golden_is_rejected():
    # C1: an offset that PASSes one probe but FAILs another must not be trusted (worst-of-N).
    state = {}

    def f(off):
        state[off] = state.get(off, 0) + 1
        golden = not (off == 420 and state[off] == 2)  # 420 fails its 2nd of 3 probes
        return Measurement(off, golden_ok=golden, mismatch_count=0,
                           read_gbs=936.0 * (1 + 0.03 * off / COARSE), junction_c=70.0)

    r = characterize(f, gate_family=GX, max_offset_mhz=1500, samples=3)
    assert r.boundary_offset_mhz < 420          # the intermittent fail was caught


def test_nonmonotonic_corruption_below_reject_not_bypassed():
    # C2: corruption at tick 360 (below the coarse REJECT at 420), "safe" again at 390.
    # Contiguous ascent must stop at 345 and never bypass 360.
    def f(off):
        golden = off not in (360, 420)
        return Measurement(off, golden_ok=golden, mismatch_count=0,
                           read_gbs=936.0 * (1 + 0.03 * off / COARSE), junction_c=70.0)

    r = characterize(f, gate_family=GX, max_offset_mhz=1500, samples=1)
    assert r.stop_kind == gate.REJECT
    assert r.boundary_offset_mhz == 345         # last contiguous-safe tick before 360
    assert r.boundary_offset_mhz < 360


def test_maxed_reports_ceiling_limited():
    # M1: corruption-free with a low max_offset -> MAXED + ceiling_limited (gain is a lower bound)
    r = characterize(gddr6x(knee_off=10_000), gate_family=GX, max_offset_mhz=210)
    assert r.stop_kind == "MAXED"
    assert r.ceiling_limited is True
    assert r.boundary_offset_mhz == 210


def test_guard_cannot_exceed_boundary():
    # C3: negative guard ticks are rejected at construction...
    with pytest.raises(ValueError):
        Thresholds(thermal_guard_ticks=-5)
    # ...and even a huge valid guard clamps accepted into [0, boundary].
    big = Thresholds(thermal_guard_ticks=100, coverage_guard_ticks=100, hysteresis_ticks=100)
    r = characterize(gddr6x(knee_off=630), gate_family=GX, max_offset_mhz=1500, th=big)
    assert 0 <= r.accepted_offset_mhz <= r.boundary_offset_mhz
