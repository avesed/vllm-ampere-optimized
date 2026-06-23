"""Gate decision logic — pure, no GPU. Verifies the FIRST-of climb-stop ordering."""
from ampere_autotune.half_b import gate
from ampere_autotune.half_b.gate import Measurement, Thresholds, evaluate
from ampere_autotune.preflight import sku

TH = Thresholds()
GX = sku.GATE_GOLDEN_EDR_JUNCTION
G6 = sku.GATE_ECC_GOLDEN


def m(off=300, golden=True, mismatch=0, gbs: "float | None" = 900.0, junc=70.0, xid=False):
    return Measurement(off, golden_ok=golden, mismatch_count=mismatch, read_gbs=gbs,
                       junction_c=junc, xid=xid)


def test_xid_aborts_even_if_golden_ok():
    assert evaluate(m(xid=True), None, TH, GX).status == gate.ABORT


def test_junction_abort_takes_precedence_over_golden():
    # junction >= 95 -> ABORT even though golden_ok and mismatch 0
    assert evaluate(m(junc=95.0), None, TH, GX).status == gate.ABORT


def test_golden_mismatch_is_reject():
    assert evaluate(m(golden=False), None, TH, GX).status == gate.REJECT


def test_cell_mismatch_is_reject():
    assert evaluate(m(mismatch=3), None, TH, GX).status == gate.REJECT


def test_reject_beats_knee():
    # even if BW also flat (would be knee), a golden fail must REJECT, not KNEE
    prev = m(off=200, gbs=900.0)
    cur = m(off=300, golden=False, gbs=900.0)  # flat BW + golden fail
    assert evaluate(cur, prev, TH, GX).status == gate.REJECT


def test_knee_when_bw_stops_rising():
    prev = m(off=200, gbs=900.0)
    cur = m(off=300, gbs=901.0)  # +0.11% < 0.3% -> knee (real GDDR6X climbs ~0.5%/105MHz)
    assert evaluate(cur, prev, TH, GX).status == gate.KNEE


def test_pass_when_bw_still_rising():
    prev = m(off=200, gbs=900.0)
    cur = m(off=300, gbs=930.0)  # +3.3% -> still climbing
    assert evaluate(cur, prev, TH, GX).status == gate.PASS


def test_gddr6_has_no_knee_check():
    # plain GDDR6 (ECC_GOLDEN): even flat BW is PASS (no EDR knee on GDDR6)
    prev = m(off=200, gbs=900.0)
    cur = m(off=300, gbs=900.0)
    assert evaluate(cur, prev, TH, G6).status == gate.PASS


def test_descent_prev_none_skips_knee():
    # prev=None (fine descent) -> only safety checks; flat BW is fine
    assert evaluate(m(gbs=900.0), None, TH, GX).status == gate.PASS


def test_guard_band_default_three_ticks():
    assert TH.guard_band_mhz(15) == 45


def test_knee_inconclusive_bw_is_not_pass():
    # H1: on GDDR6X a missing/zero bandwidth reading must STOP (KNEE), never fall to PASS.
    prev = m(off=200, gbs=900.0)
    assert evaluate(m(off=300, gbs=None), prev, TH, GX).status == gate.KNEE      # cur BW missing
    prev0 = m(off=200, gbs=0.0)
    assert evaluate(m(off=300, gbs=900.0), prev0, TH, GX).status == gate.KNEE    # prev BW zero


def test_negative_or_zero_guard_rejected():
    # C3: guard ticks must be non-negative and total >= 1 tick.
    import pytest
    with pytest.raises(ValueError):
        Thresholds(thermal_guard_ticks=-1)
    with pytest.raises(ValueError):
        Thresholds(thermal_guard_ticks=0, coverage_guard_ticks=0, hysteresis_ticks=0)
