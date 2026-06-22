"""No-root advisory — pure logic, no GPU. Covers each branch + the load-bearing safety
invariants (no bare apply-ready offset; benefit lines carry the ungated-OC warning)."""
from ampere_autotune.half_b.advise import (
    Measurements, SkuInfo, Roofline, advise, render, _BARE_OFFSET,
)
from ampere_autotune.preflight import sku

GEFORCE_GDDR6X = SkuInfo(sku.SKU_GEFORCE, sku.MEM_GDDR6X, sku.OFFSET_SUPPORTED)
WORKSTATION = SkuInfo(sku.SKU_WORKSTATION, sku.MEM_GDDR6, sku.OFFSET_SUPPORTED)
DATACENTER = SkuInfo(sku.SKU_DATACENTER, sku.MEM_HBM, sku.OFFSET_NOT_SUPPORTED)
ROOF = Roofline(sku_peak_gbs=936.0)


def _names(recs):
    return {r.name for r in recs}


def _clean_decode(**kw):
    base = dict(achieved_gbs=608.0, decode_toks=87.0, power_w=300.0, power_limit_w=350.0,
                core_temp_c=60.0, golden_ok=True, mismatch_count=0, bw_flat_across_batch=True)
    base.update(kw)
    return Measurements(**base)


def test_stock_fail_suppresses_silicon_section():
    recs = advise(_clean_decode(golden_ok=False), GEFORCE_GDDR6X, ROOF)
    assert _names(recs) == {"A-CORRECTNESS-stock-FAIL"}
    # next-action must point at integrity/health, NOT characterize
    assert "characterize" not in recs[0].action.lower()
    assert "sha256" in recs[0].action.lower()


def test_mismatch_also_suppresses():
    recs = advise(_clean_decode(mismatch_count=2), GEFORCE_GDDR6X, ROOF)
    assert _names(recs) == {"A-CORRECTNESS-stock-FAIL"}


def test_geforce_headroom_projection_fires_bandwidth_bound():
    recs = advise(_clean_decode(achieved_gbs=608.0), GEFORCE_GDDR6X, ROOF)  # 0.65 of peak
    n = _names(recs)
    assert "A-HEADROOM-mem-oc-projection" in n
    assert "A-SKU-geforce-gated-sweep" in n
    proj = next(r for r in recs if r.name == "A-HEADROOM-mem-oc-projection")
    assert "PROJECTION" in proj.message and "UNMEASURED" in proj.message
    assert "OR NOTHING" in proj.message            # honest zero floor
    assert "do NOT" in proj.message                # co-located ungated warning
    assert not _BARE_OFFSET.search(proj.message)   # no apply-ready MHz/Gbps


def test_not_bandwidth_bound_downgrades():
    recs = advise(_clean_decode(achieved_gbs=608.0, bw_flat_across_batch=False), GEFORCE_GDDR6X, ROOF)
    n = _names(recs)
    assert "A-HEADROOM-not-bandwidth-bound" in n
    assert "A-HEADROOM-mem-oc-projection" not in n


def test_thermal_suppresses_projection():
    recs = advise(_clean_decode(throttle_reasons=["HwThermalSlowdown"]), GEFORCE_GDDR6X, ROOF)
    n = _names(recs)
    assert "A-THERMAL-throttle-active" in n
    assert "A-HEADROOM-mem-oc-projection" not in n


def test_high_core_temp_counts_as_thermal():
    recs = advise(_clean_decode(core_temp_c=85.0), GEFORCE_GDDR6X, ROOF)
    assert "A-THERMAL-throttle-active" in _names(recs)


def test_near_knee_low_headroom():
    recs = advise(_clean_decode(achieved_gbs=880.0), GEFORCE_GDDR6X, ROOF)  # 0.94 of peak
    n = _names(recs)
    assert "A-NEAR-KNEE-low-headroom" in n
    assert "A-HEADROOM-mem-oc-projection" not in n


def test_datacenter_no_projection():
    recs = advise(_clean_decode(), DATACENTER, ROOF)
    n = _names(recs)
    assert "A-SKU-datacenter-locked" in n
    assert "A-HEADROOM-mem-oc-projection" not in n
    assert "A-SKU-geforce-gated-sweep" not in n


def test_workstation_ecc_reduces_not_eliminates():
    recs = advise(_clean_decode(), WORKSTATION, ROOF)
    r = next(r for r in recs if r.name == "A-SKU-workstation-ecc")
    assert "REDUCES but does NOT eliminate" in r.message
    assert "far safer" not in r.message           # adversarial fix: no unqualified claim
    assert "back off" in r.message


def test_power_perf_per_watt():
    recs = advise(_clean_decode(power_w=240.0, power_limit_w=350.0), GEFORCE_GDDR6X, ROOF)
    assert "A-POWER-perf-per-watt" in _names(recs)


def test_closing_actionable_never_a_bare_offset():
    recs = advise(_clean_decode(), GEFORCE_GDDR6X, ROOF)
    closing = next(r for r in recs if r.name == "A-ACTIONABLE-get-root-characterize")
    assert "characterize" in closing.action
    assert not _BARE_OFFSET.search(closing.message)
    assert not _BARE_OFFSET.search(closing.action)


def test_invariant_no_bare_offset_anywhere_and_render_ok():
    for skuinfo in (GEFORCE_GDDR6X, WORKSTATION, DATACENTER):
        for extra in ({}, {"throttle_reasons": ["HwThermalSlowdown"]}, {"achieved_gbs": 900.0},
                      {"bw_flat_across_batch": False}, {"power_w": 200.0}):
            recs = advise(_clean_decode(**extra), skuinfo, ROOF)
            for r in recs:
                assert not _BARE_OFFSET.search(r.message), (r.name, r.message)
                assert not _BARE_OFFSET.search(r.action), (r.name, r.action)
            out = render(recs)   # render asserts the invariants internally
            assert "recommend-only" in out


def test_golden_unchecked_telemetry_only():
    # telemetry-only run (real NVML, no vLLM golden, no bw): golden_ok=None -> "not checked",
    # no FAIL, no headroom projection; real thermal/power telemetry recs still fire.
    m = Measurements(golden_ok=None, achieved_gbs=None, decode_toks=None,
                     power_w=240.0, power_limit_w=350.0, core_temp_c=62.0)
    recs = advise(m, GEFORCE_GDDR6X, ROOF)
    n = _names(recs)
    assert "A-CORRECTNESS-not-checked" in n
    assert "A-CORRECTNESS-stock-FAIL" not in n
    assert "A-CORRECTNESS-stock-baseline" not in n
    assert "A-HEADROOM-mem-oc-projection" not in n   # never project without a clean stock golden
    assert "A-POWER-perf-per-watt" in n              # real telemetry still actionable


def test_golden_none_is_not_a_fail():
    # regression for the `is False` fix: None must NOT be treated as a correctness FAIL
    assert "A-CORRECTNESS-stock-FAIL" not in _names(advise(Measurements(golden_ok=None), GEFORCE_GDDR6X, ROOF))


def test_headroom_needs_clean_golden():
    # bandwidth-bound but golden unchecked -> still no projection (correctness baseline required)
    m = Measurements(golden_ok=None, achieved_gbs=608.0, bw_flat_across_batch=True)
    assert "A-HEADROOM-mem-oc-projection" not in _names(advise(m, GEFORCE_GDDR6X, ROOF))


def test_bw_peak_measured_rec():
    # bw_verify peak + integrity (no root, no vLLM): real peak-BW rec, % of spec, no bare offset
    m = Measurements(golden_ok=None, peak_gbs=768.0, mismatch_count=0,
                     power_w=300.0, power_limit_w=350.0, core_temp_c=60.0)
    r = next(x for x in advise(m, GEFORCE_GDDR6X, ROOF) if x.name == "A-BW-PEAK-measured")
    assert "82%" in r.message and "768 GB/s" in r.message     # 768/936 ≈ 82%
    assert "sub-proportional" in r.message and "gated sweep" in r.message
    assert not _BARE_OFFSET.search(r.message)


def test_bw_verify_mismatch_at_stock_is_fail():
    # bw_verify mismatch>0 at stock = silent VRAM corruption -> suppress silicon section
    recs = advise(Measurements(golden_ok=None, peak_gbs=768.0, mismatch_count=5), GEFORCE_GDDR6X, ROOF)
    assert _names(recs) == {"A-CORRECTNESS-stock-FAIL"}


def test_bare_offset_regex_catches_real_offsets():
    # guard the guard: the regex must actually catch the things we forbid
    assert _BARE_OFFSET.search("set +1000 MHz")
    assert _BARE_OFFSET.search("+2000 MT/s")
    assert _BARE_OFFSET.search("21 Gbps")
    assert not _BARE_OFFSET.search("65% of peak, up to +8% tok/s")
