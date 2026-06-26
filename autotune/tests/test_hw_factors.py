"""HALF-A measured-HW-factor decode model — validated against the real 9B-w4a8 mem-OC run."""
import pytest

from ampere_autotune.half_a.hw_factors import (
    fit_decode_two_points, decode_from_one_point, memoc_decode_gain_pct, DecodeModel,
    ridge_batch, prefill_ceiling_toks, max_num_seqs_from_ridge, MeasuredHw,
)

# the real measurement this session: single-stream 9B-w4a8, offset 0 (838 GB/s) vs +1000 (888).
REAL = [(838.0, 85.0), (888.0, 88.2)]


def test_two_point_fit_recovers_the_split():
    m = fit_decode_two_points(REAL)
    assert m is not None
    assert m.toks(838.0) == pytest.approx(85.0, abs=0.5)
    assert m.toks(888.0) == pytest.approx(88.2, abs=0.5)
    # decode is ~64% bandwidth-bound at the stock clock (the rest is fixed compute)
    assert m.bw_bound_frac(838.0) == pytest.approx(0.64, abs=0.03)
    # even at infinite bandwidth, single-stream caps ~239 tok/s (the compute wall)
    assert m.compute_ceiling_toks() == pytest.approx(239.0, abs=10.0)


def test_memoc_gain_matches_measured_plus_3_8_pct():
    m = fit_decode_two_points(REAL)
    assert memoc_decode_gain_pct(m, 838.0, 888.0) == pytest.approx(3.8, abs=0.3)
    # a bigger mem-OC (838->950) predicts a larger but still sub-proportional gain
    g = memoc_decode_gain_pct(m, 838.0, 950.0)
    assert 5.0 < g < (950 / 838 - 1) * 100   # below the proportional 13.4% (compute caps it)


def test_one_point_split_with_weight_bytes():
    # one point + ~6.35 GB/token weight read -> same split as the two-point fit
    m = decode_from_one_point(85.0, 838.0, 6.35e9)
    assert m is not None and m.toks(838.0) == pytest.approx(85.0, abs=0.5)
    assert m.bw_bound_frac(838.0) == pytest.approx(0.64, abs=0.03)


def test_one_point_rejects_inconsistent_bytes():
    # 20 GB/token weight read alone would exceed the measured TPOT -> inconsistent -> None
    assert decode_from_one_point(85.0, 838.0, 20e9) is None


def test_fit_rejects_single_bandwidth():
    assert fit_decode_two_points([(838.0, 85.0), (838.0, 85.1)]) is None


def test_compute_bound_model_shows_tiny_memoc_gain():
    # a model dominated by fixed compute (small bw_coef) -> mem-OC barely helps
    m = DecodeModel(bw_coef=0.5, fixed_t=0.01)
    assert m.bw_bound_frac(838.0) < 0.1
    assert memoc_decode_gain_pct(m, 838.0, 950.0) < 1.0


# ---- actual COMPUTE -> max-num-seqs (the throughput-side factor) ------------------------------

def test_ridge_batch_predicts_the_max_num_seqs_ceiling():
    # 3090 ~284 INT8 TOPS, 838 GB/s achievable, int4 weights (0.5 B/param) -> ridge ~85,
    # matching the project's empirical max-num-seqs<=82.
    rb = ridge_batch(284e12, 838.0, 0.5)
    assert rb == pytest.approx(85.0, abs=8.0)


def test_ridge_scales_with_compute_and_inverse_bandwidth():
    base = ridge_batch(284e12, 838.0, 0.5)
    assert ridge_batch(2 * 284e12, 838.0, 0.5) == pytest.approx(2 * base)   # 2x compute -> 2x ridge
    assert ridge_batch(284e12, 2 * 838.0, 0.5) == pytest.approx(base / 2)   # 2x bw -> half ridge


def test_max_num_seqs_picks_min_of_ridge_and_capacity():
    assert max_num_seqs_from_ridge(85.0, 200) == (85, "compute-ridge")     # compute caps first
    assert max_num_seqs_from_ridge(85.0, 32) == (32, "KV-capacity")        # KV memory caps first


def test_prefill_ceiling_positive():
    assert prefill_ceiling_toks(284e12, 9.0) > 1000   # 9B prefill compute ceiling (tok/s)


def test_measured_hw_ridge_uses_live_values_not_spec():
    # the ridge comes from LIVE-measured bw + compute at the under-load clock (P2 9501, not spec 9751)
    hw = MeasuredHw(bw_gbs=838.0, tflops=250.0, sm_mhz=1950, mem_mhz=9501, compute_dtype="int8")
    assert hw.ridge(0.5) == pytest.approx(ridge_batch(250e12, 838.0, 0.5))
    assert hw.mem_mhz == 9501            # proof it's the under-load clock, not the 9751 spec


def test_measured_hw_ridge_none_without_measurement():
    assert MeasuredHw(bw_gbs=None, tflops=250.0, sm_mhz=None, mem_mhz=None).ridge(0.5) is None
    assert MeasuredHw(bw_gbs=838.0, tflops=None, sm_mhz=None, mem_mhz=None).ridge(0.5) is None


def test_gemm_bench_script_is_valid_int8():
    from ampere_autotune.half_a.hw_factors import _gemm_script
    s = _gemm_script("int8", 4096, 80)
    assert "torch._int_mm" in s and "4096" in s and "tflops" in s
    compile(s, "<gemm>", "exec")   # the bench we ship into the serving image must be valid python
