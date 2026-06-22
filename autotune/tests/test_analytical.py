"""HALF-A analytical predictor — validated against the REAL sweep data measured on the 9B."""
from ampere_autotune.half_a.analytical import (
    Sample, predict, fit_kv_linear, fit_roofline, render, predict_from_baseline,
)


def test_predict_from_single_baseline_gives_wall():
    # ONE real point (the user's "spin up vLLM once as a baseline") -> feasible wall + fp8 unlock
    p = predict_from_baseline(Sample(128, 0.88, 3062.0))
    assert 130 < p.feasible_seqs < 145          # ~138, matches the 2-point fit
    assert 250 < p.feasible_seqs_fp8 < 300      # fp8 ~doubles
    assert p.feasible_seqs_fp8 >= 2 * p.feasible_seqs - 2


def test_baseline_needs_real_kv():
    assert predict_from_baseline(Sample(128, 0.0, 3000.0)) is None

# measured on Qwen3.5-9B-w4a8, 1x3090, ctx 4096, oversubscribed to 128 concurrency:
EMPIRICAL = [Sample(64, 0.42, 2622.0), Sample(128, 0.87, 3061.0)]


def test_kv_linear_fit_predicts_feasible_wall():
    a, b = fit_kv_linear(EMPIRICAL)
    assert abs(a - (0.45 / 64)) < 1e-4          # ~0.00703 KV per seq
    feasible = (0.95 - b) / a
    assert 130 < feasible < 150                  # ~139 max-num-seqs before 95% KV


def test_roofline_fit_ceiling_and_knee():
    c1, c2 = fit_roofline(EMPIRICAL)
    assert 3400 < 1.0 / c2 < 3900               # aggregate ceiling ~3676 tok/s


def test_predict_is_capacity_bound_and_recommends_fp8():
    p = predict(EMPIRICAL)
    assert p.capacity_bound is True              # feasible (~139) is below the throughput knee (~489)
    assert p.rec_fp8 is True                     # so halving KV bytes buys more concurrency -> throughput
    assert 130 < p.rec_seqs < 150               # push to ~the wall without fp8
    assert p.rec_seqs_fp8 > 250                  # fp8 ~doubles the wall
    assert p.pred_tps > 3061                     # predicted best beats the measured 128 point
    assert "--max-num-seqs" in render(p, EMPIRICAL)


def test_predict_roofline_bound_skips_fp8():
    # throughput flattens early, capacity is ample -> roofline-bound, fp8 pointless
    s = [Sample(16, 0.05, 1000.0), Sample(32, 0.10, 1050.0)]
    p = predict(s)
    assert p.capacity_bound is False
    assert p.rec_fp8 is False


def test_predict_needs_two_distinct_points():
    assert predict([Sample(64, 0.4, 2600.0)]) is None
