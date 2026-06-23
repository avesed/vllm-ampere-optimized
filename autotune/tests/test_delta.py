"""Closed-loop delta — pure render (no filesystem)."""
from ampere_autotune.half_a.delta import render_delta


def test_render_delta_improvement():
    out = render_delta({"single_tps": 80.0}, {"single_tps": 92.0})
    assert "80.0 -> 92.0" in out and "+15%" in out and "improved" in out


def test_render_delta_flags_regression():
    out = render_delta({"single_tps": 92.0}, {"single_tps": 80.0})   # higher-is-better, dropped >5%
    assert "REGRESSION" in out


def test_render_delta_stable_within_noise():
    out = render_delta({"single_tps": 65.2}, {"single_tps": 65.3})   # +0.1% -> noise
    assert "stable" in out and "improved" not in out


def test_render_delta_no_comparable_prior():
    assert "no comparable" in render_delta({"x": 1.0}, {"y": 2.0})
