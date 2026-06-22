"""Economics block — pure (no GPU)."""
from ampere_autotune.half_a.economics import metrics, fmt


def test_metrics_energy_needs_only_power():
    m = metrics(100.0, 300.0)                       # 100 tok/s @ 300 W
    assert abs(m["j_per_tok"] - 3.0) < 1e-6         # 300 W / 100 tok/s
    assert abs(m["tok_per_w"] - (1 / 3)) < 1e-6
    assert "usd_per_mtok" not in m                  # no price -> no $


def test_metrics_dollars():
    m = metrics(100.0, 300.0, cost_per_hr=0.36)     # $/1M = 0.36*1e6/(100*3600) = 1.0
    assert abs(m["usd_per_mtok"] - 1.0) < 1e-6


def test_metrics_empty_without_power():
    assert metrics(100.0, None) == {}
    assert metrics(None, 300.0) == {}


def test_fmt_blank_without_power():
    assert fmt(100.0, None) == ""


def test_fmt_includes_energy_and_dollar():
    s = fmt(100.0, 300.0, 0.36)
    assert "tok/W" in s and "J/tok" in s and "/1M" in s and "1.00" in s


def test_fmt_hints_for_cost_when_absent():
    s = fmt(100.0, 300.0)
    assert "J/tok" in s and "--cost-per-hour" in s
