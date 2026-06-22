"""HALF-A co-tuning sweep — pure helpers (no server)."""
import pytest

from ampere_autotune.half_a.cotune import (
    parse_sweep, expand_grid, config_flags, score, render, make_restart_fn, SweepPoint,
    auto_tune, Trial,
)


def _quiet(*a, **k):
    pass


def test_auto_tune_adaptive_finds_optimum_without_grid():
    # fake server: tps rises with seqs to 256; OOM past 128 unless fp8; mnbt=8192 adds 10%.
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        fp8 = cfg.get("--kv-cache-dtype") == "fp8"
        mnbt = int(cfg.get("--max-num-batched-tokens", "0"))
        if seqs > 256 or (seqs > 128 and not fp8):
            return Trial(cfg, float("-inf"), False, note="OOM")
        tps = min(seqs, 256) * 10.0 * (1.10 if mnbt == 8192 else 1.0)
        kv = min(0.99, seqs / (256.0 if fp8 else 128.0))
        return Trial(cfg, tps, True, kv, 0.0)

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=256, log=_quiet)
    assert best.config["--max-num-seqs"] == "256"               # expanded past the 128 OOM wall
    assert best.config["--kv-cache-dtype"] == "fp8"             # AUTO-applied the coupled unlock
    assert best.config["--max-num-batched-tokens"] == "8192"    # secondary knob picked up
    assert len(hist) >= 5                                       # it actually searched


def test_auto_tune_stops_at_plateau_without_unlocking():
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        return Trial(cfg, min(seqs, 64) * 10.0, True, min(0.5, seqs / 512.0), 0.0)  # plateaus at 64, KV never high

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=256, log=_quiet)
    assert best.config["--max-num-seqs"] == "64"
    assert "--kv-cache-dtype" not in best.config               # no wall -> no unlock attempted


def test_parse_sweep():
    g = parse_sweep("--max-num-seqs=32,64,96;--kv-cache-dtype=auto,fp8")
    assert g == {"--max-num-seqs": ["32", "64", "96"], "--kv-cache-dtype": ["auto", "fp8"]}


def test_parse_sweep_rejects_bad_segment():
    with pytest.raises(ValueError):
        parse_sweep("--max-num-seqs")


def test_expand_grid_cartesian():
    grid = expand_grid({"--max-num-seqs": ["32", "64"], "--kv-cache-dtype": ["auto", "fp8"]})
    assert len(grid) == 4
    assert {"--max-num-seqs": "64", "--kv-cache-dtype": "fp8"} in grid


def test_config_flags_omits_defaulty_values():
    assert config_flags({"--max-num-seqs": "64", "--kv-cache-dtype": "auto"}) == "--max-num-seqs 64"
    assert config_flags({"--max-num-seqs": "96", "--kv-cache-dtype": "fp8"}) == "--max-num-seqs 96 --kv-cache-dtype fp8"


def test_config_flags_store_true_toggle():
    # enforce-eager=true -> bare flag (no value); false -> omitted (cudagraph stays on = default)
    assert config_flags({"--max-num-seqs": "128", "--enforce-eager": "true"}) == "--max-num-seqs 128 --enforce-eager"
    assert config_flags({"--max-num-seqs": "128", "--enforce-eager": "false"}) == "--max-num-seqs 128"


def test_score_infeasible_is_neg_inf():
    assert score(SweepPoint({}, feasible=False)) == float("-inf")


def test_score_throughput_disqualifies_thrashing():
    thrash = SweepPoint({}, feasible=True, decode_tps_max_c=2000.0, preempt_per_s=0.5)
    clean = SweepPoint({}, feasible=True, decode_tps_max_c=1700.0, preempt_per_s=0.0)
    assert score(thrash, "throughput") == float("-inf")   # preemption disqualifies
    assert score(clean, "throughput") == 1700.0


def test_score_latency_uses_single_stream():
    p = SweepPoint({}, feasible=True, decode_tps_single=90.0, decode_tps_max_c=1700.0)
    assert score(p, "latency") == 90.0


def test_render_marks_best_and_fails():
    pts = [
        SweepPoint({"--max-num-seqs": "96"}, feasible=True, decode_tps_max_c=1900.0, preempt_per_s=0.0),
        SweepPoint({"--max-num-seqs": "64"}, feasible=True, decode_tps_max_c=1700.0, preempt_per_s=0.0),
        SweepPoint({"--max-num-seqs": "128"}, feasible=False, note="never became ready (OOM)"),
    ]
    out = render(pts, "throughput")
    assert "WIN" in out and "FAIL" in out
    assert "BEST: --max-num-seqs 96" in out


def test_make_restart_fn_requires_placeholder():
    with pytest.raises(ValueError):
        make_restart_fn("docker run ... no placeholder", "http://x")
