"""HALF-A co-tuning sweep — pure helpers (no server)."""
import pytest

from ampere_autotune.half_a.cotune import (
    parse_sweep, expand_grid, config_flags, score, render, make_restart_fn, SweepPoint,
    auto_tune, render_curve, render_lowc_advice, banned_in, Trial,
)


def test_banned_flags_detected_with_replacements():
    bad = banned_in({"--max-num-seqs": ["64"], "--swap-space": ["4"], "--num-scheduler-steps": ["8"],
                     "--cuda-graph-sizes": ["1,2"]})
    assert set(bad) == {"--swap-space", "--num-scheduler-steps", "--cuda-graph-sizes"}   # not max-num-seqs
    assert bad["--cuda-graph-sizes"] == "--cudagraph-capture-sizes"                       # points to the real name


def _quiet(*a, **k):
    pass


def test_auto_predict_verify_takes_knee_not_wall():
    # throughput PLATEAUS at 128 though the KV wall is far -> take the knee. kv-dtype is NOT swept.
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        tps = 3000.0 if seqs >= 128 else 3000.0 * seqs / 128.0   # hard plateau at 128
        return Trial(cfg, tps, True, seqs / 512.0, 0.0)

    best, hist, recs = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "128"             # the throughput knee
    assert "--kv-cache-dtype" not in best.config             # never auto-swept


def test_auto_recommends_fp8_when_capacity_bound_but_never_sweeps_it():
    # capacity-bound (still rising at the KV wall) -> fp8 is a RECOMMENDATION, NOT in the swept config/history
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        if seqs > 140:
            return Trial(cfg, float("-inf"), False, note="OOM")
        tps = 4000.0 * seqs / (seqs + 200.0)                     # still rising at the wall
        return Trial(cfg, tps, True, seqs / 140.0, 0.0)

    best, hist, recs = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert "--kv-cache-dtype" not in best.config                 # not swept
    assert any("fp8" in r for r in recs)                         # but recommended (opt-in)
    assert all("kv-cache-dtype" not in config_flags(t.config) for t in hist)   # never tried in the sweep


def test_auto_returns_none_when_everything_thrashes():
    # H1: a feasible-but-thrashing config (score -inf) must NEVER be returned as best
    def fake(cfg):
        return Trial(cfg, float("-inf"), True, 0.99, 0.5, "thrash")   # feasible but preempting
    best, hist, recs = auto_tune(fake, seed_seqs=32, log=_quiet)
    assert best is None


def test_auto_does_not_reprobe_a_measured_oom_config():
    # H3: when s*2 OOMs at probe, the climb must not re-probe that config
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        if seqs > 40:
            return Trial(cfg, float("-inf"), False, note="OOM")
        return Trial(cfg, seqs * 10.0, True, seqs / 100.0, 0.0)

    best, hist, recs = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert len([t for t in hist if t.config.get("--max-num-seqs") == "64"]) == 1


def test_auto_patience_passes_a_noisy_flat_rung():
    # M9: one flat intermediate rung must not end the ascent when a higher rung is much better
    tbl = {32: 800, 64: 1500, 128: 2000, 256: 2010, 512: 3000}      # 256 ~flat vs 128, 512 jumps
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        return Trial(cfg, float(tbl.get(seqs, 3000)), True, seqs / 2000.0, 0.0)   # KV tiny -> wall huge

    best, hist, recs = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "512"


def test_auto_probe_halves_down_when_seed_ooms():
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        if seqs > 40:
            return Trial(cfg, float("-inf"), False, note="OOM")
        return Trial(cfg, seqs * 10.0, True, seqs / 100.0, 0.0)

    best, hist, recs = auto_tune(fake, seed_seqs=128, seqs_ceiling=512, log=_quiet)   # seed OOMs -> halve
    assert best is not None and int(best.config["--max-num-seqs"]) <= 40


def test_render_curve_shows_per_session_and_tpot():
    out = render_curve([(1, 85.0, 85.0), (128, 3060.0, 23.9)])
    assert "per-session" in out and "TPOT" in out
    assert "85" in out and "3060" in out and "11.8" in out      # TPOT@85 tok/s ~ 11.8 ms
    assert "SLA" in out                                          # the operating-point guidance


def test_render_lowc_advice_recommends_mtp_keeps_cudagraph_no_sweep():
    out = render_lowc_advice(85.0, 1)
    assert "85" in out and "TPOT" in out
    assert "MTP" in out and "enforce-eager" in out              # opt-in MTP lever + cudagraph guardrail
    assert "auto-sweep" in out.lower() or "opt-in" in out.lower()


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
