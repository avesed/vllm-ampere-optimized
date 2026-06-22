"""HALF-A co-tuning sweep — pure helpers (no server)."""
import pytest

from ampere_autotune.half_a.cotune import (
    parse_sweep, expand_grid, config_flags, score, render, make_restart_fn, SweepPoint,
    auto_tune, Trial,
)


def _quiet(*a, **k):
    pass


def test_auto_predict_verify_climbs_with_fp8_when_throughput_still_rising():
    # throughput genuinely keeps rising; KV wall ~128 (auto) / ~256 (fp8) -> capacity-bound, fp8 pays.
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        fp8 = cfg.get("--kv-cache-dtype") == "fp8"
        if seqs > (270 if fp8 else 135):
            return Trial(cfg, float("-inf"), False, note="OOM")
        tps = 4000.0 * seqs / (seqs + 200.0)                   # saturating but still rising at 256
        kv = seqs / (270.0 if fp8 else 135.0)
        return Trial(cfg, tps, True, kv, 0.0)

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "256"              # climbed past the 128 wall...
    assert best.config["--kv-cache-dtype"] == "fp8"            # ...because fp8 was needed AND throughput rose
    assert all(int(t.config.get("--max-num-seqs", "0")) <= 270 for t in hist)  # never blind-probed past the wall


def test_auto_predict_verify_takes_knee_not_wall_and_skips_pointless_fp8():
    # throughput PLATEAUS at 128 though the KV wall is far (486) -> take the knee, do NOT push to wall / fp8.
    # (this is exactly the 9B lesson: 278+fp8 maxed KV for ~0 gain and must be rejected.)
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        tps = 3000.0 if seqs >= 128 else 3000.0 * seqs / 128.0   # hard plateau at 128
        return Trial(cfg, tps, True, seqs / 512.0, 0.0)          # KV never binds in range

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "128"             # the throughput knee
    assert "--kv-cache-dtype" not in best.config             # no pointless KV-maxing fp8


def test_auto_returns_none_when_everything_thrashes():
    # H1: a feasible-but-thrashing config (score -inf) must NEVER be returned as best
    def fake(cfg):
        return Trial(cfg, float("-inf"), True, 0.99, 0.5, "thrash")   # feasible but preempting
    best, hist = auto_tune(fake, seed_seqs=32, log=_quiet)
    assert best is None


def test_auto_rejects_kv_maxing_fp8_with_no_throughput_gain():
    # H4/H5: throughput plateaus at 128; fp8 lets KV go higher but tps is FLAT -> fp8 must be rejected
    # (this is the exact 9B 278+fp8 trap).
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        fp8 = cfg.get("--kv-cache-dtype") == "fp8"
        if seqs > (400 if fp8 else 140):
            return Trial(cfg, float("-inf"), False, note="OOM")
        tps = 3000.0 if seqs >= 128 else 3000.0 * seqs / 128.0        # hard plateau at 128
        return Trial(cfg, tps, True, seqs / (400.0 if fp8 else 140.0), 0.0)

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "128"
    assert "--kv-cache-dtype" not in best.config                     # fp8 evaluated but rejected (no gain)


def test_auto_does_not_reprobe_a_measured_oom_auto_config():
    # H3: when s*2 OOMs at probe, the auto climb must not re-probe that same (auto) config
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        cap = 80 if cfg.get("--kv-cache-dtype") == "fp8" else 40
        if seqs > cap:
            return Trial(cfg, float("-inf"), False, note="OOM")
        return Trial(cfg, seqs * 10.0, True, seqs / 100.0, 0.0)

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    auto64 = [t for t in hist if t.config.get("--max-num-seqs") == "64" and "--kv-cache-dtype" not in t.config]
    assert len(auto64) == 1                                          # probed once, never re-probed


def test_auto_patience_passes_a_noisy_flat_rung():
    # M9: one flat intermediate rung must not end the ascent when a higher rung is much better
    tbl = {32: 800, 64: 1500, 128: 2000, 256: 2010, 512: 3000}      # 256 ~flat vs 128, 512 jumps
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        return Trial(cfg, float(tbl.get(seqs, 3000)), True, seqs / 2000.0, 0.0)   # KV tiny -> wall huge

    best, hist = auto_tune(fake, seed_seqs=32, seqs_ceiling=512, log=_quiet)
    assert best.config["--max-num-seqs"] == "512"


def test_auto_probe_halves_down_when_seed_ooms():
    def fake(cfg):
        seqs = int(cfg.get("--max-num-seqs", "32"))
        if seqs > 40:
            return Trial(cfg, float("-inf"), False, note="OOM")
        return Trial(cfg, seqs * 10.0, True, seqs / 100.0, 0.0)

    best, hist = auto_tune(fake, seed_seqs=128, seqs_ceiling=512, log=_quiet)   # seed OOMs -> halve to 32
    assert best is not None and int(best.config["--max-num-seqs"]) <= 40


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
