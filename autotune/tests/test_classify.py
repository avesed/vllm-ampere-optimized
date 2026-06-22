"""HALF-A classifier — pure R1-R5, no server/GPU."""
from ampere_autotune.half_a.classify import ServerState, HwSpec, classify

HW = HwSpec()  # 3090 / 9B / W4A8


def _rules(recs):
    return {r.rule for r in recs}


def _base(**kw):
    d = dict(max_num_seqs=64, kv_cache_usage=0.40, num_running=10, num_waiting=0,
             preempt_per_s=0.0, decode_tps_single=85.0, decode_tps_max_c=700.0,
             throughput_still_rising=False, prefix_hit_rate=None, mean_prompt_toks=0.0, qps=0.0)
    d.update(kw)
    return ServerState(**d)


def test_roofline_always_present():
    recs = classify(_base(), HW)
    assert "R0-roofline" in _rules(recs)
    r0 = recs[0]
    assert "ceiling" in r0.finding and "tok/s" in r0.finding


def test_decode_ceiling_math():
    # 936*8/(9*4) = 208 tok/s spec ceiling; with achieved 764 -> ~170
    assert round(HW.decode_ceiling_tps()) == 208
    assert round(HW.decode_ceiling_tps(764.0)) == 170


def test_r2_kv_pressure():
    recs = classify(_base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5), HW)
    r = next(x for x in recs if x.rule == "R2-kv-pressure")
    assert r.flags["--kv-cache-dtype"] == "fp8"
    assert r.flags["--max-num-seqs"] == 48     # 64*0.75


def test_r5_saturation_raises_when_kv_has_headroom():
    recs = classify(_base(num_running=64, num_waiting=30, kv_cache_usage=0.50), HW)
    r = next(x for x in recs if x.rule == "R5-saturation")
    assert r.flags["--max-num-seqs"] == 96     # 64*1.5


def test_r5_kv_bound_scales_out():
    recs = classify(_base(num_running=64, num_waiting=30, kv_cache_usage=0.85), HW)
    # KV>=0.80 but <0.88 (no R2) and saturated -> scale-out, not a flag
    assert "R5-saturation-kv-bound" in _rules(recs)
    assert "R5-saturation" not in _rules(recs)


def test_r2_takes_precedence_over_r5_raise():
    # saturated + queue + KV>=0.88+preempt -> R2 (don't raise concurrency into OOM)
    recs = classify(_base(num_running=64, num_waiting=30, kv_cache_usage=0.92, preempt_per_s=0.1), HW)
    assert "R2-kv-pressure" in _rules(recs)
    assert "R5-saturation" not in _rules(recs)


def test_r1_under_batched():
    recs = classify(_base(throughput_still_rising=True, num_waiting=0, num_running=20), HW)
    assert "R1-under-batched" in _rules(recs)


def test_r3_prefix_cache():
    recs = classify(_base(prefix_hit_rate=0.10, qps=20.0, mean_prompt_toks=300.0), HW)
    r = next(x for x in recs if x.rule == "R3-prefix-cache")
    assert r.flags.get("--enable-prefix-caching") is True


def test_r3_skipped_without_volume():
    recs = classify(_base(prefix_hit_rate=0.10, qps=0.1, mean_prompt_toks=10.0), HW)
    assert "R3-prefix-cache" not in _rules(recs)
