"""HALF-A classifier — pure R1-R5, no server/GPU."""
from ampere_autotune.half_a.classify import ServerState, HwSpec, classify, objective_plans

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


def test_r2_confidence_low_when_pressure_is_transient():
    # KV pressured but only in 40% of load windows -> low confidence + the "seen in X%" note
    s = _base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5, kv_window_frac=0.4)
    r = next(x for x in classify(s, HW) if x.rule == "R2-kv-pressure")
    assert r.confidence == "low" and "40%" in r.finding


def test_r2_confidence_high_and_no_note_by_default():
    r = next(x for x in classify(_base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5), HW)
             if x.rule == "R2-kv-pressure")           # kv_window_frac defaults to 1.0
    assert r.confidence == "high" and "windows" not in r.finding


def test_r5_confidence_from_sat_windows():
    s = _base(num_running=64, num_waiting=30, kv_cache_usage=0.50, sat_window_frac=0.6)
    r = next(x for x in classify(s, HW) if x.rule == "R5-saturation")
    assert r.confidence == "med" and "60%" in r.finding


def test_r6_spec_decode_pointer_below_ceiling():
    recs = classify(_base(decode_tps_single=85.0), HW)   # eff 85/208 = 0.41 < 0.6
    r = next(x for x in recs if x.rule == "R6-spec-decode-pointer")
    assert "speculative" in r.flags["--speculative-config"].lower() or "mtp" in r.flags["--speculative-config"].lower()


def test_r6_absent_near_ceiling():
    recs = classify(_base(decode_tps_single=200.0), HW)   # eff 0.96
    assert "R6-spec-decode-pointer" not in _rules(recs)


def test_r7_token_budget_limited_not_concurrency():
    # queue while running is BELOW the cap + KV fine -> token-budget limited (R7), not R5
    recs = classify(_base(num_waiting=5, num_running=10, kv_cache_usage=0.40), HW)
    assert _rules(recs).issuperset({"R7-batched-token-budget"})
    assert "R5-saturation" not in _rules(recs)
    r = next(x for x in recs if x.rule == "R7-batched-token-budget")
    assert r.flags["--max-num-batched-tokens"] == 8192


def test_r10_max_model_len_trim_under_kv_pressure():
    recs = classify(_base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5, max_model_len=32768), HW)
    assert "R10-max-model-len-trim" in _rules(recs)


def test_high_concurrency_plan_is_multivariable():
    # the user's case: saturated @ cap with KV headroom -> a COORDINATED plan, not a lone max-num-seqs
    s = _base(max_num_seqs=32, num_running=32, num_waiting=16, kv_cache_usage=0.25, decode_tps_single=84.0)
    p = next(x for x in objective_plans(s, HW) if "high concurrency" in x.objective)
    assert p.primary["--max-num-seqs"] == 48
    txt = " ".join(p.couple).lower()
    assert "kv" in txt and "cudagraph" in txt and "max-num-batched-tokens" in txt   # 3+ coupled knobs
    assert "aggregate" in p.ceiling.lower() and "per-stream" in p.ceiling.lower()   # honest ceiling


def test_kv_pressure_plan_is_a_decision_not_one_flag():
    s = _base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5)
    p = next(x for x in objective_plans(s, HW) if "capacity" in x.objective.lower())
    blob = (str(p.primary) + " ".join(p.couple)).lower()
    assert "fp8" in blob and "max-model-len" in blob and "gpu-memory-utilization" in blob


def test_per_stream_decode_plan_points_to_spec_not_flags():
    p = next(x for x in objective_plans(_base(decode_tps_single=84.0), HW) if "per-stream" in x.objective)
    assert "spec" in (str(p.primary) + " ".join(p.couple)).lower()
    assert "bandwidth" in p.ceiling.lower()


def test_no_removed_or_renamed_flags_ever_emitted():
    # guardrail: a broken restart command (nonexistent/inert v0.23 flags) must never be produced
    bad = {"cuda_graph_sizes", "--cuda-graph-sizes", "max_seq_len_to_capture", "--max-seq-len-to-capture",
           "swap_space", "--swap-space", "num_scheduler_steps", "--num-scheduler-steps"}
    states = [_base(),
              _base(num_waiting=5, num_running=10),
              _base(num_running=64, num_waiting=30, kv_cache_usage=0.50),
              _base(kv_cache_usage=0.92, preempt_per_s=0.1, num_waiting=5, max_model_len=32768),
              _base(decode_tps_single=200.0)]
    for st in states:
        for r in classify(st, HW):
            assert not (set(r.flags) & bad), (r.rule, r.flags)
