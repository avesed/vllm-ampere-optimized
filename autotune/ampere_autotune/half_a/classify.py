"""HALF-A bottleneck classifier + vLLM-flag prescription — PURE (no HTTP/NVML), so it is
unit-testable with no GPU/server. Mirrors jungledesh/profile's measure->classify->prescribe with
the R1-R5 rule set; recommend-only (emits flags + reasoning + restart, never applies).

decode is weight-bandwidth-bound on Ampere -> the roofline below is the decode ceiling; the
R-rules act on OBSERVED load state (KV pressure, queue, prefix reuse, throughput curve).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

INFO = "INFO"
WARN = "WARN"


@dataclass
class HwSpec:
    peak_bw_gbs: float = 936.0       # GDDR6X 3090 spec
    params_b: float = 9.0            # billions
    weight_bits: float = 4.0         # W4A8 -> int4 weights drive the decode read (approx; GDN bf16 ignored)
    vram_gb: float = 24.0

    def decode_ceiling_tps(self, achieved_bw_gbs: Optional[float] = None) -> float:
        bw = achieved_bw_gbs if achieved_bw_gbs else self.peak_bw_gbs
        return bw * 8.0 / (self.params_b * self.weight_bits)   # tok/s ceiling (1 token = read all weights)


@dataclass
class ServerState:
    max_num_seqs: int
    kv_cache_usage: float            # 0..1 (peak observed)
    num_running: float               # at the highest concurrency probed
    num_waiting: float
    preempt_per_s: float
    decode_tps_single: float         # measured single-stream output tok/s
    decode_tps_max_c: float          # measured aggregate output tok/s at the highest concurrency
    throughput_still_rising: bool    # tok/s still climbing at the top of the concurrency sweep
    prefix_hit_rate: Optional[float] = None   # None = prefix caching off / unknown
    mean_prompt_toks: float = 0.0
    qps: float = 0.0
    max_model_len: Optional[int] = None       # from /v1/models (for the R10 capacity trim)
    has_mtp_head: Optional[bool] = None        # None = unknown (can't tell from the endpoint)
    # multi-window confidence: fraction of under-load samples where the condition held (1.0 = single
    # window / no multi-window data). Kills false positives from a transient one-scrape spike.
    kv_window_frac: float = 1.0               # frac of windows with KV >= 0.88
    sat_window_frac: float = 1.0              # frac of windows with running >= 0.95*max_num_seqs


# multi-window confidence thresholds (borrowed from jungledesh/profile's "seen in X% of windows")
def _conf(frac: float) -> str:
    return "high" if frac >= 0.75 else ("med" if frac >= 0.5 else "low")


@dataclass
class FlagRec:
    rule: str
    severity: str
    finding: str
    flags: Dict[str, object] = field(default_factory=dict)   # suggested launch flags (empty = client-side / no flag)
    reason: str = ""
    confidence: str = "high"                                 # high/med/low from multi-window evidence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    """An OBJECTIVE-oriented coordinated bundle (vs an isolated single-knob rule). Tuning is
    multi-variable: one goal couples several knobs that gate/trade off against each other."""
    objective: str
    primary: Dict[str, object] = field(default_factory=dict)   # the lead knob(s)
    couple: List[str] = field(default_factory=list)            # coupled co-adjustments, each with its own gate
    tradeoffs: List[str] = field(default_factory=list)
    ceiling: str = ""                                          # honest note: what this can / cannot move

    def to_dict(self) -> dict:
        return asdict(self)


def classify(s: ServerState, hw: HwSpec, achieved_bw_gbs: Optional[float] = None) -> List[FlagRec]:
    """Apply R1-R5. Returns ordered recommendations (recommend-only; flags are SUGGESTIONS)."""
    recs: List[FlagRec] = []
    # SINGLE-STREAM decode ceiling = bandwidth / weight-bytes (1 token reads all weights once).
    # Batched aggregate tok/s legitimately EXCEEDS this (the weight read is amortized across the
    # batch), so efficiency is single-stream/ceiling — NOT batch-peak/ceiling.
    ceiling = hw.decode_ceiling_tps(achieved_bw_gbs)
    eff = (s.decode_tps_single / ceiling) if ceiling > 0 else 0.0
    saturated = (s.num_running >= 0.95 * s.max_num_seqs)
    queue_ratio = (s.num_waiting / s.max_num_seqs) if s.max_num_seqs else 0.0

    # context: roofline placement (not a rule; informs the others)
    recs.append(FlagRec(
        "R0-roofline", INFO,
        f"single-stream decode {s.decode_tps_single:.0f} tok/s = {eff:.0%} of the ~{ceiling:.0f} tok/s "
        f"bandwidth ceiling; batched aggregate peaks at {s.decode_tps_max_c:.0f} tok/s (weight read "
        f"amortized over the batch). Under load: KV {s.kv_cache_usage:.0%}, running {s.num_running:.0f}/"
        f"{s.max_num_seqs}, waiting {s.num_waiting:.0f}, preempt {s.preempt_per_s:.2f}/s.",
        reason="roofline + observed load"))

    # R2 — KV pressure (check before R5 raise, so we don't recommend raising into OOM)
    kv_pressure = s.kv_cache_usage >= 0.88 and (s.preempt_per_s > 0.02 or s.num_waiting > 2)
    if kv_pressure:
        seen = f" (KV-pressured in {s.kv_window_frac:.0%} of load windows)" if s.kv_window_frac < 1.0 else ""
        recs.append(FlagRec(
            "R2-kv-pressure", WARN,
            f"KV cache at {s.kv_cache_usage:.0%} with preemption/queueing -> thrashing.{seen}",
            {"--kv-cache-dtype": "fp8", "--max-num-seqs": max(1, int(s.max_num_seqs * 0.75))},
            reason="lower concurrency or halve KV bytes (fp8) to stop preemption; or raise gpu-mem-util if headroom",
            confidence=_conf(s.kv_window_frac)))

    # R5 — concurrency saturation (only raise if KV is NOT pressured)
    if saturated and queue_ratio >= 0.30 and not kv_pressure:
        seen = f" (saturated in {s.sat_window_frac:.0%} of load windows)" if s.sat_window_frac < 1.0 else ""
        if s.kv_cache_usage < 0.80:
            recs.append(FlagRec(
                "R5-saturation", WARN,
                f"running==max_num_seqs ({s.num_running:.0f}/{s.max_num_seqs}) with a queue and KV only "
                f"{s.kv_cache_usage:.0%} -> under-provisioned concurrency.{seen}",
                {"--max-num-seqs": int(s.max_num_seqs * 1.5)},
                reason="raise max-num-seqs; KV has headroom", confidence=_conf(s.sat_window_frac)))
        else:
            recs.append(FlagRec(
                "R5-saturation-kv-bound", WARN,
                f"saturated AND KV {s.kv_cache_usage:.0%} -> can't raise concurrency on this card.{seen}",
                {}, reason="add a replica / scale out (KV-bound, not a single-box flag)",
                confidence=_conf(s.sat_window_frac)))

    # R1 — under-batching (server starved): throughput still rising + low queue
    if s.throughput_still_rising and s.num_waiting < 2 and not saturated:
        recs.append(FlagRec(
            "R1-under-batched", INFO,
            "throughput was still rising at the top of the sweep with little queue -> the SERVER is "
            "under-fed, not mis-tuned.",
            {}, reason="send more concurrent CLIENT requests; raise max-num-seqs only if it then saturates"))

    # R3 — low prefix reuse (only worth it with reuse potential + enough prompt volume)
    if (s.prefix_hit_rate is not None and s.prefix_hit_rate < 0.35
            and s.qps * s.mean_prompt_toks >= 1000):
        recs.append(FlagRec(
            "R3-prefix-cache", INFO,
            f"prefix-cache hit rate {s.prefix_hit_rate:.0%} with high prompt-token volume.",
            {"--enable-prefix-caching": True},
            reason="shared prefixes (system prompts / few-shot) would be reused"))

    # R6 — the ONLY real DECODE-tok/s lever: spec-decode / MTP (amortizes the weight-byte read
    # across k accepted tokens). A pointer, not a flag HALF-A owns (needs an MTP head).
    if eff < 0.6:
        recs.append(FlagRec(
            "R6-spec-decode-pointer", INFO,
            f"single-stream decode is {eff:.0%} of the bandwidth ceiling -> the #1 way past the wall is "
            f"MTP/speculative decode (one forward -> k tokens). This is the only flag-level lever that moves "
            f"DECODE tok/s here; flag tuning won't (decode is bandwidth-bound).",
            {"--speculative-config": "(MTP, if the checkpoint ships an mtp head; Qwen3.5/3.6 do)"},
            reason="measured +25% (9B) / +52% (27B) at K=2-3; keep mtp.fc bf16 + in the quant ignore"))

    # R7 — TOKEN-BUDGET limited (NOT concurrency-limited): queue with running BELOW the seq cap ->
    # the per-step batched-token budget is the bottleneck, not max-num-seqs. Moves PREFILL/TTFT.
    if s.num_waiting > 0 and not saturated and not kv_pressure:
        recs.append(FlagRec(
            "R7-batched-token-budget", INFO,
            f"requests queue ({s.num_waiting:.0f} waiting) while running ({s.num_running:.0f}) is BELOW the "
            f"seq cap ({s.max_num_seqs}) -> limited by the per-step token budget, not concurrency.",
            {"--max-num-batched-tokens": 8192},
            reason="raise the prefill token budget (>=8192 for throughput); lower to 512-1024 if ITL spikes "
                   "when prefills land. PREFILL/TTFT lever, not steady decode."))

    # R10 — CAPACITY: configured context far exceeds what's used, while KV is pressured -> trim
    # max-model-len to reclaim per-seq KV reservation -> more concurrency. NOT a per-stream change.
    if kv_pressure and s.max_model_len and s.max_model_len > 8192:
        recs.append(FlagRec(
            "R10-max-model-len-trim", INFO,
            f"KV pressured ({s.kv_cache_usage:.0%}) and max-model-len is {s.max_model_len} -> if real prompts "
            f"are far shorter, trimming it reclaims per-seq KV reservation for more concurrency.",
            {"--max-model-len": "<your true p99 context>"},
            reason="CAPACITY lever (more concurrent seqs), not a per-stream decode gain"))

    # GUARDRAIL — never emit flags that don't exist / are inert in v0.23 (broken restart command):
    #   cuda_graph_sizes, max_seq_len_to_capture  -> renamed (cudagraph_capture_sizes / max_cudagraph_capture_size)
    #   swap_space, num_scheduler_steps           -> removed/inert in V1
    # cudagraph_capture_sizes / cudagraph_mode / enforce_eager are startup-mem/latency, NOT tok/s, and
    # need launch-arg/startup-log inspection (no /metrics signal) -> documented + deferred, not emitted here.
    return recs


def objective_plans(s: ServerState, hw: HwSpec, achieved_bw_gbs: Optional[float] = None) -> List[Plan]:
    """Coordinated, OBJECTIVE-oriented bundles — the multi-variable view a single rule can't give.
    Each plan names the lead knob AND the knobs it is coupled to (which gate it or trade off)."""
    plans: List[Plan] = []
    ceiling = hw.decode_ceiling_tps(achieved_bw_gbs)
    eff = (s.decode_tps_single / ceiling) if ceiling > 0 else 0.0
    saturated = s.num_running >= 0.95 * s.max_num_seqs
    kv_pressure = s.kv_cache_usage >= 0.88 and (s.preempt_per_s > 0.02 or s.num_waiting > 2)
    kv_room = max(0.0, 1.0 - s.kv_cache_usage)

    # OBJECTIVE: aggregate decode throughput under HIGH CONCURRENCY (the user's example) —
    # NOT one knob: concurrency cap is gated by KV budget, coupled to cudagraph coverage + ITL.
    if (saturated or s.num_waiting > 0) and not kv_pressure:
        plans.append(Plan(
            objective="aggregate decode throughput @ high concurrency",
            primary={"--max-num-seqs": int(s.max_num_seqs * 1.5)},
            couple=[
                f"KV budget GATES how far you can push it: KV is at {s.kv_cache_usage:.0%} ({kv_room:.0%} free). "
                f"Each added seq costs KV -> if it climbs toward ~85%, buy room with --kv-cache-dtype fp8 "
                f"(~2x seqs/byte), --gpu-memory-utilization up, or --max-model-len trim. Don't raise seqs into preemption.",
                f"cudagraph coverage moves WITH it: the default capture set is min(2*max_num_seqs, 512), so raising "
                f"seqs grows captures (more VRAM, competes with KV) AND the new top batch must be captured or it falls "
                f"to eager at high batch = a latency cliff. Watch capture-OOM / startup time.",
                "ITL is the other half of 'decode speed under load': more concurrent seqs + prefills interleaving "
                "raises per-token latency. Tune --max-num-batched-tokens (raise for prefill MFU, or lower to protect "
                "decode ITL) — measure TPOT under load, not just aggregate tok/s.",
            ],
            tradeoffs=[
                "VRAM is split between KV blocks and cudagraph captures — pushing seqs pulls on both.",
                "Aggregate tok/s up does NOT mean per-stream decode up; high batch can raise each stream's TPOT.",
            ],
            ceiling="Raises AGGREGATE throughput (more streams sharing each weight read). Per-stream decode rate is "
                    "bandwidth-bound and unchanged — only spec-decode/MTP moves that (and it helps LESS at high batch).",
        ))

    # OBJECTIVE: stop KV thrashing / win capacity — a 4-way decision, not one flag.
    if kv_pressure:
        plans.append(Plan(
            objective="stop KV thrashing (capacity)",
            primary={"--kv-cache-dtype": "fp8"},
            couple=[
                "Four coupled levers, pick by cause: --kv-cache-dtype fp8 (halve KV bytes, ~lossless <=16k), "
                "--max-model-len trim (if real ctx << configured), --gpu-memory-utilization up (if host RAM/VRAM headroom), "
                "or --max-num-seqs down (last resort — costs concurrency).",
                "After buying KV room, RE-CHECK the high-concurrency plan: more KV room may let you RAISE max-num-seqs.",
            ],
            tradeoffs=["fp8 KV at >32k ctx can cost a little accuracy; trimming max-model-len caps long prompts."],
            ceiling="Capacity/anti-preemption — restores throughput lost to thrashing; not a per-stream speedup.",
        ))

    # OBJECTIVE: per-stream decode speed (TPOT) — bandwidth-bound; flags barely move it.
    if eff < 0.6:
        plans.append(Plan(
            objective="per-stream decode speed (TPOT)",
            primary={"--speculative-config": "(MTP, if the checkpoint ships an mtp head)"},
            couple=[
                "Decode is bandwidth-bound: flag tuning does NOT raise per-stream tok/s. The lever is spec-decode/MTP "
                "(one forward -> k tokens). Keep cudagraph on (enforce_eager is the anti-pattern) so the draft+verify "
                "step is graphed; at long ctx, fp8-KV cuts the per-step KV read.",
            ],
            tradeoffs=["spec-decode helps single/low-concurrency most; its win shrinks as batch already amortizes weights."],
            ceiling=f"single-stream is {eff:.0%} of the ~{ceiling:.0f} tok/s bandwidth ceiling; only fewer bytes "
                    "(int4/fp8) or spec-decode move it, never scheduler flags.",
        ))

    return plans
