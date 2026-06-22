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


@dataclass
class FlagRec:
    rule: str
    severity: str
    finding: str
    flags: Dict[str, object] = field(default_factory=dict)   # suggested launch flags (empty = client-side / no flag)
    reason: str = ""

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
        recs.append(FlagRec(
            "R2-kv-pressure", WARN,
            f"KV cache at {s.kv_cache_usage:.0%} with preemption/queueing -> thrashing.",
            {"--kv-cache-dtype": "fp8", "--max-num-seqs": max(1, int(s.max_num_seqs * 0.75))},
            reason="lower concurrency or halve KV bytes (fp8) to stop preemption; or raise gpu-mem-util if headroom"))

    # R5 — concurrency saturation (only raise if KV is NOT pressured)
    if saturated and queue_ratio >= 0.30 and not kv_pressure:
        if s.kv_cache_usage < 0.80:
            recs.append(FlagRec(
                "R5-saturation", WARN,
                f"running==max_num_seqs ({s.num_running:.0f}/{s.max_num_seqs}) with a queue and KV only "
                f"{s.kv_cache_usage:.0%} -> under-provisioned concurrency.",
                {"--max-num-seqs": int(s.max_num_seqs * 1.5)},
                reason="raise max-num-seqs; KV has headroom"))
        else:
            recs.append(FlagRec(
                "R5-saturation-kv-bound", WARN,
                f"saturated AND KV {s.kv_cache_usage:.0%} -> can't raise concurrency on this card.",
                {}, reason="add a replica / scale out (KV-bound, not a single-box flag)"))

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
