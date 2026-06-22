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
    ceiling = hw.decode_ceiling_tps(achieved_bw_gbs)
    eff = (s.decode_tps_max_c / ceiling) if ceiling > 0 else 0.0
    saturated = (s.num_running >= 0.95 * s.max_num_seqs)
    queue_ratio = (s.num_waiting / s.max_num_seqs) if s.max_num_seqs else 0.0

    # context: roofline placement (not a rule; informs the others)
    recs.append(FlagRec(
        "R0-roofline", INFO,
        f"decode ceiling ~{ceiling:.0f} tok/s (bandwidth-bound); single-stream {s.decode_tps_single:.0f}, "
        f"batch-peak {s.decode_tps_max_c:.0f} tok/s = {eff:.0%} of ceiling. KV peak {s.kv_cache_usage:.0%}, "
        f"running {s.num_running:.0f}/{s.max_num_seqs}, waiting {s.num_waiting:.0f}, preempt {s.preempt_per_s:.2f}/s.",
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

    return recs
