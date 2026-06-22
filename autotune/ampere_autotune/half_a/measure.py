"""HALF-A measurement — drive a running vLLM and read its state. Endpoint-bound (no GPU compute
of our own); the pure classify() consumes what this returns. NO privilege.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .classify import ServerState

_LINE = re.compile(r"^(vllm:[a-zA-Z0-9_]+)\{?[^}]*\}?\s+([0-9.eE+-]+)\s*$")


def scrape_metrics(endpoint: str) -> Dict[str, List[float]]:  # pragma: no cover - needs a server
    import requests
    out: Dict[str, List[float]] = {}
    try:
        txt = requests.get(endpoint.rstrip("/") + "/metrics", timeout=10).text
    except Exception:
        return out
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = _LINE.match(line.strip())
        if m:
            out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out


def _sum(d: Dict[str, List[float]], name: str) -> float:
    return sum(d.get(name, []))


def _max(d: Dict[str, List[float]], name: str) -> float:
    vals = d.get(name, [])
    return max(vals) if vals else 0.0


def model_id(endpoint: str) -> str:  # pragma: no cover - needs a server
    import requests
    return requests.get(endpoint.rstrip("/") + "/v1/models", timeout=10).json()["data"][0]["id"]


def _one_completion(endpoint: str, mid: str, prompt: str, max_tokens: int) -> int:  # pragma: no cover
    import requests
    r = requests.post(endpoint.rstrip("/") + "/v1/completions", timeout=300,
                      json={"model": mid, "prompt": prompt, "max_tokens": max_tokens,
                            "temperature": 0.0, "ignore_eos": True})
    r.raise_for_status()
    return int(r.json().get("usage", {}).get("completion_tokens", max_tokens))


def concurrency_sweep(endpoint: str, mid: str, levels=(1, 8, 32, 64),
                      max_tokens: int = 128) -> List[Tuple[int, float]]:  # pragma: no cover - needs a server
    """For each concurrency C, fire C identical completions and return (C, aggregate_decode_tok/s)."""
    prompt = "Write a detailed paragraph about the history of computing, then continue at length."
    results: List[Tuple[int, float]] = []
    for c in levels:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            toks = list(ex.map(lambda _: _one_completion(endpoint, mid, prompt, max_tokens), range(c)))
        dt = time.time() - t0
        results.append((c, (sum(toks) / dt) if dt > 0 else 0.0))
    return results


def build_state(endpoint: str, levels=(1, 8, 32, 64)) -> Optional[ServerState]:  # pragma: no cover - needs a server
    """Drive the sweep + scrape metrics -> ServerState for classify(). None if unreachable."""
    try:
        mid = model_id(endpoint)
    except Exception:
        return None
    pre = scrape_metrics(endpoint)
    pre_preempt = _sum(pre, "vllm:num_preemptions_total")
    t0 = time.time()
    sweep = concurrency_sweep(endpoint, mid, levels)
    dt = max(1e-3, time.time() - t0)
    post = scrape_metrics(endpoint)

    tps = [s for _, s in sweep]
    single = next((s for c, s in sweep if c == 1), tps[0] if tps else 0.0)
    peak = max(tps) if tps else 0.0
    rising = len(tps) >= 2 and tps[-1] > tps[-2] * 1.05
    # max_num_seqs inferred as the running plateau when pushed past the cap
    running_peak = _max(post, "vllm:num_requests_running") or float(max(levels))
    return ServerState(
        max_num_seqs=int(round(running_peak)) or max(levels),
        kv_cache_usage=_max(post, "vllm:gpu_cache_usage_perc") or _max(post, "vllm:kv_cache_usage_perc"),
        num_running=_max(post, "vllm:num_requests_running"),
        num_waiting=_max(post, "vllm:num_requests_waiting"),
        preempt_per_s=max(0.0, (_sum(post, "vllm:num_preemptions_total") - pre_preempt) / dt),
        decode_tps_single=single,
        decode_tps_max_c=peak,
        throughput_still_rising=rising,
        prefix_hit_rate=_prefix_hit(post),
        mean_prompt_toks=0.0,
        qps=0.0,
    )


def _prefix_hit(d: Dict[str, List[float]]) -> Optional[float]:
    q = _sum(d, "vllm:prefix_cache_queries_total") or _sum(d, "vllm:gpu_prefix_cache_queries_total")
    h = _sum(d, "vllm:prefix_cache_hits_total") or _sum(d, "vllm:gpu_prefix_cache_hits_total")
    return (h / q) if q > 0 else None
