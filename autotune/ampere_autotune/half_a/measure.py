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


def model_info(endpoint: str) -> Tuple[str, Optional[int]]:  # pragma: no cover - needs a server
    import requests
    d = requests.get(endpoint.rstrip("/") + "/v1/models", timeout=10).json()["data"][0]
    return d["id"], d.get("max_model_len")


def model_id(endpoint: str) -> str:  # pragma: no cover - needs a server
    return model_info(endpoint)[0]


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
    try:                                  # warm-up: the first post-restart request is cold (caches),
        _one_completion(endpoint, mid, prompt, 8)   # which would tank the c=1 single-stream number
    except Exception:
        pass
    for c in levels:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            toks = list(ex.map(lambda _: _one_completion(endpoint, mid, prompt, max_tokens), range(c)))
        dt = time.time() - t0
        results.append((c, (sum(toks) / dt) if dt > 0 else 0.0))
    return results


def lowc_throughput(endpoint: str, c: int = 1, max_tokens: int = 128, reps: int = 2) -> Optional[float]:  # pragma: no cover - needs a server
    """Decode tok/s at a fixed LOW concurrency (c=1 single-session, or a few). Median of reps after
    a warm-up. This is the objective for single/few-session max throughput (per-stream regime)."""
    try:
        mid, _ = model_info(endpoint)
    except Exception:
        return None
    vals = [sw[0][1] for sw in (concurrency_sweep(endpoint, mid, (c,), max_tokens) for _ in range(reps)) if sw]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def _burst_and_scrape(endpoint: str, mid: str, concurrency: int, max_tokens: int = 256,
                      poll_s: float = 8.0) -> Dict[str, float]:  # pragma: no cover - needs a server
    """Hold `concurrency` long generations IN FLIGHT and scrape /metrics WHILE they run (not after),
    so running/KV/waiting are the real under-load peaks. Returns peak signals + preempt rate."""
    prompt = "Write an exhaustive essay on the history of computing; keep going in great detail."
    pre = scrape_metrics(endpoint)
    pre_preempt = _sum(pre, "vllm:num_preemptions_total")
    peak = {"running": 0.0, "waiting": 0.0, "kv": 0.0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one_completion, endpoint, mid, prompt, max_tokens) for _ in range(concurrency)]
        while time.time() - t0 < poll_s and any(not f.done() for f in futs):
            d = scrape_metrics(endpoint)
            peak["running"] = max(peak["running"], _max(d, "vllm:num_requests_running"))
            peak["waiting"] = max(peak["waiting"], _max(d, "vllm:num_requests_waiting"))
            peak["kv"] = max(peak["kv"],
                             _max(d, "vllm:gpu_cache_usage_perc") or _max(d, "vllm:kv_cache_usage_perc"))
            time.sleep(0.5)
        for f in futs:
            try:
                f.result(timeout=120)
            except Exception:
                pass
    dt = max(1e-3, time.time() - t0)
    post = scrape_metrics(endpoint)
    peak["preempt_per_s"] = max(0.0, (_sum(post, "vllm:num_preemptions_total") - pre_preempt) / dt)
    ph = _prefix_hit(post)
    peak["prefix_hit"] = ph if ph is not None else -1.0
    return peak


def build_state(endpoint: str, levels=(1, 8, 32), burst_c: int = 48) -> Optional[ServerState]:  # pragma: no cover - needs a server
    """Throughput from a concurrency sweep + load state from an under-load burst -> ServerState."""
    try:
        mid, max_len = model_info(endpoint)
    except Exception:
        return None
    sweep = concurrency_sweep(endpoint, mid, levels)
    tps = [s for _, s in sweep]
    single = next((s for c, s in sweep if c == 1), tps[0] if tps else 0.0)
    peak_tps = max(tps) if tps else 0.0
    rising = len(tps) >= 2 and tps[-1] > tps[-2] * 1.05
    load = _burst_and_scrape(endpoint, mid, concurrency=burst_c)
    running_peak = load["running"] or float(burst_c)
    hit = load.get("prefix_hit", -1.0)
    return ServerState(
        max_num_seqs=int(round(running_peak)),         # plateau under an over-subscribed burst ~= the cap
        kv_cache_usage=load["kv"],
        num_running=load["running"],
        num_waiting=load["waiting"],
        preempt_per_s=load["preempt_per_s"],
        decode_tps_single=single,
        decode_tps_max_c=peak_tps,
        throughput_still_rising=rising,
        prefix_hit_rate=(None if hit < 0 else hit),
        mean_prompt_toks=0.0,
        qps=0.0,
        max_model_len=max_len,
    )


def _prefix_hit(d: Dict[str, List[float]]) -> Optional[float]:
    q = _sum(d, "vllm:prefix_cache_queries_total") or _sum(d, "vllm:gpu_prefix_cache_queries_total")
    h = _sum(d, "vllm:prefix_cache_hits_total") or _sum(d, "vllm:gpu_prefix_cache_hits_total")
    return (h / q) if q > 0 else None
