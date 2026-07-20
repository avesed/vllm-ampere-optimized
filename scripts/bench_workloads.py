#!/usr/bin/env python3
"""Multi-workload decode/accept bench against ONE running `vllm serve`.
Prompts differ but the graph/K are fixed, so no recompile — one server serves
all workloads; accept_len is isolated per-workload via /metrics counter deltas
(num_accepted/num_drafts), decode tok/s from client first/last-token timing.
Spans the entropy range: coding (templated/low-H) -> summary -> math (reasoning)
-> writing (open-ended/high-H), the regime where greedy draft would lose the
most accept-len and R04 (probabilistic) could finally bite.
Env: BASE, LABEL (server tag e.g. K2-greedy), MAX_TOKENS."""
import json
import os
import time

import requests

BASE = os.environ.get("BASE", "http://localhost:8000")
LABEL = os.environ.get("LABEL", "run")
MAXTOK = int(os.environ.get("MAX_TOKENS", "512"))

PARA = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
        "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")

WORKLOADS = {
    "coding": ("Write a complete, production-quality Python implementation of a "
               "thread-safe LRU cache: a class `LRUCache` with O(1) get/put, "
               "capacity-based eviction, type hints, docstrings, and pytest unit "
               "tests covering eviction order and concurrency."),
    "summary": ("请仔细阅读以下材料并总结：\n" + PARA * 90 +
                "\n请用中文系统总结上述材料。"),
    "math": ("一个袋子里有5个红球、3个蓝球、2个绿球。不放回地连续取3个球，"
             "求恰好取到2个红球的概率。请一步步详细推理，并给出最简分数。"),
    "writing": ("写一篇约800字的科幻短篇小说，主题是一个能听见植物思想的人。"
                "要求有完整的起承转合、具体的场景细节和情感转折，文笔细腻。"),
}


def spec_metrics():
    out = {}
    try:
        for ln in requests.get(f"{BASE}/metrics", timeout=10).text.splitlines():
            for k in ("vllm:spec_decode_num_drafts_total",
                      "vllm:spec_decode_num_accepted_tokens_total"):
                if ln.startswith(k + " ") or ln.startswith(k + "{"):
                    out[k] = out.get(k, 0.0) + float(ln.split()[-1])
    except Exception:
        pass
    return out


def model_id():
    return requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]


def one(model, wl, prompt, warmup=False):
    mt = 16 if warmup else MAXTOK
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0.6, "top_p": 0.95,
            "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    m0 = spec_metrics()
    tfirst = tlast = None
    ctoks = None
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=1200)
    for raw in r.iter_lines():
        if not raw:
            continue
        s = raw.decode()
        if not s.startswith("data: "):
            continue
        s = s[6:]
        if s.strip() == "[DONE]":
            break
        ev = json.loads(s)
        now = time.perf_counter()
        ch = ev.get("choices") or []
        d = (ch[0].get("delta") or {}) if ch else {}
        if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
            if tfirst is None:
                tfirst = now
            tlast = now
        if ev.get("usage"):
            ctoks = ev["usage"].get("completion_tokens")
    if warmup:
        return
    m1 = spec_metrics()
    dec = (ctoks - 1) / (tlast - tfirst) if (ctoks and ctoks > 1 and tlast and tfirst) else 0
    nd = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    na = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    acc = (1 + na / nd) if nd else 1.0
    print(f"RESULT {LABEL}/{wl} ctoks={ctoks}: decode={dec:.1f} tok/s | accept_len={acc:.2f}", flush=True)


def main():
    model = model_id()
    one(model, "_warm", WORKLOADS["summary"], warmup=True)
    for wl, prompt in WORKLOADS.items():
        one(model, wl, prompt)


if __name__ == "__main__":
    main()
