#!/usr/bin/env python3
"""OpenAI-API decode/prefill benchmark client (the ONLY sanctioned test path — never offline LLM()).
Streams chat/completions against a running `vllm serve`; TTFT = client time-to-first-token,
decode tok/s = (completion_tokens-1)/(t_last-t_first) (burst-safe for MTP), prefill = prompt_tokens/
TTFT. Accept-len from /metrics delta. Env: BASE (http://localhost:8000), REP, K, LABEL."""
import json
import os
import time

import requests

BASE = os.environ.get("BASE", "http://localhost:8000")
PARA = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
        "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")


def model_id():
    return requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]


def spec_metrics():
    out = {}
    try:
        for ln in requests.get(f"{BASE}/metrics", timeout=10).text.splitlines():
            for k in ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens"):
                if ln.startswith(k + " ") or ln.startswith(k + "{"):
                    out[k] = out.get(k, 0.0) + float(ln.split()[-1])
    except Exception:
        pass
    return out


def one(model, prompt, k, label, warmup=False):
    mt = 16 if warmup else int(os.environ.get("MAX_TOKENS", "256"))
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0.6, "top_p": 0.95,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    m0 = spec_metrics()
    t0 = time.perf_counter()
    tfirst = tlast = None
    ptoks = ctoks = None
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
        if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):  # reasoning models stream delta.reasoning
            if tfirst is None:
                tfirst = now
            tlast = now
        if ev.get("usage"):
            ptoks = ev["usage"].get("prompt_tokens")
            ctoks = ev["usage"].get("completion_tokens")
    if warmup:
        return
    m1 = spec_metrics()
    ttft = (tfirst - t0) * 1000 if tfirst else 0
    dec = (ctoks - 1) / (tlast - tfirst) if (ctoks and ctoks > 1 and tlast and tfirst) else 0
    pref = ptoks / (ttft / 1000) if (ptoks and ttft) else 0
    nd = m1.get("vllm:spec_decode_num_drafts", 0) - m0.get("vllm:spec_decode_num_drafts", 0)
    na = m1.get("vllm:spec_decode_num_accepted_tokens", 0) - m0.get("vllm:spec_decode_num_accepted_tokens", 0)
    acc = (1 + na / nd) if (k > 0 and nd) else 1.0
    print(f"RESULT {label} plen={ptoks}: decode={dec:.1f} tok/s | prefill={pref:.0f} tok/s "
          f"(TTFT={ttft:.0f}ms) | accept_len={acc:.2f}", flush=True)


def main():
    rep = int(os.environ.get("REP", "90"))
    k = int(os.environ.get("K", "2"))
    label = os.environ.get("LABEL", "run")
    model = model_id()
    prompt = "请仔细阅读以下材料并总结：\n" + PARA * rep + "\n请用中文系统总结上述材料。"
    one(model, prompt, k, label, warmup=True)
    one(model, prompt, k, label)


if __name__ == "__main__":
    main()
