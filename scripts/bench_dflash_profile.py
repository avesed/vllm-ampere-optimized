#!/usr/bin/env python3
"""DFlash accept-PROFILE bench: per-position acceptance curve (the GAP-2 diagnostic)
+ aggregate accept_len + decode tok/s, against ONE running `vllm serve`.

The shape over draft positions 0..N-1 is the whole Step-2 signal:
  - GENTLE monotone decay  -> SWA healthy (GAP-1 enough)
  - SHARP CLIFF after pos~4 -> GAP-2 window-semantics bug (non-causal block needs
    a symmetric window; FA2 emits left-only (W-1,0) under attn_type=DECODER)

per-position rate[i] = Δ num_accepted_tokens_per_pos{position=i} / Δ num_drafts.
accept_len = 1 + Δ num_accepted_tokens_total / Δ num_drafts_total.
decode tok/s = client first/last-token timing (burst-safe for block draft).
Env: BASE, LABEL, MAX_TOKENS (default 512), WL (coding|summary|math|writing, default math)."""
import json
import os
import time

import requests

BASE = os.environ.get("BASE", "http://localhost:8000")
LABEL = os.environ.get("LABEL", "dflash")
MAXTOK = int(os.environ.get("MAX_TOKENS", "512"))
WL = os.environ.get("WL", "math")

PARA = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
        "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
WORKLOADS = {
    "coding": ("Write a complete, production-quality Python implementation of a "
               "thread-safe LRU cache: class `LRUCache` with O(1) get/put, eviction, "
               "type hints, docstrings, and pytest unit tests."),
    "summary": "请仔细阅读以下材料并总结：\n" + PARA * 90 + "\n请用中文系统总结上述材料。",
    "math": ("一个袋子里有5个红球、3个蓝球、2个绿球。不放回地连续取3个球，"
             "求恰好取到2个红球的概率。请一步步详细推理，并给出最简分数。"),
    "writing": ("写一篇约800字的科幻短篇小说，主题是一个能听见植物思想的人。"
                "要求完整起承转合、场景细节和情感转折。"),
}


def metrics():
    """Return (drafts, accepted_total, {pos: accepted_at_pos})."""
    drafts = acc_total = 0.0
    per_pos = {}
    try:
        for ln in requests.get(f"{BASE}/metrics", timeout=10).text.splitlines():
            if ln.startswith("#"):
                continue
            if ln.startswith("vllm:spec_decode_num_drafts_total") or (
                    ln.startswith("vllm:spec_decode_num_drafts") and "_tokens" not in ln
                    and "per_pos" not in ln):
                drafts += float(ln.split()[-1])
            elif ln.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
                pos = None
                if 'position="' in ln:
                    pos = int(ln.split('position="')[1].split('"')[0])
                if pos is not None:
                    per_pos[pos] = per_pos.get(pos, 0.0) + float(ln.split()[-1])
            elif ln.startswith("vllm:spec_decode_num_accepted_tokens_total") or (
                    ln.startswith("vllm:spec_decode_num_accepted_tokens")
                    and "per_pos" not in ln):
                acc_total += float(ln.split()[-1])
    except Exception:
        pass
    return drafts, acc_total, per_pos


def model_id():
    return requests.get(f"{BASE}/v1/models", timeout=10).json()["data"][0]["id"]


def run(model, prompt, warmup=False):
    mt = 16 if warmup else MAXTOK
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0.6, "top_p": 0.95,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    d0, a0, p0 = metrics()
    tfirst = tlast = None
    ctoks = None
    txt = ""
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
        dd = (ch[0].get("delta") or {}) if ch else {}
        piece = dd.get("content") or dd.get("reasoning") or dd.get("reasoning_content")
        if piece:
            if tfirst is None:
                tfirst = now
            tlast = now
            txt += piece
        if ev.get("usage"):
            ctoks = ev["usage"].get("completion_tokens")
    if warmup:
        return
    d1, a1, p1 = metrics()
    dec = (ctoks - 1) / (tlast - tfirst) if (ctoks and ctoks > 1 and tlast and tfirst) else 0
    nd = d1 - d0
    acc_len = (1 + (a1 - a0) / nd) if nd else 1.0
    # per-position accept rate (fraction of draft steps that accepted position i)
    prof = []
    for i in sorted(set(p0) | set(p1)):
        rate = ((p1.get(i, 0) - p0.get(i, 0)) / nd) if nd else 0.0
        prof.append(rate)
    prof_s = " ".join(f"{r:.2f}" for r in prof)
    print(f"RESULT {LABEL}/{WL} ctoks={ctoks}: decode={dec:.1f} tok/s | "
          f"accept_len={acc_len:.2f} | drafts={int(nd)}", flush=True)
    print(f"PROFILE {LABEL}/{WL} pos0..N accept-rate: {prof_s}", flush=True)
    snip = txt.replace("\n", " ")[:200]
    print(f"COHERENCE {LABEL}/{WL} head200: {snip}", flush=True)


def main():
    model = model_id()
    prompt = WORKLOADS[WL]
    run(model, prompt, warmup=True)
    run(model, prompt)


if __name__ == "__main__":
    main()
