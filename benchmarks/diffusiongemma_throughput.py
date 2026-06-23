#!/usr/bin/env python3
"""Throughput benchmark for DiffusionGemma over the chat endpoint (run in-container via docker exec).
Fires CONC concurrent chat requests (threads), each generating MAXTOK tokens, measures aggregate
output tok/s. Env: CONC(16), MAXTOK(128), URL.
"""
import os, json, time, threading
import urllib.request

CONC = int(os.environ.get("CONC", "16"))
MAXTOK = int(os.environ.get("MAXTOK", "128"))
URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
PROMPTS = [
    "Explain tensor parallelism in three sentences.",
    "Write a Python function to compute fibonacci numbers.",
    "What are the main causes of the French Revolution?",
    "Describe how a transformer attention layer works.",
    "用三句话解释什么是扩散语言模型。",
    "Give me a recipe for a simple tomato pasta.",
]
results = [None] * CONC


def worker(i):
    p = PROMPTS[i % len(PROMPTS)]
    body = json.dumps({"model": "dg", "messages": [{"role": "user", "content": p}],
                       "max_tokens": MAXTOK, "temperature": 0.6}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        ct = d.get("usage", {}).get("completion_tokens", 0)
        results[i] = (ct, time.time() - t0)
    except Exception as e:
        results[i] = (0, time.time() - t0)
        print(f"  req{i} FAIL {type(e).__name__} {str(e)[:60]}", flush=True)


def main():
    print(f"[tput] CONC={CONC} MAXTOK={MAXTOK}", flush=True)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONC)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = sum(r[0] for r in results if r)
    ok = sum(1 for r in results if r and r[0] > 0)
    print(f"[tput] requests={ok}/{CONC} total_completion_tokens={toks} wall={wall:.1f}s", flush=True)
    print(f"[tput] AGGREGATE THROUGHPUT = {toks/wall:.1f} tok/s  (per-req avg {toks/max(ok,1)/wall*ok:.1f})", flush=True)
    print("TPUT_DONE", flush=True)


if __name__ == "__main__":
    main()
