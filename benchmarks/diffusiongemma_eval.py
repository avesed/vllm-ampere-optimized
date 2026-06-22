#!/usr/bin/env python3
"""Concurrent GSM8K + MMLU-Pro eval over the chat endpoint (run in-container). Reports accuracy
AND aggregate throughput (the concurrent run doubles as a throughput test).
Env: MODE(gsm8k|mmlupro), N(500), CONC(8), MAXTOK(640), DATA(path), URL.
"""
import os, re, json, time, ast, threading, urllib.request
from collections import deque

MODE = os.environ.get("MODE", "gsm8k")
N = int(os.environ.get("N", "500"))
CONC = int(os.environ.get("CONC", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "640"))
URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
DATA = os.environ.get("DATA", "/gsm8k.jsonl" if MODE == "gsm8k" else "/mmlu_pro.jsonl")
LETTERS = "ABCDEFGHIJ"


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    return m[-1] if m else None


def build(r):
    if MODE == "gsm8k":
        sys = "Solve the math problem step by step. End with: The answer is <number>"
        return sys, r["question"], r["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
    opts = r["options"]
    if isinstance(opts, str):
        opts = ast.literal_eval(opts)
    body = r["question"] + "\n\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    sys = "Answer the multiple-choice question. Reason briefly, then end with: The answer is (X)"
    return sys, body, r["answer"].strip().upper()


def correct(pred_text, gold):
    if MODE == "gsm8k":
        m = re.search(r"answer is\s*\**\s*(-?\$?\d[\d,]*\.?\d*)", pred_text, re.I)
        p = num(m.group(1)) if m else num(pred_text)
        try:
            return p is not None and abs(float(p) - float(gold)) < 1e-3
        except ValueError:
            return False
    m = re.findall(r"answer is\s*\(?([A-J])\)?", pred_text, re.I) or re.findall(r"\(([A-J])\)", pred_text) or re.findall(r"\b([A-J])\b", pred_text)
    return bool(m) and m[-1].upper() == gold


rows = [json.loads(l) for l in open(DATA)][:N]
ok = [0]
toks = [0]
done = [0]
lock = threading.Lock()
work = deque(range(len(rows)))


def worker():
    while True:
        with lock:
            if not work:
                return
            i = work.popleft()
        sysmsg, q, gold = build(rows[i])
        b = json.dumps({"model": "dg", "messages": [{"role": "system", "content": sysmsg},
                        {"role": "user", "content": q}], "max_tokens": MAXTOK, "temperature": 0.2}).encode()
        req = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=400).read())
            txt = d["choices"][0]["message"]["content"]
            ct = d.get("usage", {}).get("completion_tokens", 0)
            hit = correct(txt, gold)
        except Exception:
            ct, hit = 0, False
        with lock:
            ok[0] += hit
            toks[0] += ct
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(rows)} acc={ok[0]/done[0]:.1%}", flush=True)


print(f"[eval] MODE={MODE} N={len(rows)} CONC={CONC} MAXTOK={MAXTOK}", flush=True)
t0 = time.time()
ths = [threading.Thread(target=worker) for _ in range(CONC)]
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.time() - t0
print(f"EVAL_RESULT MODE={MODE} acc={ok[0]}/{len(rows)}={ok[0]/len(rows):.1%} | "
      f"throughput={toks[0]/wall:.1f} tok/s (agg, CONC={CONC}) | {toks[0]} tok / {wall:.1f}s", flush=True)
print("EVAL_DONE", flush=True)
