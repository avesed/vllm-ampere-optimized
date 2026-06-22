#!/usr/bin/env python3
"""GSM8K accuracy eval for DiffusionGemma served over the OpenAI chat endpoint (localhost:8000).
Quality gate for the cyankiwi W4A16 ckpt (the 704-dim 4-bit concern). Reads gsm8k.jsonl
({question, answer}; gold = text after '####'), sends each via /v1/chat/completions, extracts the
model's final number, reports accuracy. Run inside the serving container (docker exec) so localhost
reaches the engine. Env: N(50), MAXTOK(512), TEMP(0.2), GSM8K(/gsm8k.jsonl), URL.
"""
import os, re, json, urllib.request

N = int(os.environ.get("N", "50"))
MAXTOK = int(os.environ.get("MAXTOK", "512"))
TEMP = float(os.environ.get("TEMP", "0.2"))
PATH = os.environ.get("GSM8K", "/gsm8k.jsonl")
URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
SYS = "Solve the math problem step by step. On the last line write exactly: The answer is <number>"


def gold_of(ans):
    return ans.split("####")[-1].strip().replace(",", "").replace("$", "")


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    return m[-1] if m else None


def pred_of(text):
    # prefer the number after "answer is", else last number in the reply
    m = re.search(r"answer is\s*\**\s*(-?\$?\d[\d,]*\.?\d*)", text, re.I)
    if m:
        return num(m.group(1))
    return num(text)


def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return False


rows = [json.loads(l) for l in open(PATH)][:N]
ok = 0
for i, r in enumerate(rows):
    body = json.dumps({"model": "dg", "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": r["question"]}],
        "max_tokens": MAXTOK, "temperature": TEMP}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[{i}] REQ FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        continue
    g, p = gold_of(r["answer"]), pred_of(out)
    hit = eq(p, g)
    ok += hit
    print(f"[{i+1}/{len(rows)}] gold={g} pred={p} {'OK' if hit else 'X'} (run {ok}/{i+1}={ok/(i+1):.1%})", flush=True)

print(f"GSM8K_RESULT acc={ok}/{len(rows)}={ok/len(rows):.1%} (N={len(rows)}, maxtok={MAXTOK}, temp={TEMP})", flush=True)
