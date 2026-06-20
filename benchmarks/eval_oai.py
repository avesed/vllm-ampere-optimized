#!/usr/bin/env python3
"""Curl-style OpenAI eval client (runs on the dev box -> served :dev endpoint).
Tasks: gsm8k / mmlu_pro. Thinking-aware: the server's --reasoning-parser strips
<think>, so message.content is the post-think answer. Reads JSONL (no parquet dep)."""
import re, json, time, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def chat(base, model, prompt, max_tokens, timeout=1200):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    err = "?"
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            m = d["choices"][0]["message"]
            return (m.get("content") or "") or (m.get("reasoning_content") or "")
        except Exception as e:
            err = str(e); time.sleep(3)
    return "__ERR__:" + err

def load_jsonl(p, n, shuffle=False):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if shuffle:
        import random
        random.seed(1234)
        random.shuffle(rows)
    return rows[:n] if n else rows

def gsm8k_gold(r):
    m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
    return m.group(1).replace(",", "").strip() if m else None

def gsm8k_pred(t):
    ns = re.findall(r"-?\d+\.?\d*", t.replace(",", ""))
    return ns[-1].rstrip(".") if ns else None

def num_eq(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-4
    except Exception:
        return False

L = "ABCDEFGHIJ"
def mmlu_prompt(r):
    lines = "\n".join(f"{L[i]}. {x}" for i, x in enumerate(r["options"]))
    return (f"Question: {r['question']}\nOptions:\n{lines}\n\n"
            "Think step by step, then end with 'The answer is (X)' where X is the correct option letter.")

def mmlu_pred(t, no):
    ls = L[:no]
    for pat in [rf"answer is[:\s\*\(]*([{ls}])\b", rf"\\boxed\{{\s*([{ls}])",
                rf"answer[:\s\*\(]*([{ls}])\b", rf"\(([{ls}])\)"]:
        m = re.findall(pat, t, re.I)
        if m:
            return m[-1].upper()
    m = re.findall(rf"\b([{ls}])\b", t)
    return m[-1].upper() if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "mmlu_pro"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()
    rows = load_jsonl(a.data, a.n, a.shuffle)
    print(f"[{a.task}] n={len(rows)} model={a.model} conc={a.conc} maxtok={a.max_tokens} base={a.base}", flush=True)
    GI = "\nSolve it step by step, then end with '#### <number>'."

    def work(ir):
        i, r = ir
        if a.task == "gsm8k":
            txt = chat(a.base, a.model, r["question"] + GI, a.max_tokens)
            ok = num_eq(gsm8k_pred(txt), gsm8k_gold(r))
        else:
            txt = chat(a.base, a.model, mmlu_prompt(r), a.max_tokens)
            ok = (mmlu_pred(txt, len(r["options"])) == r["answer"])
        return ok, txt.startswith("__ERR__")

    correct = err = done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        futs = [ex.submit(work, (i, r)) for i, r in enumerate(rows)]
        for fu in as_completed(futs):
            ok, e = fu.result()
            done += 1; correct += int(ok); err += int(e)
            if done % 50 == 0:
                print(f"  {done}/{len(rows)} acc={correct/done:.3f} err={err} {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.task}] FINAL acc={correct/len(rows):.4f} ({correct}/{len(rows)}) err={err} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
