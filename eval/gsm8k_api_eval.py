#!/usr/bin/env python3
"""GSM8K accuracy eval against a RUNNING `vllm serve` (OpenAI API), not offline LLM().

Serving is the path users actually run, and this project's rule is to test over the API. Scoring is
imported from gsm8k_eval.py so both harnesses judge identically.

Serve the model WITHOUT `--reasoning-parser` so the raw `<think>...</think>` stays in `content` —
the extractor takes the span after the last `</think>`.

  python3 gsm8k_api_eval.py --base http://localhost:8000 --tag w4a16 --n 250 \
      --temperature 1.0 --top-p 0.95 --top-k 20 --concurrency 32 --out /tmp/w4a16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_eval import gold_number, load_gsm8k_test, numbers_match, pred_number  # noqa: E402

_lock = threading.Lock()


def post(base: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--reasoning-effort", default=None)
    # Without this, extraction has to guess: the model often states the answer in bold and then
    # keeps talking, so "last number in the span" picks up trailing commentary and scores a correct
    # answer wrong. Asking for \boxed{} is the standard reasoning-model GSM8K protocol.
    ap.add_argument("--boxed", action="store_true",
                    help=r"append an instruction to put the final answer in \boxed{}")
    args = ap.parse_args()
    suffix = "\n\nPut your final numeric answer inside \\boxed{}." if args.boxed else ""

    problems = load_gsm8k_test(args.n)
    model = json.load(urllib.request.urlopen(args.base + "/v1/models", timeout=30))["data"][0]["id"]
    print(f"[eval {args.tag}] model={model} n={len(problems)} conc={args.concurrency} "
          f"temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"effort={args.reasoning_effort or 'default'} boxed={args.boxed}", flush=True)

    done = [0]
    t0 = time.time()

    def run(i_p):
        i, p = i_p
        body = {
            "model": model,
            "messages": [{"role": "user", "content": p["question"].strip() + suffix}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        if args.reasoning_effort:
            body["reasoning_effort"] = args.reasoning_effort
        try:
            r = post(args.base, body, args.timeout)
            msg = r["choices"][0]["message"]
            text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
            ctoks = r.get("usage", {}).get("completion_tokens", 0)
            finish = r["choices"][0].get("finish_reason")
        except Exception as e:  # a failed request scores as wrong, and is reported
            text, ctoks, finish = "", 0, f"ERR:{type(e).__name__}"
        with _lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  [{done[0]}/{len(problems)}] {time.time()-t0:.0f}s", flush=True)
        return i, text, ctoks, finish

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(run, enumerate(problems)))
    wall = time.time() - t0

    outs.sort(key=lambda x: x[0])
    results, n_ok, n_think, n_trunc, n_err, toks = [], 0, 0, 0, 0, 0
    for (i, text, ctoks, finish), p in zip(outs, problems):
        pred, gold = pred_number(text), gold_number(p["answer"])
        ok = numbers_match(pred, gold)
        n_ok += int(ok)
        n_think += int("</think>" in text)
        n_trunc += int(finish == "length")
        n_err += int(str(finish).startswith("ERR:"))
        toks += ctoks
        # Keep the answer span (and the whole text when wrong) so scoring changes can be replayed
        # offline instead of costing another serve + generate cycle.
        from gsm8k_eval import split_after_think
        results.append({"idx": i, "gold": gold, "pred": pred, "ok": ok,
                        "completion_tokens": ctoks, "finish": finish,
                        "answer_span": split_after_think(text)[:1500],
                        "text": None if ok else text[:8000]})

    acc = 100.0 * n_ok / len(problems)
    print(f"\nRESULT {args.tag}: GSM8K {n_ok}/{len(problems)} = {acc:.1f}%")
    print(f"  thinking spans: {n_think}/{len(problems)} | truncated: {n_trunc} | errors: {n_err}")
    print(f"  mean completion tokens: {toks/max(1,len(problems)):.0f} | wall {wall/60:.1f} min", flush=True)
    if args.out:
        json.dump({"tag": args.tag, "n": len(problems), "correct": n_ok, "acc": acc,
                   "thinking": n_think, "truncated": n_trunc, "errors": n_err,
                   "mean_completion_tokens": toks / max(1, len(problems)),
                   "wall_s": wall, "temperature": args.temperature, "top_p": args.top_p,
                   "top_k": args.top_k, "reasoning_effort": args.reasoning_effort, "boxed": args.boxed,
                   "results": results}, open(args.out, "w"))
        print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
