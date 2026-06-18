#!/usr/bin/env python3
"""GSM8K accuracy eval for Qwen3.5-9B W4A16 vs W4A8 — isolates the int8-activation cost.

Same base model, same int4/group-128 weights; ONLY activation precision differs
(w4a16 = fp16 acts, w4a8 = int8 dynamic per-token acts). Both run with DEFAULT
attention on the same image so the single variable is the GEMM quant.

Eval protocol (Qwen3.5 is a thinking model — these are mandatory or you get false collapse):
  * vLLM OFFLINE batched inference (LLM + SamplingParams), not a server.
  * NEVER greedy: temperature=0.6, top_p=0.95, fixed seed.
  * Force thinking via the chat template (thinking enabled, default for Qwen3.5).
  * max_tokens=24576 cap, max_model_len=32768, max_num_seqs<=82.
  * Post-think extraction: take the span after the last </think>, parse \boxed{} if
    present else the last number. Gold = number after #### in the answer field.

Usage:
  python3 gsm8k_eval.py --model /m16 --n 250 --out /out/eval_w4a16.json --tag w4a16
  python3 gsm8k_eval.py --model /m8  --n 250 --out /out/eval_w4a8.json  --tag w4a8
  # smoke:
  python3 gsm8k_eval.py --model /m16 --n 5 --smoke --tag w4a16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

# --------------------------------------------------------------------------- #
# Scoring / extraction — lifted verbatim from dense2moe-ci src/eval/scoring.py
# (the single source of truth for how GSM8K is judged in this project), plus a
# \boxed{} extractor as the primary answer span per the task protocol.
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def split_after_think(text: str) -> str:
    """Answer span after the LAST </think>; unchanged if no </think> (back-compat)."""
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):] if idx != -1 else text


def gold_number(answer_field: str) -> str | None:
    """Gold numeric answer from a GSM8K answer field (value after ####)."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    return m.group(1).replace(",", "") if m else None


def _boxed_number(text: str) -> str | None:
    """Last \\boxed{...} numeric content, if present (handles nested-ish braces)."""
    last = None
    for m in re.finditer(r"\\boxed\s*\{", text):
        i = m.end()
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        content = "".join(buf).replace(",", "").replace("$", "").replace("\\", " ")
        nums = _NUM.findall(content)
        if nums:
            last = nums[-1].rstrip(".")
    return last


def pred_number(completion: str) -> str | None:
    """Predicted number: \\boxed{} in the post-</think> span first, else last number."""
    ans = split_after_think(completion)
    boxed = _boxed_number(ans)
    if boxed is not None:
        return boxed
    ans = ans.replace(",", "")
    nums = _NUM.findall(ans)
    return nums[-1].rstrip(".") if nums else None


def numbers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Dataset — read the cached openai/gsm8k 'main' test parquet (no internet needed).
# --------------------------------------------------------------------------- #
def load_gsm8k_test(n: int):
    candidates = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/"
            "*/main/test-00000-of-00001.parquet"
        )
    ) + glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--gsm8k/snapshots/"
            "*/main/test-00000-of-00001.parquet"
        )
    ) + glob.glob("/out/gsm8k_test.parquet")
    if not candidates:
        raise FileNotFoundError("gsm8k main test parquet not found in HF cache or /out")
    path = sorted(candidates)[0]
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    cols = tbl.to_pydict()
    qs, ans = cols["question"], cols["answer"]
    rows = [{"question": qs[i], "answer": ans[i]} for i in range(len(qs))]
    print(f"[data] loaded {len(rows)} gsm8k test rows from {path}", flush=True)
    return rows[:n]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=24576)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    problems = load_gsm8k_test(args.n)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Build chat prompts WITH thinking enabled (Qwen3.5 default). The question is a
    # single user turn; add_generation_prompt opens the assistant turn.
    prompts = []
    for p in problems:
        msgs = [{"role": "user", "content": p["question"].strip()}]
        try:
            txt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            # template may not accept enable_thinking kwarg; Qwen3.5 thinks by default
            txt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(txt)

    print(f"[prompt sample]\n{prompts[0][:600]}\n...", flush=True)

    # Qwen3.5-9B is a VL (image-text-to-text) checkpoint; GSM8K is text-only.
    # Disable the multimodal limit so vLLM does not reserve image/video slots.
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=False,
        seed=args.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0

    # vLLM may reorder; map back by prompt index via request output ordering
    # (llm.generate returns outputs in input order).
    results = []
    n_correct = 0
    n_thought = 0
    for i, (p, o) in enumerate(zip(problems, outs)):
        text = o.outputs[0].text
        thought = "</think>" in text
        n_thought += int(thought)
        pred = pred_number(text)
        gold = gold_number(p["answer"])
        ok = numbers_match(pred, gold)
        n_correct += int(ok)
        results.append({
            "idx": i,
            "question": p["question"],
            "gold": gold,
            "pred": pred,
            "passed": ok,
            "thought": thought,
            "gen_len": len(o.outputs[0].token_ids),
            "completion": text,
        })

    n = len(results)
    acc = n_correct / n if n else 0.0
    summary = {
        "tag": args.tag,
        "model": args.model,
        "n": n,
        "correct": n_correct,
        "accuracy": acc,
        "thought_frac": n_thought / n if n else 0.0,
        "wall_s": round(wall, 1),
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_tokens": args.max_tokens, "seed": args.seed},
    }
    print("\n==== SUMMARY ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    out_path = args.out or f"/out/eval_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f)
    print(f"[saved] {out_path}", flush=True)

    if args.smoke:
        print("\n==== SMOKE SAMPLES ====", flush=True)
        for r in results[:5]:
            tail = split_after_think(r["completion"])[-300:]
            print(f"--- idx {r['idx']} gold={r['gold']} pred={r['pred']} "
                  f"ok={r['passed']} thought={r['thought']} len={r['gen_len']}",
                  flush=True)
            print(f"    post-think tail: {tail!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
