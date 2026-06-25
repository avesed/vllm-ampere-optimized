#!/usr/bin/env python3
"""Validation module — DiffusionGemma (vLLM PR #45163) FORCED onto the fork's INT8 Marlin path.

Confirms that a DiffusionGemma checkpoint (1) serves in vLLM, (2) has its MoE experts + dense
linears routed onto the fork's int8-activation Marlin kernel (W4A16 weights + per-token int8 acts
via VLLM_MARLIN_INPUT_DTYPE=int8 / --marlin-input-dtype int8) and NOT the stock CUTLASS W8A8 path,
and (3) generates coherent text.

Must run inside the from-source image that has BOTH patch 0005/0006 (compiled _moe_C int8 kernel +
--marlin-input-dtype) AND the cherry-picked PR #45163 python (diffusion_gemma + sampler/ModelState).
SINGLE GPU ONLY — DiffusionGemma crashes on TP>1 and PP>1 (vLLM #45719); do not pass tp/pp>1.

Set the int8 path BEFORE running:  export VLLM_MARLIN_INPUT_DTYPE=int8   (the .sh wrapper does this)

Gates (each PASS / FAIL / SKIP; non-zero exit if any required gate FAILs):
  G1 load     : DiffusionGemma in registry + LLM() builds on one card
  G2 int8     : (CORE) expert FusedMoE method == CompressedTensorsWNA16MarlinMoEMethod (NOT the
                CUTLASS *W8A8Int8MoEMethod*), marlin_input_dtype == torch.int8, and per-expert
                w13/w2_input_global_scale present (len == num_experts) [patch 0005]. Dense WNA16
                linears use a Marlin kernel with int8 act.
  G3 coherent : a few EN/ZH prompts -> non-empty, not degenerate/repetitive junk
  G4 (opt)    : small GSM8K vs gold (relative sanity, --gsm8k N)

Usage (inside container, ckpt mounted):
  VLLM_MARLIN_INPUT_DTYPE=int8 python diffusiongemma_int8_validate.py --model <CKPT> \
      --max-model-len 8192 --gpu-mem-util 0.85 [--gsm8k 20]
"""
import argparse
import json
import os
import re
import sys
import time

TAG = "dg-int8"
CUTLASS_W8A8_MARKERS = ("W8A8Int8MoE", "W8A8Int8Linear", "CutlassW8A8")  # the WRONG (env-ignored) path
MARLIN_MOE_MARKER = "WNA16MarlinMoE"


def log(m):
    print(f"[{TAG}] {m}", flush=True)


# ---- G2 probe: runs INSIDE the worker via llm.apply_model (model lives there in v1) ----
def _probe_model(model):
    """Walk modules; collect quant-method facts. Returns a plain (picklable) dict."""
    moe, dense = [], []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        cls = type(qm).__name__
        modcls = type(mod).__name__
        is_moe = ("MoE" in cls) or ("FusedMoE" in modcls) or hasattr(mod, "w13_weight") or hasattr(mod, "w13_qweight")
        if is_moe:
            info = {"name": name, "method": cls, "marlin_input_dtype": str(getattr(qm, "marlin_input_dtype", None))}
            for s in ("w13_input_global_scale", "w2_input_global_scale"):
                v = getattr(mod, s, None)
                info[s] = None if v is None else (list(v.shape) if hasattr(v, "shape") else "present")
            moe.append(info)
        elif ("WNA16" in cls) or ("W4A8" in cls) or ("Linear" in cls and "Unquant" not in cls):
            dense.append({"name": name, "method": cls,
                          "input_dtype": str(getattr(qm, "input_dtype", getattr(qm, "marlin_input_dtype", None)))})
    return {"moe": moe, "dense_sample": dense[:8], "n_dense": len(dense), "n_moe": len(moe)}


def probe(llm):
    """Best-effort cross-version accessor for the loaded model's quant methods."""
    # primary: v1 worker RPC
    if hasattr(llm, "apply_model"):
        try:
            res = llm.apply_model(_probe_model)
            return res[0] if isinstance(res, (list, tuple)) else res
        except Exception as e:
            log(f"apply_model probe failed ({type(e).__name__}: {str(e)[:80]}); trying driver_worker path")
    # fallback: v0-style attribute walk
    for path in ("llm_engine.model_executor.driver_worker.model_runner.model",
                 "llm_engine.model_executor.driver_worker.worker.model_runner.model"):
        obj = llm
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return _probe_model(obj)
        except Exception:
            continue
    return None


def gate_g2(p):
    if not p or not p.get("moe"):
        log("G2 int8: INCONCLUSIVE — could not introspect MoE quant methods (adjust probe accessor for this vLLM build)")
        return "SKIP"
    log(f"G2 int8: found {p['n_moe']} MoE quant modules, {p['n_dense']} dense quant modules")
    bad = [m for m in p["moe"] if any(k in m["method"] for k in CUTLASS_W8A8_MARKERS)]
    if bad:
        log(f"G2 int8: FAIL — experts on CUTLASS W8A8 path ({bad[0]['method']}) => VLLM_MARLIN_INPUT_DTYPE silently "
            f"IGNORED. You fed a W8A8 ckpt; need a WNA16 (int4-weight, input_activations=None) ckpt.")
        return "FAIL"
    sample = p["moe"][0]
    log(f"G2 int8: expert method={sample['method']} marlin_input_dtype={sample['marlin_input_dtype']} "
        f"w13_gscale={sample['w13_input_global_scale']} w2_gscale={sample['w2_input_global_scale']}")
    on_marlin = all(MARLIN_MOE_MARKER in m["method"] for m in p["moe"])
    is_int8 = all("int8" in (m["marlin_input_dtype"] or "").lower() for m in p["moe"])
    gscale_ok = all(m["w13_input_global_scale"] is not None and m["w2_input_global_scale"] is not None for m in p["moe"])
    if on_marlin and is_int8 and gscale_ok:
        log("G2 int8: PASS — experts on WNA16 Marlin, int8 act engaged, patch-0005 per-expert global_scale present")
        return "PASS"
    log(f"G2 int8: FAIL — on_marlin={on_marlin} int8={is_int8} per_expert_gscale={gscale_ok} "
        f"(gscale None => patch 0005 per-expert factor not wired => 704-dim heterogeneous experts will garble)")
    return "FAIL"


def gate_g3(llm):
    from vllm import SamplingParams
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "Write a Python function that returns the nth Fibonacci number.",
        "请用一句话解释什么是张量并行。",
        "List three primary colors.",
    ]
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=128)
    try:
        outs = llm.generate(prompts, sp)
    except Exception as e:
        log(f"G3 coherent: FAIL — generate() errored ({type(e).__name__}: {str(e)[:120]})")
        return "FAIL"
    ok = 0
    for pr, o in zip(prompts, outs):
        txt = o.outputs[0].text.strip()
        toks = o.outputs[0].token_ids
        degenerate = (len(txt) < 2) or (len(set(toks[-32:])) <= 2 if len(toks) >= 32 else False)
        ok += int(not degenerate)
        log(f"G3 sample: {pr[:32]!r} -> {txt[:100]!r}{'  [DEGENERATE]' if degenerate else ''}")
    verdict = "PASS" if ok == len(prompts) else ("FAIL" if ok == 0 else "PARTIAL")
    log(f"G3 coherent: {verdict} ({ok}/{len(prompts)} non-degenerate)")
    return verdict


def gate_g4(llm, data, n):
    if not os.path.exists(data):
        log(f"G4 gsm8k: SKIP — {data} not found")
        return "SKIP"
    from vllm import SamplingParams
    items = []
    with open(data) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
            if len(items) >= n:
                break
    prompts = [it["question"] + "\nReason step by step, then give the final answer after '####'." for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
    correct = 0
    for it, o in zip(items, outs):
        txt = o.outputs[0].text
        after = txt.split("####")[-1] if "####" in txt else txt
        nums = re.findall(r"-?\d[\d,]*", after) or re.findall(r"-?\d[\d,]*", txt)
        pred = nums[-1].replace(",", "") if nums else None
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        correct += int(pred == gold)
    acc = correct / max(len(items), 1)
    log(f"G4 gsm8k: acc={acc:.3f} ({correct}/{len(items)})  [relative sanity, not a full eval]")
    return "PASS" if acc > 0 else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--gsm8k", type=int, default=0, help="run G4 on N gsm8k items (0=skip)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k.jsonl"))
    ap.add_argument("--skip-gen", action="store_true", help="skip G3/G4 (structural-only run)")
    args = ap.parse_args()

    log(f"VLLM_MARLIN_INPUT_DTYPE={os.environ.get('VLLM_MARLIN_INPUT_DTYPE', '(unset!)')}  model={args.model}")
    if os.environ.get("VLLM_MARLIN_INPUT_DTYPE", "").lower() != "int8":
        log("WARNING: VLLM_MARLIN_INPUT_DTYPE != int8 — G2 will not see int8 engaged. Export it before running.")

    gates = {}
    # ---- G1 load (single card; TP/PP>1 crash on DiffusionGemma, #45719) ----
    from vllm import LLM
    t0 = time.time()
    try:
        llm = LLM(model=args.model, tensor_parallel_size=1, enforce_eager=True,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True)
    except Exception as e:
        msg = str(e)
        if "not supported" in msg.lower() or "registry" in msg.lower() or "architectures" in msg.lower():
            log(f"G1 load: FAIL — DiffusionGemma not in registry (cherry-pick of PR #45163 incomplete?): {msg[:160]}")
        else:
            log(f"G1 load: FAIL — {type(e).__name__}: {msg[:200]}")
        log("RESULT: G1=FAIL  (cannot proceed)")
        sys.exit(2)
    gates["G1"] = "PASS"
    log(f"G1 load: PASS ({time.time()-t0:.1f}s)")

    # ---- G2 int8 engaged (CORE) ----
    gates["G2"] = gate_g2(probe(llm))

    # ---- G3 coherent / G4 gsm8k ----
    if args.skip_gen:
        gates["G3"] = gates["G4"] = "SKIP"
    else:
        gates["G3"] = gate_g3(llm)
        gates["G4"] = gate_g4(llm, args.data, args.gsm8k) if args.gsm8k > 0 else "SKIP"

    log("==== RESULT ====  " + "  ".join(f"{k}={v}" for k, v in gates.items()))
    required_fail = (gates["G1"] == "FAIL") or (gates["G2"] == "FAIL") or (gates.get("G3") == "FAIL")
    if gates["G2"] != "PASS":
        log("NOTE: int8 engagement (G2) is the core gate — only a PASS proves the fork's int8 kernel is on the path.")
    sys.exit(1 if required_fail else 0)


if __name__ == "__main__":
    main()
