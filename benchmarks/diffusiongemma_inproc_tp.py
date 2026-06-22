#!/usr/bin/env python3
"""In-process DiffusionGemma TP validation (bypasses the eb28452 v0.24.0-dev HTTP API-server
`_IncludedRouter` routing bug; the engine itself is fine — 9B/35B were validated in-process too).

Loads the cyankiwi W4A16 ckpt with tensor_parallel_size=2 (needs the #46177 TP soft-embed fix
overlaid onto diffusion_gemma.py), runs coherence prompts, and probes the expert quant method.
Gate 0 = run without VLLM_MARLIN_INPUT_DTYPE; Gate 1 = run with it set to int8.

NOTE: must run under `if __name__ == "__main__"` — TP>1 spawns workers via multiprocessing,
which re-imports this module; module-level LLM() would recursively spawn (freeze_support error).

Env: MODEL(/m), TP(2), MAXLEN(2048), MAXSEQS(1), GPUMEM(0.90).
"""
import os
from vllm import LLM, SamplingParams

M = os.environ.get("MODEL", "/m")
TP = int(os.environ.get("TP", "2"))


def main():
    mode = os.environ.get("VLLM_MARLIN_INPUT_DTYPE", "(default fp16-act)")
    print(f"[inproc] model={M} TP={TP} marlin_input_dtype={mode}", flush=True)

    llm = LLM(model=M, tensor_parallel_size=TP,
              max_num_seqs=int(os.environ.get("MAXSEQS", "1")),
              max_model_len=int(os.environ.get("MAXLEN", "2048")),
              enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GPUMEM", "0.90")),
              limit_mm_per_prompt={"image": 0, "video": 0})

    # --- expert quant-method probe (G2: experts must hit the Marlin WNA16 MoE path, not CUTLASS) ---
    def probe(model):
        seen = {}
        for n, mod in model.named_modules():
            qm = getattr(mod, "quant_method", None)
            if qm is not None and ("experts" in n or "mlp" in n):
                t = type(qm).__name__
                seen[t] = seen.get(t, 0) + 1
        print("[inproc] quant_method histogram (experts/mlp):", seen, flush=True)
    try:
        llm.apply_model(probe)
    except Exception as e:
        print(f"[inproc] probe skipped: {type(e).__name__}: {e}", flush=True)

    # --- coherence ---
    sp = SamplingParams(max_tokens=56, temperature=0.6, top_p=0.95)
    prompts = ["The capital of France is",
               "用一句话解释什么是张量并行：",
               "Write a Python function that reverses a string:",
               "List three prime numbers:"]
    outs = llm.generate(prompts, sp)
    print("=== COHERENCE ===", flush=True)
    for p, o in zip(prompts, outs):
        print(f"  [{p[:30]}] -> {o.outputs[0].text[:180]!r}", flush=True)
    print("INPROC_DONE", flush=True)


if __name__ == "__main__":
    main()
