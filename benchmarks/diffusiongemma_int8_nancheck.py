#!/usr/bin/env python3
"""Diagnose WHY int8-act on DiffusionGemma produces EMPTY output (64 all-pad tokens), not garbage.
Hypothesis: per-token int8 activation quant divides by a zero/degenerate scale on the canvas's
masked/noised positions -> NaN -> degenerate logits -> all-pad. (The marlin kernel test passed on
randn activations, which never hit zero-norm tokens.) Registers NaN hooks on MoE/mlp + dumps the
output token_ids (all-identical = NaN-argmax signature). TP=1, canvas64 via hf_overrides to fit one card.
Set VLLM_MARLIN_INPUT_DTYPE=int8 in env (or not, to compare). Run with eb28452 image, entrypoint python3.
"""
import os
import torch
from vllm import LLM, SamplingParams

M = os.environ.get("MODEL", "/m")
mode = os.environ.get("VLLM_MARLIN_INPUT_DTYPE", "(fp16-act)")


def main():
    print(f"[nan] marlin_input_dtype={mode}", flush=True)
    llm = LLM(model=M, tensor_parallel_size=1, max_num_seqs=1, max_model_len=1024,
              enforce_eager=True, gpu_memory_utilization=0.92,
              hf_overrides={"canvas_length": 64}, limit_mm_per_prompt={"image": 0, "video": 0})

    for q in ["What is the capital of France?", "用一句话解释什么是张量并行",
              "Write a Python function that reverses a string."]:
        out = llm.chat([{"role": "user", "content": q}],
                       SamplingParams(max_tokens=96, temperature=0.6))
        print(f"[nan] [{q[:26]}] -> {out[0].outputs[0].text[:180]!r}", flush=True)
    print("NANCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
