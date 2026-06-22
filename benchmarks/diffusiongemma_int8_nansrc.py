#!/usr/bin/env python3
"""Find the SOURCE module of the int8-act NaN on DiffusionGemma (confirmed: logits=NaN, all-token-0).
Registers a forward hook on every leaf module that prints NANSRC when output is non-finite, tagging
whether the inputs were finite (finite-in + NaN-out = the source). Module-level hook fn so apply_model
can pickle it (VLLM_ALLOW_INSECURE_SERIALIZATION=1). int8 via env. Run eb28452 image, entrypoint python3.
"""
import os
import torch
from vllm import LLM, SamplingParams

M = os.environ.get("MODEL", "/m")


def _finite(x):
    if isinstance(x, (tuple, list)):
        return all(_finite(t) for t in x)
    if torch.is_tensor(x) and x.dtype.is_floating_point:
        return bool(torch.isfinite(x).all())
    return True


def _mk_hook(name):
    def hook(_m, inp, out):
        of = _finite(out)
        if not of:
            inf = _finite(inp)
            print(f"NANSRC|{name}|in_finite={inf}|out_finite=False", flush=True)
    return hook


def add_hooks(model):
    n = 0
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:  # leaf modules only
            mod.register_forward_hook(_mk_hook(name))
            n += 1
    print(f"NANSRC_HOOKED={n}", flush=True)


def main():
    llm = LLM(model=M, tensor_parallel_size=1, max_num_seqs=1, max_model_len=1024,
              enforce_eager=True, gpu_memory_utilization=0.92,
              hf_overrides={"canvas_length": 64}, limit_mm_per_prompt={"image": 0, "video": 0})
    llm.apply_model(add_hooks)
    out = llm.chat([{"role": "user", "content": "What is the capital of France?"}],
                   SamplingParams(max_tokens=8, temperature=0.6))
    print(f"text={out[0].outputs[0].text!r}", flush=True)
    print("NANSRC_DONE", flush=True)


if __name__ == "__main__":
    main()
