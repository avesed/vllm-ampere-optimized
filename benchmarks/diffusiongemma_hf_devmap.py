#!/usr/bin/env python3
"""Isolation test: load cyankiwi W4A16 DiffusionGemma in pure transformers with device_map="auto"
(spreads across both 3090s -> no OOM) and run the OFFICIAL block-diffusion generate. This has NO
vLLM, NO tensor-parallel soft-embed all_reduce, NO #46177 patch — so coherent output here means the
W4A16 quant is good and the garbage is a vLLM/TP issue; garbage here means the quant itself is bad.
Run in dgquant-venv (transformers 5.12 + compressed-tensors). CUDA_VISIBLE_DEVICES=0,1.
"""
import os, torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer

M = os.environ.get("MODEL", "/home/coder/models/diffusiongemma-26B-A4B-it-AWQ-INT4")
print(f"[hf] loading {M} device_map=auto ...", flush=True)
tok = AutoTokenizer.from_pretrained(M)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(M, device_map="auto", dtype=torch.bfloat16)
model.eval()
print("[hf] device map:", getattr(model, "hf_device_map", "n/a"), flush=True)

prompts = ["The capital of France is",
           "用一句话解释什么是张量并行：",
           "Write a Python function that reverses a string:",
           "List three prime numbers:"]
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = tok(p, return_tensors="pt").input_ids
    ids = ids.to(model.device if hasattr(model, "device") else "cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.95)
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    print(f"[{p[:28]}] -> {txt[:200]!r}", flush=True)
print("HF_DEVMAP_DONE", flush=True)
