#!/usr/bin/env python3
"""Generic recipe runner — quantize a model with a YAML recipe via llm-compressor oneshot.

  python run_recipe.py <hf_model> <recipe.yaml> <out_dir> [num_calib=256] [max_len=2048]

Loads on CPU (the AWQ/GPTQ sequential pipeline onloads one layer to GPU at a time; pre-filling the
GPU OOMs it). Needs `pip install llmcompressor datasets` in the env. Works for the W4A16 (AWQ+mse)
and W8A8 (int8) recipes in this dir."""
import os, sys, random
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None
from datasets import load_dataset, Dataset
from llmcompressor import oneshot

MODEL, RECIPE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NUM = int(sys.argv[4]) if len(sys.argv) > 4 else 256
MAXLEN = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
# Optional: sequential-pipeline target granularity. "Linear" onloads one Linear at a time
# (low GPU mem) — use for schemes without AWQ. AWQ recipes instead keep the default
# (decoder-layer, needed for the smooth->balance mappings) and offload the cache via the
# recipe's offload_device: cpu.
SEQT = sys.argv[6] if len(sys.argv) > 6 else None
SEED = 1234

def build_calib(tok):
    random.seed(SEED)
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    idx = list(range(len(ds))); random.shuffle(idx)
    samples = []
    for i in idx:
        if len(samples) >= NUM:
            break
        ex = ds[i]
        msgs = [{"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["response"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = ex["instruction"] + "\n" + ex["response"]
        if text and text.strip():
            samples.append({"text": text})
    print(f"[calib] {len(samples)} Magicoder samples (seed={SEED})", flush=True)
    return Dataset.from_list(samples)

print(f"[run_recipe] {MODEL} -> {OUT}  recipe={RECIPE}  calib={NUM} maxlen={MAXLEN} seq_targets={SEQT}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
calib = build_calib(tok)
ok = dict(model=model, dataset=calib, recipe=RECIPE, processor=tok, max_seq_length=MAXLEN,
          num_calibration_samples=len(calib), output_dir=OUT, save_compressed=True)
if SEQT:
    ok["sequential_targets"] = [SEQT]
oneshot(**ok)
tok.save_pretrained(OUT)
# Save the full processor too (preprocessor_config.json / video_preprocessor_config.json) so VL
# checkpoints are servable as-is; no-op for text-only models that have no processor.
if AutoProcessor is not None:
    try:
        AutoProcessor.from_pretrained(MODEL, trust_remote_code=True).save_pretrained(OUT)
    except Exception as e:
        print(f"[run_recipe] no processor saved ({type(e).__name__})", flush=True)

import json  # noqa: E402
qc = json.load(open(os.path.join(OUT, "config.json"))).get("quantization_config", {})
g = qc.get("config_groups", {}).get("group_0", {})
ia = g.get("input_activations")
print(f"[run_recipe] DONE: format={qc.get('format')} "
      f"w.bits={g.get('weights', {}).get('num_bits')} w.gs={g.get('weights', {}).get('group_size')} "
      f"act={ia.get('num_bits') if ia else 'none(W*A16)'}", flush=True)
