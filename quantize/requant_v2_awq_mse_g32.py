#!/usr/bin/env python3
"""Re-quantize Qwen3.5-9B (dense hybrid VL) with the USER'S proven recipe
(ref: Avesed/Qwopus3.6-35B-A3B-v1-int4-mixed recipe.yaml): AWQModifier (scale-search
smoothing) + QuantizationModifier with observer=mse, group_size=32, symmetric int4 weights.
NOT SmoothQuant+GPTQ+minmax (that produced garbage).

The method is expressed as a SERIALIZED YAML recipe (the user's exact form) so it is
immune to llmcompressor Python-API drift. In llmcompressor 0.12.0 the Python symbol
`from llmcompressor.modifiers.awq import AWQModifier` is a deprecated FACTORY that returns
a *list* `[AWQTransformModifier, QuantizationModifier]` (NOT a Modifier), which silently
breaks `recipe=[awq, quant]`. The YAML name `AWQModifier` instead resolves to the real
`llmcompressor.modifiers.transform.awq.AWQModifier` (the AWQ scale-search smoother).

W4A16 (input_activations: null) and W4A8 (int8 dynamic per-token act) share the SAME AWQ
smoothing + SAME int4/g32/mse weight quant -> only the activation config differs ->
clean isolation of activation precision.

Usage:  python requant_v2_awq_mse_g32.py <out_dir> {w4a16|w4a8}
Load the VL instruct on CPU with AutoModelForImageTextToText (AWQ/quant onloads layers to
the GPU one at a time; device_map=auto OOMs). Calib = Magicoder-Evol-Instruct-110K
(cached & real on the sandbox; ultrachat_200k there is metadata-only and fails offline).
"""
import os
import random
import sys

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B")  # native instruct VL, in HF cache
OUT = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "w4a16"
assert MODE in ("w4a16", "w4a8")
NUM = int(os.environ.get("NUM_CALIB", "512"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))
SEED = int(os.environ.get("SEED", "1234"))

# ---------------------------------------------------------------------------
# Recipe (serialized YAML == the user's reference method, adapted to the dense 9B).
#
# AWQModifier smoothing mappings for DENSE Qwen3.5-9B (layers under
# model.language_model.layers.N.*): the 8 full-attn layers (3,7,11,...,31) have
# self_attn; the 24 GatedDeltaNet layers have linear_attn (kept bf16 / ignored).
#   * post_attention_layernorm -> mlp.gate_proj/up_proj   (ALL layers)
#   * mlp.up_proj              -> mlp.down_proj            (ALL layers)
#   * input_layernorm         -> self_attn.{q,k,v}_proj   (FULL-ATTN layers only)
#
# QuantizationModifier: int4 group_size=32 symmetric weights, observer=mse;
# input_activations null (w4a16) or int8 dynamic per-token (w4a8).
# ignore: lm_head, embed_tokens, linear_attn (GDN), visual tower, mtp head.
# ---------------------------------------------------------------------------
FULL_ATTN = r"3|7|11|15|19|23|27|31"
INPUT_ACTS = (
    "{num_bits: 8, type: int, symmetric: true, strategy: token, dynamic: true, observer: null}"
    if MODE == "w4a8"
    else "null"
)

RECIPE = f"""
default_stage:
  default_modifiers:
    AWQModifier:
      mappings:
      - smooth_layer: re:.*post_attention_layernorm$
        balance_layers: ['re:.*mlp[.]gate_proj$', 're:.*mlp[.]up_proj$']
      - smooth_layer: re:.*mlp[.]up_proj$
        balance_layers: ['re:.*mlp[.]down_proj$']
      - smooth_layer: re:.*layers\\.({FULL_ATTN})\\.input_layernorm$
        balance_layers: ['re:.*self_attn[.]q_proj$', 're:.*self_attn[.]k_proj$', 're:.*self_attn[.]v_proj$']
      duo_scaling: both
      n_grid: 20
    QuantizationModifier:
      config_groups:
        group_0:
          targets: [Linear]
          weights:
            num_bits: 4
            type: int
            symmetric: true
            group_size: 32
            strategy: group
            dynamic: false
            observer: mse
          input_activations: {INPUT_ACTS}
          output_activations: null
      targets: [Linear]
      ignore: [lm_head, 're:.*embed_tokens', 're:.*linear_attn.*', 're:.*visual.*', 're:.*mtp.*']
"""


def build_calib(tok) -> Dataset:
    """Magicoder-Evol-Instruct-110K via chat template; fully OFFLINE, same seed both modes."""
    random.seed(SEED)
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    idx = list(range(len(ds)))
    random.shuffle(idx)
    samples = []
    for i in idx:
        if len(samples) >= NUM:
            break
        ex = ds[i]
        msgs = [
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = ex["instruction"] + "\n" + ex["response"]
        if text and text.strip():
            samples.append({"text": text})
    print(f"[calib] built {len(samples)} Magicoder samples (seed={SEED})", flush=True)
    return Dataset.from_list(samples)


print(f"[requant] {MODEL} -> {OUT}  mode={MODE}  AWQ+mse g32 (user YAML recipe)", flush=True)
print(RECIPE, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(  # VL instruct; CPU
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
calib = build_calib(tok)

oneshot(
    model=model,
    dataset=calib,
    recipe=RECIPE,
    processor=tok,
    max_seq_length=MAXLEN,
    num_calibration_samples=len(calib),
    output_dir=OUT,
    save_compressed=True,
)
tok.save_pretrained(OUT)

import json  # noqa: E402

qc = json.load(open(os.path.join(OUT, "config.json")))["quantization_config"]
g = qc["config_groups"]["group_0"]
print(
    f"[requant] DONE {MODE}: format={qc.get('format')} "
    f"w.bits={g['weights']['num_bits']} w.gs={g['weights']['group_size']} "
    f"w.observer={g['weights'].get('observer')} "
    f"input_act={(g['input_activations'] or {}).get('num_bits')}",
    flush=True,
)
print("REQUANT_DONE", OUT, MODE, flush=True)
