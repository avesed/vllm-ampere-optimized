"""Unified, FAIR re-quantization of Qwen3.5-9B (INSTRUCT) for a clean W4A16 vs W4A8 pair.

Goal: produce a W4A16 and a W4A8 checkpoint that share ONE high-quality weight-quant
method so the ONLY variable is activation precision (fp16 acts vs int8 dynamic-per-token
acts). The pre-existing ~/models/Qwen3.5-9B-{w4a16,w4a8} were made with two DIFFERENT
methods (w4a16 = RTN no-calib; w4a8 = GPTQ no-smoothing) and are not apples-to-apples.

Unified method (applied IDENTICALLY to both):
  * Activation smoothing: SmoothQuantModifier(smoothing_strength=0.8)  (AWQ-style:
    migrates activation outliers into the weights before rounding). In llmcompressor
    0.12 the standalone `AWQModifier` is a deprecated shim split into a transform +
    QuantizationModifier and does RTN rounding, not GPTQ; SmoothQuant is the supported
    way to compose activation smoothing BEFORE GPTQ Hessian rounding.
  * Weight rounding: GPTQModifier (Hessian/OBQ rounding) with an EXPLICIT config_groups
    so we get group_size=32 symmetric int4 weights (the preset W4A8/W4A16 are g128).
  * group_size = 32, symmetric int4, observer memoryless_minmax.
  * Same calibration set, same order, same seed for both runs.

FAIRNESS / weight-sharing:
  Both runs use the SAME weight scheme (int4 g32 sym) and the SAME smoothing + SAME
  calib + SAME seed. The two recipes differ ONLY in `input_activations`:
      W4A16 -> input_activations = None              (fp16 acts)
      W4A8  -> input_activations = int8 token-dynamic (saves format=int-quantized ->
               vLLM CompressedTensorsW4A8Int)
  GPTQ weight rounding is driven by the Hessian of the (smoothed) calibration acts; the
  int8-act setting does not change the stored int4 weights' rounding path, so the int4
  weights are produced by the identical pipeline (same-method; bit-equality is verified
  post-hoc by the companion compare step, not assumed here).

Model is the VL instruct `Qwen3_5ForConditionalGeneration` (has a vision tower + GDN
linear-attn layers + an MTP head). We MUST load it with AutoModelForImageTextToText
(NOT AutoModelForCausalLM) and ignore the same branches the prior ckpt ignored:
  lm_head, embed_tokens, *.linear_attn.* (GDN), *.visual.* (vision), *.mtp.* (MTP head).

Load on CPU (NO device_map=auto): GPTQ's sequential pipeline onloads ONE layer/GPU; a
pre-filled GPU OOMs.

Usage:
  python3 requant_awqgptq_g32.py --scheme w4a8  --out /out/models/Qwen3.5-9B-w4a8-g32-awqgptq
  python3 requant_awqgptq_g32.py --scheme w4a16 --out /out/models/Qwen3.5-9B-w4a16-g32-awqgptq
  # or both in one process (builds calib once, shares it):
  python3 requant_awqgptq_g32.py --both --out-w4a16 <dir16> --out-w4a8 <dir8>
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

MODEL_ID = "Qwen/Qwen3.5-9B"  # the INSTRUCT (non -Base) VL hybrid
GROUP_SIZE = 32
SMOOTH = 0.8
NUM_CALIB = 512
MAX_LEN = 2048
SEED = 1234

# Match the branches the prior W4A8 ckpt left in bf16 (verified from its config.json):
# vision tower, all GDN linear_attn projections, MTP head, lm_head, embeddings.
IGNORE = [
    "lm_head",
    "re:.*embed_tokens",
    "re:.*linear_attn.*",
    "re:.*visual.*",
    "re:.*mtp.*",
]


def weight_args() -> QuantizationArgs:
    # int4, group_size=32, symmetric, static group scales — identical for both schemes.
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=GROUP_SIZE,
        strategy=QuantizationStrategy.GROUP,
        observer="memoryless_minmax",
        dynamic=False,
    )


def scheme_for(scheme: str) -> QuantizationScheme:
    w = weight_args()
    if scheme == "w4a8":
        ia = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            symmetric=True,
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
            observer=None,
        )
    elif scheme == "w4a16":
        ia = None
    else:
        raise ValueError(scheme)
    return QuantizationScheme(targets=["Linear"], weights=w, input_activations=ia)


# SmoothQuant mappings for the Qwen3.5 hybrid VL text stack.
# The default Llama mapping (smooth input_layernorm -> [q,k,v]_proj) FAILS here because the
# 24 GatedDeltaNet (linear_attn) layers have an `input_layernorm` but NO q/k/v_proj (their
# linear_attn.* projections are kept bf16 / in the ignore list). So smooth attention ONLY on
# the 8 FULL-ATTENTION layers (indices 3,7,11,15,19,23,27,31 = every full_attention_interval),
# and smooth the MLP (post_attention_layernorm -> gate/up_proj) on ALL layers.
FULL_ATTN_LAYERS = "3|7|11|15|19|23|27|31"
SMOOTHQUANT_MAPPINGS = [
    [  # attention: only full-attn layers
        ["re:.*self_attn.q_proj", "re:.*self_attn.k_proj", "re:.*self_attn.v_proj"],
        rf"re:.*layers\.({FULL_ATTN_LAYERS})\.input_layernorm",
    ],
    [  # MLP: all layers (both GDN and full-attn have a standard SwiGLU MLP)
        ["re:.*mlp.gate_proj", "re:.*mlp.up_proj"],
        "re:.*post_attention_layernorm",
    ],
]


def build_recipe(scheme: str):
    # SmoothQuant (AWQ-style activation smoothing) THEN GPTQ Hessian rounding @ g32.
    return [
        SmoothQuantModifier(smoothing_strength=SMOOTH, mappings=SMOOTHQUANT_MAPPINGS),
        GPTQModifier(
            config_groups={"group_0": scheme_for(scheme)},
            ignore=IGNORE,
            dampening_frac=0.01,
        ),
    ]


def build_calib(tok) -> Dataset:
    """Shared calibration set, fully OFFLINE (HF_HUB_OFFLINE=1).

    Primary: ise-uiuc/Magicoder-Evol-Instruct-110K — a real, diverse instruction
    dataset (instruction/response), formatted through the model's chat template. It
    is fully present in the sandbox HF cache (ultrachat_200k is metadata-only there,
    and streaming re-hits the Hub so it fails offline). Fallback: wikitext-2-raw-v1.
    The SAME set/order/seed is used for BOTH schemes (built once in main()).
    """
    random.seed(SEED)
    samples = []
    src = "magicoder-evol-instruct"
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        idx = list(range(len(ds)))
        random.shuffle(idx)
        for i in idx:
            if len(samples) >= NUM_CALIB:
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
    except Exception as e:
        print(f"[calib] magicoder unavailable ({str(e)[:80]}); falling back to wikitext",
              flush=True)
        samples = []
        src = "wikitext-2-raw-v1"
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        for r in ds:
            if len(samples) >= NUM_CALIB:
                break
            if len(r["text"].strip()) > 200:
                samples.append({"text": r["text"]})
    print(f"[calib] built {len(samples)} samples ({src}, seed={SEED})", flush=True)
    return Dataset.from_list(samples)


def run_one(scheme: str, out: str, calib: Dataset, tok):
    print(f"\n[q] ===== {scheme.upper()} -> {out} =====", flush=True)
    print(f"[q] load {MODEL_ID} on CPU (VL instruct)", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    recipe = build_recipe(scheme)
    print(f"[q] oneshot SmoothQuant({SMOOTH})+GPTQ int4 g{GROUP_SIZE} "
          f"act={'int8-token-dyn' if scheme=='w4a8' else 'fp16'} n={len(calib)}", flush=True)
    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        processor=tok,
        max_seq_length=MAX_LEN,
        num_calibration_samples=len(calib),
    )
    print("[q] save", flush=True)
    model = model.to("cpu")
    torch.cuda.empty_cache()
    model.save_pretrained(out, save_compressed=True, max_shard_size="4GB")
    tok.save_pretrained(out)
    # report what landed
    qc = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    g = qc["config_groups"]["group_0"]
    print(f"[q] DONE {scheme}: format={qc.get('format')} "
          f"w.bits={g['weights']['num_bits']} w.gs={g['weights']['group_size']} "
          f"input_act={(g['input_activations'] or {}).get('num_bits')}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["w4a8", "w4a16"])
    ap.add_argument("--out")
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--out-w4a16")
    ap.add_argument("--out-w4a8")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    calib = build_calib(tok)  # built ONCE, shared across both schemes for fairness

    if args.both:
        assert args.out_w4a16 and args.out_w4a8, "need --out-w4a16 and --out-w4a8"
        run_one("w4a16", args.out_w4a16, calib, tok)
        run_one("w4a8", args.out_w4a8, calib, tok)
    else:
        assert args.scheme and args.out, "need --scheme and --out"
        run_one(args.scheme, args.out, calib, tok)


if __name__ == "__main__":
    main()
