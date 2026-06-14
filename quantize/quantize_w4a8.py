"""Quantize an HF model to W4A8 (int4 group weights + int8 dynamic-per-token activations)
for the Ampere Marlin path in this repo.

KEY GOTCHA — use the `scheme="W4A8"` shortcut, NOT a hand-written `config_groups`:
  * `scheme="W4A8"`  -> saves `format: int-quantized`  -> vLLM routes to CompressedTensorsW4A8Int
  * manual groups    -> saves `format: pack-quantized` -> vLLM routes to weight-only WNA16 (no int8 act!)

Also: load the model on CPU (NO `device_map="auto"`). llm-compressor's GPTQ sequential
pipeline onloads ONE layer at a time to the GPU (~3 GB); pre-filling the GPU with
`device_map="auto"` makes that pipeline OOM.

Usage:
    python quantize_w4a8.py <hf_model> <out_dir> [num_calib=256] [max_len=2048]

For a plain text LLM, swap AutoModelForImageTextToText -> AutoModelForCausalLM and trim the
`ignore` list (drop the visual / linear_attn / mtp entries).
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL = sys.argv[1]
OUT = sys.argv[2]
NUM = int(sys.argv[3]) if len(sys.argv) > 3 else 256
MAXLEN = int(sys.argv[4]) if len(sys.argv) > 4 else 2048

# scheme="W4A8" => int4 group weights + int8 dynamic per-token act, saved as int-quantized.
recipe = GPTQModifier(
    scheme="W4A8",
    targets=["Linear"],
    # keep the embedding / output / non-FFN branches in full precision.
    # for hybrid/VL models add: "re:.*linear_attn.*", "re:.*visual.*", "re:.*mtp.*"
    ignore=["lm_head", "re:.*embed_tokens"],
    dampening_frac=0.01,
)

print(f"[q] load {MODEL} to CPU; GPTQ onloads layers to GPU (num={NUM} maxlen={MAXLEN})", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(MODEL)

print("[q] build calibration (ultrachat_200k)", flush=True)
stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
samples = []
for ex in stream:
    if len(samples) >= NUM:
        break
    try:
        samples.append({"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
    except Exception:
        continue
calib = Dataset.from_list(samples)

print(f"[q] oneshot start (n={len(calib)})", flush=True)
oneshot(model=model, dataset=calib, recipe=recipe, processor=tok,
        max_seq_length=MAXLEN, num_calibration_samples=NUM)

print("[q] save", flush=True)
model = model.to("cpu")
torch.cuda.empty_cache()
model.save_pretrained(OUT, save_compressed=True, max_shard_size="4GB")
tok.save_pretrained(OUT)
print(f"[q] DONE -> {OUT}", flush=True)
