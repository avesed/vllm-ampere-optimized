# quantize/ — making a W4A8 checkpoint for this fork

W4A8 = **int4 group weights + int8 dynamic-per-token activations**. The weights are quantized once;
the int8 activations are computed at runtime (no activation calibration). vLLM routes a W4A8
checkpoint (`format: int-quantized`) to `CompressedTensorsW4A8Int` → this fork's Ampere Marlin path.

A **W4A16** checkpoint is the *same* int4 weights with `input_activations: null` (fp16 activations).
On Ampere, W4A8 ≈ W4A16 on quality but is **+36–49 % faster** on prefill / batched serving (see the
top-level README); single-stream decode is at parity (both are int4-weight bandwidth-bound).

## Two ways to quantize

### 1. Generic, quick — `quantize_w4a8.py` (GPTQ)
```bash
python quantize/quantize_w4a8.py <hf_model> <out_dir> [num_calib=256] [max_len=2048]
```
GPTQ with the `scheme="W4A8"` shortcut (int4 g128 + int8 dynamic act, `format: int-quantized`). Works
on any architecture with no per-model tuning. This is the proven baseline (the original 27B W4A8 was
made this way). The `ignore` list excludes ALL quant-sensitive branches by default — `lm_head`,
`embed_tokens`, `re:.*mtp.*` (MTP head), `re:.*linear_attn.*` (GatedDeltaNet), `re:.*visual.*` (vision);
each regex is a no-op on models that lack the module, so it is safe everywhere (MoE also add
`re:.*mlp[.]gate$`). **Keeping `re:.*mtp.*` ignored is mandatory for spec-decode**: it stops the MTP
head being dropped, and — because llm-compressor writes the recipe `ignore` into the output
`config.json` `quantization_config.ignore` — it makes vLLM load the head as bf16 (else the
`qwen3_5_mtp` loader runs it through the quantized path → 0% acceptance, silently).

### 2. Recommended for quality — AWQ + mse + g32  (`requant_v2_awq_mse_g32.py`)
```bash
python quantize/requant_v2_awq_mse_g32.py <out_dir> {w4a16|w4a8}
```
The user's validated recipe (ref `Avesed/Qwopus3.6-35B-A3B-v1-int4-mixed`): **`AWQModifier`** (scale-
search activation smoothing) **+ `QuantizationModifier`** with **`observer=mse`**, **group_size=32**,
symmetric int4 — expressed as a serialized YAML recipe (`recipe_qwen35_9b_awq_mse_g32.yaml`). On
Qwen3.5-9B this beats plain GPTQ and lands W4A8 at GSM8K **85.6 %** vs W4A16 **81.6 %** (N=250, thinking)
— i.e. int8 activations are ~free on quality. Same recipe produces both W4A16 and W4A8 (only
`input_activations` differs), so the pair is a clean apples-to-apples comparison.

**`AWQModifier` is arch-specific** — the `mappings` (`smooth_layer → balance_layers`) and `ignore`
must match the model. The shipped recipe is wired for the dense Qwen3.5-9B hybrid (32 MLP layers +
8 full-attn layers; the 24 GatedDeltaNet `linear_attn` layers and the vision/MTP heads are ignored).
For another model, adapt the `mappings`/`ignore` in the script's YAML (or use `quantize_w4a8.py`).

### Gotchas (both)
- **Load on CPU** (no `device_map="auto"`) — the AWQ/GPTQ sequential pipeline onloads one layer to the
  GPU at a time; pre-filling the GPU OOMs it.
- **`scheme="W4A8"` shortcut, not a hand-written weight-only `config_groups`** — the latter saves
  `format: pack-quantized` → vLLM loads it weight-only (no int8 act). `requant_v2` sets `input_activations`
  to int8-dynamic explicitly, which *does* save `int-quantized` — verify `format: int-quantized` in the
  output `config.json`.
- **llmcompressor ≥ 0.12 trap:** `from llmcompressor.modifiers.awq import AWQModifier` is a deprecated
  factory returning a *list*, not a Modifier — pass AWQ via the **YAML recipe** (as `requant_v2` does),
  where the name resolves to the real `transform.awq.AWQModifier`.
- Quantizing in the fork image needs `pip install llmcompressor datasets` (not in the runtime image).
- W4A8 + cudagraph relies on patch **0003** (AOT cache-key) — otherwise a W4A16→W4A8 run sharing a
  compile cache crashes `KeyError: weight_zero_point`. The fork carries 0003;
  `scripts/int8_cudagraph_regression.sh` asserts it.
