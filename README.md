# vllm-3090-optimized

Patches and notes for running **vLLM optimally on NVIDIA Ampere consumer GPUs (RTX 3090, sm_86)** — squeezing the quantization paths that vLLM ships gated-to-Hopper or mis-configured for Ampere.

> Status: working set of hot-patches (pure-Python, no recompile) + reproducible benchmarks. Validated on 2× RTX 3090 (no NVLink), vLLM v0.21-dev, a 27B Qwen3.5 hybrid model.

---

## 1. W4A8-INT8 on Ampere (the flagship fix)

**TL;DR:** W4A8 (int4 weights + int8 dynamic-per-token activations) is the **best serving quant on the 3090** — but vLLM silently blocks it on Ampere. A 3-file, ~10-line Python patch enables it. No CUDA recompile.

### Why W4A8 is the sweet spot

Decode is **memory-bandwidth-bound** (read all weights per token) → int4 weights win. Prefill is **compute-bound** → int8 tensor cores win. W4A8 gets both. Measured on 2× RTX 3090, vLLM tp2, CUDA graphs on:

| quant (kernel) | single-stream decode | batch-16 decode | prefill |
|---|---|---|---|
| W4A16 (Marlin WNA16) | 50.6 tok/s | 387 tok/s | 1045 tok/s |
| W8A8 (CUTLASS int8) | 38.8 tok/s | 343 tok/s | **1250 tok/s** |
| **W4A8 (Marlin, patched)** | **50.4 tok/s** | **393 tok/s** | **1229 tok/s** |

W4A8 = W4A16's decode **and** W8A8's prefill. Bonus: W4A8's small int4 weights leave room for CUDA-graph capture, whereas **W8A8 OOMs graph capture at serving batch on 24 GB** (forced to eager → slower).

### The problem on Ampere

- vLLM's dedicated W4A8 kernels — `CutlassW4A8LinearKernel`, `MacheteLinearKernel` — **require compute capability 9.0 (Hopper)**. Machete uses `wgmma`, an sm_90-only instruction; it cannot be compiled for sm_86.
- The **Marlin** kernel *can* do W4A8-int8 on Ampere (its CUDA side already accepts `is_a_8bit`), but the Python layer never routes to it: `create_weights` passes `act_type=params_dtype` (bf16) instead of `torch.int8`, so Marlin never enters the int8 branch and the layer falls through to weight-only WNA16 → `Failed to find a kernel that can implement the WNA16 linear layer`.

This is upstream [vllm#38064](https://github.com/vllm-project/vllm/issues/38064); the fix is [vllm#38066](https://github.com/vllm-project/vllm/pull/38066).

### Apply the patch

```bash
python patches/w4a8_int_marlin_ampere.py            # auto-detects the installed vllm
# or: python patches/w4a8_int_marlin_ampere.py /path/to/site-packages/vllm
```

It edits 3 files (backing each up to `.bak`):
1. `compressed_tensors/schemes/compressed_tensors_w4a8_int.py` — `act_type=torch.int8`
2. `kernels/linear/mixed_precision/marlin.py` — allow `int4` in the 8-bit-act assert; pack signed int4 → uint4b8 layout; pass effective `wtype=uint4b8`
3. `quantization/utils/marlin_utils.py` — add `scalar_types.int4` to supported types

Then on the 3090 you'll see `Using MarlinLinearKernel for CompressedTensorsW4A8Int`, CUDA graphs on, coherent output.

### Quantize a model to W4A8 (for this path)

Use llm-compressor's **`scheme="W4A8"` shortcut** — NOT a hand-written `config_groups` (a manual config saves `format: pack-quantized`, which vLLM routes to weight-only WNA16; `scheme="W4A8"` saves `format: int-quantized`, which routes to `W4A8Int`):

```python
from llmcompressor.modifiers.quantization import GPTQModifier
recipe = GPTQModifier(scheme="W4A8", targets="Linear",
                      ignore=["lm_head"], dampening_frac=0.01)
```

(For hybrid/VL models, extend `ignore` with the non-quantized branches, e.g. `re:.*linear_attn.*`, `re:.*visual.*`, `re:.*mtp.*`, `re:.*embed_tokens`.)

A runnable end-to-end helper is in `quantize/quantize_w4a8.py`:

```bash
python quantize/quantize_w4a8.py <hf_model> <out_dir> [num_calib=256] [max_len=2048]
```

> Don't pass `device_map="auto"` to the loader — it pre-fills the GPU and the GPTQ sequential
> pipeline then OOMs. Load on CPU; the pipeline onloads one layer at a time (~3 GB).

---

## 2. Notes / gotchas collected along the way

- **W8A8 int8** works out of the box on Ampere (`CutlassInt8ScaledMMLinearKernel`) but: (a) **slower decode than int4** (int8 weights = 2× the bytes; decode is bandwidth-bound), (b) **OOMs CUDA-graph capture** at serving batch on 24 GB → eager only. Use it only for prefill-bound / batch-throughput workloads, or on bigger cards.
- **W8A8 actorder clash**: GPTQ defaults `actorder: static`, which vLLM rejects with a `channel` weight strategy (`Must use group or tensor_group ... to apply activation ordering`). Patch the saved `config.json` `weights.actorder` to `null` (harmless — no `g_idx` is saved for static).
- **DEBUG logging tanks throughput**: `VLLM_LOGGING_LEVEL=DEBUG` logs tensor `repr`s per op → forces GPU→CPU sync every rms_norm → ~200× decode slowdown. Use INFO; the kernel-selection line is already INFO.
- llm-compressor GPTQ: **don't** pass `device_map="auto"` (pre-fills the GPU → the sequential pipeline OOMs). Load on CPU; the pipeline onloads one layer at a time to GPU (~3 GB).
- **W4A8 is int8-sized on disk, not int4-sized.** The `int-quantized` format the W4A8Int kernel needs does **not** bit-pack — each 4-bit weight occupies a full `int8` byte, so the model file is ~the same size as W8A8 (and a touch bigger, from group-128 vs per-channel scales). VRAM *is* int4 — the patch repacks `int8 → packed int4` on load (proof: W4A8 decodes like W4A16, not W8A8). If you want a small **download**, ship W4A16 (`pack-quantized`, truly bit-packed). See [`benchmarks/results.md`](benchmarks/results.md).

---

## Repo layout

```
patches/   w4a8_int_marlin_ampere.py   # the flagship hot-patch (auto-detects vllm; --revert)
quantize/  quantize_w4a8.py            # llm-compressor recipe -> int-quantized W4A8 model
benchmarks/ vllm_verify.py             # single-stream + batch-16 decode + prefill, prints kernel
            vllm_batch_sweep.py        # batch 16/64/256 crossover sweep
            results.md                 # full numbers + interpretation
```

## Benchmarks

`benchmarks/vllm_verify.py` — single-stream + batch-16 decode + prefill throughput, prints the chosen kernel. `benchmarks/vllm_batch_sweep.py` — batch-size sweep (16/64/256) to find the memory-bound→compute-bound crossover. Full numbers and interpretation in [`benchmarks/results.md`](benchmarks/results.md).

## Roadmap
- [ ] Package the W4A8 patch as a proper `pip install` shim / monkeypatch on import.
- [ ] Ampere-tuned CUDA-graph capture sizes to fit W8A8 at serving batch.
- [ ] Hadamard/SpinQuant-style rotations for W4A4 on Ampere (research).

## Credits
W4A8 Marlin enablement follows [vllm#38066](https://github.com/vllm-project/vllm/pull/38066) (upstream). QQQ (an alternative Ampere W4A8 path) is [vllm#5218](https://github.com/vllm-project/vllm/pull/5218) / [HandH1998/QQQ](https://github.com/HandH1998/QQQ).
