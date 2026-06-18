# W4A8 int8-act cudagraph crash — `KeyError: 'weight_zero_point'` — root cause + fix

**Date:** 2026-06-18 · **vLLM:** v0.23.0 (image `ghcr.io/avesed/vllm-ampere-optimized:v0.23.0`)
**Model:** `Qwen3.5-9B-w4a8-g32-awqmse` (dense-hybrid VL; compressed-tensors `int-quantized`,
int4 g32 symmetric weights + int8 dynamic-per-token activations → `CompressedTensorsW4A8Int` →
MarlinLinearKernel).

## TL;DR

The crash is **NOT** in the W4A8 / Marlin / compressed-tensors quant code, and **NOT** in eager
vs cudagraph per se. It is a **torch.compile AOT-compile cache-key COLLISION**: the AOT-compile
cache key does not incorporate the quantization scheme, so a `pack-quantized` **W4A16** checkpoint
and an `int-quantized` **W4A8** checkpoint of the *same architecture* hash to the **same on-disk
AOT-graph path**. When a persistent torch.compile cache already contains the W4A16 graph (which
registers a `weight_zero_point` parameter) and the W4A8 model is then served against that same
cache, the W4A8 module **loads the W4A16 compiled graph** and crashes at runtime when the loaded
graph tries to bind its captured `weight_zero_point` input to the W4A8 module — which, being
symmetric, never registered one → `KeyError: 'weight_zero_point'`.

- **Eager works** because eager never builds/loads an AOT graph.
- **Fresh cache works** (any config, fp16 or bf16, mml 4k or 32k) — proven, see below.
- **Crash requires a polluted/shared persistent cache** populated by a *different-quant* run of the
  *same architecture* — exactly what the eval harness does: `run_eval.sh` runs **w4a16 then w4a8**
  with a shared `/root/.cache/vllm` mount.

## Exact crash site

Deterministically reproduced (decisive test below). Full traceback tail:

```
INFO decorators.py:311 Directly load AOT compilation from path
   .../torch_aot_compile/a66e0d4da453.../rank_0_0/model        <-- loads the W4A16 graph
...
File vllm/compilation/decorators.py:573, in __call__
    output = self.aot_compiled_fn(self, *args, **kwargs)
File torch/_dynamo/aot_compile.py:224, in __call__
    return self.fn(*args, **kwargs)
File vllm/model_executor/models/qwen3_next.py:505, in forward
    def forward(
KeyError: 'weight_zero_point'
```

- The `KeyError` literal `'weight_zero_point'` is raised **inside torch** (`_dynamo/aot_compile.py`)
  binding the loaded graph's captured parameter inputs to the live module — NOT by any vLLM source
  line. (`grep -rEn "\['weight_zero_point'\]"` over the entire installed vLLM tree = **0 hits**;
  the W4A8/Marlin path uses `getattr(layer, ..., None)` everywhere — that would be an
  `AttributeError`, never a `KeyError`.)
- `weight_zero_point` is registered only by the **W4A16** scheme
  (`compressed_tensors_wNa16.py:198 register_parameter("weight_zero_point", qzeros)`).
  The **W4A8** symmetric scheme registers `weight`, `weight_packed`, `weight_scale` (+ the
  marlin-internal `w_zp` empty buffer named `"w_zp"`, set via `setattr`); it **never** registers a
  param named `weight_zero_point`. Hence loading a W4A16 graph against a W4A8 module KeyErrors.
- Frame line numbers match the original report exactly: `qwen3_next.py:505 forward`,
  `cuda_graph.py:254`, `gpu_model_runner._dummy_run`/`profile_run`,
  `determine_available_memory`, `_initialize_kv_caches`.

## The cache-key collision (source-level)

AOT cache dir = `VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/{hash_key}` where
(`vllm/compilation/decorators.py` `__call__`):

```python
factors = aot_compile_hash_factors(self.vllm_config)   # [env_hash, vllm_config.compute_hash()]
factors.append(_model_hash_key(self.forward))          # hash of forward source (same arch -> same)
hash_key = sha256(str(factors)).hexdigest()
```

`aot_compile_hash_factors` (`vllm/compilation/caching.py`) folds in
`vllm_config.compute_hash()` → `ModelConfig.compute_hash()`. **For the compiled language-model
backbone submodule, `ModelConfig.compute_hash()` does NOT capture the resolved quantization
scheme.** The discriminating quant info (`format` = pack-quantized vs int-quantized, per-group
weight/activation `num_bits`/`strategy`/`symmetric`) lives in `hf_config.quantization_config`,
which is not reflected in the hashed factors at the backbone level (the `quantization` field is the
same string `"compressed-tensors"` for both).

**Instrumented proof** (printed `self.vllm_config` inside the real worker `__call__`):

| run | `qfmt` | `model_cfg_hash` | `vllm_cfg_hash` | AOT `hash_key` |
|---|---|---|---|---|
| W4A16 (pack-quantized) | `pack-quantized` | `b9e50f67…` | `79a8e191f1` | `a66e0d4da453…` |
| W4A8  (int-quantized)  | `int-quantized`  | `b9e50f67…` | `79a8e191f1` | `a66e0d4da453…` |

Identical `model_cfg_hash`, `vllm_cfg_hash`, and `hash_key` despite different quant schemes →
collision. (Independent confirmation: three production-style runs — w4a16, and w4a8 in two separate
fresh caches — all wrote/read the **same** dir `a66e0d4da453b413336f784946a415feb2f05107cd82531651654203380c3af7`.)

## Isolation matrix (all reproduced on RTX 3090, GPU1)

| test | cache | result |
|---|---|---|
| W4A8-g32-awqmse, cudagraph, minimal repro (bf16, mml 4k) | **fresh** | ✓ coherent ("…implied premise…") |
| W4A8-g32-awqmse, cudagraph, **exact gsm8k_eval cfg** (fp16, mml 32768, mns 64, gmem 0.92) | **fresh** | ✓ coherent ("Paris … Rome") |
| W4A16-g32-awqmse, cudagraph | **fresh** | ✓ coherent; saves AOT graph `a66e0d4da453…` |
| **W4A8-g32-awqmse, cudagraph, against the cache the W4A16 run wrote** | **shared/polluted** | **✗ `KeyError: 'weight_zero_point'`** (loads `a66e0d4da453…`) |

So the crash is **not** g32-specific, **not** AWQ-vs-GPTQ-save-specific, and **not**
config-specific. The single necessary+sufficient condition is: **a persistent AOT cache already
holding the same-architecture W4A16 graph, then loaded by the W4A8 model.** (The OLD g128 W4A8
"works ✓" only because it was tested without a W4A16 graph polluting its cache.)

## The fix (vLLM source patch — belongs in the vendored `vllm/`)

`vllm/compilation/caching.py` → `aot_compile_hash_factors`: append a stable hash of the resolved
quantization scheme (method name + the full `hf_config.quantization_config` dict) so distinct quant
schemes never share an AOT-compile cache key. Patch: `patches/0003-aot-compile-cache-quant-scheme-key.patch`.
See validation below — with the patch, the W4A8 model gets a **different** `hash_key`, does NOT load
the W4A16 graph, compiles fresh, and runs under cudagraph even with the W4A16 graph in the shared
cache.

### Operational workaround (no rebuild)

Until the fork image carries the patch, prevent the collision at deploy time, any one of:
- give each model its own cache: `-e VLLM_CACHE_ROOT=/root/.cache/vllm/<model-tag>` (or a distinct
  `-v …:/root/.cache/vllm` per model), or
- `-e VLLM_DISABLE_COMPILE_CACHE=1` (recompiles each start; correct but slower), or
- don't share one persistent cache across the w4a16 **and** w4a8 checkpoints in the eval harness
  (`run_eval.sh` runs both with one mounted cache — split the cache per tag).
