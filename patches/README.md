# patches/ — the fork edit recipe

This repo is a **vendored fork**: the complete modified source lives in `vllm/` (upstream v0.23.0)
and `flashinfer/` (upstream v0.6.12), committed with all our edits baked in. `patches/` is no longer
applied at build time — it is the **recipe** that regenerates those vendored trees from a fresh
upstream checkout (so an upstream bump is reproducible and drift is detectable).

The complete fork is built **from source** by the maintainer running `scripts/build_image_source.sh`
**locally** (vLLM from `vllm/` → sm_80+sm_86 fatbin + the int8-QK FlashInfer overlaid from
`flashinfer/`) and pushing to ghcr. There is **no CI auto-build**: the from-source CUDA build needs a
GPU, and a self-hosted GPU runner on a public repo is a security risk (a malicious PR could run code
on it). The only CI is the github-hosted `patch-drift-check` canary below. Native changes (`.cu/.cuh`)
ship only from source — which is exactly why this is vendored rather than a pure-Python overlay.

## The edits (the recipe)

**vLLM** (`vllm/`):
| edit | what it does |
|---|---|
| `0001-marlin-w4a8-int8-ampere.patch` (+ `regenerate.py`) | Wires int4-weight + int8-activation (W4A8) through **Marlin** on Ampere — vLLM gates Cutlass/Machete W4A8 to Hopper. Edits `compressed_tensors_w4a8_int.py` (`act_type=torch.int8`), `mixed_precision/marlin.py` (int4 in the 8-bit-act assert, pack signed int4→uint4b8, effective `wtype`), `marlin_utils.py` (int4 supported). Pure-Python. Upstream [vllm#38064](https://github.com/vllm-project/vllm/issues/38064)/[#38066](https://github.com/vllm-project/vllm/pull/38066). Applied by `regenerate.py` (anchor-based, fails loudly on skew). |
| `0002-marlin-int8-8row-decode-ampere.patch` | **Native.** int8 (`kS8`) `m_block_size_8` 8-row decode tile in Marlin (upstream gates it to 16-bit acts) via four transposed-`m16n8k32`-layout fixes, all `is_a_8bit`/`m_block_size_8`-gated. Touches `csrc/`. Applied by `git apply -p1 --directory=vllm`. |
| `0003-aot-compile-cache-quant-scheme-key.patch` | **Pure-Python.** Folds the resolved quantization scheme into the **torch.compile AOT-compile cache key** (`compilation/caching.py` `aot_compile_hash_factors`). Without it, two checkpoints of the SAME architecture but DIFFERENT quant schemes (e.g. pack-quantized **W4A16** vs int-quantized **W4A8**) collide on the on-disk AOT-graph hash; a W4A8 model served against a persistent cache holding the W4A16 graph loads it and crashes `KeyError: 'weight_zero_point'` (W4A16 registers `weight_zero_point`; symmetric W4A8 does not). Root cause + reproduction + validation in [`eval/INT8_CUDAGRAPH_ROOTCAUSE.md`](../eval/INT8_CUDAGRAPH_ROOTCAUSE.md). Applied by `git apply -p1` (additive; anchor = `aot_compile_hash_factors`). |
| `flashinfer_int8/int8qk_backend.py` | vLLM V1 CUSTOM attention backend: intercepts hd256 full-attn prefill (fresh + cached-prefix chunks), per-token int8-quant Q/K → int8 FlashInfer prefill, fp16 PV. Dropped into `vllm/vllm/v1/attention/backends/`. Now ships `register_int8qk()`, the `vllm.general_plugins` entry-point (registered by 0004). Everything non-hd256 / decode / unsupported FA-falls-back. **Runtime-couples to the int8-QK FlashInfer overlay below** — stock flashinfer 0.6.12 lacks `torch.int8` in `dtype_map_kv`, so the kernel `KeyError`s the instant it fires; both ship together via `build_image_source.sh`. |
| `0004-int8qk-general-plugin-entrypoint.patch` | **Pure-Python (`pyproject.toml`).** Registers `int8qk = "...int8qk_backend:register_int8qk"` under `[project.entry-points."vllm.general_plugins"]`. vLLM's `load_general_plugins()` runs this in **every process** — engine-core AND each TP/PP worker subprocess (`v1/worker/worker_base.py:init_worker`, before backend selection) — so the int8-QK override reaches all workers and fires under `-tp N` / `-pp N` (the old in-process monkeypatch only worked with `VLLM_ENABLE_V1_MULTIPROCESSING=0`). **Opt-in via `VLLM_INT8QK=1`** (the entry-point is baked into the from-source image; default-on would swap the global FLASH_ATTN backend for every model). The backend's `get_name()` returns `"FLASH_ATTN"` so the `AttentionBackendEnum[get_name()]` lookup in `attention.py` resolves (no launcher monkeypatch needed). Applied by `git apply -p1 --directory=vllm` **after** the `int8qk_backend.py` cp. |

**FlashInfer** (`flashinfer/`) — `flashinfer_int8/apply_to_source.py` (runs i1_apply + i4_apply + i4_compute_qk):
native int8-QK IMMA (`m16n8k32 s8s8s32`) wired into `compute_qk` (mma.cuh wrapper, s32 accum, per-token
q/k dequant + smooth_k, PV fp16) + the int8 dtype path through the JIT codegen + per-token scale plumbing,
incl. per-request q AND k scale offsets (`q_indptr` / `maybe_kv_scale_indptr`) for MULTI-request batched
prefill (paged + ragged). Validated real RTX 3090: cos 0.9999 vs fp16 single- AND multi-request (N≥3,
head_dim 128/256, GQA, causal/non-causal, paged+ragged, qo<kv append); head_dim 64 guarded unsupported
(k64B swizzle); e2e Qwen3.5-9B-W4A8 64k +1.9% / 128k chunked +2.0% TTFT. See `flashinfer_int8/NOTES.md`.

## Experimental — NOT applied to the vendored trees

| dir | what it is |
|---|---|
| `flashinfer_fp16pv/` | **fp16-accumulate PV** for the FlashInfer prefill kernel (pure-fp16 `o_frag`). Validated RTX 3090 sm_86: **+24-26% op-level prefill**, worst-row cos 0.99998, e2e prefill TTFT +1.1/2.1/4.1% @16/32/64k single-card. **NOT wired into the build / NOT in `flashinfer/`** — GeForce-GA10x sm_86-only (zero benefit + less precise on A100/A40/A6000/A10), DTypeProb=half-only, prefill-only, accuracy is fake-quant op-level not closed W4A8 e2e. Default-on needs a runtime GeForce-SKU probe + a gated `USE_FP16_PV_REDUCTION` template flag + an autoregressive accuracy gate (see `flashinfer_fp16pv/NOTES.md` + `docs/RESEARCH-fp16-accum-pv.md`). Far smaller lever than the MTP-verify KV-split fix for the MTP deployment. |

## Re-vendor on an upstream bump

`watch-upstream.yml` opens an issue when upstream releases a newer tag. To re-vendor:

```bash
scripts/revendor.sh <vllm_tag> <flashinfer_tag>   # e.g. v0.23.0 v0.6.12
# clones the fresh tags into vllm/ + flashinfer/, replays the recipe (regenerate.py + 0002 + 0003 +
# apply_to_source.py); any drifted anchor FAILS LOUDLY. Then:
git diff                                           # review
git commit -am "revendor vllm@<tag> + flashinfer@<tag>"
OWNER=<you> scripts/build_image_source.sh          # build from source + push to ghcr (local; no CI auto-build)
```

`patch-drift-check.yml` replays the recipe onto the LATEST upstream tags daily (in temp checkouts,
never touching the committed trees) and opens an issue if an anchor drifted, so refreshes are caught early.
