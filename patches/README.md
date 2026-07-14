# patches/ — the fork edit recipe

This repo is a **vendored fork**: the complete modified source lives in `vllm/` (upstream v0.23.0)
and `flashinfer/` (upstream v0.6.12), committed with all our edits baked in. `patches/` is no longer
applied at build time — it is the **recipe** that regenerates those vendored trees from a fresh
upstream checkout (so an upstream bump is reproducible and drift is detectable).

The complete fork is built **from source** by the maintainer running `scripts/build_image_source.sh`
**locally** (vLLM from `vllm/` → sm_80+sm_86 fatbin + the fp16-PV FlashInfer from
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
| `flashinfer_int8/int8qk_backend.py` | **🗑 REMOVED 2026-06-25** (was: vLLM V1 int8-QK CUSTOM attention backend, per-token int8-quant Q/K → int8 FlashInfer prefill + fp16 PV). int8-QK measured net-negative everywhere → the vendored copy is deleted and `revendor.sh` no longer drops it in. The int8-QK FlashInfer kernel overlay (`flashinfer_int8/apply_to_source.py`) is currently DEAD but still in the vendored `flashinfer/` tree (it shares files with the fp16-PV patch 0007); a clean flinfer revert is a separate task. |
| `0004-int8qk-general-plugin-entrypoint.patch` | **Pure-Python (`pyproject.toml`).** Registers `int8qk = "...int8qk_backend:register_int8qk"` under `[project.entry-points."vllm.general_plugins"]`. vLLM's `load_general_plugins()` runs this in **every process** — engine-core AND each TP/PP worker subprocess (`v1/worker/worker_base.py:init_worker`, before backend selection) — so the int8-QK override reaches all workers and fires under `-tp N` / `-pp N` (the old in-process monkeypatch only worked with `VLLM_ENABLE_V1_MULTIPROCESSING=0`). **Opt-in via `VLLM_INT8QK=1`** (the entry-point is baked into the from-source image; default-on would swap the global FLASH_ATTN backend for every model). The backend's `get_name()` returns `"FLASH_ATTN"` so the `AttentionBackendEnum[get_name()]` lookup in `attention.py` resolves. **🗑 REMOVED 2026-06-25** — int8-QK was measured NET-NEGATIVE in every scenario (fresh 16/32/64k + cached-prefix; see 0008 row); the vendored `int8qk_backend.py`, this entry-point, and the `VLLM_INT8QK`/`VLLM_FLASHAMPERE_INT8QK` envs are all DELETED, and `revendor.sh` no longer applies 0004. The patch file is kept for historical record only. flashampere now defaults to the fp16-PV legs. |
| `0008-flashampere-unified-attn-backend.patch` | **Pure-Python (routing).** `flashampere` — ONE `FlashAmpereImpl(FlashAttentionImpl)` registered into **`Backend.CUSTOM`** (not the FLASH_ATTN masquerade) and auto-selected by `platforms/cuda.py` for Ampere (sm major 8, guarded on `AttentionBackendEnum.CUSTOM.is_overridden()`). `forward()` classifies the phase (CPU-only) and dispatches **hd256 prefill → fp16-PV** (`use_fp16_pv_reduction` from 0007): fp16-served query → `fp16pv` leg; bf16-served query → `bf16cvt` leg; decode / MTP-verify (uniform q=1+K → base FA `fwd_kvcache` fix) / encoder / fp8-KV / non-Ampere → `super().forward()` (bit-faithful). Lets the fp16-PV legs + MTP-verify **compose in one routing target** (vLLM binds one impl per layer-group, so they couldn't stack before). **int8-QK was REMOVED** (and the patch-0004 standalone deleted): a sweep measured it net-negative in EVERY scenario — fresh 16/32/64k (+3.4/4.0/6.0%, gap grows because its per-token dequant is O(L²)) + cached-prefix (+14.4%) — the quant/gather/dequant tax always exceeds the ~1.7% IMMA-QK gain, while fp16-PV is −2.3~2.9% and never regresses. Half-only + GeForce-GA10x capability gating in `capability.py`; opt-in master `VLLM_FLASHAMPERE=1` + per-leg `VLLM_FLASHAMPERE_{PV_FP16,BF16CVT}` (both **default-on**, GeForce-gated so a no-op on pro Ampere) + `_SAGE` (off). New pkg `vllm/v1/attention/backends/flashampere/` (dispatch/capability/impl/kernels/backend) + edits to `cuda.py`/`envs.py`/`pyproject.toml` + CPU unit tests `tests/v1/attention/test_flashampere_dispatch.py`. Validated real 3090: CUSTOM auto-selected, bf16cvt FIRED hd256, coherent, cudagraph-captured. **`bf16cvt` leg** (query dtype is a 3-state `QSrc` enum: HALF→`fp16pv`, BF16→`bf16cvt`, OTHER→sink): fp16-PV is half-only, so a bf16-served model (Qwen3.x default) can't fire `fp16pv`; `bf16cvt` upcasts Q/K/V bf16→fp16 at runtime (lossless — fp16 carries 10 mantissa bits vs bf16's 7) and runs the SAME fp16-PV cubin, delivering the win to bf16 deploys **without int8-QK's per-token quant/gather tax** (a 64k single-prefill A/B measured int8-QK NET-NEGATIVE: +4.6% vs stock, while fp16-PV is −2.9%). Reuses `fp16pv_prefill` verbatim (already up/downcasts) + a NaN/inf/>fp16-max guard on Q/K/V before the cast. `VLLM_FLASHAMPERE_BF16CVT` **default-on**, GeForce-GA10x-gated. MEASURED real 3090: single-card 64k stock-bf16 10.780s → bf16cvt 10.531s (**−2.3%**); **27B-W4A16 TP2** both tp workers FIRED, flat (no-NVLink all-reduce dilution); **35B-A3B-MoE PP2** both pp stages FIRED, −1.1%; all coherent. 3-agent design + 3-agent adversarial review both SHIP. |
| `0009-famp-marlin-config.patch` (+ vendored `flashampere/marlin/`, `build_image_source.sh` **stage 3**) | **Pure-Python config + vendored-from-source kernel.** Widens `compressed_tensors_wNa16.py`'s int8-act (`VLLM_MARLIN_INPUT_DTYPE`) override from `is MarlinLinearKernel` to also include **FampMarlinKernel** — the fork's standalone Marlin GEMM (`torch.ops.famp_marlin.*`, a byte-mirror of stock `_C` Marlin, bit-exact per `test_kernel_equiv`), registered via a `vllm.general_plugins` entry point. The `.so` is **compiled from `flashampere/marlin/csrc`** in `build_image_source.sh` stage 3 for `FAMP_MARLIN_ARCH` (default Ampere `sm_80,sm_86`); `register_fampmarlin()` gates selection to the built arches so non-Ampere GPUs fall back to stock `_C` (bit-identical). Lets the fork **own** the W4A8/W4A16 Marlin path (was 4 marlin patches → now the vendored kernel + this 1 config patch). Import-guarded, so the patch is a no-op without the plugin installed. |

**FlashInfer** (`flashinfer/`) — `flashinfer_int8/apply_to_source.py` (runs i1_apply + i4_apply + i4_compute_qk):
native int8-QK IMMA (`m16n8k32 s8s8s32`) wired into `compute_qk` (mma.cuh wrapper, s32 accum, per-token
q/k dequant + smooth_k, PV fp16) + the int8 dtype path through the JIT codegen + per-token scale plumbing,
incl. per-request q AND k scale offsets (`q_indptr` / `maybe_kv_scale_indptr`) for MULTI-request batched
prefill (paged + ragged). Validated real RTX 3090: cos 0.9999 vs fp16 single- AND multi-request (N≥3,
head_dim 128/256, GQA, causal/non-causal, paged+ragged, qo<kv append); head_dim 64 guarded unsupported
(k64B swizzle); e2e Qwen3.5-9B-W4A8 64k +1.9% / 128k chunked +2.0% TTFT. See `flashinfer_int8/NOTES.md`.

**FlashInfer** (`flashinfer/`) — `0007-fp16-accum-pv-gated-flashinfer.patch`: **gated, half-only fp16-accumulate
PV** for the prefill kernel, productionized from the experimental `flashinfer_fp16pv/` below. Edits
`include/.../prefill.cuh` (`compute_sfm_v` PV-MMA → `f16f16f16` into a uint32[4] `o_acc` behind
`if constexpr (FA_PV16<KTraits>)`, materialize→float `o_frag` epilogue; `FA_PV16 = (FA_USE_FP16_PV!=0) &&
DTypeProb-is-half` via a `std::void_t` detection idiom so bf16-prob kernels auto-fall-back to stock and the
`DTypeProb`-less `kMultiItemScoring` traits compile) + `flashinfer/prefill.py` + `jit/attention/modules.py`
(thread `use_fp16_pv_reduction` → a `_f16pv_{flag}` URI suffix + a `-DFA_USE_FP16_PV=1` nvcc cflag on the
fa2 module, default `#define FA_USE_FP16_PV 0` = OFF). **VALIDATED flinfer 0.6.12, fresh JIT venv**
(clean-JIT vs patched-JIT — NOT AOT-vs-JIT, which is a false-negative): ON **+25.5%** cos 0.999990, OFF
**bit-identical** (cos 1.0, zero-regression default-off); composes with int8-QK (int8-Q → `DTypeProb=half`)
in one cubin. GeForce-GA10x + `VLLM_FLASHAMPERE_PV_FP16=1` gate it on; consumed by 0008's `flashampere`.

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
