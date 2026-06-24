# flashinfer_fp16pv — fp16-accumulate PV for the FlashInfer prefill kernel (EXPERIMENTAL)

**Status: validated kernel optimization, NOT wired into the default build. Opt-in / research only.**
Do NOT apply to the vendored `flashinfer/` tree or default-on — it would regress pro Ampere and ship
an accuracy-on-W4A8-e2e-unproven path. This dir is the recipe + validation, sibling to
`patches/flashinfer_int8/`. Full rationale: `docs/RESEARCH-fp16-accum-pv.md`.

## What it is

The FlashInfer prefill PV step (`compute_sfm_v`) accumulates `o_frag` in **fp32** (8 regs/fragment).
On GA10x consumer Ampere (sm_86: RTX 3090/3080/3070), f16-input/**f16-accum** HMMA runs at **2×** the
f16/f32-accum rate, and at hd256 the fp32 `o_frag` (128 regs) drives the kernel into the 255-reg cap
WITH stack spill. `apply_fp16pv.py` makes the PV loop accumulate in a pure-**fp16** `o_acc` (uint32[4]
half-packed, 4 regs, via the `f16f16f16` HMMA + half2 rescale), then materializes back to the kept
`float o_frag[8]` after the kv loop (so `threadblock_sync`/`transform_output`/`write_o_reg_gmem` are
untouched). **The spill relief — not the 2× MMA — is the dominant effect** (it speeds the whole
attention kernel, not just the PV slice).

## Measured (sandbox RTX 3090 sm_86)

| metric | result |
|---|---|
| register (hd256 f16) | spilling kernels **9→0** (split probe) / 9→4 (this materialize variant); maxSTACK 96→0/80 |
| op-level prefill | **+25.6% / +25.2% / +23.7%** @ 2k / 4k / 8k; worst-row cos 0.999995 / 0.999990 / 0.999979 |
| accuracy (fake-quant, `validate_accuracy.py`) | pure-fp16 worst-row cos **0.999928** (≈ shipped int8-QK 0.99992); two-level 0.999999 |
| e2e (`vllm bench serve`, 9B-w4a16 single-card, FLASHINFER+fp16) | prefill TTFT **+1.1% @16k / +2.1% @32k / +4.1% @64k** (grows with ctx) |

Why op +25% → e2e ~1-4%: the deployment is a **hybrid** (only 8/32 layers are full-attn; 24 are
GatedDeltaNet + MLP), so attention is a minority of prefill — its fraction grows O(L²) with context, so
the win grows with context. On tp2 it's ~0 (TP all-reduce ~67% of prefill). Decode = 0 (prefill-only).

## Coverage tested 2026-06-24 (clean vs patched, RTX 3090 sm_86)

`validate_coverage.py` (kernel cos + speedup) and `validate_needle_e2e.py` (real-model retrieval):

| scenario | result |
|---|---|
| **bf16 hd256** | worst-cos **0.20 = GARBAGE** — CONFIRMS the half-only gating is mandatory; the unconditional apply BREAKS bf16 (f16f16f16 reads bf16 bits as fp16). Do NOT serve bf16 with this patch. |
| **f16 hd128** | cos 0.999994, **+31.6%** (correct; even faster than hd256) |
| **f16 hd256 PAGED batch prefill** (deployment path) | cos 0.999995, **+20.2%** (BatchPrefillWithPagedKVCache correct) |
| **e2e needle 8k / 32k** (9B-w4a16, FLASHINFER+fp16, no-think temp=0) | patched output **byte-identical to clean**, correct retrieval both lengths — no argmax flips |

Still UNTESTED: int8-Q/W4A8 perf (correct by construction — PV code is shared, DTypeProb=half identical;
int8-QK only touches QK), full GSM8K/MMLU eval, tp2/pp2, cudagraph-captured prefill (structurally safe),
real attention-map capture, 35B-MoE, soak.

## Hard limits (why it is NOT default-on)

- **GeForce-GA10x sm_86 ONLY.** A100 (sm_80) and **pro sm_86 (A40 / A6000 / A10)** run f32-accum at FULL
  rate → ZERO benefit AND fp16-accum is LESS precise there. A `__CUDA_ARCH__>=860` gate is **WRONG**
  (enables the slower, less-precise path on the pro line). A **runtime GeForce-SKU / rate probe is
  REQUIRED and UNBUILT.**
- **DTypeProb=half only** (W4A8/int8-Q or fp16-served). bf16-compute (the default deploy dtype) needs a
  P→fp16 + V bf16→fp16 cast (SageAttention-style), not applied here — `f16f16f16` is half-only.
- **The default deployment serve picks FA2 (FLASH_ATTN), not FlashInfer** (`cuda.py` backend list), so
  this only fires when `attention_backend=FLASHINFER` + fp16/int8-Q is in use — i.e. via the int8-QK
  path or a forced FlashInfer backend, single-card long-ctx.
- **Accuracy is op-level fake-quant cos only**, NOT a closed autoregressive W4A8 e2e gate
  (needle / long-CoT / Chinese / GSM8K). Required before default-on.

## To productionize for default-on (remaining work)

1. `USE_FP16_PV_REDUCTION` template flag + JIT URI key (gated OFF) instead of the unconditional edit
   here, so two cubins don't collide and it's opt-in (mirror the QK-side `USE_FP16_QK_REDUCTION`).
2. Runtime GeForce-GA10x SKU / measured-rate startup probe (NOT an arch macro).
3. Autoregressive accuracy gate on real W4A8 (needle 32/64/128k + GSM8K + Chinese-CoT, temp 0.6,
   VLLM_BATCH_INVARIANT) — current evidence is fake-quant op-level only.
4. Decide value vs the e2e envelope (~1-4% long-ctx single-card; ~0 on the tp2 deployment). The MTP-verify
   KV-split fix (`131f9ae` on dev) is a far larger lever for this MTP deployment.

## Files

- `apply_fp16pv.py` — the kernel patcher (installed-package layout; `--restore` to revert). Strict anchors.
- `validate_accuracy.py` — the fake-quant cos gate (two-level vs pure-fp16 vs fp32).
- `bench.py` — single-prefill perf + cos bench (run clean → patched → cmp).
- `validate_coverage.py` — kernel cos+speedup battery: bf16 (expect garbage), hd128, hd256 paged batch.
- `validate_needle_e2e.py` — real-model needle-in-haystack retrieval (clean vs patched, no-think temp=0).
