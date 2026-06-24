# RESEARCH: fp16-accumulate two-level PV for FlashInfer on Ampere

**Status:** CONDITIONAL — do the zero-CUDA falsification first
**Date:** 2026-06-23
**Scope:** General Ampere line (sm_80 + sm_86), shippable fork patch. 2×3090 = test rig only.
**Method:** 10-agent workflow (6 investigators + 3 adversarial verdicts + synthesis).

---

## TL;DR / Verdict box

| | |
|---|---|
| **Verdict** | **MEASURED GO on the kernel** (2026-06-24, supersedes the CONDITIONAL below). Real correct pure-fp16 PV kernel built + benched: **+24-26% op-level speedup, worst-row cos 0.99998-0.99999** (correct), robust 2k/4k/8k hd256 f16 prefill. The register-spill relief — not the PV-matmul Amdahl slice — is the dominant effect (it accelerates the whole attention kernel). Remaining before ship: confirm on W4A8/i8 path, op→e2e TTFT, autoregressive e2e accuracy. |
| **What it is** | Replace the unconditionally-fp32 PV accumulator with a two-level scheme: accumulate each `CTA_TILE_KV` tile's P@V in a transient **fp16** block-partial via the existing `f16f16f16` HMMA (`mma.cuh:629`), then **promote** into the kept-fp32 `o_frag` master at each online-softmax rescale boundary. |
| **Physics** | REAL but **GeForce-GA10x-segment-gated**, a strict subset of sm_86. 3090/3080/3070 = 2.0×; A40/A6000/A10 (pro sm_86) = **1.0×, zero benefit**; A100 (sm_80) = 1.0×; Ada/4090 = 1.0× (nerf moved to FP8). |
| **Applicability** | **All GeForce-GA10x models, bf16 included** (CORRECTED 2026-06-23 — the earlier "W4A8-only" was wrong). The `f16f16f16` primitive is half-only, but you cast the PV operands to fp16 at the matmul boundary exactly as SageAttention does: P (softmax probs ∈[0,1]) → half is free + *more* precise than bf16; V bf16→fp16 is lossless except `|V|>65504` overflow. W4A8 already has `DTypeProb=half`; bf16 adds one V cast + an outlier-overflow gate item. |
| **Expected gain** | MEASURED **+24-26% op-level** on the attention prefill kernel (2k/4k/8k hd256 f16) — the §3 Amdahl model (+1.4% e2e) UNDER-counted because it scoped only the PV-matmul slice and treated the register-spill relief as a separate "unmeasured upside"; that relief is in fact the dominant effect. e2e TTFT win = +25% × attention-fraction-of-prefill (less than op-level, but meaningful for long-ctx). Decode = ~0. |
| **Gating risk** | Accuracy on hd256: synthetic fake-quant GREEN (§4, 2026-06-24) — two-level cos 0.999999, **pure-fp16 cos 0.999928 ≈ shipped int8-QK**. Still UNPROVEN on REAL Qwen3.5/3.6 attention maps + autoregressive e2e (the argmax-flip test). |
| **First step** | Two cheap independent gates (either NO-GO kills it): **(A)** `cuobjdump -res-usage` register probe on a throwaway pure-fp16-`o_frag` build (~1h, no model) — if it doesn't relieve spill, no >3% path exists; **(B)** zero-CUDA fake-quant worst-row-cos sweep on real hd256 maps (<1 day). Both before a single e2e kernel line. |
| **Ship as** | Patch 0007, default-OFF behind `VLLM_PV_FP16ACC=1` + a **runtime GeForce-GA10x guard** (NOT `__CUDA_ARCH__ >= 860`). |

This corrects the prior memory note's "SageAttn patch0003 flagship" framing: the shipped int8 path is the FlashInfer int8-QK patch 0004; this fp16-accum lever is a separate, deferred, conditional patch 0007.

---

## 1. Physics — verified, but per-SKU not per-die

The load-bearing claim is "on sm_86 consumer Ampere, fp16-input HMMA with fp16 accumulate runs at 2× the rate of fp16-input with fp32 accumulate." Verified against the NVIDIA GA102 GPU Architecture whitepaper. **One correction the fork MUST encode:** the nerf is GeForce product-segmentation, **not** an sm_86 architectural property.

| Card | Die | SM | FP16-accum | FP32-accum | Ratio | Benefit |
|---|---|---|---|---|---|---|
| RTX 3090 / 3090Ti | GA102 | sm_86 | 142 | 71 | **2.0×** | **YES** (test rig) |
| RTX 3080 / 3080Ti | GA102 | sm_86 | 119 | 59.5 | **2.0×** | **YES** |
| RTX 3070 / 3070Ti | GA104 | sm_86 | 81.3 | 40.6 | **2.0×** | **YES** |
| RTX 3060 / 3050 | GA106/107 | sm_86 | half-rate FP32-accum | — | **2.0×** | **YES** |
| **A40** | GA102 (pro) | sm_86 | 149.7 | 149.7 | **1.0×** | **NO** |
| **RTX A6000** | GA102 (pro) | sm_86 | 154.8 | 154.8 | **1.0×** | **NO** |
| A10 / A16 / A2 | pro | sm_86 | full-rate | = | **1.0×** | **NO** |
| A100 | GA100 | sm_80 | 312 | 312 | 1.0× | NO (624 fig = 2:4 sparsity) |
| RTX 4090 | AD102 | sm_89 | ~330 | ~330 | 1.0× | NO (Ada restored it) |
| Jetson AGX Orin | GA10B | sm_87 | up to 85 | not published | ? | PLAUSIBLE, UNCONFIRMED |

**Footgun:** a `__CUDA_ARCH__ >= 860` compile gate silently enables the slower, less-precise fp16-accum path on A40/A6000/A10 — all sm_86 — with **zero speedup and a real accuracy cost**. The gate MUST be a **runtime GeForce-GA10x SKU/device-name check** or a one-time measured-rate micro-probe at startup. The 2:1 ratio is a per-cycle datapath property → clock-invariant (3090 thermal throttle at ~104 °C hotspot scales both modes equally, ratio holds).

The kernel premise holds: `prefill.cuh:1473-1477` — **both** `DTypeQKAccum` branches today feed `mma_sync_m16n16k16_row_col_f16f16f32<DTypeProb>` into the float `o_frag`. The PV accumulator is unconditionally fp32 today, so today's PV MMA runs at the 71-TFLOPS half rate on a 3090.

**Datapath note (avoid a common conflation):** the "fp32" in "fp16×fp16 → **fp32 accumulate**" is the *Tensor-Core MMA accumulator* (the C fragment), which on A100 runs at the full **312 TFLOPS** FP16-TC rate (== fp16-accum, no nerf). It is NOT the *CUDA-core* scalar/vector FP32 path — A100's CUDA-core FP32 is only **19.5 TFLOPS** (TF32-TC = 156). Those 19.5 TFLOPS feed the softmax `exp2`/SFU, the `o_scale` rescale multiplies, and other elementwise work — a genuinely slow path, but one this PV-accum lever does **not** touch and cannot speed up. So A100's TC fp16:fp32-accum ratio is still 1.0× → zero benefit; the 19.5 figure is about a different engine and does not reopen the A100 case.

---

## 2. Kernel design — two-level, keep the fp32 master

The naive "make `o_frag` half" would touch all six fp32 sites (init 614, rescale 1296 / 1327-1330 / 1370-1373, PV 1473-1477, split-K reduce 1554, finalize, write) and break online-softmax precision + the split-K reduction. The **correct, narrow** scope keeps `o_frag` fp32 everywhere and adds a transient fp16 partial inside `compute_sfm_v`:

```c++
// compute_sfm_v (1390) — the ONLY surgery site
uint32_t o_partial[NUM_MMA_Q][NUM_MMA_D_VO][4];   // 4 packed-half regs, vs 8 fp32
for (mma_kv ...) {
  mode = (mma_kv == 0) ? kInit : kInplaceUpdate;
  if constexpr (PV_FP16_ACCUM)
    mma::mma_sync_m16n16k16_row_col_f16f16f16<mode>(o_partial[mma_q][mma_d], s_frag_f16, b_frag);  // mma.cuh:629
  else
    mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeProb>(o_frag[mma_q][mma_d], ..., b_frag);        // current 1473-1477
}
// PROMOTE (after the mma_kv loop, ~1488-1492), before return:
if constexpr (PV_FP16_ACCUM)
  for each frag: o_frag[mma_q][mma_d][k] += __half22float2(o_partial...);  // f16 C[0..1]->fp32 C[0..3], C[2..3]->C[4..7]
```

**Correctness invariant:** `o_scale` (the online-softmax rescale at `update_mdo_states` 1327-1330) only ever multiplies the **fp32 master**. The fp16 partial is born fresh (`kInit`) each tile and promoted into the already-rescaled master at tile end, so it never needs rescaling, and a single tile's `sum(P_tile) <= 1` keeps it in fp16 range structurally. `update_mdo_states`, `threadblock_sync_mdo_states` (1554, fp32 split-K), `transform_output`, `write_o_reg_gmem` — **zero change**.

**Register layout (verified in this repo):** `mma.cuh:357` f32 wrapper writes `float* C` = C[0..7] (8 regs). `mma.cuh:629` f16 primitive writes `uint32_t* C` = C[0..3] (4 packed-half regs = HALF). The `DTypeQKAccum` precedent at **`prefill.cuh:1090-1103`** already dispatches this exact f16f16f32-vs-f16f16f16 pair on the QK side — PV is the symmetric output twin.

**Dispatch (mirror `USE_FP16_QK_REDUCTION` byte-for-byte):**
- KTraits: `using DTypeOAccum = std::conditional_t<USE_FP16_PV_REDUCTION && is_same_v<DTypeProb, half>, half, float>;`
- Python `use_fp16_pv_reduction: bool` in `flashinfer/prefill.py` → append `_f16pv_{flag}` to the JIT URI (`modules.py` ~339, exactly as `f16qk_{flag}`; **MANDATORY** cache-key component — two different cubins must not collide) → `constexpr bool USE_FP16_PV_REDUCTION = {{...}};` in `single_prefill_customize_config.jinja:29` / `batch_prefill_customize_config.jinja:28` → DISPATCH_context.

**bf16 — applicable via operand cast (CORRECTED).** The f16f16f16 primitive (`mma.cuh:629`) is a non-template, half-only PTX emitter (no bf16 branch, unlike the templated f32 wrapper at 357). The earlier conclusion "therefore bf16 models can't use the lever" was WRONG — it conflated "the current code's `DTypeProb` follows `DTypeQ`" with a physical impossibility. The PV operands are cast at the matmul boundary regardless: P (softmax probs ∈[0,1]) is already `vec_cast` to `DTypeProb` at `prefill.cuh:1402` — forcing that to `half` is free and *more* precise than bf16 (11 vs 8 mantissa bits); V is loaded from bf16 smem and cast bf16→fp16, which is lossless except `|V|>65504` overflow→inf. This is exactly what SageAttention does for bf16 models. So the dispatch must NOT gate on `is_same_v<DTypeProb,half>`; instead, when `USE_FP16_PV_REDUCTION` is on, cast both PV operands to `half` independent of `DTypeProb`. **Shippable surface = all GeForce-GA10x models** (W4A8 already has `DTypeProb=half`; bf16 adds one V cast + an outlier-overflow gate item, falsified in §4's V-clamp self-check).

### 2a. Register pressure — the UNKNOWN that decides whether gain clears the bar

This is the load-bearing risk, so it gets its own subsection. (Corrected twice — the original "delta ~0 / smaller accumulator relieves spill" was wrong for the two-level design.)

**The ceiling is register-bound, proven empirically.** Kernel is already at the 255-reg/thread cap WITH stack spill (`NOTES.md:186` cuobjdump STACK:48-416, hd256). `NOTES.md`'s two probes settle the cause: `noload` (smem reads → arith, 1.094/1.118× @16k/64k) vs `scaleconst` (real loads, scales forced const = 0 scale regs, 1.083/1.100×) — real loads with the scale registers removed reach nearly the noload ceiling, so the ~1.09-1.12× wall is set by **register pressure / occupancy, NOT bandwidth or MMA-issue rate**. Removing live regs (scales → epilogue) recovered the gap; adding them (kd-outer's 64-int32 accumulator) was a wash. So the fp16-PV lever lives or dies on its register delta, not its 2×-MMA.

**Two spill sources, scaling with different tile dims** (this matters because "split more tiles" only touches one):

| Fragment | Size | Scales with | Shrunk by |
|---|---|---|---|
| `o_frag` output accumulator (`prefill.cuh:1871`) | `NUM_MMA_Q × NUM_MMA_D_VO × 8` fp32 | query rows × **head dim (fixed)** | smaller Q-tile / more `NUM_WARPS_Q` only |
| `s_frag` QK logits | `NUM_MMA_Q × NUM_MMA_KV × 8` | query rows × **KV-tile** | smaller `CTA_TILE_KV` |
| scale arrays (16 k-scale) | `~NUM_MMA_KV` | **KV-tile** | smaller `CTA_TILE_KV` (or scales→epilogue, already done) |

At hd256, `NUM_MMA_D_VO = HEAD_DIM_VO/16 = 16` is **fixed by the model**, so `o_frag` has a hard floor of **128 regs** (NUM_MMA_Q=1; =256 at NUM_MMA_Q=2, over cap by itself). Key facts: `o_frag` does NOT scale with `NUM_MMA_KV` (it accumulates *across* kv-tiles, reused); `NUM_WARPS_KV` split-K does NOT shrink it either — `threadblock_sync_mdo_states` (`:1554`) confirms each warp holds the *full* `o_frag` and reduces via smem. Only fewer query rows per warp (more `NUM_WARPS_Q`) cuts `o_frag`, down to the 128-reg floor; going below needs head-dim parallelism (invasive, prefill doesn't do it).

**"Move registers to memory more often" — only SMEM helps, never VRAM.** Register allocation is static (ptxas fixes per-thread count for the whole launch; you reduce it by shortening live ranges, not by runtime free). Spilling to *local memory = VRAM* (L1/L2-cached) is the disease we're avoiding — doing it manually saves nothing. *SMEM* (on-chip, ~20-30cyc) is the only useful staging target. Caveat: the MMA accumulator (`o_frag`) MUST be in registers during every PV MMA (it's the C operand) and is live across the whole kv-loop, so it can't be parked during compute — *except* the two-level fp32 master, which is idle during a tile's fp16-partial MMAs (touched only at the promote). `NOTES.md:211` already evaluated and rejected smem-staging the scales (~0.04× for <1% e2e, intrusive).

**Register-relief menu** (the design choice; `cuobjdump -res-usage` REG+STACK + an occupancy-calculator smem check decide the winner):

| Variant | reg Δ (hd256) | Accuracy | Cost / catch |
|---|---|---|---|
| Two-level, `mma_kv`-outer (naive) | **+64** (worsens spill) | safe | partial live across whole `compute_sfm_v` — perf-negative, do not ship |
| **Two-level, `mma_d`-outer restructure** | **+4** | safe | reorder V-smem traversal / ldmatrix (invasive but local) — **likely best** |
| Smaller `CTA_TILE_KV` | frees `s_frag`/scale room for the partial | safe + **more accurate** (shallower fp16 chains, §4) | more `update_mdo_states` rescales (128 fp32 mults/tile on the slow CUDA-core path); joint accuracy+register knob |
| smem-resident fp32 master | inner-loop peak **−64** | safe | per-tile smem RMW; trades reg pressure for **smem-capacity** occupancy limit (reuses `cta_sync_o_smem` `:105`, but it's `float[1]` when `NUM_WARPS_KV==1` → fresh alloc) |
| Pure fp16 `o_frag` (replace master) | **−64** (kills spill) | ⚠️ **swamping risk** | the only variant that could relieve the *existing* spill, but deletes the fp32 master → flat-row failure (§4) |

The tension: the variant that most relieves spill (pure-fp16) is the accuracy-risky one; the accuracy-safe two-level either adds regs (naive) or needs a restructure (`mma_d`-outer) / a resource-trade (smem master) to stay neutral. Measure all variants (plus the composed int8-QK + fp16-PV config — same 255-cap kernel) with `cuobjdump -res-usage` BEFORE writing the e2e path; if none drops STACK without losing the fp32 master, the lever stays a <3% niche.

**MEASURED 2026-06-23 (sandbox 2×3090 sm_86, CUDA 13, flashinfer 0.6.8.post1 JIT, single_prefill hd256):**

| Build | kernels at REG=255 | spilling (STACK>0) | max STACK (B) | avgREG |
|---|---|---|---|---|
| Baseline i8 hd256 (deployment kernel) | most | yes | **416** | — |
| Baseline f16 hd256 (controlled) | 16 / 56 | 9 | **96** | 160 |
| Variant B: two-level `mma_kv`-outer (+64) | 26 / 56 | 19 | **144** | — |
| **Variant −64: pure-fp16 `o_frag` (split probe)** | **0 / 56** | **0** | **0** | **140** |

Baseline confirms `NOTES.md` exactly (255-capped, STACK to 416 on i8). Variant B (naive accuracy-safe two-level, fp16 `o_partial` promoted into the kept fp32 `o_frag`, ~30-line change to `compute_sfm_v` only) **worsens spill: +10 kernels hit the 255 cap, +10 begin spilling, max STACK 96→144 (+50%)** → the naive two-level will not ship.

**Variant −64 MEASURED 2026-06-24 (the decisive register probe, GO):** a split probe — loop accumulator `o_acc[NUM_MMA_Q][NUM_MMA_D_VO][4]` carried through `init_states`/`update_mdo_states`/`compute_sfm_v` (PV MMA → transient `_t[8]` kInit, add `[0..3]` into `o_acc`), then materialized into the kept `o_frag[8]` post-loop (so `threadblock_sync`/`transform_output`/`write_o_reg_gmem` are untouched). Numerically garbage but register-faithful (`o_acc[4]` = 4 regs/frag = the exact fp16 `o_frag` footprint; conservative — real fp16 is even lighter, no `_t[8]`). Result: **spill ENTIRELY ELIMINATED — 9→0 spilling kernels, 16→0 at the 255 cap, max STACK 96→0, avgREG 160→140.** So freeing the 64 `o_frag` registers is decisive: the −64 relief is real and large, NOT a marginal effect. avgREG only dropping 160→140 (not to ~50) confirms the kernels still do real work — this is genuine spill relief, not dead-code elimination.

**Implication:** the register-relief thesis is GREEN — the accuracy-safe variants that capture this relief (`mma_d`-outer restructure, smem-resident master) are now worth their real builds. Two caveats remain before a perf claim: (1) relief→perf is not automatic (the hd128-vs-hd256 confound showed the JIT can re-spend freed regs on bigger tiles — but here the hd256 tile config is FIXED, so the relief should convert); (2) accuracy of fp16 accumulation is still ungated (§4). Both multi-function rewrites touch `init_states`/`update_mdo_states`/`compute_sfm_v` + the materialize; the accuracy-safe ones keep the fp32 master so they also touch the promote — but the register headroom to add the partial now provably exists.

**Cudagraph:** NO landmine (categorical). Pure-device constexpr, zero host code, zero CPU sync — not the `int8qk_backend.py:217-227` `.tolist()` class. Capture-safe by construction; still test under FULL cudagraph + spec-decode. Decode hard-routes to fp32 (decode PV ~0 gain) — verify prefill-only gating is airtight.

---

## 3. Amdahl — why this is a low-single-digit lever

The per-kv-tile schedule (prefill.cuh main loop ~1967-2024): `compute_qk` → logits_transform/mask → `update_mdo_states` (exp2 + `o_frag *= o_scale`) → `compute_sfm_v` (P@V). fp16-accum touches **only** the PV-matmul issue.

| Slice | Fraction of prefill wall-time | Touched by fp16-PV? |
|---|---|---|
| fp32 softmax (exp2/SFU) + rescale + V-IO + epilogue | ~76–80% | **No** |
| QK-matmul | ~8–12% | No (that was int8-QK) |
| **PV-matmul** | **~8–12%** | **Yes (≤2×)** |

The kernel is register/IO-latency-bound, not MMA-issue-bound (measured no-load ceiling only ~1.09–1.12×), so halving PV-MMA issue recovers ~1.3–2.0× of the slice, not the full 2×. Amdahl:

- op-speedup: PV=0.08,s=1.3 → +1.9%; PV=0.10,s=1.5 → +3.5%; PV=0.12,s=2.0 → +6.4%
- **e2e TTFT (single-card):** +0.3% (low, attn~15%) / **+1.4% (expected)** / +3.2% (high, 64k+ attn~55%)

Regime: prefill-only, single-card, long-context (≥32–64k), GeForce-GA10x, and **only the 8/32 full-attn layers** (3,7,…,31) of the hybrid (GDN layers never call prefill.cuh). Empirical sibling: shipped int8-QK netted 1.03–1.16× op / +2.05% best e2e — same order. **The adversarial gain-bracket verdict refuted "meaningfully worth building" in the general case.** On the 2×3090 box TP all-reduce is 67% of prefill → the win is invisible there; value is single-card long-ctx only.

---

## 4. Accuracy plan — cheapest falsification BEFORE any kernel

Two-level fp16-PV is **strictly less precise** than fp32 `o_frag`: it adds one irrecoverable fp16 rounding per `CTA_TILE_KV` tile (eps_fp16 ≈ 4.88e-4). The fp32 master only removes the sequence-length swamp — it does NOT un-round the per-tile partial. Worst on **flat, long, cancellation-heavy rows** (uniform attention, large NUM_MMA_KV, most promotes) = exactly the needle / Chinese-long-CoT regime at hd256 (16 PV k-tiles, more promotes than any prior art). SageAttention's "no accuracy loss" is diffusion/English-hd128 — physics is a perf argument, never an accuracy one.

**MEASURED 2026-06-24 (synthetic, GATE B GREEN):** faithful fake-quant (full online-softmax + per-tile fp16 CHAIN across width-16 groups → promote; NOT a lazy single-cast) swept tile{16,32,64,128,256} × L{8k,32k,128k} × flatness{uniform,flat,peaked}. **two-level (fp32 master + fp16 partial): worst-row cos 0.999999.** **pure-fp16 (no fp32 master, the −64 register winner): worst-row cos 0.999928** — also clears the 0.9999 bar, ≈ the shipped int8-QK's 0.99992 (e2e-validated). The online-softmax per-tile rescale bounds O, so the feared swamping does NOT appear. Pure-fp16 error grows with smaller tile / longer ctx / peakier attention (more fp16 master rescales → larger tile is *better* for pure-fp16, opposite of two-level). **Consequence: pure-fp16 — the simplest variant — carries BOTH the −64 register relief AND acceptable accuracy.** Remaining accuracy work = the real-model capture (Step 1, below) + the autoregressive e2e (the argmax-flip test, below). Harness: `pv_fp16_acc2.py` (sandbox).

**Step 1 (capture, NOT yet done):** UNMODIFIED bf16 Qwen3.5-9B + Qwen3.6-27B, hook post-online-softmax P [q_len×kv_len] + V [kv_len×256] bf16 on the 8 full-attn layers {3,19,31}, ~8 heads. Use eager/HF (`output_attentions`) if the fused path hides P. Validates the synthetic GREEN against real learned attention structure (sinks, local+global).

**Step 2 (worst-case prompts):** (a) Chinese long-CoT >8k tokens; (b) needle 32k/64k/128k in near-uniform filler; (c) **uniform-attention synthetic control** P=1/L over L∈{1k,8k,32k,128k} — non-negotiable, bounds worst case independent of prompt luck.

**Step 3 (fake-quant, zero CUDA):** REF = `P.float()@V.float()`. TEST = per `CTA_TILE_KV` tile, accumulate the partial in `torch.float16` **emulating the fp16 accumulator CHAIN across NUM_MMA_KV width-16 groups** (NOT a single cast of the tile sum — most fragile step, lazy cast = false GO), promote to fp32 master, apply `*= o_scale` between tiles. Sweep `CTA_TILE_KV ∈ {16,32,64,128,256}`.

**Step 4 (GO threshold):** per-q-row cos; report **MIN over rows** + that row's flatness + kv_len + non-finite/underflow. **HARD GATE: worst-row cos > 0.9999** (matches int8-QK 0.99992–0.99996) across ALL captured (layer,head,prompt) including the uniform control. SOFT-GO 0.999–0.9999 only if smaller `CTA_TILE_KV` recovers it. NO-GO if ≤0.999 or any underflow/non-finite.

**Step 5 (self-checks):** (a) confirm `sqrt(NUM_MMA_KV)` error growth (else emulation wrong); (b) smallest-tile bound `CTA_TILE_KV=16` (if even that fails, lever is dead); (c) clamp V to fp16-normal to isolate the bf16→fp16 subnormal-underflow absolute-error mode.

**ONLY IF Step 4 passes — autoregressive e2e is a HARD BLOCKER** (a per-forward cos 0.9999 can still flip an argmax that cascades over thousands of decode steps fed by perturbed prefill output). Build the GeForce-gated kernel, vs fp32 baseline on a real 3090, under `VLLM_BATCH_INVARIANT`, **temp=0.6/top_p=0.95 (never greedy on Qwen3.5 thinking)**: (1) Chinese long-CoT coherence (no loop/collapse); (2) needle 32k/64k/128k EXACT retrieval; (3) GSM8K within 0.5pp of fp32 (ref int4 96.8%); (4) MMLU-Pro within 0.5pp (ref 82.4% / 80.2%). The flat-row corrupt-but-coherent-token failure is this project's hardest-to-detect mode; non-English is the canary.

---

## 5. Shippability

Clears the fork's two bars (generalizes across Ampere + shippable) as gated opt-in: patch 0001/0002 already ships the "+19–49% sm_86, ~0 sm_80" shape, gated. fp16-PV is identical. Ship as **patch 0007** over the vendored FlashInfer template, default-OFF behind `VLLM_PV_FP16ACC=1` + the runtime GeForce-GA10x guard. NOT the `vllm.general_plugins` entrypoint (that swaps a backend class; this is a kernel-internal constexpr below the Python boundary). The fragment-layout promote (f16 C[0..3] → fp32 C[0..7]) is the same m16n8k* layout-bug family that cost multiple int8-QK debug sessions — needs a standalone cos harness (like `i4_test.py`) before e2e. Upstream: clean PR candidate (symmetric to the documented `allow_fp16_qk_reduction`) but low maintainer pull (FlashInfer roadmap is Hopper/Blackwell) — plan fork-local maintenance.

---

## 6. First step — two cheap independent gates, either NO-GO kills it

Both run before any e2e kernel work; they're independent, so run in parallel.

**Gate A — register probe — DONE 2026-06-24, GREEN (§2a).** The split −64 probe on real sm_86 (sandbox 3090, flashinfer JIT + `cuobjdump -res-usage`) ELIMINATED spill entirely (9→0 spilling kernels, 16→0 at the 255 cap, max STACK 96→0). The relief is real and large → the only path to a >3% win is open, justifying the `mma_d`-outer / smem-master builds that capture it *without* losing the fp32 master (§2a menu). (Pass.)

**Gate B — zero-CUDA fake-quant cos sweep (<1 day, no GPU-kernel).** §4 Steps 1-4: HF eager-forward Qwen3.5-9B (+ a bf16 model for the V→fp16 overflow check) with `output_attentions` on ~6 prompts (Chinese long-CoT + 32k needle) + the uniform synthetic control; dump (P,V) for the 8 full-attn layers; emulate the two-level fp16 accumulator CHAIN; worst-row cos vs fp32 at `CTA_TILE_KV ∈ {16,32,64,128,256}`. Worst-row cos ≤ 0.999 on the uniform/flat-long control → lever **dead before a kernel line**.

Only if BOTH pass: build the GeForce-gated kernel (best `mma_d`-outer variant per Gate A), standalone cos harness for the fragment-layout promote, then the autoregressive e2e blocker (§4). Gate B runs locally (5070 Ti is enough for 9B eager-forward); Gate A needs the FlashInfer build toolchain on an sm_86 target.

---

## 7. Open risks

1. **Accuracy unproven at hd256 LM long-CoT/non-English** — the gating risk; diffusion/English-hd128 evidence does not transfer.
2. **Gain below the 3% bar** in the general case (adversarial: refuted) — only the unmeasured spill-relief path clears it; gate on cuobjdump.
3. **Gate footgun** — runtime GeForce-GA10x check required, NOT `__CUDA_ARCH__ >= 860` (A40/A6000/A10 get zero benefit + accuracy cost).
4. **bf16 V→fp16 cast overflow** — bf16 IS applicable via operand cast (corrected), but `|V|>65504` overflows to inf; falsify in §4's V-clamp self-check.
5. **Register pressure is the make-or-break, and accuracy-safe ≠ perf-winning** — pure-fp16 relieves spill but risks swamping; two-level is safe but reg-additive unless `mma_d`-outer / smem-master (§2a menu). Gate on `cuobjdump -res-usage` across all variants.
6. **Fragment-layout bug class** — needs standalone cos harness first.
7. **Untested variants** — split-K, `use_softmax=false`, soft-cap, RoPE claimed-unaffected-by-construction, unproven.
8. **Box-invisible** — validate perf single-card long-ctx, never on the tp2 rig.
9. **Upstream** — fork-local maintenance expected.
