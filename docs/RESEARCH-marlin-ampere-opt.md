# famp-Marlin Ampere Optimization — Research & Verdict (2026-06-27)

> 52-agent ultracode workflow (map -> multi-lens levers -> 2-lens adversarial verify -> synth -> critique).
> 19 levers, **13 NO-GO / 3 CONDITIONAL / 1 GO**. Bottom line: the kernel is already well-tuned for Ampere.

## Confirmed against the actual code (the critique caught 3 synthesis errors)
1. **fp16-accum is sm75-ONLY** — `use_fp16_accum` is non-false only inside `#if __CUDA_ARCH__ == 750`
   (marlin_template.h:288-304); `#else = false`. There is NO sm86 fp16-accum path. So "un-gate fp16-accum
   for sm86 GeForce" (Bundle C) rests on a false premise -> deep-research / NO-GO for Ampere, NOT Tier-2.
   The "fp16-accum-PV GO precedent" is a DIFFERENT kernel (FlashInfer PV), not this Marlin arm.
2. **The weight (B) load already uses `cp.async.cg`** (L1-bypass streaming; marlin_template.h:860 -> cp_async4
   = marlin.cuh:158-163). So the critique's one NEW lever ("mark weight loads .cg to stop L2 pollution") is
   ALREADY DONE. .ca is used only for reused data (A/scales). Kernel is already optimal here.
3. **The blocks_per_sm=2 "C_tmp OOB corruption" footgun is RETRACTED** — C_tmp is indexed by output slice,
   not (block,blocks_per_sm); the cited line 691 does not demonstrate the overrun. The lever is still inert
   (occupancy is smem-pinned 1-block/SM), but the specific OOB claim was unsubstantiated.

## Net actionable shortlist
- **Recipe g32->g128** (the ONLY real lever; decode ~2% realistic, NOT a kernel edit, g128 kernels already
  built). HARD gate: the int8 coarse-group fold `if (group_blocks==2 || k==1)` (marlin_template.h:1374)
  hardcodes k==1 for g128 -> validate for a CONFIG-SPECIFIC fold-timing bug on the (64,256,256) large-batch
  config + int32-accum saturation, then per-layer GSM8K/MMLU-Pro (keep g32 on outlier down_proj).
- **Source hygiene** — strip Hopper/Ada dead code (0 runtime, compile-only; #if-guard, don't delete).
- **Follow-up RESEARCH (workflow coverage gaps, per critique):** MoE/batched-M (the real 35B-A3B deployment was
  excluded by fiat; phantom-expert masking is the one real MoE decode lever); sm80-specific tiling (the "no tile
  pruned" math used sm86's ~100KB cap — sm80 has 164KB + full FP32 -> wider tiles/deeper stages may be valid on
  sm80, so the stages=3 NO-GO is sm86-only); asym-activation int8 (kernel has has_zp scaffolding).

---

I'll write the roadmap directly from the provided map and verdicts. No file exploration needed — all claims are grounded in the inputs.

# famp-Marlin Ampere-Specific Optimization Roadmap (sm80 + sm86)

## (1) Executive Summary

**Bottom line on the two phases is now empirically settled, not theoretical:**

- **DECODE (M=1) is bandwidth-bound and dead to kernel work.** ncu measured the M=1 path at **81% DRAM SOL / 10.6% IMMA** (`project_int8_path_roadmap_ampere`), and the int8 8-row tile was *fully repaired* yet measured **ZERO decode tok/s change** (`project_ampere_int8_throughput_ceiling`). Every kernel-internal lever (stages, gather, IMMA, occupancy, launch_bounds, blocks_per_sm, fp16-accum) is bounded ~0 at M=1 by construction. The weight bytes (k·n/2 int4) are irreducible without changing precision (W4A4 = NO-GO). **The only decode levers that move bytes are RECIPE-side** (scale/zp side-streams) or **out-of-Marlin** (MTP, lm_head→W4A16, fp8-KV).

- **PREFILL is compute-bound and CAN win — but the int8 GEMM is already tapped out.** W4A8 already delivers +15-18% prefill over W4A16 via int8 tensor cores; ncu shows prefill at **IMMA~68% / SM_SOL≈IMMA / low long_scoreboard** = tensor-op-bound, single persistent wave (grid=82, waves=1.00), with the *recoverable* int8-GEMM gap bounded **≤~1.7%**. Smem-prune, stages=3, launch_bounds, and blocks_per_sm levers all collapsed under exact arithmetic (no reachable tile is pruned on sm86; occupancy is smem-pinned at 1 block/SM, not register-pinned).

**The 2-3 highest-confidence Ampere wins:**

1. **g32→g128 coarser quant group** (recipe, fewer-bytes) — the single largest non-weight byte saver: cuts the scale side-stream 4× (~8-9% of the weight stream), realistic **~2-5% decode** on large-n projections. CONDITIONAL on per-model/per-layer accuracy + numeric validation of the never-run int8 coarse-group fold.
2. **GPTQ-int4 (kU4B8) over AWQ-int4 (kU4)** where accuracy allows — drops the zp side-stream entirely (~1% decode). *But the shipped flagship is already symmetric (no zp), so this is largely a no-op on the deliverable.*
3. **fp16-accum un-gate for sm86 GeForce, W4A16-fp16 only** — the one prefill *compute* lever with a GO precedent (fp16-accum-PV), but N/A to the W4A8/bf16 flagship and needs real template-machinery work + a hard accuracy gate.

**Honest meta-result:** Of 17 verified levers, **13 are NO-GO, 3 CONDITIONAL, 1 GO (the negative-result guardrail)**. The kernel is structurally well-tuned for Ampere already; the remaining real surface is recipe-side bytes for decode and a narrow fp16-only compute lever for prefill.

---

## (2) Tiered Lever Table

### Tier-1 GO

| Lever | Phase | Impact | Effort | Risk | Rationale |
|---|---|---|---|---|---|
| **Negative-result guardrail**: no kernel-side (tiling/stage/gather/dequant/occupancy/IMMA) lever moves M=1 decode; gate any future decode claim on ncu dram SOL≥85% | decode | 0% (records constraint) | low | none | All 6 code anchors verified; corroborates every prior (81% SOL, 8-row=0, W4A4 NO-GO). Prevents wasted kernel-side decode effort. Scope: M=1 single-stream only (batched M=8..64 + MoE phantom-expert byte lever explicitly out). |

### Tier-2 CONDITIONAL

| Lever | Phase | Impact | Effort | Risk | Rationale |
|---|---|---|---|---|---|
| **g32→g128 group size** (recipe) | decode | ~2-5% (large-n proj) | medium | HIGH (accuracy) | Largest real fewer-bytes lever (scale stream 4×); g128 kernels already built. Gated on: (a) numeric validation of the **never-run int8 coarse-group `k==1` fold** path, (b) per-model/per-layer GSM8K+MMLU-Pro (keep g32 on outlier down_proj). Impact corrected down from 5-8% (weight stream ≠ all decode DRAM; bf16 GDN/lm_head/KV untouched). |
| **Strip Hopper/Ada/Turing dead code** (fp8/NVFP4/MXFP MMA+dequant, sm75 asm, fp8 fragments) | both | 0% runtime; compile/.so cleanup | medium | low | Genuinely dead on single-arch sm86 build (verified: SUPPORT_FP8 sm89+ only; if-constexpr discarded). **MUST carve out: #if-guard (not delete) `use_fp16_accum` f16f16f16 arms + W8A16/kU8B128 path** — those are future-lever scaffolding (B7/D1, fp8-W8A16). Compile/size win smaller than claimed (uninstantiated templates emit ~0 SASS). |
| **fp16-accum un-gate for sm86 GeForce, W4A16-fp16 only** | prefill | +1-3% (fp16 W4A16 on GeForce GA10x) | high | MED-HIGH (accuracy) | Real 2×-HMMA mechanism (arm exists marlin_mma.h:45-67); GO precedent (fp16-accum-PV). **But: `use_fp16_accum` is constexpr — "runtime GeForce-SKU select" needs a new template axis + doubled instantiations + host SKU detection, NOT a gate flip.** N/A to W4A8/bf16 flagship. Hard GSM8K/MMLU-Pro gate at K=4096-18432 (fp16-accum overflow risk). |

### NO-GO (dropped) — with the prior finding each contradicts

| Lever | Phase | Verdict rationale (prior contradicted) |
|---|---|---|
| **sm86 stages=3 family** (W4A16 large-M) | prefill | Premise falsified by own formula: max reachable stages=4 footprint = **71.7KB << 100.9KB sm86 cap** — NO tile is pruned. The cited 103KB (128,128,256)@m=4 is **unreachable** (small-batch-only, m≤1). stages=3 only shrinks the prefetch pipeline → net-negative. |
| **sm86 stages=3 for W4A16 wide tile** (variant) | prefill | Same; conflated W4A16 (int4-weight, ~70KB) with W8A16 (int8-weight, ~101KB). W8A16 is dropped from famp subset anyway. Contradicts nothing — refuted on own terms. |
| **blocks_per_sm=2 in determine_exec_config** | both | Mechanism real but: (a) occupancy is smem-pinned (each CTA reserves full opt-in smem), (b) **as written = silent OOB corruption — C_tmp is sized `sms*...` not `sms*blocks_per_sm*...`** (marlin.cu:691) → overruns on default bf16 fp32-reduce path. ncu shows waves=1.00 (1 block/SM by design). Contradicts persistent-grid finding. |
| **__launch_bounds__(256,2) floor occupancy** (×3 variants) | both | Occupancy is **smem-bound, not register-bound** — each CTA requests full opt-in smem via cudaFuncSetAttribute, so 2 blocks/SM is physically impossible; the hint is inert or forces spills. Contradicts ncu (grid=82/waves=1.00 by design, IMMA=10.6% M=1 / 68% prefill tensor-op-bound). |
| **Prune unreachable (128,128,256)@m>1** | both | Config already not emitted (verified 0 instances in sm_86 codegen); smem checked at runtime with loud TORCH_CHECK (not silent fallback). Pure churn — fails feedback_squeeze_every_bit's "no non-fixes" bar. (One salvageable fragment: optional build-time smem-fit assert.) |
| **Hoist g32 int8 scale-fold out of K-loop** | prefill | At g32 (group_blocks=2) the int8 k-tile = **exactly one group**; the `group_blocks==2` gate is a **correctness requirement** (single shared frag_c_tmp), not redundant overhead. Proposed test → no-op (div_ceil(2,4)=1) or corruption. Contradicts int8-GEMM-tapped-out. |
| **Fuse int8 act-scale with I2F convert** | prefill | Already done in source (1976-1979 are adjacent register-local statements; nvcc emits I2F+FMUL with no store-back). Epilogue is O(output) not O(K), runs once per slice. ~0. Contradicts int8-GEMM-tapped-out (confirms it). |
| **Arch-gate cross-block reduce to fp16-store** | prefill | The fp16-reduce **m_block_size_8 branch is flagged latent-broken** (project_ampere_int8_throughput_ceiling) and fires for W4A8 at M≤8. Reduce is once-per-slice epilogue; prefill is IMMA/dequant-bound not L2-bound. bf16 worst-precision. |
| **Pre-expand g32 int16 scales to drop __shfl_sync** | both | The shuffle is an **irreducible cross-lane permutation** (owner lane%4 → consumer lane/4), not a hoistable broadcast. At g32 already at min frequency (once/group). Decode=0 (DRAM-SOL). High verify burden (4-bug history) vs noise payoff. |
| **L2 residency via cudaAccessPolicyWindow** | decode | Self-closed: per-layer int4 weight (10-45MB) >> GA102 L2 (6MB); persisting region > L2 = no reuse + evicts KV/attention lines = net-negative. Confirms decode-bandwidth wall. |
| **GPTQ-int4 over AWQ-int4 (drop zp)** | decode | Real mechanism but **shipped flagship is already symmetric (has_zp=false)** — no-op on the deliverable. Only applies to a non-default asym AWQ ckpt, which the project deliberately avoids (asym weight quality ≈ 0). Contradicts validated symmetric recipe. |
| **Make bf16 sm90-atomicAdd statically unreachable** | both | Footgun already defended: vLLM caller `should_use_atomic_add_reduce` explicitly gates `sm<9 && bf16 → False`, plus default-off env + shape gate. Redundant; the bf16x2 atomicAdd is also emulated/legal on sm86, not "illegal." Throughput 0. |

---

## (3) The #1 Lever, Fully Specified: g32→g128 Coarser Quant Group

This is the only lever with a path to exceed the ~1% squeeze floor with a real, byte-grounded mechanism. It is a **recipe change with zero kernel edits** — the g128 kernels already exist — but it requires a numeric-correctness validation of an int8 code path that has likely never executed in production.

### Why it's #1 (grounded)
- Scale tensor is bf16, streamed every launch: `s_gl_stride = prob_n/8` (W4A8, is_8bit_scale=false), fetched per-group at `marlin_template.h:881-891`. At g32: scales = k·n/16 = **12.5% of the k·n/2 int4 weight stream**; at g128: k·n/64 = **3.1%**. Delta ≈ **9.4% of weight-adjacent bytes** — the only term that moves 4× with group size.
- Decode is 81% DRAM-SOL → byte removal converts ~linearly below SOL.
- **Corrected impact: ~2-5% decode** (not 5-8%): the int4 weight stream still dominates, and bf16-kept GDN in_proj/router/shared_expert/norm/lm_head(248k vocab)/embed are untouched, so the scale delta is a smaller fraction of *total* decode DRAM.

### Files / regions
- **No kernel edit.** Kernels exist: `generate_kernels.py:77,84,92,100` instantiate `group_blocks=[-1,2,4,8]` for both kS8 (W4A8) configs across `thread_m_blocks=[0.5,1,2,3,4]`.
- **Recipe**: `quantize/requant_v2_awq_mse_g32.py` (and siblings) — change `group_size=32` → `128`. Keep AWQ-smoothing + QuantizationModifier symmetric=True.
- Host plumbing already correct: `marlin.cu:344` (`group_blocks = group_size/16` → 8), scale cache `marlin.cu:149-177` (`tb_groups=div_ceil(tb_k,group_size)` → smem *shrinks* 4×, only relaxes is_valid_config).

### Concrete change
1. Quantize one model (start with the dense 27B, then the hybrid 9B) at **g128** and at **per-layer-mixed** (g128 everywhere except outlier-heavy down_proj which stays g32 — feasible since Marlin selects group_blocks per-GEMM).
2. Ensure each layer's K is divisible by its own group_blocks (`marlin.cu:345` TORCH_CHECK; g128 needs K%8==0).

### Build steps (only needed for the numeric-correctness harness, not the recipe)
The recipe change needs no rebuild. To run the GEMM correctness/ncu harness against famp_marlin:
```
# On the sm86 box (3090). build is single -gencode sm_86 from live device cap.
rm -rf /home/trevor/vllm-ampere-optimized/flashampere/marlin/build/   # nuke sticky .so + generated *kernel_*.cu
python -c "from flashampere.marlin.build import get_famp_marlin; get_famp_marlin()"
# build.py: nvcc -rdc per .cu (4 workers) -> nvcc -dlink -> g++ -shared famp_marlin.so
# ~15min; -rdc + device-link mandatory (cross-TU __global__ address refs). NEVER nvcc --threads (silent symbol drop).
```

### A/B benchmark plan on sm86 (2×3090)

**Step 0 — CORRECTNESS FIRST (gating, the int8 coarse-group fold is unvalidated):**
- Unit GEMM: `benchmarks/bench_marlin_gemm_imma.py`-style, compare marlin int8 **g128** output vs fp32 dequant-matmul reference, across BOTH int8 thread configs: `(128,128,256)` small-batch/decode (thread_k_blocks=8, one pipe = one g128 group, fold at `k==1`) AND `(64,256,256)` large-batch (thread_k_blocks=4, half a group per pipe). The `k==1` fold (`marlin_template.h:1373-1374`) has never run for coarse int8 groups in production. Require **cos > 0.9999 and bit-exact vs reference** on REAL (non-all-ones) activations across all 4 W4A8 layers (qkv/o/gate_up/down), including the 8-row transposed branch (1384-1413).
- ncu `dram__bytes_read` at M=1 → confirm the ~9.4% scale-byte cut materializes.

**Step 1 — Accuracy gate (hard):**
- `vllm serve` (W4A8, `VLLM_MARLIN_INPUT_DTYPE=int8`, tp2, `--shm-size=8g`, `--max-num-seqs 32`).
- GSM8K (`benchmarks/gsm8k.jsonl`) + MMLU-Pro (`benchmarks/mmlu_pro.jsonl`) + zh-CoT: g32 baseline vs g128 vs per-layer-mix. **Adopt only within run-to-run noise.** Expect uniform g128 to regress on down_proj (50000× outlier) → per-layer-mix is the de-risk.

**Step 2 — Perf (only if Step 0+1 pass):**
- Decode tok/s via `scripts/api_bench_client.py` (streaming OpenAI API, client-side `decode=(ctoks-1)/(tlast-tfirst)`) AND clean manual timing `scripts/bench_decode_clean.py` (t(N)-t(1), prefix-cache OFF) — **NOT** get_metrics TPOT (~2× inflated). Single-card 27B b1, g32 vs g128.
- Report decode/prefill as **tok/s, never ms**.

**GO criteria:** Step 0 bit-exact + Step 1 within accuracy noise + Step 2 shows measurable decode tok/s gain on large-n projections. If the int8 coarse-group fold is not bit-exact, NO-GO immediately (read-only analysis could not rule out a latent off-by-group fold bug).

---

## (4) Do-Together Bundles + Implementation Sequence

**Bundle A — Recipe fewer-bytes (decode), one quantize+eval cycle:**
- g32→g128 (#1) + GPTQ-vs-AWQ zp check. Both are recipe-only, share the same quantize→serve→GSM8K/MMLU-Pro→decode-tok/s harness. Run them in the *same* eval cycle per model (g32-AWQ baseline / g128-sym / per-layer-mix). Note GPTQ-zp is likely a no-op (flagship already symmetric) — fold it in as a free comparison, don't spend a separate cycle.

**Bundle B — Source hygiene (build-time only), one rebuild:**
- Strip Hopper/Ada/Turing dead code **+** the optional build-time smem-fit assert (the one salvageable fragment of the prune lever). **Carve-out is mandatory:** #if-guard `use_fp16_accum` arms and W8A16/kU8B128 dequant rather than deleting, to preserve Bundle C's scaffolding. Measure `rm -rf build && time build` + .so/SASS before/after — pitch as compile-time only, never tok/s.

**Bundle C — fp16-accum prefill (sm86 GeForce, W4A16-fp16), standalone, high-effort:**
- Only if a real fp16-served W4A16 deployment exists. Requires the new `use_fp16_accum` template axis + doubled instantiations + host SKU detection (cudaGetDeviceProperties.name; pro-sm86 A40/A6000/A10 share sm86 but run full-rate FP32 → must NOT fire). Gate: Phase-0 ncu M-sweep (confirm HMMA-issue-bound) → GSM8K/MMLU-Pro at largest K → serve-API prefill tok/s.

**Suggested sequence:**
1. **Ship the GO guardrail** (Tier-1) into docs/MEMORY now — zero risk, stops future agents re-attempting the 13 NO-GO kernel-side decode levers. (Already effectively done via this roadmap + the memory entries.)
2. **Bundle A** — highest expected value (~2-5% decode), recipe-only, but front-load the **int8 coarse-group fold numeric validation (Step 0)** before any accuracy/perf work; it's the real risk and it's read-only-undecidable.
3. **Bundle B** — cheap maintainability win, do whenever a rebuild is convenient; the carve-out preserves Bundle C.
4. **Bundle C** — last, high-effort, narrow scope (W4A16-fp16 + GeForce only), N/A to the W4A8/bf16 flagship. Only if a matching deployment exists.

**Do NOT pursue:** any stages/launch_bounds/blocks_per_sm/tile-prune/scale-fold/shuffle/L2 kernel lever — all 13 are arithmetically or empirically refuted against the actual code (smem-pinned 1-block/SM occupancy, no reachable pruned tile, DRAM-SOL decode floor). The blocks_per_sm variant is additionally a **latent OOB corruption** (C_tmp sizing) if anyone reopens it.

---

# Completeness Critique (adversarial)

The key fold mechanics are now clear. The fold loop is `for (int k = 0; k < b_sh_wr_iters; k++) matmul_a8(k)`. The gate `group_blocks == 2 || k == 1` folds at every inner iteration when group_blocks==2 (g32), but for g128 (group_blocks==8) folds only at the single iteration `k==1`. 

The crucial question: does frag_s hold the SAME scale across all k iterations within a tile when group_blocks >= thread_k_blocks (which is true for g128 if the tile's K < group)? From fetch_scales lines 1004-1017: when `group_blocks >= thread_k_blocks`, the scale is loaded once per `pipe % g == 0` and held — so a single scale is valid for the whole tile. In that regime the fold at k==1 is correct ONLY IF the tile spans exactly one group. But the gate hardcodes `k==1` — it does NOT check whether the accumulation has actually covered a full group. This is the latent risk. Note `s_type_id = c_type_id` (bf16 for the flagship), so `is_8bit_scale=false`, confirming scale stride = `prob_n/8`.

I have enough verified grounding. Let me compile the critique.

Verified against source. Critique follows.

---

## Verified factual corrections to the roadmap (these change the conclusions)

**1. The fp16-accum lever (Bundle C + the Bundle B carve-out) is mis-architected against the actual gate. This is the biggest error.**
`use_fp16_accum` is not merely `constexpr` — it is hardwired to `__CUDA_ARCH__ == 750` (Turing/sm75). marlin_template.h:288-303: only the `#if __CUDA_ARCH__ == 750` branch can ever set it true; the `#else` is `constexpr bool use_fp16_accum = false`. So on the sm80/sm86 build the f16f16f16 MMA arms (marlin_mma.h:46-62) are dead, and the "future-lever scaffolding" Bundle B is told to preserve is **sm75 scaffolding, not sm86**. The roadmap's "GO precedent (fp16-accum-PV)" is a different kernel (FlashInfer PV), not this Marlin arm. "Un-gate for sm86 GeForce" is therefore not "a new template axis + host SKU detect" on top of an existing sm86 path — it requires lifting an arch-gate that upstream deliberately restricted to sm75, then validating fp16-accum overflow on sm86 from scratch. Effort is higher and precedent is weaker than stated; arguably this should drop to NO-GO/research, not Tier-2 CONDITIONAL.

**2. The C_tmp / blocks_per_sm OOB claim is wrong as written.** The roadmap asserts "C_tmp is sized `sms*...` not `sms*blocks_per_sm*...` (marlin.cu:691) → overruns." But C_tmp (marlin.cu:688-693) is sized `sms * max_m_block_size * max_thread_n` and is indexed by **slice**, not by physical (block_idx, blocks_per_sm) pair. The fp32-reduce buffer is partitioned across the grid's logical output slices; doubling blocks_per_sm changes occupancy/grid, not the per-slice C_tmp footprint, so the specific "OOB corruption" mechanism cited is not demonstrated by line 691. The blocks_per_sm lever may still be inert (smem-pinned occupancy is the real reason, and that part is correct), but the roadmap's headline "latent OOB corruption" footgun is unsubstantiated and should be retracted or re-grounded — leaving a false specific bug in a roadmap is its own hazard.

**3. The #1 lever's central risk is real but mis-stated, and there is a sharper, un-flagged correctness concern.** The fold gate is `if (group_blocks == 2 || k == 1)` (marlin_template.h:1374). For g32 (group_blocks==2) it folds every inner iteration — fine. For g128 (group_blocks==8) it folds **only at the single hardcoded `k==1`** inside `for (k=0; k<b_sh_wr_iters; k++)`. The gate hardwires `k==1` rather than "after exactly one group of int32 accumulation has completed." Whether that is correct depends on `b_sh_wr_iters` vs the group/tile geometry for each int8 config — and the small-batch `(thread_k=...,256)` vs large-batch `(...,256)` configs have different `thread_k_blocks`, so `k==1` is correct for one and potentially mis-timed for the other. The roadmap says "validate the k==1 fold" but frames it as a generic bit-exactness check; the actual question is **a specific off-by-group fold-timing bug between the two int8 thread configs at g128** — that is what Step-0 must target, with the (64,256,256) large-batch config being the high-risk one, not just "across both configs." Also: because int8 holds raw int32 in frag_c_tmp across the whole tile, g128 means int32 accumulation over 8 k-blocks before scaling — worth a saturation sanity check, not only a cos>0.9999 check.

**4. Scale-stride arithmetic confirmed, but the "9.4%" is the wrong denominator for the headline.** s_gl_stride = `prob_n / (is_8bit_scale ? 16 : 8)` (marlin_template.h:562) and for W4A8 `s_type_id = c_type_id` = bf16 → `is_8bit_scale=false` → `/8`, so the scale stream is bf16 and the g32→g128 4× cut is real. The roadmap's own corrected "~2-5% of *total* decode DRAM" is the honest number; the "9.4% of weight-adjacent bytes" should not appear near the headline since weight-adjacent ≠ total stream and the table already had to walk 5-8% back to 2-5%. Net: the #1 lever's realistic ceiling is closer to the low end (~2%), and that should be stated up front, not buried.

---

## Missing Ampere-specific levers (not in the 17)

**A. cp.async cache-hint / L2 streaming on the weight load (`cp.async.cg` vs `.ca`, evict_first).** The roadmap treats L2 only as the (correctly rejected) `cudaAccessPolicyWindow` persistence angle. The opposite Ampere lever is un-examined: the int4 weight stream is pure streaming (read-once, never reused across the M=1 launch), so marking those `cp.async` loads `evict_first` / `.cg` (bypass-L2-ish) would stop them **polluting** L2 and protect KV/activation residency — the same net effect the roadmap wanted from persistence, but achievable and decode-relevant. sm80/sm86 cp.async supports the cache-global hint. Worth one ncu `lts__t_sector_hit_rate` A/B; near-free if it composes.

**B. sm80-vs-sm86 separate tile/stage tuning — the roadmap collapses the two arches.** The smem-fit math ("71.7KB << 100.9KB cap") uses the **sm86** ~100KB opt-in cap to declare "no tile is pruned." But sm80 (A100) has the 164KB cap *and* full-rate FP32 *and* more SMs — the reachable/optimal tile and stage count differ. "No tile pruned on sm86" does not transfer to sm80, yet the roadmap's NO-GO on stages=3 is asserted for both. At minimum the stages=3 NO-GO is sm86-only; sm80 large-M with the 164KB budget can host deeper pipelines or wider tiles that sm86 cannot, and that is exactly the "general Ampere sm80+sm86" scope the project claims. This is a completeness gap, not just optimism.

**C. The dual-issue / half-rate-FP32 asymmetry as a *prefill* lever, specifically for the bf16 flagship.** sm86 runs FP32 at half rate and has fewer tensor cores/SM than sm80. The bf16-accumulate (`f32.bf16.bf16.f32`, marlin_mma.h:71) epilogue/dequant FP32 work is therefore relatively more expensive on sm86 than sm80. The roadmap declares prefill "tapped out ≤1.7%" from one ncu run, but never asks whether the dequant/scale FP32 path is disproportionately costing the sm86 GeForce relative to sm80 — i.e., whether an int32→scale fold (already int-domain for int8) or a reduced-FP32 epilogue helps sm86 more. The "tapped out" claim rests on a single arch's ncu and is over-generalized.

**D. MoE / batched M is excluded by fiat, but the project's real box runs MoE.** The Tier-1 guardrail explicitly scopes out "batched M=8..64 + MoE phantom-expert byte lever." Given memory says phantom-expert masking is "the one real decode lever" for MoE and the flagship includes a 35B-A3B MoE, scoping it out of a *completeness* roadmap is the single largest coverage hole. The guardrail's "no kernel lever moves decode" is only true for dense M=1; for MoE the per-expert M is small-but->1 and routing-byte/phantom-tile effects are live. This belongs in the roadmap as an open Tier, not a footnote exclusion.

**E. zp side-stream removal is dismissed as "no-op (flagship symmetric)" — but asym *activation* is in the memory as "the real lever."** The roadmap kills GPTQ-vs-AWQ correctly for weight-zp, then stops. It never notes that the kernel's `has_zp` int8 path (matmul_a8 zp-dequant, marlin_template.h:1334-1342) is the scaffolding for the asym-activation direction memory flags as the genuine accuracy/byte frontier. Not a decode-speed lever, but its omission makes the "13 NO-GO" count look more closed than the surface actually is.

---

## Over-optimism / unverified claims to flag

- **"≤~1.7% recoverable prefill, IMMA~68%, tapped out"** — single-ncu, single-arch (sm86), single-shape assertion generalized to "PREFILL tapped out." Not validated across K=4096..18432 or on sm80. Treat as sm86-point-estimate, not a law.
- **"smem-pinned 1 block/SM, occupancy not register-bound"** is asserted as the reason every occupancy lever is inert, but no `cudaFuncAttributes` / `-Xptxas -v` register count is cited — only ncu waves=1.00 (which is consistent with persistent-grid-by-design *or* occupancy-pinned; the roadmap can't distinguish them from waves alone). The "physically impossible 2 blocks/SM" claim needs the actual per-CTA smem request vs cap, per arch (sm86 ~100KB vs sm80 164KB → 2 blocks/SM may be possible on sm80).
- **"kernels already built for g128"** — true in codegen (generate_kernels.py:92,100 emit group_blocks=8), but the roadmap's own Step-0 admits the g128 int8 fold may never have executed. "Built" ≠ "validated"; the GO-criteria correctly gate on bit-exactness, but the Tier-2 "effort: medium" understates that this is a latent-bug hunt in a transposed 8-row fragment path with a 4-bug history, on the highest-risk (64,256,256) config.
- **Build recipe risk:** the doc instructs `rm -rf build` then rebuild "~15min" and warns "NEVER nvcc --threads" — but the environment rule is read-only/no-GPU-builds, and the roadmap presents the build as routine. Fine as documentation, but the GO criteria depend on a build+ncu the analysis itself cannot run, so every "verified" Step-0 claim is actually "must-verify," and the roadmap occasionally blurs that line (e.g., "kernels exist" stated as derisking).

**Net:** The decode-wall thesis and the 13 kernel-side NO-GOs are sound and well-grounded. The three errors that matter: (1) fp16-accum is sm75-gated, not sm86-scaffolded — Bundle C and the Bundle-B carve-out rest on a false premise; (2) the C_tmp OOB footgun is not supported by the cited line and should be retracted; (3) the #1 lever's real risk is a config-specific g128 fold-timing bug on (64,256,256), not a generic bit-exact check. The biggest completeness gaps are MoE/batched-M (excluded despite being the real deployment), sm80≠sm86 tile/stage divergence (collapsed into one arch's smem math), and the cp.async L2-eviction-hint angle (the achievable version of the rejected L2-persistence lever).

Files inspected: `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin_template.h`, `marlin.cu`, `marlin_mma.h`, `generate_kernels.py`.