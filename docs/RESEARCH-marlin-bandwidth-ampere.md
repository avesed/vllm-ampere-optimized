# famp-marlin Bandwidth-Utilization — Research & Verdict (2026-06-27)

> ultracode run (domain=bw, 45 agents, 8 survivors). Builds on WF0. Bottom line: the dense kernel is
> bandwidth-TAPPED-OUT; the 19%% DRAM-SOL gap is BARRIER-bound (one __syncthreads/K-tile, stages-independent),
> NOT recoverable waste. Real wins = recipe g32->g128 (~2%% decode) + OUT-of-Marlin (MTP, lm_head->W4A16, fp8-KV).
> Doc correction: A/scales loads are .cg (NOT .ca as WF0 said); only per-token a_scale + int8 C-reduce are .ca.

## Roadmap

Code anchors confirmed. Key facts verified: `s_gl_stride = prob_n/(is_8bit_scale?16:8)` (line 562), fold gate `group_blocks==2 || k==1` (line 1374), `s_type_id = c_type_id` then fp8 override only for kFE2M1f/kFE4M3fn (lines 586-597), weight load `.cg` / per-token scale `.ca` (marlin.cuh:153/163 vs 114/127/140), `int stages = 4` flat (marlin.cu:396), and the codegen filter forcing `(64,256)` for m_blocks>1 (generate_kernels.py). All claims in the MAP/VERDICTS hold against source. Writing the roadmap now.

# famp-marlin Bandwidth-Utilization / Memory-Subsystem Efficiency — Ampere Optimization Roadmap

Scope: the vendored **dense** Marlin GEMM (`flashampere/marlin/csrc/marlin/`) on sm80 (A100) + sm86 (GA10x / 2×3090). Flagship = dense 27B **W4A8** (int4 weight + int8 act, sym AWQ kU4 / GPTQ kU4B8). Builds on workflow-0's 13 NO-GOs (DRAM-SOL decode wall, occupancy smem-pinned, .cg already streaming, int8-GEMM tapped-out, W4A4 NO-GO, L2-persistence NO-GO). This frontier asks: **is the 19% SOL gap recoverable, and which byte-cuts are real?**

---

## 1. Executive Summary

**The honest bottom line for this frontier: the dense Marlin kernel itself is bandwidth-tapped-out. Every byte-stream is already coalesced, .cg-streamed, and byte-tight; the 19% decode DRAM-SOL gap is barrier-bound, not request-starvation, so it is NOT recoverable by any in-kernel change. The only real wins are (a) ONE in-kernel-adjacent recipe lever and (b) three out-of-Marlin byte amortizers on the same bus.**

Highest-confidence findings, grounded:

- **The 19% SOL gap is structural, not waste.** ncu (workflow-0 / int8-path roadmap) measured M=1 decode at 81% DRAM-SOL / 10.6% IMMA with the dominant stall = **barrier** (not long_scoreboard). The main loop fires exactly one `__syncthreads` per K-tile (`marlin_template.h` wait_for_stage), a count **independent of `stages`**, so deeper pipelines cannot move it. The int8 8-row tile fix — a kernel-internal change to this exact M=1 path — measured **+0 decode**. Three independent levers that attack the gap (deeper stages, ldmatrix A-gather, steady-state re-measure) all collapse to ~0.

- **The weight stream is irreducible and already optimal.** Repack output is byte-exact zero-padding (`gptq_marlin_repack.cu:313-315`); the GEMM reads it as fully-coalesced 16B `cp.async.cg` per thread (sector-optimal on GA102's 32-sector L2 lines). No tile pruned, no tail over-read on any served (64-aligned) shape. W4A4 is a confirmed NO-GO.

- **The ONE real in-kernel-adjacent byte lever = recipe g32→g128.** `s_gl_stride = prob_n/8` (bf16 scales, `marlin_template.h:562`), fetched at group cadence. g32 scale stream ≈12.5% of weight bytes → g128 ≈3.1% (4× cut). The g128 kernels are **already built** (`generate_kernels.py` group_blocks=[-1,2,4,8]); zero kernel edits. Realistic **~2% decode**. Gated on (i) bit-exact validation of the int8 coarse-group fold (`group_blocks==2 || k==1`, line 1374) and (ii) a GSM8K/MMLU-Pro quality gate — and complicated by the fact that **g32 is the team's deliberate quality requant** (g128 is the upstream preset the team moved *away* from), so the quality gate likely fails uniform g128 and forces a per-layer mix (g32 on the L0 down_proj 50000× outlier).

- **The real throughput is out-of-Marlin, on the same 936 GB/s bus.** Since decode is a bytes-per-token wall the kernel can't shrink, the wins come from removing *other* bf16 consumers and emitting more tokens per weight pass: **MTP spec-decode** (amortize the whole int4 stream over accepted tokens, the #1 decode lever), **lm_head→W4A16/W8A16** (cut the ~2GB/step 248k-vocab bf16 logits read), and **fp8-E4M3 KV** (capacity-only enabler at ≥24-32k).

**Tapped-out (do not work): in-kernel tiling/stages/gather/dequant/occupancy/IMMA/coalescing, fp8 weight-scales (structurally unreachable for int4 weights), zp-stream (sym flagship carries zero), L2 persistence/streaming windows (weight ≫ L2; .cg already minimizes pollution), MoE per-expert N-padding (no per-expert N; the dense path isn't even the MoE GEMM), large-batch tile re-selection (the claimed thread_n=64 fallback doesn't occur — N%128==0 already selects a 128-wide tile).**

---

## 2. Tiered Lever Table (ALL levers)

| Lever | Phase | Impact | Effort | Risk | Rationale / prior finding |
|---|---|---|---|---|---|
| **GO / CONDITIONAL** ||||||
| **#1 Recipe g32→g128** (4× scale-stream cut) | both (decode-dom) | **~2% decode**, ~0 prefill | low (recipe) + med (validate) | correctness (int8 fold k==1) + accuracy (coarser group) | WF0's sole dense lever. Kernels already built. *Caveat:* g128 is the upstream preset the team requants AWAY from for quality → likely needs per-layer mix, not uniform g128. |
| **MTP spec-decode (K=2)** | decode | **~1.4-1.8× best-case** (flagship 27B/35B); ~+5-19% realistic tp2-prose | medium | accept collapse if MTP head zeroed; long-ctx cliff w/o patch-A; XOR fp8-KV | Out-of-Marlin. The corpus-blessed AMORTIZE escape from the byte-wall. GEMM unchanged (rides the already-shipped small-batch/8-row tile at M=1+K). |
| **lm_head→W4A16 / W8A16** | decode | **+2-7%** (~2-4% on tp2; high end single-card) + ~1.5GB VRAM | medium | tie-embeddings corruption; argmax regression; MTP shared-head aliasing | Out-of-Marlin. ~2GB bf16 read/step at 936GB/s. *Code reality:* under `VLLM_MARLIN_INPUT_DTYPE=int8` the head runs W4A8 not W4A16 — tighter quality gate; ship W8A16 first. |
| **fp8-E4M3 KV cache** | both (long-ctx) | **+7-14% @≥24-32k; ~0 ≤16k** (capacity, not speed) | low (config) | FA2→Triton routing is STRUCTURAL on Ampere → forfeits patch-A MTP verify; EITHER/OR with MTP long-ctx | Out-of-Marlin. `flash_attn.py:189-192` Hopper-gates fp8-KV → Ampere falls to Triton. Deployment-aware XOR, not a stack. |
| **Doc-correct the .ca/.cg split** | n/a | 0 perf (hygiene) | low | none | Corrects WF0 doc line 13 (A/scales are .cg, NOT .ca; only per-token a_scale + int8 C-reduce are .ca). Prevents a future regression. |
| **Confirm zero tail over-read (comment only)** | n/a | 0 perf (doc) | low | none | The `prob_n%thread_n==0` invariant is ALREADY unconditionally asserted (`marlin.cu` is_valid_config + the post-resolve TORCH_CHECK). Drop the proposed redundant assert; ship the comment + MoE-shard table only. |
| **Keep sym kU4B8 flagship (no zp)** | both | ~0 (do-not-regress) | low | none | Sym carries ZERO zp bytes (`has_zp` false). Avoid migrating to asym kU4 (adds ~3.1% zp side-stream, ~0 accuracy gain per memory). Already shipped sym. |
| **NO-GO (dropped — prior finding contradicted)** ||||||
| Deepen pipeline stages 4→6/8 | decode | ~0 | medium | spill, smem | Contradicts DRAM-SOL wall + barrier-bound stall (NOT long_scoreboard) + int8-8row=+0. Barrier count is stages-independent. |
| ldmatrix A-gather replacement | decode | ~0 (likely negative) | high | silent decode corruption (transposed m16n8k32 operand) | Contradicts int8-8row=+0 + the gather is a double-buffered PREFETCH decoupled from the mma (not on the critical leg). Also operand-role + swizzle make it infeasible. |
| ncu-re-decompose 81% SOL | n/a | 0 (diagnostic) | low | none | Premise ("never decomposed") is false — P0-A ran 2026-06-16 (barrier-bound). Fold into the g128 ncu session if at all. |
| fp8 weight-scales (is_8bit path) | decode | ~0.8pt on g128 | high | new quant scheme + fp8 group-scale accuracy | Structurally UNREACHABLE: `s_type_id=c_type_id` for int4 weights (fp8 only for NVFP4/MXFP8 Hopper/Blackwell); static_assert forces bf16; no source format (AWQ/GPTQ store bf16). |
| Per-column g128+g32 residual scale | both | ~0 (= g128) | high | composes with int32-fold incorrectly | Weight scale is folded as INTEGER into int32 accum (line 1378-1382), NOT a bf16 epilogue multiply. Dominated by per-layer-mix. |
| Drop (64,256,256)→add (128,128,256) large tile | prefill | ~0 (negative) | medium | smem, build size | Premise false: N%128==0 already selects the 128-wide `{64,128,128}`; thread_n=64 only fires on N%128≠0 (a deliberate grid-fill win). Total weight bytes = N·K regardless of tile width. |
| MoE per-expert N-pad to 256 | both | 0 | high | garbage cols into next layer | Premise false: MoE weights are ONE stacked tensor w/ shared N; no per-expert independent N. Also this csrc is the DENSE path, not the MoE GEMM. moe_intermediate_size already 256-aligned. |
| L2 streaming-window on weight | both | ~0 | medium | stream-attr leak | Weight already .cg (no normal-priority L2 install to downgrade); 11-115MB ≫ 6MB L2 cycles cache regardless of priority. Same physics as WF0 persistence NO-GO. |
| L2 persisting-window on scales (sm80) | both | ~0 | medium | global state, sm86 negative | Scales are read-ONCE-streaming (`.cg`, monotonic s_gl_rd) — no intra-kernel re-read for L2 to convert to a hit. a_scales already `.ca` + 4 bytes at M=1. |

---

## 3. The #1 Lever — Fully Specified: Recipe g32→g128

The single real in-kernel-adjacent byte lever. **No kernel-source edit** — the g128 kernels are already instantiated.

### Files + regions

- **Kernel (read-only, the validation target):** `flashampere/marlin/csrc/marlin/marlin_template.h`
  - `s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);` (**line 562**) — the bf16 scale stride; unchanged by g128, but read 4× less often.
  - Scale group-cadence fetch: `if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0)` (**~line 885**) — at g128 (group_blocks=8) the cp_async4 of scales fires 4× less often than g32 (group_blocks=2).
  - **The hard validation gate** — int8 coarse-group fold: `if (group_blocks == 2 || k == 1)` (**line 1374**), int32 fold via `reinterpret_cast<uint16_t*>(&frag_s...)` (**lines 1376-1382**). For g128 this fires only at `k==1` (tile-end). Static trace says correct (`b_sh_wr_iters==2` for all 4 W4A8 configs → k∈{0,1}, so k==1 is always last; large-batch tkb=4 folds twice with the same persisted scale = algebraically identical), **but it must be executed bit-exact** because the m_block_size_8 transposed branch has a multi-bug history.
- **Codegen (confirms kernels exist):** `flashampere/marlin/csrc/marlin/generate_kernels.py` — `group_blocks: [-1, 2, 4, 8]` for both W4A8 configs (kU4, kU4B8). g128 = group_blocks=8 is already emitted. `s_type = quant_config.get("s_type", c_type)` keeps scales bf16.
- **The actual change (NOT in csrc):** the per-model **quant export config** — set `group_size=128` (and, per the quality gate, keep `group_size=32` on the L0 down_proj outlier), plus the **served recipe** pointing at the g128 checkpoint.

### Concrete change

1. Re-quantize the 27B dense W4A8 flagship at `group_size=128` (AWQ kU4 / GPTQ kU4B8), routing through the **same int8-act export path** (the int8 path applies the weight scale as an integer fold; the g128 scales must be the int-fold per-group scales, not weight-only). Keep g32 on the outlier down_proj if the quality gate demands it.
2. Set the served recipe to the g128 checkpoint. No code edit — Marlin selects group_blocks per-GEMM from the packed scale shape.

### Build steps

No GPU build for the kernel (the g128 templates already exist; this environment is read-only/no-GPU per the task constraint). The only "build" is the offline re-quantization + the validation harness build, both run on the sandbox box:
1. `quantize/requant_awqgptq_g32.py`-style tooling, but targeting g128 (the inverse of the current g32 requant) through the int8-act export path.
2. Build the GEMM correctness probe (drives `marlin.cu` against the g128 kernels vs an fp32 dequant reference) — this is the read-only-undecidable gate that must run before serving.

### sm86 A/B benchmark plan (serve + OpenAI API)

**Gate order (each must pass before the next):**

1. **Correctness (HARD GATE, first):** bit-exact GEMM probe of the g128 int8 fold vs fp32 dequant reference, on REAL (non-all-ones) activations, for BOTH configs — decode `{128,128,256}` (thread_k_blocks=8, fold once-per-128-group) and large-batch `{64,256,256}` (thread_k_blocks=4, fold per-64 across 2 pipes) — AND the m_block_size_8 8-row `__shfl`-permute branch (lines 1384-1413). Require bit-exact / cos>0.9999. Check int32-accum saturation over the 8-k-block span.
2. **Accuracy (HARD GATE):** GSM8K + MMLU-Pro on 27B W4A8 g128 vs the validated g32 baseline. Because g32 is the team's deliberate quality recipe, expect uniform g128 to regress → fall back to per-layer mix (g128 everywhere except the L0 down_proj kept at g32) and re-clear within run-to-run noise.
3. **Perf (the win):** `vllm serve` + HTTP client (OpenAI API, **never offline `LLM()`**); decode tok/s via `bench_decode_clean.py` (manual `t(N)-t(1)` timing, **prefix-cache OFF, NOT get_metrics TPOT** which is ~2× inflated), on the large-N projections (qkv / gate_up / o) where the scale fraction is largest. Confirm ~2% clears noise.
4. **ncu A/B (confirm the mechanism):** `dram__bytes_read` at M=1 drops by the ~9.4pt scale-stream delta; overall decode `dram__throughput` unchanged-or-down. Run the standalone decode-GEMM micro-bench (NOT full vLLM under `--target-processes all`, which hangs), in a DooD sibling container with `--cap-add=SYS_ADMIN`.

---

## 4. Do-Together Bundles + Sequence

**Bundle A — Phase-0 ncu gate (do FIRST, unlocks/caps everything):** one ncu session on single-card 9B-W4A8 b1 decode + its following attention kernel. Capture `dram__throughput.pct_of_peak`, `smsp__...long_scoreboard`, `l1tex__data_bank_conflicts...`, and the steady-vs-prologue split. This single session **closes 4 NO-GO levers definitively** (deeper stages, ldmatrix gather, SOL re-decompose, L2 windows) and **provides the dram__bytes A/B baseline for the g128 lever**. Per prior data the answer is already known (barrier-bound, ~0 recoverable in-kernel), so this is confirmation + g128 baseline, not exploration.

**Bundle B — the real decode stack (out-of-Marlin, these COMPOUND):**
- MTP K=2 + lm_head→W8A16/W4A16 + g128 recipe.
- These stack cleanly: MTP amortizes the int4 weight stream over (1+accept) tokens; lm_head-quant cuts the ~2GB/step logits read that MTP's draft reads (1+K)× (so it compounds *more* under MTP); g128 trims the scale side-stream the GEMM still pays each pass.
- Shared prerequisite for MTP: `re:.*mtp.*` in quant-ignore + `mtp.fc` bf16 (else 0% accept), graft head for AWQ/GPTQ, and **cherry-pick patch-A fwd_kvcache** (current dev-dllm branch lacks it) for long-ctx verify.

**Bundle C — long-ctx capacity (deployment-aware XOR):** fp8-E4M3 KV is mutually exclusive on Ampere with MTP's patch-A FA2 verify (fp8-KV forces Triton). Decide per deployment: capacity-bound long-ctx (fp8-KV-Triton, no patch-A MTP) XOR latency/short-ctx (bf16-KV + patch-A MTP). Do NOT ship both on the same serve.

**Bundle D — documentation hygiene (free, do alongside Bundle A):** correct the .ca/.cg split in `docs/RESEARCH-marlin-ampere-opt.md:13` + memory `project_marlin_ampere_opt_verdict.md:26`; add the comment at `marlin_template.h:379`; close the MoE-per-expert open question with a static N-alignment table.

**Recommended sequence:**
1. Bundle A (Phase-0 ncu) + Bundle D (docs) — one session, free, caps the in-kernel frontier and baselines g128.
2. g128 correctness probe → accuracy gate → per-layer-mix fallback (the only in-kernel-adjacent ~2%).
3. lm_head→W8A16 (calib-free, near-lossless, in-tree per-output-channel) → W4A16 follow-up if accuracy clears.
4. MTP K=2 (cherry-pick patch-A first; measure accept_len with the `0<accept_len<K` collapse canary).
5. fp8-KV only if a deployment is capacity-bound at ≥24-32k AND not running patch-A MTP.

**Honest tapped-out verdict:** the dense Marlin GEMM is a closed bandwidth frontier — the 19% SOL gap is barrier-bound and irrecoverable in-kernel, the weight bytes are floor-irreducible, and every side-stream is already coalesced/.cg/byte-tight. The only in-kernel-adjacent lever is the ~2% g128 recipe (with a real quality gate that may force a per-layer mix). All meaningful throughput is out-of-Marlin: amortize the stream (MTP), remove the other bf16 consumer (lm_head-quant), or buy long-ctx capacity (fp8-KV).

---

## Critique

I have enough to characterize the int8 fold. The fold gate `group_blocks == 2 || k == 1` (line 1374) and `b_sh_wr_iters` is the inner-k count per pipe-stage. With the decode tile `{128,128,256}` thread_k_blocks=8, b_sh_stage = b_sh_stride*8/2, so b_sh_wr_iters... the roadmap claims b_sh_wr_iters==2 for all 4 W4A8 configs. That means `k` ranges {0,1} and `k==1` is the fold point. But with group_blocks=8 (g128), the scale is constant across the full 128-element thread_k span, so the fold at k==1 per pipe accumulates only 2 k-blocks of int32 into frag_c_tmp before fold — NOT the full 8-block group. This means g128's int32 intermediate spans are NOT deeper than g32's at the fold granularity; the saturation concern the roadmap raises is largely moot (fold cadence is per-pipe k==1, identical). That's a real over-cautious gate, but minor.

I have verified all load-bearing claims. Let me check one more: the `frag_c_tmp` second-dim is `is_a_8bit ? 2 : 4`, confirming the int8 path uses a 2-wide C_tmp (relevant to the 8-row branch correctness). I have what I need.

Here is my critique.

---

## Verified against source (claims that HOLD)
- `.cg` weight/A load, `.ca` for ca-variants: `marlin.cuh:153` (`cp_async4_pred` = `.cg`), `:163` (`cp_async4` = `.cg`), vs `:114/127/142` (`cp_async{1,2,4}_ca_pred` = `.ca`). The doc-correction is correct in spirit, but cite **142** not 140 for `cp_async4_ca`.
- `s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8)` — `marlin_template.h:562`. Holds.
- int8 fold gate `group_blocks == 2 || k == 1` — `marlin_template.h:1374`, int32 integer fold at `:1408-1410`. Holds.
- `int stages = 4` flat (sm75→2 override) — `marlin.cu:396-398`. Holds.
- g128 kernels emitted: `generate_kernels.py:77,84,92,100` group_blocks include 8. Holds.
- sym carries zero zp: `has_zp` path gated, `zp_sh_stage = has_zp ? ... : 0` (`marlin_template.h:589`). Holds.

## The load-bearing ERROR — the thread_n=64 NO-GO is wrong on mechanism AND misses a live decode lever
The roadmap asserts: *"thread_n=64 only fires on N%128≠0 (a deliberate grid-fill win); N%128==0 already selects {64,128,128}."* **This is false.** `marlin.cu:448-459` unconditionally overrides to `{128, 64, 128}` (thread_n=64) whenever `prob_n/thread_n * div_ceil(prob_m, tm*16) * 4 <= sms` — i.e. whenever the wide-tile grid **under-fills the SMs**, which at M=1 happens for *small-N, perfectly-128-divisible* projections (e.g. a 4096-N o_proj on 82 SMs). thread_n=64 is therefore an *occupancy/grid-fill lever already firing at decode*, not a divisibility fallback. The roadmap reached the right "don't add tiles" conclusion via a wrong model, and in doing so **missed an existing, tunable bandwidth lever**: the `*4 <= sms` heuristic threshold is a magic constant that decides 64-wide vs 128-wide at decode, and it was never measured/swept. That is a genuine gap the roadmap should have flagged.

## Missing levers
- **C_tmp / partial-sum traffic (`use_atomic_add` + split-K reduce).** `marlin.cu:412-419` sets `max_par` and there is a `C_tmp` global buffer + `use_atomic_add` path (`marlin_mm` signature, `marlin.cu:323`). At M=1 with split-K (small prob_m, large prob_k), partial int32 C tiles are written to and re-read from global `C_tmp` — extra DRAM traffic *on the same 936 GB/s bus the roadmap calls saturated*. The roadmap never inventories the reduce/atomic path; whether decode shapes trigger split-K (and pay this) is unverified and could be a real byte sink or a free win (force `use_atomic_add` to skip the C_tmp round-trip). This is the most defensible missing in-kernel item.
- **`get_kernel_cache_size` / smem-fit as the real tile selector.** Tile choice flows through `is_valid_config`→`get_kernel_cache_size <= max_shared_mem` (`marlin.cu:244-248`). The roadmap treats tiling as closed but never checks whether the g128 scale-smem reduction (`s_sh_stage` shrinks) *changes which tiles pass the smem gate* — g128 could unlock a tile g32 can't fit, a second-order effect of the #1 lever that's entirely unanalyzed.

## Over-optimistic / under-justified
- **g128 "scale stream ≈12.5% of weight bytes → 4× cut → ~2% decode."** The 12.5% is the static byte ratio, but scales are read at *group cadence* through smem with their own pipeline; at M=1 the kernel is barrier-bound (the roadmap's own finding), so cutting a side-stream that isn't on the critical-latency leg may yield **< the byte-proportional ~2%**, possibly ~0 — the same logic that made int8-8row "+0". The roadmap inconsistently invokes "barrier-bound, byte-cuts don't help" for the NO-GOs but then credits a byte-cut with ~2%. Either DRAM bytes gate decode (then re-examine the NO-GOs) or barriers do (then g128 is also suspect). This tension is unresolved.
- **int32-accum saturation gate is over-cautious.** With `b_sh_wr_iters==2`, the fold fires at `k==1` *every pipe* (`:1374`), so g128 accumulates the same 2 k-blocks into `frag_c_tmp` before folding as g32 does — the int32 intermediate span does **not** deepen with g128. The roadmap's "check saturation over the 8-k-block span" gate targets a span that doesn't exist; the real (already-present, group-size-independent) saturation risk is per-pipe and identical to shipped g32. Keep the bit-exact probe, drop the "g128 deepens accumulation" framing.

## Prior finding misapplied
The roadmap repeatedly leans on "barrier-bound, not DRAM-starved" to kill stages/gather, yet builds its one GO (g128) and its whole out-of-Marlin thesis (MTP/lm_head) on "decode is a bytes-per-token DRAM wall." Both can be true (barrier-bound *within* the kernel's steady state, DRAM-bound *across* the token), but the roadmap never states which regime g128 lives in — and that determines whether the ~2% is real or another "+0."

## Files
- `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin.cu` (tile selector :265-313, thread_n=64 override :448-459, stages :396, C_tmp/atomic :323/412)
- `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin_template.h` (scale stride :562, int8 fold :1374-1434, frag_c_tmp :771)
- `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin.cuh` (.cg :153/163, .ca :114/127/142)
- `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/generate_kernels.py` (codegen filter :155-163, group_blocks :77-100)