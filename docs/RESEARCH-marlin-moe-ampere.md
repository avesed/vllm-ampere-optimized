# famp-marlin MoE / batched-M Ampere — Research & Verdict (2026-06-27)

> TWO convergent ultracode runs (49 agents each; survivors 6 + 9). Both agree on the structural verdict;
> their adversarial critiques CROSS-CORRECTED each other and surfaced the lever both roadmaps missed.

## Synthesized bottom line (after reconciling both runs + both critiques)
- **No flagship MoE kernel win.** Per-expert M at decode = B*top_k/E < 1 (35B-A3B = **E=128**, top_k=8) -> each
  active expert streams its full int4 slab for ~1 token = the dense-M=1 DRAM-SOL wall, replicated per expert.
- **DO-NOT-VENDOR the MoE marlin (scope GO).** famp owns ONLY the dense marlin; the MoE expert-GEMM is the
  unvendored upstream `csrc/moe/marlin_moe_wna16/` (carried via patch 0005, int8-act per-expert scale). It is
  ALREADY more Ampere-tuned than dense: adaptive blocks_per_sm via cudaFuncGetAttributes occupancy
  (ops.cu:266-334) + part1-DP/part2-StreamK grid-fill, vs dense `return {1,th_config}` (marlin.cu:309).
  Vendoring = 0 perf, pure ownership. (Dense blocks_per_sm>1 IS a real C_tmp-OOB — see correction.)
- **THE one genuinely untapped MoE lever (both roadmaps MISSED it; the critiques found it): inter-GEMM +
  epilogue BANDWIDTH.** (a) The marlin-MoE epilogue topk-reduce is `torch.sum(moe_output.view(-1,topk,K),dim=1)`
  (marlin_moe.py:393-396) — a full `[M*top_k, K]` bf16 materialize-then-reduce DRAM round-trip; fuse the topk
  reduction into the GEMM2 epilogue. (b) The GEMM1->silu/mul->int8-requant->GEMM2 intermediate `[tokens,inter]`
  round-trips HBM; fuse act+requant into the GEMM1 epilogue. This inter-GEMM traffic — NOT the tile shape — is
  the likely source of the int8-MoE -4.4% decode regression (the int8 8-row tile ALREADY EXISTS in the MoE
  template: m_block_size_8 is pervasive + composes with is_a_8bit — so the "port 8-row" NO-GO was a false premise).
- **Real-but-small:** batched-M tokens-per-expert amortization (larger moe_block_size / capacity batching ->
  more rows per int4 slab read — the actual "batched-M crossover" the scope asked for, prefill/batched only);
  g32->g128 on MoE expert scales (~1-2% batched/prefill, recipe-only, bundle w/ dense g128 fold validation);
  phantom-expert masking (realistically ~0 at deployed b1 — routing temporal-locality + most experts already
  active at top_k=8/E=128; a 1-line runner input_ids-zero is a cheap correctness cleanup, NOT a #1 lever; note
  an EXISTING ungated kernel-native skip `if(tmp_expert_id==-1)continue` at ops.cu:108 + ignore_invalid_experts).
- **CORRECTION to docs/RESEARCH-marlin-ampere-opt.md:** the dense C_tmp-OOB on blocks_per_sm>1 is **REAL** (C_tmp
  sized `sms` slots, blockIdx.x over-indexes when gridDim=sms*bps; marlin.cu:689-693 / marlin_template.h:420,1691).
  WF0's critique that RETRACTED the OOB was WRONG. Dense adaptive-occupancy stays NO-GO anyway (smem-pinned 1blk/SM).

---

## RUN A roadmap

All claims verified:
- L2's NO-GO is confirmed: the `expert_id == -1` skip (lines 132/230/279/310) is **gated behind `has_expert_map`**; the only unconditional guard is `expert_id >= num_experts` (line 126/303). So a raw `-1` in topk_ids without expert_map is OOB, not a clean drop.
- kS8 (int8-act) configs use `thread_m_blocks = [1,2,3,4]` (no 0.5), while W4A16/bf16 configs use `[0.5, 1, 2, 3, 4]` — confirming int8-act MoE has no 8-row tile.

I have everything needed. Writing the roadmap now.

# famp-marlin MoE Expert-GEMM + Batched-M Ampere Optimization Roadmap

**Scope:** Qwen3.6-35B-A3B MoE (E=256, top_k=8, expert-only W4A16/W8A8) expert-GEMM decode/prefill + the dense W4A8/W4A16 batched-M crossover, on sm80 (A100) and sm86 (GA10x: RTX 3090/A10/A40). Builds on workflow-0's 13 dense-M=1 NO-GOs (DRAM-SOL wall, occupancy smem-pinned, cp.async.cg already used, fp16-accum sm75-only, int8-GEMM tapped-out, W4A4 NO-GO, L2-persist NO-GO). Read + reason only; no GPU builds run. Code anchors verified against `/home/trevor/vllm-ampere-flashampere/vllm` and `/home/trevor/vllm-ampere-optimized/flashampere`.

---

## 1. Executive Summary

**The honest bottom line for this frontier: there is no flagship MoE kernel win.** Every per-tile / per-expert-GEMM compute lever is bounded ~0 at the deployed regime by the same DRAM-SOL bandwidth wall workflow-0 found for dense M=1, replicated *per active expert*. The MoE decode GEMM at M=1 streams ~8 distinct experts' int4 weight slabs once each (per-expert M = B·top_k/E = B/32, i.e. ~0.03 real tokens/expert at b1, ~8 even at B=256), so it is **pure weight-streaming with zero arithmetic-intensity gain from batching at decode**. This is the structural reason the established int8 8-row dense tile fix measured decode **+0**, and the same physics carries to MoE.

**Highest-confidence findings (grounded):**

1. **famp owns only the DENSE marlin.** Verified: `flashampere/marlin/csrc/marlin/` has no `*moe*` source; the MoE expert-GEMM is the **stock-upstream** `moe_wna16_marlin_gemm` in the fork's `csrc/moe/marlin_moe_wna16/`. Any MoE kernel work is a fork/upstream change, not a famp-vendored-kernel edit.

2. **DO-NOT-VENDOR the MoE marlin (GO).** The strongest scope conclusion: the upstream MoE `determine_exec_config` (`ops.cu:266-334`) is **strictly more SM-aware** than the dense one — it computes real occupancy (`cudaFuncGetAttributes` → `allow_count = min(reg_budget, smem_budget)`, capped 4/2, with grid-fill backoff), whereas the dense path hardcodes `blocks_per_sm=1` (`marlin.cu:309`). Vendoring would replicate the `moe_align`/`sorted_token_ids`/`expert_ids`/per-expert-`global_scale` machinery (883-line `ops.cu` + 795-line `moe_align_sum_kernels.cu` + 2264-line template) for **zero scheduling win** and a likely regression if the dense template's naive exec_config were reused. The one in-kernel claim to correct: the part1/part2 stream-K tail-wave fold is NOT MoE-exclusive — the dense vendored template has it identically.

3. **The ONE real byte-moving MoE-decode lever is phantom-expert masking, and it is a runner/Python change — proving vendoring is unnecessary for it (CONDITIONAL).** Verified asymmetry: `gpu_model_runner.py:3492` slices `input_ids.gpu[:num_input_tokens]` (stale pad tail) while `:3502-3503` zeroes the positions tail. Stale ids → phantom routing → extra int4 expert tiles streamed. Fix = zero the `input_ids` tail (1 line, mirror positions). **Bounded + UNPROVEN**: exactly 0 at b1 (no padding) and ~0 at large batch (experts saturated); only mid-batch padded buckets pay, and stale-id/real-routing overlap may erase it. MUST gate on a `RoutedExpertsCapturer` distinct-active-experts probe before claiming. Deployed MoE serving (PP2 single-stream) is b1 → exactly 0 there.

**Per-expert-M regime — bandwidth vs compute:** The crossover to compute-bound (where deeper m-tiles/occupancy could help) requires per-expert M ≥ 8, i.e. aggregate B ≥ ~256-512 PREFILL — a workload the fork doesn't serve. At decode and small-batch the MoE GEMM never reaches `thread_m_blocks > 1`; the int8-act floor pins block_size_m ≥ 16 (`marlin_moe.py:338-339`, verified), and the int8-act path is itself decode-NEGATIVE (-4.4%), shipped opt-in default-off.

**What is tapped-out vs real:**
- **TAPPED-OUT:** int8-act MoE GEMM (68% IMMA, +1.7% over W8A8, tiles hardcoded, no tuned-config lever); the MoE 8-row int8 tile port (would inherit decode +0); block_size_m threshold/floor tuning (selection loop already tracks per-expert M; floor only bites at B≤128 where the regime is bandwidth-bound); MoE routing kernels (topk/align/sort/sum — integer/index work off the weight stream, captured in the full decode cudagraph).
- **REAL (small, gated):** phantom-expert masking (mid-batch valleys only, probe-gated); `moe_sum` topk=8 specialization (sub-1%, a tidy-efficiency cleanup, not a tok/s mover); the recipe-level g32→g128 on expert scales (same as the surviving dense lever, but more diluted by bf16-kept shared_expert/router/GDN/lm_head).

**The 19% SOL gap (dense, carried from WF0) is NOT recoverable here.** The MoE per-expert GEMM sits on the *same* wall, and the byte-cuts that are real (phantom-expert tiles) only exist in a narrow mid-batch valley the deployment doesn't occupy.

---

## 2. Tiered Lever Table (all levers)

| # | Lever | Phase | Impact | Effort | Risk | Tier / Rationale |
|---|-------|-------|--------|--------|------|------------------|
| V8 | **DO-NOT-VENDOR the MoE marlin** (scope verdict; MoE occupancy is upstream-owned and ahead of dense) | both | 0 tok/s (avoids regression + large lift) | low | low | **GO** — code-confirmed: MoE `ops.cu:266-334` computes real occupancy + grid-fill backoff; dense `marlin.cu:309` hardcodes `blocks_per_sm=1`. Correct one nit: stream-K tail-fold is shared, not MoE-exclusive. |
| V11 | **Routing-overhead is a near-dead lens** (record + optional `num_experts` startup assert) | both | ~0 (prevents wasted effort) | low | none | **GO** (as a guardrail) — gather fused into GEMM smem; topk-256 hits fused case; topk-weight fused into GEMM2; align/sort/sum captured in full decode cudagraph. Famp does NOT vendor a MoE marlin (corrects citation). |
| L1 | **Zero the cudagraph-padded `input_ids` tail** (phantom-expert masking at source) | decode | bounded, mid-batch valleys only; ~0 b1, ~0 large-batch; plausibly 0 after probe | low | low (output-neutral; spec/MTP read own buffer) | **CONDITIONAL** — mechanism verified (`:3492` vs `:3502-3503`); MUST run `RoutedExpertsCapturer` delta>0 probe first. Correct rationale: the zero runs *eagerly pre-replay*, not "inside the captured region." |
| V (moe-vendor) | **Phantom-masking is a runner lever, not a vendor reason** (implement upstream) | decode | same as L1 (bounded/unproven) | low | medium (spec-decode buffer check) | **CONDITIONAL** — reinforces DO-NOT-VENDOR; same probe gate as L1. |
| L3 (moe_sum) | **Specialize `moe_sum` for topk=8** (drop ATen `at::sum_out` fallback) | both | sub-1%; decode flat-to-noise, prefill a hair up | low | low (bf16-acc vs ATen fp32-acc; GSM8K check) | **CONDITIONAL** — code-confirmed (only cases 2/3/4 specialized; topk=8→`at::sum_out`); serving path DOES reach `ops.moe_sum`. Tidy efficiency, not a tok/s mover. Ship only if A/B cleanly positive + GSM8K within noise. |
| L2 | **Sentinel-mask padded topk_ids → -1** | decode | claimed ≥L1; actually = L1 at best | medium | **high (correctness)** | **NO-GO (dropped)** — contradicts `moe_align_sum_kernels.cu`: `==-1` skip is **gated behind `has_expert_map`** (lines 132/230/279/310). Single-node 35B has `expert_map=None` → `-1` is OOB atomicAdd, not a clean drop. Correct sentinel would be `≥num_experts`, voiding the "zero kernel change" claim. |
| L3 (capture) | **Right-size cudagraph capture buckets** to shrink pad valleys | decode | ~0 | low | low | **NO-GO (dropped)** — contradicts `project_scope_general_ampere` (capture-size is an EXCLUDED per-deploy knob) and `project_autotune_gpu_oc` (capture list = startup/VRAM, NOT throughput). Multiplier on a near-zero base; default ladder already has dense low buckets [1,2,4]. |
| V (dense-occ) | **Port MoE `allow_count` occupancy into DENSE `determine_exec_config`** (blocks_per_sm>1 for batched-M) | both | claimed +3-8%; ~0% | medium | medium (C_tmp/locks OOB) | **NO-GO (dropped)** — contradicts WF0 prefill IMMA~68%/waves=1.00 tensor-bound (can't hide non-bottleneck latency) AND register cap: at 256-thread {64,256,256}, `reg_budget` term → `allow_count=1` regardless of smem. Incomplete change would OOB C_tmp (`sms` slots) + locks via `locks_off=blockIdx.x`. |
| V (n-backoff) | **Generalize dense low-M `{128,64,128}` backoff to allow_count search** | both | claimed +2-5%; ~0% | low | low | **NO-GO (dropped)** — contradicts persistent-grid/stream-K model: `blocks = sms*blocks_per_sm` independent of n_tiles; under-fill already recruited via K-split. Depends on the (refuted) blocks_per_sm>1; degenerate fallback admits it "picks the same {128,64,128} it picks today." |
| L (8-row port) | **Port dense int8 8-row tile into MoE kS8 path** / lower block_size_m floor | prefill | claimed +3-6%; ~0-1% | high | high (4 transposed-epilogue bugs + per-expert scale) | **NO-GO (dropped)** — contradicts int8-act-MoE-tapped-out + dense int8 8-row=+0. Windows disjoint: floor only bites at M_total<256 (bandwidth-bound, int8 net-negative there); int8-act win is at M_total~8192 where padding is already 1.00×. Note: upstream v0.23.0 already ships much of the 8-row int8 MoE epilogue. |
| L (bsm tune) | **Tune block_size_m crossover threshold (0.9) per-model** | both | claimed +1-4%; ~0 / net-negative | low | low | **NO-GO (dropped)** — selection loop already picks block_size_m≈per-expert-M; forcing bsm=32 early DOUBLES padding for ~0 weight saving (ratio 0.996-1.000 at M=64-256); real amortization (0.697) only at M~512 where the existing 0.9 already crosses. For E=128 35B-A3B, bsm stays 8 for all decode M. |
| L (bsm select W4A16) | **Tune W4A16 block_size_m selection per-model (the "TODO tune")** | both | ~0 | low | low | **NO-GO (dropped)** — over-padding wastes only MMA on the DRAM-SOL decode wall (free); decode buckets b1..b64 already use the minimum compiled tile (8 for E=128 W4A16). No mid-decode headroom. |
| L (small_batch gate) | **Raise `small_batch_expert_mode` gate to cover 256 experts** (fuse 2-launch align) | decode | ~0; **not launchable** | low | n/a | **NO-GO (dropped)** — host launch cost already amortized by full decode cudagraph; AND smem at 256 experts = 258KB > sm86 ~99KB AND > sm80 163KB AND > 48KB default (no opt-in call on the non-LoRA path) → crashes on ALL Ampere. |
| L (quant fuse) | **Fuse per-token int8 act-quant into MoE marlin prologue** (kill Triton round-trip) | both | claimed "flips -4.4%→+"; ~0 / negative | medium | high | **NO-GO (dropped)** — contradicts measured int8-act decode root-cause (weight-bound, cudagraph already amortizes launch yet stays -4.4%). Option A premise false (`is_a_8bit` is a compile-time mutually-exclusive arm; no kernel loads bf16 + reads a_scales); per-token absmax is a K-axis reduction the K-sliced smem tile can't do. |
| L (int8 default-on) | **Re-validate + un-gate int8-act MoE as regime-auto default** | both | unlocks measured prefill +5-19% as default | medium | medium | **NO-GO (dropped)** — architecturally infeasible: int16 scale codes baked DESTRUCTIVELY at load time (`int_wna16.py:490-506`), dtype/kernel bound once; no per-phase swap on one weight set + one FULL graph. Regime-gating is also an excluded per-deploy knob. Coherence already validated (GSM8K 95.83%); prize already reachable via the env flag. |

---

## 3. The #1 Lever, Fully Specified

**L1 — Zero the cudagraph-padded `input_ids` tail (phantom-expert masking at the source).**

This is #1 because it is the **only verified byte-moving MoE-decode lever**, it is correctness-neutral, low-effort, no rebuild, sm80==sm86, and it concretely demonstrates the DO-NOT-VENDOR conclusion (the highest-value MoE-decode lever lives in the runner, not the kernel). It is CONDITIONAL — ship the probe first; adopt only if the probe shows a positive distinct-active-experts delta.

### Files + region

`/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py`, text-only branch, immediately after line 3492 (the `input_ids` slice), mirroring the positions-zero block at lines 3502-3503.

Current (verified):
```python
3492:            input_ids = self.input_ids.gpu[:num_input_tokens]
...
3502:            if num_input_tokens > num_scheduled_tokens:
3503:                self.positions[num_scheduled_tokens:num_input_tokens].zero_()
```

### Concrete change (one line, guarded, persistent-buffer)

Add right after line 3492:
```python
            if num_input_tokens > num_scheduled_tokens:
                self.input_ids.gpu[num_scheduled_tokens:num_input_tokens].zero_()
```

Critical correctness points (all verified):
- **Operate on the persistent `self.input_ids.gpu` buffer, not the `input_ids` slice view.** The write runs **eagerly in `_preprocess` before the cudagraph replay** (capture region is only `_model_forward`); it works exactly like `positions.zero_()` — pre-replay write to the persistent buffer that the captured graph then reads. (Correct the prior "inside the captured region" framing.)
- Guard with the same `num_input_tokens > num_scheduled_tokens` condition.
- `input_ids` is int32; `.zero_()` is valid; token-0 is a valid vocab embedding row.
- Output-neutral: padded rows are discarded downstream by `moe_sum`.
- Spec/MTP safe: draft reads `draft_token_ids = self.input_ids.gpu[logits_indices]` (`:2797`, explicit sampled positions) and target reads stay ≤ `num_scheduled_tokens` / `total_num_tokens = query_start_loc_cpu[-1]` (real token range); none touch the `[num_scheduled:num_input]` pad tail. Assert this at runtime and verify empirically (accept-len + GSM8K).

### Build steps

None for the change itself — pure Python, no kernel/rebuild, no CMake, identical on sm80/sm86. Deploy the modified `gpu_model_runner.py` into the running fork venv (overlay/copy; no `_C`/`_moe_C` recompile needed).

### sm86 A/B benchmark plan (35B-A3B, 2×3090 tp2 + shm8g)

**Step 0 — GO/NO-GO probe FIRST (mandatory, no build):**
Enable `RoutedExpertsCapturer` (`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`). On 35B-A3B (W4A16, tp2, `--shm-size 8g`, `--max-num-seqs 32`, expandable_segments), at fixed mid-batch cudagraph buckets (b2/b4/b8), count distinct-active-experts per MoE layer **with stale tail vs zeroed tail** (simulate the zero in eager). **Proceed only if delta > 0 at mid-batch and not overlap-erased.** If delta == 0 → NO-GO (the byte model overstates it; stale ids correlate with recent routing).

**Step 1 — serve + OpenAI-API, manual-time tok/s across the M-sweep (only if delta>0):**
```
vllm serve <35B-A3B-W4A16> --tensor-parallel-size 2 \
    --max-num-seqs 32 --gpu-memory-utilization 0.92
# env: VLLM_USE_V1=1, ipc=host / --shm-size 8g, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Drive via an HTTP streaming OpenAI-API client (`scripts/api_bench_client.py`), measuring client-side TTFT/TPOT. Report **decode tok/s = 1000/TPOT** (never ms), prefix-cache OFF, manual timing per the decode-bench artifact rule (NOT `get_metrics` TPOT). Sweep concurrency to hit the padded buckets b1/b2/b4/b8/b16/b32. Expected shape: exactly 0 at b1, possible low-single-digit % only at mid-batch padded buckets, ~0 at large batch.

**Step 2 — kernel confirmation:** nsys decode trace at b2/b4/b8 — confirm `moe_wna16_marlin_gemm` grid/block count (and active-expert count) drops with the zeroed tail; ncu DRAM bytes on the expert GEMM fall by the masked-expert tiles.

**Step 3 — correctness check:** output bit-identical on a fixed prompt set (padded rows discarded ⇒ must match unmasked-correct output); GSM8K + spec-decode accept-len unchanged with MTP on (the spec-buffer safety assertion, validated empirically).

**Adopt only if:** probe delta>0 AND Step-1 decode tok/s shows a non-noise gain at mid-batch AND Step-3 is clean. Default-on, correctness-neutral. The alternate `topk_ids→-1` masking (L2) is **NOT a valid fallback** as specified (OOB without expert_map); if a runner-side zero ever conflicts, the correct sentinel is `≥num_experts`, not `-1`.

---

## 4. Do-Together Bundles + Sequence

**Bundle A — "Close the MoE frontier honestly" (documentation + 1 probe, ~1 session, no build):**
- Record **DO-NOT-VENDOR (V8)** in `docs/RESEARCH-marlin-ampere-opt.md`: MoE occupancy/scheduling is upstream-owned and ahead of dense; the only future vendor reason is co-locating the patch-0005 int8 fix / an 8-row port, never scheduling. Correct the one nit (stream-K tail-fold is shared, not MoE-exclusive).
- Record **routing-overhead as a near-dead lens (V11)**; add the optional `num_experts` startup assert in `router/fused_topk_router.py` (defensive, zero-cost, prevents the slow 2-kernel softmax+topk fallback on a future non-{pow2, mult-of-64} expert count).
- Run the **one nsys decode trace** (b1 + a mid-batch valley, single-card/pp2 to avoid the all-reduce mask) summing topk_softmax+moe_align+count_and_sort+moe_sum as a fraction of TPOT — closes the routing lens with data.

**Bundle B — "The one real byte lever" (gated, after Bundle A's trace tooling is up):**
- **L1 probe** (`RoutedExpertsCapturer` masked-vs-unmasked at b2/b4/b8). This is the single GO/NO-GO that decides whether any phantom-masking work proceeds.
- If delta>0: land the L1 one-liner + the spec-decode safety assert; A/B per §3.
- `moe_sum` topk=8 specialization (L3-moe_sum) can ride along *only* if from-source `_moe_C` is built and an A/B shows clean positive or neutral-with-cleaner-kernel + GSM8K parity — otherwise hold as a tidy-efficiency note.

**Sequence:**
1. **Bundle A first** — it's free, it converts the bulk of this frontier (12 of 14 levers) into recorded NO-GOs/guardrails, and it produces the trace tooling Bundle B needs.
2. **Then Bundle B's L1 probe** — the only gate worth GPU time. If delta==0, the entire phantom-masking direction is closed and the frontier reduces to "no MoE kernel win; serve W4A16, keep int8-act opt-in for prefill/batched."
3. **Do NOT** start any kernel work (8-row port, occupancy port, vendoring, quant-fusion) — all are NO-GO against the DRAM-SOL wall + tapped-out int8-act + the load-time-baked-scale architecture.

**Net deliverable of this frontier:** documentation + one defensive assert + one probe-gated one-line runner change. No flagship MoE kernel optimization exists on Ampere for the deployed W4A16-decode regime — the per-expert GEMM is the dense M=1 bandwidth wall, replicated per expert.

## RUN A critique

I have verified all the load-bearing claims. Note the dense path already contains the `{128,64,128}` low-M backoff (line 451-461) AND grid-underfill detection (`prob_n/thread_n * ... * 4 <= sms`) — this directly refutes the V(n-backoff) lever's premise that this backoff doesn't exist. The roadmap correctly NO-GO'd it but with a different rationale. I have everything needed.

---

## Critique of the famp-marlin MoE + batched-M roadmap

### MISSING LEVERS (the roadmap's biggest gap is the reduction + the work-bounding mechanism)

**M1. The marlin MoE epilogue reduction is `torch.sum`, not a fused/specialized kernel — and that is the one un-examined byte-mover.** Verified at `marlin_moe.py:393-396`: the serving path does `if moe_sum is None: return torch.sum(moe_output.view(-1, topk, K), dim=1, out=output)`. The intermediate `moe_output` is `[M·topk, K]` bf16 — for 35B-A3B decode that is `M·8·K` bf16 elements written by the GEMM then re-read by `torch.sum`. This full materialize-then-reduce round-trip through DRAM is a *real* per-token byte cost the roadmap never costs out, and fusing the topk reduction into the marlin GEMM2 epilogue (write the summed output directly) would cut it. The roadmap instead spent a whole lever (L3-moe_sum) on the wrong target.

**M2. The roadmap's L3-moe_sum lever is aimed at dead code.** It claims "serving path DOES reach `ops.moe_sum`" — false. No caller passes `moe_sum=` to `fused_marlin_moe` (grep of `fused_moe/`, `quantization/` finds zero), so `moe_sum is None` always holds and the path is `torch.sum`, never the CUDA `ops.moe_sum` kernel whose topk=8 specialization the lever proposes. Specializing `ops.moe_sum` for topk=8 changes nothing on the marlin path. Either retarget to M1 (epilogue fusion / replacing `torch.sum`) or drop the lever entirely.

**M3. `ignore_invalid_experts` / `expert_ids == -1` is an EXISTING, ungated phantom-mask hook the roadmap overlooked.** Verified: the marlin MoE GEMM skips blocks via `if (tmp_expert_id == -1) continue;` at `ops.cu:108` — **unconditionally**, not gated by `has_expert_map`. And `moe_align_block_size(..., ignore_invalid_experts=...)` (`moe_align_block_size.py:16`) already exists to drop tokens. This means phantom-expert masking may be achievable by routing pad tokens to a sentinel expert_id and relying on the existing `-1` block-skip — a cleaner, kernel-native alternative to the L1 runner-zero that the roadmap never evaluated. It also weakens the "DO-NOT-VENDOR proven by L1 being runner-only" narrative.

### OVER-OPTIMISTIC / UNVERIFIED CLAIMS

**C1. "Dense hardcodes `blocks_per_sm=1` (marlin.cu:309)" is imprecise and undersells the dense path.** The dense `determine_exec_config` doesn't hardcode a constant — it `return {1, th_config}` at the first valid config (`marlin.cu:309`), but the surrounding `marlin_mm` *already supports* `blocks_per_sm > 1` (reads it at line 471, splits smem at 472-473). So blocks_per_sm>1 is a **supported-but-disabled** path, not absent infrastructure. This makes the V(dense-occ) NO-GO's "incomplete change would OOB C_tmp + locks" rationale shaky — the smem-split plumbing is present; the real reason it's inert is the register/smem `allow_count` math, which the roadmap should lean on instead.

**C2. The V(n-backoff) NO-GO has the right verdict but cites a non-existent absence.** The dense path *already contains* both the grid-underfill check (`prob_n/thread_n * div_ceil(...) * 4 <= sms`, line 449-451) and the `{128,64,128}` low-M backoff (line 452-461). The lever proposes "generalizing" a backoff that is already implemented; the correct NO-GO rationale is "already done," not the stream-K argument given.

**C3. L1's "exactly 0 at b1" is asserted, not the only failure mode.** Verified `M = hidden_states.size(0)` (`marlin_moe.py:318`) and `block_size_m` floor logic at `:334-339`: for E=128/256 at decode, `M·topk/E/8 < 0.9` always breaks at `block_size_m=8`. The padded pad-tokens enter `num_tokens_post_padded` only if they route to experts not already active. The roadmap's own probe gate is correct, but it understates a second erasure path: if pad tokens route into a `block_size_m=8` block that an *already-active* expert is paying for anyway (very likely at top_k=8, E=128, since most experts are active), the marginal tile cost is zero even at mid-batch. The "low-single-digit %" expectation is probably optimistic; realistic outcome is ~0.

**C4. The `>= num_experts` guard is NOT universal — the L2 NO-GO is right by luck.** The roadmap says the unconditional guard is at "line 126/303." Verified: the first kernel (`:126`) and `_count_and_sort` (`:305`) have it, but the `tokens_cnts` kernel at `moe_align_sum_kernels.cu:225-234` has **no** `>= num_experts` guard before indexing `tokens_cnts[... + expert_id]`. So a sentinel `≥num_experts` (the roadmap's proposed "correct" alternative to `-1`) would itself be OOB in that second kernel. Both sentinel directions are unsafe; only the runner-zero (L1) or the existing `ignore_invalid_experts` path (M3) are correct.

### IGNORED / MISAPPLIED PRIOR FINDINGS

**P1. The `small_batch_expert_mode` smem-crash NO-GO is correctly verified but the gate is `num_experts <= 64` (`:525`), and 35B-A3B is E=128 (per `project_qwen36_quants`: "35B=expert-only"), not E=256.** The roadmap's table oscillates between E=256 and E=128 (§1 says E=256, §2 bsm rows say E=128). For the actual deployed 35B-A3B (E=128), the small_batch gate is already missed (128 > 64), so that path never engages regardless — making the smem-crash argument moot, not load-bearing. Pin the expert count.

**P2. The DRAM-SOL "replicated per expert" claim conflates two regimes the memory already separates.** WF0's wall is M=1 dense weight-streaming. At MoE decode b1, per-expert M ≈ B·top_k/E < 1, so each active expert streams its slab for a *single* token — strictly worse arithmetic intensity than dense M=1, which the roadmap states. But it doesn't reconcile this with `project_cudagraph_decode_bandwidth`'s "MoE phantom-expert masking = the ONE real lever (near-free, bounded, measure first)" — the roadmap demotes that memory's headline lever to "plausibly 0 after probe" without noting it is *contradicting* a prior GO-with-measurement verdict. That's defensible (skepticism is warranted) but should be flagged as a deliberate downgrade, not silently.

### Bottom line

The roadmap's structural conclusion (no flagship MoE kernel win; per-expert GEMM is the dense bandwidth wall) is sound and well-defended. But it **misidentifies its own #2 byte-lever**: the real un-examined DRAM cost is the `torch.sum` epilogue reduction (M1), not `ops.moe_sum` (which the marlin path never calls, M2). It also overlooks the existing ungated `expert_ids==-1` skip + `ignore_invalid_experts` path (M3), which both offers a kernel-native phantom-mask alternative and weakens the "L1-is-runner-only proves DO-NOT-VENDOR" framing. Several NO-GOs reach the right verdict via wrong/imprecise code citations (C1, C2, C4). Fix the expert-count inconsistency (E=128 vs 256, P1) — it invalidates at least one NO-GO's load-bearing arithmetic.

Files: `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:318,334-339,393-396`; `/home/trevor/vllm-ampere-flashampere/vllm/csrc/moe/marlin_moe_wna16/ops.cu:105-108,266-334`; `/home/trevor/vllm-ampere-flashampere/vllm/csrc/moe/moe_align_sum_kernels.cu:124-138,225-234,305-307,525,648-655`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:11-46`; `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin.cu:265-312,449-473`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py:3492,3501-3503`.

---

## RUN B roadmap

Confirmed the decisive correctness point: the `-1` skip at line 132 is inside `if (has_expert_map)` — so a raw `topk_ids=-1` sentinel WITHOUT expert_map falls through to `expert_id >= num_experts` (false for -1) then to `atomicAdd(&shared_counts[... -1 ...])` = OOB. This validates the verdict that the constant-pad-id fix (not the -1 sentinel) is the only safe route. All anchors verified. Writing the roadmap.

# famp-Marlin MoE Expert-GEMM + Batched-M Ampere Optimization Roadmap

Scope: the 35B-A3B MoE expert path (the real deployment) + dense batched-M crossover (M=8..64). Builds on workflow-0's 13 dense-M=1 NO-GOs — does NOT re-derive them. All code anchors below were verified against `/home/trevor/vllm-ampere-optimized/flashampere/marlin/` (dense) and `/home/trevor/vllm-ampere-flashampere/vllm/` (fork MoE) this session.

---

## 1. Executive Summary

**Structural fact that frames everything:** famp vendored ONLY the dense Marlin (`flashampere/marlin/csrc/marlin/` — `grep` for `expert|moe|sorted_token|B_expert_off` returns **empty**, confirmed). The 35B-A3B MoE expert-GEMM is the **unvendored upstream** kernel at `vllm/csrc/moe/marlin_moe_wna16/`, carried only via patch 0005 (int8-act per-expert scale). Any MoE lever edits the fork, not the famp SoT tree.

**The honest bottom line for this frontier — three independent walls, all confirmed:**

1. **Per-expert-M is the dense-M=1 wall, multiplied.** At decode, per-expert avg M = B·topk/E is *tiny* (0.03 at B=1, 1.0 at B=32 for E=256/topk=8). Each touched expert streams its FULL int4 weight slab for an 8-or-16-row GEMM — same DRAM-SOL currency as dense M=1, just × min(E, B·topk) active experts. Aggregation across experts does NOT raise per-GEMM intensity; it only raises the COUNT. So **decode MoE is bandwidth-bound exactly like dense**, and the deployed regime (35B-A3B activates ~3B/token) is *more* weight-bound than dense-27B, not less.

2. **The MoE kernel is already MORE Ampere-tuned than the dense one.** It has adaptive `blocks_per_sm` (allow_count 1–4, `ops.cu:282-330`), part1-DP/part2-StreamK grid-fill, and `-1` trailing-block skip — all of which the dense kernel LACKS (dense hard-returns `{1, th_config}` at `marlin.cu:309`, verified). **Vendoring the MoE kernel buys ownership, not performance (0%).** The asymmetry runs MoE→dense, not the reverse — but porting MoE's adaptive occupancy INTO dense is also dead (the C_tmp reduce buffer is sized `sms`, not `sms*blocks_per_sm`, so `blocks_per_sm>1` causes a global-memory OOB; that's the real reason the dense-occupancy lever is NO-GO).

3. **The 19%-SOL-gap analog for MoE is phantom-expert byte-count, and it's the only survivor — but bounded near 0.** Every dense kernel-internal lever (8-row tile, occupancy, intermediate tiles, g32→g128-on-MoE-as-perf) is NO-GO or near-0. The single mechanism that reduces *bytes* (not flops, so it doesn't fight the SOL wall) is **collapsing cudagraph-pad phantom-expert routing** — and it is exactly 0 at b1, ~0 at saturated batch, and pays only at mid-batch padded valleys, possibly erased by stale-id/real-routing overlap. Measure-first.

**Highest-confidence actionable wins (all CONDITIONAL, measure-gated):**
- **L1 — zero cudagraph-pad `input_ids`** (the #1 lever): 1-line runner change, correctness-neutral, no rebuild, captures the phantom-expert byte win where it exists.
- **g32→g128 MoE expert scales** (recipe-only): ~1-2% in the cudagraph-batched/prefill regime (NOT decode), bundled with WF0's dense g128 fold validation.
- **Phase-A block_size_m heuristic sweep** (env-tunable, zero-build): marginal, but free and fills a `TODO`.

**Tapped-out / dead (do not pursue):** MoE int8 8-row tile port (decode +0 per WF0, + full transposed-datapath rewrite risk), kernel-side zero-valid-block early-out (the empty-block set is provably empty in `[0,parallel)`), porting adaptive occupancy to dense (C_tmp OOB), intermediate (128,128,256) tile (smem math is *backwards* — the proposed tile is larger), int8-act-decode-parity (the −4.4% residual is irreducible per-token-quant against the int4-weight wall).

---

## 2. Tiered Lever Table

### TIER 1 — CONDITIONAL (build only after the measure-gate)

| Lever | Phase | Impact | Effort | Risk | Rationale / gate |
|---|---|---|---|---|---|
| **L1** Zero cudagraph-pad `input_ids` (collapse phantom experts to token-0) | decode | 0 at b1/saturated; ~0–2% mid-batch valleys, plausibly 0 | low (1 line, no rebuild) | low (correctness-neutral; pad output already sliced off) | Mechanism code-verified: `gpu_model_runner.py:3492` slices `input_ids.gpu[:num_input_tokens]` while `:3503` zeroes only positions → stale ids route to valid phantom experts → full int4 slab streamed. **Gate:** RoutedExpertsCapturer must show distinct-expert delta > 0 at off-stride batches {3,5,9,17,25}. |
| **g32→g128 on MoE expert scales** (recipe, no kernel edit) | batched/prefill | ~1-2% (NOT decode) | low | medium (quality + unvalidated g128 `k==1` fold) | MoE generator already instantiates kS8 group_blocks [-1,2,4,8] (`generate_kernels.py:126,134`); g128 quarters scale-byte stream. Per-expert amax denominator is group-count-invariant → no re-collapse. **Bundle with WF0 dense g128 fold bit-exact validation** (shared template code). |
| **L3** Wire `token_mask` through `moe_align` (mm-safe superset of L1) | decode | ≥ L1 but same bound; only net-new value = multimodal/inputs_embeds coverage | medium (op-schema + ForwardContext plumbing) | medium (torch op surface) | `token_mask` infra exists (`moe_align_sum_kernels.cu:136,232,284`, `nullptr` non-LoRA). Covers `inputs_embeds` path (`:3484`) L1 can't reach. **Escalate to L3 only if L1 insufficient AND a multimodal-MoE deploy exists.** |
| **Phase-A** block_size_m heuristic env-tunable sweep | both | ~0-1% (int8 floor blocks reaching 8; only 16-vs-{32,48,64} rebalance) | low (zero-build) | low (selects among built tiles) | `marlin_moe.py:333` explicit `TODO: tune this further`; floor at `:338-339,508-509`. Free; A/B on pp2/single-card to unmask the all-reduce-bound tp2 box. |

### TIER 3 — CONTESTED

| Lever | Verdict split | Resolution |
|---|---|---|
| **Vendor MoE marlin for perf** | NO-GO (perf) vs GO (adopt the do-NOT-vendor recommendation) | **Do NOT vendor for performance = 0%.** MoE is already more occupancy-aware than dense (`ops.cu:282-330` adaptive vs dense `marlin.cu:309` `{1,th_config}`). Vendoring relocates code, same SASS. Only defensible as consolidation, gate on a 2nd MoE patch existing (today only patch 0005). Two sub-arguments in the lever were falsified: dense DOES have StreamK; the int8 8-row writeback divergence runs MoE-lacks-it, not dense. |

### TIER 4 — NO-GO (dropped, with the prior finding contradicted)

| Lever | Phase | Why dead | Prior finding |
|---|---|---|---|
| **L2** Zero-valid-block kernel early-out | decode | Premise falsified: per-expert CEILDIV padding lands in the SAME block as the expert's last valid tokens (`moe_align_sum_kernels.cu:153`); trailing -1 blocks are beyond `parallel = num_tokens_past_padded/moe_block_size` (`template:386`). Zero-valid-block set in `[0,parallel)` = **empty**. + StreamK deadlock risk for 0 gain. | Confirms DRAM-SOL; the real byte lever is L1 at routing, not a kernel early-out. |
| **Port adaptive `blocks_per_sm` to dense** | both | C_tmp reduce buffer sized `sms` (`marlin.cu:691-692`), NOT `sms*blocks_per_sm`; returning `blocks_per_sm>1` → `locks_off=blockIdx.x` over-indexes `c_cur_offset` in `global_reduce_fp32` = global-mem OOB on the default bf16 path. Reg term likely caps at 1 on the 256-thread tile anyway (no `__launch_bounds__`). | **Re-instates WF0's C_tmp-OOB NO-GO** (WF0's own critique that retracted it was wrong). Also: occupancy smem-pinned 1 block/SM. |
| **Intermediate (128,128,256) large-batch tile** | both | smem math is **inverted**: at the M=17-32 band (tmb=2) the proposed tile = 53248B > current (64,256,256) = 45056B because sh_a DOUBLES with tb_k 64→128 while dominant sh_b (32768) is unchanged. On sm86 (~99KB) 2 blocks need 106496B > cap. Persistent grid = 82 CTAs always (`marlin.cu:471`); narrower N can't "fill more SMs". | Contradicts get_kernel_cache_size smem math + persistent-StreamK grid=82/waves=1.00. |
| **Lower dense int8 8-row gate M≤8 → M≤16** | both | 8-row tile physically emits 8 rows; host par-loop (`marlin.cu:417-528`) hands 9..16 to ONE launch (par_count=0, no 8+rest split) → rows 8..15 silently DROPPED = wrong output. Builds clean, fails at runtime. | WF0 8-row int8 = +0 decode (the verdict stands; the challenge is unreachable). |
| **Port int8 8-row tile to MoE + lift floor to 8** | decode | WF0 dense 8-row = +0 (DRAM-SOL). MoE template has NONE of patch-0002's 4 transposed-layout fixes (plain `ldsm<2>` gather at `:1086`, `matmul_a8` has no 8-row int8 branch, writeback `:1797` hardcodes 64). Builds clean → silent garbage on real activations. HIGH effort, 0 gain. | WF0 int8-8row=+0; project_w4a8_moe_ampere int8-MoE decode −4.4%. |
| **PIECEWISE 2-launch moe_align fold** | decode | Spec/MTP decode runs FULL cudagraph on Ampere (all backends UNIFORM_BATCH=2, downgrade gate `2<2` never fires); even under PIECEWISE, moe_align is NOT a split-op so it's captured inside the segment. E=256 single-launch smem=258KB > sm86 99KB anyway. Launch-latency, not bandwidth. | project_cudagraph_decode_bandwidth (decode = one FULL graph). |
| **TOPK=8 moe_sum specialization** | decode | Stale: `marlin_moe.py:788` already passes `moe_sum=ops.moe_sum` (the torch.sum branch is dead); topk=8 already uses fused `at::sum_out` (`:653`), one launch. ~0, inside FULL graph. | project_ampere_int8_throughput_ceiling (M=1 reduce negligible vs weight stream). |
| **Validate/ship int8-act MoE decode parity** | both | Already executed + shipped (patch 0005, opt-in default-off). decode-b1 = −4.4% measured under cudagraph; both recovery levers (8-row, phantom-mask) are ~0 at b1 by construction. Residual = irreducible per-token-quant against int4-weight wall. | project_w4a8_moe_ampere ("TAPPED OUT"); decode-b1 −4.4%. |

---

## 3. The #1 Lever — FULLY SPECIFIED

### L1: Zero the cudagraph decode `input_ids` pad region (kill stale-token phantom routing)

**Why #1:** Only lever that (a) reduces actual HBM bytes (the one currency the DRAM-SOL wall permits), (b) is correctness-neutral by construction, (c) needs no kernel rebuild, (d) is the named follow-up in project_cudagraph_decode_bandwidth. It is the MoE analog of "recover the SOL gap": it doesn't go faster per byte, it streams fewer phantom-expert slabs.

**Mechanism (all anchors verified this session):**
- `gpu_model_runner.py:3492` → `input_ids = self.input_ids.gpu[:num_input_tokens]` (full padded region, no zero).
- `gpu_model_runner.py:3503` → `self.positions[num_scheduled_tokens:num_input_tokens].zero_()` (positions ARE zeroed — the asymmetry).
- Pad slots hold STALE valid token ids → embed → router → valid `topk_ids` (NOT -1) → `moe_align` counts them → `moe_wna16_marlin_gemm` advances `B_expert_off` and streams those experts' full int4 gate/up/down slabs (`marlin_moe_wna16/marlin_template.h:563`) for output later discarded by `[:num_scheduled_tokens]`.
- The `-1` skip does NOT fire (experts are valid). **Critical (verified `:132`):** the `-1` skip is inside `if (has_expert_map)` — so the alternative `topk_ids=-1` sentinel WITHOUT expert_map falls through to `atomicAdd(&shared_counts[...-1...])` = **OOB**. This is why the constant-pad-id fill is the ONLY safe fix, not the sentinel.

**Exact file + region:**
`/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py`, text-only branch at line 3492, mirroring the existing positions-zero at 3503.

**Concrete change:**
```python
# at ~line 3492, text-only branch
input_ids = self.input_ids.gpu[:num_input_tokens]
if num_input_tokens > num_scheduled_tokens:
    # mirror the positions-zero below: collapse cudagraph pad-slot stale
    # token ids onto token 0 so pad rows don't route to phantom experts
    self.input_ids.gpu[num_scheduled_tokens:num_input_tokens].zero_()
```
- Must be in-place `.zero_()` on the persistent static `self.input_ids.gpu` buffer (allocated once, "Persistent buffer for CUDA graphs" comment at `:717`; `CpuGpuBuffer.gpu` never realloc'd) — replay-safe by construction, identical mechanism to the proven positions-zero.
- Runs eagerly in `_preprocess` before the captured `self.model(...)` replay, writing the static buffer the graph reads.
- Correctness-neutral: pad outputs discarded by the caller's `[:num_scheduled_tokens]` slice. Confirm token id 0 is a valid embedding index (not a special sentinel that asserts).

**Build steps:** NONE. Pure-Python runner edit, no kernel recompile on either marlin. (Note: the byte-stream citations `template:563`/`:386` live in the STOCK `csrc/moe/marlin_moe_wna16/`, not the famp dense tree — but no kernel change is needed.)

**sm86 A/B benchmark plan:**

*Pre-build measure-gate (decisive — do this FIRST, abandon if it fails):*
```bash
# Enable RoutedExpertsCapturer on real 35B-A3B; count DISTINCT active experts
# per MoE layer for masked (pad input_ids zeroed) vs unmasked, at OFF-STRIDE
# batches b in {3,5,9,17,25} (NOT 1,2,4,8,16 — those are pad=0, will show 0).
# GATE: if distinct-expert delta < ~1 expert/layer avg → STOP (overlap erased it).
```

*Serve (only if gate passes):*
```bash
vllm serve <35B-A3B W4A16 ckpt> \
  --marlin-input-dtype int8 \
  -pp2 -tp1 \           # PP2 unmasks (no-NVLink box: tp2 all-reduce 67% + 64MB-shm landmine)
  --shm-size 8g \       # implied via --ipc=host / docker --shm-size=8g
  --max-num-seqs 32
```

*Manual-time tok/s M-sweep (NEVER get_metrics TPOT — ~2x inflated):*
```bash
# bench_decode_clean.py style: manual t(N)-t(1), prefix-cache OFF,
# via OpenAI HTTP client (streaming), NOT offline LLM().
# Report tok/s = 1000/TPOT at off-stride decode batches {3,5,9,17,25}, masked vs unmasked.
```

*Byte confirmation:*
```bash
ncu --metrics dram__bytes_read.sum \
  --kernel-name regex:moe_wna16_marlin_gemm \
  ...  # GEMM1 + GEMM2, with/without the zero, at a valley batch
# Confirm byte delta == (distinct_experts_masked - unmasked) * per_expert_slab_bytes
# 35B-A3B: ~0.5MB w1 shard + ~0.5MB w2 per extra distinct expert / 936GB/s
```

*Correctness check:* GSM8K (gate ≥95%, baseline 95.83%) — output is unchanged by construction (pad outputs discarded), so this confirms no capture/replay regression.

**Ship gate:** distinct-expert delta > 0 at off-stride batches AND >1% sustained decode tok/s. Per feedback_squeeze_every_bit, a correctness-neutral 1-line win is worth shipping even if small — but only after the delta>0 gate. If RoutedExpertsCapturer shows delta ~0, L1 becomes a zero-op (still harmless to leave in).

---

## 4. Do-Together Bundles + Sequence

**Phase 0 — One measurement session (gates the whole frontier, no builds):**
1. RoutedExpertsCapturer distinct-active-expert probe at off-stride batches {3,5,9,17,25} for L1/L3. → If delta ~0, drop L1+L3 entirely.
2. `nsys` one decode step: confirm 35B-A3B decode runs FULL cudagraph (kills all PIECEWISE launch-latency levers immediately).
3. `ncu` MoE GEMM1/GEMM2 at b1/b8/b32: confirm DRAM-SOL + that `determine_exec_config` already selects `allow_count>1` (confirms vendoring/dense-port are 0).

**Bundle A — Phantom-expert byte lever (after Phase-0 gate 1 passes):**
- L1 (1-line input_ids zero) first. If insufficient AND a multimodal-MoE deploy exists → escalate to L3 (token_mask plumbing). Never both, never L3 first.

**Bundle B — Recipe scale-bytes (independent, runs in parallel with A):**
- g32→g128 MoE expert ckpt **bundled with WF0's dense g128 `k==1` fold bit-exact validation** (shared template code; validate ONCE for both dense + MoE on the (64,256,256) config with real heterogeneous activations). Quality gate: GSM8K≥95%/MMLU-Pro≥80%, per-layer-mix keeping g32 on the down_proj outlier. Perf A/B in the **cudagraph-batched/prefill** regime (NOT decode), on pp2/single-card to unmask.
- Phase-A block_size_m env-tunable sweep rides along (zero-build, correctness-neutral) — but expect ~0 and gate on pp2/single-card, not the all-reduce-masked tp2 box.

**Sequence:** Phase 0 → (Bundle A ∥ Bundle B). Total kernel work if everything passes: **zero new kernels.** L1 is Python; g128 is a recipe + a shared fold-validation already owed to WF0; everything kernel-side is NO-GO.

**Honest expected aggregate:** if Phase-0 gates pass, ~1-2% in cudagraph-batched/prefill (g128) + low-single-digit% at mid-batch decode valleys (L1). Decode-b1 single-stream throughput is **unmovable** — it is the int4-weight DRAM-SOL wall, confirmed for both dense and MoE. The frontier's real ceiling is "don't regress, harvest the bounded byte-count valleys, ship g128," not a step-change.

## RUN B critique

All confirmed. The MoE writeback at line 1797 has an explicit `if constexpr (m_block_size_8)` branch — it is NOT "hardcoded 64". The `64` is inside the 8-row branch (`c_sh_wr += 64 * (threadIdx.x / 32)`), which is the 8-row-specific layout, the opposite of what the roadmap claims. The `ldsm` is templated. The `m_block_size_8 || row < 8` guard at line 991 is exactly the kind of 8-row layout fix the roadmap claims is absent.

I have thoroughly verified the load-bearing claims. Writing the critique.

---

# Critique of the famp-Marlin MoE Expert-GEMM + Batched-M Roadmap

I verified every load-bearing code claim against the actual files. The roadmap's three structural walls (per-expert-M = dense-M=1 wall × active-experts; MoE is more occupancy-tuned than dense; phantom-byte is the only survivor) are sound and well-anchored. But several key NO-GO justifications rest on falsified premises, and the highest-value MoE lever is missing. Details below, most→least important.

## Corrections (load-bearing claims that are WRONG)

**1. FALSIFIED — the "Port int8 8-row tile to MoE" NO-GO is built on a premise that is false: the int8 + 8-row tile ALREADY EXISTS in the MoE template.** The roadmap claims (Tier-4) "MoE template has NONE of patch-0002's 4 transposed-layout fixes (plain `ldsm<2>` gather at `:1086`, `matmul_a8` has no 8-row int8 branch, writeback `:1797` hardcodes 64)." All three sub-claims are wrong: `m_block_size_8` is pervasive (`marlin_template.h:49,243,308,375,750-751,816,822,991`), it composes with `is_a_8bit` (`:354`) throughout, the gather is `ldsm<m_block_size_8 ? 2 : 4, a_type_id>` (`:1086`, templated not hardcoded), and the writeback at `:1791-1797` has an explicit `if constexpr (m_block_size_8)` branch where the `64` is the 8-row-specific layout term, not a hardcoded 16-row value. The 8-row row-guard fix (`if (!m_block_size_8 || row < 8)`) is present at `:991`. The NO-GO conclusion (decode +0 per WF0 DRAM-SOL) may still hold, but the stated reason ("builds clean → silent garbage, the fixes are absent") is invalid — there is no port to do, so the "HIGH effort" cost is also wrong.

**2. IMPRECISE — the dense C_tmp OOB justification mis-describes the buffer and overstates the blocker.** The roadmap says (Tier-4) C_tmp is "sized `sms` (`marlin.cu:691-692`), NOT `sms*blocks_per_sm`." Actual: `marlin.cu:689-693` sizes it `sms * max_m_block_size * max_thread_n` floats — i.e. `sms` *slots*, each `max_m_block_size*max_thread_n`. The OOB mechanism is real and I confirmed it: `c_cur_offset = locks_off * c_size` (`marlin_template.h:1691`) with `locks_off = blockIdx.x` (`:420`), so `blocks_per_sm>1` → `gridDim.x = sms*bps` → `blockIdx.x` overruns the `sms`-slot buffer. BUT the MoE kernel proves this is a one-line alloc fix, not an inherent blocker: MoE sizes its buffer `sms * 4 * moe_block_size * max_thread_n` ("max num of threadblocks is sms * 4", `ops.cu:694-696`) and launches `blocks = sms * exec_cfg.blocks_per_sm` (`ops.cu:480`) — same `locks_off=blockIdx.x` indexing (`:458`), just a correctly-sized buffer. So "dense can't do adaptive occupancy" should read "dense's C_tmp alloc would need the MoE-style `*max_blocks_per_sm` widening first" — and the real reason it's still NO-GO is WF0's smem-pinned 1-block/SM, which makes the alloc fix pointless, not the OOB itself.

**3. OVERSTATED — the "-1 sentinel = OOB" claim is directionally right but the guard story is incomplete.** Verified `:126` has `if (expert_id >= num_experts) continue;` BEFORE the `has_expert_map` block, and `:132` has the `-1` skip inside `if (has_expert_map)`. A `-1` without expert_map passes the `>=num_experts` guard and reaches `atomicAdd(&shared_counts[warp_idx*experts_per_warp + expert_offset])` with `expert_offset = -1 % N = -1` → OOB. Correct conclusion (constant-pad-id, not sentinel). BUT note the guard is inconsistent across the four align kernels: the SGL path at `:226-228` and `:275-278` have NO `>=num_experts` guard at all (only the `has_expert_map` block), so they'd OOB on `-1` even more directly. This strengthens the "constant-pad-id only" verdict but the roadmap presents the single `:132` anchor as the whole story.

## Gaps (missing levers / unverified claims)

**4. MISSING — the dominant batched-M MoE lever is grouping/sorting tokens to raise per-expert M, which the roadmap never raises.** The entire roadmap treats per-expert-M as fixed at `B·topk/E` and concludes "aggregation only raises COUNT, not intensity." That is true for the *current* token→expert layout, but the standard batched-MoE win on bandwidth-bound hardware is the opposite framing: at the cudagraph-batched/prefill regime the roadmap explicitly targets, increasing tokens-per-expert (larger `moe_block_size`, expert-parallel packing, or capacity-factor batching across decode steps) is what amortizes the int4 slab read across more rows — the same "fewer bytes per useful flop" currency. The roadmap's own L1 mechanism (collapsing phantom experts) is a *special case* of this (reducing distinct experts), yet the general lever — fewer distinct experts per launch via routing/batching, or larger M per touched expert — is absent. This is the actual "batched-M crossover" the scope asked for, and it's the gap.

**5. UNVERIFIED / OVER-OPTIMISTIC — L1's win is asserted to require RoutedExpertsCapturer, but the more likely outcome (delta≈0) is under-weighted given the roadmap's own evidence.** I confirmed the stale-pad mechanism is real: `input_ids.gpu` is written only at scheduled indices (`:1785` copy, `:1797/:1821` scatter), never the pad region, so pad slots hold stale ids; `:3492` doesn't zero them while `:3503` zeroes positions. So L1 is correctly motivated. BUT at decode the dominant batch sizes ARE the cudagraph capture strides (the pad-to-next-stride is small relative to E=256), and stale ids from the *immediately prior* step route to experts that overlap heavily with the current step's experts (temporal locality of routing) — the roadmap flags this ("plausibly 0", "overlap erased it") but still ranks it #1 and budgets a full ncu/serve campaign. Given it's bounded to mid-batch valleys AND subject to routing overlap AND the buffer is int32 (a zeroed slot routes token-0's embedding through the router → still a valid expert, just a *consistent* one, so the win is "fewer *distinct* phantom experts," not "zero phantom work"), the honest expected value is closer to 0 than the "#1 lever" framing implies. It's a 1-line correctness-neutral change worth landing, but the measure campaign is disproportionate.

**6. MISSING — no treatment of GEMM1/GEMM2 fusion or the intermediate activation (silu+mul) bandwidth between the two MoE GEMMs.** The 35B-A3B expert path is two back-to-back marlin GEMMs with a silu/mul + (for int8-act) a re-quantization of the intermediate in between. On a bandwidth wall, the intermediate write-then-read of the `[tokens, inter_size]` activation and its int8 re-quant scale computation is real DRAM traffic the roadmap never accounts for. Whether the fork fuses act+requant into the GEMM1 epilogue, or round-trips through HBM, is unexamined — and it's a per-token cost that scales with batched-M, directly in scope.

**7. UNVERIFIED — the g32→g128 "no re-collapse" claim for per-expert scales is asserted, not checked.** The roadmap states "per-expert amax denominator is group-count-invariant → no re-collapse" and bundles validation with the dense fold. The MoE scale path (`b_s_ptr`, `g_s_ptr` in `ops.cu:526`) and the `group_blocks` math (`:301`) differ from dense in the int8-act per-expert-scale patch-0005 datapath; the claim that quartering scale bytes is quality-neutral per-expert needs the same gemma-down_proj-outlier caveat the roadmap applies elsewhere (it does keep g32 on down_proj in Bundle B, but the "group-count-invariant" theoretical claim is stated as fact without an anchor).

## Prior findings possibly misapplied

**8. The roadmap leans on "WF0: int8 8-row = decode +0" to NO-GO the MoE 8-row port — but per correction #1 there's no port, and the int8-MoE −4.4% decode regression (project_w4a8_moe_ampere) is a DIFFERENT phenomenon (per-token-quant overhead), conflated here.** The roadmap correctly cites both but then attributes the −4.4% residual to "irreducible per-token-quant against the int4-weight wall" while simultaneously claiming the 8-row recovery lever doesn't exist in MoE — when it does. If the 8-row int8 path is already present and the regression persists, that's stronger evidence the −4.4% is the quant epilogue / intermediate round-trip (#6), not the tile shape — which redirects effort toward fusion (#6), the actual untapped lever.

## Bottom line

The pessimism on decode-b1 (DRAM-SOL wall, unmovable) is correct and well-anchored. The roadmap's failures are: (a) it declares a MoE int8 8-row port NO-GO on a false "the code lacks it" premise when the code already has it (#1) — the real question is why int8-MoE still regresses, pointing at #6; (b) it misses the genuine batched-M lever (token-per-expert amortization via routing/block-size/capacity, #4) that the scope explicitly asked for; (c) it misses GEMM1→GEMM2 intermediate/requant bandwidth (#6); and (d) it over-ranks L1, whose realistic value is near-0 by its own overlap argument (#5). The C_tmp-OOB and `-1`-sentinel mechanics are real but described imprecisely (#2, #3).

Files verified: `/home/trevor/vllm-ampere-optimized/flashampere/marlin/csrc/marlin/marlin.cu` (`:309,689-693`), `.../marlin_template.h` (`:420,1683,1691,1701-1716`), `/home/trevor/vllm-ampere-flashampere/vllm/csrc/moe/marlin_moe_wna16/ops.cu` (`:282-330,458,480,694-696`), `.../marlin_moe_wna16/marlin_template.h` (`:354,375,991,1086,1728,1736,1791-1797`), `/home/trevor/vllm-ampere-flashampere/vllm/csrc/moe/moe_align_sum_kernels.cu` (`:126,132,226,275`), `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py` (`:718,1785,1797,3492,3501-3503`).