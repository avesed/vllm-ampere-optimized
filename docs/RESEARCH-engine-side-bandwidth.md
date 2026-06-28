# Engine-side DRAM bandwidth-utilization — Verdict (2026-06-27)

> ultracode (47 agents, 17 levers -> 4 survivors, 0 Tier-1 GO). Bottom line: the engine CANNOT move the
> decode DRAM wall. b1 single-stream = only MTP (shipped, accept-len-capped). Batched wins (max_num_seqs /
> gpu_mem_util / kv-dtype / chunked-prefill) = PER-DEPLOYMENT config -> a serving-recipe DOC, not fork patches.
> The only new shippable engine code lever = MoE phantom-expert masking (input_ids pad-tail zero, =R07) — and
> the critique shrank it further (int8-act block_size_m=16 -> up to 16 rows share a tile, so phantom rows on
> already-active experts cost ZERO extra reads). lm_head/sampler fused-read = NO-GO (~0): temp=0.6 never enters
> all_greedy, <0.1% of stream. Critique corrections: FA max_num_splits IS live (the split-KV NO-GO was wrong);
> async-guard must also assert disable_padded_drafter_batch + executor support.

Confirmed. The phantom-expert mechanism is real: in `_preprocess` (line 3492) and the dummy/capture path (5841), `input_ids` slices to the padded length while only `positions` is zeroed (3503). The `off_experts == -1` skip (fused_moe.py:161,415) and `ignore_invalid_experts` (1436, already True at 1669) exist as fix hooks. Output discard at line 4315 (`hidden_states[logits_indices]`) confirms pad-row outputs never reach sampling — so zeroing the tail is correctness-neutral. I have enough to write the grounded roadmap.

---

# Engine-Side Bandwidth-Utilization Roadmap — Ampere W4A8 hybrid (qwen3_5 GDN+full-attn 27B) + 35B-A3B MoE

## 1. Executive Summary — the honest bottom line, split by regime

**Decode is a hard DRAM-bandwidth wall, and the engine cannot move the wall.** At ~81% DRAM-SOL the int4 weight stream (~6.5GB/card tp2 dense, dominant routed-expert tiles MoE) is read essentially once per forward and that read sets the floor. The engine's only jobs are: **(A) amortize that one read over more useful tokens**, **(B) fill the idle SMs during the memory-bound phase**, **(C) cut the few non-weight bytes that rival the stream**, **(D) enable bigger batches**. Everything else (cudagraph forcing, L2 residency, metadata narrowing, int8 8-row tile, padding-grid density, sampler fp32 byte-cuts) was measured or derived to ~0 and must not be re-proposed.

### Regime b1 (single-stream serving) — essentially a wall
- **MTP (amortize, SHIPPED) is the only mover.** 1-fwd→k-tokens past the bandwidth wall. Already shipped (graft_mtp + patch-A FA2 verify). Its ceiling is **accept-len (~1.6–2.6 from Qwen's 1-layer head)** = a training/ckpt problem, NOT engine.
- **Async-scheduling host-seam** is the only *other* b1-applicable lever — and it is **already default-on** for the MTP shape. Worth a regression *guard* only (bounded ~1–3% b1, gated by an unmeasured nsys host-bubble; likely <2–3%).
- **Everything else at b1 is ~0**: no waiting prefill to fill SMs, KV/state reads are tiny vs the weight stream, capacity is irrelevant (one running req). **Be honest: at b1 the engine is out of levers beyond MTP.**

### Regime BATCHED / multi-request serving — where the engine actually earns its keep
- The whole prize is **keeping the batch full** (one weight read serves B tokens, ~linear to the compute crossover) and **overlapping prefill compute onto decode's idle SMs**.
- **But the two biggest batched levers are scope-excluded** (per `project_scope_general_ampere`): they are per-deployment knobs (max_num_seqs, gpu_memory_utilization, mamba_cache_mode, kv-cache-dtype, chunked-prefill thresholds), not shippable fork patches. They belong in a **serving-recipe doc**, not the fork.
- **The one genuinely-new, shippable, engine-side weight-byte lever is MoE phantom-expert masking** (zero the cudagraph-padded `input_ids` tail). It is the lone item that survives as a code change — but it is **MoE-only, ~0 at b1, probe-gated** (must measure the distinct-active-expert delta first; may net ~0 if stale ids correlate with real routing).

**Bottom line: there is no new flagship decode-throughput patch hiding in the engine.** MTP is shipped and accept-len-capped; the batched wins are deployment config (doc, not patch); the only new fork code is a small, probe-gated MoE masking one-liner plus two cheap correctness/regression guards.

---

## 2. Tiered table

Impact is **decode tok/s** (1000/TPOT) or **aggregate decode tok/s** (batched). R## = already in the prioritized roadmap R01–R29.

| Lever | Regime | Impact | Effort | Risk | Rationale / status |
|---|---|---|---|---|---|
| **GO (shippable fork code)** | | | | | |
| **MoE phantom-expert masking** — `input_ids[ns:ni].zero_()` mirroring the positions-zero | batched (MoE-only) | probe-gated; 0 at b1, low-single-digit % mid-batch valleys (upper-bounded ~20% at MTP-verify M3→4 under independent routing, realistically far less) | low | low (output discarded at 4315; fixed-shape, cudagraph-safe) | **= R07.** The lone real weight-byte engine lever. Mechanism verified (3492/5841 slice vs 3503 zero). MUST gate on RoutedExpertsCapturer distinct-active-expert delta — STOP if unchanged. |
| **Async+MTP composition guard** (startup assert async ON ∧ MTP ON; sync-parity smoke) | b1 | ~0 new; insures ~1–3% b1 against silent loss | low | none (verifies default-on path) | **= R11/B2 / P03.** Fold into the FULL-decode startup assert. Not a fresh win; regression insurance. |
| **FULL-decode-mode startup assert** (decode_mode()==FULL; no attn group < UNIFORM_BATCH) | both | ~0 new; catches the linear_attn UNIFORM_SINGLE_TOKEN landmine that would silently drop decode off cudagraph AND break MTP capture | low | none | Regression guard. Catches a cliff that would cost the whole decode path. |
| **NON-mamba max_num_seqs > max_concurrency startup WARNING** (extend the shipped mamba ValueError to MoE/dense; never auto-clamp) | batched | 0 if tuned; recovers a thrash cliff if mis-set | low | low (warn-only) | Residual of the auto-cap lever. Hybrid already hard-raises (compilation.py:1453-1468); gap is only non-mamba MoE/dense. Warning, not clamp. |
| **CONTESTED (verdict-split — measure decides)** | | | | | |
| **fp8-E4M3 KV cache** (capacity, MTP-OFF or long-ctx only) | batched, ≥24–32k | ~0 at deployed 4k; +7–14% long-ctx MTP-off; **net-NEGATIVE with MTP-on** (forces verify off FA2→Triton on Ampere) | medium | MTP conflict; E5M2 quality; per-token-head scale bytes | **= R12/B8.** Deploy knob + already-verdicted (`project_kv_quant_ampere_verdict`). Document, don't patch. |
| **GDN ssm-state bf16** (`--mamba-ssm-cache-dtype`) | both | **verdict-split**: one verifier found Qwen3.5 default already resolves to bf16 (`auto`→model bf16) = no-op; other found fp32-config-dependent +~3% short-ctx capacity | medium | GDN recurrent-state precision UNTESTED | **= R19/B11 subset.** Verify the model's resolved ssm dtype FIRST — if already bf16, DROP. Quality-gate (GSM8K/MMLU-Pro) before trusting. |
| **Cap prefill chunk size** (`long_prefill_token_threshold`) | batched/mixed | goodput + TPOT-SLO knob; **~0 decode-bandwidth** (cross-request batching already amortizes; this is prefill-fills-idle-SM, win cat (b) not (a)) | low | GDN drops off FULL graph on any blend (binary num_prefills==0) | **= D11/P07.** Per-deploy doc knob; one A/B run for the serving doc. |
| **NO-GO (dropped — prior finding contradicted)** | | | | | |
| mamba_cache_mode "all"→"align" | batched | **0 — false premise** | low | — | qwen3_5 default is already "align"/"none"; "all" hard-raises NotImplementedError (qwen3_5.py:459). |
| Auto-cap max_num_seqs to max_concurrency | batched | ~0 | low | — | Hybrid cliff already hard-guarded (compilation.py:1453-1468); max_concurrency already logged. Scope-excluded knob. |
| Relax `scheduler_reserve_full_isl` | batched | thin window few-%, often net-neg | low | preemption thrash on GDN | Duplicate P07; scope-excluded; binding cap is mamba-state not this flag. |
| Tighten cudagraph capture granularity | batched | ~0 bandwidth (padded rows ride shared weight read; per-row work L2-absorbed) | med | denser grid steals KV VRAM → lower concurrency | Contradicts cudagraph-SOL verdict; = `--performance-mode interactivity`, latency-only, scope-excluded. |
| Anti-mixing (segregate prefills) | batched | ~0/neg | med | higher prefill TTFT | Inverse of D11; protects ~0-value FULL graph (cudagraph nets ~0 on weight stream). |
| FA split-KV count (`max_num_splits`) | b1 long-ctx | **0 — impossible** | low | — | FA2 hard-raises `NotImplementedError(num_splits>1)`; knob is FA3/Hopper-gated. |
| Order self.running for cascade | batched | **0 — misread** | med | fairness | `get_num_common_prefix_blocks` is order-independent global-AND over allocated reqs; reorder can't raise it. |
| Skip sampler fp32 upcast on greedy | both | **0 deployed** | low | — | temp=0.6 default never enters all_greedy; MTP verify uses rejection_sampler not sampler.py:96; <0.1% of stream. = R02 subset. |
| Top-k candidate all-gather (TP2) | both | ~0/neg | high | MTP rejection-sampling needs full-vocab softmax denom = correctness bug | Off the weight stream (int8-QK failure mode); = R17, deferred. |
| Raise gpu_memory_utilization + ESTIMATE_CUDAGRAPHS | batched | ~0/neg | low | re-triggers OOM | ESTIMATE_CUDAGRAPHS already default-True; box already OOM-tight at 0.92; binding cap is GDN state not KV pool. Scope-excluded. |

---

## 3. The #1 engine lever, fully specified

**MoE phantom-expert masking** — the only new, shippable, engine-side weight-byte cut (= R07). It is #1 *among engine code deliverables* (MTP is already shipped and accept-len-capped; the bigger batched wins are scope-excluded deploy config).

### Mechanism (verified)
Decode/MTP-verify steps are cudagraph-padded: `num_input_tokens` (next bucket) > `num_scheduled_tokens`. The runner zeroes the **positions** tail but only **slices** `input_ids`:

- `gpu_model_runner.py:3503` — `self.positions[num_scheduled_tokens:num_input_tokens].zero_()` (positions tail cleared)
- `gpu_model_runner.py:3492` — `input_ids = self.input_ids.gpu[:num_input_tokens]` (tail keeps **stale prior-step token ids**)

Stale ids → `embed_input_ids` → MoE router → "phantom" experts no real token chose → each distinct phantom expert pulls one extra int4 expert-tile from HBM (the Marlin `fused_marlin_moe` path picks `block_size_m=8` at deployed low M, so one block per distinct expert, no block-sharing rescue). Output of pad rows is discarded (`hidden_states[logits_indices]` at `4315`), so masking is correctness-neutral.

### Exact change
File: `vllm/v1/worker/gpu_model_runner.py`, `_preprocess`, immediately after line 3503's positions-zero, **inside the same guard**:

```python
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > num_scheduled_tokens:
                self.positions[num_scheduled_tokens:num_input_tokens].zero_()
                self.input_ids.gpu[num_scheduled_tokens:num_input_tokens].zero_()  # route pad rows to a single deterministic expert (phantom-expert mask)
```

- Touch **only the text-only branch** (mm/prompt-embeds branches embed `[:num_scheduled_tokens]`, unaffected).
- Mirror the same guard in the dummy/capture path around **line 5841** so captured graphs match the live tail.
- **Stronger bound (only if the token-0 variant's probe shows residual concentration):** route the pad region to expert `-1` via `moe_align_block_size` so `off_experts == -1` (fused_moe.py:161,415) + `ignore_invalid_experts` (1436, already True at 1669) **skip** those tiles entirely rather than reading expert-0's set once. This is more than a one-liner (needs per-position validity plumbed into align) — do it only if step-1 shows token-0 still pulls phantom tiles.

### A/B bench plan (decode tok/s, never ms)

**Stage 1 — mechanism gate (no perf claim until this passes).** Serve 35B-A3B with `init_routed_experts_capturer` enabled; drive fixed b4 concurrent decode over the **OpenAI API** (`vllm serve` + HTTP client). Dump **distinct-active-experts / MoE-layer / step** WITH vs WITHOUT the tail-zero. **If unchanged → STOP, mark ~0.**

**Stage 2 — perf (only if Stage 1 shows a drop).**
- `vllm serve` 35B-A3B tp2, shm8g, OpenAI API, **non-streaming** client (streaming TTFT buffers on tp2 — garbage).
- Decode tok/s via **manual T(N)−T(1)** timing, **prefix-caching OFF** (NOT `get_metrics` TPOT — inflates ~2× at long ctx).
- Sweep concurrency **b1 / b2 / b4 / b8** (expect 0 at b1; valleys mid-batch).
- Report **aggregate decode tok/s delta**.
- Confirm MTP intact: `vllm:spec_decode_num_accepted` from `/metrics` unchanged.
- Quality spot-check (GSM8K ~200 prompts) — should be identical (output bit-unchanged).

**Honest expectation:** 0 at b1; bounded low-single-digit % in low-expert-overlap mid-batch valleys (incl. the deployed b1+MTP-K2 M3→bucket4 case = 1 phantom row); ~0 at large batch (most experts already active); pure no-op on the dense 27B (no routed experts). Ship behind the probe; the magnitude is genuinely unproven.

---

## 4. Do-together bundles + sequence

**Phase 0 — one measurement session gates everything (no GPU here; run on the 2×3090 rig).**
1. **nsys b1 host-bubble** at W4A8+MTP: inter-step / draft↔verify GPU-idle seam. `<2%` ⇒ the entire b1 host-duty cluster is dead (only the async guard ships). `5–20%` ⇒ a real b1 protect-target.
2. **RoutedExpertsCapturer** distinct-active-expert delta (masked vs stale) at b2/b4 on 35B-A3B — the single gate for the #1 lever.
3. **ncu DRAM-SOL** on the captured W4A8 GEMM b1 (confirm ≥85% — caps the whole byte/L2/occupancy family at ~0) and locate the **M-sweep compute crossover** (caps the entire batching prize).

**Bundle G1 — Engine guards (ship together, low-risk, correctness-neutral).** FULL-decode startup assert + async∧MTP composition guard + sync-parity smoke + non-mamba max_num_seqs>max_concurrency **warning**. One PR; protects shipped MTP + the cudagraph decode path.

**Bundle G2 — MoE phantom-expert masking (the one new perf code).** The 3492/5841 tail-zero, **behind Phase-0 gate #2**. If the probe shows a drop, ship the one-liner; only escalate to the `-1`-sentinel variant if token-0 concentration remains.

**Bundle D1 — Serving-recipe DOC (NOT fork patches; scope-excluded knobs consolidated).** The hybrid-W4A8 capacity recipe: `max_num_seqs` knee (read the logged `max_concurrency`), `gpu_memory_utilization`, `--kv-cache-dtype fp8 E4M3` (MTP-off / ≥24–32k only), `--mamba-ssm-cache-dtype bf16` (after quality gate; verify not already bf16), `long_prefill_token_threshold` (TPOT-SLO). Each is a deploy decision; the deliverable is documentation + one A/B run each, not code.

**Sequence:** Phase 0 → G1 (ships regardless, insures the shipped wins) → G2 (gated on probe) → D1 (doc, parallel). 

**The unembellished conclusion:** the engine's remaining decode headroom is small. MTP (shipped) is the wall-breaker and its ceiling is a ckpt/training problem. The batched prize is real but lives in deployment config the project scope excludes from the fork. The only new fork code worth writing is a probe-gated MoE one-liner and two cheap guards — and even those may measure to ~0. Spend the effort on Phase-0 measurement before any kernel/runner work; it will cap or unlock the entire tier.

Files referenced (all absolute): `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py` (3492, 3503, 4315, 5841), `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` (161, 415, 1436, 1669), `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/core/sched/scheduler.py`, `/home/trevor/vllm-ampere-flashampere/vllm/vllm/config/compilation.py` (1453-1468), `/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory/project_scope_general_ampere.md`.

---

# Critique

Confirmed: `MTPModelTypes` (line 34, includes `"qwen3_5_mtp"` and `"mtp"`) is a member of `EagleModelTypes` (line 56-57). So the roadmap's claim that async-scheduling is default-on for the MTP shape is **correct** — MTP passes the `method in EagleModelTypes` check. BUT the more important caveat the roadmap omits: it's only on if `disable_padded_drafter_batch` is False AND the executor (multiprocessing tp2) supports it. The roadmap's "async guard" should assert those, not just "async ON ∧ MTP ON."

I have enough verified evidence. Final critique.

---

# Critique — Engine-Side Bandwidth Roadmap (verified against `/home/trevor/vllm-ampere-flashampere/vllm`)

## Load-bearing code claims: what holds, what breaks

**HOLDS (verified):**
- `input_ids` stale-tail vs positions-zero: `gpu_model_runner.py:3492` (`input_ids = self.input_ids.gpu[:num_input_tokens]`) and `:3503` (`self.positions[num_scheduled_tokens:num_input_tokens].zero_()`) — exact, and the edit site is correctly inside the non-mrope/non-xdrope `else` branch. Output discard at `:4315` (`hidden_states[logits_indices]`) confirmed.
- qwen3_5 `"all"` cache mode hard-raises: `qwen3_5.py:459-461`. The "mamba_cache_mode all→align = false premise" NO-GO is correct.
- compilation.py mamba cudagraph guard: `compilation.py:1453-1468`, gated on `has_full_cudagraphs() and has_mamba_layers and max_num_reqs > num_blocks`. Correct, and it is indeed mamba-only (the non-mamba warning gap is real).
- async-scheduling default-on for MTP: confirmed — `qwen3_5_mtp`/`mtp` ∈ `MTPModelTypes` ∈ `EagleModelTypes` (`speculative.py:34,46,48,56-57`), so the `elif method not in EagleModelTypes` disable branch (`vllm.py:968-988`) does NOT fire for MTP. The b1 async claim stands.

**BREAKS / WRONG:**

1. **The central #1-lever mechanism mis-cites the block size.** The roadmap says "Marlin `fused_marlin_moe` picks `block_size_m=8` at deployed low M, so one block per distinct expert, no block-sharing rescue." Actual code (`experts/marlin_moe.py:334-340`): the loop picks the smallest `block_size_m` with `M*topk/E/block_size_m < 0.9`, **then** for int8-act (`input_dtype.itemsize == 1`, which is exactly W4A8) `block_size_m = max(block_size_m, 16)`. So the deployed path uses **block 16, not 8**, and up to 16 rows routing to the same expert **share one tile read** — i.e. block-sharing IS the rescue. A phantom row landing on an already-active expert costs **zero** extra tile reads. This shrinks the lever's upper bound well below the roadmap's "1 block per distinct phantom expert" framing.

2. **The "-1 sentinel escalation variant" is a no-op as described.** Roadmap proposes routing pad rows to expert `-1` so `off_experts == -1` (fused_moe.py:161/415) skips the tiles. But `ignore_invalid_experts=True` is **already passed in the live Marlin path** (`marlin_moe.py:348`) and `ignore_invalid_experts` only marks experts **outside the local EP `expert_map`** as -1 — phantom experts chosen by stale ids are *valid local* experts, never -1. There is no plumbing that turns a zeroed token-id into an invalid-expert sentinel. The escalation can't be built without inventing a per-position validity mask in `moe_align_block_size`; "more than a one-liner" undersells it — it's a different feature.

3. **The FA2 `num_splits>1` NO-GO is factually wrong.** Roadmap: "FA2 hard-raises `NotImplementedError(num_splits>1)`; knob is FA3/Hopper-gated." There is **no such raise** in `flash_attn.py`. `max_num_splits` is a live field (`:255,333,373`), set from `flash_attn_max_num_splits_for_cuda_graph` and passed to every `flash_attn_varlen_func`/`*_with_kvcache` call (`:468,853,876,976,1009`). The two `NotImplementedError`s at `:700` (fused output quant) and `:1046` (encoder KV quant) are unrelated. Split-KV count is a real, reachable Ampere knob — the NO-GO row should be deleted or re-verdicted, not asserted impossible.

## Over-optimistic / wrong-regime

4. **The phantom-expert "upper-bounded ~20% at M3→4" number is not credible** given finding #1. At MTP-K2 verify the pad is typically 1 row into a block-16 tile already holding 3 real tokens' experts; with top-k routing those 3 tokens already activate several experts, and the 1 phantom row very likely collides with an already-loaded expert (block-shared, zero cost) or adds at most one expert out of many already streamed. The honest bound is **sub-1% even in valleys**, and plausibly a hard 0 on the deployed shape. The roadmap's own "realistically far less" hedge should be the headline, not the 20%.

5. **The async guard is under-specified for its own regime.** The roadmap's guard asserts "async ON ∧ MTP ON," but the actual disable conditions (`vllm.py:945-988`) that matter for this box are `disable_padded_drafter_batch=True` and `executor_supports_async_sched` (mp tp2). A guard that doesn't assert those two will pass while async is silently off. Cite the real predicates.

## Misapplied / duplicated prior findings

6. **"Tighten cudagraph capture granularity ~0" is correct but the listed reason double-books the decode-cudagraph~0 verdict in the wrong place.** That verdict is about *replay* netting ~0 on the weight stream; capture *granularity* (bucket density) is a separate axis whose real cost is KV-VRAM-vs-concurrency (which the row also states). Fine to NO-GO, but it's not "contradicts cudagraph-SOL verdict" — it's orthogonal to it.

## The genuinely MISSING engine-side lever

7. **Mixed prefill+decode weight-stream fusion (the one real cat-(a) amortizer the roadmap omits).** The roadmap treats prefill→decode overlap only as cat-(b) "fill idle SMs" and then scope-excludes it as a chunked-prefill *threshold knob*. But there is a distinct **engine-code** lever: when a decode step and a chunked-prefill chunk are co-scheduled, the int4 weight tile read by the prefill GEMM is the *same* tile the decode rows need — batching decode rows **into** the prefill GEMM (shared-M) amortizes one weight read across both, which is cat-(a), not (b). vLLM v1 already mixes them in one forward; the question is whether the GEMM actually fuses the M dims or runs two passes. This is a real runner/layer inspection target and is more promising at b1-with-background-prefill than anything in the GO tier — it's absent from the table entirely.

8. **KV-read layout for the GDN/full-attn hybrid is dismissed too fast at b1.** "KV/state reads are tiny vs the weight stream" is true for full-attn at 4k, but the GDN recurrent **state** is read AND written every decode step and at the deployed tp2 dense shape its per-step traffic is not negligible relative to the per-card weight slice at long context. The roadmap never measures the GDN state byte-rate; it should be a Phase-0 nsys line, not assumed tiny.

## Bottom line on the roadmap's own bottom line

The "no new flagship hiding in the engine" conclusion is probably right, and the discipline (probe-gate everything, measure before kernel work) is sound. But the #1 GO lever is **weaker than stated** (block-16 sharing, finding #1), its escalation path **doesn't exist** (finding #2), one NO-GO is **factually false** (finding #3, num_splits is live), and one real cat-(a) lever (**mixed prefill/decode GEMM-M fusion**, finding #7) is missing. Net: the GO tier shrinks toward "two cheap guards + a likely-0 MoE one-liner," which strengthens the roadmap's pessimism — the honest read is even flatter than written.

Files cited (absolute): `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/worker/gpu_model_runner.py:3492,3503,4315,5841`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:334-348`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:161,415,1436,1470,1669`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/v1/attention/backends/flash_attn.py:255,333,373,468,700,1046`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/config/vllm.py:945-997`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/config/speculative.py:34-57`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/config/compilation.py:1453-1468`; `/home/trevor/vllm-ampere-flashampere/vllm/vllm/model_executor/models/qwen3_5.py:459`.