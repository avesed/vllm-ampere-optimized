# I-4b CHUNKED (cached-prefix) int8-QK prefill — results

**STATUS: chunked / cached-prefix path BUILT + e2e VALIDATED on the real Qwen3.5-9B-W4A8 (single
RTX 3090, sm_86, image v0.23.0).** The original I-4b backend fired int8 only on PURE-FRESH prefill
(whole prompt in one step → capped ~64k single-card by the single-step activation memory). This
extension fires int8 on the CACHED-PREFIX chunks too, so chunked prefill (small
max_num_batched_tokens) makes 128k FIT ON ONE CARD and runs int8 on every chunk (not just chunk-0).

## The approach (gather + single_prefill, NO int8 KV cache)
On each eligible full-attn prefill chunk: reshape_and_cache_flash writes the NEW chunk's K/V into
the paged cache (as FA does) → read the FULL context K/V (prefix + chunk = first `seq_len` tokens)
from the paged cache via the block_table → per-token int8-quant Q(chunk)/K(full)+smooth_k,
per-tensor int8 V → `single_prefill_with_kv_cache(q[chunk], k[full], v[full], causal=True)`. qo<=kv
+ causal aligns the chunk's queries to the END of the kv (chunk token i attends to all prior
context + itself). Pure-fresh chunks skip the gather (in-hand K/V; bit-identical to the old path).

## qo<kv causal-alignment VERIFICATION (the #1 correctness risk) — PASS
`i4b_align_test.py` (isolated, vs a full-prefill reference, D=256 H8 GQA-g4):
- single_prefill(q=new chunk [C], k/v=FULL context [S], causal=True) == the LAST C rows of the full
  causal prefill: **cos 1.00000** across 6 splits incl page-unaligned (P=2000,C=333 ; P=777,C=256).
- holds for BOTH **fp16** (the alignment convention) AND **int8** (the kernel preserves it).
- NEGATIVE control: cand vs FIRST C rows = cos 0.156, vs LAST C rows = 1.00000 → END-aligned confirmed.

## int8 fires on cached-prefix chunks (not just chunk-0) — PROVEN by the fire SPLIT
The harness logs the fresh/cached fire split. At 128k with 8192-token chunks (16 chunks):
**fresh=8, cached=120 per run** = 8 full-attn layers × (1 fresh chunk + 15 cached chunks) = 128
int8 calls/run. cached=120 (not 0) ⇒ int8 fired on every cached-prefix chunk.

## TTFT — single-card chunked, int8 vs FA (median of 3, +1 tok to isolate prefill TTFT)
All matched config (same MBT / GPUMEM / chunked); int8 TTFT INCLUDES the per-chunk on-the-fly
gather+quant overhead. needle retrieved CORRECTLY (== FA answer "73914") at every length.

| len  | chunking            | FA TTFT (s) | int8 TTFT (s) | speedup | needle | fire/run (fresh+cached) |
|------|---------------------|-------------|---------------|---------|--------|-------------------------|
| 16k  | MBT 4096 (4 chunks) | 2.637       | 2.704         | −2.5%   | OK     | 8 + 24                  |
| 64k  | MBT 16384 (4 chunks)| 14.054      | 13.964        | +0.6%   | OK     | 8 + 24                  |
| 128k | MBT 8192 (16 chunks)| 37.049      | 36.895        | +0.42%  | OK     | 8 + 120                 |
| 128k | MBT 16384 (8 chunks)| 37.181      | 36.532        | **+1.75%** | OK  | 8 + 56                  |
| 128k | MBT 24576 (6 chunks)| 37.276      | 36.511        | **+2.05%** | OK  | 8 + 40                  |

**Best 128k single-card = MBT 24576 (6 chunks): +2.05% TTFT, needle correct, int8 on all 6 chunks.**
Larger chunks → fewer re-gathers → bigger int8 win (16→8→6 chunks = +0.42%→+1.75%→+2.05%); bounded
by per-chunk activation memory (24576 fits at GPUMEM 0.88 single-card).

Reference (single-STEP, NO chunking, prior I-4b — bounded ≤64k single-card by single-step activation):
16k −0.9%, 32k ±0%, 48k +1.4%, 64k +1.9% — the Amdahl trajectory that projected ~+4%@128k. Chunked
128k at the largest chunk (+2.05%) lands above the single-step 64k point and on-trajectory once the
re-gather count is minimized.

## 256k single-card = NOT POSSIBLE (hardware ceiling, FA-shared — NOT an int8-path limit)
256k single-card OOMs for BOTH FA and int8 with the IDENTICAL error (tried 192 MiB, 127 MiB free):
the 256k KV-cache pool (~21 GiB) + weights (9.5 GiB) exceeds one 24 GiB 3090, independent of chunking
(chunking only shrinks per-step ACTIVATION, not the KV pool, which must hold all 256k tokens). Also
the model's max_position_embeddings is exactly 262144, so 256k is the model's absolute RoPE ceiling.
256k needs 2 cards (TP/PP). **The int8 path reaches the same single-card ceiling as FA → 128k is the
single-card max for this model on a 3090, and int8 is net-positive there.**

## HONEST finding: the per-chunk gather+re-quant tax erodes the int8 win at 128k
The single-step trajectory projected ~+4%@128k. CHUNKED int8 measures only **~+0.4–0.6%** because the
gather+single_prefill approach RE-GATHERS and RE-QUANTIZES the GROWING full context every chunk
(chunk N quantizes all N·MBT tokens of K and V in PyTorch). That O(Σ context) tax (which FA's fused
varlen kernel avoids — it reads the paged cache in-kernel, no separate materialize+quant) almost
exactly cancels the int8-QK matmul win. Fewer/larger chunks reduce the re-gather count → see the
MBT-16384 row. The int8-QK lever itself is real and net-positive (the single-step trend), but the
chunked-prefix MEMORY enabler carries a host-side gather/quant cost that nets to roughly break-even
at 128k single-card for this hybrid.

The BIG deliverable that DOES land: **chunked + cached-prefix int8 makes 128k single-card prefill
FIT and stay coherent (needle correct, int8 on all chunks)** — the prior path OOMed >64k single-card.

## Memory: streaming quant was required to fit 128k single-card
Naive full-context fp32 quant temporaries (~1GB per K and per V at 128k) OOMed (25–447 MiB free at
the failure). Fix = (1) gather+quant K and V SEPARATELY with eager frees (no k_bf16 + k_fp32 +
v_bf16 coexisting); (2) STREAM the per-token int8 quant over the sequence in 16384-row slices so the
fp32 working buffer is bounded by one slice (~128MB), not the full context. Numerics unchanged
(per-token scale is a row reduction; smooth_k mean computed full-context first). Fits at GPUMEM 0.88,
chunk 8192. (FA 128k chunked also fits — chunking is what enables single-card 128k for both.)

## Files
- `int8qk_backend.py` — extended backend (cached-prefix gather + streaming int8 quant + fresh fast path).
- `i4b_e2e.py` — harness (MAX_BATCHED_TOKENS chunk control + fresh/cached fire split reporting).
- `run_i4b.sh` — runner (MBT env for chunk size).
- `i4b_align_test.py` — the qo<kv causal-alignment isolation test (ALIGN_PASS).
