# I-4b — int8-QK compute_qk D256 perf tuning (real RTX 3090 sm_86, FlashInfer 0.6.12)

Goal: recover the production per-token int8-QK `compute_qk` at head_dim=256 (Qwen3.5's real hd)
from a NET-NEGATIVE 0.95×@D256 toward the structural ceiling, while keeping cos>0.99 & mag∈[0.95,1.05].
Op-level speedup = fp16-FlashInfer ms / int8-per-token ms, cuda-event median, causal, H=8 MHA,
real per-token symmetric int8 scales + smooth_k (`i4_time.py`).

## Root cause (the diagnostic that cracked it)
The kernel is REGISTER-BOUND at 255 regs WITH stack spill (cuobjdump `-res-usage`: STACK:48-416
bytes on the high-NUM_MMA_KV configs, on the f16 path too). Two perturbation probes on the PRODUCTION
(per-token, real-scale) kernel isolated the bottleneck:
- `noload` (smem reads → arith hash, IMMA + scales kept): **D256 1.094×@16k / 1.118×@64k**, D128 1.119/1.106.
- `scaleconst` (REAL smem loads kept, scales forced const = zero scale registers): **D256 1.083×@16k /
  1.100×@64k**, D128 1.201/1.214.

Either removing the loads OR removing the scale registers recovers ~the full gap → the overhead is
REGISTER PRESSURE from the per-token scale machinery (16 k-scale + a few q-scale floats held live
across the load+mma loop forced spills), NOT the un-vectorized direct-index smem load. The prompt's
lever-1 (vectorize/ldmatrix the load) is therefore NOT the lever here; lever-2 (scale amortization /
keep scales off the register-critical loop) is. The fix recovers ~60-70% of the available headroom.

## Variant sweep (op speedup; cos = i4_sweep ALL_PASS unless noted)
| variant | D256 16k | D256 64k | D128 16k | D128 64k | cos | note |
|---|---|---|---|---|---|---|
| **baseline (un-tuned, shipped i4)** | 0.961 | 0.986 | 1.159 | 1.055 | 0.9999 | NET-NEG @D256 |
| V1 kd-outer A/B preload+reuse (cut loads 2.7×) | 0.964 | 0.978 | 1.087 | 1.047 | 0.9999 | flat — 64-reg acc array adds pressure, cancels |
| V2 k-scale read in epilogue (drop 16-reg array) | 0.999 | 1.028 | 1.138 | 1.089 | 0.9999 | first net-positive @D256-64k |
| V3 both q&k scales inline in epilogue | 1.027 | 1.074 | 0.984 | 0.928 | 0.9999 | best D256, but D128 REGRESSES (inline divmod) |
| V4 q-array read in epilogue + k deferred | 0.993 | 1.016 | 1.104 | 1.083 | 0.9999 | balanced but D256-16k ~break-even |
| **V5 = SHIPPED: D-aware (defer q-scale iff HD>=256, else pre-array) + k deferred** | **1.043 / 1.026** | **1.059** | **1.114-1.163** | **1.076-1.081** | 0.9999 | net-positive BOTH dims |
| ceiling: scaleconst probe (real loads, 0 scale regs) | 1.083 | 1.100 | 1.201 | 1.214 | — | |
| ceiling: noload probe (0 loads, scales kept) | 1.094 | 1.118 | 1.119 | 1.106 | — | |

## Result (V5, shipped) — before → after  (confirmed over 3 back-to-back runs)
| D | L | before (op speedup) | after (op speedup, 3-run range) | cos | mag |
|---|---|---|---|---|---|
| 256 | 16384 | 0.961× | **1.035–1.044×** | 0.99992-0.99995 | 0.9995-1.0005 |
| 256 | 65536 | 0.986× | **1.060–1.064×** | 0.99992-0.99995 | 0.9995-1.0005 |
| 128 | 16384 | 1.159× | **1.109–1.115×** | 0.99992-0.99995 | 0.9999-1.0006 |
| 128 | 65536 | 1.055× | **1.074–1.089×** | 0.99992-0.99995 | 0.9999-1.0006 |

D256 cleared >1.0× (the must-hit bar) on both lengths; ~+3-6% op. Reaches ~1.04/1.06 vs the
~1.09/1.11 noload ceiling = ~60-70% of the available headroom recovered. Correctness sweep (i4 single,
i3 paged, i3 ragged) ALL_PASS, cos 0.9999, mag≈1.0 — unchanged from before tuning.

## Why V5 is D-aware (the divmod-vs-prearray flip)
The q-scale GQA `group_size.divmod` placement trades off against k-tile count:
- D256 (8 k-tiles, most register-critical): DEFER q-scale into the epilogue → nothing q-related lives
  during the loop → +D256. (Inlining the divmod is fine because the per-(mma_q,mma_kv) cost is hidden
  by the load-bound profile at 8 tiles.)
- D128 (4 k-tiles, less pressure): PRE-ARRAY q-scale (4 regs) so the divmod is off the epilogue → +D128.
  (Inlining the divmod here costs more than the 4 regs save → V3 regressed D128 to 0.93-0.98.)
`constexpr bool DEFER_QSCALE = (HD >= 256)` selects the path at compile time; the f16 path is untouched.

## What did NOT help / not worth it
- V1 register-blocking (kd-outer, preload all A/B, cross-product mma): cuts smem loads 2.7× but the
  64-int32 accumulator array held across the kd loop adds pressure that cancels the gain (flat, and
  D128 regressed). Confirms loads are not throughput-bound — they are latency-bound but reducing count
  via more live registers is a wash on a 255-reg-capped kernel.
- Lever-1 vectorized/ldmatrix load: NOT pursued — the scaleconst probe shows real loads already reach
  ~1.08-1.10×@D256 (≈ the noload ceiling), so the smem load is NOT the residual bottleneck. A faster
  load cannot beat the noload ceiling; the residual gap is register-spill + k-scale gmem latency.
- Re-introducing a per-mma_kv k-scale array to de-dup the redundant mma_q reads: reintroduces the
  16-reg pressure V2/V5 removed → net wash. The residual k-scale gmem latency would need smem staging
  (a SharedStorage field) to remove — intrusive, ≤~0.03× gain, not taken.

## Remaining gap to ceiling (~0.04-0.05× op @D256)
The kernel still spills (STACK up to 416 on high-NUM_MMA_KV variants — shared with the f16 path). The
residual is (a) spill of the shared fp16-PV/softmax state, not int8-specific, and (b) k-scale gmem
`__ldg` latency in the epilogue (8 reads, 16 unique). Closing it needs smem-staging the kv-tile
k-scales (a SharedStorage scratch field + cooperative fill + __syncwarp) — intrusive for ~3% more op
(~<1% e2e). Not taken: V5 is net-positive on both dims, correctness intact, and the lever is structurally
capped (~88% of the kernel is the shared fp16-PV+softmax that int8-QK does not touch).
