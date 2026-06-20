# I-2 NOTES — native int8-QK (IMMA m16n8k32) wired into FlashInfer FA2 compute_qk

**STATUS: I-2 FUNCTIONALLY DONE.** int8 `single_prefill_with_kv_cache(int8 q,k / fp16 o, fa2)`
matches the fp16 reference at **cos ≈ 0.9999** across all tested shapes (target was cos>0.95).
Validated on real RTX 3090 (sm_86), image `ghcr.io/avesed/vllm-ampere-optimized:v0.23.0`, FlashInfer 0.6.12.

## Final cosine results (int8 vs fp16 reference, head_dim=128)
| shape | causal | cos | finite |
|---|---|---|---|
| L=16/32/64/128/256/512, H=8 | yes | 0.99993–0.99995 | 1.0 |
| L=16, H=8 | no | 0.99995 | 1.0 |
| L=33, L=200, H=8 | yes | 0.99996 / 0.99993 | 1.0 |
| L=256, H=8, Hkv=2 (GQA g4) | yes | 0.99994 | 1.0 |
| L=128, H=4, Hkv=1 (GQA g4) | yes | 0.99995 | 1.0 |

(The `|O_i8|/|O_ref|` ratio ≈ 53–61 is the uniform V per-tensor scale 1/sv — cosine-invariant, expected.)

## What the working compute_qk int8 path does (see `i2_compute_qk.py`, `i2_prefill.diff`)
1. **Guard** `if constexpr (sizeof(DTypeQ)==1)` at the top of `compute_qk`; `return`s before the f16 path.
2. **Direct-index the b128-swizzled int8 smem (BYPASS ldmatrix)** to build the validated m16n8k32
   fragments: `g=lane>>2, t=lane&3`; head-dim looped in k32 tiles `kd∈[0,HD/32)`, `base=kd*32+t*4`;
   `A[0..3]=Q[{g,g+8}][base+{0..3 | 16..19}]`, `B[0,1]=K[n=g (+8 for 2nd n8 tile)][base+{0..3|16..19}]`.
   smem offset = `(row*UPCAST_STRIDE + (jb ^ (row&7)))*16 + e` where `jb=dcol>>4, e=dcol&15`.
3. Run `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` into an int32 accumulator (the I-0 wrapper).
4. **s32 → float** into `s_frag` in the exact C-frag order `logits_transform`/`logits_mask` expect:
   `sf[0..3]=acc[n8=0]` (kv 0-7), `sf[4..7]=acc[n8=1]` (kv 8-15); within each: c0,c1=row g, c2,c3=row g+8.
5. PV stays fp16 via FlashInfer's existing int8→f16 V upcast (unchanged).

## Three real bugs found & fixed beyond the bare QK matmul (none were in the I-0 mma layout)
1. **PV/softmax used DTypeQ (=int8) as the f16 probability/PV type** → truncated softmax probs to
   int8 → 0 → denom 0 → NaN. Fix: add `DTypeProb = conditional<sizeof(DTypeQ)==1, half, DTypeQ>`
   to KernelTraits and use it in `compute_sfm_v` for the prob fragment, the float→prob cast, the V
   upcast target, and the two PV `mma_sync_m16n16k16_row_col_f16f16f32<...>` template args.
2. **`load_q_global_smem` head-dim geometry was hardcoded 16-bit** (`mma_do < NUM_MMA_D_QK/4`,
   rewind `2*NUM_MMA_D_QK`). For int8 (16 elems/b128, 8 threads cover 128 dims in ONE step) the f16
   count did a 2nd OOB step that corrupted Q smem and dropped head-dim lanes 4/8/12. Fix: make it
   dtype-aware like `produce_kv`: count `NUM_MMA_D_QK/(8/sizeof(DTypeQ))`, rewind `sizeof(DTypeQ)*NUM_MMA_D_QK`
   (f16 unchanged). (`produce_kv` for K was already dtype-correct via `8/sizeof(DTypeKV)`.)
3. **Causal mask leaked** because FlashInfer's `math::inf` is a FINITE `5e4`, so masking only
   works when `5e4 * sm_scale_log2 >> 1`. The per-tensor scale folded into a tiny `sm_scale`
   (~1.7e-5) made the mask-fill represent only ~-1.5 in scaled space → masked logits survived
   softmax (leak ∝ masked-fraction; cos stuck ~0.67–0.74). Fix for the harness: pre-scale s_frag by
   `INT8_QK_RCP=1/256` in-kernel and multiply `sm_scale` by 256 in the caller (exact softmax,
   restores mask-fill dominance). **Production note:** real per-token dequant applies q_scale*k_scale
   in-kernel and keeps `sm_scale=1/sqrt(d)` (normal magnitude), so this normalization is a
   test-harness device, not a numerical change — but the deploy path MUST keep effective sm_scale
   O(0.01+) (don't fold a tiny scalar into sm_scale) or causal mask leaks the same way.

## Validation methodology (the diagnostics that cracked it)
- `harness_qk_smem.cu` / `test_mma_s8_layout.cu` (standalone, PASS) — necessary but NOT sufficient:
  a self-consistent write+read passes even if it mismatches the real kernel's loaders.
- `sim_swizzle.cpp` (host) — models the REAL produce_kv / load_q write addressing vs my read
  (found the load_q chunk-count assumption; once fixed, 0 mismatches for both Q and K).
- Argmax/one-hot probes (q peaks at dim m, V=readout) localized bug #2 (dropped dims 4/8/12) and
  bug #3 (causal leak ∝ masked fraction; `MASKFORCE`→-1e30 and small-int8 tests proved the 5e4 issue).

## Files (in this dir)
- `i2_compute_qk.py` — the reproducible compute_qk int8 edit (apply ON TOP of `i1_apply.py`).
- `i2_test.py` — cos validation harness (L/H/CAUSAL via env).
- `i2_prefill.diff` — full prefill.cuh diff (I-1 + I-2) vs pristine 0.6.12.
- `harness_qk_smem.cu`, `sim_swizzle.cpp` — the de-risking harnesses.
- `fi_headers/` — pristine 0.6.12 headers pulled from the image for reference.

## Dev loop (sandbox, per `--rm` container)
```
ssh -p 60022 -i ~/.ssh/ssh-key coder@192.168.100.1 \
 'docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e HOME=/out \
   -v /mnt/coder/workspaces/trevor/d2m:/out --entrypoint bash \
   ghcr.io/avesed/vllm-ampere-optimized:v0.23.0 -lc "
     python3 /out/i1_apply.py && python3 /out/i2_compute_qk.py &&
     rm -rf /out/.cache/flashinfer &&
     L=256 H=8 CAUSAL=1 python3 /out/i2_test.py 2>&1 | tail -3"'
```

## Recommended next steps (toward I-3/I-4)
- I-3 (paged): the shared `compute_qk` means the int8 path is reused; validate the paged kernel
  (`BatchPrefillWithPagedKVCacheKernel`) — the same `tid`/warp-base logic and DTypeProb fixes apply
  to its compute_sfm_v/load too. Re-run the cos harness via the paged wrapper.
- I-4 (vLLM e2e): for real per-token int8, apply q_scale[token]*k_scale[token] dequant INSIDE
  compute_qk (read scales from a small smem/gmem buffer indexed by the C-frag q/kv formula from
  `logits_transform`), keeping sm_scale=1/sqrt(d) — this both fixes the magnitude/mask issue
  cleanly AND gives true per-token accuracy. Plumb q_scale/k_scale as fa2 additional tensors
  (I-1 E2/E3 — currently NOT done; the harness sidesteps it with per-tensor + sm_scale folding).

---

# I-4a + I-3 NOTES — PRODUCTION per-token in-kernel dequant + scale plumbing + paged path

**STATUS: I-4a DONE + I-3 (paged AND ragged) DONE.** The I-2 test HACK (per-tensor quant, sq*sk
folded into sm_scale, in-kernel `INT8_QK_RCP=1/256` magnitude normalization → |O_i8|/|O_ref|≈53) is
REMOVED. Real per-token symmetric int8 dequant is applied IN-KERNEL; caller passes
`sm_scale=1/sqrt(d)` unchanged. Validated on real RTX 3090 (sm_86), image v0.23.0, FlashInfer 0.6.12.

## Final cos + magnitude (REAL per-token scales, all PASS = cos>0.99 AND mag∈[0.95,1.05])
- **I-4a single_prefill** (`i4_sweep.py`, 16 cfgs D∈{128,256} × L∈{256,2048} × causal/non × MHA/GQA-g4):
  ALL PASS — cos 0.99992–0.99995, **mag 0.9995–1.0005** (was 53× with the hack → real dequant proven).
- **I-3 paged** (`i3_sweep.py`, 11 cfgs incl non-page-aligned L=200/333, page 16/32, D256, GQA):
  ALL PASS — cos 0.99992–0.99995, **mag 0.9997–1.0008**.
- **I-3 ragged** (`i3_ragged_sweep.py`, 11 cfgs): ALL PASS — cos 0.99992–0.99995, **mag 0.9995–1.0004**.
- Representative single point (D=128, L=2048, causal, GQA-g4): cos 0.99994, mag 1.0006.

## The dequant + token-index formula that worked (the crux)
In `compute_qk`'s `if constexpr (sizeof(DTypeQ)==1)` branch, after the s8s8s32 IMMA:
```
sf[reg] = (float)acc[reg] * q_scale[q_idx(reg)] * k_scale[kv_idx(reg)]   // natural f16 magnitude
```
The C-frag (reg 0..7) → (q_row, kv_col) mapping MIRRORS `logits_transform` (prefill.cuh~L949), with
g=lane>>2, t=lane&3:
- **packed q row** = `qo_packed_idx_base + mma_q*16 + g + 8*((reg%4)/2)` → `group_size.divmod` → `q_idx`
  (the divmod de-groups GQA exactly as logits_transform does). reg{0,1,4,5}=row g; reg{2,3,6,7}=row g+8.
- **kv_idx** = `kv_idx_base + mma_kv*16 + 2*t + 8*(reg/4) + (reg%2)`  (logical kv position in request).
- `kv_idx_base = chunk_start + (iter*NUM_WARPS_KV + warp_kv)*NUM_MMA_KV*16` — HOISTED above the
  compute_qk call at all 3 sites (it was originally computed just AFTER) and passed in.
Caller keeps `sm_scale=1/sqrt(d)`: `sm_scale_log2 ≈ 0.18 (d128)/0.09 (d256)` → finite mask-fill
(−5e4) dominates the causal mask (the I-2 bug-3 leak is gone WITHOUT the ×256 hack). OOB tile padding
(q_idx≥qo_len / kv_idx≥kv_len) reads scale 1.0 (bounded deref, no OOB memory access) and is then
overwritten by `logits_mask` → MaskFillValue before softmax. Perf: q-scale (per mma_q, via divmod) and
k-scale (per mma_kv, 4 gmem __ldg) are PRECOMPUTED ONCE into register arrays before the mma loops
(re-reading per (mma_q,mma_kv) regressed perf to 0.79–0.94×; hoisting recovered it).

## Scale plumbing (i4_apply.py) — q_scale/k_scale as fa2 ADDITIONAL TENSORS (I-1 E2/E3/E8)
- **modules.py** (P1): `gen_single_prefill_module` + `gen_batch_prefill_module` fa2 branches — when
  `dtype_q==torch.int8`, APPEND `["maybe_q_scale","maybe_k_scale"]` / `["float","float"]` to
  `additional_tensor_names/dtypes`. `generate_additional_params` then AUTO-emits in the generated config:
  Params decl `float* maybe_q_scale; float* maybe_k_scale;`, the `Optional<ffi::Tensor>` func params,
  and the nullptr-tolerant setter `params.maybe_q_scale = maybe_q_scale ? ...data_ptr() : nullptr;`.
- **prefill.py** (P2): the 3 fa2 C++ run calls (`run`/`ragged_run`/`paged_run`) forward `scale_q,scale_k`
  positionally right after `maybe_*_cache_sf` (== the new tensors' slot) when `q.dtype==int8`.
- **prefill.py** (P3/P4): paged + ragged WRAPPER `.run()` extract per-token `scale_q,scale_k` from
  `*args` for int8 q (mirrors the existing fp8 extract); `wr.run(q_i8, kv, scale_q, scale_k)`.
- **compute_qk** reads `params.maybe_q_scale/maybe_k_scale` via SFINAE-guarded accessors
  `get_q/k_dequant_scale<KTraits>(params, request_idx)` (helper injected INSIDE `namespace flashinfer`
  — must NOT re-open the namespace or you get `flashinfer::flashinfer` and `::Error` breaks). For batch
  the q-scale is offset by `params.q_indptr[request_idx]` (field detected via SFINAE; single Params lacks
  it → offset 0). f16 modules lack the fields → `i4_has_maybe_q_scale<Params>` is false → nullptr → 1.0
  fallback → f16/other dtypes unaffected.

## TWO bugs found & fixed beyond the dequant math
1. **Helper re-opened `namespace flashinfer`** → nested `flashinfer::flashinfer::Error` → "namespace has
   no member Error". Fix: inject the SFINAE helpers WITHOUT a namespace wrapper (the insertion point is
   already inside the file's `namespace flashinfer{...}`); add `#include <type_traits>/<utility>` at
   global scope (before the namespace) for `std::void_t`/`std::declval`.
2. **Ragged wrapper forced bf16 output for ALL 1-byte q** (`out_dtype = bfloat16 if q.dtype.itemsize==1`
   — a STOCK FlashInfer fp8 assumption). Our int8 module is compiled with `o_data_type=float16`
   (DTypeO=half) → kernel writes half bits into a bf16 buffer → bit-reinterpret → garbage (cos 0.14,
   |O|~1e10–1e14). This was NOT a kernel bug: the kernel's acc/sf/m/d/o_frag were all PROVEN correct
   (probe-matched the single kernel; partition_kv=0). Manifested ONLY via the ragged wrapper's internal
   `out=None` alloc; passing `out=` explicitly already worked. Fix (P5): for `q.dtype==int8`, honor
   `self._cached_o_data_type` (set in `plan()`), not bf16. Single_prefill + paged wrapper have different
   out logic → were never hit. (Debug method: probe printf of scales→non-null+correct; of acc/sf→correct;
   of m/d/o_frag→correct & finite; → isolated `O.dtype==bfloat16` despite `_cached_o_data_type==float16`.)

## Perf (real 3090, per-token scales, `i4_time.py`, cuda-event median, causal H8)
| D | L | fp16 ms | int8 ms | op speedup |
|---|---|---|---|---|
| 128 | 16384 | ~9.3 | ~8.4 | **1.08–1.12×** |
| 128 | 65536 | ~130.7 | ~123.2 | **1.06×** |
| 256 | 16384 | ~17.9 | ~18.6 | **0.95–0.97×** |
| 256 | 65536 | ~289 | ~295 | **0.98×** |
D=128 matches/exceeds the I-2 hack (1.20×@16k / —) and the project's ~1.05–1.13× op band. **D=256 is a
~3–5% op REGRESSION vs the hack's 1.065×** (the hack used a single per-tensor constant = 0 gmem reads;
per-token needs the divmod + 4 k-scale gmem reads/kv-tile + scale-array register pressure, and D256 has
8 k-tiles so the QK fraction is thin). This is consistent with the documented STRUCTURAL ceiling (≈88%
of the kernel is the shared fp16-PV+softmax; QK-matmul is only ~8–12%) — D256 op-perf was already only
~1.05–1.09× even with the hack, so the per-token machinery erodes that thin margin. **[SUPERSEDED by the
I-4b tuning below — D256 is now NET-POSITIVE.]**

---

# I-4b NOTES — D256 perf TUNING (the 0.95×@D256 regression FIXED, real RTX 3090 sm_86)

**STATUS: D256 cleared the >1.0× must-hit bar and is net-positive on both lengths. Correctness intact
(i4 single + i3 paged + i3 ragged sweeps ALL_PASS, cos 0.9999, mag≈1.0 — unchanged).** Full table +
variant sweep + ceiling-probe evidence in repo `i4_tuned_results.md`. The tuned `compute_qk` is the
single source of truth in `i4_compute_qk.py` (still applies ON TOP of i1_apply.py + i4_apply.py).

## Before → after (op speedup, fp16-FlashInfer / int8-per-token, cuda-event median, causal H8)
| D | L | BEFORE | AFTER (shipped V5) | cos | mag |
|---|---|---|---|---|---|
| 256 | 16384 | 0.961× | **1.026–1.043×** | 0.9999 | ≈1.0 |
| 256 | 65536 | 0.986× | **1.059×** | 0.9999 | ≈1.0 |
| 128 | 16384 | 1.159× | **1.114–1.163×** | 0.9999 | ≈1.0 |
| 128 | 65536 | 1.055× | **1.076–1.081×** | 0.9999 | ≈1.0 |

## Root cause = REGISTER PRESSURE, not the load (the prompt's lever-1 was a red herring here)
cuobjdump `-res-usage`: the kernel is at the 255-reg cap WITH stack spill (STACK:48–416 on high-
NUM_MMA_KV configs, on the f16 path too). Two perturbation probes on the PRODUCTION (per-token,
real-scale) kernel:
- `noload` (smem reads→arith hash, IMMA+scales kept): D256 **1.094×@16k / 1.118×@64k**.
- `scaleconst` (REAL loads kept, scales forced const = ZERO scale regs): D256 **1.083×@16k / 1.100×@64k**.
Real loads already reach ~the noload ceiling once the scale registers are removed ⇒ the smem load is
NOT the bottleneck; the per-token SCALE-REGISTER PRESSURE (16 k-scale + q-scale floats held across the
load+mma loop → spills) is. So lever-2 (keep scales off the register-critical loop) is the fix; lever-1
(vectorize/ldmatrix the load) cannot beat the noload ceiling and was correctly skipped.

## The fix (V5, shipped): keep ALL scale state off the load+mma loop
- **k-scales**: never pre-expanded into the 16-reg `ks*_a[NUM_MMA_KV]` arrays; read 4 `__ldg` per
  mma_kv in the dequant EPILOGUE (after the kd/mma loop, when A/B load fragments are dead).
- **q-scales**: D-AWARE via `constexpr bool DEFER_QSCALE = (HD >= 256)` — the GQA-divmod-vs-prearray
  trade flips with k-tile count. D256 (8 k-tiles, register-critical): compute q-scale inline in the
  epilogue (nothing q-related lives during the loop → +D256). D128 (4 k-tiles): pre-array up-front
  (4 regs) so the divmod is off the epilogue (→ +D128; inlining it regressed D128 to 0.93–0.98). f16
  path untouched.

## What did NOT help
- kd-outer register-blocking (preload all A/B per kd, cross-product mma): cuts smem loads 2.7× but the
  64-int32 accumulator held across the kd loop adds pressure that cancels it (flat; D128 regressed).
  Confirms loads are latency- not throughput-bound, and adding live regs on a 255-reg-capped kernel is
  a wash.
- Re-introducing a per-mma_kv k-scale array to de-dup redundant mma_q reads: reintroduces 16-reg
  pressure = net wash. smem-staging the kv-tile k-scales (a SharedStorage field + cooperative fill +
  __syncwarp) could claw the residual ~0.04× toward the ceiling but is intrusive for <1% e2e — not taken.

## Residual gap (~0.04–0.05× op @D256, 1.04/1.06 vs ~1.09/1.11 ceiling)
Shared fp16-PV/softmax spill (not int8-specific) + k-scale gmem `__ldg` latency in the epilogue. The
lever is structurally capped (~88% of the kernel is the shared fp16-PV+softmax int8-QK never touches).
Net-positive on both dims with correctness intact is the deliverable; chasing the last ~3% needs smem
staging for <1% e2e.

## Multi-request batch (NOTE for vLLM I-4): single-request validated; for >1 request the q-scale offset
is handled (`params.q_indptr[request]`), but the **k-scale per-request offset is NOT yet plumbed** — the
kernel indexes `maybe_k_scale[kv_idx]` with kv_idx logical-within-request and offset 0 (no
`maybe_kv_scale_indptr` field exists; SFINAE → 0). For multi-request batch the caller must pass a
request-local k-scale view OR add a `maybe_kv_scale_indptr` Params field + offset (q-scale already shows
the pattern). The matmul/dequant math is correct; only the global→request k-scale tensor offset remains.
**[CLOSED by I-5 below — `maybe_kv_scale_indptr` plumbed; multi-request paged + ragged validated.]**

## smooth_k composition
smooth_k = subtract per-(head,channel) mean of K BEFORE int8 quant (caller side, as in the tests). The
mean term is a per-kv-row-constant logit shift → absorbed by softmax row-shift invariance, so it composes
trivially with per-token k_scale (the kernel just sees the smoothed-then-quantized K and its per-token
scale). No in-kernel smooth_k handling needed.

## Files (I-4a/I-3, in this dir)
- `i4_apply.py` — scale plumbing (modules.py P1, prefill.py P2/P3/P4/P5). Apply AFTER i1_apply.py.
- `i4_compute_qk.py` — per-token dequant compute_qk + 3 call sites + SFINAE scale accessors.
  SUPERSEDES i2_compute_qk.py (contains all i2 fixes: DTypeProb, compute_sfm_v, load_q geometry).
- `i4_test.py` / `i4_sweep.py` — I-4a single cos+magnitude (env L/H/HKV/D/CAUSAL ; sweep = full matrix).
- `i3_test.py` / `i3_sweep.py` — I-3 PAGED cos+magnitude (BatchPrefillWithPagedKVCacheWrapper fa2).
- `i3_ragged_test.py` / `i3_ragged_sweep.py` — I-3 ragged cos+magnitude (BatchPrefillWithRagged…).
- `i4_time.py` — perf (fp16 vs int8 per-token) op-level.
- `i4_diffs/{prefill_cuh,prefill_py,modules_py}.diff` — applied diffs (i4 layer, on top of i1).
- `fi_src/` — modules.py/prefill.py/jit_attention_utils.py/csrc templates/variants.cuh pulled from the
  image for reference (the exact 0.6.12 sources the edits anchor on).

## Dev loop (I-4a/I-3, per `--rm` container)
```
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e HOME=/out \
  -v /mnt/coder/workspaces/trevor/d2m:/out --entrypoint bash IMG -lc "
    python3 /out/i1_apply.py >/dev/null && python3 /out/i4_apply.py >/dev/null &&
    python3 /out/i4_compute_qk.py >/dev/null && rm -rf /out/.cache/flashinfer &&
    python3 /out/i4_sweep.py 2>&1 | tail -3 && python3 /out/i3_sweep.py 2>&1 | tail -2"
```
(Sandbox workspace `~`==`/home/coder` == container `/out`; the prompt's `/mnt/coder/.../d2m` host path
and ssh `~` are bind-mounted to the SAME dir. Re-apply edits each `--rm` run; `rm -rf .cache/flashinfer`
between header edits or a stale .so masks changes.)

---

# I-4b CHUNKED (cached-prefix) NOTES — int8 on the cached-prefix chunks; 128k single-card

**STATUS: chunked / cached-prefix int8-QK path BUILT + e2e VALIDATED (real Qwen3.5-9B-W4A8, single
RTX 3090).** Extends the I-4b backend so int8 fires on cached-prefix chunks (num_computed_tokens>0),
not just pure-fresh prefill. This lets CHUNKED prefill fit 128k on ONE card AND run int8 on every
chunk. Full table + the qo<kv alignment proof in repo `I4B_PAGED_RESULTS.md`.

## What changed in int8qk_backend.py
- Dropped the `_is_pure_fresh_prefill` hard gate. Now per-request: route DECODE rows (q_len==1 &&
  seq_len>1) to FA; route PREFILL chunks (fresh or cached) through int8.
- Cached-prefix path: reshape_and_cache_flash writes the new chunk's K/V → GATHER the full context
  K/V (first seq_len tokens) from the paged cache via block_table → int8 single_prefill(q[chunk],
  k[full], v[full], causal=True). qo<=kv + causal aligns q to the END of kv (the chunked-prefix
  semantics). Pure-fresh chunks skip the gather (in-hand K/V; bit-identical to old path).
- Per-request fresh/cached fire counters (INT8QK_FIRE["fresh"/"cached"]) so the harness PROVES int8
  fired on the cached chunks.

## qo<kv causal-alignment (THE #1 correctness risk) — VERIFIED in isolation (i4b_align_test.py)
single_prefill(q=chunk[C], k/v=full[S], causal=True) == last-C rows of the full causal prefill:
cos **1.00000** for both fp16 AND int8, 6 splits incl page-unaligned; negative control (vs FIRST C
rows) cos 0.156. End-aligned semantics confirmed — exactly what chunked prefill needs.

## Memory: STREAMING quant required to fit 128k single-card (the OOM fight)
Naive full-context fp32 quant temporaries (~1GB per K and per V at 128k) OOMed (25–447 MiB free at
failure; the greedy KV pool eats all GPUMEM slack, leaving only FA's profiled activation headroom,
which the larger int8 transients exceed). FIX: (1) gather+quant K and V SEPARATELY with eager `del`
(never hold k_bf16 + k_fp32 + v_bf16 at once); (2) STREAM the per-token int8 quant over the seq in
16384-row slices → fp32 working buffer bounded by one slice (~128MB) not the full context. Numerics
identical (per-token scale = row reduction; smooth_k mean computed full-context first). Fits at
GPUMEM 0.88, chunk 8192/16384 single-card. needle correct at every length.

## TTFT (single-card chunked, int8 vs FA, matched config; int8 INCLUDES gather+quant overhead)
| len  | chunks  | FA TTFT | int8 TTFT | speedup | needle | fire/run |
|------|---------|---------|-----------|---------|--------|----------|
| 16k  | 4       | 2.637s  | 2.704s    | −2.5%   | OK     | 8+24     |
| 64k  | 4       | 14.054s | 13.964s   | +0.6%   | OK     | 8+24     |
| 128k | 16(8k)  | 37.049s | 36.895s   | +0.42%  | OK     | 8+120    |
| 128k | 8(16k)  | 37.181s | 36.532s   | +1.75%  | OK     | 8+56     |
| 128k | 6(24k)  | 37.276s | 36.511s   | **+2.05%** | OK  | 8+40     |

**256k single-card = HARDWARE CEILING, NOT an int8 limit: FA AND int8 OOM identically (256k KV pool
~21GB + 9.5GB weights > one 24GB 3090; chunking shrinks ACTIVATION not the pool, which must hold all
256k tokens). Model RoPE max_position_embeddings = 262144. So 128k is the single-card max for this
model on a 3090 and int8 is net-positive there (+2.05% at the best-fitting chunk). 256k needs 2 cards.**

## HONEST finding: the per-chunk gather+re-quant tax erodes the int8 win — fewer/larger chunks help
The single-STEP I-4b trend projected ~+4%@128k. CHUNKED int8 is lower because the gather+single_prefill
approach RE-GATHERS + RE-QUANTIZES the GROWING full context every chunk (chunk N quantizes all N·MBT
tokens in PyTorch) — an O(Σ context) host-side tax that FA's fused varlen kernel avoids (it reads the
paged cache in-kernel). The tax scales with CHUNK COUNT: 16 chunks → +0.42%, 8 chunks → +1.75%,
6 chunks → +2.05% at 128k.
So larger chunks (fewer re-gathers) recover the lever toward the single-step trend, bounded by per-chunk
activation memory. The int8-QK matmul lever itself is real + net-positive; the chunked MEMORY enabler
carries the gather/quant cost. **The landing deliverable: chunked+cached-prefix int8 makes 128k
single-card prefill FIT and stay coherent (needle correct, int8 on ALL chunks) — the prior path OOMed
>64k single-card.** For max int8 benefit use the LARGEST chunk that fits.

---

# I-5 NOTES — per-request k-scale offset for MULTI-request batched int8-QK (the gap CLOSED)

**STATUS: DONE + validated on real RTX 3090 (sm_86), image v0.23.0, FlashInfer 0.6.12.** The I-4b
flagged gap ("k-scale per-request offset is NOT plumbed; single-request correct, multi-request needs a
request-local k-scale view") is CLOSED. A batched prefill of N requests with DISTINCT per-request
per-token k-scales is now numerically correct for EVERY request in both the paged and ragged fa2 paths.

## The bug (confirmed before fixing — req0 OK, req1..N-1 corrupted)
The kernel reads `maybe_k_scale + maybe_kv_scale_indptr[request_idx]`, but `maybe_kv_scale_indptr` was
never declared (SFINAE → offset 0), so EVERY request used request-0's per-token k-scale segment.
q-scale was already offset by `params.q_indptr[request_idx]` (correct), so req0 stayed correct and the
error grew with request index. The trap: with statistically-similar per-request scales the error only
nicks cos ~1e-4 (FALSE PASS on a weak test). Built `i5_paged_test.py`/`i5_ragged_test.py` with DISTINCT
per-request K magnitudes (KMAG=[0.1,1.0,8.0]) so a wrong k-scale offset is unmistakable:
- BEFORE (current recipe, N=3 L=128/333/777, GQA g4, D128, causal):
  - paged: req0 cos 0.99995 PASS / req1 cos **0.98234** FAIL / req2 cos **0.30845** FAIL
  - ragged: identical (req2 cos 0.308, mag 0.715)

## The fix — `maybe_kv_scale_indptr` (kv-TOKEN prefix sum), mirror of the q-scale offset
- **i4_apply.py P1 (modules.py BATCH branch)**: append `maybe_kv_scale_indptr` / `int32_t` to the int8
  batch additional tensors (after maybe_k_scale). `generate_additional_params` auto-emits the
  `int32_t* maybe_kv_scale_indptr;` Params field + `Optional<ffi::Tensor>` func param + nullptr-tolerant
  setter — exactly like the other `maybe_*` tensors. (single_prefill is one request → no batch field.)
- **i4_apply.py P6**: add trailing `scale_kv_indptr=None` param to `paged_run`/`ragged_run` (the
  registered ops in `get_batch_prefill_module`; `register_custom_op` is a no-op passthrough in this build
  so KEYWORD args work). The int8 C++ forwarding (P2) passes `scale_kv_indptr` right after `scale_k` so
  it maps to `maybe_kv_scale_indptr`.
- **i4_apply.py P8 (paged wrapper run)**: derive the kv-TOKEN prefix sum from the planned
  `_paged_kv_indptr_buf` (pages) + `_paged_kv_last_page_len_buf` + `page_size`
  (`kvlen = (pages-1)*page_size + last_page_len`, cumsum), pass `scale_kv_indptr=` by keyword.
- **i4_apply.py P9 (ragged wrapper run)**: `self._kv_indptr_buf` IS the kv-token prefix sum → pass it
  directly. Both P8/P9 guard the keyword on `self._jit_module is None` (the user-jit passthrough uses
  `*args`, no such param).
- **i4_compute_qk.py (get_k_dequant_scale)**: the SFINAE offset already existed; added a **runtime
  nullptr guard** on `maybe_kv_scale_indptr` because the field now ALWAYS exists for int8 batch modules
  but is nullptr for single-request / callers that don't pass it (→ offset 0, bit-identical to before).
Caller CONTRACT unchanged from single-request: pass flat `scale_q` [Σqo] + flat `scale_k` [Σkv] in
request-major logical order; the wrapper auto-derives both per-request offsets (q via its own
qo_indptr, k via the new kv_scale_indptr). Layout = exactly the SFINAE accessor expects.

## Final cos + magnitude (DISTINCT per-request K mags; PASS = EVERY req cos>0.99 AND mag∈[0.95,1.05])
- **AFTER fix, N=3 L=128/333/777 GQA g4 D128 causal** (the prior FAIL case):
  - paged:  req0 cos 0.99995 / req1 cos **0.99993** / req2 cos **0.99982** — ALL PASS (was 0.982/0.308)
  - ragged: req0 0.99995 / req1 0.99993 / req2 0.99982 — ALL PASS
- **i5_sweep.py** (paged + ragged each): N∈{3,5,8}, GQA groups {g1,g2,g4,g8,MHA}, D∈{128,256},
  causal+non-causal, page {16,32}, plus a qo<kv APPEND case (qo_len<kv_len end-aligned, D128+D256) —
  **ALL_PASS, worst-request cos 0.99969** across every config. Single-request regression (i4 single /
  i3 paged / i3 ragged sweeps) re-run: ALL_PASS, cos 0.9999, mag ~1.0 — no regression.
- **Perf** (single_prefill, real-scale, cuda-event median, causal H8): D128 L16384 **1.166×**,
  D256 L16384 **1.037×** — within the documented bands (no regression; the offset is one
  nullptr-guarded pointer-add per CTA, not per element).

## Other int8-dtype gap found + closed during the audit: head_dim 64
While sweeping head_dim {64,128,256}, **head_dim=64 int8 was found PRE-EXISTING broken** (never
validated — prior sweeps were D128/D256 only): single-request D64 cos 0.91–0.98 (NOT a multi-request
issue). Root cause: for int8 KV with `HEAD_DIM_VO==64` the KV smem uses the **k64B swizzle** (`off =
i*4 + (j ^ ((i/2)%4))` + a different token→row layout), whereas the int8 IMMA read assumed k128B
(`j ^ (i%8)`). Refactored the int8 `ldq`/`ldk` to use the canonical `smem_t::get_permuted_offset<US>`
(swizzle-mode-correct by construction; behaviorally identical for the deployed k128B D128/D256 →
0 perf/accuracy change). The k64B path ALSO needs a different token-row mapping (k64B `produce_kv`
uses `kv_idx = base + warp*8 + lane/4`, 2-row interleave), which the int8 read does not implement, so
D64 is **GUARDED unsupported**: `gen_*_prefill_module` asserts `head_dim_vo != 64` for int8 q (loud
error, never silent garbage). head_dim 64 is in NO deployed int8-QK path (the vLLM backend targets
hd256; models are hd128/hd256), so guarding is the complete+stable resolution; a real k64B
implementation is left as future work if an hd64 int8 model ever appears. Validated: i5_sweep's
"head_dim 64 guard" case confirms the assert fires.

## Files (I-5, in this dir)
- `i5_paged_test.py` / `i5_ragged_test.py` — the N≥3 multi-request gate (distinct per-request K mags;
  per-request cos+mag vs each request's fp16 single reference). env: H/HKV/D/PAGE/CAUSAL/LENS/KMAG.
- `i5_sweep.py` — full multi-request matrix (paged+ragged, N/GQA/D/causal/page/qo<kv) + the D64 guard.
- Recipe edits: `i4_apply.py` (P1 modules.py kv_scale_indptr + D64 guard; P2 C++ scale_kv_indptr
  forward; P6 run() signatures; P8 paged wrapper kv-token prefix-sum; P9 ragged wrapper kv_indptr) and
  `i4_compute_qk.py` (get_permuted_offset swizzle-correct ldq/ldk + kv_scale_indptr nullptr guard).
- Vendored `flashinfer/` regenerated; `apply_to_source.py` on a FRESH `git clone --depth 1 --branch
  v0.6.12` reproduces all 5 recipe-touched files **byte-identical** to the committed tree (all anchors
  match) — verified.

## Dev loop (I-5, per `--rm` container) — same as I-4
```
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e HOME=/out \
  -v /mnt/coder/workspaces/trevor/d2m:/out --shm-size=2g --entrypoint bash IMG -lc "
    python3 /out/i1_apply.py && python3 /out/i4_apply.py && python3 /out/i4_compute_qk.py &&
    rm -rf /out/.cache/flashinfer && python3 /out/i5_sweep.py 2>&1 | tail -30"
```
