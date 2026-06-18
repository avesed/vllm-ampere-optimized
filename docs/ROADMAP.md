# vllm-ampere-optimized — optimization roadmap

**Scope (locked):** optimize the *old hardware* (Ampere sm_80 A100 / sm_86 3090·A40·A6000·A10 —
big install base, still-decent compute) *for modern model architectures* (hybrid linear-attn like
GatedDeltaNet/Mamba2 + a few full-attn layers, large head_dim ~256; modern MoE). Every item must
clear three bars: (1) generalizes across the Ampere line, (2) ships in the fork (patch / pinned
per-arch dep / tuned-config data / tooling), (3) serves modern architectures — **not** old
pure-full-attention dense, **not** per-deployment config.

Status legend: ✅ shipped · 🔨 build-now · ⏳ gated on measurement · 🧪 eval-then-ship · ❌ NO-GO.

---

## Already shipped ✅
- **W4A8-int8 Marlin un-gate** (patch 0001) + int8 8-row decode tile (patch 0002). The flagship.
  Measured: decode = W4A16 parity (weight-bandwidth-bound), prefill +15-18% (int8 TC).
- **RTX-3090 fused_moe Triton config** (configs/fused_moe). NOTE: used by the **Triton** fused-MoE
  path (bf16/fp16/int8_w8a8), confirmed triggered. NOT read by the Marlin `moe_wna16` path. Keep;
  document which path it serves. (Earlier "delete it as dead" was wrong — it's live for Triton MoE.)
- **Diagnostic harness (this session):** `benchmarks/bench_marlin_gemm_imma.py` (ncu IMMA occupancy),
  `benchmarks/torch_prof_phase.py` + `analyze_torch_prof.py` + `prof_decode_batchsweep.py` (kineto
  decode/prefill kernel-share, comm-excluded). Reusable "is this lever worth it" tooling.

## Tier A — build-now, low-risk (the real shippable value = anti-regression + tooling) 🔨
The deep-dives concluded the kernel-perf space is largely already captured upstream; the durable
fork value is *guaranteeing the captured wins don't silently regress on Ampere across upstream bumps*.
1. **Triton-on-Ampere kernel CI — BUILT + sm_86-VALIDATED ✅.** The fla/mamba/causal-conv1d/GDN kernels
   are JIT Triton (not AOT) — `git apply --check`/source build prove nothing about Ampere compile/numerics.
   Smartest impl: RUN the UPSTREAM mamba/GDN kernel pytest suite on the Ampere runner against the shipped
   image (upstream CI doesn't gate consumer-Ampere). Shipped: `scripts/ampere_kernel_ci.sh <image> <tag>`
   (host sparse-clones the tag's tests/, mounts ro into the image, `pip install pytest einops tblib`, runs
   the vetted subset; skips w/o GPU) — run locally after `build_image_source.sh` (no CI auto-build).
   Validated on real RTX 3090 (sm_86): test_fused_gdn_post_conv + test_causal_conv1d = 228
   passed; test_mamba_ssm = 286 passed; test_mamba_ssm_ssd / test_ssu_dispatch (8) / test_mamba_ssm_configs
   (6) pass; test_gdn_forward_core_split is sm_86-gated→skipped→excluded. Combined: 641 cases collect clean.
   E2E-VALIDATED against the SHIPPED image: pulled ghcr overlay v0.23.0, ran the subset inside it on
   sm_86 → **622 passed, 19 skipped, exit 0** (~27min, cold-Triton-JIT slow but green). The e2e caught +
   fixed a real bug: the official image has NO `python` (only `python3`) — script now uses `python3 -m
   pip`/`python3 -m pytest` + `--entrypoint /bin/bash`. Push-ready (test deps installed in-container:
   pytest einops tblib). Runs locally ~27min cold (after build_image_source.sh, per-release).
2. **Ampere runtime path-selection smoke — BUILT + WIRED; sandbox-validation pending 🔨.**
   `scripts/ampere_defaults_check.sh <image>` loads a tiny hybrid (Qwen3.5-0.8B-Base) in the shipped image
   on an Ampere GPU and asserts the runtime SELECTS the right paths: GDN prefill → Triton/FLA, attention →
   FlashAttention (catches a regression class the kernel tests can't — an upstream bump silently routing
   Ampere to a slow/broken path). Run locally alongside the kernel CI after a build. (Dropped the
   triton≥3.4 + check_shared_mem asserts as redundant — the kernel CI already exercises both.) REMAINING:
   sandbox-validate the in-image model-load smoke (was blocked by a transient Bash classifier outage).
3. **Ship the diagnostic harness ✅** — `benchmarks/README.md` written (documents the ncu IMMA-occupancy,
   comm-excluded decode-share batch-sweep, GDN kernel sweeps + their decision rules + the "decode is
   bandwidth-bound, measure before patching" lesson). Tools already in `benchmarks/`.
4. **Docs ✅** — README now has "Other automatic Ampere fast paths" (W8A16-fp8 Marlin auto <sm89 ~1.6x;
   AllSpark uint8b128 W8A16 small-M decode; W8A8-int8 CUTLASS cap75) + "Hybrid / linear-attention models"
   (fla/causal-conv1d vendored → don't pip-pin; CI-guarded; GDN state bf16 by default). "Also included"
   corrected: fused_moe configs serve the Triton MoE path, NOT the Marlin moe_wna16 path.

## Tier B — GDN recurrent decode: launch-tuning DEAD by measurement ❌; only bf16-state remains 🧪
5. **GDN `fused_recurrent_*packed_decode` launch-constant tuning — DEAD.** Decode-share is real (W4A8-9B,
   tp1: 2.0/10.4/15.7/**21.6%** of non-comm at batch 1/16/32/64), BUT the micro-bench
   (`benchmarks/bench_gdn_recurrent_decode.py`, sm_86) swept BV×num_warps×num_stages (48 cfgs/batch) and
   found **best vs default = 1.00-1.01x = zero**. The hardcoded `num_warps=1` is already optimal. Reason:
   the kernel is **fp32-state-bandwidth-bound** — state `[HV,V,K]=[32,128,128]` fp32 = 2MB/seq/step; at
   batch 64, 318us ≈ 85% of the state read+write roofline. num_warps/stages/BV don't change bytes moved.
   (Glad we micro-benched before writing a patch.)
6. **bf16 recurrent state — NOT a lever: it is ALREADY the default ❌** (resolved by source recon). The
   Ampere decode path (`qwen_gdn_linear_attn.py:1684`) passes `ssm_state` directly to fused_recurrent with
   NO fp32 cast (the `.to(float32)` at :267 is the FlashInfer PREFILL path, unused on Ampere), and
   `linear_attention_state_dtype = get_kv_cache_torch_dtype("auto", model_dtype) = bf16`. So the model
   already runs bf16 recurrent state (the perf bench's "fp32 baseline" is a config the model never uses;
   the recurrent kernel is already bf16-bandwidth-optimal). A "bf16 patch" would be a no-op. Sub-bf16
   (fp8/int8) state = the recurrent-accumulation NO-GO. **=> Tier B fully closed; the entire linear-attn
   kernel-efficiency area has no shippable kernel lever in v0.23.** Tooling kept: `benchmarks/bench_gdn_recurrent_decode.py`,
   `bench_gdn_state_dtype.py`.

## Tier C — eval-then-ship config data 🧪
6. **Mamba2 `selective_state_update` Ampere tuned-JSON.** The one real *structural* gap: upstream ships
   only Hopper/Blackwell JSON; Ampere falls to a heuristic ("Performance might be sub-optimal!"); this
   kernel cannot runtime-autotune (overwrites in-place state) → JSON is the only lever. Covers 7 Mamba2
   families (Nemotron-H/Bamba/Jamba/Zamba2/FalconH1/GraniteMoE-Hybrid/Plamo2). Does NOT help Qwen3.5/3.6
   GDN. Needs real-Ampere `benchmark_selective_state_update.py --save-configs --all-dstates`; sm_86 now,
   sm_80 blocked (no A100 runner). Entry: `mamba/ops/mamba_ssm.py:36-153`, `configs/selective_state_update/`.
7. **Better 4-bit PTQ recipes** (AWQ+GPTQ desc_act / rotations) in `quantize/` — buys accuracy headroom,
   loads into the shipped W4A8 path. 0 tok/s, accuracy-only.

## NO-GO — confirmed dead, stop re-chasing ❌
- **SageAttention / int8-QK attention** — only helps D≤128 pure-full-attn (old arch); D=256 unsupported;
  linear-attn layers have no softmax to accelerate. Out of scope. (Kernel + math validated, just not the target.)
- **int8-GEMM kernel work** (QServe int8-domain dequant, large-M tile sweep, Stream-K) — measured Marlin
  prefill 68% IMMA vs W8A8 82-88% but only ~1.7% wall-clock gap → ≤2% prefill, tapped out.
- **int8 / quantized GDN recurrent** — fp32 rank-1 (no tl.dot for TC) + error accumulation over 10-30k steps.
- **W4A4 / W3 / W2** — no Ampere int3/2/4-act Marlin kernel; Qwen-class accuracy collapse; int4 TC EOL.
- **int8-PV / fp8 attention (SageAttn v2/v3, INT-FlashAttn)** — softmax-prob int8 unacceptable; fp8-PV
  needs Hopper/Ada fp8 TC (Ampere has none).
- **Machete / FA3 / FlashMLA / FlashInfer-MoE / CuteDSL Mamba** — Hopper/Blackwell ISA (wgmma/TMA/tcgen05).
- **prefill chunk-kernel tuning, gating/l2norm/post-conv fusion, SSD static JSON** — already upstream
  (runtime-autotuned / fused / <1.5% of prefill).
- **box-deployment config** — TP/PP, P2P/NVLink, gpu_mem_util, max_num_seqs, cudagraph sizes, attention-
  backend selection, comm/topology. Per-deployment knobs, not fork deliverables. (The no-NVLink 2×3090
  "67% all-reduce prefill" is a box artifact, not a general Ampere property.)

## Hardware gaps for validation
sm_86 (3090) testable in the sandbox now. **sm_80 (A100) is a CI + tuning gap** — no A100 runner; any
"sm_80 tuned" claim must wait for one. New CUDA kernels need smem-split tuning (sm_80 192KB / sm_86 100KB).

## Next investigation candidates (un-run dimensions from the modern-arch scope)
Modern MoE on Ampere · sparse/long-context attention (NSA/Quest for hybrid full-attn layers) ·
quant/KV/MTP breadth (lm_head-int4, hybrid KV #37121 over-alloc, MTP spec-decode). Run as focused
workflows when chosen.
