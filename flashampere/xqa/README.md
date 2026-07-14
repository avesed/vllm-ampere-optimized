# famp's own XQA — Ampere MTP spec-decode verify kernel

An **independent, editable** copy of FlashInfer's XQA attention, vendored into flashampere so we
own the spec-decode verify kernel and can deepen it ourselves. Used for the MTP **verify** (uniform
q=1+K): XQA is decode-shaped, KV-split, warp-specialized, with ~flat q-scaling, so q=1+K is much
cheaper than a prefill-style verify.

## Layout
- `csrc/` — vendored XQA source, **pruned to the Ampere include-closure only** (16 files, 340K).
  The sm90 (Hopper GMMA/TMA: `mha_sm90.cu`, `gmma*.cuh`, `tma.h`, `tensorMap.*`) and sm120 (MLA:
  `mla_sm120.*`) files are removed — nothing in the Ampere closure includes them (the
  `#if __CUDA_ARCH__>=900` guards are inline-asm, not `#include`s). `mha.cu` is the kernel we edit.
- `_jit_xqa.py` — builder: a copy of FI `jit/xqa.py` with the csrc path repointed here and Ampere
  (sm80/86) **un-gated**. Reuses stock FI **only as the JIT toolchain** (`gen_jit_spec`, etc.).
- `_xqa.py` — wrapper: a copy of FI `xqa.py`, un-gated, GQA-only (MLA path dropped).
- The cudagraph-safe runtime glue lives one level up in `../xqa_verify.py`.

## Status (2026-06-25)
- **Correct**: `famp.xqa` q=1+K **cos = 1.00000** vs FA2 (verified on a stock-*gated* FI → confirms
  independence). cudagraph capture+replay cos=1.0.
- **Fast (op level)**: q=1+K is **1.8–4.3× faster** than FA2 `fwd_kvcache` at 32–128k.
- **e2e on the hybrid W4A8 model**: **parity**. The 4× is diluted — only 8/32 layers are full-attn
  (the rest are GDN + MLP), and patch-A (`fwd_kvcache` KV-split) already removed the verify cliff,
  so the verify is no longer the bottleneck. Same dilution as fp16-PV. → ships **opt-in, default-off**
  (`VLLM_FLASHAMPERE_XQA_VERIFY=0`). Expected to **win on dense full-attn models** and at very long ctx.

## Maintenance (freeze + test-on-bump, NOT permanent rebase)
The kernel is frozen. On a FlashInfer bump: diff upstream `csrc/xqa/mha.cu`; if there's an Ampere-
relevant improvement, port it in; otherwise leave ours untouched. The "un-gate" is two conceptual
edits (allow major 8 in the builder arch list + the wrapper's SM check) — re-apply only if re-vendoring.

## Why we own it — the optimization platform
Owning `mha.cu` lets us deepen the spec-decode kernel directly. Candidate levers, by where they pay:
- **Ampere occupancy tuning** (most promising for the *bandwidth-bound* verify): XQA is H100-tuned
  (`nbCtaPerSM=1` on Ampere vs 2 on sm90, smem sized for H100). Raising occupancy/KV-split for sm86
  targets the actual bottleneck (KV bandwidth + SM utilization), not compute.
- **fp16-QKPV** (half-accumulate QK/PV): a *compute* optim. The verify is bandwidth-bound at long
  ctx, so the win is small there; it helps when compute dominates (short ctx, large K, dense models).
  Worth measuring now that we can edit the mma accumulation in `mha.cu`.
- **Future** points as discovered — the kernel is ours to change.
