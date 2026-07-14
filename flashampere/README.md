# flashampere — Ampere attention backend (source of truth)

This directory is the **single source of truth** for the `flashampere` vLLM attention backend
(registered into `Backend.CUSTOM`, opt-in behind `VLLM_FLASHAMPERE=1`) and every kernel we own/patch.
All development happens HERE; `./sync_to_fork.sh` deploys a copy into the vLLM fork's backend tree
(`vllm/v1/attention/backends/flashampere`) where vLLM actually loads it.

## Layout
- `backend.py` — `FlashAmpereBackend` (Backend.CUSTOM), `register_flashampere` plugin entrypoint.
- `impl.py` — `FlashAmpereImpl`: per-call phase classify + dispatch; sinks to stock FA.
- `dispatch.py` — `DispatchKey` -> ordered kernel legs (FP16PV/BF16CVT/XQA_VERIFY/SAGE head sets).
- `kernels.py` — `fp16pv_prefill` (FI single_prefill + fp16-PV), `_Hd512DecodeState` (FI BatchPrefill
  q-pad-2) + `_XqaHd512DecodeState` (owned XQA hd512 decode, opt-in `VLLM_FAMP_XQA_HD512=1`),
  the mixed-batch q-pad-2 fix.
- `xqa_verify.py` — MTP spec-verify via the owned XQA kernel.
- `capability.py` — GPU cap detection + leg gating.
- `xqa/` — **vendored XQA kernel** (csrc + `_jit_xqa.py` JIT harness + `_xqa.py` wrapper). Owns our
  Ampere un-gating + hd512 (headElemsQK/headElems split) + mixed-batch changes.
- `vendor/` — **pristine upstream kernel baselines** (NOT deployed; excluded from sync). Untouched stock
  copies for diff/rollback during Ampere kernel R&D. See `vendor/README.md`.

## Fallback / rollback (safety net for Ampere kernel R&D with limited testing)
Building Ampere-exclusive kernels is risky under limited testing; the backend is architected so a buggy
kernel NEVER corrupts output or breaks production — it falls back to the stock kernel:

**RUNTIME (production safety) — the real "原装回退":**
1. `VLLM_FLASHAMPERE=0` (default) → famp is a no-op; 100% stock vLLM.
2. Each Ampere kernel is **per-leg opt-in** (`VLLM_FAMP_XQA_HD512`, `VLLM_FLASHAMPERE_XQA_VERIFY`,
   `VLLM_FLASHAMPERE_PV_FP16`, …) — default off → the stock path runs.
3. Every leg **declines (`raise KernelDecline`) to the stock path** (vLLM FA via `super().forward()`, or
   the flashinfer API) on ANY unsupported/uncertain case — the universal bit-faithful sink (`impl.py`).

**CONTRACT for every new Ampere kernel** (keep the fallback intact):
- (a) gate behind a per-leg env flag, **default off**;
- (b) on any shape/dtype/capture case you haven't validated, **decline** (never best-effort — declining
      yields the bit-faithful stock result, a wrong kernel yields silent garbage);
- (c) the stock path (FA / FI) must stay reachable for that shape (for hd>256 where FA can't, the stock
      fallback is the FI BatchPrefill/prefill path, not FA — keep it wired).

**SOURCE (dev rollback):** `vendor/` holds pristine upstream baselines (`diff -u vendor/xqa-upstream/mha.cu
xqa/csrc/xqa/mha.cu` to see exactly what we changed); git history reverts individual changes.

## Internal vs irreducible-external references
Our code is already self-contained: intra-package imports are **relative** (`from . import ...`,
`from .xqa._jit_xqa import ...`). The following are **irreducible** infra deps (cannot be vendored
without forking large subtrees — even the vendored XQA uses them):
- `vllm.v1.attention.backends.flash_attn` — `FlashAttentionImpl` base class (flashampere extends it) +
  `reshape_and_cache_flash`; the universal bit-faithful sink.
- `flashinfer.jit` (`gen_jit_spec`, `CompilationContext`) + `flashinfer.data` (cutlass/cccl headers) —
  the JIT toolchain that compiles the vendored csrc.
- `vllm.config` / `vllm.logger` / `vllm.platforms` / `torch`.

## Kernels we own (vendored csrc) vs call via API
- OWNED (vendored): the XQA kernel (`xqa/csrc`) — decode hd512 + MTP verify.
- CALLED via flashinfer API (Phase-2 vendor target): `single_prefill_with_kv_cache`
  (fp16-PV prefill) + `BatchPrefillWithPagedKVCacheWrapper` (hd512 decode default). Vendoring these
  would let us own the fp16-PV (use_fp16_pv_reduction) patch in-tree and drop the patched-flashinfer
  dependency — pending (they are flashinfer's templated FA2 prefill, not a single file like XQA).
- NOT used by flashampere: DeepSeek MLA (`mla.cuh`) — routes through vLLM's MLA backend -> FI fa2,
  verdict NO-GO to own on Ampere.

## Deploy
`./sync_to_fork.sh` (or `FORK_BACKEND=/path ./sync_to_fork.sh`) rsyncs this dir into the fork backend
path. Then build/serve the fork image as usual.
