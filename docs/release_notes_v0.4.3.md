**vLLM 0.29.0 → 0.31.0 (vendored), FlashInfer 0.6.18.post1. The V1 model runner stays the default.**

- **Upgrade:** all fork edits carried onto vLLM 0.31.0 — W4A8 / int8 Marlin, famp-marlin, flashampere, DSpark,
  mixed-SWA DFlash heads on V1, and the exactly-(K+1)-token prefill guard. The image keeps the V1 model runner
  (`VLLM_USE_V2_MODEL_RUNNER=1` opts into V2). FlashInfer stays on the 0.6.18 line; 0.7 is not adopted.
- **Fix (spec-decode correctness, also affects v0.4.2):** on hybrid GDN models (Qwen3.5 / 3.6 / 3.8) with
  speculative decoding, prefix caching and async scheduling — all three on by default — an output could turn to
  garbage from its second token. The previous step's accepted-token counts were read from batch rows that were
  being moved, so a request's first verify after prefill could resume from an unwritten state slot (upstream
  vllm#51571). Measured on 27B + DFlash K=7 at 16 concurrent requests: 0.9% of outputs before, 0 after. Upstream
  reproduces the same race on vLLM 0.28–0.30, so v0.4.2 serves without flashampere are likely affected (flashampere
  happened to mask it); workaround on v0.4.2: `--no-async-scheduling`. The fix costs ~1–2% MTP decode.
- **FA2 paged decode:** vllm-flash-attention is built from
  [avesed/flash-attention](https://github.com/avesed/flash-attention/tree/vllm-ampere-optimized/capture-safe-kvcache).
  The upstream pin's `mha_fwd_kvcache` checked paged-KV lengths with a device-to-host sync on every call, which
  broke CUDA-graph capture of the fork's FA2 spec-verify path and stalled the MTP drafter once per step (−8% MTP
  decode). The check now runs on the device.
- **flashampere / famp-marlin:** ported to 0.31 (upstream removed the CPU `seq_lens` and Marlin act-order).
- **Release matrix:** new FA2-contract probe (T0), famp-marlin bit-exactness (T5e), greedy draft A/B (T6g),
  spec-vs-no-spec canary (T6c), draft-head A/B (T6h), perf medians (T10), and a checkpoint lost-write-hole gate.
  The soak (T9) now gates on abnormal output starts and adds a default-env MTP soak.
- **Perf** (2× RTX 3090, Qwen3.6-27B W4A16, medians): no-spec decode 71.5 tok/s and 8k prefill 2387 tok/s, equal to
  v0.4.2; MTP K=2 decode 124 tok/s short / 102 tok/s at 16k context (v0.4.2: 129 / 103).

Image: `ghcr.io/avesed/vllm-ampere-optimized:0.4.3` · `:latest` · `:v0.31.0-ampere-cu130` (from-source, full
multi-arch). No-NVLink multi-GPU: add `--disable-custom-all-reduce`.
