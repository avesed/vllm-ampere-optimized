**Patch release on v0.4 (same vendored vLLM 0.25.1). Primarily a spec-decode correctness fix.**

- **Fix (spec-decode correctness):** GDN-hybrid models (Qwen3.6 / Qwen3-Next) + speculative decoding produced garbage when a prompt tokenized to **exactly `num_speculative_tokens + 1` tokens**. An upstream vLLM `_is_uniform_decode` shape-only check mis-dispatched that prefill through the FULL spec-verify CUDA graph, skipping the recurrent-state write. Affects MTP / DSpark / DFlash / ngram, any quantization; short raw-completion prompts (chat-template traffic is effectively immune). Also filed upstream (vllm#49918, PR #47123).
- **Fix (flashampere):** the 0.25.1 `sliding_window=(-1,-1)` sentinel was treated as a real window, which had **disabled all hd≤256 flashampere routing in the v0.4 image** (including the 27B hd128 flagship). Restored.
- **Fix (flashampere xqa_verify):** the MTP spec-verify leg is now prewarmed before CUDA-graph capture (it was silently declining every verify to stock FA); cleanly self-disables on mamba-aligned hybrid page geometry instead of erroring; padded-batch reshape crash fixed.
- **DSpark:** accepts the upstream / DeepSeek-official head format (`Qwen3DSparkModel`) — one checkpoint now serves both this fork and upstream vLLM ≥ 0.26.
- **flashampere prefill:** opt-in batched paged-KV prefill (`VLLM_FAMP_BATCH_PREFILL=1`, default off) — +2.3% / +4.5% clean full-prefill @ 4k / 8k; `CTA_TILE_Q=64` on sm86 for hd128 (+6–18% short/mid prefill); sync-free leg dispatch.

Image: `ghcr.io/avesed/vllm-ampere-optimized:0.4.1` · `:latest` · `:v0.25.1-ampere-cu130` (from-source, full multi-arch). No-NVLink multi-GPU: add `--disable-custom-all-reduce`.
