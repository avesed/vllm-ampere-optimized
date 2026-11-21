**vLLM 0.31.0 → 0.34.0 (vendored), Model Runner V2, FlashInfer 0.7.3.**

vLLM 0.34 removed Model Runner V1 and Marlin W4A8. Upstream now covers most of what the fork used to carry, so this
release moves to the upstream paths and drops the fork's own copies.

- **Breaking — W4A8 moves to Humming.** `--marlin-input-dtype` and `VLLM_MARLIN_INPUT_DTYPE` are gone (vLLM refuses to
  start with the variable set). Serve a W4A16 compressed-tensors checkpoint as W4A8 with `--quantization humming` and
  `VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8","group_size":0}'`. Humming is faster than the fork's retired
  famp_marlin kernel: 27B 8k prefill 3804 vs 3500 tok/s, decode 65.2 vs 62.6; 35B-A3B decode 173 vs 160.
- **Breaking — V2 only.** `VLLM_USE_V2_MODEL_RUNNER` must be unset or `1`. The image no longer pins V1. The fork's
  V1 speculative-decoding code (DSpark, mixed sliding/full DFlash, the exactly-(K+1)-token guard, the vllm#51571
  race fix) is gone: upstream V2 serves the published DSpark and DFlash heads and does not have either bug.
- **Breaking — flashampere retired.** On vLLM 0.34 it gained 1.5% prefill at 32k context and nothing below
  (decode unchanged; gemma-4 never selected it). `VLLM_FLASHAMPERE*` now only draws an unknown-variable warning.
  The vendored FlashInfer tree went with it: the image uses the FlashInfer 0.7.3 wheel vLLM pins.
- **Kept — FA2 split-KV spec-decode verify,** ported to V2's FULL CUDA graphs: MTP K=2 decode at 16k context
  95.4 tok/s vs 61.4 without it (stock vLLM 0.34: 61.3). The route now only takes batches up to the spec-decode
  verify width; before, a lone prefill chunk of uniform shape could also take it. The capture-safe paged-KV
  bounds check is rebased onto the new flash-attention pin.
- **New fixes carried:** Humming's fallback crash (vllm#68194; 35B-A3B W4A8 failed to load) and sampling-parameter
  size limits (vllm PR #55164; an oversized `logit_bias`, `allowed_token_ids` or stop-token list raised inside the
  worker and killed the engine — it is now a 400).
- **Packaging fix:** vLLM's own `.gitignore` matches a few files upstream force-adds (`*_hip*` covers the real
  module `kernels/linear/mxfp4/rdna3_hip.py`). Vendoring had dropped them; `revendor.sh` now stages them and the
  release matrix checks every vendored module is in the image.
- **Measured recipes** (2× RTX 3090 NVLink, 27B W4A16 + MTP K=2, short / 16k decode tok/s): baseline 121.8 / 94.9;
  draft-only NVFP4 `lm_head` (`"draft_lm_head_quantization":"nvfp4"`) 137.4 / 103.4; plus custom all-reduce on the
  NVLink pair 149.7 / 110.8; plus Humming W4A8 149.3 / 118.1 with 8k prefill 3612 (vs 2087). See the README.
- **Release matrix:** T0 adds a package-completeness check, the sampling-limit check and a Humming 35B load check;
  T6m adds GSM8K on the MTP serve and checks the FA2 verify route is captured into the graphs; famp blocks
  (T1, T2, T5e) are removed.
- **Validated:** T0, T6m, T10, T6, T4, T9 (12-minute soaks: DFlash 1408 and MTP 1632 requests, 0 errors, 0
  abnormal starts). Paths that are now pure upstream (quant GSM8K rows, multimodal, hybrid 9B, greedy draft A/B)
  were not re-run for this release.

Image: `ghcr.io/avesed/vllm-ampere-optimized:0.5.0` · `:latest` · `:v0.34.0-ampere-cu130` (from-source, full
multi-arch).
