# FA2 `fwd_kvcache` for the MTP spec-decode verify (FlashDecoding split-KV)

Routes the **MTP / speculative-decode verify attention** (the uniform `q = 1 + num_speculative`
query tokens over a long paged KV) through the compiled-but-unwrapped FA2 op
`torch.ops._vllm_fa2_C.fwd_kvcache` instead of `flash_attn_varlen_func`.

## Problem

On FA2 (Ampere sm_80/sm_86), `flash_attn_varlen_func` with `q > 1` runs the `flash_fwd_splitkv`
kernel **without** KV-splitting — its only parallelism is `batch * num_kv_heads` tiles. At
`num_kv_heads = 4` / `head_dim = 256` (Qwen3.5/3.6 full-attn) that is ~5% SM occupancy, and the
per-call cost grows with KV length. Single-token decode (`q = 1`) hits the packed-GQA FlashDecoding
path and stays fast, but the verify (`q = 1 + K`) does not — so **MTP decode goes net-negative
beyond ~10k context** (e.g. 9B 32k: +32% @4k → −36% @32k vs no-MTP).

Root cause confirmed with torch.profiler (GDN/Mamba kernels are byte-identical with/without MTP; the
delta is entirely `flash_fwd_splitkv`) and reproduced on stock `vllm/vllm-openai:v0.23.0` and on a
single GPU (not a fork / TP / NVLink issue). Known upstream but unfixed for FA on Ampere
(vllm-project/vllm #14486, #6052; #40750 is MLA-only; FlashInfer's spec-as-decode is SM100-gated).

## Fix

`fwd_kvcache` is the FlashDecoding "append" entry — same `flash_fwd_splitkv` kernel **plus** the
`flash_fwd_splitkv_combine` reduction, i.e. it actually splits over KV. The verify tokens are already
in the paged cache (written by `reshape_and_cache` before attention), so we call it with `k = v =
None`, `seqlens_k = full per-request length`, `causal = True`, `num_splits = 0`. It reproduces
`flash_attn_varlen_func` (q=1 cos=1.000000, q=3 cos=0.999996 — bf16 split-combine rounding) at
**~4.3× the kernel speed** (32k: 2.84ms → 0.66ms), and is **cudagraph-capturable at num_splits=0**
(PyTorch's graph private pool captures the combine workspace) — so it captures into the FULL
cudagraph and the win lands in the real serve.

## Measured (OpenAI API, cudagraph, decode tok/s)

| model | ctx | MTP off-patch | MTP on-patch | gain |
|-------|-----|---------------|--------------|------|
| 9B  (tp1) | 16k | 68.2 | 104.7 | +54% |
| 9B  (tp1) | 32k | 50.4 | 88.6  | +76% |
| 27B (tp2) | 16k | 55.2 | 88.2  | +60% |
| 27B (tp2) | 32k | 39.7 | 81.9  | +106% |

MTP is now **net-positive at every context and beats no-MTP** (27B 32k: 81.9 vs no-MTP 61, +34%).
`accept-len` is preserved (9B 2.1–2.3, 27B 2.6–3.0 — the correctness canary), `VLLM_FA2_KVCACHE_VERIFY=0`
is byte-identical to baseline, and prefill / non-spec decode are untouched.

## Files changed (see `fa2-kvcache-verify.patch`)

- `vllm/vllm_flash_attn/flash_attn_interface.py` — `flash_attn_kvcache_verify` wrapper (+ import-time
  schema-arity==20 assert).
- `vllm/vllm_flash_attn/__init__.py`, `vllm/v1/attention/backends/fa_utils.py` — re-export.
- `vllm/v1/attention/backends/flash_attn.py` — guarded import + the dispatch in
  `FlashAttentionImpl.forward` (non-cascade branch).
- `vllm/envs.py` — `VLLM_FA2_KVCACHE_VERIFY` (default on).

The dispatch predicate is all-AND and falls back to the unchanged `flash_attn_varlen_func` for every
unsupported case (fp8/quantized KV, sliding window, alibi, softcap, sinks, FA3, batch-invariant,
non-uniform / prefill-mixed batches), so blast radius on un-targeted models is zero.

## Tests

Standalone kernel tests (need a CUDA box with the FA2 extension; no model download):

- `correctness_probe.py` — `fwd_kvcache` vs `flash_attn_varlen_func`, cos > 0.9999 for q∈{1,3} @ {4k,16k,32k}.
- `wrapper_test.py` — the shipped `flash_attn_kvcache_verify` wrapper (in-place out) vs varlen.
- `capturability_test.py` — capture `fwd_kvcache` in a CUDA graph, replay, assert correct.
- `kvcache_bench.py` / `kernel_name_probe.py` — the ~4.3× speedup and the `splitkv + combine` evidence.

E2E (serve + OpenAI API): set `VLLM_FA2_KVCACHE_VERIFY=1` vs `0`, MTP on, compare decode at 16k/32k;
`accept-len` from the `SpecDecoding metrics` log line must stay ~2.1–3.0.
