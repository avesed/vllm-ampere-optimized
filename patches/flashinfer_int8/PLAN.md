# int8-QK prefill in FlashInfer (Approach I) — Ampere sm_80/86, hd256 hybrids

Goal: int8-QK (IMMA) + fp16-PV prefill attention inside FlashInfer's FA2 Ampere kernel, covering
paged-context + ragged-new-tokens. Decided over Approach II (extend SageAttn) for the FlashInfer
platform bet (perf + unified substrate incl MoE serving). Honest payoff (measured): full-attn is
O(L²) → int8-QK worth ~15% prefill TTFT @128k, ~23% @256k on Qwen3.5-9B-W4A8 (marginal <64k).

## FlashInfer 0.6.12 edit map (headers ship in-package for JIT → editable in place)
- `flashinfer/data/include/flashinfer/mma.cuh` — fp8 clone template `mma_sync_m16n16k32_row_col_f8f8f32` @~L217. ADD `mma_sync_m16n16k32_row_col_s8s8s32` (see mma_s8s8s32_wrapper.cuh). [I-0 done]
- `flashinfer/data/include/flashinfer/attention/prefill.cuh` — `compute_qk()` (QK matmul = `mma_sync_m16n16k16_row_col_f16f16f32`; accum `DTypeQKAccum s_frag[..][8]`; head-dim loop `NUM_MMA_D_QK*16`; the `if constexpr(sizeof(DTypeKV)==1)` branch UPCASTS 8-bit→f16 = opposite of needed). int8 path: s32 accum, k16→k32 tiling, int8 fragment load (no upcast), dequant s32→f32 with q_scale*k_scale + smooth_k before softmax. ONE shared compute_qk covers BOTH paged+ragged. [I-2]
- `flashinfer/jit/attention/modules.py` — `assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"` (gen_batch_prefill_module ~L651, gen_single_prefill ~L342). Add int8 dtype branch; must NOT inherit the fp8 gate. [I-1]
- `flashinfer/data/csrc/batch_prefill_customize_config.jinja` + `batch_prefill_paged_kernel_inst.jinja` + `batch_prefill_ragged_kernel_inst.jinja` — add int8 dtype_map entries + q_scale/k_scale (+smooth_k bias) to Params (via additional_params) + int8 instantiations. [I-1]
- `flashinfer/prefill.py` — wrapper plan()/run() accept int8 q_data_type/kv_data_type + scale tensors. [I-1]
- vLLM glue: custom FlashInfer-backed int8 backend + Q/K int8 quant + scale buffers; register via vllm.general_plugins entry-point (for TP workers). [I-4]

## Phases (each independently testable)
- I-setup: FlashInfer source checkout, editable install in container, confirm JIT recompiles after an edit.
- I-0 [DONE]: int8 IMMA wrapper, compiles sm_86 (RC=0), correct m16n8k32 s8s8s32 PTX. → mma_s8s8s32_wrapper.cuh
- I-1: add the int8 dtype path through JIT/template/python (un-gate fp8 assert, new kernel_inst, Params+scales); prove a new int8 prefill module COMPILES + is callable from python.
- I-2: wire wrapper into compute_qk (k32 tiling, s32 accum, no-upcast int8 fragment load, dequant+smooth_k); validate ragged single_prefill output vs fp16 (rel-err / cos).
- I-3: validate paged path (free via shared compute_qk).
- I-4: vLLM backend + e2e on Qwen3.5-9B-W4A8 @128k (TTFT vs FlashInfer-fp16 + needle/zh-CoT).

## Build/dev facts (sandbox)
- Image ghcr.io/avesed/vllm-ampere-optimized:v0.23.0 has nvcc (CUDA 13) + flashinfer 0.6.12.
- Headers at `/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/`.
- nvcc compile-check: `nvcc -arch=sm_86 -Xptxas -v -c x.cu` (CUDA13: `-Xptxas -v`, NOT `-ptxas-options`).
- Biggest risk = I-2 numerics (fragment/thread layout of m16n8k32 s8 + dequant); de-risk with a standalone numerical mma test (load A/B per PTX m16n8k32 .s8 layout, compare s32 C vs CPU ref) before editing compute_qk.
