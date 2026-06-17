# benchmarks/ — serving harness + Ampere diagnostic tooling

Two kinds of tool here: the **serving** harness behind [`results.md`](results.md), and a **diagnostic**
harness for deciding *whether a kernel optimization is worth building on Ampere* (the "measure before you
patch" tooling — it repeatedly saved writing no-op patches). Run them on the target GPU (sm_86 sandbox now;
sm_80/A100 is a coverage gap — no runner). Use a single card (`--tp 1`) or PP to avoid the no-NVLink TP
all-reduce artifact polluting shares.

## Serving throughput
- **`vllm_verify.py <model> <tag>`** — single-stream decode, batch-16 decode, prefill tok/s (the results.md table).
- **`vllm_batch_sweep.py <model> <tag>`** — batch 16/64/256 crossover + the cudagraph-capture-size workaround.

## Diagnostic: is a kernel lever worth it?
- **`bench_marlin_gemm_imma.py`** / **`profile_marlin_w4a8_imma.py`** — `ncu` IMMA (int8 tensor-core) occupancy
  of the W4A8 Marlin GEMM, bucketed by prefill M. **Decision rule:** IMMA ≥ 65% → GEMM saturated, kernel work
  (QServe dequant port / tile sweep / Stream-K) won't pay; < 45% + dominant `long_scoreboard` stall → dequant-tax
  bound, worth it; per-SM skew → Stream-K. *Measured:* W4A8 prefill ≈ 68% IMMA, only ~1.7% wall-clock below pure
  int8 → int8-GEMM kernel work is tapped out. (ncu in a container needs `--cap-add=SYS_ADMIN`.)
- **`torch_prof_phase.py`** + **`analyze_torch_prof.py`** + **`prof_decode_batchsweep.py`** — vLLM-builtin torch
  profiler (captures tp>1 worker kernels) → kernel-time breakdown by bucket (gemm_marlin / attn_full / attn_linear
  / comm / …), **comm EXCLUDED** from the denominator (the no-NVLink TP all-reduce is a box artifact, not compute).
  `prof_decode_batchsweep.py` loads the model ONCE and sweeps batch {1,16,32,64}. **Decision rule:** if a bucket's
  share stays < ~5% even at batch 32+, that path's kernel tuning is dead; ~20-30% → worth a per-arch config.
  *Measured:* GDN linear-attn decode 2→21.6% of non-comm as batch 1→64 (W4A8 9B). Use `profiler_config=
  {"profiler":"torch","torch_profiler_dir":...}` (this build deprecated `VLLM_TORCH_PROFILER_DIR`).
- **`bench_gdn_recurrent_decode.py`** — sweep the GatedDeltaNet recurrent-decode kernel launch constants
  (BV × num_warps × num_stages). *Measured:* best vs hardcoded default = 1.00–1.01x → launch tuning DEAD; the kernel
  is fp32→bf16 state-bandwidth-bound (and bf16 state is already the vLLM default), so nothing to tune.
- **`bench_gdn_state_dtype.py`** — fp32 vs bf16 GDN state perf (kernel runs bf16 with no edit). *Measured:* ~2x for
  bf16 state — but bf16 is already the default on Ampere, so it's the status quo, not a lever.
- **`nsys_phase.py`** / **`analyze_nsys.py`** — nsys-timeline variant. NOTE: the nsys bundled inside Nsight Compute
  can't convert its own `.qdstrm` (`Invalid version prefix`) — prefer the torch-profiler tools above.

## Recurring lesson (why these exist)
Decode is **weight-/state-bandwidth-bound** on Ampere: no tile/launch/config change moves it — only *fewer bytes*
(W4A8 weights, int8 KV, bf16 state) do. Prefill compute (int8 GEMM, GDN chunk kernels) is already near-optimal
upstream. Every "promising" kernel lever this harness checked turned out tapped-out or already-default — so always
measure here before writing a kernel patch.
