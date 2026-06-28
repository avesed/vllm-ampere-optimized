# fused_silu_int8 — serve + A/B bench recipe (sandbox 2×3090 tp2)

PREFILL lever: fuse SiluAndMul + per-token int8 quant into the dense W4A8 FFN, feeding the
pre-quantized activation straight into `down_proj`'s marlin GEMM (skips marlin's internal
`per_token_quant_int8`, eliminates the bf16 `[M, intermediate]` HBM round-trip). Decode (M=1) is
bit-exact and ~0 gain — **prefill tok/s is the only metric**.

Sandbox: `ssh coder@192.168.100.1 -p 60022 -i ~/.ssh/ssh-key` (chmod 600 the key).
Ckpt: `/home/coder/models/Qwen3.6-27B-W4A16` (W4A8 = this W4A16 dir + `VLLM_MARLIN_INPUT_DTYPE=int8`).
venv: `/home/coder/vllm-build-venv/bin`.
Host MUST use `--shm-size=8g --ipc=host` — the 64MB default `/dev/shm` corrupts tp2 output and
**looks identical to prequant garbage**. Always rule shm out first.

---

## 0. Pre-serve gate (MUST pass before any serve)

```bash
# on the GPU box (sm86)
cd /home/trevor/vllm-ampere-optimized
VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_FAMP_FUSED_SILU=1 \
  python -m flashampere.fused_silu_int8.test_mlp_equiv
```
Expect every row `cos > 0.9999`, `max_rel` ~1e-3 or less, and `ALL EQUIVALENCE CHECKS PASSED`.
Note on what each assertion catches: a missing/forced `input_global_scale` (the #1 failure, ~4096×
off) is caught by the MAGNITUDE check (`assert_close` rtol/atol), NOT by `cos` — cosine is
scale-invariant. `cos < 0.9999` instead flags a wrong arg slot / direction error (wrong
`w_s`/`w_zp`/`g_idx`). The gate needs BOTH; if either fires → DO NOT serve.

This run also gates the patched forward directly: `run_patched_forward` asserts the dense path
(expert_gate=None) fuses and matches stock, and that a SHARED-EXPERT instance (expert_gate set)
bails to stock EXACTLY (atol=0) — the kernel-level compare alone cannot catch the dropped
expert_gate. `run_kernel_vs_real_silu` confirms the fused int8 codes match the REAL serve
`torch.ops._C.silu_and_mul` (not just `F.silu(.float())*up`) to ≤1 LSB.

---

## 1. Common env (BOTH A and B)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_MARLIN_INPUT_DTYPE=int8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/trevor/vllm-ampere-optimized      # REQUIRED: top-level `flashampere` pkg
# If VLLM_PLUGINS is set anywhere, it is an ALLOWLIST — include fused_silu (else plugin is skipped):
# export VLLM_PLUGINS=fused_silu            # (or leave VLLM_PLUGINS unset so all plugins load)
```
`PYTHONPATH` landmine: `build.py` does `from flashampere.marlin.build import ...`; the top-level
`flashampere` package is the PROJECT ROOT, not the vendored backend subtree `sync_to_fork.sh`
deploys. Without the project root on `PYTHONPATH`, the plugin import fails silently (patch skipped,
no error — you'd just measure baseline twice).

## 2. Serve command (identical for A and B)

```bash
vllm serve /home/coder/models/Qwen3.6-27B-W4A16 \
  --tensor-parallel-size 2 --max-model-len 34000 \
  --max-num-seqs 16 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 --trust-remote-code \
  --reasoning-parser deepseek_r1
```
- **A (baseline / stock):** `VLLM_FAMP_FUSED_SILU` UNSET (or `0`).
- **B (treatment / fused):** `export VLLM_FAMP_FUSED_SILU=1`.

Watch the worker log for `fused_silu plugin: patched <Class>.forward` (B only). Its absence in B
means the plugin did not load (PYTHONPATH / VLLM_PLUGINS / build failure) — the run is not valid.
CONFIRM the logged class is the one the served model actually instantiates, or B == baseline:
- Qwen3.6-27B dense (qwen3_5_text / qwen3_next arch) → `Qwen2MoeMLP` (Qwen3NextMLP alias).
- qwen3_moe arch (e.g. some 35B-A3B builds) → `Qwen3MoeMLP`.
- plain Qwen2/Qwen3 dense → `Qwen2MLP`.
On a MoE model the fused path fires ONLY for the per-layer DENSE MLP; SHARED EXPERTS (expert_gate
set) bail to stock by design (they apply a sigmoid gate the fused path does not). So a pure-MoE-only
checkpoint with no dense MLP layers will see ~no fused activity — that is expected, not a failure.

Restart between A and B: `pkill -f 'vllm serve'; sleep 8`. One server per SSH session (sandbox sshd
is flaky on long connections). 27B-W4A8 is memory-tight on 2×24GB — keep `--max-num-seqs 16` +
shm8g + expandable_segments or it OOMs at `determine_available_memory` (GDN mamba/conv state cache).

If serving from docker, equivalent flags: `--gpus all --shm-size=8g --ipc=host` and pass the same
envs with `-e`, plus mount/`-e PYTHONPATH=/path/to/vllm-ampere-optimized` inside the container.

---

## 3. Prefill A/B protocol (the perf metric)

The fused kernel is a prefill lever → drive long prompts with small `max_tokens` so TTFT dominates,
and report **prefill tok/s = prompt_tokens / TTFT_sec** (`api_bench_client.py:67`). Take the
**median of N≥3 reps** per context length to beat TTFT-ramp noise.

```bash
cd /home/trevor/vllm-ampere-optimized
export BASE=http://localhost:8000/v1
export MAX_TOKENS=32          # small: TTFT dominates
export K=0                    # no spec-decode for the prefill A/B (isolates the FFN lever)

# REP 90 / 354 / 708  ->  ~4k / 16k / 32k input tokens. 3 reps each; median the prefill tok/s.
for REP in 90 354 708; do
  for i in 1 2 3; do
    REP=$REP LABEL="ctx${REP}_rep${i}" python scripts/api_bench_client.py
  done
done
```
The script auto-warms (one `max_tokens=16` call) before the measured call — for B this also triggers
the plugin's eager kernel build in the workers, so the FIRST measured rep is not paying JIT.

Report B/A prefill tok/s at matched `plen`. Decode tok/s will NOT move (~0 lever) — not the metric.
Expected win is modest (eliminates one bf16 `[M, intermediate_per_rank]` write+read per FFN per
layer; larger at big M / long ctx). Any prefill **regression** in B → investigate (kernel overhead
or extra copy), not necessarily a NO-GO.

---

## 4. Correctness gates (B MUST match A — any drop = prequant path wrong → NO-GO)

Run EVERY gate on BOTH servers: A (fused OFF) and B (fused ON). Compare B against the A run from
THIS session — not a remembered number. Restart between A and B (`pkill -f 'vllm serve'; sleep 8`).

### 4a. Coherence (catches silent corruption, e.g. a missing input_global_scale = output ~4096× off)
```bash
# run with server A up, then again with server B up
bash scripts/coherence_check.sh        # 3 zh+en prompts, temp0.6/top_p0.95; human-read for garbage
```

### 4b. GSM8K (numeric correctness; pass bar = B matches the A run within noise)
```bash
# run against A, record acc_A; then against B, record acc_B
python benchmarks/eval_oai.py --task gsm8k --data benchmarks/gsm8k.jsonl \
  --base http://localhost:8000/v1 --max-tokens 24576 --conc 32
```
Pass bar: **acc_B ≥ acc_A within noise** (W4A8 int8-act ~96.5%, int4 ~96.8% for reference only — gate
against the live A run, not the constant). Any meaningful drop → reject the fused path (the prequant
math is wrong even if perf looked good). `mmlu_pro.jsonl` available as a second gate if desired.

### 4c. MoE / shared-expert gate (REQUIRED if the served model has shared experts)
The dense-only 27B GSM8K CANNOT detect a shared-expert regression (dense layers set
expert_gate=None and are correct). If you enable B on a MoE checkpoint (e.g. 35B-A3B), ALSO run 4a/4b
there: the patched forward bails to stock on shared experts (expert_gate set), so B must still
match A. If it does not, the bail guard is not firing — stop and investigate.

---

## Quick reference — the 8 make-or-break items (all handled in integrate.py)

1. `a_scales = a_scales * layer.input_global_scale` — mandatory for g32 (the 27B recipe); guarded
   `if igs is not None` for channelwise. Skipping/forcing = silent garbage.
2. 7th positional to `ops.marlin_gemm` (`global_scale`) = ALWAYS `None`; global scale folded into
   `a_scales`.
3. `wtype = uint4b8` forced + asserted (int4 weights repacked at load).
4. `weight_scale` (`w_s`) passed UNCHANGED (int16 codes inside a float-dtype tensor).
5. `use_atomic_add = False` hardcoded (do NOT derive from the int8 input dtype).
6. All-reduce after the kernel apply (down_proj is RowParallel; we bypass its `.forward`). NO
   post-reduce bias add — Qwen down_proj is bias=False and stock fuses bias into the GEMM on rank 0
   only; the patched forward asserts bias is None and bails to stock if a biased down_proj appears.
7. Gate on `kernel.config.act_type == torch.int8` AND `VLLM_MARLIN_INPUT_DTYPE=int8`; else fall back
   to stock `act_fn`+`down_proj`.
8. expert_gate bail — `Qwen2MoeMLP`/`Qwen3MoeMLP` (== shared experts) apply
   `F.sigmoid(expert_gate(x)) * out`; the patched forward bails to stock whenever
   `self.expert_gate is not None` so shared experts are never fused (the fused path drops the gate).
9. Plugin runs in worker subprocesses (`vllm.general_plugins`), idempotent, eager-build in try/except.
