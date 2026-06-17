"""P0-A diagnostic: is the Ampere W4A8 Marlin PREFILL GEMM int8-tensor-core-bound, or
starving on the int4->int8 dequant tax? This single number gates ALL downstream int8
kernel work (QServe int8-domain dequant port, large-M tile sweep, Stream-K) so we don't
burn another ~18 from-source rebuilds chasing a kernel that won't pay.

It runs Nsight Compute (`ncu`) on a real single-card W4A8 prefill (enforce_eager, so every
Marlin GEMM is a raw kernel launch ncu can attribute — cudagraph would hide them) in TWO
passes:

  Phase A  (timing, single-pass, ALL kernels)  -> group kernels by name, report the
           ATTENTION wall-clock share of prefill. This is the *bonus* output that decides
           whether a SageAttention INT8-QK prefill backend is even worth building
           (its ceiling ~= attn-core-share x 0.5). Linear-attn (GatedDeltaNet) kernels are
           bucketed separately because SageAttn cannot touch them.

  Phase B  (occupancy, multi-pass, Marlin GEMM only) -> the real question. Reports, bucketed
           by launch grid size (= a free occupancy-vs-M sweep, since vLLM's startup dummy run
           and our prompt hit different M):
             - sm__pipe_tensor_op_imma_cycles_active  (INT8 tensor-core busy %)
             - the dominant warp stall reason (long_scoreboard == memory/dequant latency)
             - SM / DRAM speed-of-light, waves-per-SM (wave quantization / tail)
           then prints an AUTO-VERDICT mapped to the int8 roadmap's next action.

Usage (run on a single RTX 3090 with the W4A8 model present):
    python profile_marlin_w4a8_imma.py --model /path/to/Qwen3.6-27B-W4A8
    python profile_marlin_w4a8_imma.py --model <moe-w4a16> --marlin-input-dtype int8   # MoE moe_wna16 path
    python profile_marlin_w4a8_imma.py --model <m> --prompt-len 8192 --launches 64
    python profile_marlin_w4a8_imma.py --model <m> --dry-run        # print ncu cmds only

  Sandbox note: ncu inside a container needs perf-counter access. Launch the GPU sibling
  container with `--cap-add=SYS_ADMIN` (and ideally `--privileged`), or set the driver param
  `NVreg_RestrictProfilingToAdminUsers=0`, else ncu aborts with ERR_NVGPUCTRPERM.

This is a DIAGNOSTIC — it changes nothing, builds nothing, and is safe to run repeatedly.
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---- Hardware constant (RTX 3090, sm_86). Override with --sm-count for A100(108)/A6000(84). ----
DEFAULT_SM_COUNT = 82

# ---- Verdict thresholds (IMMA active %, on the largest-M Marlin bucket). From the int8 roadmap. ----
IMMA_MAXED = 65.0      # >= this  -> tensor cores already saturated; kernel work won't pay
IMMA_STARVED = 45.0    # <  this  -> tensor cores starving; if long_scoreboard dominates, dequant-tax is real
SKEW_RATIO = 1.30      # per-SM cycles max/avg above this -> wave-quantization / tail imbalance

# ---- Kernel-name -> bucket classification for Phase A (substring/regex, case-insensitive). ----
BUCKETS = [
    ("linear_attn", r"gated_delta|causal_conv|chunk_(scan|fwd|o)|recurrent|mamba|ssm|delta_rule"),
    ("attn_core",   r"flash|fmha|fwd_kernel|paged|attention|\bmha\b|sdpa|flashinfer|merge_attn|single_query"),
    ("attn_aux",    r"reshape_and_cache|rotary|\brope\b|rms.*qkv|concat_kv"),
    ("gemm_marlin", r"marlin|gptq|awq"),
    ("gemm_other",  r"gemm|cutlass|cublas|\bsgemm\b|\bhgemm\b|\bigemm\b|wgmma|ampere_|turing_"),
    ("moe",         r"moe|topk|expert|grouped|sort_|scatter|gather"),
    ("norm",        r"rms_?norm|layer_?norm|\bnorm\b"),
    ("act",         r"silu|gelu|swiglu|activation|\bmul\b|elementwise|act_and"),
    ("quant",       r"quant|scaled|dequant|int8|fp8|cvt|convert"),
    ("comm",        r"all_?reduce|all_?gather|reduce_scatter|nccl|\bar_\b|broadcast"),
    ("sample",      r"sample|softmax|argmax|topp|topk_|penal|logit"),
    ("embed",       r"embed|gather_rows|index_select"),
]

# ---- Raw ncu metric-name substrings we extract from Phase B (substring match = version-robust). ----
M_IMMA = "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"
M_HMMA = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
M_SMSOL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_ISSUE = "smsp__issue_active.avg.pct_of_peak_sustained_active"
M_GRID = "launch__grid_size"
M_WAVES = "launch__waves_per_multiprocessor"
M_DUR = "gpu__time_duration.sum"


# =====================================================================================
# WORKLOAD ROLE — invoked *under* ncu; does exactly one prefill then exits.
# =====================================================================================
def run_workload(args):
    if args.marlin_input_dtype:
        os.environ["VLLM_MARLIN_INPUT_DTYPE"] = args.marlin_input_dtype
    from vllm import LLM, SamplingParams

    # tp=1 (clean single-process attribution), enforce_eager (raw kernels, no cudagraph).
    # max_model_len == max_num_batched_tokens == prompt_len+64 so the whole prompt is ONE
    # prefill chunk (GEMM M ~= prompt_len) and the startup memory-profiling forward stays small.
    win = args.prompt_len + 64
    kw = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        max_model_len=win,
        max_num_seqs=1,
        max_num_batched_tokens=win,
        gpu_memory_utilization=args.gpu_mem,
    )
    # Drop the vision tower's dummy multimodal profiling allocation (VL models OOM the
    # startup forward on a 24GB card otherwise). Harmless on text-only models.
    try:
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    except (TypeError, ValueError, AssertionError):
        llm = LLM(**kw)

    # A ~prompt_len-token prompt. Exact M is approximate; Phase B buckets by grid size anyway.
    prompt = ("The quick brown fox jumps over the lazy dog. " * ((args.prompt_len // 9) + 1))
    print(f"[workload] one prefill, ~{args.prompt_len} tok, max_tokens=1", flush=True)
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0))
    print("WORKLOAD_DONE", flush=True)


# =====================================================================================
# PARSING
# =====================================================================================
def _ncu_import_csv(rep_path):
    """Re-open a .ncu-rep and emit the raw per-metric CSV to a string (clean, no target stdout)."""
    out = subprocess.run(
        ["ncu", "-i", rep_path, "--csv", "--page", "raw"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"[FAIL] ncu --import failed:\n{out.stderr[-2000:]}")
    return out.stdout


def _rows(csv_text):
    # ncu raw CSV has a preamble line or two before the header; find the header row.
    lines = csv_text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if '"ID"' in ln or ln.startswith("ID,") or '"Kernel Name"' in ln:
            start = i
            break
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def _num(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if s in ("", "N/A", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _col(row, *names):
    for n in names:
        if n in row:
            return row[n]
    return None


def classify(name):
    low = name.lower()
    for bucket, pat in BUCKETS:
        if re.search(pat, low):
            return bucket
    return "other"


# =====================================================================================
# PROFILE ROLE — orchestrates the two ncu passes, parses, and renders the verdict.
# =====================================================================================
def _self_workload_cmd(args):
    return [sys.executable, os.path.abspath(__file__), "workload",
            "--model", args.model, "--prompt-len", str(args.prompt_len),
            "--gpu-mem", str(args.gpu_mem), "--tp", str(args.tp)] + \
           (["--marlin-input-dtype", args.marlin_input_dtype] if args.marlin_input_dtype else [])


def phase_a_timing(args, tmpdir):
    rep = os.path.join(tmpdir, "phaseA")
    cmd = ["ncu", "--target-processes", "all", "--metrics", M_DUR,
           "-c", str(args.timing_launches), "-f", "-o", rep] + _self_workload_cmd(args)
    if args.dry_run:
        print("PHASE A:", " ".join(cmd)); return None
    print(f"[phaseA] timing up to {args.timing_launches} kernels (single-pass)...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    _check_ncu(r)
    rows = _rows(_ncu_import_csv(rep + ".ncu-rep"))
    agg = {}  # bucket -> [total_ns, count]
    per_name = {}
    for row in rows:
        if M_DUR not in (_col(row, "Metric Name") or ""):
            continue
        name = _col(row, "Kernel Name") or "?"
        v = _num(_col(row, "Metric Value"))
        if v is None:
            continue
        b = classify(name)
        agg.setdefault(b, [0.0, 0]); agg[b][0] += v; agg[b][1] += 1
        per_name.setdefault(name, 0.0); per_name[name] += v
    return agg, per_name


def phase_b_occupancy(args, tmpdir):
    rep = os.path.join(tmpdir, "phaseB")
    cmd = ["ncu", "--target-processes", "all", "-k", "regex:Marlin",
           "-c", str(args.launches), "--set", "full", "-f", "-o", rep] + _self_workload_cmd(args)
    if args.dry_run:
        print("PHASE B:", " ".join(cmd)); return None
    print(f"[phaseB] full-set occupancy on up to {args.launches} Marlin GEMMs "
          f"(multi-pass replay — minutes)...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    _check_ncu(r)
    rows = _rows(_ncu_import_csv(rep + ".ncu-rep"))

    # group raw metrics per launch id, then bucket launches by grid size
    launches = {}  # id -> {metric_substr: value, "grid":..., "stalls":{reason:val}, "name":...}
    for row in rows:
        lid = _col(row, "ID")
        mname = _col(row, "Metric Name") or ""
        val = _num(_col(row, "Metric Value"))
        d = launches.setdefault(lid, {"stalls": {}, "name": _col(row, "Kernel Name") or "?"})
        if val is None:
            continue
        for key in (M_IMMA, M_HMMA, M_SMSOL, M_DRAM, M_ISSUE, M_GRID, M_WAVES):
            if key in mname:
                d[key] = val
        if "stalled" in mname:
            reason = mname.split("stalled_")[-1].split(".")[0].split("_per_")[0]
            d["stalls"][reason] = max(d["stalls"].get(reason, 0.0), val)
    return list(launches.values())


def _check_ncu(r):
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2500:]
        if "ERR_NVGPUCTRPERM" in tail or "permission" in tail.lower():
            raise SystemExit(
                "[FAIL] ncu cannot access GPU perf counters (ERR_NVGPUCTRPERM).\n"
                "  bare metal : run as root, or set NVreg_RestrictProfilingToAdminUsers=0.\n"
                "  container  : launch the GPU container with --cap-add=SYS_ADMIN (or --privileged).")
        raise SystemExit(f"[FAIL] ncu run failed (rc={r.returncode}):\n{tail}")


def render(args, a_result, b_result):
    print("\n" + "=" * 78)
    print("  P0-A RESULT — Ampere W4A8 Marlin prefill int8-TC occupancy")
    print("=" * 78)

    # ---- Phase A: attention share -> SageAttn decision ----
    sage_ceiling = None
    if a_result:
        agg, per_name = a_result
        total = sum(v[0] for v in agg.values()) or 1.0
        print("\n[A] Prefill GPU-time by kernel bucket (a SageAttn-worth-it readout):")
        for b, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"      {b:12s} {100*t/total:5.1f}%   ({c} launches)")
        attn = 100 * agg.get("attn_core", [0, 0])[0] / total
        lin = 100 * agg.get("linear_attn", [0, 0])[0] / total
        sage_ceiling = attn * 0.5
        print(f"    -> full-attn core = {attn:.1f}% of prefill;  linear-attn(GDN, untouchable) = {lin:.1f}%")
        print(f"    -> SageAttention v1 prefill ceiling ~= {sage_ceiling:.1f}%  (attn_core x ~0.5, op-level 2x)")
        top = sorted(per_name.items(), key=lambda kv: -kv[1])[:5]
        print("    top kernels: " + "; ".join(f"{n[:34]}={100*t/total:.1f}%" for n, t in top))

    if not b_result:
        if args.dry_run:
            print("\n(dry-run: no data)")
        return

    # ---- Phase B: bucket Marlin launches by grid size; verdict on the largest-M bucket ----
    by_grid = {}
    for d in b_result:
        g = d.get(M_GRID)
        if g is None or M_IMMA not in d:
            continue
        by_grid.setdefault(g, []).append(d)
    if not by_grid:
        print("\n[B] No Marlin GEMM launches captured with IMMA metric — check the kernel "
              "regex (is the GEMM named 'Marlin...'?) and that the model uses the W4A8 path.")
        return

    def avg(ds, key):
        xs = [d[key] for d in ds if key in d]
        return sum(xs) / len(xs) if xs else None

    print("\n[B] Marlin GEMM int8-TC occupancy, bucketed by launch grid size (= M sweep):")
    print(f"      {'grid(blocks)':>13} {'n':>3} {'IMMA%':>7} {'SM_SOL%':>8} {'DRAM%':>7} {'waves':>6}  dominant_stall")
    ordered = sorted(by_grid.items(), key=lambda kv: -kv[0])  # largest grid (largest M) first
    for g, ds in ordered:
        imma = avg(ds, M_IMMA); smsol = avg(ds, M_SMSOL)
        dram = avg(ds, M_DRAM); waves = avg(ds, M_WAVES)
        stalls = {}
        for d in ds:
            for k, v in d["stalls"].items():
                stalls[k] = stalls.get(k, 0.0) + v
        dom = max(stalls.items(), key=lambda kv: kv[1])[0] if stalls else "?"
        f = lambda x: f"{x:7.1f}" if x is not None else "    n/a"
        print(f"      {int(g):>13} {len(ds):>3} {f(imma)} {f(smsol)} {f(dram)} "
              f"{(f'{waves:6.2f}' if waves is not None else '   n/a')}  {dom}")

    # verdict on the largest-M bucket (most compute-bound = where kernel headroom matters)
    big = ordered[0][1]
    imma = avg(big, M_IMMA); waves = avg(big, M_WAVES)
    stalls = {}
    for d in big:
        for k, v in d["stalls"].items():
            stalls[k] = stalls.get(k, 0.0) + v
    dom = max(stalls.items(), key=lambda kv: kv[1])[0] if stalls else ""
    lsb_dominant = "long_scoreboard" in dom

    print("\n" + "-" * 78)
    if imma is not None:
        print(f"  VERDICT  (largest-M bucket: grid={int(ordered[0][0])}, IMMA={imma:.1f}%)")
    else:
        print("  VERDICT  (no IMMA reading)")
    print("-" * 78)
    if imma is None:
        print("  Inconclusive — IMMA metric missing. Re-run with --set full and verify ncu version.")
    elif imma >= IMMA_MAXED:
        print(f"  ✅ TENSOR CORES SATURATED (IMMA {imma:.0f}% >= {IMMA_MAXED:.0f}%).")
        print("     Marlin large-M already maxes int8 TC. Kernel work (QServe dequant port, tile")
        print("     sweep, Stream-K) will NOT move prefill. STOP int8-GEMM kernel investment.")
        print("     Pursue instead: int8 per-token KV (long-ctx decode), and SageAttn ONLY if")
        print(f"     [A] shows attention is a real prefill share (ceiling ~{sage_ceiling:.0f}%)."
              if sage_ceiling is not None else "     [A] attention share.")
    elif imma < IMMA_STARVED and lsb_dominant:
        print(f"  🔧 DEQUANT/MEMORY-TAX BOUND (IMMA {imma:.0f}% < {IMMA_STARVED:.0f}%, dominant stall = {dom}).")
        print("     Tensor cores starve waiting on the int4->int8 dequant / smem path. The headroom")
        print("     is REAL. Next: (1) cheap g=-1 per-channel requant A/B (reuses patch 0001 line 60);")
        print("     (2) standalone-test QServe omniserve int8-domain dequant on your shapes, then port")
        print("     its dequant idiom into marlin_template.h (improve patch 0002). Gate on Chinese CoT.")
    elif waves is not None and waves < 4 and abs(waves - round(waves)) > 0.15:
        print(f"  🔧 WAVE-QUANTIZATION / TAIL (IMMA {imma:.0f}%, waves/SM={waves:.2f} — fractional, few waves).")
        print("     SMs idle on the tail wave. Next: Stream-K / split-K dispatch for these prefill shapes.")
    else:
        print(f"  ⚖️  PARTIALLY BOUND (IMMA {imma:.0f}%, between {IMMA_STARVED:.0f}-{IMMA_MAXED:.0f}%, no single dominant tax).")
        print("     Moderate headroom. Do the CHEAP experiments first: g=-1 requant A/B + large-M tile")
        print("     sweep (generate_kernels.py large-M kS8 — NEVER the 0.5 decode rows). A multi-week")
        print("     QServe port is marginal here — standalone-test it before committing.")
    print(f"\n  (stalls on largest-M bucket: " +
          ", ".join(f"{k}={v:.2f}" for k, v in sorted(stalls.items(), key=lambda kv: -kv[1])[:4]) + ")")
    print("=" * 78)


def profile(args):
    if not args.dry_run and not shutil.which("ncu"):
        raise SystemExit("[FAIL] `ncu` (Nsight Compute) not on PATH. It ships with the CUDA "
                         "toolkit (…/nsight-compute/…). Add it to PATH or `module load`.")
    if not args.model:
        raise SystemExit("[FAIL] --model is required.")
    with tempfile.TemporaryDirectory(prefix="marlin_imma_") as tmp:
        a = phase_a_timing(args, tmp)
        b = phase_b_occupancy(args, tmp)
        render(args, a, b)


# =====================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--model", help="path/HF-id of the W4A8 (or W4A16+--marlin-input-dtype) model")
        sp.add_argument("--prompt-len", type=int, default=4096, help="approx prefill token count (GEMM M)")
        sp.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (2 = real serving topo, fits 27B on 2x24GB)")
        sp.add_argument("--gpu-mem", type=float, default=0.85, help="gpu_memory_utilization")
        sp.add_argument("--marlin-input-dtype", default=None,
                        help="set VLLM_MARLIN_INPUT_DTYPE (use 'int8' for the MoE moe_wna16 path)")

    pp = sub.add_parser("profile", help="run ncu + parse + verdict (default)")
    common(pp)
    pp.add_argument("--launches", type=int, default=48, help="Phase B: max Marlin GEMMs to profile")
    pp.add_argument("--timing-launches", type=int, default=3000, help="Phase A: max kernels to time")
    pp.add_argument("--sm-count", type=int, default=DEFAULT_SM_COUNT)
    pp.add_argument("--dry-run", action="store_true", help="print ncu commands, run nothing")

    wp = sub.add_parser("workload", help="(internal) one prefill, invoked under ncu")
    common(wp)

    # default subcommand = profile
    argv = sys.argv[1:]
    if not argv or (argv[0] not in ("profile", "workload") and argv[0].startswith("-")):
        argv = ["profile"] + argv
    args = p.parse_args(argv)

    if args.cmd == "workload":
        run_workload(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()
