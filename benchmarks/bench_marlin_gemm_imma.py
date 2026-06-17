"""P0-A diagnostic (standalone-GEMM version): is the Ampere W4A8 Marlin GEMM int8-tensor-
core-bound at prefill M, or starving on the int4->int8 dequant tax? This gates downstream int8
kernel work (QServe int8-domain dequant port, large-M tile sweep, Stream-K).

Why standalone: profiling a FULL vLLM forward under ncu hangs — ncu --target-processes all
intercepts every CUDA call across vLLM's multiprocess (tp) startup + multi-GB weight load and
crawls. So we instead build ONE W4A8-int8 Marlin GEMM in-process (no engine, seconds to start,
reusing vLLM's own marlin_quantize + ops.marlin_gemm — the exact serving kernel) and sweep M
across the model's real Linear shapes. ncu then profiles only the GEMM launches.

Two passes:
  warmup/launch  : the `gemm` subcommand runs ops.marlin_gemm for each (shape x M), invoked under ncu.
  occupancy      : `--set full -k regex:Marlin`, bucket launches by grid size (= M/shape), report
                   sm__pipe_tensor_op_imma_cycles_active (INT8 TC busy %), the dominant warp stall
                   (long_scoreboard == memory/dequant latency), SM/DRAM SOL, waves/SM -> AUTO-VERDICT.

Usage (one RTX 3090 is enough — GEMM tensors are small):
    python bench_marlin_gemm_imma.py                          # 27B-w4a8 shapes (H5120 I17408 tp2) baked in
    python bench_marlin_gemm_imma.py --hidden 5120 --intermediate 17408 --tp 2 --m-list 1,256,2048,4096
    python bench_marlin_gemm_imma.py --dry-run

  Container ncu perms: launch the sibling container with --cap-add=SYS_ADMIN (else ERR_NVGPUCTRPERM).
Read-only diagnostic — builds/changes nothing on the model side.
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_SM_COUNT = 82
IMMA_MAXED = 65.0
IMMA_STARVED = 45.0

M_IMMA = "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"
M_SMSOL = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
M_ISSUE = "smsp__issue_active.avg.pct_of_peak_sustained_active"
M_GRID = "launch__grid_size"
M_WAVES = "launch__waves_per_multiprocessor"
M_DUR = "gpu__time_duration.sum"


def model_shapes(hidden, intermediate, tp, heads=24, kv_heads=4, head_dim=256):
    """Canonical per-GPU Marlin GEMM shapes (K, N) for the dense transformer block at tp."""
    q = heads * head_dim
    kv = kv_heads * head_dim
    return [
        ("qkv_proj", hidden, (q + 2 * kv) // tp),     # col-parallel
        ("o_proj",   q // tp, hidden),                # row-parallel
        ("gate_up",  hidden, (2 * intermediate) // tp),  # col-parallel (largest N)
        ("down_proj", intermediate // tp, hidden),    # row-parallel (largest K)
    ]


# ---------------------------------------------------------------------------
# WORKLOAD ROLE — run under ncu; builds + runs marlin GEMMs, no vLLM engine.
# ---------------------------------------------------------------------------
def run_gemm(args):
    import torch
    from vllm import _custom_ops as ops

    dtype = torch.bfloat16
    ms = [int(x) for x in args.m_list.split(",")]
    shapes = [s for s in model_shapes(args.hidden, args.intermediate, args.tp)
              if (args.shapes == "all" or s[0] in args.shapes.split(","))]
    torch.manual_seed(0)

    if args.kernel == "marlin":  # W4A8: int4 weight (uint4b8) x int8 act — the serving kernel
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.scalar_type import scalar_types
        b_type = scalar_types.uint4b8
        workspace = marlin_make_workspace_new(torch.device("cuda"))
        for label, K, N in shapes:
            b_weight = torch.randn((K, N), dtype=dtype, device="cuda")
            w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(
                b_weight, b_type, 128, False, input_dtype=torch.int8)
            s2 = (s / s.max() * 4096).round().to(torch.int16).view(dtype)
            for M in ms:
                a = torch.randint(-100, 101, (M, K), dtype=torch.int8, device="cuda")
                a_s = (torch.full((M, 1), 0.02, dtype=torch.float32, device="cuda") / 4096 * s.max()).float()
                out = torch.empty((M, N), dtype=dtype, device="cuda")
                print(f"LAUNCH marlin {label} M={M} K={K} N={N}", flush=True)
                for _ in range(args.iters):
                    ops.marlin_gemm(a, out, q_w, None, s2, a_s, None, None, g_idx, sort_idx,
                                    workspace, b_type, M, N, K, is_k_full=True,
                                    use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
                torch.cuda.synchronize()

    elif args.kernel == "cutlass_int8":  # W8A8: pure int8 x int8 CUTLASS — the "no-dequant" IMMA ceiling
        for label, K, N in shapes:
            b = torch.randint(-8, 8, (N, K), dtype=torch.int8, device="cuda").t()  # (K,N) col-major
            sb = (torch.rand((1, N), dtype=torch.float32, device="cuda") * 0.02 + 0.001)
            for M in ms:
                a = torch.randint(-100, 101, (M, K), dtype=torch.int8, device="cuda")
                sa = (torch.rand((M, 1), dtype=torch.float32, device="cuda") * 0.02 + 0.001)
                print(f"LAUNCH cutlass_int8 {label} M={M} K={K} N={N}", flush=True)
                for _ in range(args.iters):
                    ops.cutlass_scaled_mm(a, b, sa, sb, dtype)
                torch.cuda.synchronize()
    else:
        raise SystemExit(f"unknown --kernel {args.kernel}")
    print("GEMM_DONE", flush=True)


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------
def _ncu_import_csv(rep_path):
    out = subprocess.run(["ncu", "-i", rep_path, "--csv", "--page", "raw"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"[FAIL] ncu --import failed:\n{out.stderr[-2000:]}")
    return out.stdout


def _rows(csv_text):
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


def _check_ncu(r):
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2500:]
        if "ERR_NVGPUCTRPERM" in tail or "permission" in tail.lower():
            raise SystemExit(
                "[FAIL] ncu cannot access GPU perf counters (ERR_NVGPUCTRPERM).\n"
                "  container: launch with --cap-add=SYS_ADMIN (or --privileged).")
        raise SystemExit(f"[FAIL] ncu run failed (rc={r.returncode}):\n{tail}")


# ---------------------------------------------------------------------------
# PROFILE ROLE
# ---------------------------------------------------------------------------
def _self_cmd(args):
    return [sys.executable, os.path.abspath(__file__), "gemm",
            "--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
            "--tp", str(args.tp), "--m-list", args.m_list, "--shapes", args.shapes,
            "--kernel", args.kernel, "--iters", str(args.iters)]


# Explicit metric set (single-pass, fast, deterministic — no --set full replay storm).
# No -k name filter: the build's GEMM symbol may not contain "Marlin"; we isolate the GEMM
# launches in the parser by IMMA>0 (only tensor-core kernels have it; setup kernels = 0).
NCU_METRICS = ",".join([
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "launch__grid_size",
    "launch__waves_per_multiprocessor",
    "gpu__time_duration.sum",
])


def profile(args):
    if not args.dry_run and not shutil.which("ncu"):
        raise SystemExit("[FAIL] `ncu` not on PATH (ships with CUDA toolkit).")
    with tempfile.TemporaryDirectory(prefix="marlin_gemm_imma_") as tmp:
        rep = os.path.join(tmp, "occ")
        cmd = ["ncu", "--target-processes", "all", "-c", str(args.launches),
               "--metrics", NCU_METRICS, "-f", "-o", rep] + _self_cmd(args)
        if args.dry_run:
            print("NCU:", " ".join(cmd)); return
        print(f"[occ] profiling up to {args.launches} kernels (explicit metrics, single-pass)...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        _check_ncu(r)
        rows = _rows(_ncu_import_csv(rep + ".ncu-rep"))
        render(args, rows)


def _find(row, sub):
    """Wide-format raw CSV: return the numeric value of the first column whose name contains `sub`."""
    for k, v in row.items():
        if k and sub in k:
            return _num(v)
    return None


def render(args, rows):
    # `--page raw` CSV is WIDE: one row per kernel, each metric a column. The 1st data row is units.
    launches = []
    for row in rows:
        idv = (row.get("ID") or "").strip()
        if not idv.isdigit():
            continue  # skip the units row
        imma = _find(row, "pipe_tensor_op_imma_cycles_active.avg.pct_of_peak")
        if imma is None or imma <= 0.01:
            continue  # not a tensor-core kernel -> setup/elementwise; keep only the GEMMs
        launches.append({
            "kname": row.get("Kernel Name", "?"),
            "imma": imma,
            "grid": _find(row, "launch__grid_size"),
            "waves": _find(row, "launch__waves_per_multiprocessor"),
            "sm_sol": _find(row, "sm__throughput.avg.pct_of_peak"),
            "dram": _find(row, "dram__throughput.avg.pct_of_peak"),
            "issue": _find(row, "smsp__issue_active.avg.pct_of_peak"),
            "stalls": {
                "long_scoreboard": _find(row, "stalled_long_scoreboard"),
                "short_scoreboard": _find(row, "stalled_short_scoreboard"),
                "mio_throttle": _find(row, "stalled_mio_throttle"),
                "barrier": _find(row, "stalled_barrier"),
            },
            "dur_us": _find(row, "gpu__time_duration"),
        })

    print("\n" + "=" * 78)
    print("  P0-A RESULT — W4A8 Marlin GEMM int8-TC occupancy (standalone, real 27B shapes)")
    print("=" * 78)
    if not launches:
        print("\n[!] No tensor-core (IMMA>0) launches captured — check that ncu collected the IMMA "
              "metric and that ops.marlin_gemm ran the int8 path.")
        return

    by_grid = {}
    for d in launches:
        if d["grid"] is None:
            continue
        by_grid.setdefault(d["grid"], []).append(d)

    def avg(ds, key):
        xs = [x[key] for x in ds if x.get(key) is not None]
        return sum(xs) / len(xs) if xs else None

    def stall_sum(ds):
        st = {}
        for d in ds:
            for k, v in d["stalls"].items():
                if v is not None:
                    st[k] = st.get(k, 0.0) + v
        return st

    kname = launches[0]["kname"]
    print(f"\n  GEMM kernel: {kname[:72]}")
    print(f"\n  {'grid(blk)':>9} {'n':>2} {'IMMA%':>6} {'SM_SOL%':>7} {'DRAM%':>6} {'issue%':>6} {'waves':>6} {'us':>7}  dom_stall")
    ordered = sorted(by_grid.items(), key=lambda kv: -kv[0])
    for g, ds in ordered:
        st = stall_sum(ds)
        dom = max(st.items(), key=lambda kv: kv[1])[0] if st else "?"
        f = lambda x: f"{x:6.1f}" if x is not None else "   n/a"
        w = avg(ds, "waves"); dur = avg(ds, "dur_us")
        print(f"  {int(g):>9} {len(ds):>2} {f(avg(ds,'imma'))} {f(avg(ds,'sm_sol'))} "
              f"{f(avg(ds,'dram'))} {f(avg(ds,'issue'))} "
              f"{(f'{w:6.2f}' if w is not None else '   n/a')} {(f'{dur:7.1f}' if dur is not None else '    n/a')}  {dom}")

    big = ordered[0][1]
    imma = avg(big, "imma"); waves = avg(big, "waves")
    dram = avg(big, "dram"); issue = avg(big, "issue")
    st = stall_sum(big)
    dom = max(st.items(), key=lambda kv: kv[1])[0] if st else ""
    lsb = dom in ("long_scoreboard", "short_scoreboard", "mio_throttle")

    print("\n" + "-" * 78)
    print(f"  VERDICT (largest-grid bucket: grid={int(ordered[0][0])}, IMMA={imma:.1f}%)"
          if imma is not None else "  VERDICT (no IMMA)")
    print("-" * 78)
    if imma is None:
        print("  Inconclusive — IMMA missing.")
    elif imma >= IMMA_MAXED:
        print(f"  ✅ TENSOR CORES SATURATED (IMMA {imma:.0f}% >= {IMMA_MAXED:.0f}%). Marlin large-M maxes int8 TC.")
        print("     Kernel work (QServe dequant port / tile sweep / Stream-K) will NOT move prefill.")
        print("     STOP int8-GEMM kernel investment -> pursue int8 per-token KV (long-ctx decode)")
        print("     and SageAttn (attention is still fp16) instead.")
    elif imma < IMMA_STARVED and lsb:
        print(f"  🔧 DEQUANT/MEMORY-TAX BOUND (IMMA {imma:.0f}% < {IMMA_STARVED:.0f}%, dominant stall = {dom}).")
        print("     int8 TC starve on the int4->int8 dequant / smem path -> headroom is REAL.")
        print("     Next: (1) cheap g=-1 per-channel requant A/B (patch 0001 line 60); (2) standalone-test")
        print("     QServe omniserve int8-domain dequant, then port the idiom into marlin_template.h.")
    elif waves is not None and waves < 4 and abs(waves - round(waves)) > 0.15:
        print(f"  🔧 WAVE-QUANTIZATION / TAIL (IMMA {imma:.0f}%, waves/SM={waves:.2f}). Try Stream-K / split-K.")
    else:
        print(f"  ⚖️  PARTIALLY BOUND (IMMA {imma:.0f}%, between {IMMA_STARVED:.0f}-{IMMA_MAXED:.0f}%).")
        print("     Cheap experiments first: g=-1 requant A/B + large-M tile sweep. QServe port marginal.")
    print(f"\n  (stalls@largest-grid: " +
          ", ".join(f"{k}={v:.2f}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:4]) + ")")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--hidden", type=int, default=5120)
        sp.add_argument("--intermediate", type=int, default=17408)
        sp.add_argument("--tp", type=int, default=2)
        sp.add_argument("--m-list", default="1,256,2048,4096", help="prefill M sweep (decode->prefill)")
        sp.add_argument("--shapes", default="gate_up,down_proj", help="comma list or 'all'")
        sp.add_argument("--kernel", default="marlin", choices=["marlin", "cutlass_int8"],
                        help="marlin=W4A8 int4xint8 (serving); cutlass_int8=W8A8 pure-int8 IMMA ceiling")
        sp.add_argument("--iters", type=int, default=3)

    pp = sub.add_parser("profile"); common(pp)
    pp.add_argument("--launches", type=int, default=64)
    pp.add_argument("--dry-run", action="store_true")
    gp = sub.add_parser("gemm"); common(gp)

    argv = sys.argv[1:]
    if not argv or (argv[0] not in ("profile", "gemm") and argv[0].startswith("-")):
        argv = ["profile"] + argv
    args = p.parse_args(argv)
    if args.cmd == "gemm":
        run_gemm(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()
