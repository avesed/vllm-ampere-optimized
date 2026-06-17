"""Parse a vLLM/kineto Chrome trace (one per rank) for a phase run:
  - GPU kernel-time breakdown: gemm_marlin / attn_full / attn_linear / norm / act_quant / comm / other
    (attn_full% = SageAttention attention share)
  - GPU BUBBLE fraction: 1 - (union of GPU-kernel intervals) / span  (eager vs cudagraph)

Reads cat=="kernel" GPU events (ts/dur in microseconds). Picks the busiest rank's trace.

Usage: python analyze_torch_prof.py --dir /home/coder/prof_decode_eager --label decode/eager
"""
import argparse
import glob
import gzip
import json
import os
import re

# NOTE: comm MUST precede act_quant — NCCL "AllReduce" contains "reduce"; if act_quant's
# pattern (which had a bare `reduce`) is checked first it steals the all-reduce. comm first +
# no bare `reduce` in act_quant fixes the mislabel that hid the 67% TP all-reduce as act_quant.
BUCKETS = [
    ("comm",        r"all_?reduce|all_?gather|reduce_scatter|nccl|\bar_\b|broadcast|sendrecv|reduce_kernel"),
    ("gemm_marlin", r"marlin"),
    ("attn_full",   r"flash|fmha|fwd_kernel|paged|attention|\bmha\b|sdpa|flashinfer|single_query|merge_attn"),
    ("attn_linear", r"gated_delta|causal_conv|chunk_(scan|fwd|o|bwd)|recurrent|mamba|ssm|delta_rule|fused_recurrent|post_conv|sigmoid_gating|wy_fast|solve_tril|cumsum"),
    ("norm",        r"rms_?norm|layer_?norm|\bnorm\b"),
    ("act_quant",   r"silu|gelu|swiglu|activation|\bmul\b|elementwise|quant|scaled|cvt|convert|cast|dequant"),
    ("rope",        r"rotary|\brope\b"),
    ("misc",        r"embed|gather|index_select|topk|sample|softmax|argmax|copy|memcpy|memset"),
]


def bucket(name):
    low = name.lower()
    for b, pat in BUCKETS:
        if re.search(pat, low):
            return b
    return "other"


def load_events(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def kernel_intervals(events):
    """Return [(ts, dur, name)] for GPU kernel events (kineto cat=='kernel')."""
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        cat = str(e.get("cat", "")).lower()
        if cat != "kernel":
            continue
        ts = e.get("ts"); dur = e.get("dur")
        if ts is None or dur is None:
            continue
        out.append((float(ts), float(dur), e.get("name", "?")))
    return out


def analyze(path):
    iv = kernel_intervals(load_events(path))
    if not iv:
        return None
    agg = {}
    total = 0.0
    for ts, dur, name in iv:
        b = bucket(name)
        agg[b] = agg.get(b, 0.0) + dur
        total += dur
    spans = sorted((ts, ts + dur) for ts, dur, _ in iv)
    span = spans[-1][1] - spans[0][0]
    covered = 0.0
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s > ce:
            covered += ce - cs; cs, ce = s, e
        else:
            ce = max(ce, e)
    covered += ce - cs
    busy = covered / span if span else 0.0
    return {"agg": agg, "total": total, "busy": busy, "span_ms": span / 1000.0, "n": len(iv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.json")) + glob.glob(os.path.join(a.dir, "*.json.gz")))
    print(f"\n================  {a.label}  ================")
    if not files:
        print(f"  [!] no trace files in {a.dir}")
        return
    # pick the trace with the most GPU kernels (the busiest rank)
    best = None
    for fp in files:
        try:
            r = analyze(fp)
        except Exception as ex:
            print(f"  [warn] parse failed {os.path.basename(fp)}: {type(ex).__name__}")
            r = None
        if r and (best is None or r["n"] > best[1]["n"]):
            best = (fp, r)
    if not best:
        print("  [!] no GPU kernels parsed from any trace")
        return
    fp, r = best
    total = r["total"] or 1.0
    print(f"  trace: {os.path.basename(fp)}  ({r['n']} GPU kernels, span {r['span_ms']:.1f} ms)")
    print("  GPU kernel-time breakdown (% of summed kernel time):")
    for b, t in sorted(r["agg"].items(), key=lambda kv: -kv[1]):
        print(f"      {b:13s} {100*t/total:5.1f}%")
    attn = 100 * r["agg"].get("attn_full", 0.0) / total
    print(f"    -> attn_full = {attn:.1f}%  (SageAttn v1 prefill ceiling ~= {attn*0.5:.1f}%)")
    # share EXCLUDING comm (the no-NVLink TP all-reduce is a box artifact, not real compute)
    noncomm = total - r["agg"].get("comm", 0.0)
    if noncomm > 0:
        al = 100 * r["agg"].get("attn_linear", 0.0) / noncomm
        gm = 100 * r["agg"].get("gemm_marlin", 0.0) / noncomm
        print(f"    -> of NON-COMM compute: attn_linear = {al:.1f}%  | gemm_marlin = {gm:.1f}%  "
              f"(comm excluded = {100*r['agg'].get('comm',0.0)/total:.1f}% of total)")
    print(f"  GPU timeline: BUSY {100*r['busy']:.1f}%  |  BUBBLE {100*(1-r['busy']):.1f}%  (idle gaps between kernels)")


if __name__ == "__main__":
    main()
