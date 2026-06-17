"""Parse `nsys stats` CSV (cuda_gpu_kern_sum + cuda_gpu_trace) for a phase run:
  - kernel-time breakdown: gemm_marlin / attn_full / attn_linear / norm / act_quant / comm / other
    (attn_full% is the SageAttention ceiling input)
  - GPU BUBBLE fraction: 1 - (union of kernel intervals) / span  (eager vs cudagraph)

Usage:
    python analyze_nsys.py --kern kern_sum.csv --trace gpu_trace.csv --label "decode/eager"
"""
import argparse
import csv
import io
import re

BUCKETS = [
    ("gemm_marlin", r"marlin"),
    ("attn_full",   r"flash|fmha|fwd_kernel|paged|attention|\bmha\b|sdpa|flashinfer|single_query|merge_attn"),
    ("attn_linear", r"gated_delta|causal_conv|chunk_(scan|fwd|o|bwd)|recurrent|mamba|ssm|delta_rule"),
    ("norm",        r"rms_?norm|layer_?norm|\bnorm\b"),
    ("act_quant",   r"silu|gelu|swiglu|activation|\bmul\b|elementwise|quant|scaled|cvt|convert|cast|dequant"),
    ("comm",        r"all_?reduce|all_?gather|reduce_scatter|nccl|\bar_\b|broadcast|sendrecv"),
    ("rope",        r"rotary|\brope\b|rope_"),
    ("embed",       r"embed|gather_rows|index_select|topk|sample|softmax|argmax"),
]


def bucket(name):
    low = name.lower()
    for b, pat in BUCKETS:
        if re.search(pat, low):
            return b
    return "other"


def read_csv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    # nsys stats may prepend blank/SQL lines before the CSV header
    lines = txt.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if "," in ln and ("Name" in ln or "Total Time" in ln or "Start" in ln or "Duration" in ln):
            start = i
            break
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def col(row, *subs):
    for k in row:
        if k and any(s.lower() in k.lower() for s in subs):
            return row[k]
    return None


def num(x):
    if x is None:
        return None
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return None


def kern_breakdown(path):
    rows = read_csv(path)
    agg = {}
    total = 0.0
    for r in rows:
        name = col(r, "Name")
        t = num(col(r, "Total Time", "Total"))
        if name is None or t is None:
            continue
        b = bucket(name)
        agg[b] = agg.get(b, 0.0) + t
        total += t
    return agg, total


def bubble(path):
    rows = read_csv(path)
    iv = []
    for r in rows:
        s = num(col(r, "Start"))
        d = num(col(r, "Duration"))
        if s is None or d is None:
            continue
        iv.append((s, s + d))
    if not iv:
        return None, None, None
    iv.sort()
    span = iv[-1][1] - iv[0][0]
    covered = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            covered += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    covered += ce - cs
    busy = covered / span if span else None
    return (1 - busy) if busy is not None else None, busy, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kern")
    ap.add_argument("--trace")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    print(f"\n================  {a.label}  ================")
    if a.kern:
        agg, total = kern_breakdown(a.kern)
        if total:
            print("  GPU kernel-time breakdown (% of summed kernel time):")
            for b, t in sorted(agg.items(), key=lambda kv: -kv[1]):
                print(f"      {b:13s} {100*t/total:5.1f}%")
            attn = 100 * agg.get("attn_full", 0.0) / total
            print(f"    -> attn_full = {attn:.1f}%  (SageAttn v1 prefill ceiling ~= {attn*0.5:.1f}%)")
    if a.trace:
        bub, busy, span = bubble(a.trace)
        if bub is not None:
            print(f"  GPU timeline: BUSY {100*busy:.1f}%  |  BUBBLE {100*bub:.1f}%  (idle gaps between kernels)")
            print(f"               span = {span/1e6:.1f} ms")


if __name__ == "__main__":
    main()
