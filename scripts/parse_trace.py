#!/usr/bin/env python3
"""Summarize a torch-profiler chrome trace: top GPU kernels by total time + call count. Reveals
where MTP decode spends time at long context. Usage: parse_trace.py <profiler_dir>."""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.json*")), key=os.path.getmtime)
if not files:
    print("no trace files in", d); sys.exit(1)
f = files[-1]
opener = gzip.open if f.endswith(".gz") else open
with opener(f, "rt") as fh:
    data = json.load(fh)
ev = data["traceEvents"] if isinstance(data, dict) else data

dur = defaultdict(float)
cnt = defaultdict(int)
for e in ev:
    if e.get("cat") in ("kernel", "Kernel", "gpu_op", "gpu_memcpy") and isinstance(e.get("dur"), (int, float)):
        n = e.get("name", "?")
        # collapse template/param noise a bit
        key = n.split("<")[0][:80]
        dur[key] += e["dur"]
        cnt[key] += 1
tot = sum(dur.values())
print(f"trace={os.path.basename(f)}  total_GPU_kernel_time={tot/1e3:.1f}ms  (top 22 by time)")
for n, v in sorted(dur.items(), key=lambda x: -x[1])[:22]:
    print(f"  {v/1e3:8.1f}ms  {100*v/tot:5.1f}%  x{cnt[n]:6d}  {n}")
