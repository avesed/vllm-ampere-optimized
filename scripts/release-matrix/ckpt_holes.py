#!/usr/bin/env python3
"""Find 4 MiB-aligned all-zero extents in safetensors files and name the tensors they fall in.

Lost writes left 16-36 MiB zeroed holes in three staged checkpoints (found 2026-10-07); file sizes
were intact, nothing failed to load, and the only symptom was quality. Trained float weights are
essentially never 4 MiB of exact zeros, so any hit is reported. Stdlib only (runs on the bare host).

    ckpt_holes.py <models_dir_or_checkpoint_dir> [...]
"""
import json
import pathlib
import struct
import sys

BLK = 4 << 20
ZERO = bytes(BLK)

for root in map(pathlib.Path, sys.argv[1:]):
    for f in sorted(root.rglob("*.safetensors")):
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
            base = 8 + hlen
            spans = sorted((base + v["data_offsets"][0], base + v["data_offsets"][1], k)
                           for k, v in header.items() if k != "__metadata__")
            size = f.stat().st_size
            zero_blocks = []
            for off in range(0, size - BLK + 1, BLK):
                fh.seek(off)
                if fh.read(BLK) == ZERO:
                    zero_blocks.append(off)
        hits: dict[str, int] = {}
        for z in zero_blocks:
            for s, e, k in spans:
                if s < z + BLK and z < e:
                    hits[k] = hits.get(k, 0) + 1
        status = "CLEAN" if not zero_blocks else f"{len(zero_blocks)} zero 4MiB blocks"
        print(f"{f.relative_to(root)}: {status}", flush=True)
        for k, n in sorted(hits.items(), key=lambda x: -x[1])[:6]:
            print(f"    {n:4d} blocks in {k} {header[k]['dtype']} {header[k]['shape']}")
