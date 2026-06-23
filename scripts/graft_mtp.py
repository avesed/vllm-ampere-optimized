#!/usr/bin/env python3
"""Graft the native MTP (multi-token-prediction) head back into a quantized checkpoint.

AWQ/GPTQ quantization drops the `mtp.*` tensors (the HF load path doesn't instantiate the
draft head, so the quantizer never sees it). Without them, enabling MTP spec-decode in vLLM
loads the drafter as ZEROS → 0% acceptance → silent no-op. This re-grafts the bf16 head from a
SAME-LINEAGE unquantized base and makes the checkpoint serve MTP as-is.

It is additive + non-destructive: it writes `mtp.safetensors`, (re)builds
`model.safetensors.index.json` to include the mtp.* tensors, and adds `re:.*mtp.*` to the
checkpoint's `quantization_config.ignore` (else compressed-tensors quantizes `mtp.fc` → 0%
accept; see GOTCHA1). The main quantized weight shards are left byte-identical.

HEAD-LINEAGE MATTERS: the MTP head is aligned to ITS base's final hidden states. Graft a head
from the matching unquantized base (e.g. Qwen3.6-27B head -> Qwen3.6-27B-* quants), not a
cross-lineage one, or acceptance drops. Spec-decode stays lossless either way (the target
verifies every token), so a mismatched head only costs SPEED, never correctness.

Usage:
    python graft_mtp.py --quant <quant_dir> --head-src <unquant_base_or_mtp_file> [--dry-run]

`--head-src` may be a directory (its mtp.* tensors are pulled from a standalone mtp file or
scanned across its shards) or a single .safetensors file. Grafts in place into `--quant`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

MTP_PREFIX = "mtp."


def _safetensors_files(path: str) -> list[str]:
    return sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))


def collect_mtp_tensors(head_src: str) -> dict:
    """Return {tensor_name: tensor} for every mtp.* tensor found in head_src (dir or file)."""
    mtp: dict = {}
    if os.path.isfile(head_src):
        files = [head_src]
    else:
        # Prefer a standalone mtp file; else scan all shards (lazy headers, only mtp.* read).
        named = [f for f in _safetensors_files(head_src) if "mtp" in f.lower()]
        files = [os.path.join(head_src, f) for f in (named or _safetensors_files(head_src))]
    for fp in files:
        with safe_open(fp, framework="pt") as h:
            for k in h.keys():
                if k.startswith(MTP_PREFIX):
                    mtp[k] = h.get_tensor(k)
    return mtp


def existing_weight_map(quant_dir: str) -> tuple[dict, int]:
    """weight_map of the quant ckpt (from its index, or synthesized for a single-file ckpt)."""
    idx_path = os.path.join(quant_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        total = int(idx.get("metadata", {}).get("total_size", 0))
        return dict(idx["weight_map"]), total
    # Single-file ckpt: map every tensor in model.safetensors to it.
    wm: dict = {}
    total = 0
    for fn in _safetensors_files(quant_dir):
        fp = os.path.join(quant_dir, fn)
        with safe_open(fp, framework="pt") as h:
            for k in h.keys():
                wm[k] = fn
                t = h.get_slice(k)
                total += _nbytes(t.get_shape(), t.get_dtype())
    return wm, total


_DT_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
             "F64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I4": 1, "U4": 1}


def _nbytes(shape, dtype) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * _DT_BYTES.get(str(dtype).upper().replace("DTYPE.", ""), 2)


def patch_config_ignore(quant_dir: str, dry: bool) -> bool:
    cfg_path = os.path.join(quant_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if qc is None:
        print("  [config] no quantization_config (unquantized?) — skipping ignore patch")
        return False
    ignore = qc.setdefault("ignore", [])
    if any("mtp" in str(p).lower() for p in ignore):
        print("  [config] mtp already in quantization_config.ignore — ok")
        return False
    ignore.append("re:.*mtp.*")
    print(f"  [config] added 're:.*mtp.*' to quantization_config.ignore ({len(ignore)} entries)")
    if not dry:
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, help="quantized ckpt dir to graft into (in place)")
    ap.add_argument("--head-src", required=True, help="unquant base dir or an mtp .safetensors file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    quant, src, dry = args.quant, args.head_src, args.dry_run
    print(f"== graft_mtp: {quant}  <-  {src} {'(dry-run)' if dry else ''} ==")

    wm, _ = existing_weight_map(quant)
    already = [k for k in wm if k.startswith(MTP_PREFIX)]
    if already:
        print(f"  [skip] {len(already)} mtp.* tensors already present in {quant} — nothing to graft")
        return 0

    mtp = collect_mtp_tensors(src)
    if not mtp:
        print(f"  [ERROR] no mtp.* tensors found in head source {src}", file=sys.stderr)
        return 2
    n_params = sum(t.numel() for t in mtp.values())
    print(f"  [head] {len(mtp)} mtp.* tensors, {n_params/1e6:.1f}M params, dtype "
          f"{ {str(t.dtype) for t in mtp.values()} }")

    mtp_file = "mtp.safetensors"
    if not dry:
        save_file(mtp, os.path.join(quant, mtp_file), metadata={"format": "pt"})
    print(f"  [write] {mtp_file}")

    # Rebuild a complete index that references the main shards + the new mtp file.
    wm2, total = existing_weight_map(quant)  # re-read (single-file path lists real tensors)
    for k, t in mtp.items():
        wm2[k] = mtp_file
        total += t.numel() * t.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(wm2.items()))}
    if not dry:
        with open(os.path.join(quant, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    print(f"  [index] model.safetensors.index.json -> {len(wm2)} tensors "
          f"({len(mtp)} mtp + {len(wm2)-len(mtp)} base), total_size={total/1e9:.2f}GB")

    patch_config_ignore(quant, dry)
    print("  [done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
