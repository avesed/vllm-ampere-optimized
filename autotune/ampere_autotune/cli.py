"""ampere-autotune CLI.

Selector: --vllm (DEFAULT, no privilege) vs --hw (host root, opt-in).
Preflight runs first on every invocation; any --hw action the matrix did not unlock is
HARD-REFUSED with the exact missing capability + fix.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, preflight


def _add_common(p: argparse.ArgumentParser) -> None:
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--vllm", dest="tier", action="store_const", const="vllm",
                     help="HALF-A: recommend vLLM flags (default, no privilege)")
    sel.add_argument("--hw", dest="tier", action="store_const", const="hw",
                     help="HALF-B: GPU silicon tuning (host root, opt-in)")
    p.set_defaults(tier="vllm")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ampere-autotune", description=__doc__)
    p.add_argument("--version", action="version", version=f"ampere-autotune {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("preflight", help="print the capability/permission matrix (changes nothing)")
    _add_common(pre)
    pre.add_argument("--write-probe", action="store_true",
                     help="(--hw) prove OC write-permission via a no-op offset set (still moves no clock)")

    rec = sub.add_parser("recommend", help="HALF-A: recommend vLLM flags for a running server")
    _add_common(rec)
    rec.add_argument("--endpoint", default="http://localhost:8000", help="vLLM /metrics base URL")

    tune = sub.add_parser("tune", help="HALF-B: characterize a stable clock profile (host root)")
    _add_common(tune)
    tune.add_argument("--mode", choices=["characterize", "monitor-only", "safe-adapt"],
                      default="monitor-only")
    tune.add_argument("--dry-run", action="store_true",
                      help="MANDATORY before any real --hw write")
    tune.add_argument("--profile", help="per-GPU profile path (default ~/.config/ampere-autotune/<uuid>.json)")

    mon = sub.add_parser("monitor", help="HALF-B: monitor-only watchdog (revert/derate only)")
    _add_common(mon)

    rev = sub.add_parser("revert", help="HALF-B: reset clocks/offsets to stock")
    _add_common(rev)
    return p


def _emit(obj, as_json: bool, render_fn=None) -> None:
    if as_json:
        print(json.dumps(obj.to_dict() if hasattr(obj, "to_dict") else obj, indent=2))
    elif render_fn is not None:
        print(render_fn(obj))
    else:
        print(obj)


def _hw_unlocked(matrix) -> bool:
    return bool(matrix.gpus) and any(g.half_b_unlocked for g in matrix.gpus)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Preflight ALWAYS runs first.
    write_probe = getattr(args, "write_probe", False)
    matrix = preflight.collect(write_probe=write_probe)

    if args.cmd == "preflight":
        _emit(matrix, args.json, preflight.render)
        return 0

    if args.tier == "vllm" or args.cmd == "recommend":
        if not matrix.half_a_available:
            print("HALF-A unavailable: cannot read NVML and no reachable /metrics endpoint.", file=sys.stderr)
            return 2
        # TODO(half_a): wire metrics->classify->prescribe. Scaffolded stub for now.
        from .half_a import prescribe
        return prescribe.run(args, matrix)

    # --hw path: gate on preflight.
    if not _hw_unlocked(matrix):
        print("HALF-B REFUSED by preflight. Run `ampere-autotune preflight --hw` for details:", file=sys.stderr)
        print(preflight.render(matrix), file=sys.stderr)
        return 3

    if args.cmd in ("tune", "monitor"):
        if args.cmd == "tune" and getattr(args, "mode", "") == "characterize" and not args.dry_run:
            print("Refusing: `tune --hw --mode characterize` requires a prior `--dry-run`.", file=sys.stderr)
            return 4
        from .half_b import monitor
        return monitor.run(args, matrix)

    if args.cmd == "revert":
        from .half_b import silicon
        return silicon.revert_all(matrix)

    return 1
