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
    p.add_argument("--gpu", default=None,
                   help="restrict to ONE GPU by index (e.g. 1) or UUID substring; "
                        "writes/advisory target only it (never the others)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--yes", action="store_true",
                   help="(--hw) skip the hardware-damage + card-in-use confirmation prompts "
                        "(non-interactive; you accept the risk)")


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

    ct = sub.add_parser("cotune", help="HALF-A: MEASURED co-tuning sweep (restart per config, find best)")
    _add_common(ct)
    ct.add_argument("--endpoint", default="http://localhost:8000", help="vLLM base URL")
    ct.add_argument("--restart-cmd", default=None,
                    help="advanced: shell template that (re)launches the server; MUST contain {flags}")
    ct.add_argument("--model", default=None,
                    help="BUILT-IN launcher: model path/HF-id to serve; the tool builds the restart "
                         "command itself (no --restart-cmd needed)")
    ct.add_argument("--launcher", choices=["docker", "vllm"], default="docker",
                    help="(--model) how to launch: docker (default) or bare `vllm serve`")
    ct.add_argument("--image", default="vllm/vllm-openai:latest",
                    help="(--model --launcher docker) image (pass the Ampere fork image for W4A8)")
    ct.add_argument("--gpus", default="all", help="(--model --launcher docker) docker --gpus spec")
    ct.add_argument("--port", type=int, default=8000, help="(--model) server port (endpoint follows it)")
    ct.add_argument("--tp", type=int, default=None, help="(--model) tensor-parallel-size for the launched server")
    ct.add_argument("--serve-extra", default=None, help="(--model) extra fixed serve flags appended verbatim")
    ct.add_argument("--batch-curve", action="store_true",
                    help="profile the RUNNING server across concurrency (NO restart): aggregate + "
                         "per-session tok/s + TPOT per batch -> pick the operating batch")
    ct.add_argument("--levels", default="1,2,4,8,16,32,64,128", help="(--batch-curve) concurrency levels")
    ct.add_argument("--sweep", default=None,
                    help='MANUAL grid, e.g. "--max-num-seqs=32,64,96;--kv-cache-dtype=auto,fp8"')
    ct.add_argument("--auto", action="store_true",
                    help="ADAPTIVE search instead of a manual grid (picks knobs/values itself)")
    ct.add_argument("--seed", type=int, default=32, help="(--auto) starting max-num-seqs")
    ct.add_argument("--seqs-ceiling", type=int, default=256, help="(--auto) max max-num-seqs to probe")
    ct.add_argument("--objective", default="throughput", choices=["throughput", "latency"],
                    help="throughput=aggregate @ high concurrency; latency=single/few-session max throughput")
    ct.add_argument("--concurrency", type=int, default=1,
                    help="(--objective latency / --mtp-sweep) sessions to optimize for (1=single, e.g. 4=few)")
    ct.add_argument("--mtp-sweep", action="store_true",
                    help="sweep MTP/spec-decode K (num_speculative_tokens); needs an mtp-head checkpoint")
    ct.add_argument("--mtp-ks", default="0,1,2,3", help="(--mtp-sweep) K values to try; 0 = baseline (no spec)")
    ct.add_argument("--spec-method", default="qwen3_5_mtp", help="(--mtp-sweep) speculative method name")
    ct.add_argument("--scenario", default=None, choices=["general", "code", "writing", "chat", "reasoning"],
                    help="preset benchmark prompt (workload shape — accept-rate / optimal K depend on it)")
    ct.add_argument("--prompt-file", default=None, help="benchmark on YOUR prompt (overrides --scenario)")
    ct.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; OMIT to use the model's default (tests never force temp=0)")
    ct.add_argument("--ready-timeout", type=int, default=600,
                    help="GUARD: max seconds to wait for /health per vLLM bring-up (hard-capped at 600)")
    ct.add_argument("--output", "-o", default=None,
                    help="save the result here (file or dir); default ~/.local/state/ampere-autotune/results/ "
                         "(in docker, mount a volume + point here — the default is ephemeral). Always also printed.")

    tune = sub.add_parser("tune", help="HALF-B: characterize a stable clock profile (host root)")
    _add_common(tune)
    tune.add_argument("--mode", choices=["characterize", "monitor-only", "safe-adapt"],
                      default="monitor-only")
    tune.add_argument("--dry-run", action="store_true",
                      help="MANDATORY before any real --hw write")
    tune.add_argument("--profile", help="per-GPU profile path (default ~/.config/ampere-autotune/<uuid>.json)")
    tune.add_argument("--endpoint", default=None,
                      help="running vLLM base URL; the no-root advisory uses it for the stock "
                           "golden + decode-tok/s (-> real headroom projection)")

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
    matrix = preflight.collect(write_probe=write_probe, gpu_filter=getattr(args, "gpu", None))

    if args.cmd == "preflight":
        _emit(matrix, args.json, preflight.render)
        return 0

    if args.cmd == "cotune":   # HALF-A measured sweep (no NVML/privilege needed; drives the endpoint)
        from .half_a import cotune
        return cotune.run(args)

    if args.tier == "vllm" or args.cmd == "recommend":
        if not matrix.half_a_available:
            print("HALF-A unavailable: cannot read NVML and no reachable /metrics endpoint.", file=sys.stderr)
            return 2
        # TODO(half_a): wire metrics->classify->prescribe. Scaffolded stub for now.
        from .half_a import prescribe
        return prescribe.run(args, matrix)

    # --hw path: gate on preflight.
    if not _hw_unlocked(matrix):
        # No OC-write privilege. `tune` degrades to the no-root ADVISORY (measure + recommend)
        # instead of refusing; monitor/revert still need write privilege.
        advisory_capable = bool(matrix.gpus) and any(g.advisory_capable for g in matrix.gpus)
        if args.cmd == "tune" and advisory_capable:
            from .half_b import advise
            return advise.run_advisory(matrix, endpoint=getattr(args, "endpoint", None))
        print("HALF-B REFUSED by preflight (need host root for OC writes). "
              "Run `ampere-autotune preflight --hw` for details:", file=sys.stderr)
        print(preflight.render(matrix), file=sys.stderr)
        return 3

    if args.cmd in ("tune", "monitor"):
        # Launch-time consent for the real OC-write path (skip on a no-op --dry-run preview).
        real_write = not (args.cmd == "tune" and getattr(args, "dry_run", False))
        if real_write:
            from .half_b import safety, silicon
            if not safety.confirm_oc_damage_warning(force=getattr(args, "yes", False)):
                print("Aborted: hardware-damage warning not accepted.", file=sys.stderr)
                return 5
            for g in (matrix.gpus or []):                 # per-target-card 'in use?' consent
                uuid = silicon._gpu_uuid(g)
                if not uuid:
                    continue
                if not safety.confirm_vram_in_use(silicon.mem_used_mib(uuid),
                                                  silicon.running_proc_count(uuid),
                                                  force=getattr(args, "yes", False)):
                    print(f"Aborted: card {uuid[:20]} is in use, not confirmed.", file=sys.stderr)
                    return 5
        if args.cmd == "tune" and getattr(args, "mode", "") == "characterize" and not args.dry_run:
            print("Refusing: `tune --hw --mode characterize` requires a prior `--dry-run`.", file=sys.stderr)
            return 4
        from .half_b import monitor
        return monitor.run(args, matrix)

    if args.cmd == "revert":
        from .half_b import silicon
        return silicon.revert_all(matrix)

    return 1
