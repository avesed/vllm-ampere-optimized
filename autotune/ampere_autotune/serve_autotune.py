"""`vllm serve --autotune` glue — the docker-friendly entry.

Shipped as fork patch 0005: vLLM's serve subcommand gets a few --autotune* flags and, when --autotune
is set, hands off here INSTEAD of launching one long-running server. We reuse the user's OWN serve
args (model / tensor-parallel / port / ...) as the single source of truth: reconstruct the base
`vllm serve ...` command from argv (minus the --autotune* flags), use it as the per-config restart
template, and run the HALF-A autotuner against http://localhost:<port>. Same command shape as a normal
serve (one extra flag) -> no entrypoint/CMD change in docker.

The patch stays THIN: serve.py only calls add_autotune_args() + run(); all logic is here.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import List, Optional

# autotune flags injected into `vllm serve`; value-True = consumes the next token (or =value form).
AUTOTUNE_FLAGS = {
    "--autotune": False,             # master switch -> hand off here
    "--autotune-objective": True,    # throughput | latency
    "--autotune-mtp": False,         # MTP/spec-decode K-sweep
    "--autotune-batch-curve": False,  # no-restart throughput<->latency frontier
    "--autotune-scenario": True,     # code | writing | chat | reasoning | general
    "--autotune-prompt-file": True,
    "--autotune-temperature": True,
    "--autotune-concurrency": True,
    "--autotune-seed": True,
    "--autotune-seqs-ceiling": True,
    "--autotune-mtp-ks": True,
    "--autotune-spec-method": True,
}

_PIDFILE = "/tmp/ampere-autotune-vllm.pid"


def add_autotune_args(parser) -> None:
    """Called by the serve.py patch (after make_arg_parser) to expose the --autotune* flags."""
    g = parser.add_argument_group("autotune (Ampere) — sweep serving config instead of serving")
    g.add_argument("--autotune", action="store_true",
                   help="run the Ampere autotuner using THESE serve args, instead of serving")
    g.add_argument("--autotune-objective", choices=["throughput", "latency"], default="throughput")
    g.add_argument("--autotune-mtp", action="store_true", help="MTP/spec-decode K-sweep")
    g.add_argument("--autotune-batch-curve", action="store_true",
                   help="throughput<->latency frontier on the running server (no restart sweep)")
    g.add_argument("--autotune-scenario", choices=["general", "code", "writing", "chat", "reasoning"],
                   default=None)
    g.add_argument("--autotune-prompt-file", default=None)
    g.add_argument("--autotune-temperature", type=float, default=None)
    g.add_argument("--autotune-concurrency", type=int, default=1)
    g.add_argument("--autotune-seed", type=int, default=32)
    g.add_argument("--autotune-seqs-ceiling", type=int, default=256)
    g.add_argument("--autotune-mtp-ks", default="0,1,2,3")
    g.add_argument("--autotune-spec-method", default="qwen3_5_mtp")


def strip_autotune(tokens: List[str]) -> List[str]:
    """Remove the --autotune* flags (and their values) from the serve flag tokens, leaving the BASE
    serve command's flags. Handles both `--flag value` and `--flag=value`."""
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        head = t.split("=", 1)[0]
        if head in AUTOTUNE_FLAGS:
            if AUTOTUNE_FLAGS[head] and "=" not in t:
                i += 2                       # skip the flag AND its separate value
            else:
                i += 1                       # store_true flag, or --flag=value (one token)
            continue
        out.append(t)
        i += 1
    return out


def base_serve_restart_cmd(serve_flags: List[str]) -> str:
    """A per-config restart template: relaunch `vllm serve <base flags> {flags}` as a child, killing
    only the prior CHILD (by pidfile — never the orchestrator, whose cmdline also has 'vllm serve')."""
    base = " ".join(serve_flags)
    return (f"[ -f {_PIDFILE} ] && kill $(cat {_PIDFILE}) 2>/dev/null; sleep 2; "
            f"nohup vllm serve {base} {{flags}} > /tmp/ampere-autotune-vllm.log 2>&1 & echo $! > {_PIDFILE}")


def run(args, argv: Optional[List[str]] = None) -> int:  # pragma: no cover - drives a server
    """Intercept entry from the patched serve.cmd(). Reconstruct the base serve cmd from argv and
    drive the HALF-A autotuner against the user's own model/port/TP."""
    from .half_a import cotune
    argv = argv if argv is not None else sys.argv
    # argv = [prog, "serve", <serve flags...>]; keep the flags after the "serve" token.
    try:
        serve_flags = strip_autotune(argv[argv.index("serve") + 1:])
    except ValueError:
        serve_flags = strip_autotune(argv[2:])
    port = getattr(args, "port", None) or 8000
    ns = SimpleNamespace(
        endpoint=f"http://localhost:{port}",
        restart_cmd=base_serve_restart_cmd(serve_flags),
        objective=getattr(args, "autotune_objective", "throughput"),
        auto=not (getattr(args, "autotune_mtp", False) or getattr(args, "autotune_batch_curve", False)),
        mtp_sweep=getattr(args, "autotune_mtp", False),
        batch_curve=getattr(args, "autotune_batch_curve", False),
        sweep=None, levels="1,2,4,8,16,32,64,128",
        scenario=getattr(args, "autotune_scenario", None),
        prompt_file=getattr(args, "autotune_prompt_file", None),
        temperature=getattr(args, "autotune_temperature", None),
        concurrency=getattr(args, "autotune_concurrency", 1),
        seed=getattr(args, "autotune_seed", 32),
        seqs_ceiling=getattr(args, "autotune_seqs_ceiling", 256),
        mtp_ks=getattr(args, "autotune_mtp_ks", "0,1,2,3"),
        spec_method=getattr(args, "autotune_spec_method", "qwen3_5_mtp"),
        model=None, ready_timeout=600, json=False,
    )
    print(f"[vllm serve --autotune] driving the autotuner on {ns.endpoint} "
          f"(base serve: vllm serve {' '.join(serve_flags)})")
    return cotune.run(ns)
