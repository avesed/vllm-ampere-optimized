"""HALF-A co-tuning sweep — the MEASURED autotuner (vs recommend's single heuristic pass).

Engine flags are startup-baked, so co-tuning = restart-per-config + measure + rank. Deployment-
agnostic: the caller supplies a `--restart-cmd` shell template with a `{flags}` placeholder
(docker / systemd / k8s / a launch script — the tool never assumes docker). For each config it
restarts, waits for /health, measures (concurrency sweep + under-load scrape), scores, and reports
the empirical best. NO privilege. The pure helpers (grid/parse/score/flags) are unit-tested.
"""
from __future__ import annotations

import itertools
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List

from . import measure

# values meaning "use the server default" -> omit the flag entirely
_DEFAULTY = {"auto", "default", "", "-"}


@dataclass
class SweepPoint:
    config: Dict[str, str]
    feasible: bool = False
    decode_tps_max_c: float = 0.0      # batched aggregate (the throughput objective)
    decode_tps_single: float = 0.0
    kv_peak: float = 0.0
    preempt_per_s: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_sweep(spec: str) -> Dict[str, List[str]]:
    """'--max-num-seqs=32,64,96;--kv-cache-dtype=auto,fp8' -> {flag: [values]}."""
    out: Dict[str, List[str]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"bad sweep segment (need flag=v1,v2): {part!r}")
        k, vs = part.split("=", 1)
        out[k.strip()] = [v.strip() for v in vs.split(",") if v.strip()]
    return out


def expand_grid(spec: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Cartesian product of the per-flag value lists."""
    if not spec:
        return [{}]
    keys = list(spec.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(spec[k] for k in keys))]


def config_flags(cfg: Dict[str, str]) -> str:
    """Render a config to a flag string; default-y values are omitted (server default)."""
    return " ".join(f"{k} {v}" for k, v in cfg.items() if str(v).lower() not in _DEFAULTY)


def score(p: SweepPoint, objective: str = "throughput") -> float:
    """Higher = better. Infeasible -> -inf. Thrashing (preemption) disqualifies a throughput win."""
    if not p.feasible:
        return float("-inf")
    if objective == "throughput":
        return float("-inf") if p.preempt_per_s > 0.05 else p.decode_tps_max_c
    if objective == "latency":          # best single-stream tok/s (interactive)
        return p.decode_tps_single
    return p.decode_tps_max_c


def make_restart_fn(template: str, endpoint: str, ready_timeout: int = 600,
                    settle_s: float = 3.0) -> Callable[[Dict[str, str]], bool]:  # pragma: no cover - drives a server
    """Build a restart hook from a shell template: substitute {flags}, run it, wait for /health."""
    if "{flags}" not in template:
        raise ValueError("--restart-cmd must contain the {flags} placeholder")

    def restart(cfg: Dict[str, str]) -> bool:
        cmd = template.replace("{flags}", config_flags(cfg))
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return _wait_ready(endpoint, ready_timeout) and (time.sleep(settle_s) or True)

    return restart


def _wait_ready(endpoint: str, timeout: int) -> bool:  # pragma: no cover - needs a server
    import requests
    base = endpoint.rstrip("/")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(base + "/health", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def run_sweep(grid: List[Dict[str, str]], restart_fn, endpoint: str,
              objective: str = "throughput", log=print) -> List[SweepPoint]:  # pragma: no cover - drives a server
    points: List[SweepPoint] = []
    for i, cfg in enumerate(grid):
        log(f"[cotune] {i + 1}/{len(grid)}: {config_flags(cfg) or '(defaults)'}")
        try:
            up = restart_fn(cfg)
        except Exception as e:
            points.append(SweepPoint(cfg, note=f"restart error: {e}"))
            log(f"    x restart error: {e}")
            continue
        if not up:
            points.append(SweepPoint(cfg, note="never became ready (OOM/crash?)"))
            log("    x not ready (likely OOM at this config)")
            continue
        st = measure.build_state(endpoint)
        if st is None:
            points.append(SweepPoint(cfg, note="measure failed"))
            continue
        p = SweepPoint(cfg, feasible=True, decode_tps_max_c=st.decode_tps_max_c,
                       decode_tps_single=st.decode_tps_single, kv_peak=st.kv_cache_usage,
                       preempt_per_s=st.preempt_per_s)
        points.append(p)
        log(f"    -> aggregate {p.decode_tps_max_c:.0f} tok/s | single {p.decode_tps_single:.0f} | "
            f"KV {p.kv_peak:.0%} | preempt {p.preempt_per_s:.2f}/s")
    points.sort(key=lambda p: score(p, objective), reverse=True)
    return points


def render(points: List[SweepPoint], objective: str) -> str:
    lines = [f"ampere-autotune — HALF-A co-tuning sweep (measured), objective={objective}\n"]
    best = next((p for p in points if p.feasible and score(p, objective) != float("-inf")), None)
    for p in points:
        tag = "WIN " if p is best else ("    " if p.feasible else "FAIL")
        cfg = config_flags(p.config) or "(defaults)"
        if p.feasible:
            lines.append(f"[{tag}] {cfg}\n        aggregate {p.decode_tps_max_c:.0f} tok/s | "
                         f"single {p.decode_tps_single:.0f} | KV {p.kv_peak:.0%} | preempt {p.preempt_per_s:.2f}/s"
                         + (f"  ({p.note})" if p.note else ""))
        else:
            lines.append(f"[{tag}] {cfg}\n        infeasible: {p.note}")
    if best:
        lines.append(f"\nBEST: {config_flags(best.config) or '(defaults)'} -> {best.decode_tps_max_c:.0f} tok/s aggregate.")
        lines.append("Apply it (restart with these flags), then re-measure with `recommend` to confirm.")
    else:
        lines.append("\nNo feasible config — every point OOM'd / thrashed. Loosen the grid.")
    return "\n".join(lines)


def run(args) -> int:  # pragma: no cover - drives a server
    grid = expand_grid(parse_sweep(args.sweep))
    endpoint = (getattr(args, "endpoint", None) or "http://localhost:8000").rstrip("/")
    obj = getattr(args, "objective", "throughput")
    print(f"[cotune] grid = {len(grid)} configs; each restarts the server (~minutes). objective={obj}")
    restart = make_restart_fn(args.restart_cmd, endpoint, getattr(args, "ready_timeout", 600))
    points = run_sweep(grid, restart, endpoint, obj)
    print("\n" + render(points, obj))
    return 0
