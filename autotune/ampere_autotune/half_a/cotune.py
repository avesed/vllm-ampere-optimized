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
from typing import Callable, Dict, List, Optional

from . import measure

# values meaning "use the server default / disabled toggle" -> omit the flag entirely
_DEFAULTY = {"auto", "default", "", "-", "false", "off", "no", "0"}
# values meaning "enable this store_true toggle" -> emit the bare flag (no value)
_TRUEY = {"true", "on", "yes", "1"}


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
    """Render a config to a flag string. Default-y values are omitted; a store_true toggle
    (value true/on) emits the bare flag; everything else emits 'flag value'."""
    parts: List[str] = []
    for k, v in cfg.items():
        s = str(v).lower()
        if s in _DEFAULTY:
            continue
        parts.append(k if s in _TRUEY else f"{k} {v}")
    return " ".join(parts)


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
        # oversubscribe past the largest swept max-num-seqs so every config is actually saturated
        # (a fair throughput comparison; otherwise a big-cap config is never exercised)
        st = measure.build_state(endpoint, levels=(1, 32, 128), burst_c=160)
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


# ------------------------------------------------------------------------------------------------
# AUTO mode — adaptive coordinate-ascent search (no manual grid; the tuner picks knobs + values).
# ------------------------------------------------------------------------------------------------

@dataclass
class Trial:
    config: Dict[str, str]
    score: float
    feasible: bool
    kv: float = 0.0
    preempt: float = 0.0
    note: str = ""


def _seqs_ladder(start: int, ceiling: int) -> List[int]:
    v, out = start, []
    while v <= ceiling:
        out.append(v)
        v *= 2
    return out


def auto_tune(trial_fn: Callable[[Dict[str, str]], Trial],
              seed_seqs: int = 32, seqs_ceiling: int = 256, kv_high: float = 0.90, eps: float = 0.03,
              mnbt_candidates=("8192",), log=print):
    """Adaptive multi-variable search. trial_fn(config)->Trial restarts+measures one config.
    Returns (best Trial or None, history). PURE given trial_fn (tested with a fake)."""
    history: List[Trial] = []

    def T(cfg):
        t = trial_fn(cfg)
        history.append(t)
        log(f"  try {config_flags(cfg) or '(defaults)'} -> {t.score:.0f} "
            f"(kv {t.kv:.0%}, preempt {t.preempt:.2f}{'' if t.feasible else ', ' + t.note})")
        return t

    # Phase A — expand the primary knob (max-num-seqs) until it stops paying or hits a wall.
    best: Optional[Trial] = None
    capped = False
    for v in _seqs_ladder(seed_seqs, seqs_ceiling):
        t = T({"--max-num-seqs": str(v)})
        if not t.feasible or t.preempt > 0.05:
            capped = True
            break
        if best is not None and t.score <= best.score * (1 + eps):
            break                                   # diminishing returns
        best = t
        if t.kv >= kv_high:
            capped = True                           # KV ceiling -> a coupled unlock may extend it
            break
    if best is None:
        return None, history

    # Phase B — if a wall stopped us, AUTO-apply the coupled unlock (fp8) and resume climbing.
    if capped:
        best_v = int(best.config["--max-num-seqs"])
        for v in _seqs_ladder(best_v * 2, seqs_ceiling):
            t = T({"--max-num-seqs": str(v), "--kv-cache-dtype": "fp8"})
            if not t.feasible or t.preempt > 0.05:
                break
            if t.score <= best.score * (1 + eps):
                break
            best = t

    # Phase C — at the winning concurrency, sweep the secondary knob (token budget).
    for mnbt in mnbt_candidates:
        t = T({**best.config, "--max-num-batched-tokens": mnbt})
        if t.feasible and t.score > best.score * (1 + eps):
            best = t

    return best, history


def _live_trial(restart_fn, endpoint: str, objective: str):  # pragma: no cover - drives a server
    def trial(cfg: Dict[str, str]) -> Trial:
        try:
            up = restart_fn(cfg)
        except Exception as e:
            return Trial(cfg, float("-inf"), False, note=f"restart error: {e}")
        if not up:
            return Trial(cfg, float("-inf"), False, note="not ready (OOM?)")
        st = measure.build_state(endpoint, levels=(1, 32, 128), burst_c=160)
        if st is None:
            return Trial(cfg, float("-inf"), False, note="measure failed")
        sp = SweepPoint(cfg, True, st.decode_tps_max_c, st.decode_tps_single,
                        st.kv_cache_usage, st.preempt_per_s)
        return Trial(cfg, score(sp, objective), True, st.kv_cache_usage, st.preempt_per_s)
    return trial


def render_auto(best: Optional[Trial], history: List[Trial], objective: str) -> str:
    lines = [f"ampere-autotune — HALF-A AUTO-tune (adaptive search), objective={objective}\n",
             "search path:"]
    for t in history:
        tag = "ok  " if t.feasible else "FAIL"
        lines.append(f"  [{tag}] {config_flags(t.config) or '(defaults)'} -> "
                     + (f"{t.score:.0f} (kv {t.kv:.0%})" if t.feasible else t.note))
    if best:
        lines.append(f"\nBEST: {config_flags(best.config)} -> {best.score:.0f} ({objective}); kv {best.kv:.0%}.")
        lines.append(f"  {len(history)} configs tried (auto-chosen). Apply + re-measure with `recommend`.")
    else:
        lines.append("\nNo feasible config found from the seed — loosen the seed/ceiling.")
    return "\n".join(lines)


def run(args) -> int:  # pragma: no cover - drives a server
    endpoint = (getattr(args, "endpoint", None) or "http://localhost:8000").rstrip("/")
    obj = getattr(args, "objective", "throughput")
    restart = make_restart_fn(args.restart_cmd, endpoint, getattr(args, "ready_timeout", 600))

    if getattr(args, "auto", False):
        print(f"[cotune] AUTO adaptive search (no manual grid). objective={obj}")
        best, history = auto_tune(_live_trial(restart, endpoint, obj),
                                  seed_seqs=getattr(args, "seed", 32),
                                  seqs_ceiling=getattr(args, "seqs_ceiling", 256))
        print("\n" + render_auto(best, history, obj))
        return 0

    if not getattr(args, "sweep", None):
        print("[cotune] need --sweep <grid> (manual) or --auto (adaptive search).")
        return 2
    grid = expand_grid(parse_sweep(args.sweep))
    print(f"[cotune] grid = {len(grid)} configs; each restarts the server (~minutes). objective={obj}")
    points = run_sweep(grid, restart, endpoint, obj)
    print("\n" + render(points, obj))
    return 0
