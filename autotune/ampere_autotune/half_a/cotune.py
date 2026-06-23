"""HALF-A co-tuning sweep — the MEASURED autotuner (vs recommend's single heuristic pass).

Engine flags are startup-baked, so co-tuning = restart-per-config + measure + rank. Deployment-
agnostic: the caller supplies a `--restart-cmd` shell template with a `{flags}` placeholder
(docker / systemd / k8s / a launch script — the tool never assumes docker). For each config it
restarts, waits for /health, measures (concurrency sweep + under-load scrape), scores, and reports
the empirical best. NO privilege. The pure helpers (grid/parse/score/flags) are unit-tested.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

from . import measure


# ------------------------------------------------------------------------------------------------
# Built-in launcher — `ampere-autotune cotune --model X ...` auto-builds the restart-cmd so the user
# never hand-writes a docker/serve template. Explicit --restart-cmd always wins (escape hatch).
# ------------------------------------------------------------------------------------------------
_LAUNCH_NAME = "ampere-autotune-vllm"


def build_restart_cmd(args) -> Optional[str]:
    """Return the {flags}-templated restart command. Priority: explicit --restart-cmd > --model
    (auto-built) > None. The auto path supports launcher=docker (default) or vllm (bare)."""
    rc = getattr(args, "restart_cmd", None)
    if rc:
        return rc
    model = getattr(args, "model", None)
    if not model:
        return None
    port = getattr(args, "port", 8000) or 8000
    tp = getattr(args, "tp", None)
    extra = getattr(args, "serve_extra", None) or ""
    serve = f"--port {port}"
    if tp:
        serve += f" --tensor-parallel-size {tp}"
    if extra:
        serve += f" {extra}"
    launcher = getattr(args, "launcher", "docker") or "docker"

    if launcher == "vllm":                                  # bare process (vllm must be on PATH)
        return (f"pkill -f 'vllm serve' >/dev/null 2>&1 || true; sleep 2; "
                f"nohup vllm serve {model} {serve} {{flags}} > /tmp/{_LAUNCH_NAME}.log 2>&1 &")

    # docker (default). Absolute path -> mount its PARENT as /models (so sibling-relative symlinks
    # resolve) and serve /models/<base>; otherwise pass the model straight through (HF id / in-image).
    image = getattr(args, "image", None) or "vllm/vllm-openai:latest"
    gpus = getattr(args, "gpus", None) or "all"
    if os.path.isabs(model):
        parent, base = os.path.split(model.rstrip("/"))
        mount = f"-v {parent}:/models:ro "
        model_arg = f"/models/{base}"
    else:
        mount = ""
        model_arg = model
    return (f"docker rm -f {_LAUNCH_NAME} >/dev/null 2>&1; "
            f"docker run -d --name {_LAUNCH_NAME} --gpus {gpus} --shm-size=8g {mount}"
            f"-p {port}:8000 {image} --model {model_arg} {serve} {{flags}}")

# values meaning "use the server default / disabled toggle" -> omit the flag entirely
_DEFAULTY = {"auto", "default", "", "-", "false", "off", "no", "0"}
# values meaning "enable this store_true toggle" -> emit the bare flag (no value)
_TRUEY = {"true", "on", "yes", "1"}

# Flags that DO NOT EXIST / are inert in vLLM v0.23 (verified against source, see autotune/PARAMS.md)
# -> emitting any of these makes a BROKEN restart command. Maps the bad flag to its replacement.
BANNED_FLAGS = {
    "--cuda-graph-sizes": "--cudagraph-capture-sizes",
    "--max-seq-len-to-capture": "--max-cudagraph-capture-size",
    "--swap-space": "--kv-offloading-size",
    "--num-scheduler-steps": "(removed in V1 — single-step only)",
    "--num-lookahead-slots": "(removed in V1)",
    "--preemption-mode": "(removed in V1 — recompute only)",
    "--max-parallel-loading-workers": "(inert in V1 — ignored)",
}


def banned_in(flags) -> Dict[str, str]:
    """Subset of `flags` that are removed/renamed/inert in v0.23 -> {bad: replacement}."""
    return {f: BANNED_FLAGS[f] for f in flags if f in BANNED_FLAGS}


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


# GUARD: every vLLM bring-up waits AT MOST this many seconds for /health, then gives up (so a
# broken config / crash-loop never hangs the sweep). Hard cap — a larger --ready-timeout is clamped.
MAX_WAIT_S = 600


def _clamp_wait(t) -> int:
    """Bound any requested readiness wait to (0, MAX_WAIT_S]; missing/0 -> MAX_WAIT_S."""
    try:
        t = int(t)
    except (TypeError, ValueError):
        t = MAX_WAIT_S
    return max(1, min(t if t > 0 else MAX_WAIT_S, MAX_WAIT_S))


def make_restart_fn(template: str, endpoint: str, ready_timeout: int = MAX_WAIT_S,
                    settle_s: float = 3.0) -> Callable[[Dict[str, str]], bool]:  # pragma: no cover - drives a server
    """Build a restart hook from a shell template: substitute {flags}, run it, wait for /health
    (bounded by the MAX_WAIT_S guard so a bad config never hangs the run)."""
    if "{flags}" not in template:
        raise ValueError("--restart-cmd must contain the {flags} placeholder")
    ready_timeout = _clamp_wait(ready_timeout)

    def restart(cfg: Dict[str, str]) -> bool:
        cmd = template.replace("{flags}", config_flags(cfg))
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        return _wait_ready(endpoint, ready_timeout) and (time.sleep(settle_s) or True)

    return restart


def _wait_ready(endpoint: str, timeout: int = MAX_WAIT_S) -> bool:  # pragma: no cover - needs a server
    import requests
    base = endpoint.rstrip("/")
    deadline = time.time() + _clamp_wait(timeout)
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


def _predict_walls(samples):
    """(feasible_wall, feasible_wall_fp8) from the capacity model — the RELIABLE part of the math.
    2 points -> roofline+KV fit; else the MOST CONSERVATIVE single point (highest per-seq KV ->
    smallest wall, so we never over-predict feasibility). None if it can't fit."""
    from .analytical import predict, predict_from_baseline, Sample
    pts = [Sample(s, kv, tps) for (s, kv, tps) in samples]
    if len({p.seqs for p in pts}) >= 2:
        p = predict(pts)
        if p:
            return p.feasible_seqs, p.feasible_seqs_fp8
    cons = max(pts, key=lambda p: (p.kv / p.seqs) if p.seqs else 0.0)
    bp = predict_from_baseline(cons)
    return (bp.feasible_seqs, bp.feasible_seqs_fp8) if bp else (None, None)


def _valid(t) -> bool:
    """A usable result: feasible (server came up) AND not thrashing (finite score; score() -inf's
    a feasible-but-preempting config)."""
    return t.feasible and t.score > float("-inf")


def auto_tune(trial_fn: Callable[[Dict[str, str]], Trial],
              seed_seqs: int = 32, seqs_ceiling: int = 512, eps: float = 0.03, patience: int = 2,
              mnbt_candidates=(), log=print):
    """PREDICT-THEN-VERIFY adaptive search. Two cheap probes anchor the CAPACITY model (reliable);
    the predicted feasible wall BOUNDS the climb (no blind-OOM rungs); verify upward to the THROUGHPUT
    KNEE (not the wall, with patience for noise); fp8 is tried ONLY if capacity stopped us, anchored
    apples-to-apples (fp8 at the same seqs first) and adopted ONLY if it beats the non-fp8 best by eps
    (so a KV-maxing config with ~0 gain is rejected). Returns (best Trial or None, history)."""
    history: List[Trial] = []

    def T(cfg):
        t = trial_fn(cfg)
        history.append(t)
        log(f"  try {config_flags(cfg) or '(defaults)'} -> {t.score:.0f} "
            f"(kv {t.kv:.0%}, preempt {t.preempt:.2f}{'' if t.feasible else ', ' + t.note})")
        return t

    # PROBE — a feasible AND non-thrashing seed (halve down otherwise), then one rung up.
    s = seed_seqs
    a = T({"--max-num-seqs": str(s)})
    while not _valid(a) and s > 1:
        s //= 2
        a = T({"--max-num-seqs": str(s)})
    if not _valid(a):
        return None, history, []
    b = T({"--max-num-seqs": str(s * 2)})
    samples = [(s, a.kv, a.score)]
    best = a
    b_oom = not b.feasible                                   # hard OOM at s*2 (a measured infeasibility)
    if _valid(b):
        samples.append((s * 2, b.kv, b.score))
        if b.score > a.score:
            best = b

    # PREDICT — the feasible wall bounds the climb (never probe past what the math says fits, and
    # never past a MEASURED OOM at s*2).
    wall, wall_fp8 = _predict_walls(samples)
    wall = min(wall or seqs_ceiling, seqs_ceiling)
    wall_fp8 = min(wall_fp8 or wall, seqs_ceiling)
    if b_oom:
        wall = min(wall, s * 2 - 1)
    log(f"  [predict] feasible wall ~{wall} seqs (fp8 ~{wall_fp8}); climb to the throughput KNEE within it")

    def climb(extra, start):
        """Verify upward (x2) from `start` up to the relevant wall; keep the argmax; stop at the
        throughput knee (patience consecutive non-improving rungs). Returns (argmax, hit_capacity)
        where hit_capacity = stopped by OOM/thrash/ceiling rather than a genuine plateau."""
        ceiling = wall_fp8 if extra.get("--kv-cache-dtype") == "fp8" else wall
        cur = start
        stale = 0
        v = int(start.config["--max-num-seqs"]) * 2
        while v <= ceiling:
            t = T({**extra, "--max-num-seqs": str(v)})
            if not t.feasible or t.preempt > 0.05:          # OOM or thrash -> capacity wall
                return cur, True
            if t.score > cur.score * (1 + eps):
                cur, stale = t, 0
            else:
                if t.score > cur.score:
                    cur = t                                 # keep the argmax even on a sub-eps gain
                stale += 1
                if stale >= patience:
                    return cur, False                       # genuine throughput knee
            v *= 2
        return cur, (stale == 0)                            # exhausted the wall while still improving

    # CLIMB (only the accuracy-neutral knob) to the throughput knee within the predicted wall.
    best, hit_cap = climb({}, best)

    # Secondary accuracy-neutral knob (token budget) — kept only if it clears the eps bar.
    for mnbt in mnbt_candidates:
        t = T({**best.config, "--max-num-batched-tokens": mnbt})
        if _valid(t) and t.score > best.score * (1 + eps):
            best = t

    # EXTRA recommendations — opt-in, NOT auto-swept (they touch precision/accuracy or the checkpoint;
    # the operator decides against their quality budget). kv-cache-dtype / MTP / mamba-dtype live here,
    # never in the sweep above.
    recs: List[str] = []
    if hit_cap and _valid(best):
        recs.append(f"capacity-bound at ~{best.config['--max-num-seqs']} seqs (KV wall) while throughput was "
                    f"still rising. To push further -> --kv-cache-dtype fp8 ~2x KV room (wall ~{wall} -> ~{wall_fp8}; "
                    f"NOT auto-swept, verify accuracy — sm80/86 fp8 is emulated/storage), or the accuracy-neutral "
                    f"--gpu-memory-utilization up / --max-model-len trim.")
    recs.append("per-stream (single/few-session) decode speed is bandwidth-bound; the lever is MTP/spec-decode "
                "(needs an mtp head + an accept-rate check), not these flags. See --objective latency.")
    return (best if _valid(best) else None, history, recs)


def _live_trial(restart_fn, endpoint: str, objective: str, temperature=None):  # pragma: no cover - drives a server
    def trial(cfg: Dict[str, str]) -> Trial:
        try:
            up = restart_fn(cfg)
        except Exception as e:
            return Trial(cfg, float("-inf"), False, note=f"restart error: {e}")
        if not up:
            return Trial(cfg, float("-inf"), False, note="not ready (OOM?)")
        st = measure.build_state(endpoint, levels=(1, 32, 128), burst_c=160, temperature=temperature)
        if st is None:
            return Trial(cfg, float("-inf"), False, note="measure failed")
        sp = SweepPoint(cfg, True, st.decode_tps_max_c, st.decode_tps_single,
                        st.kv_cache_usage, st.preempt_per_s)
        return Trial(cfg, score(sp, objective), True, st.kv_cache_usage, st.preempt_per_s)
    return trial


def render_lowc_advice(tps, c: int) -> str:
    """Single/few-session: nothing accuracy-neutral to AUTO-SWEEP (per-stream is bandwidth-bound).
    Measure the running server (no restart) + emit the opt-in levers."""
    if tps is None:
        return "single/few-session: endpoint unreachable."
    head = (f"ampere-autotune — single/few-session ({c}-concurrency) throughput = {tps:.0f} tok/s "
            f"(TPOT ~{1000.0 / tps:.1f} ms)" if tps > 0 else "single/few-session: no throughput measured")
    return "\n".join([head, "",
        "Nothing to auto-sweep here: per-stream decode is bandwidth-bound and accuracy-neutral flags don't move it.",
        "EXTRA levers (opt-in, NOT auto-swept — they touch accuracy or the checkpoint):",
        "  - MTP / spec-decode K=2 — THE per-stream lever; needs an mtp head + an accept-rate/quality check.",
        "  - long-ctx only: --kv-cache-dtype fp8 / --mamba-cache-dtype (the GDN-state analogue) — trade a little",
        "    accuracy for bytes; verify quality (sm80/86 fp8 is emulated/storage).",
        "  - max-num-seqs / max-model-len are CAPACITY knobs — they do NOT change per-stream speed."])


def mtp_sweep(restart_fn, endpoint: str, ks=(0, 1, 2, 3), method: str = "qwen3_5_mtp",
              c: int = 1, prompt=None, temperature=None, log=print):  # pragma: no cover - needs a server
    """RESTART-class sweep of MTP/spec-decode K (num_speculative_tokens). For each K: restart with
    the spec config (K=0 = baseline, no spec), measure single-stream decode tok/s + accept-rate.
    ACCEPT-RATE (hence optimal K) is WORKLOAD-DEPENDENT — measured on the given prompt only."""
    results = []
    for k in ks:
        if k == 0:
            cfg = {}                                            # baseline: no speculative decoding
        else:
            # single-quoted JSON survives config_flags (opaque value) + the shell restart template
            cfg = {"--speculative-config": f"'{{\"method\":\"{method}\",\"num_speculative_tokens\":{k}}}'"}
        try:
            up = restart_fn(cfg)
        except Exception as e:
            results.append((k, None, None, f"restart error: {e}"))
            continue
        if not up:
            results.append((k, None, None, "not ready (OOM / bad spec config?)"))
            log(f"  K={k}: not ready")
            continue
        tps = measure.lowc_throughput(endpoint, c=c, prompt=prompt, temperature=temperature)
        acc = measure.spec_accept_len(endpoint) if k > 0 else None
        results.append((k, tps, acc, ""))
        log(f"  K={k}: {tps:.0f} tok/s" + (f", accept-len {acc:.2f}" if acc is not None else ""))
    return results


def render_mtp(results, c: int) -> str:
    lines = [f"ampere-autotune — MTP/spec-decode K-sweep (single-stream c={c}; decode tok/s)\n",
             "  K | decode tok/s | accept-len | note"]
    base = next((t for (k, t, _, _) in results if k == 0 and t is not None), None)
    feasible = [(k, t, a) for (k, t, a, _) in results if t is not None]
    for k, tps, acc, note in results:
        if tps is None:
            lines.append(f"  {k} | (failed: {note})")
            continue
        gain = f" ({(tps / base - 1) * 100:+.0f}% vs K=0)" if base and k != 0 else ""
        accs = f"{acc:.2f}" if acc is not None else "n/a"
        lines.append(f"  {k} | {tps:>11.0f}{gain} | {accs:>10}")
    if feasible:
        bk, bt, _ = max(feasible, key=lambda r: r[1])
        lines.append(f"\nBEST on THIS prompt: K={bk} -> {bt:.0f} tok/s single-stream.")
    lines.append("\nWORKLOAD-DEPENDENT — the crux: accept-rate (so the optimal K) varies by content. Structured /")
    lines.append("code / repetitive text accepts MORE (higher K pays); creative / high-entropy accepts LESS")
    lines.append("(K=1, or spec off). This sweep used the GIVEN prompt -> run with --scenario/--prompt-file")
    lines.append("matching your real traffic before fixing K. Also: MTP helps LOW concurrency only (spec compute")
    lines.append("hurts aggregate at high batch, no v0.23 auto-disable) -> gate spec on/off per QPS regime at startup.")
    return "\n".join(lines)


def batch_curve(endpoint: str, levels, max_tokens: int = 128,
                prompt=None, temperature=None):  # pragma: no cover - needs a server
    """Profile the RUNNING server (no restart) across offered concurrency -> per-batch aggregate AND
    per-session tok/s. Shows the throughput<->latency tradeoff so you pick the operating batch."""
    try:
        mid, _ = measure.model_info(endpoint)
    except Exception:
        return []
    sweep = measure.concurrency_sweep(endpoint, mid, tuple(levels), max_tokens, prompt, temperature)
    return [(c, agg, (agg / c if c else 0.0)) for (c, agg) in sweep]


def render_curve(rows) -> str:
    if not rows:
        return "batch-curve: endpoint unreachable."
    lines = ["ampere-autotune — batch (concurrency) curve: aggregate throughput <-> per-session speed\n",
             "  batch | aggregate tok/s | per-session tok/s | per-session TPOT ms"]
    for c, agg, ps in rows:
        tpot = (1000.0 / ps) if ps > 0 else 0.0
        lines.append(f"  {c:>5} | {agg:>15.0f} | {ps:>17.0f} | {tpot:>17.1f}")
    lines.append("\nPer-session tok/s falls as batch grows (the same weight read is shared by more streams).")
    lines.append("Pick the LARGEST batch whose per-session tok/s still meets your latency SLA -> that is the")
    lines.append("max-num-seqs that maximizes aggregate throughput within the per-user speed you require.")
    return "\n".join(lines)


def render_auto(best: Optional[Trial], history: List[Trial], recs, objective: str) -> str:
    lines = [f"ampere-autotune — HALF-A AUTO-tune (adaptive search), objective={objective}\n",
             "search path (accuracy-neutral knobs only):"]
    for t in history:
        tag = "ok  " if t.feasible else "FAIL"
        lines.append(f"  [{tag}] {config_flags(t.config) or '(defaults)'} -> "
                     + (f"{t.score:.0f} (kv {t.kv:.0%})" if t.feasible else t.note))
    if best:
        lines.append(f"\nBEST: {config_flags(best.config)} -> {best.score:.0f} ({objective}); kv {best.kv:.0%}.")
        lines.append(f"  {len(history)} configs tried (auto-chosen). Apply + re-measure with `recommend`.")
    else:
        lines.append("\nNo feasible config found from the seed — loosen the seed/ceiling.")
    if recs:
        lines.append("\nEXTRA recommendations (opt-in, NOT auto-swept — touch accuracy / the checkpoint):")
        for r in recs:
            lines.append(f"  - {r}")
    return "\n".join(lines)


def resolve_prompt(args):
    """--prompt-file (custom) > --scenario (preset) > None (default prose). Returns the prompt string."""
    pf = getattr(args, "prompt_file", None)
    if pf:
        with open(pf, "r") as f:
            return f.read().strip()
    scen = getattr(args, "scenario", None)
    if scen:
        from .measure import SCENARIO_PROMPTS
        return SCENARIO_PROMPTS.get(scen, SCENARIO_PROMPTS["general"])
    return None


def run(args) -> int:  # pragma: no cover - drives a server
    from .results import emit
    endpoint = (getattr(args, "endpoint", None) or "http://localhost:8000").rstrip("/")
    if getattr(args, "model", None):                # built-in launcher -> endpoint follows --port
        endpoint = f"http://localhost:{getattr(args, 'port', 8000) or 8000}"
    obj = getattr(args, "objective", "throughput")
    prompt = resolve_prompt(args)
    temp = getattr(args, "temperature", None)          # None -> don't send -> vLLM model default (never temp=0)
    pnote = (f" [prompt={getattr(args, 'scenario', None) or 'file' if getattr(args, 'prompt_file', None) else 'default'}"
             f", temp={'default' if temp is None else temp}]")

    if getattr(args, "batch_curve", False):         # no restart — profile the running server as-is
        levels = [int(x) for x in str(getattr(args, "levels", "1,2,4,8,16,32,64,128")).split(",") if x.strip()]
        print(f"[cotune] batch curve (no restart) over concurrency {levels}{pnote}")
        emit("\n" + render_curve(batch_curve(endpoint, levels, prompt=prompt, temperature=temp)), "batch-curve", args)
        return 0

    if obj == "latency":                            # single/few-session: measure + recommend, NO restart/sweep
        c = getattr(args, "concurrency", 1)
        print(f"[cotune] single/few-session ({c}-conc): measure + recommend{pnote}.")
        tps = measure.lowc_throughput(endpoint, c=c, prompt=prompt, temperature=temp)
        report = render_lowc_advice(tps, c)
        if tps:                                     # closed-loop delta: before->after vs the last run
            from . import delta
            try:
                mid = measure.model_info(endpoint)[0]
            except Exception:
                mid = "?"
            key = f"{endpoint}|{mid}|latency-c{c}"
            prev = delta.load_result(key)
            if prev:
                report += "\n\n" + delta.render_delta(prev.get("metrics", {}), {"single_tps": round(tps, 1)},
                                                      prev.get("ts"))
            delta.save_result(key, {"single_tps": round(tps, 1)})
        emit("\n" + report, "latency", args, data={"single_tps": tps, "concurrency": c})
        return 0

    rc = build_restart_cmd(args)                    # --restart-cmd > --model (auto-built) > None
    if not rc:
        print("[cotune] --sweep/--auto need a server to (re)launch: pass --model (built-in launcher) "
              "or --restart-cmd \"...{flags}...\" (--batch-curve / --objective latency need neither).")
        return 2
    restart = make_restart_fn(rc, endpoint, getattr(args, "ready_timeout", 600))

    if getattr(args, "mtp_sweep", False):
        ks = tuple(int(x) for x in str(getattr(args, "mtp_ks", "0,1,2,3")).split(",") if x.strip())
        method = getattr(args, "spec_method", "qwen3_5_mtp")
        c = getattr(args, "concurrency", 1)
        print(f"[cotune] MTP/spec-decode K-sweep {ks} (method={method}, c={c}){pnote}; each K restarts.")
        res = mtp_sweep(restart, endpoint, ks, method, c, prompt=prompt, temperature=temp)
        emit("\n" + render_mtp(res, c), "mtp-sweep", args,
             data={"k": [r[0] for r in res], "tok_s": [r[1] for r in res], "accept": [r[2] for r in res]})
        return 0

    if getattr(args, "auto", False):
        print(f"[cotune] AUTO predict-then-verify (accuracy-neutral sweep). objective={obj}{pnote}")
        best, history, recs = auto_tune(_live_trial(restart, endpoint, obj, temperature=temp),
                                        seed_seqs=getattr(args, "seed", 32),
                                        seqs_ceiling=getattr(args, "seqs_ceiling", 256))
        emit("\n" + render_auto(best, history, recs, obj), "auto", args,
             data={"best": best.config if best else None})
        return 0

    if not getattr(args, "sweep", None):
        print("[cotune] need --sweep <grid> (manual) or --auto (adaptive search).")
        return 2
    spec = parse_sweep(args.sweep)
    bad = banned_in(spec)
    if bad:                                          # refuse a sweep that would build a broken restart
        for f, repl in bad.items():
            print(f"[cotune] REFUSED: {f} does not exist / is inert in vLLM v0.23 -> use {repl} (see autotune/PARAMS.md)")
        return 2
    grid = expand_grid(spec)
    print(f"[cotune] grid = {len(grid)} configs; each restarts the server (~minutes). objective={obj}")
    points = run_sweep(grid, restart, endpoint, obj)
    emit("\n" + render(points, obj), "sweep", args, data={"points": [p.to_dict() for p in points]})
    return 0
