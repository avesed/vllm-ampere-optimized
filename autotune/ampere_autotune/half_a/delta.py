"""Closed-loop delta (borrowed from jungledesh/profile delta.rs/drift.rs) — persist each run's
result keyed by endpoint+model, and on the next run show before->after + flag regressions. Lets you
verify a config change actually helped (vs guessing). Pure render; tiny JSON state file.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, Optional

def _dir() -> str:
    from .results import state_dir
    return os.path.join(state_dir(), "delta")


def _path(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:180]
    return os.path.join(_dir(), safe + ".json")


def save_result(key: str, metrics: Dict[str, float]) -> None:  # pragma: no cover - filesystem
    try:
        os.makedirs(_dir(), exist_ok=True)
        json.dump({"ts": time.time(), "metrics": metrics}, open(_path(key), "w"))
    except OSError:
        pass


def load_result(key: str) -> Optional[dict]:  # pragma: no cover - filesystem
    try:
        return json.load(open(_path(key)))
    except (OSError, ValueError):
        return None


def render_delta(prev_metrics: Dict[str, float], cur_metrics: Dict[str, float],
                 prev_ts: Optional[float] = None, regress_frac: float = 0.05,
                 higher_is_better=("single_tps", "aggregate_tps", "decode_tps", "tok_s")) -> str:
    """before -> after table over the SHARED metrics, with a +/-% and a REGRESSION tag."""
    shared = [m for m in cur_metrics if m in prev_metrics]
    if not shared:
        return "closed-loop delta: no comparable prior run (saved this one as the new baseline)."
    age = ""
    if prev_ts:
        mins = max(0, (time.time() - prev_ts) / 60.0)
        age = f" (prior run {mins:.0f} min ago)"
    lines = [f"== closed-loop delta vs last run{age} =="]
    for m in shared:
        a, b = prev_metrics[m], cur_metrics[m]
        pct = ((b / a - 1) * 100) if a else 0.0
        hib = any(k in m for k in higher_is_better)
        improved = (b > a) if hib else (b < a)
        regressed = (b < a * (1 - regress_frac)) if hib else (b > a * (1 + regress_frac))
        if regressed:
            tag = "  <-- REGRESSION"
        elif abs(pct) < 2.0:
            tag = "  (stable)"                       # within noise — not a real change
        elif improved:
            tag = "  (improved)"
        else:
            tag = ""
        lines.append(f"  {m}: {a:.1f} -> {b:.1f} ({pct:+.0f}%){tag}")
    return "\n".join(lines)
