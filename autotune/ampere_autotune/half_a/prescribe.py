"""HALF-A: classify the bottleneck (roofline + R1-R5) and PRESCRIBE vLLM startup flags.

Recommend-only: emits a flag set + the exact restart command; never applies anything (engine
flags are startup-baked, no hot-reload — see ../docs/RESEARCH-autotune-gpu-oc.md §5.1). Preferred
home is upstream jungledesh/profile (Apache-2.0); this is the thin fallback.
"""
from __future__ import annotations

from typing import Dict

from .classify import HwSpec, classify, FlagRec


def render(recs, endpoint: str) -> str:
    lines = [f"ampere-autotune — HALF-A vLLM-flag recommender (recommend-only) [{endpoint}]\n"]
    merged: Dict[str, object] = {}
    for r in recs:
        lines.append(f"[{r.severity}] {r.rule}\n  {r.finding}")
        if r.flags:
            lines.append("  suggest: " + " ".join(f"{k}={v}" for k, v in r.flags.items()))
        if r.reason:
            lines.append(f"  why: {r.reason}")
        lines.append("")
        # Only LITERAL flag values join the copy-paste restart command; pointer/placeholder
        # values (e.g. "(MTP if...)", "<your true p99 context>") stay in their per-rule suggest only.
        for k, v in r.flags.items():
            sv = str(v)
            if "(" not in sv and "<" not in sv and " " not in sv:
                merged[k] = v
    if merged:
        flagstr = " ".join(f"{k} {v}" for k, v in merged.items())
        lines.append("To apply (engine flags are startup-baked -> RESTART the server, drained):")
        lines.append(f"  vllm serve <model> {flagstr}")
        lines.append("(verify the delta by re-running `ampere-autotune recommend`.)")
    else:
        lines.append("No flag change recommended at the probed load.")
    return "\n".join(lines)


def run(args, matrix) -> int:
    endpoint = (getattr(args, "endpoint", None) or "http://localhost:8000").rstrip("/")
    from . import measure
    state = measure.build_state(endpoint)
    if state is None:
        print(f"[half_a] no reachable vLLM at {endpoint} (start one, or pass --endpoint). "
              "HALF-A needs a running server to measure.")
        return 2
    # HwSpec: default 3090/9B/W4A8; refine from the local SKU if a GPU is visible.
    hw = HwSpec()
    if matrix.gpus:
        nm = (matrix.gpus[0].sku.get("name") or "").upper()
        if "3080" in nm:
            hw.peak_bw_gbs = 760.0
    recs = classify(state, hw)
    if getattr(args, "json", False):
        import json
        print(json.dumps([r.to_dict() if isinstance(r, FlagRec) else r for r in recs], indent=2))
    else:
        print(render(recs, endpoint))
    return 0
