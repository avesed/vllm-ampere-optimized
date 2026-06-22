"""Economics block (borrowed from jungledesh/profile) — turn tok/s into energy + $ per token.

Pure math + an NVML power read; no privilege. J/tok and tok/W need only power; $/1M needs a price.
"""
from __future__ import annotations

from typing import Dict, Optional


def metrics(tps: Optional[float], power_w: Optional[float], cost_per_hr: Optional[float] = None) -> Dict[str, float]:
    """{j_per_tok, tok_per_w, usd_per_mtok} — only the keys that have the inputs to compute them."""
    out: Dict[str, float] = {}
    if tps and tps > 0 and power_w and power_w > 0:
        out["j_per_tok"] = power_w / tps          # joules per token = W / (tok/s)
        out["tok_per_w"] = tps / power_w
    if tps and tps > 0 and cost_per_hr and cost_per_hr > 0:
        out["usd_per_mtok"] = cost_per_hr * 1e6 / (tps * 3600.0)
    return out


def fmt(tps: Optional[float], power_w: Optional[float], cost_per_hr: Optional[float] = None) -> str:
    """One-line economics summary; '' if there's nothing to show (no power)."""
    m = metrics(tps, power_w, cost_per_hr)
    if not m:
        return ""
    parts = []
    if "tok_per_w" in m:
        parts.append(f"{m['tok_per_w']:.2f} tok/W")
        parts.append(f"{m['j_per_tok']:.3f} J/tok")
    if "usd_per_mtok" in m:
        parts.append(f"${m['usd_per_mtok']:.2f}/1M tok")
    suffix = "" if "usd_per_mtok" in m else "  (pass --cost-per-hour for $/1M)"
    return "economics: " + " | ".join(parts) + (f"  @ {power_w:.0f} W" if power_w else "") + suffix


def gpu_power_w() -> Optional[float]:  # pragma: no cover - needs a GPU
    """Total board power (W) summed across the visible (non-idle) GPUs, via NVML. None if unreadable."""
    from ..preflight import _nvml
    with _nvml.Session() as sess:
        if not sess.ok:
            return None
        total = 0.0
        got = False
        for i in range(sess.device_count()):
            h = sess.handle(i)
            if h is None:
                continue
            c = _nvml.call("nvmlDeviceGetPowerUsage", h)   # milliwatts
            u = _nvml.call("nvmlDeviceGetUtilizationRates", h)
            busy = (not u.ok) or (getattr(u.value, "gpu", 0) or 0) > 5
            if c.ok and busy:
                total += c.value / 1000.0
                got = True
        return total if got else None
