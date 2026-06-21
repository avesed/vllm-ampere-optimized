"""HALF-B: mode dispatch + the monitor-only watchdog (default).

monitor-only [DEFAULT]: a subtract-only watchdog that reverts/derates on any
Xid/throttle/junction/golden trip, ACTS BEFORE IT LOGS, and never raises an offset.
safe-adapt: applies a validated profile (FREE-knob hold + offset derate-only).
characterize: the real climb (gated; requires a prior --dry-run).
STATUS: scaffold. See DESIGN.md §9 / ../docs/RESEARCH-autotune-gpu-oc.md §9.
"""
from __future__ import annotations


def run(args, matrix) -> int:
    mode = getattr(args, "mode", "monitor-only")
    dry = getattr(args, "dry_run", False)
    unlocked = [g.index for g in matrix.gpus if g.half_b_unlocked]
    print(f"[half_b/monitor] mode={mode} dry_run={dry} gpus={unlocked}")
    if mode == "characterize":
        print("[half_b/monitor] TODO: adaptive search (105up/30down/15snap) + golden gate + heat-soak.")
    elif mode == "safe-adapt":
        print("[half_b/monitor] TODO: load per-UUID profile; FREE-knob hold + derate-only.")
    else:
        print("[half_b/monitor] TODO: 4Hz subtract-only watchdog (Xid/throttle/junction/golden -> revert).")
    return 0
