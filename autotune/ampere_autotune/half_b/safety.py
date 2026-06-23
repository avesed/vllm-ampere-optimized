"""HALF-B launch-time consent gates. These fire ONLY on the real --hw WRITE path (never on
--dry-run / advisory). All interactive: the standalone tool prompts the user on the host.

Two gates:
  1. confirm_oc_damage_warning  — a prominent HARDWARE-DAMAGE warning at startup; requires an
     explicit 'yes' (or --yes for non-interactive). mem-OC on no-ECC GeForce can silently corrupt,
     hang the GPU, or shorten its life / void warranty.
  2. confirm_vram_in_use        — if ANOTHER app already holds VRAM on the target card, warn and
     ask before proceeding (the OC is GLOBAL to the card, the soak may OOM, and the back-side
     chips behind the other app won't be fully soaked).

The verdict/text helpers are pure (unit-tested); only the input() prompt touches a TTY.
"""
from __future__ import annotations

import sys
from typing import Optional, Tuple

OC_DAMAGE_WARNING = (
    "\n========================  HARDWARE RISK — READ  ========================\n"
    " ampere-autotune --hw OVERCLOCKS GPU MEMORY (raises the GDDR clock).\n"
    " On consumer GeForce there is NO ECC, so an unstable clock can SILENTLY\n"
    " CORRUPT results; an excessive clock can HANG the GPU (needs reboot); and\n"
    " sustained over-clock/over-heat can cause PERMANENT HARDWARE DAMAGE and may\n"
    " VOID YOUR WARRANTY. You proceed AT YOUR OWN RISK.\n"
    " (The tool gates with memtest + golden + a thermal abort, but CANNOT remove\n"
    "  the risk — junction temp is invisible on most GeForce drivers.)\n"
    "======================================================================="
)


def _ask(question: str, *, reader=input, stream=None) -> bool:
    stream = stream or sys.stderr
    print(question, file=stream, flush=True)
    try:
        ans = reader("Type 'yes' to continue (anything else aborts): ")
    except (EOFError, KeyboardInterrupt):
        return False
    return str(ans).strip().lower() == "yes"


def confirm_oc_damage_warning(*, force: bool = False, reader=input, stream=None) -> bool:
    """Show the damage warning; require an explicit 'yes' unless force (--yes)."""
    stream = stream or sys.stderr
    print(OC_DAMAGE_WARNING, file=stream, flush=True)
    if force:
        print("[--yes] hardware-risk warning acknowledged non-interactively.", file=stream, flush=True)
        return True
    return _ask("Proceed with GPU memory overclocking?", reader=reader, stream=stream)


def vram_in_use(used_mib: Optional[float], n_procs: Optional[int],
                threshold_mib: float = 200.0) -> Tuple[bool, str]:
    """Pure: is the card already in use by another app, + the message. >threshold resident OR any
    running process counts (idle GeForce sits at a few MiB)."""
    used = used_mib or 0
    procs = n_procs or 0
    in_use = used > threshold_mib or procs > 0
    msg = (f"Another application is using this card ({used:.0f} MiB resident"
           + (f", {procs} process(es)" if procs else "") + ").")
    return in_use, msg


def confirm_vram_in_use(used_mib: Optional[float], n_procs: Optional[int], *, force: bool = False,
                        threshold_mib: float = 200.0, reader=input, stream=None) -> bool:
    """If the card is in use, warn + ask 'do you want to proceed?'. Clear cards pass silently."""
    stream = stream or sys.stderr
    busy, msg = vram_in_use(used_mib, n_procs, threshold_mib)
    if not busy:
        return True
    print(msg, file=stream, flush=True)
    if force:
        print("[--yes] proceeding despite the card being in use.", file=stream, flush=True)
        return True
    return _ask("Do you want to proceed? (the OC affects the WHOLE card; the soak may OOM and the "
                "chips behind the other app won't be fully soaked)", reader=reader, stream=stream)
