"""HALF-B per-GPU-UUID validated clock profile persistence.

Path: ~/.config/ampere-autotune/<uuid>.json (override base_dir for tests). Re-validate on
temp-delta / driver-change / UUID-mismatch. NEVER a shipped default (silicon lottery).
Stores BOTH the NVML clock-domain offset and the GDDR MT/s number (the half-rate footgun).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

DEFAULT_BASE_DIR = os.path.expanduser("~/.config/ampere-autotune")

# how far the junction temp may drift from the validated point before we re-characterize
REVALIDATE_TEMP_DELTA_C = 8.0


@dataclass
class Profile:
    uuid: str
    arch: str                          # "8.6"
    driver: str                        # "565.57"
    stock_mem_mhz: int
    max_stable_mem_offset_mhz: int     # NVML clock-domain offset (the value we set)
    max_stable_gpc_offset_mhz: int
    power_limit_w: Optional[int]
    validated_temp_c: Optional[float]
    decode_gain_pct: Optional[float]
    prefill_gain_pct: Optional[float]
    vllm_flags: dict

    @property
    def max_stable_mem_offset_mtps(self) -> int:
        """GDDR transfer-rate equivalent = 2x the clock-domain offset (footgun)."""
        return self.max_stable_mem_offset_mhz * 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["max_stable_mem_offset_mtps"] = self.max_stable_mem_offset_mtps  # store both
        return d


def profile_path(uuid: str, base_dir: str = DEFAULT_BASE_DIR) -> str:
    return os.path.join(base_dir, f"{uuid}.json")


def save(profile: Profile, base_dir: str = DEFAULT_BASE_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = profile_path(profile.uuid, base_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile.to_dict(), f, indent=2, sort_keys=True)
    os.replace(tmp, path)  # atomic
    return path


def load(uuid: str, base_dir: str = DEFAULT_BASE_DIR) -> Optional[Profile]:
    """Load a profile, or None if absent / corrupt / inconsistent (-> force re-characterize).

    Never raises on a bad file (H2): a truncated/garbage JSON, schema drift, or a tampered
    half-MT/s field all degrade to None rather than crashing the caller or — worse — loading
    a 2x-too-high corrupting clock.
    """
    path = profile_path(uuid, base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
        mtps = d.pop("max_stable_mem_offset_mtps", None)  # derived; not a constructor field
        prof = Profile(**d)
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return None  # corrupt / schema drift -> treat as absent
    # FOOTGUN integrity: the stored MT/s MUST equal 2x the clock-domain offset. A mismatch
    # means a tampered/confused file (e.g. the offset field holding the MT/s value) -> refuse.
    if mtps is not None and mtps != prof.max_stable_mem_offset_mtps:
        return None
    return prof


def needs_revalidation(profile: Profile, *, uuid: str, driver: str,
                       current_temp_c: Optional[float]) -> Optional[str]:
    """Return a human reason string if the profile must be re-validated, else None."""
    if profile.uuid != uuid:
        return f"UUID mismatch ({profile.uuid} != {uuid}) — never reuse another card's profile"
    if profile.driver != driver:
        return f"driver changed ({profile.driver} -> {driver})"
    if (current_temp_c is not None and profile.validated_temp_c is not None
            and abs(current_temp_c - profile.validated_temp_c) > REVALIDATE_TEMP_DELTA_C):
        return (f"junction drifted {abs(current_temp_c - profile.validated_temp_c):.0f}C "
                f"from validated {profile.validated_temp_c:.0f}C")
    return None
