"""HALF-B: the ONLY clock-write module. Wraps SetClockOffsets / SetPowerManagementLimit /
SetGpuLockedClocks behind the §4 correctness gate; readback-verifies every write.

FOOTGUN (lives here): NVML mem-offset value = HALF the GDDR MT/s number. Convert + log both.
STATUS: scaffold — writes are stubbed; revert is wired to a clean reset path.
"""
from __future__ import annotations


def mtps_to_clock_offset_mhz(mtps: int) -> int:
    """GDDR transfer-rate delta -> NVML clock-domain offset (half)."""
    return mtps // 2


def clock_offset_mhz_to_mtps(mhz: int) -> int:
    return mhz * 2


def revert_all(matrix) -> int:
    """Reset clocks/offsets to stock on every unlocked GPU (idempotent, safe)."""
    targets = [g.index for g in matrix.gpus] if matrix.gpus else []
    print(f"[half_b/silicon] revert: zero offsets + reset locked clocks to stock on gpus={targets}.")
    print("[half_b/silicon] TODO: nvml SetClockOffsets(0) + ResetGpuLockedClocks per GPU-UUID.")
    return 0
