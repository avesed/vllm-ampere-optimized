"""HALF-B: the ONLY clock-write module. Wraps the GDDR mem-clock VF offset (and reset) behind
the §4 correctness gate; readback-verifies every write. Roll-own NVML via ctypes (no pynvml dep
— every OC tool bottoms out on these same calls; a dep only adds a daemon between us and an
instant revert).

FOOTGUN (lives here): NVML mem-offset value = HALF the GDDR MT/s number. Convert + log both.
VRAM-FILL (lives here): the 3090/3080 carry GDDR6X on BOTH PCB sides; the back-side chips have
the weakest cooling and are the heat-soak failure point, but VRAM fills front-first — an 8GB
test buffer never touches them. So the bw/soak buffer MUST fill (near) all free VRAM, sized so
even the top chip's address range is covered. ``fill_gib`` computes that (pure, unit-tested).

STATUS: writes implemented (NVML SetMemClkVfOffset); GPU-bound, root-only. Pure helpers tested.
"""
from __future__ import annotations

import atexit
import ctypes
from typing import Optional, Set, Tuple

_LIB = "libnvidia-ml.so.1"
_GIB = 1024 ** 3
# fail-closed hard cap used ONLY when the device VF-offset range can't be read (legacy/transient):
# refuse to write above this rather than fall through with no bound (a too-high offset hard-hangs).
_HARD_CAP_MHZ = 1500
# UUIDs we've written a NON-ZERO offset to -> zeroed by the atexit net (crash/Ctrl-C insurance).
_applied: Set[str] = set()
# headroom for the CUDA context so the fill doesn't OOM, but SMALL so we still reach the top chip:
# on a 24GB/12x2GB card with ~23.56GiB free the context is ~0.5GiB, so 0.5 -> fill 23 (reaches the
# 22-24GB range = the top back-side chip). 0.7 would round to 22 and leave that chip cold (the bug).
_FILL_HEADROOM_GIB = 0.5


def mtps_to_clock_offset_mhz(mtps: int) -> int:
    """GDDR transfer-rate delta -> NVML clock-domain offset (half)."""
    return mtps // 2


def clock_offset_mhz_to_mtps(mhz: int) -> int:
    return mhz * 2


def fill_gib(free_bytes: Optional[int], headroom_gib: float = _FILL_HEADROOM_GIB) -> int:
    """Largest whole-GiB buffer that fits in ``free_bytes`` leaving ``headroom_gib`` for the CUDA
    context. Filling (near) all VRAM is REQUIRED so the BACK-side GDDR6X chips are exercised — a
    small buffer only hits the cool front side and a soak there is meaningless (the back chips,
    which heat-soak to ~100-110C, never get warm). Whole-GiB so the top chip's range is reached;
    headroom avoids an OOM that would abort the soak. Returns 0 if free is unknown/too small.
    """
    if not free_bytes or free_bytes <= 0:
        return 0
    usable = int(free_bytes) - int(headroom_gib * _GIB)
    return max(0, usable // _GIB)


# ---- NVML (GPU-bound, root for writes) -------------------------------------------------------

def _nvml():  # pragma: no cover - needs the driver
    lib = ctypes.CDLL(_LIB)
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit failed (driver present?)")
    return lib


def _handle(lib, uuid: str):  # pragma: no cover - needs a GPU
    h = ctypes.c_void_p()
    rc = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h))
    if rc != 0:
        raise RuntimeError(f"nvmlDeviceGetHandleByUUID({uuid}) rc={rc}")
    return h


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


def mem_info(uuid: str) -> Optional[_Mem]:  # pragma: no cover - needs a GPU
    lib = _nvml()
    mi = _Mem()
    if lib.nvmlDeviceGetMemoryInfo(_handle(lib, uuid), ctypes.byref(mi)) != 0:
        return None
    return mi


def mem_free_bytes(uuid: str) -> Optional[int]:  # pragma: no cover - needs a GPU
    mi = mem_info(uuid)
    return int(mi.free) if mi else None


def mem_used_mib(uuid: str) -> Optional[float]:  # pragma: no cover - needs a GPU
    mi = mem_info(uuid)
    return (int(mi.used) / (1024 * 1024)) if mi else None


def running_proc_count(uuid: str) -> int:  # pragma: no cover - needs a GPU
    """Best-effort count of OTHER compute processes on the card (for the in-use consent prompt)."""
    lib = _nvml()
    try:
        c = ctypes.c_uint(0)
        # NVML sets the required count even when passed a null buffer (INSUFFICIENT_SIZE/SUCCESS).
        lib.nvmlDeviceGetComputeRunningProcesses_v3(_handle(lib, uuid), ctypes.byref(c), None)
        return int(c.value)
    except Exception:
        return 0


def mem_offset_range_mhz(uuid: str) -> Tuple[Optional[int], Optional[int]]:  # pragma: no cover - GPU
    lib = _nvml()
    mn, mx = ctypes.c_int(0), ctypes.c_int(0)
    rc = lib.nvmlDeviceGetMemClkMinMaxVfOffset(_handle(lib, uuid), ctypes.byref(mn), ctypes.byref(mx))
    return (mn.value, mx.value) if rc == 0 else (None, None)


def get_mem_offset_mhz(uuid: str) -> Optional[int]:  # pragma: no cover - needs a GPU
    """Read the current mem-clock VF offset. NOTE: returns 0 on pre-R570 drivers even when set —
    treat a 0 readback as 'unverifiable', not 'failed'."""
    lib = _nvml()
    cur = ctypes.c_int(-1)
    rc = lib.nvmlDeviceGetMemClkVfOffset(_handle(lib, uuid), ctypes.byref(cur))
    return cur.value if rc == 0 else None


def set_mem_offset_mhz(uuid: str, mhz: int, *, enforce_range: bool = True
                       ) -> Tuple[int, Optional[int]]:  # pragma: no cover - GPU+root
    """Set the GDDR mem-clock VF offset (MHz; = MT/s // 2). Needs root. Readback-verifies.
    Returns (rc, readback_mhz); rc 0 = the SET succeeded. enforce_range=False is for reset (0),
    which must ALWAYS be allowed even if the device range is weird/unreadable."""
    mhz = int(mhz)
    if enforce_range and mhz != 0:
        mn, mx = mem_offset_range_mhz(uuid)
        if mn is None or mx is None:
            # FAIL CLOSED: the range getter is the same legacy family as the offset getter and can
            # return unsupported on R555-R570; without a bound a too-high write can hard-hang. Cap.
            if mhz > _HARD_CAP_MHZ:
                raise RuntimeError(f"VF offset range unreadable; refusing {mhz} MHz > hard cap "
                                   f"{_HARD_CAP_MHZ} (fail-closed)")
            print(f"[half_b/silicon] WARN range unreadable; allowing {mhz}<= cap {_HARD_CAP_MHZ}")
        elif not (mn <= mhz <= mx):
            raise ValueError(f"offset {mhz} MHz outside device range [{mn}, {mx}]")
    lib = _nvml()
    rc = lib.nvmlDeviceSetMemClkVfOffset(_handle(lib, uuid), ctypes.c_int(mhz))
    rb = get_mem_offset_mhz(uuid)
    # readback HARD-fail: where readback is meaningful (not None/0), rb != mhz means the apply did
    # NOT take -> force rc nonzero so collect()/assemble() treat the measurement as unsafe.
    if rc == 0 and rb not in (None, 0) and rb != mhz:
        print(f"[half_b/silicon] readback {rb} != set {mhz} -> apply FAILED")
        rc = 1
    if rc == 0:
        _applied.add(uuid) if mhz != 0 else _applied.discard(uuid)
    print(f"[half_b/silicon] gpu={uuid[:20]} set mem offset = {mhz} MHz "
          f"(= {clock_offset_mhz_to_mtps(mhz)} MT/s) rc={rc} readback={rb}")
    return rc, rb


def reset_mem_offset(uuid: str) -> int:  # pragma: no cover - needs a GPU + root
    rc, _ = set_mem_offset_mhz(uuid, 0, enforce_range=False)   # 0 must always be allowed
    return rc


def _gpu_uuid(g) -> Optional[str]:
    """The UUID lives in the driver_state DICT (matrix GpuCapabilities has no .uuid attr) — the
    write paths read it from there, so revert MUST too (else it no-ops on every card)."""
    ds = getattr(g, "driver_state", None)
    if isinstance(ds, dict):
        return ds.get("uuid")
    return getattr(ds, "uuid", None) if ds is not None else None


def revert_all(matrix, reset_fn=reset_mem_offset) -> int:
    """Reset mem offset to stock (0) on every GPU (idempotent, safe). reset_fn injectable for test."""
    rc = 0
    for g in (matrix.gpus or []):
        uuid = _gpu_uuid(g)
        if not uuid:                                  # no real UUID -> cannot GetHandleByUUID; skip
            continue
        try:
            rc |= reset_fn(str(uuid))
        except Exception as e:
            print(f"[half_b/silicon] revert {uuid}: {e}")
            rc |= 1
    return rc


def _revert_applied() -> None:  # pragma: no cover - atexit/interrupt safety net
    """Zero every offset we applied — last-line insurance so a crash/Ctrl-C never leaves a
    corrupting clock live for the rest of the host session."""
    for uuid in list(_applied):
        try:
            reset_mem_offset(uuid)
        except Exception:
            pass


atexit.register(_revert_applied)
