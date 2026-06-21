"""Probe 3 — OC (clock-write) PERMISSION.

Proves whether we *could* set clock offsets WITHOUT moving a clock, and cleanly separates
the unprivileged-container case (NVML_ERROR_NO_PERMISSION) from locked datacenter HW
(NVML_ERROR_NOT_SUPPORTED).

Two depths:
  - default (read-only inference): euid/CAP_SYS_ADMIN/container/driver -> WRITE_LIKELY.
  - write_probe=True (used by `tune --hw`): a no-op set of the offset to its CURRENT value
    + read-back -> WRITE_OK. Still moves nothing (same value), but it is the only real
    write, so it is opt-in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Tuple

from . import _nvml

# privilege state
HOST_ROOT_PRIVILEGED = "HOST_ROOT_PRIVILEGED"
ROOT_NO_CAP = "ROOT_NO_CAP"
UNPRIVILEGED_CONTAINER = "UNPRIVILEGED_CONTAINER"
UNPRIVILEGED = "UNPRIVILEGED"

# write capability
WRITE_OK = "WRITE_OK"
WRITE_LIKELY = "WRITE_LIKELY"            # inferred, not write-probed
WRITE_NO_PERMISSION = "NO_PERMISSION"
WRITE_NOT_SUPPORTED = "NOT_SUPPORTED"
WRITE_READBACK_MISMATCH = "WRITE_READBACK_MISMATCH"
WRITE_NOT_PROBED = "NOT_PROBED"

_MIN_DRIVER: Tuple[int, int] = (555, 85)   # SetClockOffsets floor


@dataclass
class OcPermResult:
    euid_root: bool
    cap_sys_admin: bool
    in_container: bool
    privileged_container: bool
    priv: str
    driver: str
    driver_ok: bool
    write_cap: str
    fix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _cap_sys_admin() -> bool:
    """Parse CapEff from /proc/self/status; bit 21 == CAP_SYS_ADMIN."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    bits = int(line.split()[1], 16)
                    return bool(bits & (1 << 21))
    except (OSError, ValueError):
        pass
    return False


def _in_container() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            blob = f.read()
        return any(k in blob for k in ("docker", "kubepods", "containerd", "lxc"))
    except OSError:
        return False


def _driver_tuple(s: str) -> Tuple[int, int]:
    try:
        parts = s.split(".")
        return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (int(parts[0]), 0)
    except (ValueError, IndexError):
        return (0, 0)


def probe_oc_perm(sess: "_nvml.Session", idx: int, write_probe: bool = False) -> OcPermResult:
    euid_root = (os.geteuid() == 0)
    cap = _cap_sys_admin()
    in_ctr = _in_container()
    privd = _can_open_dev_mem()

    if in_ctr and not (euid_root and privd):
        priv = UNPRIVILEGED_CONTAINER
    elif euid_root and cap:
        priv = HOST_ROOT_PRIVILEGED
    elif euid_root:
        priv = ROOT_NO_CAP
    else:
        priv = UNPRIVILEGED

    drv_c = _nvml.call("nvmlSystemGetDriverVersion")
    driver = drv_c.value.decode() if (drv_c.ok and isinstance(drv_c.value, bytes)) else (drv_c.value if drv_c.ok else "?")
    driver = str(driver)
    driver_ok = _driver_tuple(driver) >= _MIN_DRIVER

    write_cap = WRITE_NOT_PROBED
    fix = ""
    if priv == UNPRIVILEGED_CONTAINER:
        write_cap = WRITE_NO_PERMISSION
        fix = "clocks are GLOBAL host state; run on the HOST (systemd unit Before= the vLLM container), not in the image"
    elif not driver_ok:
        fix = f"driver {driver} < R555.85 — SetClockOffsets unavailable; upgrade driver (or use legacy VF-offset)"
    elif priv in (HOST_ROOT_PRIVILEGED, ROOT_NO_CAP):
        if not write_probe:
            write_cap = WRITE_LIKELY  # inferred; the real no-op write happens in `tune --hw`
        else:
            write_cap = _write_probe(sess, idx)
    else:
        fix = "need root (sudo) to set GPU clocks"

    return OcPermResult(euid_root, cap, in_ctr, privd, priv, driver, driver_ok, write_cap, fix)


def _write_probe(sess: "_nvml.Session", idx: int) -> str:
    """No-op set of the MEM offset to its current value, then read back. Moves nothing."""
    h = sess.handle(idx)
    if h is None:
        return WRITE_NOT_PROBED
    cur = _nvml.call("nvmlDeviceGetClockOffsets", h)
    if not cur.ok:
        return WRITE_NOT_SUPPORTED if cur.kind in ("NOT_SUPPORTED", "MISSING_SYMBOL") else WRITE_NOT_PROBED
    # NOTE: exact SetClockOffsets signature is driver/struct-specific; the real silicon.py
    # owns the struct. Here we only classify the *permission*, so we attempt and classify.
    setc = _nvml.call("nvmlDeviceSetClockOffsets", h, cur.value)
    if setc.ok:
        rb = _nvml.call("nvmlDeviceGetClockOffsets", h)
        return WRITE_OK if (rb.ok and rb.value == cur.value) else WRITE_READBACK_MISMATCH
    if setc.kind == "NO_PERMISSION":
        return WRITE_NO_PERMISSION
    if setc.kind in ("NOT_SUPPORTED", "MISSING_SYMBOL"):
        return WRITE_NOT_SUPPORTED
    return WRITE_NOT_PROBED


def _can_open_dev_mem() -> bool:
    try:
        fd = os.open("/dev/mem", os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False
