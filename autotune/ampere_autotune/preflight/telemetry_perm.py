"""Probe 2 — TELEMETRY-READ permission.

Three tiers of "can we read what we need to gate on":
  - NVML read   : clocks/power/core-temp — always, no privilege.
  - BAR0 junction: GDDR6X mem-junction via the gputemps MMIO tool — FOUR independent prereqs.
  - DCGM-PROF   : DCGM_FI_DEV_MEMORY_TEMP (field 140) — nonzero only on datacenter cards.
All checks are read-only / environment inspection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

from . import _nvml

# NVML read
READ_OK = "OK"
READ_NO_PERMISSION = "NO_PERMISSION"
READ_UNAVAILABLE = "UNAVAILABLE"

# BAR0 junction
JUNCTION_READABLE = "JUNCTION_READABLE"
MISSING_ROOT = "MISSING_ROOT"
MISSING_DEV_MEM = "MISSING_DEV_MEM"
MISSING_IOMEM_RELAXED = "MISSING_IOMEM_RELAXED"
SECUREBOOT_ON = "SECUREBOOT_ON"
NO_MMIO_TOOL = "NO_MMIO_TOOL"            # gputemps not built / SKU unsupported (e.g. TU102)

# DCGM
DCGM_PROF_AVAILABLE = "DCGM_PROF_AVAILABLE"
DCGM_BLANK = "DCGM_BLANK"


@dataclass
class TelemetryResult:
    nvml_read: str
    junction: str
    junction_fix: str
    dcgm: str

    def to_dict(self) -> dict:
        return asdict(self)


def _proc_cmdline() -> str:
    try:
        with open("/proc/cmdline", "r") as f:
            return f.read()
    except OSError:
        return ""


def _can_open_dev_mem() -> bool:
    try:
        fd = os.open("/dev/mem", os.O_RDONLY)
        os.close(fd)
        return True
    except OSError:
        return False


def _secureboot_on() -> bool:
    """Best-effort Secure Boot detection via the efivars GUID. Unknown -> assume OFF
    (we only use this to *explain* a junction-read failure, never to gate reads)."""
    import glob
    for path in glob.glob("/sys/firmware/efi/efivars/SecureBoot-*"):
        try:
            with open(path, "rb") as f:
                data = f.read()
            # 4-byte attr prefix then the value byte; 1 == enabled
            return len(data) >= 5 and data[4] == 1
        except OSError:
            continue
    return False


def probe_telemetry(sess: "_nvml.Session", idx: int) -> TelemetryResult:
    h = sess.handle(idx)

    # --- NVML read ---
    if h is None:
        nvml_read = READ_UNAVAILABLE
    else:
        c = _nvml.call("nvmlDeviceGetTemperature", h, 0)  # 0 == NVML_TEMPERATURE_GPU
        if c.ok:
            nvml_read = READ_OK
        elif c.kind == "NO_PERMISSION":
            nvml_read = READ_NO_PERMISSION
        else:
            nvml_read = READ_UNAVAILABLE

    # --- BAR0 junction (gputemps): four independent, ordered prereqs ---
    fix = ""
    if os.geteuid() != 0:
        junction, fix = MISSING_ROOT, "run as root (sudo)"
    elif not _can_open_dev_mem():
        junction, fix = MISSING_DEV_MEM, "host /dev/mem not accessible (privileged container or host)"
    elif "iomem=relaxed" not in _proc_cmdline():
        junction, fix = MISSING_IOMEM_RELAXED, "add kernel boot param iomem=relaxed and reboot"
    elif _secureboot_on():
        junction, fix = SECUREBOOT_ON, "disable Secure Boot (BAR0 MMIO blocked)"
    else:
        # Prereqs met; actual readability depends on the gputemps build + GA10x support.
        # The matrix layer will downgrade to NO_MMIO_TOOL if the binary is absent or the
        # idle core-temp sanity check vs NVML fails (e.g. TU102 / wrong offsets).
        junction, fix = JUNCTION_READABLE, ""

    # --- DCGM PROF (datacenter only; blank on GeForce). Probed lazily/optionally. ---
    dcgm = DCGM_BLANK  # default; a real DCGM probe is optional (datacenter), left to matrix

    return TelemetryResult(nvml_read, junction, fix, dcgm)
