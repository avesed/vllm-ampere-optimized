"""Capability + permission preflight — runs first on every invocation.

Public API:
    collect(write_probe=False) -> CapabilityMatrix
    render(matrix) -> str
The four probes (sku, telemetry_perm, oc_perm, driver_state) are read-only or no-op; the
matrix aggregates them into a DECISION that unlocks tiers/modes. Importing this package
pulls NO privileged write code (HALF-A safe).
"""
# Populate submodules on the package namespace BEFORE matrix imports them as siblings
# (keeps `from . import sku` resolvable and avoids an init-time cycle).
from . import _nvml, sku, telemetry_perm, oc_perm, driver_state  # noqa: F401
from .matrix import CapabilityMatrix, GpuCapabilities, collect, render

__all__ = [
    "CapabilityMatrix", "GpuCapabilities", "collect", "render",
    "_nvml", "sku", "telemetry_perm", "oc_perm", "driver_state",
]
