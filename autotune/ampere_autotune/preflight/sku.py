"""Probe 1 — GPU-OC-SUPPORT / SKU tier ceiling + gate family.

All calls are READ-ONLY. The offset-support probe uses ``nvmlDeviceGetClockOffsets``
(a *getter*) — it never writes. Datacenter cards return NOT_SUPPORTED here, which is how we
detect the locked SKUs without touching anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

from . import _nvml

# --- outcome vocab (plain strings; stable across versions) ---
OFFSET_SUPPORTED = "OFFSET_SUPPORTED"
OFFSET_NOT_SUPPORTED = "OFFSET_NOT_SUPPORTED"
OFFSET_UNKNOWN = "OFFSET_UNKNOWN"

MEM_GDDR6X = "GDDR6X"
MEM_GDDR6 = "GDDR6"
MEM_HBM = "HBM2e"
MEM_UNKNOWN = "UNKNOWN"

SKU_GEFORCE = "GEFORCE"
SKU_WORKSTATION = "WORKSTATION"
SKU_DATACENTER = "DATACENTER"
SKU_UNKNOWN = "UNKNOWN"

# gate families (see DESIGN.md / RESEARCH §4)
GATE_GOLDEN_EDR_JUNCTION = "GOLDEN_EDR_JUNCTION"  # GDDR6X no-ECC: golden + EDR knee + junction
GATE_ECC_GOLDEN = "ECC_GOLDEN"                    # GDDR6 workstation: ECC counters + golden, NO EDR knee
GATE_ECC = "ECC"                                  # HBM / datacenter locked: ECC counters only

# Heuristic name tables. Matched with WORD BOUNDARIES (re \b) so "A40" does NOT match
# "A4000" and "A10" does NOT match "A100" (the naive-substring bug a test caught).
_GDDR6X = ("3090", "3080")                        # RTX 3090/3090Ti/3080/3080Ti/3080-12G
_WORKSTATION = ("A6000", "A5000", "A4500", "A4000", "A2000")  # RTX A-series workstation
_DATACENTER = ("A100", "A40", "A30", "A16", "A10", "A2", "H100", "H200")


def _tok(name: str, token: str) -> bool:
    """Whole-token match (word boundaries) so model numbers don't cross-match."""
    return re.search(r"\b" + re.escape(token) + r"\b", name) is not None


@dataclass
class SkuResult:
    index: int
    name: str
    brand: str
    cc: Optional[str]            # "8.6"
    mem_type: str
    sku_class: str
    offset_support: str
    tier_ceiling: str            # T0 | T1 | T2 | T3 | T3-EXPERIMENTAL
    gate_family: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_mem(name: str, sku_class: str) -> str:
    n = name.upper()
    if sku_class == SKU_DATACENTER and (_tok(n, "A100") or _tok(n, "H100") or _tok(n, "H200")):
        return MEM_HBM
    # GeForce GDDR6X parts (workstation A-series is GDDR6, never GDDR6X)
    if sku_class != SKU_WORKSTATION and any(_tok(n, k) for k in _GDDR6X):
        return MEM_GDDR6X
    # workstation + remaining datacenter Ampere are GDDR6 (A40/A10/A2/A6000/A5000/A4000)
    if sku_class in (SKU_WORKSTATION, SKU_DATACENTER):
        return MEM_GDDR6
    if re.search(r"\bRTX 30\d\d", n) or re.search(r"\b30\d\d\b", n):  # other 30-series -> GDDR6
        return MEM_GDDR6
    return MEM_UNKNOWN


def _classify_sku(name: str) -> str:
    n = name.upper()
    # Workstation BEFORE datacenter: A4000/A2000 must not fall to the A40/A2 datacenter tokens.
    if any(_tok(n, k) for k in _WORKSTATION):
        return SKU_WORKSTATION
    if any(_tok(n, k) for k in _DATACENTER):
        return SKU_DATACENTER
    if "GEFORCE" in n or re.search(r"\bRTX [345]0", n):
        return SKU_GEFORCE
    return SKU_UNKNOWN


def _ceiling_and_gate(mem_type: str, sku_class: str, offset_support: str, name: str):
    n = name.upper()
    if mem_type == MEM_HBM or sku_class == SKU_DATACENTER:
        return "T1", GATE_ECC
    if offset_support == OFFSET_NOT_SUPPORTED:
        return "T1", (GATE_GOLDEN_EDR_JUNCTION if mem_type == MEM_GDDR6X else GATE_ECC)
    if mem_type == MEM_GDDR6X:
        return "T3", GATE_GOLDEN_EDR_JUNCTION
    if mem_type == MEM_GDDR6 and sku_class == SKU_WORKSTATION:
        ceiling = "T3-EXPERIMENTAL" if "A4000" in n else "T3"
        return ceiling, GATE_ECC_GOLDEN  # GDDR6: NO EDR knee -> ECC/golden gate
    return "T1", GATE_ECC


def probe_sku(sess: "_nvml.Session", idx: int) -> SkuResult:
    h = sess.handle(idx)
    name = _nvml.call("nvmlDeviceGetName", h).value if h is not None else ""
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")
    name = name or "(unknown)"

    brand_c = _nvml.call("nvmlDeviceGetBrand", h) if h is not None else _nvml.Call(False)
    brand = str(brand_c.value) if brand_c.ok else "?"

    cc_c = _nvml.call("nvmlDeviceGetCudaComputeCapability", h) if h is not None else _nvml.Call(False)
    cc = f"{cc_c.value[0]}.{cc_c.value[1]}" if cc_c.ok and cc_c.value else None

    sku_class = _classify_sku(name)
    mem_type = _classify_mem(name, sku_class)

    # SAFE read-only offset-support probe (getter; never writes).
    off = OFFSET_UNKNOWN
    note = ""
    if h is not None:
        c = _nvml.call("nvmlDeviceGetClockOffsets", h)  # newer drivers only
        if c.ok:
            off = OFFSET_SUPPORTED
        elif c.kind == "NOT_SUPPORTED":
            off = OFFSET_NOT_SUPPORTED
        elif c.kind == "MISSING_SYMBOL":
            off = OFFSET_UNKNOWN
            note = "GetClockOffsets missing — driver < R555.85; fall back to legacy VF-offset or treat as unsupported"
    else:
        note = "no NVML handle (no GPU / no driver)"

    ceiling, gate = _ceiling_and_gate(mem_type, sku_class, off, name)
    return SkuResult(idx, name, brand, cc, mem_type, sku_class, off, ceiling, gate, note)
