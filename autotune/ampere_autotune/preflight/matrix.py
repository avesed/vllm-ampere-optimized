"""Aggregate the four probes into a CapabilityMatrix: a human table, a --json dump, and a
DECISION that unlocks tiers/modes. Every refusal carries the missing capability + the fix."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List

from . import _nvml
from . import sku as _sku
from . import telemetry_perm as _tel
from . import oc_perm as _oc
from . import driver_state as _ds


@dataclass
class GpuCapabilities:
    index: int
    sku: dict
    telemetry: dict
    oc: dict
    driver_state: dict
    half_b_unlocked: bool
    max_tier: str               # highest tier the host may attempt for this GPU
    gate_family: str
    refusals: List[str] = field(default_factory=list)
    decision: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CapabilityMatrix:
    half_a_available: bool
    gpus: List[GpuCapabilities]
    note: str = ""

    def to_dict(self) -> dict:
        return {"half_a_available": self.half_a_available, "note": self.note,
                "gpus": [g.to_dict() for g in self.gpus]}


def _decide(s: _sku.SkuResult, t: _tel.TelemetryResult, o: _oc.OcPermResult,
            d: _ds.DriverState):
    refusals: List[str] = []

    # HALF-B gate: privileged host root + a real/likely write cap + new-enough driver.
    write_ok = o.write_cap in (_oc.WRITE_OK, _oc.WRITE_LIKELY)
    half_b = (o.priv == _oc.HOST_ROOT_PRIVILEGED) and write_ok and o.driver_ok

    if o.priv == _oc.UNPRIVILEGED_CONTAINER:
        refusals.append("HALF-B REFUSED: unprivileged container — " + o.fix)
    elif not o.driver_ok and o.priv != _oc.UNPRIVILEGED_CONTAINER:
        refusals.append("HALF-B REFUSED: " + (o.fix or "driver too old for SetClockOffsets"))
    elif o.priv == _oc.UNPRIVILEGED:
        refusals.append("HALF-B REFUSED: " + (o.fix or "need root"))

    # Tier ceiling from the SKU, then clamp by what we can safely gate.
    max_tier = s.tier_ceiling if half_b else "T1"
    if s.offset_support == _sku.OFFSET_NOT_SUPPORTED:
        max_tier = "T1"  # datacenter / locked HW

    # T3 mem-OC needs a thermal abort sensor. On GDDR6X (golden-EDR-junction gate) the
    # junction reader is mandatory; if unreadable, cap below T3.
    if max_tier.startswith("T3") and s.gate_family == _sku.GATE_GOLDEN_EDR_JUNCTION:
        if t.junction != _tel.JUNCTION_READABLE:
            refusals.append(
                f"T3 mem-OC REFUSED: GDDR6X junction unreadable ({t.junction}) — "
                + (t.junction_fix or "no headless silent-overheat guard"))
            max_tier = "T2"

    # ECC refines the gate family: ECC_ON -> SBE/DBE counters are ground truth;
    # ECC_OFF on GDDR6X -> golden-token gate is MANDATORY.
    ecc_note = ""
    if d.ecc_current == _ds.ECC_ON:
        ecc_note = "ECC on (SBE/DBE counters)"
    elif d.ecc_current == _ds.ECC_OFF and s.gate_family == _sku.GATE_GOLDEN_EDR_JUNCTION:
        ecc_note = "no ECC -> golden-token gate MANDATORY"
    if half_b and d.persistence == _ds.PERSIST_DISABLED:
        refusals.append("note: persistence-mode DISABLED -> offsets wiped on reboot; "
                        "re-apply via a systemd unit (nvidia-persistenced recommended)")

    if half_b:
        decision = (f"T0/T1{'/T2' if max_tier in ('T2','T3','T3-EXPERIMENTAL') else ''}"
                    f"{'/T3' if max_tier in ('T3','T3-EXPERIMENTAL') else ''} unlocked"
                    f" — monitor-only default; characterize allowed after --dry-run"
                    f" (gate: {s.gate_family}{'; ' + ecc_note if ecc_note else ''})")
    elif o.priv in (_oc.UNPRIVILEGED, _oc.UNPRIVILEGED_CONTAINER):
        decision = "HALF-A only (no privilege for silicon tuning)"
    else:
        decision = f"T0/T1 only ({'offsets NOT_SUPPORTED by HW' if s.offset_support==_sku.OFFSET_NOT_SUPPORTED else 'limited'})"

    return half_b, max_tier, refusals, decision


def collect(write_probe: bool = False) -> CapabilityMatrix:
    with _nvml.Session() as sess:
        if not sess.ok:
            return CapabilityMatrix(
                half_a_available=True,  # HALF-A can still target a remote /metrics endpoint
                gpus=[],
                note=f"NVML unavailable ({sess.reason}). HALF-A may still run against a remote endpoint; HALF-B needs a local GPU.")
        n = sess.device_count()
        gpus: List[GpuCapabilities] = []
        any_read = False
        for i in range(n):
            s = _sku.probe_sku(sess, i)
            t = _tel.probe_telemetry(sess, i)
            o = _oc.probe_oc_perm(sess, i, write_probe=write_probe)
            d = _ds.probe_driver_state(sess, i)
            any_read = any_read or (t.nvml_read == _tel.READ_OK)
            half_b, max_tier, refusals, decision = _decide(s, t, o, d)
            gpus.append(GpuCapabilities(
                index=i, sku=s.to_dict(), telemetry=t.to_dict(), oc=o.to_dict(),
                driver_state=d.to_dict(), half_b_unlocked=half_b, max_tier=max_tier,
                gate_family=s.gate_family, refusals=refusals, decision=decision))
        return CapabilityMatrix(half_a_available=any_read or n > 0, gpus=gpus)


def render(matrix: CapabilityMatrix) -> str:
    lines: List[str] = []
    if not matrix.gpus:
        lines.append(matrix.note or "no GPUs detected")
        lines.append(f"HALF-A available: {matrix.half_a_available}")
        return "\n".join(lines)
    for g in matrix.gpus:
        s, t, o, d = g.sku, g.telemetry, g.oc, g.driver_state
        lines.append(f"GPU {g.index}  {s.get('name')}  cc{s.get('cc')}  "
                     f"{s.get('mem_type')}  {s.get('sku_class')}  [{(d.get('uuid') or '')[:16]}]")
        lines.append(f"  tier-ceiling : {s.get('tier_ceiling'):<16} gate-family  : {g.gate_family}")
        lines.append(f"  NVML-read    : {t.get('nvml_read'):<16} junction-read: {t.get('junction')}")
        lines.append(f"  OC-priv      : {o.get('priv'):<16} OC-write     : {o.get('write_cap')}")
        lines.append(f"  driver       : {o.get('driver')} ({'OK' if o.get('driver_ok') else 'TOO_OLD'})"
                     f"   persistence: {d.get('persistence')}   ECC: {d.get('ecc_current')}")
        lines.append(f"  DECISION     : {g.decision}")
        for r in g.refusals:
            lines.append(f"    ! {r}")
    return "\n".join(lines)
