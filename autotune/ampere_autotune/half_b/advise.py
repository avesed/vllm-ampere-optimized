"""HALF-B no-root ADVISORY — recommend-only silicon tuning from no-root signals.

When there is no OC-write privilege, the tool still MEASURES (NVML reads + bw_verify kernel +
a stock golden token-id check + vLLM /metrics — none need root) and emits tuning
RECOMMENDATIONS, instead of refusing. PURE + no-GPU: callers inject the measurements, so the
whole recommendation logic is unit-testable with no GPU (the collector is the GPU/endpoint
part, stubbed).

SAFETY (load-bearing, from the adversarial review):
  - NEVER print an apply-ready offset magnitude (no MHz / Gbps). Headroom is a dimensionless
    decode-tok/s PROJECTION only; "the safe offset is found by the gate, not stated".
  - Every benefit-bearing message co-locates the ungated-OC-corrupts warning.
  - Projection floor is ZERO ("up to +X% — or nothing"), labelled UPPER-BOUND / UNMEASURED.
  - A stock correctness FAIL suppresses the whole silicon section.
  - Thermal/throttle suppresses the encouraging projection.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from ..preflight import sku as _sku

# severities
INFO = "INFO"
WARN = "WARN"
CRITICAL = "CRITICAL"

UNGATED_WARNING = (
    "an ungated GDDR6X offset silently corrupts output (mismatch can read 0) — the ONLY safe "
    "path is root + the gated sweep; do NOT type an offset into nvidia-settings / nvidia_oc / Coolbits"
)
# any benefit/% message must carry this; tests assert it. No bare offset magnitude may appear.
_BARE_OFFSET = re.compile(r"[+-]?\d+\s*(MHz|Gbps|MT/s)", re.IGNORECASE)


@dataclass
class Measurements:
    achieved_gbs: Optional[float] = None       # DECODE achieved bandwidth (needs /metrics; how bw-bound decode is)
    peak_gbs: Optional[float] = None           # bw_verify saturating PEAK GB/s (real achievable roofline, no root)
    decode_toks: Optional[float] = None        # 1000/TPOT from vLLM /metrics
    prefill_toks: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    core_temp_c: Optional[float] = None
    throttle_reasons: List[str] = field(default_factory=list)
    golden_ok: Optional[bool] = None   # None = not checked (e.g. telemetry-only run, no vLLM endpoint)
    mismatch_count: int = 0
    ecc_current: str = "ECC_UNKNOWN"
    bw_flat_across_batch: bool = True           # True => genuinely bandwidth-bound (corroboration)


@dataclass
class SkuInfo:
    sku_class: str        # _sku.SKU_GEFORCE | SKU_WORKSTATION | SKU_DATACENTER
    mem_type: str         # _sku.MEM_GDDR6X | MEM_GDDR6 | MEM_HBM
    offset_support: str   # _sku.OFFSET_SUPPORTED | OFFSET_NOT_SUPPORTED | OFFSET_UNKNOWN


@dataclass
class Roofline:
    sku_peak_gbs: float
    nominal_bw_headroom_pct: float = 8.0   # per-SKU nominal mem headroom (dimensionless; NEVER printed as MHz)
    subprop_hi: float = 0.8                # decode realizes only ~0.6-0.8 of the BW gain


@dataclass
class Recommendation:
    name: str
    severity: str
    message: str
    action: str
    root_needed: bool
    trigger_fired: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_THERMAL_REASONS = ("thermal", "powerbrake", "hwslowdown", "sw_thermal", "hwthermal", "swthermal")


def _is_thermally_limited(m: Measurements) -> bool:
    if any(any(t in r.lower() for t in _THERMAL_REASONS) for r in m.throttle_reasons):
        return True
    if m.core_temp_c is not None and m.core_temp_c >= 80.0:
        return True
    return False


def _actionable_rec() -> Recommendation:
    return Recommendation(
        "A-ACTIONABLE-get-root-characterize", INFO,
        "To actually apply any of the root-needed items above, the only safe path is the gated sweep — "
        "it finds the safe clock by measurement; you never type an offset.",
        "sudo ampere-autotune tune --hw --dry-run   (proves write-perm, moves nothing)\n"
        "sudo ampere-autotune tune --hw --mode characterize   (adaptive gated sweep)\n"
        "sudo ampere-autotune monitor --hw   (revert-only watchdog)",
        root_needed=True, trigger_fired="closing")


def _sku_guidance(sku: SkuInfo) -> List[Recommendation]:
    """SKU-specific guidance that needs NO live measurement (used in static mode)."""
    if sku.mem_type == _sku.MEM_HBM or sku.sku_class == _sku.SKU_DATACENTER:
        return [Recommendation(
            "A-SKU-datacenter-locked", INFO,
            "Datacenter SKU: memory clock is locked — no mem-OC. Only Tier-0 power-limit and Tier-1 "
            "locked-clocks (anti-throttle); ECC SBE/DBE counters are the health source of truth.",
            "With root: cap/pin within the factory envelope (Tier-0/1).", True, "datacenter")]
    if sku.sku_class == _sku.SKU_WORKSTATION:
        return [Recommendation(
            "A-SKU-workstation-ecc", INFO,
            "Workstation SKU (GDDR6 + toggleable ECC): ECC REDUCES but does NOT eliminate the "
            "silent-corruption risk — ungated manual OC is still unsafe. " + UNGATED_WARNING,
            "With root: enable ECC (needs reboot) + run the gated sweep.", True, "workstation")]
    if sku.sku_class == _sku.SKU_GEFORCE and sku.mem_type == _sku.MEM_GDDR6X:
        return [Recommendation(
            "A-SKU-geforce-gated-sweep", WARN,
            "GeForce GDDR6X is the only family where mem-OC is real, and the riskiest: NO ECC. " + UNGATED_WARNING,
            "With root: run the gated characterize.", True, "geforce_gddr6x")]
    return []


def advise(m: Optional[Measurements], sku: SkuInfo, roof: Roofline) -> List[Recommendation]:
    """Pure: measurements -> ordered recommendations. No NVML/CUDA/HTTP here.

    ``m=None`` = live measurement unavailable (no CUDA GPU / no endpoint): emit static SKU
    guidance only (never a fabricated correctness/headroom claim)."""
    if m is None:
        recs: List[Recommendation] = [Recommendation(
            "A-MEASUREMENT-unavailable", WARN,
            "Live measurement unavailable (needs a CUDA GPU for bw_verify + a reachable vLLM endpoint "
            "for the stock golden check). Static SKU guidance only — no headroom estimate.",
            "Run on the serving host with the model loaded to get the bandwidth/correctness analysis.",
            root_needed=False, trigger_fired="no_measurement")]
        recs.extend(_sku_guidance(sku))
        recs.append(_actionable_rec())
        return recs

    recs = []

    # --- A-CORRECTNESS: stock baseline. A FAIL suppresses the entire silicon section. ---
    if m.golden_ok is False or m.mismatch_count > 0:
        recs.append(Recommendation(
            "A-CORRECTNESS-stock-FAIL", CRITICAL,
            "Output is NOT coherent at STOCK clocks (golden mismatch / bw_verify mismatch>0). "
            "This is not an overclock problem — do not tune silicon.",
            "Verify checkpoint sha256 vs source, then RAM/VRAM health (memtest). Re-run after clean.",
            root_needed=False, trigger_fired="golden_fail_or_mismatch"))
        return recs
    if m.golden_ok is True:
        recs.append(Recommendation(
            "A-CORRECTNESS-stock-baseline", INFO,
            "Coherent at stock (exact golden token-id match, mismatch_count==0) — this is the reference "
            "the gated sweep would hold every step against.",
            "None.", root_needed=False, trigger_fired="stock_clean"))
    else:  # None = not checked (telemetry-only run; no vLLM golden available)
        recs.append(Recommendation(
            "A-CORRECTNESS-not-checked", WARN,
            "Stock correctness NOT verified (no vLLM golden run available). The thermal/power notes "
            "below stand, but any mem-OC must FIRST establish a clean stock golden — the gated sweep "
            "does that; do not project a headroom gain until it is confirmed.",
            "Run on the serving host with the model loaded for the golden + bandwidth analysis.",
            root_needed=False, trigger_fired="golden_unchecked"))

    # --- A-BW-PEAK: real measured peak bandwidth + VRAM integrity (bw_verify, no root). ---
    if m.peak_gbs is not None and roof.sku_peak_gbs > 0:
        pct = m.peak_gbs / roof.sku_peak_gbs
        recs.append(Recommendation(
            "A-BW-PEAK-measured", INFO,
            f"Measured peak bandwidth {m.peak_gbs:.0f} GB/s = {pct:.0%} of the {roof.sku_peak_gbs:.0f} GB/s "
            f"spec roofline; VRAM integrity clean (mismatch_count==0). Decode is weight-bandwidth-bound on "
            f"Ampere, so raising the mem clock raises this ceiling — a gated mem-OC is worth characterizing "
            f"(the sweep also establishes the golden baseline). The realized decode gain is sub-proportional "
            f"and unmeasured; it is confirmed only by the gated sweep, never by this peak number.",
            "Get root and run the gated characterize (below) to turn this into a measured decode gain.",
            root_needed=True, trigger_fired="bw_peak_measured"))

    bw_fraction = None
    if m.achieved_gbs is not None and roof.sku_peak_gbs > 0:
        bw_fraction = m.achieved_gbs / roof.sku_peak_gbs

    thermally_limited = _is_thermally_limited(m)

    # --- A-THERMAL: if already throttling, suppress any encouraging projection. ---
    if thermally_limited:
        recs.append(Recommendation(
            "A-THERMAL-throttle-active", WARN,
            "The GPU is thermally constrained right now (throttle reason or high core temp). "
            "Overclocking will not help while throttling; the mem-OC projection is suppressed. "
            "Note: readable core temp does NOT see GDDR6X memory-junction heat-soak.",
            "Improve cooling (airflow / thermal pads) or apply a power cap (Tier-0), then re-measure.",
            root_needed=True, trigger_fired="thermal"))

    # --- SKU-specific guidance + (GeForce GDDR6X only) the projection ---
    if sku.mem_type == _sku.MEM_HBM or sku.sku_class == _sku.SKU_DATACENTER:
        recs.append(Recommendation(
            "A-SKU-datacenter-locked", INFO,
            "Datacenter SKU: memory clock is locked — no mem-OC. Only Tier-0 power-limit and "
            "Tier-1 locked-clocks (anti-throttle) are available; ECC SBE/DBE counters are the "
            "health source of truth.",
            "With root: cap/pin within the factory envelope (Tier-0/1). No silicon offset.",
            root_needed=True, trigger_fired="datacenter"))
    elif sku.sku_class == _sku.SKU_WORKSTATION:
        recs.append(Recommendation(
            "A-SKU-workstation-ecc", INFO,
            "Workstation SKU (GDDR6 + toggleable ECC): ECC REDUCES but does NOT eliminate the "
            "silent-corruption risk — ungated manual OC is still unsafe. Rising SBE counts during "
            "a sweep mean back off. " + UNGATED_WARNING,
            "With root: enable ECC (nvidia-smi -e 1, needs reboot) + run the gated sweep.",
            root_needed=True, trigger_fired="workstation"))
    elif sku.sku_class == _sku.SKU_GEFORCE and sku.mem_type == _sku.MEM_GDDR6X:
        # The only family where mem-OC is real — and the riskiest (no ECC).
        # Only PROJECT headroom when stock correctness is confirmed clean (golden is True).
        if not thermally_limited and m.golden_ok is True and bw_fraction is not None \
                and 0.5 <= bw_fraction < 0.85 and sku.offset_support == _sku.OFFSET_SUPPORTED:
            if m.bw_flat_across_batch:
                proj_hi = roof.nominal_bw_headroom_pct * roof.subprop_hi
                toks = f"~{m.decode_toks * (1 + proj_hi / 100):.0f} tok/s" if m.decode_toks else "more tok/s"
                recs.append(Recommendation(
                    "A-HEADROOM-mem-oc-projection", INFO,
                    f"PROJECTION (UPPER-BOUND, UNMEASURED on any 3090): decode is bandwidth-bound at "
                    f"{bw_fraction:.0%} of peak, so a gated mem-OC could yield UP TO +{proj_hi:.0f}% "
                    f"decode tok/s ({toks}) — OR NOTHING. Unknowable without actually OCing: the stable "
                    f"clock ceiling, the EDR-knee position (silicon lottery), and thermal-steady behavior. "
                    + UNGATED_WARNING,
                    "Worth characterizing: get root and run the gated sweep (next).",
                    root_needed=True, trigger_fired="headroom_bandwidth_bound"))
            else:
                recs.append(Recommendation(
                    "A-HEADROOM-not-bandwidth-bound", INFO,
                    "Bandwidth scales with batch here, so decode may be launch/occupancy-bound rather "
                    "than purely bandwidth-bound — mem-OC upside is uncertain; prefer the vLLM-flag "
                    "recommendations first.",
                    "Apply the vLLM flag recommendations; reconsider mem-OC only if still bandwidth-bound.",
                    root_needed=False, trigger_fired="not_bw_bound"))
        elif bw_fraction is not None and bw_fraction >= 0.85:
            recs.append(Recommendation(
                "A-NEAR-KNEE-low-headroom", INFO,
                f"Decode already sits at {bw_fraction:.0%} of the bandwidth roofline — little room "
                "before the EDR knee; mem-OC headroom is small. Prefer the vLLM-flag recommendations.",
                "Apply vLLM flags; mem-OC likely not worth the gated sweep.",
                root_needed=False, trigger_fired="near_knee"))
        recs.append(Recommendation(
            "A-SKU-geforce-gated-sweep", WARN,
            "GeForce GDDR6X is the only family where mem-OC is real, and the riskiest: NO ECC. The EDR "
            "knee silently rolls effective bandwidth over before any crash; the gate must use measured "
            "tok/s + golden, never an applied clock. " + UNGATED_WARNING,
            "With root: run the gated characterize (next).",
            root_needed=True, trigger_fired="geforce_gddr6x"))

    # --- A-POWER: perf/watt note (root to act; Tier-0, no GDDR6X clock, zero correctness risk) ---
    if (m.power_w is not None and m.power_limit_w and m.power_limit_w > 0
            and (m.power_w / m.power_limit_w) < 0.85):
        recs.append(Recommendation(
            "A-POWER-perf-per-watt", INFO,
            "Decode is bandwidth-bound, so you can likely cap power for near-identical tok/s with less "
            "heat (improves 24/7 stability).",
            "With root: lower the power limit (Tier-0) and re-measure decode tok/s.",
            root_needed=True, trigger_fired="power_headroom"))

    # --- A-ACTIONABLE: the ONLY realization path. Never a bare offset. ---
    recs.append(_actionable_rec())
    return recs


def render(recs: List[Recommendation]) -> str:
    """Human render. Enforces the safety invariant: no bare offset magnitude anywhere, and every
    benefit/percent line co-locates the ungated warning."""
    lines = ["ampere-autotune — HALF-B silicon advisory (no OC-write privilege: recommend-only)\n"]
    for r in recs:
        assert not _BARE_OFFSET.search(r.message), f"advisory leaked an apply-ready offset: {r.name}"
        # a projected POSITIVE gain ("+N%") must co-locate the ungated-OC warning; a bare
        # diagnostic percent (e.g. "94% of roofline") need not.
        if re.search(r"\+\s*\d+\s*%", r.message):
            assert "do NOT" in r.message or "ungated" in r.message, \
                f"benefit rec {r.name} missing the ungated-OC warning"
        tag = "" if not r.root_needed else "  [needs root]"
        lines.append(f"[{r.severity}] {r.name}{tag}\n  {r.message}\n  -> {r.action}\n")
    return "\n".join(lines)


# ----- CLI glue (the live measurement is GPU/endpoint-bound; the pure advise() above is tested) -----

_THROTTLE_BITS = (
    ("HwThermalSlowdown", "nvmlClocksThrottleReasonHwThermalSlowdown"),
    ("SwThermalSlowdown", "nvmlClocksThrottleReasonSwThermalSlowdown"),
    ("HwPowerBrakeSlowdown", "nvmlClocksThrottleReasonHwPowerBrakeSlowdown"),
    ("SwPowerCap", "nvmlClocksThrottleReasonSwPowerCap"),
)


_BW_VERIFY_BIN = Path(__file__).resolve().parents[2] / "instruments" / "bw_verify" / "bw_verify"


def _run_bw_verify(scope: str, size_gb: float = 2.0, iters: int = 5) -> Tuple[Optional[float], int]:  # pragma: no cover - needs a GPU
    """Run the compiled bw_verify kernel scoped to one GPU (CUDA_VISIBLE_DEVICES=scope, a UUID or
    index). Returns (peak_read_GB_s, mismatch_count). (None, 0) if the binary is absent or fails."""
    if not _BW_VERIFY_BIN.exists():
        return None, 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(scope))
    try:
        out = subprocess.run([str(_BW_VERIFY_BIN), str(size_gb), str(iters)],
                             capture_output=True, text=True, timeout=120, env=env)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        if "error" in data:
            return None, 0
        return float(data["read_GB_s"]), int(data.get("mismatch_count", 0))
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError, OSError):
        return None, 0


def collect_measurements(index: int, uuid: Optional[str] = None) -> Optional[Measurements]:  # pragma: no cover - needs a GPU
    """Gather the NO-ROOT advisory signals for GPU `index`.

    No root, no vLLM: NVML telemetry (power, power-limit, core temp, throttle) + the bw_verify
    kernel (peak GB/s + mismatch_count, scoped by UUID). NOT collected here (need a running vLLM
    under VLLM_BATCH_INVARIANT=1 -> left None): achieved_gbs (DECODE bw), decode_toks, golden_ok.
    advise() handles None honestly. Returns None if NVML is unavailable.
    """
    from ..preflight import _nvml
    nv = _nvml.nvml()
    with _nvml.Session() as sess:
        if not sess.ok:
            return None
        h = sess.handle(index)
        if h is None:
            return None

        def _v(fn, *a):
            c = _nvml.call(fn, h, *a)
            return c.value if c.ok else None

        power = _v("nvmlDeviceGetPowerUsage")                       # milliwatts
        limit = _v("nvmlDeviceGetEnforcedPowerLimit")
        if limit is None:
            limit = _v("nvmlDeviceGetPowerManagementLimit")
        temp = _v("nvmlDeviceGetTemperature", 0)                    # 0 == NVML_TEMPERATURE_GPU
        reasons: List[str] = []
        mask = _v("nvmlDeviceGetCurrentClocksThrottleReasons")
        if mask is not None and nv is not None:
            for label, const in _THROTTLE_BITS:
                bit = getattr(nv, const, 0)
                if bit and (int(mask) & int(bit)):
                    reasons.append(label)
        peak, mm = _run_bw_verify(uuid if uuid else str(index))
        return Measurements(
            achieved_gbs=None, peak_gbs=peak, decode_toks=None,
            power_w=(power / 1000.0 if power is not None else None),
            power_limit_w=(limit / 1000.0 if limit is not None else None),
            core_temp_c=(float(temp) if temp is not None else None),
            throttle_reasons=reasons,
            golden_ok=None, mismatch_count=mm)


def _peak_gbs(name: Optional[str], mem_type: str) -> float:
    n = (name or "").upper()
    if "3090" in n:
        return 936.0
    if "3080" in n:
        return 760.0
    if mem_type == _sku.MEM_GDDR6X:
        return 900.0
    if mem_type == _sku.MEM_GDDR6:
        return 768.0
    return 936.0


def run_advisory(matrix) -> int:
    """No-root HALF-B entry: per advisory-capable GPU, measure (or fall back to static) + advise."""
    print("ampere-autotune: no OC-write privilege -> HALF-B ADVISORY (recommend-only).\n")
    any_gpu = False
    for g in matrix.gpus:
        if not getattr(g, "advisory_capable", False):
            continue
        any_gpu = True
        s = g.sku  # dict from preflight.sku.SkuResult
        full_uuid = g.driver_state.get("uuid") or ""
        skuinfo = SkuInfo(s.get("sku_class", ""), s.get("mem_type", ""), s.get("offset_support", ""))
        roof = Roofline(sku_peak_gbs=_peak_gbs(s.get("name"), skuinfo.mem_type))
        try:
            # bw_verify scoped by full UUID (CUDA_VISIBLE_DEVICES); NVML telemetry by index
            m = collect_measurements(g.index, uuid=full_uuid or None)
        except Exception as e:                   # never let a telemetry hiccup break the advisory
            print(f"GPU {g.index} {s.get('name')}: telemetry collect failed ({e}); static guidance only.\n")
            m = None
        print(f"=== GPU {g.index} {s.get('name')} [{full_uuid[:20]}] ===")
        print(render(advise(m, skuinfo, roof)))
    if not any_gpu:
        print("No advisory-capable GPU (need NVML read access).")
        return 2
    return 0
