# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere capability detection — per-card + per-toggle gating.

The make-or-break decision is fp16-accumulate PV: it runs at 2x ONLY on GeForce-GA10x
consumer Ampere (RTX 3090/3080/3070/3060/3050); on pro Ampere (A100/A40/A6000/A10) the
FP32-accumulate path is already full-rate, so fp16-PV gives ZERO speed and only costs
accuracy. The gate is therefore the device NAME, never ``__CUDA_ARCH__ >= 860`` (which
would silently degrade the pro sm_86 cards).

detect() takes its inputs by argument so the whole policy is unit-testable on CPU with no
GPU; gather_inputs() collects the real values from the platform + env at backend init.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .dispatch import Cap

# Substring that identifies a GeForce GA10x consumer card (all RTX 30-series, incl. Ti and
# Laptop). Excludes Ada "GeForce RTX 40..." and every pro SKU (A100/A40/A6000/A10/...).
_GEFORCE_GA10X_MARK = "geforce rtx 30"


def classify_card(cc_major: int, device_name: str) -> Cap:
    """Map (compute-capability major, device name) -> capability class.

    Ampere is cc major 8 (incl. Ada 8.9, which is a pro/consumer non-GA10x => SERVER).
    """
    if cc_major != 8:
        return Cap.OTHER
    if _GEFORCE_GA10X_MARK in device_name.lower():
        return Cap.GEFORCE_GA10X
    return Cap.AMPERE_SERVER


def _envflag(env: dict[str, str], name: str, default: str = "0") -> bool:
    return env.get(name, default) in ("1", "true", "True")


@dataclass(frozen=True)
class FlashAmpereCaps:
    cap: Cap
    has_flashinfer: bool
    has_sage: bool
    has_fp16pv_kernel: bool  # vendored flashinfer exposes use_fp16_pv_reduction (patch 0007)
    fp16pv_on: bool  # VLLM_FLASHAMPERE_PV_FP16 (fp16-served fp16-PV; default on)
    bf16cvt_on: bool  # VLLM_FLASHAMPERE_BF16CVT (bf16-served fp16-PV via upcast; default on)
    sage_on: bool  # VLLM_FLASHAMPERE_SAGE

    @property
    def pv_fp16(self) -> bool:
        """Effective fp16-PV: GeForce-GA10x AND vendored kernel present AND toggled on.
        Forced off on AMPERE_SERVER/OTHER (0 gain + accuracy cost) regardless of the toggle."""
        return (
            self.cap is Cap.GEFORCE_GA10X
            and self.has_fp16pv_kernel
            and self.fp16pv_on
        )

    @property
    def pv_fp16_bf16(self) -> bool:
        """Effective bf16cvt (bf16-served fp16-PV via runtime upcast): same GeForce-GA10x +
        vendored-kernel gate as pv_fp16, but its OWN toggle. Separate from fp16pv_on because the
        bf16 source adds a runtime upcast cost and a (small) numeric round-trip, so it ships
        opt-in until a bf16-served e2e A/B promotes it. Forced off on AMPERE_SERVER/OTHER."""
        return (
            self.cap is Cap.GEFORCE_GA10X
            and self.has_fp16pv_kernel
            and self.bf16cvt_on
        )

    def enabled(self, name: str) -> bool:
        """Is the dispatch leg `name` permitted to run (lib present + toggled on)?"""
        if name == "fp16pv":
            # Standalone fp16-PV prefill: needs the kernel + an effective (GeForce) gate.
            return self.has_flashinfer and self.pv_fp16
        if name == "bf16cvt":
            # bf16-served fp16-PV (runtime upcast): same kernel + GeForce gate, own toggle.
            return self.has_flashinfer and self.pv_fp16_bf16
        if name == "sage":
            return self.has_sage and self.sage_on
        return False


def detect(
    cc_major: int,
    device_name: str,
    env: dict[str, str],
    has_flashinfer: bool,
    has_sage: bool,
    has_fp16pv_kernel: bool,
) -> FlashAmpereCaps:
    """Pure policy: build the caps from explicit inputs (unit-testable, no GPU)."""
    master = _envflag(env, "VLLM_FLASHAMPERE")
    # Sub-toggles: fp16-PV legs (fp16pv fp16-served / bf16cvt bf16-served) default ON — they are
    # the primary Ampere prefill win and are GeForce-GA10x-gated by the pv_fp16* properties, so
    # default-on is a no-op on AMPERE_SERVER/OTHER. Sage stays research/off. int8-QK was removed
    # (net-negative everywhere). The master gate must also be on for any leg to fire.
    fp16pv_on = master and _envflag(env, "VLLM_FLASHAMPERE_PV_FP16", "1")
    bf16cvt_on = master and _envflag(env, "VLLM_FLASHAMPERE_BF16CVT", "1")
    sage_on = master and _envflag(env, "VLLM_FLASHAMPERE_SAGE", "0")
    return FlashAmpereCaps(
        cap=classify_card(cc_major, device_name),
        has_flashinfer=has_flashinfer,
        has_sage=has_sage,
        has_fp16pv_kernel=has_fp16pv_kernel,
        fp16pv_on=fp16pv_on,
        bf16cvt_on=bf16cvt_on,
        sage_on=sage_on,
    )


def gather_inputs() -> dict:
    """Collect the real runtime inputs for detect() (called once at backend init)."""
    cc_major = 0
    device_name = ""
    try:
        from vllm.platforms import current_platform

        dc = current_platform.get_device_capability()
        if dc is not None:
            cc_major = dc.major
        device_name = current_platform.get_device_name()
    except Exception:
        pass

    try:
        import flashinfer  # noqa: F401

        has_flashinfer = True
    except Exception:
        has_flashinfer = False

    try:
        import sageattention  # noqa: F401

        has_sage = True
    except Exception:
        has_sage = False

    return dict(
        cc_major=cc_major,
        device_name=device_name,
        env=dict(os.environ),
        has_flashinfer=has_flashinfer,
        has_sage=has_sage,
        has_fp16pv_kernel=_flashinfer_has_fp16pv(),
    )


def _flashinfer_has_fp16pv() -> bool:
    """True iff the vendored FlashInfer exposes the use_fp16_pv_reduction lever (patch 0007).
    Probed so VLLM_FLASHAMPERE_PV_FP16=1 on a stock (un-patched) FlashInfer is a silent no-op
    rather than a crash."""
    try:
        import inspect

        import flashinfer

        sig = inspect.signature(flashinfer.single_prefill_with_kv_cache)
        return "use_fp16_pv_reduction" in sig.parameters
    except Exception:
        return False
