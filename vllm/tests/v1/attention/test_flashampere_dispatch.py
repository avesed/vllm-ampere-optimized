# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the flashampere dispatch + capability policy (no GPU, no vllm runtime).

These assert the STRUCTURAL routing table and the per-card/per-toggle gating that decide which
Ampere kernel (if any) a call uses. They are the cheap guardrail that the 4-kernel composition
stays correct across vLLM rebases. Runnable directly (`python test_flashampere_dispatch.py`) or
under pytest — both import only flashampere.{dispatch,capability}, which are torch/CUDA-free.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

# Bootstrap: import flashampere.{dispatch,capability} WITHOUT importing the heavy vllm package.
# Walk up to the repo's `vllm/vllm/v1/attention/backends` dir and put it on sys.path so the
# `flashampere` package imports as a top-level package (its __init__ is import-light).
_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    _backends = _p / "vllm" / "v1" / "attention" / "backends"
    if (_backends / "flashampere" / "dispatch.py").exists():
        sys.path.insert(0, str(_backends))
        break

from flashampere import capability as cap_mod  # noqa: E402
from flashampere import dispatch as d  # noqa: E402
from flashampere.dispatch import Cap, DispatchKey, Phase, QSrc  # noqa: E402

HEADS = (64, 96, 128, 256, 576)
PHASES = (Phase.PREFILL, Phase.DECODE, Phase.VERIFY, Phase.OTHER)
CAPS = (Cap.GEFORCE_GA10X, Cap.AMPERE_SERVER, Cap.OTHER)


def _key(phase, head_dim, *, q_src=QSrc.HALF, kv_quantized=False,
         cap=Cap.GEFORCE_GA10X, plain_decoder=True):
    return DispatchKey(phase, head_dim, q_src, kv_quantized, cap, plain_decoder)


def _names(key):
    return tuple(e.name for e in d.resolve(key))


def test_only_prefill_ever_matches():
    # No decode/verify/other call may ever hit a fast kernel -> sink to stock FA.
    for phase, hd in itertools.product((Phase.DECODE, Phase.VERIFY, Phase.OTHER), HEADS):
        assert _names(_key(phase, hd)) == (), (phase, hd)


def test_fp16pv_owns_hd256_prefill():
    # int8-QK removed (net-negative in every scenario); fp16-PV legs own the hd256 prefill slot.
    # fp16 q -> fp16pv only.
    assert _names(_key(Phase.PREFILL, 256, q_src=QSrc.HALF)) == ("fp16pv",)
    # bf16 q -> bf16cvt only (fp16-PV via runtime bf16->fp16 upcast).
    assert _names(_key(Phase.PREFILL, 256, q_src=QSrc.BF16)) == ("bf16cvt",)
    # other dtype (e.g. fp8) -> no fp16-PV leg -> sink to stock FA.
    assert _names(_key(Phase.PREFILL, 256, q_src=QSrc.OTHER)) == ()


def test_fp16pv_legs_are_dtype_exclusive():
    # fp16pv fires ONLY for fp16 source, bf16cvt ONLY for bf16 source — never both, never crossed.
    half = _names(_key(Phase.PREFILL, 256, q_src=QSrc.HALF))
    bf16 = _names(_key(Phase.PREFILL, 256, q_src=QSrc.BF16))
    assert "fp16pv" in half and "fp16pv" not in bf16
    assert "bf16cvt" in bf16 and "bf16cvt" not in half
    # bf16cvt is PREFILL-only and never leaks to decode/verify/other.
    for phase in (Phase.DECODE, Phase.VERIFY, Phase.OTHER):
        assert "bf16cvt" not in _names(_key(phase, 256, q_src=QSrc.BF16)), phase


def test_sage_owns_small_head_dims():
    for hd in (64, 96, 128):
        assert _names(_key(Phase.PREFILL, hd)) == ("sage",), hd


def test_unsupported_head_dims_sink():
    assert _names(_key(Phase.PREFILL, 576)) == ()


def test_quantized_kv_always_sinks():
    # fp8/int8 KV breaks the prefill cache-read contract -> no fast row, every head dim.
    for hd in HEADS:
        assert _names(_key(Phase.PREFILL, hd, kv_quantized=True)) == (), hd


def test_non_plain_decoder_sinks():
    # SWA/alibi/softcap/sinks/encoder => plain_decoder False => sink, every head dim.
    for hd in HEADS:
        assert _names(_key(Phase.PREFILL, hd, plain_decoder=False)) == (), hd


def test_capability_class():
    assert cap_mod.classify_card(8, "NVIDIA GeForce RTX 3090") is Cap.GEFORCE_GA10X
    assert cap_mod.classify_card(8, "NVIDIA GeForce RTX 3080 Ti") is Cap.GEFORCE_GA10X
    assert cap_mod.classify_card(8, "NVIDIA GeForce RTX 3060 Laptop GPU") is Cap.GEFORCE_GA10X
    # Pro Ampere sm_80/86 -> SERVER (fp16-PV forced off).
    for nm in ("NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA A10"):
        assert cap_mod.classify_card(8, nm) is Cap.AMPERE_SERVER, nm
    # Ada (cc 8.9, GeForce RTX 40xx) is NOT GA10x -> SERVER (fp16-PV nerf moved to fp8).
    assert cap_mod.classify_card(8, "NVIDIA GeForce RTX 4090") is Cap.AMPERE_SERVER
    # Non-Ampere.
    assert cap_mod.classify_card(9, "NVIDIA H100") is Cap.OTHER
    assert cap_mod.classify_card(7, "Tesla V100") is Cap.OTHER


def test_pv_fp16_never_on_non_geforce():
    # The load-bearing safety invariant: fp16-PV must be off everywhere but GeForce-GA10x,
    # even with the toggle on and the kernel present.
    env = {"VLLM_FLASHAMPERE": "1", "VLLM_FLASHAMPERE_PV_FP16": "1"}
    for nm, expect in (
        ("NVIDIA GeForce RTX 3090", True),
        ("NVIDIA A40", False),
        ("NVIDIA RTX A6000", False),
        ("NVIDIA A100-SXM4-80GB", False),
    ):
        caps = cap_mod.detect(8, nm, env, has_flashinfer=True, has_sage=True,
                              has_fp16pv_kernel=True)
        assert caps.pv_fp16 is expect, nm
        # fp16-PV with an un-patched flashinfer (no kernel) is a silent no-op, never on.
        caps_nokernel = cap_mod.detect(8, nm, env, has_flashinfer=True, has_sage=True,
                                       has_fp16pv_kernel=False)
        assert caps_nokernel.pv_fp16 is False, nm


def test_master_gate_off_disables_every_leg():
    env = {"VLLM_FLASHAMPERE": "0", "VLLM_FLASHAMPERE_PV_FP16": "1",
           "VLLM_FLASHAMPERE_BF16CVT": "1", "VLLM_FLASHAMPERE_SAGE": "1"}
    caps = cap_mod.detect(8, "NVIDIA GeForce RTX 3090", env, has_flashinfer=True,
                          has_sage=True, has_fp16pv_kernel=True)
    for leg in ("fp16pv", "bf16cvt", "sage"):
        assert caps.enabled(leg) is False, leg


def test_default_toggles_when_master_on():
    env = {"VLLM_FLASHAMPERE": "1"}  # only master on -> fp16-PV legs default-on (GeForce), sage off
    caps = cap_mod.detect(8, "NVIDIA GeForce RTX 3090", env, has_flashinfer=True,
                          has_sage=True, has_fp16pv_kernel=True)
    assert caps.enabled("fp16pv") is True  # fp16-PV default-on now (int8-QK removed); GeForce-gated
    assert caps.enabled("bf16cvt") is True  # bf16cvt default-on now; GeForce-gated
    assert caps.enabled("sage") is False  # Sage defaults off (research)


def test_bf16cvt_gating():
    # bf16cvt: GeForce-GA10x-only, needs the fp16-PV kernel — never on AMPERE_SERVER. (PV_FP16 set
    # off here to isolate bf16cvt's own toggle from the now-default-on fp16pv leg.)
    env = {"VLLM_FLASHAMPERE": "1", "VLLM_FLASHAMPERE_BF16CVT": "1", "VLLM_FLASHAMPERE_PV_FP16": "0"}
    on = cap_mod.detect(8, "NVIDIA GeForce RTX 3090", env, has_flashinfer=True,
                        has_sage=True, has_fp16pv_kernel=True)
    assert on.enabled("bf16cvt") is True and on.pv_fp16_bf16 is True
    # PV_FP16 explicitly off here -> fp16-served leg off, bf16cvt independent.
    assert on.enabled("fp16pv") is False
    for nm in ("NVIDIA A40", "NVIDIA A100-SXM4-80GB", "NVIDIA RTX A6000"):
        srv = cap_mod.detect(8, nm, env, has_flashinfer=True, has_sage=True,
                             has_fp16pv_kernel=True)
        assert srv.enabled("bf16cvt") is False, nm  # never on pro Ampere
    # un-patched flashinfer (no fp16-PV kernel) -> silent no-op.
    nok = cap_mod.detect(8, "NVIDIA GeForce RTX 3090", env, has_flashinfer=True,
                         has_sage=True, has_fp16pv_kernel=False)
    assert nok.enabled("bf16cvt") is False


def test_enablement_requires_library():
    env = {"VLLM_FLASHAMPERE": "1", "VLLM_FLASHAMPERE_SAGE": "1"}
    caps = cap_mod.detect(8, "NVIDIA GeForce RTX 3090", env, has_flashinfer=False,
                          has_sage=False, has_fp16pv_kernel=False)
    assert caps.enabled("fp16pv") is False  # no flashinfer / no fp16-PV kernel
    assert caps.enabled("bf16cvt") is False  # no flashinfer / no fp16-PV kernel
    assert caps.enabled("sage") is False  # no sageattention


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
