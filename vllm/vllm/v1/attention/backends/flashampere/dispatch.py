# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere dispatch core — pure, GPU-free, unit-testable.

Maps a per-call ``DispatchKey`` (phase x head_dim x dtype x kv-quant x capability x
plain-decoder) to the ordered list of Ampere kernels that may run it. An empty list
means "no fast kernel applies" -> the impl sinks to stock FlashAttention (bit-faithful).

This module imports NOTHING from torch/CUDA or the vllm runtime so the entire routing
decision can be enumerated and asserted on CPU in milliseconds (see test_dispatch.py).
Enablement (is a leg toggled on? is its library present? does the card qualify?) is NOT
decided here — that is the capability/runtime concern (capability.py + impl.forward).
The table encodes only STRUCTURAL eligibility, which is what must stay correct across
vLLM rebases.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

# Head dims each prefill leg accelerates. fp16-PV (fp16pv fp16-served / bf16cvt bf16-served) targets
# the Qwen3.x hd256 full-attn layers; SageAttn targets the smaller dense hd's. Extend to widen.
FP16PV_HEADS: tuple[int, ...] = (256,)  # fp16-served (q dtype == fp16)
BF16CVT_HEADS: tuple[int, ...] = (256,)  # bf16-served (runtime upcast to fp16)
XQA_VERIFY_HEADS: tuple[int, ...] = (64, 128, 256)  # MTP spec-verify (XQA headElems 64/128/256)
SAGE_HEADS: tuple[int, ...] = (64, 96, 128)


class Phase(Enum):
    """Per-call attention phase, derived from metadata in impl._classify (CPU-only)."""

    PREFILL = "prefill"  # fresh or chunked-prefix prefill (large query)
    DECODE = "decode"  # query_len <= 1
    VERIFY = "verify"  # uniform small-query spec-decode verify (1 < q <= 1+K)
    OTHER = "other"  # encoder / cascade / unclassifiable -> sink


class Cap(Enum):
    """SM-capability CLASS, not raw compute capability (per-card, set in capability.py)."""

    GEFORCE_GA10X = "geforce_ga10x"  # 3090/3080/3070/3060/3050 — fp16-PV runs at 2x
    AMPERE_SERVER = "ampere_server"  # A100/A40/A6000/A10/... — fp16-PV gives 0, forced off
    OTHER = "other"  # not Ampere (sm major != 8) — backend declines everything


class QSrc(Enum):
    """Query source dtype CLASS — selects which fp16-PV variant (if any) is structurally
    eligible. fp16-PV is half-only (DTypeProb=half), so a fp16-served model (HALF) routes to
    the fp16pv leg directly, while a bf16-served model (BF16) routes to the bf16cvt leg, which
    upcasts Q/K/V to fp16 at runtime (lossless: fp16 has 10 mantissa bits vs bf16's 7) then runs
    the SAME fp16 cubin. OTHER (fp8 etc.) is ineligible for either -> sink."""

    HALF = "half"  # float16
    BF16 = "bf16"  # bfloat16 — runtime-upcast to fp16 for the fp16-PV cubin
    OTHER = "other"  # anything else -> no fp16-PV leg


class DispatchKey(NamedTuple):
    phase: Phase
    head_dim: int
    q_src: QSrc  # query source dtype class (fp16->fp16pv, bf16->bf16cvt, else no fp16-PV leg)
    kv_quantized: bool  # fp8/int8 kv cache (breaks the prefill cache-read contract)
    cap: Cap
    plain_decoder: bool  # no SWA/alibi/softcap/sinks and attn_type == DECODER


class KernelEntry(NamedTuple):
    name: str
    capture_safe: bool  # may execute inside a captured cudagraph (no CPU<->GPU sync)


class KernelDecline(Exception):
    """Raised inside a kernel at run time (missing lib, overflow, unsupported slice) to
    fall back to the next candidate, ultimately stock FA. Never a correctness compromise."""


# Ordered (matcher, entry) rows. resolve() returns every structural match in order; the
# impl runs the first ENABLED + capture-compatible one, else sinks. First-registered wins
# ties (fp16pv and bf16cvt are dtype-exclusive on the hd256 prefill slot, so no real contention).
_TABLE: list[tuple[Callable[[DispatchKey], bool], KernelEntry]] = []


def register(matcher: Callable[[DispatchKey], bool], entry: KernelEntry) -> None:
    _TABLE.append((matcher, entry))


def resolve(key: DispatchKey) -> tuple[KernelEntry, ...]:
    """All structurally-eligible kernels for this call, in priority order ( () => sink)."""
    return tuple(entry for matcher, entry in _TABLE if matcher(key))


def clear() -> None:
    _TABLE.clear()


def build_default_table() -> None:
    """(Re)install the v1 routing table. Idempotent; called at import and in tests."""
    clear()

    # hd256 prefill, fp16 query: fp16-PV FlashInfer prefill (fp16-served models). The fp16-PV
    # reduction (DTypeProb=half) is the measured Ampere win; int8-QK was removed after a sweep
    # found it net-negative in every scenario (fresh 16/32/64k + cached-prefix) — its O(L^2)
    # per-token dequant + fp16-KV-read-then-requant tax always exceeds the ~1.7% IMMA-QK gain.
    register(
        lambda k: (
            k.phase is Phase.PREFILL
            and k.head_dim in FP16PV_HEADS
            and k.plain_decoder
            and not k.kv_quantized
            and k.q_src is QSrc.HALF
        ),
        KernelEntry("fp16pv", capture_safe=False),
    )
    # Same slot, bf16 query: bf16cvt = fp16pv with a runtime bf16->fp16 upcast of Q/K/V (the
    # kernel already casts; this row only widens eligibility). Delivers the half-only fp16-PV
    # win to bf16-served models (the Qwen3.x default dtype). GeForce-gated at enablement.
    register(
        lambda k: (
            k.phase is Phase.PREFILL
            and k.head_dim in BF16CVT_HEADS
            and k.plain_decoder
            and not k.kv_quantized
            and k.q_src is QSrc.BF16
        ),
        KernelEntry("bf16cvt", capture_safe=False),
    )
    # MTP spec-decode VERIFY (uniform q=1+K) -> famp's vendored XQA (decode-shaped, KV-split,
    # warp-specialized; q-scaling ~flat so q=1+K is 1.8-4.3x faster than FA2 fwd_kvcache on Ampere,
    # cos=1.0). Any-dtype (XQA handles fp16/bf16). capture_safe=True: the kernel + the slim
    # module.xqa_wrapper launch are cudagraph-capturable (persistent buffers, copy_/zero_/launch only
    # — verified capture+replay cos=1.0); this is REQUIRED for the e2e win (verify runs in the captured
    # decode graph, so per-call Python overhead is eliminated). Declines to base-FA verify on shortfall.
    register(
        lambda k: (
            k.phase is Phase.VERIFY
            and k.head_dim in XQA_VERIFY_HEADS
            and k.plain_decoder
            and not k.kv_quantized
        ),
        KernelEntry("xqa_verify", capture_safe=True),
    )
    # Small dense head dims -> SageAttn int8-QK prefill (research-grade; gated by HAS_SAGE
    # + toggle at enablement time, so the row is harmless when the package is absent).
    register(
        lambda k: (
            k.phase is Phase.PREFILL
            and k.head_dim in SAGE_HEADS
            and k.plain_decoder
            and not k.kv_quantized
        ),
        KernelEntry("sage", capture_safe=False),
    )
    # DECODE / OTHER / quantized-KV / non-plain-decoder: no row -> () -> stock FA. (DECODE q=1 is
    # bandwidth-bound; stock FA fwd_kvcache split-KV is its lever.) VERIFY now routes to xqa_verify
    # above (opt-in, env-gated); when disabled it declines/sinks to the base-FA fwd_kvcache verify.


build_default_table()
