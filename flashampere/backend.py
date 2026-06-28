# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAmpereBackend — the unified Ampere attention backend, registered into Backend.CUSTOM.

Inherits FlashAttentionBackend wholesale (KV-cache shape/layout, metadata builder, and all the
validate_configuration supports_* gates) and changes only three things: get_name()=="CUSTOM"
(round-trips through AttentionBackendEnum["CUSTOM"]), get_impl_cls()->FlashAmpereImpl, and
supports_compute_capability restricted to Ampere (sm major 8). The compute-capability gate
self-fences: on non-Ampere, validate_configuration appends "compute capability not supported" so
the cuda.py priority walk skips CUSTOM and falls to FLASH_ATTN. Inheriting FA's
supports_kv_cache_dtype also means fp8-KV makes CUSTOM invalid on Ampere (FA3+sm90-only) -> the
selector routes fp8-KV elsewhere automatically (the "fp8-KV is either/or" scoping, for free).

Registered in EVERY process (engine core + each TP/PP worker) via the vllm.general_plugins
entry-point register_flashampere; opt-in behind VLLM_FLASHAMPERE=1 so a plain image swap does not
change attention for every model.
"""
from __future__ import annotations

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from .impl import FlashAmpereImpl

logger = init_logger(__name__)


class FlashAmpereBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        # Registered into the CUSTOM slot; AttentionBackendEnum["CUSTOM"] resolves cleanly.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[FlashAmpereImpl]:
        return FlashAmpereImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Ampere only (sm_80/sm_86/sm_89). Hopper/Blackwell have better backends; non-Ampere
        # gets "compute capability not supported" -> selector falls back to FLASH_ATTN.
        return capability.major == 8

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # FA's ceiling is 256 (flash_attn supports_head_size: >256 needs FA4, not on Ampere). famp
        # extends to 512 for the PREFILL fp16-PV leg: the vendored-FI prefill IsInvalid register
        # heuristic (8*NUM_MMA_D_VO>=256) is relaxed and fp16-PV halves the O accumulator to fit
        # 256 regs -> hd512 prefill runs on Ampere (validated cos=1.0). Unlocks Gemma4's hd512
        # full-attn layers. (Decode at hd512 still sinks to FA, which rejects it -> hd512 layers are
        # prefill-only through famp for now; covered for prefill-bench / enforce-eager.)
        if head_size % 8 != 0:
            return False
        return head_size <= 512

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # VL models (Gemma4 unified) declare a multimodal-prefix capability; FA's default is False,
        # which would filter CUSTOM out of the per-layer backend selection for every VL model. famp's
        # prefill leg handles the TEXT path correctly (causal m.causal); the mm-prefix image-token
        # bidirectional mask is carried in the attention metadata. Declare support so CUSTOM is
        # selectable for Gemma4's hd512 full-attn layers (text-serving / prefill-bench).
        return True


def register_flashampere() -> None:
    """vllm.general_plugins entry-point: install FlashAmpere into Backend.CUSTOM in every process.

    Opt-in: no-op unless VLLM_FLASHAMPERE truthy. load_general_plugins() runs this in the engine
    core AND each TP/PP worker subprocess before attention-backend selection, so CUSTOM is
    registered everywhere the cuda.py priority-prepend (guarded on CUSTOM.is_overridden()) needs it.
    Idempotent.
    """
    import os

    if os.environ.get("VLLM_FLASHAMPERE", "0") not in ("1", "true", "True"):
        return

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm.v1.attention.backends.flashampere.backend.FlashAmpereBackend",
    )
    logger.info(
        "flashampere plugin: registered FlashAmpereBackend in Backend.CUSTOM (VLLM_FLASHAMPERE=1). "
        "Auto-selected for Ampere full-attn layers; hd256 prefill -> fp16-PV: fp16pv (fp16-served) "
        "| bf16cvt (bf16->fp16 upcast); GeForce-GA10x-gated, default-on; "
        "verify/decode/everything-else -> stock FA."
    )
