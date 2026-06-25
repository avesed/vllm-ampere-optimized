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
