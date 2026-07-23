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
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)

from .impl import FlashAmpereImpl

logger = init_logger(__name__)


class FlashAmpereMetadataBuilder(FlashAttentionMetadataBuilder):
    """FA builder + the CPU metadata twins the famp legs need for sync-free dispatch.

    FlashAttentionMetadata carries only GPU query_start_loc/seq_lens; the legs used to
    .tolist() them = 2 device syncs per LAYER per step (paid even when the mixed-batch
    decline then sank the call to stock FA). CommonAttentionMetadata already holds the CPU
    twins, but the impl never sees it — attach them here (once per STEP, no extra sync:
    query_start_loc_cpu is a builder input; seq_lens_cpu a lazily-cached property)."""

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        md.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        md.seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        return md


class FlashAmpereBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[FlashAmpereMetadataBuilder]:
        return FlashAmpereMetadataBuilder

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

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        # FA's supports_combination rejects use_mm_prefix unless FA4 (no FA4 on Ampere), which
        # inheritance would apply to CUSTOM for BOTH head sizes of a Gemma4 VL model. But Gemma4
        # CLEARS mm_prefix for its hd512 FULL-ATTN layers (gemma4_mm._clear_mm_prefix_for_full_attn_
        # layers) -> those layers are plain causal, which the famp hd512 legs own (XQA gemm0-once
        # decode 1.5-2.8x faster than TRITON; fp16-PV prefill 1.9-2.9x). So allow the combination
        # for head_size>256 only; the hd256 SLIDING layers keep FA's rejection (super) -> they fall
        # to TRITON/FLEX, which correctly carry their bidirectional image mm_prefix. Keep FA's sink
        # gate. hd512 has no stock fallback anyway (FA rejects >256), so famp must own it.
        if use_mm_prefix and head_size > 256:
            if has_sink and device_capability < DeviceCapability(9, 0):
                return "sink not supported on compute capability < 9.0"
            return None
        return super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )


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
