"""Model-architecture -> flashampere routing profiles (the fork's single source of truth).

Adding a validated model = add ONE entry to ``PROFILES``. ``apply_famp_profile`` runs once from
``VllmConfig.try_verify_and_update_config`` -- in the main process, before the per-architecture
config handler (e.g. the Gemma4 TRITON-force gate) reads the env AND before worker subprocesses are
spawned -- so the ``VLLM_FAMP_*`` env it sets is both seen by that gate and inherited by every worker.

Scope: only the famp ATTENTION path is routed here (it needs explicit env).

Unlisted architectures get stock defaults (no famp attention).
User-set env always wins: every write uses ``setdefault`` so an explicit ``VLLM_FLASHAMPERE=0`` (etc.)
is never overridden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # never let a logging import break config setup
    import logging

    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FampProfile:
    """Per-architecture famp routing. Attention fields are applied; note is documentation."""

    # --- attention routing (actively applied to the environment) ---
    flashampere: bool = False  # select the FlashAmpere CUSTOM attention backend
    own_prefill: bool = False  # hd512 prefill via famp's vendored FI (relaxed IsInvalid + fp16-PV)
    xqa_hd512: bool = False     # hd512 decode via famp's own vendored XQA kernel
    # --- documentation only (NOT applied here) ---
    note: str = ""


# Keyed by ``model_config.architecture`` (the resolved primary ``architectures[0]``).
PROFILES: dict[str, FampProfile] = {
    # -- Gemma4 (e.g. 12B/31B): heterogeneous heads hd256 (sliding) / hd512 (global full-attn).
    #    flashampere=True: famp OWNS the hd512 full-attn layers (prefill via own-prefill fp16-PV =
    #    1.9-2.9x TRITON; decode via XQA gemm0-once single-K-read = 1.5-2.8x TRITON, cos=1.0). The
    #    per-head-size supports_combination gate keeps the hd256 SLIDING layers on TRITON/FLEX (they
    #    carry the bidirectional image mm_prefix, which famp declines). backend gate: famp is only
    #    selectable for hd512 (mm_prefix cleared there); text + image both correct. (The old "decode
    #    -14%" was the two-launch e2e; the gemm0-once single-K-read kernel reversed it decisively.)
    "Gemma4UnifiedForConditionalGeneration": FampProfile(
        flashampere=True,
        own_prefill=True,
        xqa_hd512=True,
        note="hd512 full-attn -> famp (prefill 1.9-2.9x + XQA gemm0-once decode 1.5-2.8x vs TRITON); hd256 sliding -> TRITON",
    ),
    "Gemma4ForConditionalGeneration": FampProfile(
        flashampere=True,
        own_prefill=True,
        xqa_hd512=True,
        note="hd512 full-attn -> famp (prefill+XQA-decode); hd256 sliding -> TRITON",
    ),
    "Gemma4ForCausalLM": FampProfile(
        flashampere=True,
        own_prefill=True,
        xqa_hd512=True,
        note="hd512 full-attn text-only -> famp (prefill+XQA-decode); hd256 sliding -> TRITON",
    ),

    # -- Qwen3.6-27B dense (hd128). famp ATTENTION is NOT beneficial: hd128 sinks to stock FA (the
    #    fp16-PV prefill leg is 0 e2e), so attention stays default (FLASH_ATTN).
    "Qwen3_5ForConditionalGeneration": FampProfile(
        flashampere=False,
        note="dense hd128 -> famp attention gives 0 e2e (sinks to FA)",
    ),

    # -- Qwen3.6-35B-A3B MoE. Quantized GEMMs are all MoE experts on stock kernels; dense/attention
    #    linears are bf16. famp ATTENTION is not beneficial -> all defaults.
    "Qwen3_5MoeForConditionalGeneration": FampProfile(
        flashampere=False,
        note="MoE + GDN hybrid; serves coherently on stock paths (GSM8K 85-95%)",
    ),
}


def resolve_profile(vllm_config) -> "FampProfile | None":
    """Look up the famp profile for this run's model architecture (or None if unlisted)."""
    mc = getattr(vllm_config, "model_config", None)
    if mc is None:
        return None
    arch = getattr(mc, "architecture", None)
    if arch is None:
        archs = getattr(mc, "architectures", None) or []
        arch = archs[0] if archs else None
    if arch is None:
        return None
    return PROFILES.get(arch)


def apply_famp_profile(vllm_config) -> "FampProfile | None":
    """Set the ``VLLM_FAMP_*`` env for this arch's famp-attention profile, respecting user env.

    Called once from ``try_verify_and_update_config`` before the per-arch handler + worker spawn.
    No-op for unlisted architectures or profiles with ``flashampere=False``. Returns the profile.
    """
    prof = resolve_profile(vllm_config)
    if prof is None or not prof.flashampere:
        return prof
    applied = []
    if os.environ.get("VLLM_FLASHAMPERE") is None:
        os.environ["VLLM_FLASHAMPERE"] = "1"; applied.append("VLLM_FLASHAMPERE")
    if prof.own_prefill and os.environ.get("VLLM_FAMP_OWN_PREFILL") is None:
        os.environ["VLLM_FAMP_OWN_PREFILL"] = "1"; applied.append("VLLM_FAMP_OWN_PREFILL")
    if prof.xqa_hd512 and os.environ.get("VLLM_FAMP_XQA_HD512") is None:
        os.environ["VLLM_FAMP_XQA_HD512"] = "1"; applied.append("VLLM_FAMP_XQA_HD512")
    if applied:
        arch = getattr(getattr(vllm_config, "model_config", None), "architecture", "?")
        logger.info("flashampere model-profile: %s -> auto-set %s (%s)", arch, applied, prof.note)
    return prof
