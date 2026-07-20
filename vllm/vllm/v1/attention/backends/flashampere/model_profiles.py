"""Model-architecture -> flashampere routing profiles (the fork's single source of truth).

Adding a validated model = add ONE entry to ``PROFILES``. ``apply_famp_profile`` runs once from
``VllmConfig.try_verify_and_update_config`` -- in the main process, before the per-architecture
config handler (e.g. the Gemma4 TRITON-force gate) reads the env AND before worker subprocesses are
spawned -- so the ``VLLM_FAMP_*`` env it sets is both seen by that gate and inherited by every worker.

Scope: only the famp ATTENTION path is routed here (it needs explicit env). The famp MARLIN kernel
self-gates (plugin presence + built arch + ``can_implement``), so it needs no per-model flag -- the
``marlin`` field below is DOCUMENTATION only (the end-to-end validation status per architecture).

Unlisted architectures get stock defaults (no famp attention); marlin still self-gates as usual.
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
    """Per-architecture famp routing. Attention fields are applied; marlin/note are documentation."""

    # --- attention routing (actively applied to the environment) ---
    flashampere: bool = False  # select the FlashAmpere CUSTOM attention backend
    own_prefill: bool = False  # hd512 prefill via famp's vendored FI (relaxed IsInvalid + fp16-PV)
    xqa_hd512: bool = False     # hd512 decode via famp's own vendored XQA kernel
    # --- documentation only (famp-marlin self-gates; NOT applied here) ---
    marlin: str = ""            # famp-marlin (W4A8/W4A16) end-to-end validation status for this arch
    note: str = ""


# Keyed by ``model_config.architecture`` (the resolved primary ``architectures[0]``).
PROFILES: dict[str, FampProfile] = {
    # -- Gemma4 (e.g. 12B): heterogeneous heads hd256 (sliding) / hd512 (global full-attn). famp's
    #    hd512 prefill is +7-9% over TRITON, but famp's hd512 XQA decode is -4..-14% (net slower for
    #    decode-heavy serving). The ideal split (famp prefill + TRITON decode, same layer) needs
    #    FA<->TRITON metadata bridging and is not ready. So default = TRITON (validated 5/5, fastest
    #    decode). To opt into the famp prefill path, set VLLM_FLASHAMPERE=1 + VLLM_FAMP_OWN_PREFILL=1
    #    + VLLM_FAMP_XQA_HD512=1 explicitly. Flip flashampere=True here once the prefill/decode hybrid lands.
    "Gemma4UnifiedForConditionalGeneration": FampProfile(
        flashampere=False,
        marlin="W4A16 data-free RTN validated coherent; W4A8-int8 NO-GO (L0 down_proj outlier -> SmoothQuant)",
        note="hd512 full-attn; default TRITON (decode fastest). famp prefill +8% via explicit flags; hybrid pending",
    ),
    "Gemma4ForConditionalGeneration": FampProfile(
        flashampere=False,
        note="hd512 full-attn (Gemma4 conditional-generation); default TRITON; hybrid pending",
    ),
    "Gemma4ForCausalLM": FampProfile(
        flashampere=False,
        note="hd512 full-attn (Gemma4 text-only); default TRITON; hybrid pending",
    ),

    # -- Qwen3.6-27B dense (hd128). famp MARLIN validated (W4A16 bit-exact vs stock; W4A8-int8
    #    engaged + coherent). famp ATTENTION is NOT beneficial: hd128 sinks to stock FA (the fp16-PV
    #    prefill leg is 0 e2e), so attention stays default (FLASH_ATTN). Listed for the marlin doc.
    "Qwen3_5ForConditionalGeneration": FampProfile(
        flashampere=False,
        marlin="W4A16 famp bit-exact vs stock _C (GSM8K 87.5%); W4A8-int8 engaged + coherent (85%)",
        note="dense hd128 -> famp attention gives 0 e2e (sinks to FA); marlin auto-engages",
    ),

    # -- Qwen3.6-35B-A3B MoE. Quantized GEMMs are all MoE experts on stock CompressedTensorsWNA16
    #    MarlinMoEMethod (famp deliberately does NOT vendor MoE); dense/attention linears are bf16.
    #    So famp MARLIN does not engage and famp ATTENTION is not beneficial -> all defaults.
    "Qwen3_5MoeForConditionalGeneration": FampProfile(
        flashampere=False,
        marlin="N/A - experts use stock Marlin MoE (famp not vendored for MoE); dense/attn = bf16",
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
