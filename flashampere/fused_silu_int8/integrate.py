"""Integrate the fused silu_and_mul_int8_quant kernel into the dense W4A8-int8-act FFN.

PREFILL lever (decode ~0). Eliminates the bf16 [M, intermediate] SiluAndMul intermediate HBM
round-trip by fusing silu*mul + per-token int8 quant, then feeding the pre-quantized activation
straight into down_proj's marlin GEMM (skipping marlin's internal per_token_quant_int8).

Mechanism: a `vllm.general_plugins` entry-point (register_fused_silu), the same proven shape as
flashampere — the ONLY mechanism that runs inside every TP/PP worker subprocess where the dense
MLP forward actually executes. Env-gated (VLLM_FAMP_FUSED_SILU=1) and only meaningful with
VLLM_MARLIN_INPUT_DTYPE=int8 (which is what makes down_proj an int8-act marlin layer).

The make-or-break is apply_gptq_marlin_linear_prequant: it must EXACTLY replicate
apply_gptq_marlin_linear (marlin_utils.py:567-631) MINUS the internal marlin_quant_input step,
i.e. it keeps the `a_scales = a_scales * input_global_scale` fold (marlin_utils.py:601) and the
exact ops.marlin_gemm positional arg order (the 7th positional `global_scale` is ALWAYS None;
the global scale lives in a_scales). Getting any of that wrong = silent garbage.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
# 1a. EXACT replication of apply_gptq_marlin_linear MINUS the internal quant.
# ----------------------------------------------------------------------------- #
def apply_gptq_marlin_linear_prequant(layer, kernel, a_int8, a_scales, bias=None):
    """Run down_proj's marlin int8-act GEMM on an ALREADY-quantized activation.

    Args:
        layer:    the down_proj nn.Module (carries input_global_scale, g_idx_sort_indices, weights).
        kernel:   the MarlinLinearKernel instance for `layer` (carries config, workspace, is_k_full).
        a_int8:   int8 [..., K] — bit-exact replacement for marlin_quant_input's reshaped_x output.
        a_scales: fp32 [..., 1] — RAW per-token scale from the fused kernel (PRE global-scale).
        bias:     optional bias tensor (Qwen MLP has none; supported for generality).
    Returns:
        output reshaped to a_int8.shape[:-1] + (output_size_per_partition,).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        USE_FP32_REDUCE_DEFAULT,
    )
    from vllm.scalar_type import scalar_types

    c = kernel.config
    # Return order is EXACTLY (w_q, w_s, w_zp, w_gidx) — MPLinearKernel.py:89-94.
    w_q, w_s, w_zp, w_gidx = kernel._get_weight_params(layer)

    # int4 weights were repacked to uint4b8 codes at load; FORCE uint4b8 (marlin.py:204-208).
    wtype = scalar_types.uint4b8 if c.weight_type == scalar_types.int4 else c.weight_type
    # The int8 branch of apply_gptq_marlin_linear asserts this (marlin_utils.py:597).
    assert wtype == scalar_types.uint4b8, "prequant int8 path requires uint4b8 weights"

    output_size_per_partition = c.partition_weight_shape[1]
    input_size_per_partition = c.partition_weight_shape[0]

    reshaped_x = a_int8.reshape(-1, a_int8.shape[-1])              # 2D [M, K]
    out_shape = a_int8.shape[:-1] + (output_size_per_partition,)  # leading dims + N_out
    a_scales = a_scales.reshape(-1, 1)                            # fp32 [M, 1]

    # MAKE-OR-BREAK: fold input_global_scale into a_scales (marlin_utils.py:601).
    # Non-None for grouped int8-act (g32, num_groups>1 — the dense 27B recipe); None for
    # channelwise (group_size==-1). Mirror getattr(layer, "input_global_scale", None) exactly
    # and GUARD the multiply — unconditional multiply OR unconditional skip breaks one config.
    igs = getattr(layer, "input_global_scale", None)
    if igs is not None:
        a_scales = a_scales * igs

    # use_atomic_add: stock derives this from the ORIGINAL bf16 input.dtype, which on sm8x+bf16
    # always disables atomicAdd (should_use_atomic_add_reduce, marlin_utils.py:502-522: False by
    # default, and the sm8x+bf16 guard returns False even when VLLM_MARLIN_USE_ATOMIC_ADD=1).
    # We hardcode False so the prequant int8 dtype cannot bypass that bf16 guard and accidentally
    # enable atomicAdd (which it could if the env were set, n<2048, k>=2048). Matches stock.
    use_atomic_add = False

    output = ops.marlin_gemm(
        reshaped_x,                # a            (int8, pre-quantized)
        None,                      # c
        w_q,                       # b_q_weight
        bias,                      # b_bias
        w_s,                       # b_scales     (int16-coded, float-dtype-viewed — UNCHANGED)
        a_scales,                  # a_scales     (per-token * input_global_scale)
        None,                      # global_scale (ALWAYS None — folded into a_scales above)
        w_zp,                      # b_zeros
        w_gidx,                    # g_idx
        layer.g_idx_sort_indices,  # perm
        kernel.workspace,          # workspace
        wtype,                     # b_q_type
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=kernel.is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
        is_zp_float=False,
    )
    return output.reshape(out_shape)


# ----------------------------------------------------------------------------- #
# 1b. Resolve the down_proj MarlinLinearKernel + int8 gate (both quant formats).
# ----------------------------------------------------------------------------- #
def _resolve_w4a8_marlin_kernel(down_proj):
    """Return the MarlinLinearKernel for `down_proj`, or None if not W4A8-int8-act marlin.

    Handles BOTH quant formats:
      compressed-tensors: layer.scheme.kernel
      awq_marlin:         layer.quant_method.kernel
    The true gate is the underlying MarlinLinearKernel.config.act_type == torch.int8.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )

    kernel = None
    sch = getattr(down_proj, "scheme", None)            # compressed-tensors
    if sch is not None:
        kernel = getattr(sch, "kernel", None)
    if kernel is None:                                  # awq_marlin
        qm = getattr(down_proj, "quant_method", None)
        kernel = getattr(qm, "kernel", None)
    if not isinstance(kernel, MarlinLinearKernel):
        return None
    if getattr(kernel.config, "act_type", None) != torch.int8:   # the true gate
        return None
    return kernel


# ----------------------------------------------------------------------------- #
# 1c. Patched dense-MLP forward.
# ----------------------------------------------------------------------------- #
def _make_patched_forward(orig_forward):
    from flashampere.fused_silu_int8.build import fused_silu_mul_quant_int8

    def forward(self, x):
        # CRITICAL: this same class (Qwen2MoeMLP == Qwen3NextMLP, and the structurally-identical
        # Qwen3MoeMLP) is instantiated BOTH as the DENSE MLP (expert_gate is None — the target) AND
        # as the MoE SHARED EXPERT (expert_gate=self.shared_expert_gate; qwen3_next.py:147-153).
        # The shared-expert forward ends with `out = F.sigmoid(self.expert_gate(x)[0]) * out`
        # (qwen2_moe.py:119-120). We do NOT replicate that here, so refuse the fused path entirely
        # whenever expert_gate is set and let stock handle shared experts. Per-instance + cheap.
        if getattr(self, "expert_gate", None) is not None:
            return orig_forward(self, x)

        dp = self.down_proj
        kernel = getattr(self, "_famp_dp_kernel", "UNSET")
        if kernel == "UNSET":
            kernel = _resolve_w4a8_marlin_kernel(dp)
            self._famp_dp_kernel = kernel        # MarlinLinearKernel or None, cached forever
        if kernel is None:
            return orig_forward(self, x)         # not W4A8-int8 marlin -> stock path

        # down_proj is RowParallelLinear (input_is_parallel); gate_up already runs per-rank, so the
        # SiluAndMul output is [M, intermediate_per_rank] = down_proj input_size_per_partition (K).
        gate_up, _ = self.gate_up_proj(x)
        a8, asc = fused_silu_mul_quant_int8(gate_up)   # int8 [..., N], fp32 [..., 1]

        # Shape sanity: a8's last dim must equal down_proj's K. Decline ONCE, permanently, on a
        # mismatch (avoids per-call garbage); fall back to the correct stock path.
        if a8.shape[-1] != kernel.config.partition_weight_shape[0]:
            logger.warning(
                "fused_silu: a8 last-dim %d != down_proj K %d; disabling fused path for this MLP.",
                a8.shape[-1],
                kernel.config.partition_weight_shape[0],
            )
            self._famp_dp_kernel = None
            return orig_forward(self, x)

        # Scope invariant: every Qwen gate_up/down_proj is bias=False. Stock RowParallelLinear
        # fuses bias INTO the marlin GEMM on rank 0 ONLY, BEFORE the all-reduce (linear.py:1551:
        # `bias_ = None if (tp_rank>0 or skip_bias_add) else self.bias`), so a post-all-reduce
        # all-ranks `out + bias` would double/triple-count under TP>1. Rather than re-implement
        # that (and skip_bias_add/return_bias) for a layer that never has bias in scope, assert the
        # invariant and bail to stock if it is ever violated — never emit a wrong-bias result.
        if getattr(dp, "bias", None) is not None or getattr(dp, "skip_bias_add", False):
            self._famp_dp_kernel = None
            return orig_forward(self, x)

        out = apply_gptq_marlin_linear_prequant(dp, kernel, a8, asc, bias=None)

        # Replicate RowParallelLinear.forward tail: all-reduce across TP ranks when reduce_results
        # (linear.py:1554-1558). No bias add here — bias is asserted None above (it would have been
        # fused into the GEMM pre-all-reduce on rank 0, not added post-reduce).
        if getattr(dp, "reduce_results", True) and getattr(dp, "tp_size", 1) > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            out = tensor_model_parallel_all_reduce(out)
        return out

    forward._famp_fused_silu = True              # idempotency marker
    return forward


# ----------------------------------------------------------------------------- #
# 1d. Plugin entry point — env-gated, idempotent, eager build.
# ----------------------------------------------------------------------------- #
def register_fused_silu():
    """vllm.general_plugins entry-point: fuse silu+int8-quant into the dense W4A8 FFN.

    load_general_plugins() runs this in the engine core AND each TP/PP worker subprocess before the
    model is built, so the dense-MLP class is patched everywhere the FFN forward executes.
    Idempotent (the _famp_fused_silu marker guards against double-wrapping).

    GATING: vLLM spawns TP workers with a SANITIZED env (no PYTHONPATH, no custom VLLM_* vars), so an
    env-var opt-in is NOT readable in the worker where the FFN forward runs. So the opt-in is
    PLUGIN-PRESENCE: this entry point is only discoverable when the famp dist-info + a `.pth` (making
    `flashampere` importable without PYTHONPATH) are installed — don't install them to keep it off.
    Correctness does NOT rely on the env: the per-forward guard `_resolve_w4a8_marlin_kernel` fires
    ONLY on int8-act marlin down_proj layers and falls back to stock everywhere else, so patching the
    class on a non-W4A8 serve is a no-op at runtime. An explicit FAMP_FUSED_SILU=0 still forces off
    where the env IS readable (e.g. the main process)."""
    if (os.environ.get("FAMP_FUSED_SILU") or os.environ.get("VLLM_FAMP_FUSED_SILU") or "1") \
            not in ("1", "true", "True"):
        return

    # Eager build: compile in THIS worker now, so a build failure cleanly disables the patch
    # instead of crashing the first request (get_fused_silu_int8 is @functools.cache, so this is
    # a one-time compile per process; concurrent TP workers serialize on ninja's build lock).
    try:
        from flashampere.fused_silu_int8.build import get_fused_silu_int8

        get_fused_silu_int8()
    except Exception as e:  # noqa: BLE001 — any build/import failure must fall back to stock.
        logger.warning("fused_silu build failed, staying stock: %r", e)
        return

    # Patch the dense MLP class(es). Qwen3.6-27B dense -> Qwen2MoeMLP (qwen3_5_text Qwen3NextMLP
    # alias); Qwen2/Qwen3 -> Qwen2MLP (== Qwen3MLP, qwen3.py:52 alias); qwen3_moe arch ->
    # Qwen3MoeMLP (distinct class, NOT an alias). All MoE-MLP classes carry expert_gate and are
    # also used as shared experts; the patched forward bails to stock when expert_gate is set, so
    # patching the class is safe for both roles. Qwen2MLP has no expert_gate (always dense).
    targets = []
    try:
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

        targets.append(Qwen2MoeMLP)
    except Exception:  # noqa: BLE001
        pass
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

        targets.append(Qwen3MoeMLP)
    except Exception:  # noqa: BLE001
        pass
    try:
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        targets.append(Qwen2MLP)
    except Exception:  # noqa: BLE001
        pass

    patched = []
    for cls in targets:
        if getattr(cls.forward, "_famp_fused_silu", False):   # idempotent across processes
            continue
        cls.forward = _make_patched_forward(cls.__dict__["forward"])
        patched.append(cls.__name__)

    if patched:
        logger.info(
            "fused_silu plugin: patched %s.forward (VLLM_FAMP_FUSED_SILU=1, "
            "VLLM_MARLIN_INPUT_DTYPE=int8). Dense W4A8 FFN fuses silu*mul+int8-quant -> "
            "down_proj marlin GEMM (prefill lever; decode M=1 also runs the fused path but is "
            "bit-exact). Shared experts (expert_gate set), non-W4A8-int8 and biased down_proj "
            "layers fall back to stock per-MLP.",
            ", ".join(patched),
        )
    else:
        logger.info("fused_silu plugin: no dense-MLP class to patch (or already patched).")
