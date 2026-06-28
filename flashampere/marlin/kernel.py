"""FampMarlinKernel — an MPLinearKernel that mirrors vLLM's stock MarlinLinearKernel EXACTLY but
routes the marlin ops (gptq_marlin_repack, marlin_gemm) to torch.ops.famp_marlin.* (the vendored
famp_marlin extension) instead of torch.ops._C. awq_marlin_repack is exposed by the famp lib too but
is NOT on this code path — transform_w_q always uses gptq_marlin_repack, even for AWQ uint4 weights
(stock does the same; awq packing is handled upstream before this kernel).

Everything except the 2 reachable op-call sites (repack in process_weights_after_loading + the gemm in
apply_weights, which is INLINED so the gemm can be rerouted) is byte-for-byte identical to stock — this
is the correctness contract; the test_kernel_equiv.py gate asserts BIT-EXACT parity vs stock.

Selected over stock Marlin for W4A8-int8 / W4A16 layers by register_fampmarlin() (the
vllm.general_plugins entry point), which INSERTS this class BEFORE MarlinLinearKernel in
_POSSIBLE_KERNELS[CUDA] so choose_mp_linear_kernel (first-match-in-order) reaches it first.
"""
import logging
import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    USE_FP32_REDUCE_DEFAULT,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_quant_input,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    should_use_atomic_add_reduce,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)

from flashampere.marlin.build import get_famp_marlin

logger = logging.getLogger(__name__)


class FampMarlinKernel(MPLinearKernel):
    """Mirror of MarlinLinearKernel, but the marlin ops come from torch.ops.famp_marlin.*.

    NO __init__ / is_supported override: inherits MPLinearKernel.__init__ (asserts can_implement,
    stores config + param names) and _transform_param / _get_weight_params unchanged. is_supported
    belongs to the unrelated MMLinearKernel hierarchy — the MP selector never calls it.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # MUST match stock Marlin's can_implement EXACTLY (no act_type filter) so famp is eligible on
        # precisely the same layer set as stock — required for clean selection + the equivalence gate.
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "FampMarlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  FampMarlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "FampMarlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        return check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Worker has no toolchain: the get_famp_marlin() fast path (load prebuilt .so) must run here.
        get_famp_marlin()
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            # W4A8 also covers asym AWQ (uint4 + runtime zp): the (kS8,kU4)
            # kernel folds zp into the int8 operand via sub_zp_and_dequant.
            assert c.weight_type in (
                scalar_types.uint4b8,
                scalar_types.int4,
                scalar_types.uint4,
            ), "W4A8-INT8 marlin supports uint4b8, int4, or uint4 weights."

        if c.act_type == torch.float8_e4m3fn:
            # fp8 unreachable on Ampere (get_marlin_input_dtype rejects it < SM89); kept for parity.
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            if c.weight_type == scalar_types.int4:
                w = x.data
                assert w.shape[1] % 8 == 0, (
                    f"int4 marlin: in dim {w.shape[1]} must be a multiple of 8"
                )
                w_u4 = (w.to(torch.int32) + 8) & 0xF
                w_u4 = w_u4.reshape(w.shape[0], w.shape[1] // 8, 8)
                shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
                packed = (w_u4 << shifts[None, None, :]).sum(dim=2).to(torch.int32)
                x.data = packed.T.contiguous()
            else:
                permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            # SWAP: famp repack instead of ops.gptq_marlin_repack (-> torch.ops._C). POSITIONAL only;
            # the raw torch.ops.famp_marlin schema is (b_q_weight, perm, size_k, size_n, num_bits,
            # is_a_8bit) — identical to _C (verified famp_marlin_binding.cu) but positional avoids any
            # kw-binding edge case on the raw op.
            x.data = torch.ops.famp_marlin.gptq_marlin_repack(
                x.data.contiguous(),
                layer.g_idx_sort_indices,        # perm
                c.partition_weight_shape[0],     # size_k
                c.partition_weight_shape[1],     # size_n
                c.weight_type.size_bits,         # num_bits
                is_a_8bit,                       # is_a_8bit
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        # g_idx block above sets layer.g_idx_sort_indices BEFORE transform_w_q reads it as `perm`;
        # w_q transform runs before w_s. Preserve this order (stock).
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(layer.bias)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # INLINE of apply_gptq_marlin_linear (marlin_utils.py:584-631): do NOT call that helper — it
        # hardcodes ops.marlin_gemm -> torch.ops._C.marlin_gemm, so famp would never run the gemm.
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` ensures w_zp and w_gidx are not None for marlin.
        wtype = (
            scalar_types.uint4b8
            if c.weight_type == scalar_types.int4
            else c.weight_type
        )
        input_size_per_partition = c.partition_weight_shape[0]
        output_size_per_partition = c.partition_weight_shape[1]
        input_dtype = c.act_type
        input_global_scale = getattr(layer, "input_global_scale", None)

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (output_size_per_partition,)

        use_atomic_add = should_use_atomic_add_reduce(
            m=reshaped_x.size(0),
            n=output_size_per_partition,
            k=reshaped_x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        a_scales = None
        if input_dtype == torch.int8:
            # uint4 = asym AWQ (zp via weight_zp); uint4b8 = sym GPTQ. Both are W4A8;
            # an 8-bit weight (W8A8) is still unsupported.
            assert wtype in (scalar_types.uint4b8, scalar_types.uint4), (
                "W4A8-INT8 marlin requires a 4-bit weight (uint4b8 or uint4)."
            )
            reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)
            a_scales = a_scales * input_global_scale
        elif input_dtype == torch.float8_e4m3fn:
            assert wtype == scalar_types.uint4b8, (
                "INT8 weight + FP8 activation is not supported."
            )
            reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)

        # SWAP: famp gemm instead of ops.marlin_gemm (-> torch.ops._C). POSITIONAL, 19 args, order
        # verified against famp_marlin_binding.cu + _custom_ops.py. CRITICAL: b_type_id is wtype.id
        # (the INT scalar-type id), NOT the ScalarType object (the famp schema field is `int b_type_id`,
        # mirroring _custom_ops.py:1409 b_q_type.id).
        output = torch.ops.famp_marlin.marlin_gemm(
            reshaped_x,                       # a
            None,                             # c_or_none
            w_q,                              # b_q_weight
            bias,                             # b_bias_or_none
            w_s,                              # b_scales
            a_scales,                         # a_scales
            None,                             # global_scale
            w_zp,                             # b_zeros_or_none
            w_gidx,                           # g_idx_or_none
            layer.g_idx_sort_indices,         # perm_or_none
            self.workspace,                   # workspace
            wtype.id,                         # b_type_id (INT, not ScalarType)
            reshaped_x.shape[0],              # size_m
            output_size_per_partition,        # size_n
            input_size_per_partition,         # size_k
            self.is_k_full,                   # is_k_full
            use_atomic_add,                   # use_atomic_add
            USE_FP32_REDUCE_DEFAULT,          # use_fp32_reduce
            False,                            # is_zp_float
        )

        return output.reshape(out_shape)


# ----------------------------------------------------------------------------- #
# vllm.general_plugins entry point — insert FampMarlinKernel before stock Marlin.
# ----------------------------------------------------------------------------- #
def register_fampmarlin():
    """Insert FampMarlinKernel BEFORE the stock MarlinLinearKernel in _POSSIBLE_KERNELS[CUDA].

    load_general_plugins() runs this in the engine core AND every TP/PP worker subprocess BEFORE the
    model is built (worker_base.py:245) — i.e. before create_weights / choose_mp_linear_kernel — so
    the insertion is in place for selection in every process.

    GATING is PLUGIN-PRESENCE, not an env var: this entry point is only discoverable when the famp
    dist-info + a `.pth` (so `flashampere` imports without PYTHONPATH) are installed. choose_mp_linear_
    kernel still picks famp ONLY when its can_implement() (== stock Marlin's) passes; everything else
    falls back to stock. An explicit FAMP_MARLIN=0 forces off where the env is readable (main proc).
    """
    if (os.environ.get("FAMP_MARLIN") or "1") not in ("1", "true", "True"):
        return

    try:
        from vllm import envs
        from vllm.platforms import PlatformEnum, current_platform
        if not current_platform.is_cuda():
            return
        # famp_marlin ships only the cubins it was compiled for (FAMP_MARLIN_ARCH, default the Ampere
        # sm_80+sm_86 scope). On any other GPU there is no compatible kernel image, so selecting famp
        # would fail at the first marlin_gemm — gate to the built arches here; stock _C Marlin (full
        # fatbin) serves the rest, bit-identically (famp is a byte-mirror). The release Dockerfile bakes
        # FAMP_MARLIN_ARCH as an env so this stays in sync with whatever arches the .so was built for.
        _built = {a.strip().split("+")[0]
                  for a in (os.environ.get("FAMP_MARLIN_ARCH") or "8.0,8.6").split(",") if a.strip()}
        _cap = current_platform.get_device_capability()
        _cur = f"{_cap.major}.{_cap.minor}" if _cap is not None else None
        if _cur not in _built:
            logger.info("famp_marlin: GPU sm_%s not in built arches %s; using stock Marlin.",
                        _cur, sorted(_built))
            return
        from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
        from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
            MarlinLinearKernel,
        )
    except Exception as e:  # noqa: BLE001 — any import failure must fall back to stock.
        logger.warning("famp_marlin: registration skipped (import failed): %s", e)
        return

    # Honor the standard escape hatch: choose_mp_linear_kernel skips any kernel whose __name__ is in
    # VLLM_DISABLED_KERNELS (linear/__init__.py). If an operator force-disables FampMarlinKernel,
    # don't insert at all -> selection falls through to stock MarlinLinearKernel.
    if "FampMarlinKernel" in getattr(envs, "VLLM_DISABLED_KERNELS", ()):
        logger.info("famp_marlin: FampMarlinKernel disabled via VLLM_DISABLED_KERNELS; using stock.")
        return

    kernels = _POSSIBLE_KERNELS.setdefault(PlatformEnum.CUDA, [])

    # Idempotent: load_general_plugins may run per-process; guard against double-insert.
    if FampMarlinKernel in kernels:
        return

    # Insert immediately before the EARLIEST kernel famp competes with for marlin layers. famp's
    # can_implement == stock Marlin's, so it must win over Marlin AND over any min-cap<=86 kernel
    # listed AHEAD of Marlin that could claim a W4 layer on Ampere (today only AllSparkLinearKernel,
    # which rejects int4/uint4 -> never wins; guarded here so a future rebase that gives AllSpark W4
    # support can't silently bypass famp). It does NOT jump ahead of the Hopper-only kernels
    # (CutlassW4A8/Machete, min-cap 90) so famp does not override a genuinely-better kernel on SM90+.
    # choose_mp_linear_kernel is first-match-in-order; stock Marlin stays in the list as fallback.
    _cc = current_platform.get_device_capability()
    _cc_int = _cc.to_int() if _cc is not None else 86
    compete = {MarlinLinearKernel}
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.allspark import (
            AllSparkLinearKernel,
        )
        compete.add(AllSparkLinearKernel)
    except Exception:  # noqa: BLE001 — AllSpark optional; Marlin anchor is enough.
        pass
    idx = next(
        (i for i, k in enumerate(kernels)
         if k in compete and k.get_min_capability() <= _cc_int),
        len(kernels),
    )
    kernels.insert(idx, FampMarlinKernel)
    logger.info(
        "famp_marlin: FampMarlinKernel inserted at _POSSIBLE_KERNELS[CUDA][%d] (before Marlin/AllSpark)",
        idx,
    )

    # Eager-load the .so now so a build/load failure cleanly disables famp (remove the insert) rather
    # than crashing the first request. @functools.cache makes this a no-op on subsequent layers.
    try:
        get_famp_marlin()
    except Exception as e:  # noqa: BLE001
        logger.warning("famp_marlin: .so load failed, reverting insertion: %s", e)
        kernels.remove(FampMarlinKernel)
