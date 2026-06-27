"""Pre-serve correctness gate: FampMarlinKernel output == stock MarlinLinearKernel output, BIT-EXACT.

Same C kernels (famp_marlin.so re-uses the identical Marlin sources as _C), same MPLinearLayerConfig,
same loaded weights -> exact equality is the right bar (NOT allclose). If it ever diverges by even a
ULP, an op was routed wrong or an arg (is_a_8bit / wtype.id / input_global_scale) mismatched -> fail.

Both kernels MUTATE layer params in-place during process_weights_after_loading (repack), so we build a
FRESH layer per kernel from the SAME seeded raw tensors (the int4 weight_type path: UNPACKED [N,K]
signed codes; transform_w_q does the +8&0xF nibble-repack itself). The comparison is famp-vs-stock on
the SAME post-load weights, so any layout choice cancels — this validates the OP ROUTING, not a
reference matmul. The on-GPU GSM8K (SERVE_BENCH.md) is the end-to-end gate.

Cases:
  - W4A8-int8 g32  : num_groups=128>1 -> exercises marlin_act_int8_process_scales + input_global_scale
                     (the most fragile leg) + per_token_quant_int8 + the famp marlin_gemm a_scales arg.
  - W4A16     bf16 : act_type=bf16 -> is_a_8bit False, a_scales None, plain int4-weight gemm.
  - W4A8-int8 g128 : second grouped int8 case (different scale layout / num_groups).
  - W4A16 GPTQ-sym : weight_type=uint4b8 (GPTQ packed int32) -> exercises transform_w_q's ELSE branch
                     (permute_param_layout_ -> famp gptq_marlin_repack on PACKED int32 input), which
                     the int4-sym path NEVER reaches (it has its own nibble-pack). Closes the
                     repack-route coverage gap for the AWQ/GPTQ-packed layout.
  M=17 forces a non-tile-aligned size_m (catches padding / atomic-add bugs).

Out of test scope (rely on the byte-identical-sources argument + the GSM8K serve gate): (1) the asym
zero_points=True leg (marlin_zero_points/unpack_cols + w_zp into the gemm) — a faithful synthetic AWQ
qzeros layout can't be validated offline, and the ops swapped (repack/gemm) are already covered here;
(2) act-order has_g_idx=True (g_idx_sort wiring); (3) fp8 activations (unreachable on Ampere;
get_marlin_input_dtype rejects fp8 < SM89). All three are byte-identical to stock by inspection.

Standalone vLLM marlin layers need a current VllmConfig context + a 1-rank TP group (PackedvLLMParameter
queries the TP rank). Mirrors fused_silu_int8/test_mlp_equiv.py.

Run on the GPU box (sm80/sm86) AFTER building famp_marlin.so:
    cd /home/trevor/vllm-ampere-optimized
    python -m flashampere.marlin.build            # -> flashampere/marlin/build/famp_marlin.so
    python -m flashampere.marlin.test_kernel_equiv
Exit 0 + 'EQUIV_OK ALL' on success; raises AssertionError otherwise.
"""
import torch

from vllm.scalar_type import scalar_types
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)

from flashampere.marlin.build import get_famp_marlin
from flashampere.marlin.kernel import FampMarlinKernel

DEV = "cuda"
PACK_FACTOR = 8  # int4 -> 8 codes per int32


def _noop_loader(*args, **kwargs):
    return None


def _make_layer(K, N, group_size, seed, weight_type, zero_points):
    """Fresh layer matching how the real quant layers register weights, so the kernel's transform path
    for this (weight_type, zero_points) is exercised end-to-end.

    Seeded so stock and famp see byte-identical raw weights. Each kernel mutates these in place during
    process_weights_after_loading, hence one fresh layer per kernel.

    int4-sym  : weight stored as UNPACKED [N,K] signed codes [-8,7] int32 (transform_w_q int4 branch
                does the +8&0xF nibble-repack itself).
    uint4b8   : GPTQ-sym checkpoint layout (qweight PackedvLLMParameter input_dim=0/output_dim=1/
                packed_dim=0, shape [K//8, N] int32, exactly as auto_gptq.py:377) -> exercises
                transform_w_q's ELSE branch (permute_param_layout_ + gptq_marlin_repack on PACKED
                int32), which the int4-sym path NEVER reaches. zero_points stays False (GPTQ-sym).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    eff_group = group_size if group_size != -1 else K
    num_groups = K // eff_group

    layer = torch.nn.Module()

    if weight_type == scalar_types.int4:
        # raw SIGNED int4 codes [-8,7] as int32 [N=out, K=in]
        w_codes = torch.randint(-8, 8, (N, K), generator=g, dtype=torch.int32).to(DEV)
        layer.register_parameter("weight_packed", PackedvLLMParameter(
            input_dim=1, output_dim=0, weight_loader=_noop_loader,
            packed_factor=PACK_FACTOR, packed_dim=1, data=w_codes,
        ))
        ws = (torch.rand((N, num_groups), generator=g) * 0.02 + 0.001).to(torch.bfloat16).to(DEV)
        if group_size == -1:
            ws_param = ChannelQuantScaleParameter(
                output_dim=0, weight_loader=_noop_loader, data=ws,
            )
        else:
            ws_param = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, weight_loader=_noop_loader, data=ws,
            )
        layer.register_parameter("weight_scale", ws_param)
    else:
        # GPTQ-sym uint4b8: qweight [K//8, N] int32 packed along K (input_dim, packed_dim=0); scales
        # [num_groups, N]. Any random int32 bits are valid packed uint4 codes for repack+gemm parity.
        assert group_size != -1, "GPTQ-sym else-branch test uses a grouped layout"
        qweight = torch.randint(
            -(2**31), 2**31, (K // PACK_FACTOR, N), generator=g, dtype=torch.int64,
        ).to(torch.int32).to(DEV)
        layer.register_parameter("weight_packed", PackedvLLMParameter(
            input_dim=0, output_dim=1, weight_loader=_noop_loader,
            packed_factor=PACK_FACTOR, packed_dim=0, data=qweight,
        ))
        ws = (torch.rand((num_groups, N), generator=g) * 0.02 + 0.001).to(torch.bfloat16).to(DEV)
        layer.register_parameter("weight_scale", GroupQuantScaleParameter(
            input_dim=0, output_dim=1, weight_loader=_noop_loader, data=ws,
        ))

    layer.register_parameter("weight_shape", BasevLLMParameter(
        data=torch.tensor([K, N], dtype=torch.int64, device=DEV), weight_loader=_noop_loader,
    ))
    layer.to(DEV)
    return layer


def _run_kernel(Kernel, cfg, x, seed):
    layer = _make_layer(
        cfg.partition_weight_shape[0], cfg.partition_weight_shape[1], cfg.group_size, seed,
        cfg.weight_type, cfg.zero_points,
    )
    k = Kernel(
        cfg,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point" if cfg.zero_points else None,
        w_gidx_param_name="weight_g_idx",
    )
    k.process_weights_after_loading(layer)
    with torch.no_grad():
        return k.apply_weights(layer, x.clone(), bias=None)


def _check(label, K, N, M, act_type, group_size, x_dtype, seed=1234,
           weight_type=scalar_types.int4, zero_points=False):
    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=act_type,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=False,
    )
    x = torch.randn((M, K), dtype=x_dtype, device=DEV)
    s = _run_kernel(MarlinLinearKernel, cfg, x, seed)
    f = _run_kernel(FampMarlinKernel, cfg, x, seed)
    assert s.shape == f.shape and s.dtype == f.dtype, (
        f"{label}: shape/dtype mismatch stock={tuple(s.shape)}/{s.dtype} famp={tuple(f.shape)}/{f.dtype}"
    )
    assert torch.equal(s, f), (
        f"{label}: famp != stock; max|d|={(s.float() - f.float()).abs().max().item():.3e}"
    )
    print(f"EQUIV_OK {label} M={M} shape={tuple(s.shape)} dtype={s.dtype}")


def run_all():
    K, N, M = 4096, 4096, 17  # M=17 -> non-tile-aligned size_m
    get_famp_marlin()  # build/load famp_marlin.so
    _check("W4A8-int8 g32 ", K, N, M, torch.int8, 32, torch.bfloat16)
    _check("W4A16    bf16 ", K, N, M, torch.bfloat16, 128, torch.bfloat16)
    _check("W4A8-int8 g128", K, N, M, torch.int8, 128, torch.bfloat16)
    _check("W4A16 GPTQ-sym", K, N, M, torch.bfloat16, 128, torch.bfloat16,
           weight_type=scalar_types.uint4b8, zero_points=False)
    print("EQUIV_OK ALL")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "GPU (sm80/sm86) required"
    import torch.distributed as dist
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method="tcp://127.0.0.1:12401", local_rank=0,
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        run_all()
    print("ALL EQUIVALENCE CHECKS PASSED — FampMarlinKernel is bit-exact vs stock MarlinLinearKernel")
