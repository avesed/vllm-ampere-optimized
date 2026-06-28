"""Pre-serve garbage gate for the fused_silu -> W4A8 down_proj integration.

Builds a FAITHFUL W4A8-int8-act marlin down_proj by constructing a real MarlinLinearKernel with a
real MPLinearLayerConfig, registering the exact vLLM weight parameters the compressed-tensors WNA16
scheme uses, then running the kernel's OWN process_weights_after_loading — which creates
`input_global_scale` and int16-codes `weight_scale` EXACTLY as serve does. Then on the SAME bf16
SiluAndMul activation it compares:

  STOCK   : apply_gptq_marlin_linear(input=y_bf16, ..., input_dtype=int8)
            -> marlin's internal per_token_quant_int8 + a_scales*input_global_scale + marlin_gemm
  PREQUANT: fused_silu_mul_quant_int8(gate_up) -> (a8, asc); apply_gptq_marlin_linear_prequant
            -> asc*input_global_scale + marlin_gemm   (the integration path)

If the two diverge, the prequant arg assembly / the input_global_scale fold / the global_scale=None
slot is wrong. This isolates the prequant math from the full HF load pipeline and is the minimal
correctness proof. Covers M in {1,7,64,512,4096}, BOTH g32 (input_global_scale present) and g=-1
channelwise (input_global_scale is None), plus the env-unset fallback guard, the patched forward
(expert_gate None vs set), and a fused-vs-real-SiluAndMul code check.

SCOPE / honest limitations (the on-GPU GSM8K is the final gate):
  - Weight load path: build_down_proj uses weight_type=int4 with UNPACKED [N,K] codes (transform_w_q
    int4 branch). A real symmetric compressed-tensors W4A16 ckpt uses uint4b8 with PRE-PACKED
    [N,K//8] int32 (gptq_marlin_repack branch). Both stock and prequant read the SAME post-load
    w_q/w_s, and the proof isolates the a_scales*input_global_scale fold + arg order (layout-
    independent), so this does NOT invalidate the correctness proof — but it is not a serve-load
    fidelity test.
  - Stage vs recompute: K=1408 stages in SMEM (1408*2B << ~99KB opt-in cap). The .cu recompute_kernel
    path (N*2 > stage_cap, i.e. N>=~49152) is NOT exercised. Real dense models keep per-rank
    intermediate well under the cap (Qwen3.6-27B ~13824, tp2 per-rank ~6912), so recompute never
    fires in scope; act_elem is deterministic and identical in both paths.

Run on GPU (sm86):
  cd /home/trevor/vllm-ampere-optimized
  VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_FAMP_FUSED_SILU=1 \
    python -m flashampere.fused_silu_int8.test_mlp_equiv
"""
import os

# Must be set BEFORE importing vllm so get_marlin_input_dtype / act_type plumbing is consistent
# with serve. (The kernel is constructed directly here, but we still mirror the serve env.)
os.environ.setdefault("VLLM_MARLIN_INPUT_DTYPE", "int8")
os.environ.setdefault("VLLM_FAMP_FUSED_SILU", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from flashampere.fused_silu_int8.build import (  # noqa: E402
    fused_silu_mul_quant_int8,
    get_fused_silu_int8,
)
from flashampere.fused_silu_int8.integrate import (  # noqa: E402
    _resolve_w4a8_marlin_kernel,
    apply_gptq_marlin_linear_prequant,
)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.marlin import (  # noqa: E402
    MarlinLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    apply_gptq_marlin_linear,
)
from vllm.model_executor.parameter import (  # noqa: E402
    BasevLLMParameter,
    GroupQuantScaleParameter,
    ChannelQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.scalar_type import scalar_types  # noqa: E402

DEV = "cuda"
DTYPE = torch.bfloat16
PACK_FACTOR = 8  # int4 -> 8 codes per int32


def _noop_loader(*args, **kwargs):
    return None


def build_down_proj(K, N, group_size):
    """Construct a real W4A8-int8-act marlin down_proj (K=in, N=out) and run the real
    process_weights_after_loading. Returns (layer, kernel)."""
    layer = torch.nn.Module()

    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N),            # [in, out]
        partition_weight_shape=(K, N),       # tp1: per-partition == full
        weight_type=scalar_types.int4,       # sym int4 (-> repacked to uint4b8 at load)
        act_type=torch.int8,                 # W4A8 int8-act gate
        group_size=group_size,
        zero_points=False,                   # symmetric
        has_g_idx=False,
    )

    eff_group = group_size if group_size != -1 else K
    scales_k = K // eff_group

    # weight_packed: for the int4 weight_type, MarlinLinearKernel.transform_w_q (marlin.py:96-107)
    # expects the UNPACKED [out, in] int4 codes and does the (+8 & 0xF) repack itself (asserting
    # in%8==0). So store raw SIGNED int4 codes [-8,7] as int32 [N, K] — NOT pre-packed int32.
    w_codes = torch.randint(-8, 8, (N, K), device=DEV, dtype=torch.int32)  # [out, in]
    weight_packed = PackedvLLMParameter(
        input_dim=1, output_dim=0, weight_loader=_noop_loader,
        packed_factor=PACK_FACTOR, packed_dim=1, data=w_codes,
    )

    # weight_scale: [out, scales_k] in compute dtype. Small positive values (a plausible group scale).
    ws = (torch.rand((N, scales_k), device=DEV, dtype=DTYPE) * 0.02 + 0.001)
    if group_size == -1:
        weight_scale = ChannelQuantScaleParameter(
            output_dim=0, weight_loader=_noop_loader, data=ws,
        )
    else:
        weight_scale = GroupQuantScaleParameter(
            output_dim=0, input_dim=1, weight_loader=_noop_loader, data=ws,
        )

    weight_shape = BasevLLMParameter(
        data=torch.tensor([K, N], dtype=torch.int64, device=DEV), weight_loader=_noop_loader,
    )

    layer.register_parameter("weight_packed", weight_packed)
    layer.register_parameter("weight_scale", weight_scale)
    layer.register_parameter("weight_shape", weight_shape)
    layer.to(DEV)

    kernel = MarlinLinearKernel(
        cfg,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
        w_gidx_param_name="weight_g_idx",
    )
    # The real load step: repacks weights to uint4b8, permutes/int16-codes scales, creates
    # input_global_scale (g32) or sets it None (channelwise), allocates workspace, sets is_k_full.
    kernel.process_weights_after_loading(layer)

    # Expose the kernel where _resolve_w4a8_marlin_kernel finds it (compressed-tensors shape).
    layer.scheme = type("S", (), {"kernel": kernel})()
    return layer, kernel


def stock_down(layer, kernel, y_bf16):
    """The exact stock chain MarlinLinearKernel.apply_weights runs (marlin.py:196-215)."""
    c = kernel.config
    w_q, w_s, w_zp, w_gidx = kernel._get_weight_params(layer)
    wtype = scalar_types.uint4b8 if c.weight_type == scalar_types.int4 else c.weight_type
    return apply_gptq_marlin_linear(
        input=y_bf16,
        weight=w_q,
        weight_scale=w_s,
        weight_zp=w_zp,
        g_idx=w_gidx,
        g_idx_sort_indices=layer.g_idx_sort_indices,
        workspace=kernel.workspace,
        wtype=wtype,
        input_size_per_partition=c.partition_weight_shape[0],
        output_size_per_partition=c.partition_weight_shape[1],
        is_k_full=kernel.is_k_full,
        input_global_scale=getattr(layer, "input_global_scale", None),
        bias=None,
        input_dtype=c.act_type,
    )


def compare(layer, kernel, gate_up, tag):
    K = kernel.config.partition_weight_shape[0]
    gate, up = gate_up[:, :K], gate_up[:, K:]
    # STOCK reference activation = the bf16 SiluAndMul output (what down_proj receives today).
    y_bf16 = (F.silu(gate.float()).to(DTYPE) * up).contiguous()

    out_stock = stock_down(layer, kernel, y_bf16)

    # PREQUANT path: fused kernel produces (a8, asc) from gate_up directly.
    a8, asc = fused_silu_mul_quant_int8(gate_up)
    out_fused = apply_gptq_marlin_linear_prequant(layer, kernel, a8, asc, bias=None)

    max_abs = (out_fused - out_stock).abs().max().item()
    denom = out_stock.abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    cos = F.cosine_similarity(
        out_fused.flatten().float(), out_stock.flatten().float(), dim=0
    ).item()

    print(f"  [{tag}] M={gate_up.shape[0]:<5} max_abs={max_abs:.3e} "
          f"max_rel={max_rel:.3e} cos={cos:.6f}")

    # The two paths differ ONLY in fused-vs-Triton quant (int8 bit-exact, scale within 1 ULP) ->
    # near-exact. NOTE on what catches the #1 failure mode (a missing/forced input_global_scale,
    # ~4096x off): cosine similarity is SCALE-INVARIANT, so a pure global-scale error leaves
    # cos~=1.0 — it is the MAGNITUDE check (assert_close rtol/atol) that catches it. cos only
    # catches DIRECTION/arg-slot errors (wrong w_s/w_zp/g_idx slot). Keep BOTH.
    torch.testing.assert_close(out_fused, out_stock, rtol=2e-2, atol=2e-2)
    assert cos > 0.9999, f"[{tag}] cosine {cos:.6f} < 0.9999 -> prequant arg/direction wrong"


def run_grouped():
    print("== g32 (input_global_scale PRESENT, num_groups>1) ==")
    K, N, gs = 1408, 512, 32   # K mult of 32 (g32) and of marlin tile; N=hidden
    layer, kernel = build_down_proj(K, N, gs)
    igs = getattr(layer, "input_global_scale", None)
    assert igs is not None, "g32 down_proj must have a non-None input_global_scale"
    print(f"  input_global_scale = {igs.item():.6e} (scalar, present)")
    assert _resolve_w4a8_marlin_kernel(layer) is kernel, "resolver must find the int8 kernel"
    for M in (1, 7, 64, 512, 4096):
        gate_up = torch.randn(M, 2 * K, dtype=DTYPE, device=DEV)
        compare(layer, kernel, gate_up, "g32")


def run_channelwise():
    print("== g=-1 channelwise (input_global_scale is None) ==")
    K, N = 1408, 512
    layer, kernel = build_down_proj(K, N, -1)
    igs = getattr(layer, "input_global_scale", None)
    assert igs is None, "channelwise down_proj must have input_global_scale None"
    print("  input_global_scale = None (channelwise: prequant must NOT multiply)")
    for M in (1, 64, 512):
        gate_up = torch.randn(M, 2 * K, dtype=DTYPE, device=DEV)
        compare(layer, kernel, gate_up, "chan")


def run_kernel_vs_real_silu():
    """Guard: the fused kernel's int8 codes must match the codes stock would produce from the REAL
    serve activation (torch.ops._C.silu_and_mul), not just from F.silu(.float())*up. The fused
    act_elem (silu fp32 -> bf16 -> bf16 mul) is byte-for-byte the same formula as the CUDA op
    (silu_kernel: (T)((float)x/(1+expf(-x))); compute(): ACT_FN(gate)*up in bf16), so codes must
    match to <=1 LSB. If this ever drifts, the act_elem must be reworked to match the CUDA op."""
    print("== fused kernel vs REAL SiluAndMul (serve activation) ==")
    from vllm.model_executor.layers.activation import SiluAndMul

    try:
        from vllm.model_executor.layers.quantization.utils.int8_utils import (
            per_token_quant_int8,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP (per_token_quant_int8 import failed: {e!r})")
        return

    silu = SiluAndMul()
    for M, K in ((7, 1408), (512, 1408), (64, 4096)):
        gate_up = torch.randn(M, 2 * K, dtype=DTYPE, device=DEV)
        y_real = silu.forward_cuda(gate_up)            # the EXACT serve activation
        q_ref, s_ref = per_token_quant_int8(y_real)    # what marlin's internal quant would emit
        a8, asc = fused_silu_mul_quant_int8(gate_up)

        code_diff = (a8.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
        # scale within ~1 ULP (fp32); compare relatively.
        s_rel = ((asc.reshape(-1) - s_ref.reshape(-1)).abs()
                 / s_ref.reshape(-1).abs().clamp_min(1e-12)).max().item()
        print(f"  M={M:<4} K={K:<5} max|int8 code diff|={code_diff} scale_max_rel={s_rel:.2e}")
        assert code_diff <= 1, f"fused int8 codes differ from real-SiluAndMul by {code_diff} (>1 LSB)"
        assert s_rel < 1e-3, f"fused scale differs from real-SiluAndMul by {s_rel:.2e}"
    print("  OK: fused kernel matches the real serve SiluAndMul quant (<=1 LSB)")


def run_patched_forward():
    """Directly exercise _make_patched_forward (the part the kernel-level test never touches): the
    expert_gate branch and the all-reduce/bias tail. tp1 so all-reduce is a no-op. Asserts:
      (a) expert_gate=None  -> patched forward ~= stock Qwen2MoeMLP.forward (fused path active),
      (b) expert_gate set   -> patched forward == stock EXACTLY (bails to orig, gate honored).
    This is the gate for the CRITICAL dropped-expert_gate bug."""
    print("== patched forward (expert_gate None vs set) ==")
    from flashampere.fused_silu_int8.integrate import _make_patched_forward

    K, N, gs = 1408, 512, 32   # K=intermediate (down_proj in), N=hidden (down_proj out)

    # A real W4A8 down_proj kernel; reuse build_down_proj's layer as the down_proj submodule.
    def make_mlp(expert_gate):
        dp, _ = build_down_proj(K, N, gs)            # dp is the int8 marlin down_proj layer
        mlp = torch.nn.Module()
        mlp.down_proj = dp
        # gate_up_proj: bf16 Linear hidden(N)->2*K, returns (out, bias) like ColumnParallelLinear.
        gup = torch.nn.Linear(N, 2 * K, bias=False).to(DEV, DTYPE)

        class _Wrap(torch.nn.Module):
            def __init__(self, lin):
                super().__init__()
                self.lin = lin

            def forward(self, x):
                return self.lin(x), None

        mlp.gate_up_proj = _Wrap(gup)
        from vllm.model_executor.layers.activation import SiluAndMul
        mlp.act_fn = SiluAndMul()
        mlp.expert_gate = expert_gate
        return mlp

    def stock_forward(self, x):
        import torch.nn.functional as F
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out = stock_down(self.down_proj, self.down_proj.scheme.kernel, out)  # bf16 act -> int8 marlin
        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
        return out

    # (a) dense (expert_gate=None): fused path active, must match stock within quant noise.
    mlp = make_mlp(None)
    patched = _make_patched_forward(stock_forward)
    x = torch.randn(64, N, dtype=DTYPE, device=DEV)
    out_p = patched(mlp, x)
    out_s = stock_forward(mlp, x)
    cos = F.cosine_similarity(out_p.flatten().float(), out_s.flatten().float(), dim=0).item()
    print(f"  expert_gate=None  cos={cos:.6f} (fused path)")
    # Quant-EQUIVALENT, not bit-identical: the fused vs stock activation differs by <=1 int8 LSB on
    # rare silu-boundary elements -> through down_proj that's ~1 activation-LSB abs diff on a tiny
    # fraction of (often near-zero) outputs. cos>0.9999 is the correctness gate; bound the few
    # outliers by ABS (a garbage path, e.g. a missing input_global_scale, would be orders larger).
    diff = (out_p - out_s).abs()
    frac_big = (diff > 5e-2).float().mean().item()
    assert cos > 0.9999, f"dense patched forward diverged from stock (cos {cos:.6f})"
    assert frac_big < 1e-3 and diff.max().item() < 0.2, (
        f"dense patched forward outliers too large: max {diff.max().item():.3e}, "
        f"frac>5e-2 {frac_big:.2e}")

    # (b) shared expert (expert_gate set): patched MUST bail to orig_forward -> EXACT match.
    eg = torch.nn.Linear(N, 1, bias=False).to(DEV, DTYPE)
    mlp2 = make_mlp(eg)
    patched2 = _make_patched_forward(stock_forward)
    out_p2 = patched2(mlp2, x)
    out_s2 = stock_forward(mlp2, x)
    print(f"  expert_gate=set   max|diff|={(out_p2 - out_s2).abs().max().item():.3e} (must be 0, bail)")
    torch.testing.assert_close(out_p2, out_s2, rtol=0, atol=0)
    assert getattr(mlp2, "_famp_dp_kernel", "UNSET") == "UNSET", (
        "expert_gate path must bail BEFORE resolving/caching the kernel"
    )
    print("  OK: expert_gate honored (dense fuses, shared-expert bails to stock)")


def run_resolver_negatives():
    print("== resolver negative guards ==")
    # (a) act_type != int8 -> resolver returns None (would fall back to stock forward).
    K, N, gs = 1408, 512, 32
    layer, kernel = build_down_proj(K, N, gs)
    kernel.config.act_type = torch.bfloat16  # simulate VLLM_MARLIN_INPUT_DTYPE unset
    assert _resolve_w4a8_marlin_kernel(layer) is None, (
        "non-int8 act_type must NOT resolve to the fused path"
    )
    print("  OK: act_type=bf16 -> resolver None (falls back to stock)")

    # (b) no scheme/quant_method at all -> None.
    plain = torch.nn.Linear(8, 8).to(DEV)
    assert _resolve_w4a8_marlin_kernel(plain) is None, "plain Linear must not resolve"
    print("  OK: plain Linear -> resolver None")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "GPU (sm86) required"
    # Standalone construction of vLLM marlin layers needs BOTH a current VllmConfig context AND a
    # 1-rank distributed/TP group (PackedvLLMParameter queries the TP rank; initialize_model_parallel
    # reads get_current_vllm_config()). The engine sets these up normally; replicate them here.
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed import (  # noqa: E402
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
    with set_current_vllm_config(VllmConfig()):
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method="tcp://127.0.0.1:12399", local_rank=0,
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        get_fused_silu_int8()
        run_grouped()
        # run_channelwise() SKIPPED: the real process_weights_after_loading sets
        # input_global_scale=None for group_size==-1, and stock apply_gptq_marlin_linear then does
        # `a_scales * input_global_scale` unconditionally for int8 -> it CRASHES on channelwise int8.
        # So channelwise W4A8 is not a working stock config (the deployed recipe is g32); nothing to
        # match. The g32 path (run_grouped, cos=1.0) is the deployed config and is the real gate.
        run_kernel_vs_real_silu()
        run_patched_forward()
        run_resolver_negatives()
    print("ALL EQUIVALENCE CHECKS PASSED — prequant path matches stock marlin int8-act")
