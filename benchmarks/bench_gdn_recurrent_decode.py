"""Micro-bench the GatedDeltaNet recurrent DECODE kernel and sweep its launch constants.

`fused_recurrent_gated_delta_rule_packed_decode_kernel` in vLLM's vendored fla hardcodes
`num_warps=1, num_stages=3, BV=min(npo2(V),32)` with NO autotune. The decode-share sweep showed
this path is ~21.6% of non-comm compute at batch 64 (W4A8 Qwen3.5-9B) → a faster recurrent kernel
is a real high-batch decode win. This sweeps BV × num_warps × num_stages per batch, times with CUDA
events, and verifies each config's output matches the default (perf params must not change numerics).

Standalone: imports the real vendored kernel, no vLLM engine. Run on the target arch (sm_86 / sm_80).
    python bench_gdn_recurrent_decode.py --batches 1,16,32,64
"""
import argparse
import torch
import triton
from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel as KERN,
)

# Qwen3.5-9B GatedDeltaNet shapes
H, HV, K, V = 16, 32, 128, 128          # k/q heads, v heads, head_k_dim, head_v_dim
QKV_DIM = H * K + H * K + HV * V        # packed q + k + v = 8192
SCALE = K ** -0.5
DEFAULT = (32, 1, 3)                    # (BV, num_warps, num_stages) — the hardcoded values


def build(B, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    return dict(
        mixed_qkv=torch.randn(B, QKV_DIM, dtype=torch.bfloat16, device=dev, generator=g),
        a=torch.randn(B, HV, dtype=torch.bfloat16, device=dev, generator=g),
        b=torch.rand(B, HV, dtype=torch.bfloat16, device=dev, generator=g),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev, generator=g),
        dt_bias=torch.ones(HV, dtype=torch.float32, device=dev),
        state0=torch.randn(B, HV, V, K, dtype=torch.float32, device=dev, generator=g),
        ssm_idx=torch.arange(B, dtype=torch.int32, device=dev),
    )


def launch(cfg, t, state, out):
    BV, nw, ns = cfg
    BK = triton.next_power_of_2(K)
    grid = (triton.cdiv(V, BV), t["mixed_qkv"].shape[0] * HV)
    KERN[grid](
        mixed_qkv=t["mixed_qkv"], a=t["a"], b=t["b"], A_log=t["A_log"], dt_bias=t["dt_bias"],
        o=out, h0=state, ht=state, ssm_state_indices=t["ssm_idx"], scale=SCALE,
        stride_mixed_qkv_tok=t["mixed_qkv"].stride(0), stride_a_tok=t["a"].stride(0),
        stride_b_tok=t["b"].stride(0), stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0), stride_indices_seq=t["ssm_idx"].stride(0),
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=False, num_warps=nw, num_stages=ns,
    )


def run_once(cfg, t):
    """One launch on a fresh state copy -> (out, final_state) for correctness comparison."""
    state = t["state0"].clone()
    out = torch.empty(t["mixed_qkv"].shape[0], 1, HV, V, dtype=torch.bfloat16, device="cuda")
    launch(cfg, t, state, out)
    torch.cuda.synchronize()
    return out, state


def time_cfg(cfg, t, iters=300):
    state = t["state0"].clone()
    out = torch.empty(t["mixed_qkv"].shape[0], 1, HV, V, dtype=torch.bfloat16, device="cuda")
    for _ in range(15):  # warmup (triton compile for this BV/warps/stages)
        launch(cfg, t, state, out)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        launch(cfg, t, state, out)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000.0  # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,16,32,64")
    ap.add_argument("--bv", default="16,32,64,128")
    ap.add_argument("--warps", default="1,2,4,8")
    ap.add_argument("--stages", default="1,2,3,4")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",")]
    bvs = [int(x) for x in a.bv.split(",")]
    warps = [int(x) for x in a.warps.split(",")]
    stages = [int(x) for x in a.stages.split(",")]
    print(f"GPU: {torch.cuda.get_device_name()}  |  default cfg (BV,warps,stages)={DEFAULT}")

    for B in batches:
        t = build(B)
        ref_out, ref_state = run_once(DEFAULT, t)
        d_us = time_cfg(DEFAULT, t)
        results = []
        for bv in bvs:
            for nw in warps:
                for ns in stages:
                    cfg = (bv, nw, ns)
                    try:
                        o, _ = run_once(cfg, t)
                        md = (o.float() - ref_out.float()).abs().max().item()
                        if md > 0.05:  # config changed numerics -> reject
                            continue
                        us = time_cfg(cfg, t)
                        results.append((us, cfg, md))
                    except Exception:
                        continue
        results.sort()
        best_us, best_cfg, best_md = results[0]
        print(f"\n  batch={B:3d}  default={d_us:7.2f}us  "
              f"best={best_us:7.2f}us  cfg(BV,W,S)={best_cfg}  speedup={d_us/best_us:.2f}x  (maxdiff {best_md:.1e})")
        for us, cfg, md in results[:5]:
            tag = "  <-default" if cfg == DEFAULT else ""
            print(f"      {us:7.2f}us  BV={cfg[0]:>3} warps={cfg[1]} stages={cfg[2]}  {d_us/us:.2f}x{tag}")
    print("\nDONE")


if __name__ == "__main__":
    main()
