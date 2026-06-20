"""Decode-shape micro-bench of the NON-Marlin W4A16 MoE path (fused_experts -> wna16 cuda/triton),
to test the "force Triton/non-Marlin MoE" hypothesis on Ampere: is decode already bandwidth-roofline
-capped (→ forcing off Marlin = parity = no lever), or is there headroom? Mirrors benchmark_moe.py's
int4_w4a16 synth; no ray. Shapes = Qwen3.6-35B-A3B MoE (E=256, N=512, K=2048, topk=8, group=128)."""
import torch
from vllm.model_executor.layers.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

E, N, K, TOPK, G = 256, 512, 2048, 8, 128
SHARD = 2 * N                  # gate_up combined
DT = torch.bfloat16
BW = 936e9                     # RTX 3090 GDDR6X ~GB/s
PER_EXPERT_BYTES = SHARD * (K // 2) + K * (N // 2)   # uint8 packed int4 = bytes

def build():
    w1 = torch.randint(0, 255, (E, SHARD, K // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 255, (E, K, N // 2), dtype=torch.uint8, device="cuda")
    w1s = torch.rand((E, SHARD, K // G), dtype=DT, device="cuda")
    w2s = torch.rand((E, K, N // G), dtype=DT, device="cuda")
    qc = FusedMoEQuantConfig.make(quant_dtype=None, w1_scale=w1s, w2_scale=w2s,
                                  block_shape=[0, G], weight_dtype="int4")
    return w1, w2, qc

def bench(M, w1, w2, qc, iters=200):
    x = torch.randn(M, K, dtype=DT, device="cuda")
    g = torch.randn(M, E, dtype=torch.float32, device="cuda")
    tw, ti, _ = fused_topk(x, g, TOPK, renormalize=True)
    for _ in range(12):
        fused_experts(x, w1, w2, tw, ti, quant_config=qc)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fused_experts(x, w1, w2, tw, ti, quant_config=qc)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000.0  # us

w1, w2, qc = build()
print(f"GPU {torch.cuda.get_device_name()} | E={E} N={N} K={K} topk={TOPK} g={G} | per-expert W4={PER_EXPERT_BYTES/1e6:.2f}MB")
for M in [1, 8, 16, 32, 64]:
    us = bench(M, w1, w2, qc)
    act = min(E, M * TOPK)
    rb = act * PER_EXPERT_BYTES + M * K * 2          # activated-expert W4 + activation read
    floor = rb / BW * 1e6
    print(f"  M={M:3d}  wna16={us:8.2f}us  roofline≈{floor:7.2f}us  -> {us/floor:.1f}x roofline  (act_experts≈{act})")
print("DONE")
