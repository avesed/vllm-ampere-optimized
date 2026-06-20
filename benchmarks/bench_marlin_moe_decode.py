"""Marlin MoE decode baseline — the kernel the W4A16 MoE actually uses by default. Compare to the
wna16 path (bench_moe_wna16_decode.py) to settle "force non-Marlin?" : 35B-A3B shapes E256/N512/K2048/topk8/g128."""
import torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
from vllm.scalar_type import scalar_types

E, N, K, TOPK, G = 256, 512, 2048, 8, 128
SHARD = 2 * N
DT = torch.bfloat16
BW = 936e9
PER_EXPERT_BYTES = SHARD * (K // 2) + K * (N // 2)

def marlin_q_moe(w_bf16, g):          # w_bf16 [e, n, k]
    wl, sl = [], []
    for i in range(w_bf16.shape[0]):
        w_t = w_bf16[i].T.contiguous()
        _, wq, ws, _, _, _ = marlin_quantize(w_t, scalar_types.uint4b8, g, act_order=False)
        wl.append(wq); sl.append(ws)
    return torch.stack(wl), torch.stack(sl)

print("quantizing 256 experts to Marlin (setup)...")
w1b = (torch.randn(E, SHARD, K, dtype=DT, device="cuda") * 0.1)
w1m, w1s = marlin_q_moe(w1b, G); del w1b; torch.cuda.empty_cache()
w2b = (torch.randn(E, K, N, dtype=DT, device="cuda") * 0.1)
w2m, w2s = marlin_q_moe(w2b, G); del w2b; torch.cuda.empty_cache()

def bench(M, iters=200):
    x = torch.randn(M, K, dtype=DT, device="cuda")
    gate = torch.randn(M, E, dtype=torch.float32, device="cuda")
    tw, ti, _ = fused_topk(x, gate, TOPK, renormalize=True)
    call = lambda: fused_marlin_moe(hidden_states=x, w1=w1m, w2=w2m, bias1=None, bias2=None,
        w1_scale=w1s, w2_scale=w2s, topk_weights=tw, topk_ids=ti,
        quant_type_id=scalar_types.uint4b8.id, global_num_experts=E, input_dtype=DT, is_k_full=True)
    for _ in range(12): call()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): call()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000.0

print(f"GPU {torch.cuda.get_device_name()} | E={E} N={N} K={K} topk={TOPK} g={G}")
WNA16 = {1:262.09, 8:701.74, 16:1284.09, 32:1957.01, 64:2780.65}
for M in [1, 8, 16, 32, 64]:
    us = bench(M)
    act = min(E, M * TOPK)
    floor = (act * PER_EXPERT_BYTES + M * K * 2) / BW * 1e6
    print(f"  M={M:3d}  marlin_moe={us:8.2f}us  | wna16={WNA16[M]:8.2f}us  | roofline≈{floor:7.2f}us  | marlin {WNA16[M]/us:.1f}x faster than wna16, {us/floor:.1f}x roofline")
print("DONE")
