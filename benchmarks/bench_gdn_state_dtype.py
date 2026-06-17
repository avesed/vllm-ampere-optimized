"""Perf prong of the bf16-state lever: does halving the GDN recurrent state (fp32->bf16) actually
~2x the bandwidth-bound decode kernel? The wrapper doesn't validate state dtype and the triton
kernel's tl.load infers dtype from the pointer, so we can test bf16 state with NO kernel edit.
Times default launch config (BV32/W1/S3) with fp32 vs bf16 state across batches. (Accuracy is a
SEPARATE e2e gate — this only answers 'is the speedup real'.)"""
import torch, triton
from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel as KERN)

H, HV, K, V = 16, 32, 128, 128
QKV_DIM = H*K + H*K + HV*V
SCALE = K ** -0.5
BV, NW, NS = 32, 1, 3

def build(B, sdt, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    return dict(
        mixed_qkv=torch.randn(B, QKV_DIM, dtype=torch.bfloat16, device=dev, generator=g),
        a=torch.randn(B, HV, dtype=torch.bfloat16, device=dev, generator=g),
        b=torch.rand(B, HV, dtype=torch.bfloat16, device=dev, generator=g),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev, generator=g),
        dt_bias=torch.ones(HV, dtype=torch.float32, device=dev),
        state0=torch.randn(B, HV, V, K, dtype=sdt, device=dev, generator=g),
        ssm_idx=torch.arange(B, dtype=torch.int32, device=dev))

def launch(t, state, out):
    grid = (triton.cdiv(V, BV), t["mixed_qkv"].shape[0]*HV)
    KERN[grid](mixed_qkv=t["mixed_qkv"], a=t["a"], b=t["b"], A_log=t["A_log"], dt_bias=t["dt_bias"],
        o=out, h0=state, ht=state, ssm_state_indices=t["ssm_idx"], scale=SCALE,
        stride_mixed_qkv_tok=t["mixed_qkv"].stride(0), stride_a_tok=t["a"].stride(0),
        stride_b_tok=t["b"].stride(0), stride_init_state_token=state.stride(0),
        stride_final_state_token=state.stride(0), stride_indices_seq=t["ssm_idx"].stride(0),
        H=H, HV=HV, K=K, V=V, BK=triton.next_power_of_2(K), BV=BV, SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=False, num_warps=NW, num_stages=NS)

def time_it(t, iters=300):
    state = t["state0"].clone()
    out = torch.empty(t["mixed_qkv"].shape[0],1,HV,V, dtype=torch.bfloat16, device="cuda")
    for _ in range(15): launch(t, state, out)
    torch.cuda.synchronize()
    e0,e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): launch(t, state, out)
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/iters*1000, out

print(f"GPU {torch.cuda.get_device_name()}  | state-bytes: fp32=2.0MB/seq, bf16=1.0MB/seq")
for B in [1,16,32,64,128]:
    try:
        us32,o32 = time_it(build(B, torch.float32))
        us16,o16 = time_it(build(B, torch.bfloat16))
        md = (o32.float()-o16.float()).abs().max().item()
        print(f"  batch={B:3d}  fp32-state={us32:7.2f}us  bf16-state={us16:7.2f}us  speedup={us32/us16:.2f}x  (out maxdiff fp32-vs-bf16-state={md:.2e})")
    except Exception as e:
        print(f"  batch={B:3d}  EXC: {type(e).__name__}: {str(e)[:120]}")
print("DONE")
