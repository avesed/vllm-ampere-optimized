"""Model-free correctness check for the SageAttention prefill substitution: sageattn_varlen
(int8-QK + fp16-PV, smooth_k) vs a torch-SDPA fp32 reference, at the GQA / head-dim / causal /
varlen shapes vLLM's prefill uses. PASS if relative error is small (SageAttn = "accurate 8-bit").
"""
import torch
import torch.nn.functional as F
from sageattention import sageattn_varlen


def reln(o, r):
    return (torch.mean(torch.abs(o.float() - r.float())) / torch.mean(torch.abs(r.float()))).item()


def ref_varlen_causal(q, k, v, cu, Hq, Hkv, D):
    # per-sequence torch SDPA in fp32, GQA via repeat_interleave
    outs = []
    g = Hq // Hkv
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        qi = q[s:e].permute(1, 0, 2).float()            # [Hq, L, D]
        ki = k[s:e].permute(1, 0, 2).float().repeat_interleave(g, dim=0)
        vi = v[s:e].permute(1, 0, 2).float().repeat_interleave(g, dim=0)
        o = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)  # [Hq, L, D]
        outs.append(o.permute(1, 0, 2))                  # [L, Hq, D]
    return torch.cat(outs, 0)


def run(seqlens, Hq, Hkv, D, dtype=torch.float16):
    torch.manual_seed(0)
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
                      device="cuda", dtype=torch.int32)
    T = int(cu[-1])
    q = torch.randn(T, Hq, D, device="cuda", dtype=dtype)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=dtype)
    v = torch.randn(T, Hkv, D, device="cuda", dtype=dtype)
    mql = max(seqlens)
    out = sageattn_varlen(q, k, v, cu, cu, mql, mql, is_causal=True,
                          sm_scale=D ** -0.5, smooth_k=True)
    ref = ref_varlen_causal(q, k, v, cu, Hq, Hkv, D)
    return reln(out, ref)


if __name__ == "__main__":
    cases = [
        ("Llama-8B-ish  GQA Hq32/Hkv8 D128", [512, 200, 333], 32, 8, 128),
        ("Qwen-dense    GQA Hq40/Hkv8 D128", [1024, 777], 40, 8, 128),
        ("MHA           Hq16/Hkv16 D128", [600, 600], 16, 16, 128),
        ("D64           GQA Hq16/Hkv4 D64", [800], 16, 4, 64),
    ]
    print("rel-err < 0.05 = PASS (SageAttn accurate-8bit)")
    allpass = True
    for name, sl, hq, hkv, d in cases:
        try:
            e = run(sl, hq, hkv, d)
            ok = e < 0.05
            allpass &= ok
            print(f"  {name:40s} rel-err={e:.4f}  {'PASS' if ok else 'FAIL <<<'}")
        except Exception as ex:
            allpass = False
            print(f"  {name:40s} EXC: {type(ex).__name__}: {str(ex)[:120]}")
    print("ALL_PASS" if allpass else "SOME_FAIL")
