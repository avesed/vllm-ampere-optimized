import math, os, torch, torch.nn.functional as F, flashinfer
torch.manual_seed(0); dev="cuda"; D=128
def run(L,H,causal,Hkv=None):
    Hkv=Hkv or H
    q=torch.randn(L,H,D,device=dev,dtype=torch.float16)*0.5
    k=torch.randn(L,Hkv,D,device=dev,dtype=torch.float16)*0.5
    v=torch.randn(L,Hkv,D,device=dev,dtype=torch.float16)*0.5
    sm=1/math.sqrt(D)
    Or=flashinfer.single_prefill_with_kv_cache(q,k,v,causal=causal,backend="fa2",pos_encoding_mode="NONE",sm_scale=sm)
    def qpt(x):
        s=x.abs().amax()/127.0; return torch.clamp(torch.round(x/s),-127,127).to(torch.int8),float(s)
    km=k.float().mean(0,keepdim=True); ksm=(k.float()-km).to(torch.float16)
    qi,sq=qpt(q.float()); ki,sk=qpt(ksm); vi,sv=qpt(v.float())
    Oi=flashinfer.single_prefill_with_kv_cache(qi,ki,vi,causal=causal,backend="fa2",o_dtype=torch.float16,
        pos_encoding_mode="NONE",sm_scale=sm*sq*sk*256.0)
    a=Oi.flatten().float(); b=Or.flatten().float()
    fin=torch.isfinite(a)&torch.isfinite(b)
    cos=F.cosine_similarity(a[fin],b[fin],dim=0).item() if fin.any() else float('nan')
    print(f"L={L:4d} H={H} Hkv={Hkv} causal={int(causal)}: cos={cos:.5f} finite={fin.float().mean():.3f} {'OK' if cos>0.95 else 'FAIL'}")
for L in [16,32,64,128,256,512]:
    run(L,8,True)
run(16,8,False); run(33,8,True); run(200,8,True)
run(256,8,True,Hkv=2)   # GQA group 4
run(128,4,True,Hkv=1)   # GQA group 4
