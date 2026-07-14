import sys, torch, flashinfer
torch.manual_seed(0)
out={}
def time_run(run, o):
    for _ in range(10): run()
    torch.cuda.synchronize()
    s,e,ts=torch.cuda.Event(True),torch.cuda.Event(True),[]
    for _ in range(50): s.record(); run(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return o.float().cpu(), ts[25]

# 1. bf16 hd256 single_prefill (EXPECT garbage when patched -> validates half-only gating need)
try:
    HD=256; q=torch.randn(2048,8,HD,dtype=torch.bfloat16,device='cuda'); k=torch.randn(2048,4,HD,dtype=torch.bfloat16,device='cuda'); v=torch.randn(2048,4,HD,dtype=torch.bfloat16,device='cuda')
    o=flashinfer.single_prefill_with_kv_cache(q,k,v,causal=True,backend='fa2'); out['bf16_hd256']=(o.float().cpu(),0.0); print("bf16_hd256 ok",tuple(o.shape))
except Exception as ex: print("bf16_hd256 ERR",repr(ex)[:120])

# 2. f16 hd128 single_prefill
try:
    HD=128; q=torch.randn(2048,16,HD,dtype=torch.float16,device='cuda'); k=torch.randn(2048,4,HD,dtype=torch.float16,device='cuda'); v=torch.randn(2048,4,HD,dtype=torch.float16,device='cuda')
    run=lambda: flashinfer.single_prefill_with_kv_cache(q,k,v,causal=True,backend='fa2'); out['f16_hd128']=time_run(run,run()); print("f16_hd128 ok")
except Exception as ex: print("f16_hd128 ERR",repr(ex)[:120])

# 3. f16 hd256 PAGED batch prefill (deployment path)
try:
    HD=256; NH_Q=16; NH_KV=4; PAGE=16; nreq=3; qlen=512; klen=2048
    npages=(klen+PAGE-1)//PAGE
    wbuf=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda')
    wr=flashinfer.BatchPrefillWithPagedKVCacheWrapper(wbuf,"NHD")
    qo_indptr=torch.tensor([0,qlen,2*qlen,3*qlen],dtype=torch.int32,device='cuda')
    paged_indptr=torch.tensor([0,npages,2*npages,3*npages],dtype=torch.int32,device='cuda')
    paged_indices=torch.arange(3*npages,dtype=torch.int32,device='cuda')
    last_len=torch.full((nreq,),((klen-1)%PAGE)+1,dtype=torch.int32,device='cuda')
    wr.plan(qo_indptr,paged_indptr,paged_indices,last_len,NH_Q,NH_KV,HD,PAGE,causal=True,q_data_type=torch.float16,kv_data_type=torch.float16)
    q=torch.randn(3*qlen,NH_Q,HD,dtype=torch.float16,device='cuda')
    kv=torch.randn(3*npages,2,PAGE,NH_KV,HD,dtype=torch.float16,device='cuda')
    run=lambda: wr.run(q,kv); out['f16_hd256_paged']=time_run(run,run()); print("f16_hd256_paged ok")
except Exception as ex: print("f16_hd256_paged ERR",repr(ex)[:160])

torch.save(out, sys.argv[1]); print("saved",sys.argv[1],"scenarios:",list(out.keys()))
