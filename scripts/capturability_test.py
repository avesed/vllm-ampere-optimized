#!/usr/bin/env python3
"""Can fwd_kvcache be captured in a CUDA graph? num_splits=0 picks a runtime-sized combine
workspace (likely not capturable); a FIXED num_splits should give a fixed workspace -> capturable.
If NS=32 captures + replays correct, patch A v2 (pass attn_metadata.max_num_splits, drop the
is_current_stream_capturing guard) realizes the verify win inside the FULL cudagraph serve."""
import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401

op = torch.ops._vllm_fa2_C.fwd_kvcache
dev, dt = "cuda", torch.bfloat16
D, Hq, Hkv, BS, kv, B, ql = 256, 16, 4, 16, 32768, 1, 3
torch.manual_seed(0)
nblk = (kv + BS - 1) // BS
q = torch.randn(B, ql, Hq, D, device=dev, dtype=dt)
kc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
vc = torch.randn(B * nblk + 1, BS, Hkv, D, device=dev, dtype=dt)
bt = torch.arange(B * nblk, device=dev, dtype=torch.int32).reshape(B, nblk)
seqk = torch.full((B,), kv, device=dev, dtype=torch.int32)
out = torch.empty(B, ql, Hq, D, device=dev, dtype=dt)


def call(ns):
    op(q, kc, vc, None, None, seqk, None, None, None, None, bt, None, out,
       D ** -0.5, True, -1, -1, 0.0, False, ns)


for NS in [0, 32]:
    call(NS)
    torch.cuda.synchronize()
    ref = out.clone()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call(NS)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(NS)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        cos = F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()
        print(f"num_splits={NS:2d}: CAPTURE OK, replay cos={cos:.6f}  "
              f"{'(matches)' if cos > 0.999 else '*** REPLAY WRONG ***'}")
    except Exception as e:
        print(f"num_splits={NS:2d}: CAPTURE FAILED -> {str(e)[:140]}")
