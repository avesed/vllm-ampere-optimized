"""Does FA2 paged fwd_kvcache (the fork's MTP-verify path) survive CUDA graph capture?

Calls torch.ops._vllm_fa2_C.fwd_kvcache exactly like the fork's flash_attn_kvcache_verify
(k/v None, paged KV via block_table, per-request seqlens_k) once eagerly, then inside
torch.cuda.graph capture, then replays. Prints CAPTURE_OK / CAPTURE_FAIL.
"""

import sys

import torch

import vllm.vllm_flash_attn  # noqa: F401  (registers torch.ops._vllm_fa2_C)

torch.manual_seed(0)
dev = "cuda"
B, QLEN, H, HKV, D, PAGE, PAGES_PER_SEQ = 4, 4, 16, 4, 256, 16, 64
dt = torch.float16

num_blocks = B * PAGES_PER_SEQ
kcache = torch.randn(num_blocks, PAGE, HKV, D, dtype=dt, device=dev)
vcache = torch.randn(num_blocks, PAGE, HKV, D, dtype=dt, device=dev)
block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(B, PAGES_PER_SEQ)
seqlens_k = torch.tensor([700, 333, 1000, 64], dtype=torch.int32, device=dev)
q = torch.randn(B, QLEN, H, D, dtype=dt, device=dev)
out = torch.empty_like(q)
n_args = len(torch.ops._vllm_fa2_C.fwd_kvcache.default._schema.arguments)
print(f"torch {torch.__version__}  fwd_kvcache arity {n_args}")


def run():
    torch.ops._vllm_fa2_C.fwd_kvcache(
        q, kcache, vcache, None, None, seqlens_k, None, None, None, None,
        block_table, None, out, D ** -0.5, True, -1, -1, 0.0, False, 0,
    )


run()
torch.cuda.synchronize()
ref = out.clone()
print("eager OK")

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
try:
    with torch.cuda.stream(s):
        run()  # warmup on the side stream, as vLLM does
    torch.cuda.current_stream().wait_stream(s)
    out.zero_()
    with torch.cuda.graph(g):
        run()
    g.replay()
    torch.cuda.synchronize()
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    print(f"CAPTURE_OK replay_matches_eager={ok}")
except Exception as e:  # noqa: BLE001
    print(f"CAPTURE_FAIL {type(e).__name__}: {str(e).splitlines()[0][:300]}")
    sys.exit(1)
