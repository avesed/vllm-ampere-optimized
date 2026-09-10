"""hd512 paged DECODE A/B on Ampere: stock FlashInfer vs famp's vendored XQA. One leg per process.

Legs
  stock_decode    flashinfer BatchDecodeWithPagedKVCacheWrapper (large-head decode is SM80+ since
                  FlashInfer #3739; before that this did not exist on Ampere)
  stock_prefill   flashinfer BatchPrefillWithPagedKVCacheWrapper with q_len=1 -- the path famp
                  currently ships as the validated default for hd512 decode
  famp_xqa        famp's vendored XQA kernel (gemm0-once), called exactly as kernels.py does

Every leg is checked against an fp32 reference before timing, and the report includes effective KV
bandwidth against the card's spec so a physically impossible "win" is visible rather than believed.
Decode is DRAM-bound, so >100% of roofline means the leg is not reading the KV it claims to.
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

SPEC_GBPS = {"NVIDIA GeForce RTX 3090": 936.2}


def build(bs, ctx, hq, hkv, d, page_size, dtype, dev):
    ppr = (ctx + page_size - 1) // page_size
    npages = bs * ppr
    q = torch.randn(bs, hq, d, dtype=dtype, device=dev) * 0.5
    kc = torch.randn(npages, page_size, hkv, d, dtype=dtype, device=dev) * 0.5
    vc = torch.randn(npages, page_size, hkv, d, dtype=dtype, device=dev) * 0.5
    page_table = torch.arange(npages, dtype=torch.int32, device=dev).view(bs, ppr)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=dev)
    return q, kc, vc, page_table, seq_lens, ppr


def reference(q, kc, vc, ppr, bs, ctx, hq, hkv, d, sm_scale):
    rep = hq // hkv
    out = torch.empty(bs, hq, d, dtype=torch.float32, device=q.device)
    for b in range(bs):
        k = kc[b * ppr:(b + 1) * ppr].reshape(-1, hkv, d)[:ctx].float()
        v = vc[b * ppr:(b + 1) * ppr].reshape(-1, hkv, d)[:ctx].float()
        qb = q[b].float().unsqueeze(1)                                  # [Hq,1,D]
        kb = k.transpose(0, 1).repeat_interleave(rep, 0)
        vb = v.transpose(0, 1).repeat_interleave(rep, 0)
        out[b] = F.scaled_dot_product_attention(qb, kb, vb, scale=sm_scale).squeeze(1)
    return out


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True,
                    choices=["stock_decode", "stock_prefill", "famp_xqa"])
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--num-qo-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import flashinfer

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    d, hq, hkv = args.head_dim, args.num_qo_heads, args.num_kv_heads
    sm_scale = 1.0 / (d ** 0.5)
    name = torch.cuda.get_device_name(0)
    spec = SPEC_GBPS.get(name)

    for bs in args.batch_sizes:
        for ctx in args.ctx:
            q, kc, vc, page_table, seq_lens, ppr = build(
                bs, ctx, hq, hkv, d, args.page_size, dtype, dev)
            ref = reference(q, kc, vc, ppr, bs, ctx, hq, hkv, d, sm_scale)
            out = torch.empty(bs, hq, d, dtype=dtype, device=dev)
            kv_bytes = 2 * bs * ctx * hkv * d * q.element_size()   # K + V actually needed

            try:
                if args.leg == "famp_xqa":
                    from vllm.v1.attention.backends.flashampere.xqa._jit_xqa import gen_xqa_module
                    mod = gen_xqa_module(dtype, dtype, args.page_size, d, hq // hkv,
                                         False, dtype, 1).build_and_load()
                    nb_seq = hkv * bs
                    sem = torch.zeros((nb_seq + 1) // 2 * 2 + 2 + nb_seq + 2,
                                      dtype=torch.uint32, device=dev)
                    seq_u32 = seq_lens.view(bs, 1).to(torch.uint32)
                    scratch = torch.zeros(512 << 20, dtype=torch.uint8, device=dev)
                    smc = torch.cuda.get_device_properties(dev).multi_processor_count
                    q4, out4 = q.view(bs, 1, hq, d), out.view(bs, 1, hq, d)
                    q_scale = float(sm_scale) * (d ** 0.5)

                    def fn():
                        sem.zero_()
                        mod.xqa_wrapper(False, smc, hkv, 0, q_scale, None, out4, 1.0, q4, None,
                                        kc, vc, None, None, page_table, ppr * args.page_size,
                                        seq_u32, bs, 1.0, None, 1, None, sem, scratch, False)
                        return out
                elif args.leg == "stock_decode":
                    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
                    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="NHD")
                    indptr = torch.arange(bs + 1, dtype=torch.int32, device=dev) * ppr
                    indices = torch.arange(bs * ppr, dtype=torch.int32, device=dev)
                    last = torch.full((bs,), ctx - (ppr - 1) * args.page_size,
                                      dtype=torch.int32, device=dev)
                    w.plan(indptr, indices, last, hq, hkv, d, args.page_size,
                           q_data_type=dtype, kv_data_type=dtype)
                    kv = torch.stack([kc, vc], dim=1)   # [P,2,ps,Hkv,D]
                    fn = lambda: w.run(q, kv)           # noqa: E731
                else:  # stock_prefill (q_len=1)
                    ws = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
                    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
                    qo = torch.arange(bs + 1, dtype=torch.int32, device=dev)
                    indptr = torch.arange(bs + 1, dtype=torch.int32, device=dev) * ppr
                    indices = torch.arange(bs * ppr, dtype=torch.int32, device=dev)
                    last = torch.full((bs,), ctx - (ppr - 1) * args.page_size,
                                      dtype=torch.int32, device=dev)
                    w.plan(qo, indptr, indices, last, hq, hkv, d, args.page_size,
                           causal=False, q_data_type=dtype, kv_data_type=dtype)
                    kv = torch.stack([kc, vc], dim=1)
                    fn = lambda: w.run(q, kv)           # noqa: E731

                got = fn()
                torch.cuda.synchronize()
                cos = cosine(got.view(bs, hq, d), ref)

                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - t0) * 1e3)
                ms = statistics.median(samples)
                gbps = kv_bytes / (ms * 1e-3) / 1e9
                roof = f" {100 * gbps / spec:.0f}%roof" if spec else ""
                print(f"RESULT leg={args.leg} dtype={args.dtype} hd={d} heads={hq}/{hkv} "
                      f"bs={bs} ctx={ctx} ms={ms:.3f} cos={cos:.6f} bw={gbps:.0f}GB/s{roof}")
            except Exception as e:  # noqa: BLE001
                print(f"RESULT leg={args.leg} dtype={args.dtype} hd={d} heads={hq}/{hkv} "
                      f"bs={bs} ctx={ctx} FAILED {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    main()
