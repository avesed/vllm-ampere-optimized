"""Batch/paged prefill A/B: stock FlashInfer vs famp's fp16-PV kernel. One leg per process.

This is the path famp actually serves through (`_jit_batch_prefill.install_famp_batch_prefill`
shims flashinfer's module generator for one exact config key), unlike the single-prefill A/B in
bench_hd512_prefill_ab.py. flashinfer caches the built module per key, so each leg must run in its
own process -- see the driver loop in the accompanying shell invocation.

  --leg stock     plain flashinfer BatchPrefillWithPagedKVCacheWrapper
  --leg famp_pv0  famp's vendored kernel, FA_USE_FP16_PV=0
  --leg famp_pv1  famp's vendored kernel, FA_USE_FP16_PV=1

For bf16 the famp legs reproduce the shipped `bf16cvt` leg: upcast q/k/v to fp16, run famp's
half-only kernel, cast the output back -- so the comparison is what serving would actually do.
"""

import argparse
import time

import torch
import torch.nn.functional as F


def build_case(bs, seq_len, hq, hkv, d, page_size, dtype, dev):
    pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = bs * pages_per_req
    q = torch.randn(bs * seq_len, hq, d, dtype=dtype, device=dev) * 0.5
    kv = torch.randn(num_pages, 2, page_size, hkv, d, dtype=dtype, device=dev) * 0.5
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * seq_len
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * pages_per_req
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=dev)
    last_page = torch.full((bs,), seq_len - (pages_per_req - 1) * page_size,
                           dtype=torch.int32, device=dev)
    return q, kv, qo_indptr, kv_indptr, kv_indices, last_page, pages_per_req


def reference(q, kv, bs, seq_len, hq, hkv, d, page_size, pages_per_req, sm_scale):
    out = torch.empty(bs * seq_len, hq, d, dtype=torch.float32, device=q.device)
    rep = hq // hkv
    for b in range(bs):
        pages = kv[b * pages_per_req:(b + 1) * pages_per_req]          # [P,2,ps,Hkv,D]
        k = pages[:, 0].reshape(-1, hkv, d)[:seq_len].float()
        v = pages[:, 1].reshape(-1, hkv, d)[:seq_len].float()
        qb = q[b * seq_len:(b + 1) * seq_len].float().transpose(0, 1)  # [Hq,L,D]
        kb = k.transpose(0, 1).repeat_interleave(rep, 0)
        vb = v.transpose(0, 1).repeat_interleave(rep, 0)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
        o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, scale=sm_scale)
        out[b * seq_len:(b + 1) * seq_len] = o.transpose(0, 1)
    return out


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=["stock", "famp_pv0", "famp_pv1"])
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--num-qo-heads", type=int, default=24)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import flashinfer

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    d, hq, hkv, bs = args.head_dim, args.num_qo_heads, args.num_kv_heads, args.batch_size
    sm_scale = 1.0 / (d ** 0.5)
    famp = args.leg.startswith("famp")
    # famp's kernel is half-only; bf16 goes through the shipped bf16cvt upcast.
    run_dtype = torch.float16 if (famp and dtype is torch.bfloat16) else dtype

    if famp:
        from vllm.v1.attention.backends.flashampere.prefill._jit_batch_prefill import (
            install_famp_batch_prefill,
        )
        install_famp_batch_prefill(run_dtype, run_dtype, run_dtype, d,
                                   use_fp16_pv=(args.leg == "famp_pv1"))

    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")

    for L in args.seq_lens:
        q, kv, qo_indptr, kv_indptr, kv_indices, last_page, ppr = build_case(
            bs, L, hq, hkv, d, args.page_size, dtype, dev)
        ref = reference(q, kv, bs, L, hq, hkv, d, args.page_size, ppr, sm_scale)
        qr, kvr = (q.to(run_dtype), kv.to(run_dtype)) if run_dtype is not dtype else (q, kv)

        wrapper.plan(qo_indptr, kv_indptr, kv_indices, last_page, hq, hkv, d, args.page_size,
                     causal=True, q_data_type=run_dtype, kv_data_type=run_dtype)
        fn = lambda: wrapper.run(qr, kvr)  # noqa: E731

        try:
            out = fn()
            cos = cosine(out, ref)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1e3
            print(f"RESULT leg={args.leg} dtype={args.dtype} hd={d} heads={hq}/{hkv} "
                  f"bs={bs} L={L} ms={ms:.3f} cos={cos:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"RESULT leg={args.leg} dtype={args.dtype} hd={d} heads={hq}/{hkv} "
                  f"bs={bs} L={L} FAILED {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
