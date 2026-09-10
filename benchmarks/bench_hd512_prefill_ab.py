"""hd512 causal prefill on Ampere: stock FlashInfer vs famp's vendored fp16-PV kernel.

Upstream #3739 ("Enable Ampere FA2 large-head attention", in FlashInfer 0.6.14+) enables FA2
large-head on SM80+ through a generic VO-split, which is the same problem famp solved privately by
halving the O accumulator (fp16-PV). Since 0.6.12 could not run hd512 on Ampere at all, this A/B
only became possible with the newer FlashInfer. It decides whether famp still needs to carry its
own hd512 prefill kernel.

⚠ TIMINGS FROM THIS HARNESS ARE NOT TRUSTWORTHY -- use bench_batch_prefill_ab.py for verdicts.
Measured 2026-08-30: this harness reports famp ~2x SLOWER than stock at hd256/L=2048, while the
batch/paged harness reports famp 1.22x FASTER for the same kernel, dtype and shape. The two differ
in how the kernel is reached (this one calls famp's own `_prefill.single_prefill` marshalling with a
64 MB tmp buffer and no `plan()`); serving goes through the batch/paged wrapper. The CORRECTNESS
signal here is still useful (it is what showed famp's hd512 kernel is wrong on FlashInfer 0.6.16),
the milliseconds are not.

Legs:
  A  stock  flashinfer.single_prefill_with_kv_cache
  B  famp   vendored kernel, FA_USE_FP16_PV=0  (relaxed register heuristic, fp32 accumulate)
  C  famp   vendored kernel, FA_USE_FP16_PV=1  (fp16 accumulate -- the fork's perf lever)

Each leg is checked against an fp32 SDPA reference before it is timed.
Run inside a GPU container that has flashinfer + the famp package importable.
"""

import argparse
import time

import torch
import torch.nn.functional as F


def reference(q, k, v, sm_scale):
    # [L, H, D] -> SDPA in fp32, GQA-expanded, causal.
    qo_len, hq, d = q.shape
    kv_len, hkv, _ = k.shape
    rep = hq // hkv
    qt = q.float().transpose(0, 1)                       # [Hq, Lq, D]
    kt = k.float().transpose(0, 1).repeat_interleave(rep, 0)
    vt = v.float().transpose(0, 1).repeat_interleave(rep, 0)
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=q.device).tril(kv_len - qo_len)
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, scale=sm_scale)
    return out.transpose(0, 1)                           # [Lq, Hq, D]


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3      # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--num-qo-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 8192, 16384])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    dev = torch.device("cuda")
    d, hq, hkv = args.head_dim, args.num_qo_heads, args.num_kv_heads
    sm_scale = 1.0 / (d ** 0.5)

    import flashinfer
    print(f"flashinfer {getattr(flashinfer, '__version__', '?')} | "
          f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"hd={d} Hq={hq} Hkv={hkv} dtype={args.dtype} causal=True\n")

    try:
        from vllm.v1.attention.backends.flashampere.prefill import single_prefill as famp_prefill
        have_famp = True
    except Exception as e:  # noqa: BLE001
        print(f"!! famp prefill unavailable: {type(e).__name__}: {e}")
        have_famp = False

    rows = []
    for L in args.seq_lens:
        q = torch.randn(L, hq, d, dtype=dtype, device=dev) * 0.5
        k = torch.randn(L, hkv, d, dtype=dtype, device=dev) * 0.5
        v = torch.randn(L, hkv, d, dtype=dtype, device=dev) * 0.5
        ref = reference(q, k, v, sm_scale)

        legs = {"A stock-FI": lambda: flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=True, sm_scale=sm_scale)}
        if have_famp:
            legs["B famp pv=0"] = lambda: famp_prefill(
                q, k, v, causal=True, sm_scale=sm_scale, use_fp16_pv=False)
            legs["C famp pv=1"] = lambda: famp_prefill(
                q, k, v, causal=True, sm_scale=sm_scale, use_fp16_pv=True)

        for name, fn in legs.items():
            try:
                out = fn()
                cos = cosine(out, ref)
                ms = timed(fn, args.iters)
                rows.append((L, name, ms, cos))
                print(f"  L={L:>6}  {name:<12}  {ms:8.3f} ms   cos={cos:.6f}")
            except Exception as e:  # noqa: BLE001
                rows.append((L, name, None, None))
                print(f"  L={L:>6}  {name:<12}  FAILED: {type(e).__name__}: {str(e)[:160]}")
        print()

    print("=== summary (speedup vs stock-FI; >1 means famp is faster) ===")
    for L in args.seq_lens:
        base = next((r[2] for r in rows if r[0] == L and r[1].startswith("A")), None)
        if not base:
            continue
        parts = [f"{r[1]}={base / r[2]:.2f}x" for r in rows if r[0] == L and r[2] and not r[1].startswith("A")]
        print(f"  L={L:>6}  stock={base:.3f} ms  " + "  ".join(parts))


if __name__ == "__main__":
    main()
