"""Perf + correctness bench for the fp16-accum PV kernel (run patched vs clean, compare).

Usage (single-prefill, hd256, the kernel the patch touches):
  python3 bench.py save  <out.pt> [seqlen]   # run, save {O, median_ms}; default seqlen 4096
  python3 bench.py cmp   <test.pt> <ref.pt>   # cos(test,ref) + speedup ref/test

Typical flow: run on CLEAN flashinfer -> save clean.pt; apply_fp16pv.py -> save patched.pt; cmp.
MEASURED (RTX 3090 sm_86, hd256 f16): speedup 1.256x/1.252x/1.237x @ 2k/4k/8k, worst-row cos
0.999995/0.999990/0.999979 (the JIT recompiles the patched kernel on first call).
"""
import sys, torch, flashinfer


def run_save(out, L=4096):
    torch.manual_seed(0)
    HD, HQ, HK = 256, 16, 4
    q = torch.randn(L, HQ, HD, dtype=torch.float16, device="cuda")
    k = torch.randn(L, HK, HD, dtype=torch.float16, device="cuda")
    v = torch.randn(L, HK, HD, dtype=torch.float16, device="cuda")
    run = lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")
    o = run()
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    s, e, ts = torch.cuda.Event(True), torch.cuda.Event(True), []
    for _ in range(100):
        s.record(); run(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort()
    torch.save({"O": o.float().cpu(), "t": ts[50]}, out)
    print(f"saved {out} L={L} median={ts[50]:.4f}ms")


def cmp(test, ref):
    a, b = torch.load(test), torch.load(ref)
    cos = torch.nn.functional.cosine_similarity(a["O"].reshape(-1, 256), b["O"].reshape(-1, 256), dim=1)
    print(f"ref={b['t']:.4f}ms  test={a['t']:.4f}ms  speedup={b['t']/a['t']:.3f}x  "
          f"worst-cos={cos.min():.6f}  max-abs-err={(a['O']-b['O']).abs().max():.3e}")


if __name__ == "__main__":
    if sys.argv[1] == "save":
        run_save(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4096)
    else:
        cmp(sys.argv[2], sys.argv[3])
