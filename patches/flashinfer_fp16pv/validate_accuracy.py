"""Accuracy gate for fp16-accum PV vs full fp32 — FAITHFUL fake-quant (no CUDA kernel needed).

Emulates the EXACT kernel: full online-softmax (running max m, rescale O*=exp(m_old-m_new), running sum
d) + per CTA_TILE_KV tile a fp16 CHAIN across width-16 MMA groups -> promote (NOT a single cast of the
tile sum, which would under-state error = false GO). Sweeps tile x ctx x flatness (tau; tau=0 = uniform
attention = swamping worst case). Metric: per-query-row cos, MIN over rows. GO = worst-row cos > 0.9999.

Modes: 'two' = two-level (fp32 master + fp16 partial, register-NEUTRAL, bulletproof);
       'pure' = pure-fp16 master (the -64 register WINNER that apply_fp16pv.py ships).
MEASURED 2026-06-24 (RTX 3090): two-level worst cos 0.999999; pure-fp16 worst 0.999928 (~= int8-QK).
"""
import torch

dev = "cuda"; torch.manual_seed(0)
D = 256; R = 96; GROUP = 16


def make_V(L):
    V = torch.randn(L, D, device=dev); V[:, ::64] *= 8.0  # outlier channels
    return V.half().float()


def make_s(L, tau):
    return torch.zeros(R, L, device=dev) if tau == 0 else torch.randn(R, L, device=dev) * tau


def online(s, V, tile, mode):
    Rr, L = s.shape
    m = torch.full((Rr,), -float("inf"), device=dev); d = torch.zeros(Rr, device=dev)
    O = torch.zeros(Rr, D, device=dev, dtype=torch.float16 if mode == "pure" else torch.float32)
    for t0 in range(0, L, tile):
        st = s[:, t0:t0+tile]; Vt = V[t0:t0+tile]
        m_new = torch.maximum(m, st.max(1).values)
        scale = torch.exp(m - m_new); scale[torch.isnan(scale)] = 0
        O = (O.float() * scale[:, None]).half() if mode == "pure" else O * scale[:, None]
        d = d * scale; p = torch.exp(st - m_new[:, None]); d = d + p.sum(1)
        if mode == "fp32":
            O += p @ Vt
        else:
            part = torch.zeros(Rr, D, device=dev, dtype=torch.float16)
            for g in range(0, Vt.shape[0], GROUP):                 # fp16 chain across width-16 groups
                part = (part.float() + (p[:, g:g+GROUP] @ Vt[g:g+GROUP])).half()
            O = (O.float() + part.float()).half() if mode == "pure" else O + part.float()
        m = m_new
    return O.float() / d[:, None]


def worst_cos(L, tile, tau, V, mode):
    s = make_s(L, tau); ref = online(s, V, L, "fp32"); t = online(s, V, tile, mode)
    return torch.nn.functional.cosine_similarity(t, ref, dim=1).min().item()


def main():
    print(f"{'L':>7}{'tau':>6}{'tile':>5} {'two-level':>10} {'pure-fp16':>10}")
    wmin = {"two": 1.0, "pure": 1.0}
    for L in [8192, 32768, 131072]:
        V = make_V(L)
        for tau in [0.0, 0.5, 2.0]:                                # 0=uniform worst, 2.0=peaked
            for tile in ([64, 256] if L >= 131072 else [16, 64, 256]):
                c2 = worst_cos(L, tile, tau, V, "two"); cp = worst_cos(L, tile, tau, V, "pure")
                wmin["two"] = min(wmin["two"], c2); wmin["pure"] = min(wmin["pure"], cp)
                flag = "" if cp > 0.9999 else ("  SOFT" if cp > 0.999 else "  FAIL")
                print(f"{L:>7}{tau:>6}{tile:>5} {c2:>10.6f} {cp:>10.6f}{flag}")
        print()
    print(f"WORST two-level={wmin['two']:.6f} ({'GO' if wmin['two']>0.9999 else 'no'}) | "
          f"pure-fp16={wmin['pure']:.6f} ({'GO' if wmin['pure']>0.9999 else ('SOFT' if wmin['pure']>0.999 else 'NO-GO')})")


if __name__ == "__main__":
    main()
