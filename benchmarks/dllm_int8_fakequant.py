#!/usr/bin/env python3
"""
dLLM int8 fake-quant go/no-go — offline validation for "W4A16 weights + per-token int8 activations" on a DIFFUSION LLM.

This is Stage-0 of docs/RESEARCH-diffusion-llm.md Part 7: a ZERO-kernel, ZERO-serving-dependency gate
that answers, before any Marlin/serving work, two questions on Dream-7B (a dense AR->diffusion model
adapted from Qwen2.5-7B):

  (A) error-curve : does the int8-activation quant error stay FLAT across denoising steps (good),
                    or grow geometrically (the W4A4 failure mode in DLLMQuant Fig.2)?  -> primary go/no-go
  (B) accuracy    : fp(bf16) vs fake-quant(W4-sym-g32 weights + per-token-int8 acts) GSM8K A/B  -> coherence/quality
  (C) profile     : per-layer activation fragility (max-abs / std / kurtosis) along the denoising
                    trajectory  -> the Transformer Lab fragility profiler; picks which layers to keep in bf16

Fake-quant only (no real int8 GEMM) — it measures the QUANTIZATION ERROR, which is what gates the kernel work.
Symmetric int4 weights (group=32, matches the user's compressed-tensors recipe) + per-token dynamic int8
activations (matches the deployed per_token_quant_int8). lm_head + embed_tokens are kept full precision.

Setup (uv, per the user's preference):
    uv venv --python 3.11 && source .venv/bin/activate
    uv pip install torch==2.5.1 transformers==4.46.2 accelerate numpy
    # (matplotlib optional, for a PNG of the curve)

Run (single 24GB GPU, e.g. one 3090; needs ~16-18GB):
    python benchmarks/dllm_int8_fakequant.py --mode error-curve
    python benchmarks/dllm_int8_fakequant.py --mode accuracy --n 30 --steps 128
    python benchmarks/dllm_int8_fakequant.py --mode profile

Verified Dream API (Dream-org/Dream-v0-Instruct-7B): trust_remote_code AutoModel/DreamModel; Qwen2.5-7B
geometry (28 layers, hidden 3584, ffn 18944, vocab 152064); mask_token_id=151666; eos/pad/bos=151643;
linears at model.model.layers.{i}.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}; lm_head separate
(untied); forward(input_ids, attention_mask, position_ids).logits; model.diffusion_generate(...).
"""
import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 151666
EOS_ID = 151643
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ----------------------------- fake-quant primitives -----------------------------
def fake_quant_weight_sym_group(w: torch.Tensor, bits: int = 4, group: int = 32) -> torch.Tensor:
    """Symmetric int(bits) weight fake-quant, grouped along the input dim (per output row)."""
    out, inf = w.shape
    g = group if (group > 0 and inf % group == 0) else inf
    qmax = 2 ** (bits - 1) - 1  # int4 -> 7
    wv = w.reshape(out, inf // g, g).float()
    scale = wv.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(wv / scale), -(qmax + 1), qmax)
    return (q * scale).reshape(out, inf).to(w.dtype)


def fake_quant_act_per_token_sym(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token (per last-dim vector) symmetric dynamic int(bits) activation fake-quant."""
    qmax = 2 ** (bits - 1) - 1  # int8 -> 127
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(xf / scale), -(qmax + 1), qmax)
    return (q * scale).to(x.dtype)


class FakeQuantLinear(nn.Module):
    """Wraps an nn.Linear; toggleable weight/activation fake-quant. bake_inplace() destroys fp weights."""

    def __init__(self, lin: nn.Linear, w_bits=4, w_group=32, a_bits=8):
        super().__init__()
        self.lin = lin
        self.w_bits, self.w_group, self.a_bits = w_bits, w_group, a_bits
        self.quant_w = False
        self.quant_a = False
        self.baked = False

    @property
    def weight(self):
        return self.lin.weight

    @property
    def bias(self):
        return self.lin.bias

    def bake_inplace(self):
        with torch.no_grad():
            self.lin.weight.data.copy_(fake_quant_weight_sym_group(self.lin.weight.data, self.w_bits, self.w_group))
        self.baked = True

    def forward(self, x):
        w = self.lin.weight
        if self.quant_w and not self.baked:
            w = fake_quant_weight_sym_group(w, self.w_bits, self.w_group)
        if self.quant_a:
            x = fake_quant_act_per_token_sym(x, self.a_bits)
        return F.linear(x, w, self.lin.bias)


def wrap_linears(model, w_bits, w_group, a_bits, protect_substrings):
    wrappers = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGET_SUFFIXES:
            if any(p in name for p in protect_substrings):
                continue  # kept full precision (fragile-layer protection)
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            fq = FakeQuantLinear(mod, w_bits, w_group, a_bits)
            setattr(parent, name.split(".")[-1], fq)
            wrappers.append((name, fq))
    return wrappers


def set_quant(wrappers, w, a):
    for _, fq in wrappers:
        fq.quant_w, fq.quant_a = w, a


# ----------------------------- model loading -----------------------------
def load_model(model_id, dtype):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    model = model.to("cuda").eval()
    return model, tok


def forward_logits(model, x, attn_mask, pos_ids):
    # Dream's forward passes attention_mask straight to SDPA (is_causal=False hard-coded) and does NOT
    # convert 2D->4D internally (diffusion_generate does that). Our canvas has no padding, so full
    # bidirectional attention == attn_mask=None (== the all-zeros 4D bias diffusion_generate builds).
    # Passing the raw 2D long mask hits SDPA's dtype check and crashes. position_ids are passed explicitly.
    return model(input_ids=x, attention_mask=None, position_ids=pos_ids).logits


# ----------------------------- (A) per-step error curve -----------------------------
@torch.no_grad()
def run_error_curve(model, tok, wrappers, args):
    """Drive ONE coherent reference (fp) denoising trajectory; at each step also compute the fake-quant
    logits on the IDENTICAL partially-masked sequence and measure fp-vs-quant divergence at masked
    positions. Isolates per-step quant error without trajectory drift (the DLLMQuant Fig.2 analog)."""
    prompt = args.prompt or "Explain why the sky is blue, step by step."
    messages = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True)
    ids = enc["input_ids"].to("cuda")
    L = ids.shape[1]
    T = L + args.max_new
    x = torch.full((1, T), MASK_ID, dtype=torch.long, device="cuda")
    x[:, :L] = ids
    am = torch.ones((1, T), dtype=torch.long, device="cuda")
    pos = am.long().cumsum(-1) - 1
    timesteps = torch.linspace(1, args.eps, args.steps + 1, device="cuda")

    rows = []
    for i in range(args.steps):
        t, s = timesteps[i], timesteps[i + 1]
        mask_index = x == MASK_ID
        if mask_index.sum() == 0:
            break
        qw = args.quant in ("wa", "w")
        qa = args.quant in ("wa", "a")
        set_quant(wrappers, False, False)
        lf = forward_logits(model, x, am, pos)
        set_quant(wrappers, qw, qa)
        lq = forward_logits(model, x, am, pos)
        set_quant(wrappers, False, False)

        mp = mask_index[0]
        f = lf[0][mp].float()
        q = lq[0][mp].float()
        cos = F.cosine_similarity(f, q, dim=-1).mean().item()
        rel = ((q - f).norm(dim=-1) / (f.norm(dim=-1) + 1e-6)).mean().item()
        agree = (f.argmax(-1) == q.argmax(-1)).float().mean().item()
        rows.append((i, t.item(), int(mp.sum()), cos, rel, agree))

        # advance the reference (fp) trajectory: confidence top-k unmask (deterministic, coherent)
        num_mask = int(mask_index.sum().item())
        n_tr = num_mask if i == args.steps - 1 else int(num_mask * (1 - (s / t).item()))
        n_tr = max(n_tr, 0)
        if n_tr > 0:
            probs = lf[0].softmax(-1)
            conf, x0 = probs.max(-1)
            conf = conf.masked_fill(~mask_index[0], -1.0)
            idx = conf.topk(n_tr).indices
            x[0, idx] = x0[idx]

    _report_curve(rows, args)
    return rows


def _report_curve(rows, args):
    if not rows:
        print("no steps run", flush=True)
        return
    print("\n step      t   nmask    cos_sim   rel_L2   top1_agree", flush=True)
    show = rows if len(rows) <= 12 else [rows[0]] + rows[1:-1:max(1, len(rows) // 8)] + [rows[-1]]
    for (i, t, nm, cos, rel, ag) in show:
        print(f"  {i:4d} {t:5.3f} {nm:6d}   {cos:7.4f}  {rel:7.4f}   {ag:7.4f}", flush=True)

    out_csv = args.out or "dream_fakequant_errorcurve.csv"
    with open(out_csv, "w") as fcsv:
        fcsv.write("step,t,n_mask,cos_sim,rel_l2,top1_agree\n")
        for r in rows:
            fcsv.write(",".join(str(v) for v in r) + "\n")
    print(f"\nwrote {out_csv}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r[0] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(steps, [r[4] for r in rows]); ax[0].set_title("rel L2 (fp vs quant) per denoising step")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("rel L2"); ax[0].grid(True, alpha=.3)
        ax[1].plot(steps, [r[3] for r in rows], label="cos"); ax[1].plot(steps, [r[5] for r in rows], label="top1 agree")
        ax[1].set_title("cosine / top1-agreement"); ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(True, alpha=.3)
        png = out_csv.replace(".csv", ".png")
        fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f"wrote {png}", flush=True)
    except Exception as e:
        print(f"(matplotlib skipped: {e})", flush=True)

    # verdict — the per-step curve answers ONE thing: does quant error ACCUMULATE across denoising
    # steps (the W4A4 failure mode in DLLMQuant Fig.2) or stay bounded? Mean rel-L2 trend is the signal.
    # NOTE: top1-agreement is reported but NOT gated on — early steps have a (near-)fully-masked canvas
    # with low-confidence logits whose argmax flips on tiny perturbation (structural, not collapse), so
    # it is naturally low early and rises with context. ACCURACY A/B (--mode accuracy) is decisive.
    rels = [r[4] for r in rows]
    coss = [r[3] for r in rows]
    ags = [r[5] for r in rows]
    n = len(rows)
    early = rows[: max(n // 3, 1)]
    late = rows[-max(n // 3, 1):]
    mean_rel_early = sum(r[4] for r in early) / len(early)
    mean_rel_late = sum(r[4] for r in late) / len(late)
    growth = mean_rel_late / max(mean_rel_early, 1e-6)
    scheme = {"wa": "W4+A8", "w": "W4 weight-only", "a": "A8 act-only"}[args.quant]
    print(f"\n--- ACCUMULATION CHECK ({scheme} vs bf16) ---", flush=True)
    print(f"  rel-L2: mean={sum(rels)/n:.4f}  early3rd={mean_rel_early:.4f}  late3rd={mean_rel_late:.4f}  "
          f"late/early={growth:.2f}x", flush=True)
    print(f"  cos-sim mean={sum(coss)/n:.4f}  (informational) top1-agree mean(excl step0)="
          f"{sum(ags[1:])/max(len(ags)-1,1):.4f}", flush=True)
    if growth < 1.8:
        print("  => error is FLAT across denoising steps (no geometric accumulation = the GOOD/int8-like "
              "regime, NOT the W4A4 blow-up). Decisive test = run --mode accuracy.", flush=True)
    elif growth < 3.0:
        print("  => mild growth; watch it, but not the W4A4 geometric blow-up. Run --mode accuracy.", flush=True)
    else:
        print("  => error GROWS geometrically across steps (W4A4-style accumulation) -> likely NO-GO; "
              "confirm with --mode accuracy.", flush=True)


# ----------------------------- (B) GSM8K accuracy A/B -----------------------------
def _extract_pred(text):
    after = text.split("####")[-1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*", after)
    if not nums:
        nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else None


def _gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


@torch.no_grad()
def _gsm8k_run(model, tok, items, args, tag):
    correct = 0
    preds = []
    t0 = time.time()
    for k, it in enumerate(items):
        msg = [{"role": "user", "content": it["question"] +
                "\nReason step by step, then give the final answer after '####'."}]
        enc = tok.apply_chat_template(msg, return_tensors="pt", return_dict=True, add_generation_prompt=True)
        ids = enc["input_ids"].to("cuda")
        am = enc["attention_mask"].to("cuda")
        out = model.diffusion_generate(
            ids, attention_mask=am, max_new_tokens=args.max_new, steps=args.steps,
            temperature=args.temperature, top_p=(args.top_p if args.temperature > 0 else None),
            alg=args.alg, alg_temp=0.0, output_history=False, return_dict_in_generate=True,
        )
        gen = out.sequences[0][ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        pred, gold = _extract_pred(text), _gold(it["answer"])
        ok = pred is not None and pred == gold
        correct += int(ok)
        preds.append((pred, gold, ok))
        print(f"  [{tag}] {k + 1}/{len(items)} pred={pred} gold={gold} {'OK' if ok else 'x'}", flush=True)
    dt = time.time() - t0
    acc = correct / max(len(items), 1)
    print(f"  [{tag}] acc={acc:.3f} ({correct}/{len(items)})  {dt:.1f}s", flush=True)
    return acc, preds


def run_accuracy(model, tok, wrappers, args):
    items = _load_gsm8k(args.data, args.n)
    print(f"\n=== GSM8K A/B on {len(items)} items (steps={args.steps}, alg={args.alg}, temp={args.temperature}) ===", flush=True)

    set_quant(wrappers, False, False)
    print("\n-- baseline bf16 (fp) --", flush=True)
    acc_fp, preds_fp = _gsm8k_run(model, tok, items, args, "fp")

    print("\n-- baking weights to int4-sym-g32 in place + enabling per-token int8 acts --", flush=True)
    for _, fq in wrappers:
        fq.bake_inplace()
        fq.quant_a = True
    torch.cuda.empty_cache()
    acc_q, preds_q = _gsm8k_run(model, tok, items, args, "W4A8")

    flips = sum(1 for (a, b) in zip(preds_fp, preds_q) if a[2] != b[2])
    print("\n--- ACCURACY A/B SUMMARY ---", flush=True)
    print(f"  bf16    : {acc_fp:.3f}", flush=True)
    print(f"  W4A8(fq): {acc_q:.3f}   delta={acc_q - acc_fp:+.3f}   per-item flips={flips}/{len(items)}", flush=True)
    drop = acc_fp - acc_q
    verdict = ("GO (<=2pt drop -> int8-act quality holds; proceed to real Marlin kernel/serving)" if drop <= 0.02
               else "CAUTION (2-5pt drop -> try GPTQ over AWQ + protect fragile down_proj/ff_out via --profile)" if drop <= 0.05
               else "NO-GO (>5pt drop -> int8 unsafe on this config; profile fragile layers or drop to int8-weight-only)")
    print(f"  => {verdict}", flush=True)


def _load_gsm8k(path, n):
    if not os.path.exists(path):
        sys.exit(f"GSM8K not found at {path} (run benchmarks/load_data.py first)")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


# ----------------------------- (C) per-layer fragility profiler -----------------------------
@torch.no_grad()
def run_profile(model, tok, wrappers, args):
    """Transformer Lab-style fragility profiler: along the denoising trajectory, record per-linear
    INPUT-activation max-abs / std / kurtosis. High kurtosis + max-abs = outlier-heavy = fragile ->
    candidates to keep in bf16. (down_proj / ff_out usually dominate.)"""
    stats = {name: {"maxabs": 0.0, "sum": 0.0, "sumsq": 0.0, "sum4": 0.0, "cnt": 0} for name, _ in wrappers}
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float().flatten()
            st = stats[name]
            st["maxabs"] = max(st["maxabs"], x.abs().max().item())
            st["sum"] += x.sum().item(); st["sumsq"] += (x * x).sum().item()
            st["sum4"] += (x ** 4).sum().item(); st["cnt"] += x.numel()
        return hook

    for name, fq in wrappers:
        handles.append(fq.register_forward_hook(mk_hook(name)))

    set_quant(wrappers, False, False)
    prompt = args.prompt or "Explain why the sky is blue, step by step."
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  return_tensors="pt", return_dict=True, add_generation_prompt=True)
    ids = enc["input_ids"].to("cuda")
    L = ids.shape[1]
    x = torch.full((1, L + args.max_new), MASK_ID, dtype=torch.long, device="cuda")
    x[:, :L] = ids
    am = torch.ones_like(x); pos = am.long().cumsum(-1) - 1
    timesteps = torch.linspace(1, args.eps, args.steps + 1, device="cuda")
    for i in range(args.steps):
        t, s = timesteps[i], timesteps[i + 1]
        mask_index = x == MASK_ID
        if mask_index.sum() == 0:
            break
        lf = forward_logits(model, x, am, pos)
        num_mask = int(mask_index.sum().item())
        n_tr = num_mask if i == args.steps - 1 else int(num_mask * (1 - (s / t).item()))
        if n_tr > 0:
            probs = lf[0].softmax(-1); conf, x0 = probs.max(-1)
            conf = conf.masked_fill(~mask_index[0], -1.0)
            idx = conf.topk(max(n_tr, 1)).indices; x[0, idx] = x0[idx]
    for h in handles:
        h.remove()

    rank = []
    for name, st in stats.items():
        if st["cnt"] == 0:
            continue
        n = st["cnt"]; mean = st["sum"] / n
        var = max(st["sumsq"] / n - mean * mean, 1e-12)
        # raw 4th moment about zero -> excess kurtosis approx (activations ~ zero-mean)
        kurt = (st["sum4"] / n) / (var * var) - 3.0
        rank.append((name, st["maxabs"], math.sqrt(var), kurt))
    # fragility score = normalized(maxabs) * (1 + excess_kurtosis_clamped)  (heuristic; TL weights 3 stats)
    mx = max(r[1] for r in rank) or 1.0
    rank.sort(key=lambda r: (r[1] / mx) * (1.0 + max(r[3], 0.0)), reverse=True)
    print("\n=== per-layer activation fragility (top 20; bf16-protect candidates) ===", flush=True)
    print(f"{'layer':52s} {'max|x|':>10s} {'std':>8s} {'kurtosis':>10s}", flush=True)
    for name, ma, sd, ku in rank[:20]:
        print(f"{name:52s} {ma:10.2f} {sd:8.3f} {ku:10.1f}", flush=True)
    print(f"\nsuggest --protect for the top FFN down_proj/o_proj entries above "
          f"(keep them bf16), then re-run --mode accuracy.", flush=True)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["error-curve", "accuracy", "profile"], default="error-curve")
    ap.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--w-group", type=int, default=32)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--protect", default="", help="comma-sep substrings of layer names to keep bf16 (fragile protection)")
    ap.add_argument("--quant", choices=["wa", "w", "a"], default="wa",
                    help="error-curve: quantize weights+acts (wa), weight-only int4 (w), or act-only int8 (a) — isolates the int8-activation contribution")
    ap.add_argument("--steps", type=int, default=64, help="denoising steps (64 smoke; 128-256 quality)")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--alg", default="entropy", help="origin|maskgit_plus|topk_margin|entropy (accuracy mode)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=20, help="GSM8K items (accuracy mode)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gsm8k.jsonl"))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    protect = [p for p in args.protect.split(",") if p]

    print(f"loading {args.model} (bf16)...", flush=True)
    model, tok = load_model(args.model, torch.bfloat16)
    wrappers = wrap_linears(model, args.w_bits, args.w_group, args.a_bits, protect)
    print(f"wrapped {len(wrappers)} Linear layers "
          f"(W{args.w_bits}-sym-g{args.w_group} + A{args.a_bits}-per-token; lm_head/embed kept fp; "
          f"{len(protect)} protected)", flush=True)
    print(f"mode={args.mode} steps={args.steps} max_new={args.max_new}", flush=True)

    if args.mode == "error-curve":
        run_error_curve(model, tok, wrappers, args)
    elif args.mode == "accuracy":
        run_accuracy(model, tok, wrappers, args)
    else:
        run_profile(model, tok, wrappers, args)


if __name__ == "__main__":
    main()
