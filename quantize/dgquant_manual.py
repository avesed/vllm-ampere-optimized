#!/usr/bin/env python3
"""Manual layer-sequential GPTQ for DiffusionGemma-26B-A4B.

llmcompressor's fx-tracing `sequential` pipeline cannot trace the block-diffusion (encoder+decoder,
wrapped forward, self-conditioning) model — it auto-wraps `DiffusionGemmaModel`, mis-splits the graph
(88 expected subgraphs -> 31 traced) and runs the WHOLE decoder on one GPU -> OOM. So we drive GPTQ
ourselves: capture the decoder-layer inputs once (cheap — text-only calib skips the vision encoder),
then quantize each `DiffusionGemmaDecoderTextLayer` on ONE GPU at a time (bounded memory). Reuses
llmcompressor's GPTQ math (make_empty_hessian/accumulate_hessian/quantize_weight) + the MoE
linearizer + compressed-tensors for the W4A16 packing/save.

Env: MODEL, OUT, NUM(calib=128), MAXLEN(1024), LAYERS(0=all decoder layers; >0 = smoke test).
Run with dgquant-venv (transformers 5.12 + llmcompressor 0.12.1a). One GPU is enough.
"""
import os, json, random
import torch
from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer, AutoProcessor
from datasets import load_dataset
from compressed_tensors.quantization import (QuantizationArgs, QuantizationScheme, QuantizationConfig,
    apply_quantization_config)
from llmcompressor.modifiers.gptq.gptq_quantize import make_empty_hessian, accumulate_hessian, quantize_weight
from llmcompressor.modeling.moe.linearize import linearize_moe

M = os.environ.get("MODEL", "/home/coder/models/diffusiongemma-26B-A4B-it")
OUT = os.environ.get("OUT", "/home/coder/models/diffusiongemma-26B-A4B-it-w4a16-g32")
NUM = int(os.environ.get("NUM", "128"))
MAXLEN = int(os.environ.get("MAXLEN", "1024"))
LAYERS = int(os.environ.get("LAYERS", "0"))
DEV = torch.device("cuda:0")
IGNORE = ["lm_head", "re:.*embed.*", "re:.*router", "re:.*vision_tower.*", "re:.*self_conditioning.*"]
QARGS = QuantizationArgs(num_bits=4, type="int", symmetric=True, group_size=32,
                         strategy="group", dynamic=False, observer="mse")
SCHEME = QuantizationScheme(targets=["Linear"], weights=QARGS, input_activations=None, output_activations=None)


def log(m): print(f"[dgq] {m}", flush=True)


class _Stop(Exception):
    pass


def main():
    random.seed(1234)
    tok = AutoTokenizer.from_pretrained(M)
    log("loading model (bf16, cpu)...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        M, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    log("linearizing MoE experts (fused 3D -> nn.Linear)...")
    linearize_moe(model)

    # decoder text layers (the MoE text stack)
    layers = model.model.decoder.layers
    n_layers = LAYERS if LAYERS > 0 else len(layers)
    log(f"{len(layers)} decoder layers; quantizing {n_layers}")

    # mark target modules with the W4A16 scheme (creates weight_scale/zp params + quant config)
    qconfig = QuantizationConfig(config_groups={"group_0": SCHEME}, ignore=IGNORE,
                                 quantization_status="initialized")
    apply_quantization_config(model, qconfig)

    # ---- 1) build calib + capture layer-0 inputs (abort forward after layer 0 => cheap, no OOM) ----
    ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    idx = list(range(len(ds))); random.shuffle(idx)
    cap = []  # list of (args, kwargs) captured at layer 0 input, per sample

    def hook(_m, args, kwargs):
        cap.append(([a.detach() if torch.is_tensor(a) else a for a in args],
                    {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}))
        raise _Stop()
    h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    got = 0
    for i in idx:
        if got >= NUM:
            break
        ex = ds[i]
        msgs = [{"role": "user", "content": ex["instruction"]}, {"role": "assistant", "content": ex["response"]}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = ex["instruction"] + "\n" + ex["response"]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids
        try:
            with torch.no_grad():
                model(input_ids=ids)
        except _Stop:
            got += 1
    h.remove()
    log(f"captured {len(cap)} layer-0 input sets")

    # ---- 2) per-layer GPTQ ----
    def targets_in(layer):
        out = []
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear) and hasattr(mod, "quantization_scheme"):
                out.append((name, mod))
        return out

    cur = cap  # inputs to the current layer (list of (args, kwargs)); args[0] = hidden_states
    for li in range(n_layers):
        layer = layers[li].to(DEV)
        tgts = targets_in(layer)
        log(f"layer {li}: {len(tgts)} target Linears")
        H = {name: None for name, _ in tgts}
        ns = {name: torch.zeros(1, device=DEV) for name, _ in tgts}
        handles = []

        def mk(name, mod):
            def fwd_hook(_m, inp, _out):
                x = inp[0]
                if H[name] is None:
                    H[name] = make_empty_hessian(mod, device=DEV)
                H[name], ns[name] = accumulate_hessian(x, mod, H[name], ns[name])
            return fwd_hook
        for name, mod in tgts:
            handles.append(mod.register_forward_hook(mk(name, mod)))

        # forward calib through the (unquantized) layer to accumulate Hessians
        with torch.no_grad():
            for args, kwargs in cur:
                args = [a.to(DEV) if torch.is_tensor(a) else a for a in args]
                kwargs = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                layer(*args, **kwargs)
        for hd in handles:
            hd.remove()

        # quantize each target with GPTQ
        for name, mod in tgts:
            loss, q_w, scale, zp, g_idx = quantize_weight(mod, QARGS, H[name])
            mod.weight.data = q_w.to(mod.weight.dtype)
            mod.weight_scale.data = scale.to(mod.weight_scale.dtype)
            if zp is not None and hasattr(mod, "weight_zero_point"):
                mod.weight_zero_point.data = zp.to(mod.weight_zero_point.dtype)
            H[name] = None
        torch.cuda.empty_cache()

        # forward the QUANTIZED layer to produce next-layer inputs
        nxt = []
        with torch.no_grad():
            for args, kwargs in cur:
                a2 = [a.to(DEV) if torch.is_tensor(a) else a for a in args]
                k2 = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                out = layer(*a2, **k2)
                hs = out[0] if isinstance(out, (tuple, list)) else out
                na = list(args); na[0] = hs.detach().cpu()
                nxt.append((na, {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in kwargs.items()}))
        cur = nxt
        layers[li] = layer.cpu()
        torch.cuda.empty_cache()
        log(f"layer {li} done")

    # ---- 3) save compressed-tensors W4A16 ----
    log(f"saving to {OUT}")
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    try:
        AutoProcessor.from_pretrained(M).save_pretrained(OUT)
    except Exception as e:
        log(f"no processor: {type(e).__name__}")
    qc = json.load(open(os.path.join(OUT, "config.json"))).get("quantization_config", {})
    log(f"DONE format={qc.get('format')} groups={list(qc.get('config_groups', {}).keys())}")
    print("DGQ_MANUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
