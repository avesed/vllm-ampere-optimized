"""Decode-share profiler with a BATCH SWEEP, loading the model ONCE (no per-batch reload).
For a hybrid (GatedDeltaNet) model it answers: at decode, what fraction of NON-COMM GPU kernel
time is the linear-attn path vs the (W4A8) Linear GEMMs, as batch grows? Decision rule (from the
linear-attn roadmap): if attn_linear stays <5% of non-comm even at batch 32+, decode kernel tuning
is dead; if it climbs to ~20-30%, a tuned config datafile is worth it.

One LLM load; per batch: start_profile -> decode generate -> stop_profile -> move that batch's
kineto traces into <prof-dir>/b<batch>/ so analyze_torch_prof.py can read each batch separately.

Usage:
    python prof_decode_batchsweep.py --model <w4a8> --tp 2 --prof-dir /home/coder/prof_bsweep \
        --batches 1,16,32,64 --decode-tokens 48
"""
import argparse
import glob
import os
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prof-dir", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--batches", default="1,16,32,64")
    ap.add_argument("--prompt-len", type=int, default=48)
    ap.add_argument("--decode-tokens", type=int, default=48)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()

    batches = [int(x) for x in a.batches.split(",")]
    os.makedirs(a.prof_dir, exist_ok=True)

    import torch
    from vllm import LLM, SamplingParams

    win = a.prompt_len + a.decode_tokens + 64
    max_b = max(batches)
    kw = dict(
        model=a.model, tensor_parallel_size=a.tp, enforce_eager=True,  # eager: kernel-TIME share is bubble-independent
        max_model_len=win, max_num_seqs=max_b, max_num_batched_tokens=max(max_b * a.prompt_len, 2048),
        gpu_memory_utilization=a.gpu_mem,
        profiler_config={"profiler": "torch", "torch_profiler_dir": a.prof_dir,
                         "torch_profiler_with_stack": False},
    )
    try:
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    except (TypeError, ValueError, AssertionError):
        llm = LLM(**kw)

    prompt = "The quick brown fox jumps over the lazy dog. " * ((a.prompt_len // 9) + 1)

    def new_traces(seen):
        cur = set(glob.glob(os.path.join(a.prof_dir, "*.json*")))
        return cur - seen, cur

    seen = set(glob.glob(os.path.join(a.prof_dir, "*.json*")))
    for b in batches:
        prompts = [prompt] * b
        sp = SamplingParams(max_tokens=a.decode_tokens, temperature=0, ignore_eos=True)
        llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))  # warmup this batch
        torch.cuda.synchronize()
        llm.start_profile()
        t = time.perf_counter()
        llm.generate(prompts, sp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        llm.stop_profile()
        time.sleep(2)  # flush kineto
        sub = os.path.join(a.prof_dir, f"b{b}")
        os.makedirs(sub, exist_ok=True)
        fresh, seen = new_traces(seen)
        for fp in fresh:
            shutil.move(fp, os.path.join(sub, os.path.basename(fp)))
        tok = b * a.decode_tokens
        print(f"BATCH={b} wall={dt*1000:.0f}ms decode_tok={tok} tok/s={tok/dt:.1f} traces={len(fresh)} -> {sub}", flush=True)

    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
