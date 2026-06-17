"""nsys workload: run ONE phase (prefill or decode) of a real W4A8 vLLM forward at batch,
so an nsys timeline can measure (a) inter-kernel BUBBLE fraction (eager vs cudagraph) and
(b) the GEMM / attention / other kernel-time split (→ SageAttention attention share).

nsys only TRACES (no kernel replay), so unlike ncu it does NOT hang on vLLM's multiprocess
startup. We profile the whole short run; warmup matches the phase so the trace is phase-pure.

Usage (under nsys):
    nsys profile -t cuda -o out -f true python nsys_phase.py --model <w4a8> --phase decode [--eager]
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    if a.phase == "prefill":
        win = a.prompt_len + 64
        mnbt = max(a.batch * a.prompt_len, win)
    else:  # decode: short prompt, long generation
        a.prompt_len = min(a.prompt_len, 64)
        win = a.prompt_len + a.decode_tokens + 64
        mnbt = max(a.batch * a.prompt_len, 2048)

    kw = dict(model=a.model, tensor_parallel_size=a.tp, enforce_eager=a.eager,
              max_model_len=win, max_num_seqs=max(a.batch, 1),
              max_num_batched_tokens=mnbt, gpu_memory_utilization=a.gpu_mem)
    try:
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    except (TypeError, ValueError, AssertionError):
        llm = LLM(**kw)

    prompt = "The quick brown fox jumps over the lazy dog. " * ((a.prompt_len // 9) + 1)
    prompts = [prompt] * a.batch

    if a.phase == "prefill":
        sp = SamplingParams(max_tokens=1, temperature=0)
        warm = SamplingParams(max_tokens=1, temperature=0)
    else:
        sp = SamplingParams(max_tokens=a.decode_tokens, temperature=0, ignore_eos=True)
        warm = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)

    llm.generate(prompts, warm)          # warmup (cudagraph capture etc.) — same phase shape
    torch.cuda.synchronize()

    torch.cuda.profiler.start()          # capture-range hint (best-effort; whole run is short anyway)
    t = time.perf_counter()
    llm.generate(prompts, sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    torch.cuda.profiler.stop()
    print(f"PHASE={a.phase} eager={a.eager} batch={a.batch} wall={dt*1000:.1f}ms", flush=True)
    print("WORKLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
