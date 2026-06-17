"""Profile ONE phase of a real W4A8 vLLM forward using vLLM's built-in torch profiler
(VLLM_TORCH_PROFILER_DIR + start_profile/stop_profile). Unlike a driver-process torch.profiler,
this profiles the WORKER processes too, so it captures tp=2 GPU kernels. Emits a kineto Chrome
trace per rank into --prof-dir, which analyze_torch_prof.py parses for bubble + kernel breakdown.
nsys-free (the bundled nsys can't convert its .qdstrm).

Usage:
    VLLM_TORCH_PROFILER_DIR is set internally from --prof-dir BEFORE the engine starts.
    python torch_prof_phase.py --model <w4a8> --phase decode [--eager] --prof-dir /tmp/prof_x
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--prof-dir", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--prof-iters", type=int, default=1,
                    help="repeat the profiled generate N times (prefill needs >=5 steps to fill a profiler cycle)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    os.makedirs(a.prof_dir, exist_ok=True)

    import torch
    from vllm import LLM, SamplingParams

    if a.phase == "prefill":
        win = a.prompt_len + 512   # margin: the repeated-word prompt tokenizes a bit over prompt_len
        mnbt = max(a.batch * win, win)
    else:
        a.prompt_len = min(a.prompt_len, 64)
        win = a.prompt_len + a.decode_tokens + 64
        mnbt = max(a.batch * a.prompt_len, 2048)

    # This build deprecated VLLM_TORCH_PROFILER_DIR; profiling is configured via profiler_config.
    kw = dict(model=a.model, tensor_parallel_size=a.tp, enforce_eager=a.eager,
              max_model_len=win, max_num_seqs=max(a.batch, 1),
              max_num_batched_tokens=mnbt, gpu_memory_utilization=a.gpu_mem,
              profiler_config={"profiler": "torch", "torch_profiler_dir": a.prof_dir,
                               "torch_profiler_with_stack": False})
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

    llm.generate(prompts, warm)  # warmup (graph capture etc.), not profiled
    torch.cuda.synchronize()

    llm.start_profile()
    t = time.perf_counter()
    for _ in range(a.prof_iters):
        llm.generate(prompts, sp)
    dt = time.perf_counter() - t
    llm.stop_profile()
    time.sleep(2)  # let kineto flush traces to disk
    print(f"PHASE={a.phase} eager={a.eager} batch={a.batch} wall={dt*1000:.1f}ms", flush=True)
    print("WORKLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
