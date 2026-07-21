#!/usr/bin/env python3
"""Offline kernel profile of a ~16k-context decode (in-process engine so torch.profiler sees the
GPU kernels). Env: M, K (0=noMTP, 2=MTP). Run with VLLM_ENABLE_V1_MULTIPROCESSING=0 + enforce_eager.
Prints top CUDA kernels by total time — compare K=0 vs K=2 to see what MTP adds at long context.
Kernel-ID only (not a tok/s measurement)."""
import os

from torch.profiler import ProfilerActivity, profile
from vllm import LLM, SamplingParams


def main():
    m = os.environ["M"]
    k = int(os.environ.get("K", "0"))
    spec = {"method": "mtp", "num_speculative_tokens": k} if k > 0 else None
    llm = LLM(model=m, enforce_eager=True, max_model_len=18000, gpu_memory_utilization=0.85,
              max_num_seqs=16, tensor_parallel_size=1, trust_remote_code=True,
              speculative_config=spec)
    para = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
            "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
    prompt = "请仔细阅读并总结：\n" + para * 354 + "\n请用中文系统总结。"
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=24, ignore_eos=True)
    llm.generate([prompt], sp)  # warmup
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        llm.generate([prompt], sp)
    print(f"==== K={k} (prefill+24 decode steps @16k) top CUDA kernels by total time ====", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=24))


if __name__ == "__main__":
    main()
