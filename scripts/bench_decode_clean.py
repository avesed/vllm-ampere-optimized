#!/usr/bin/env python3
"""Metric-independent decode-rate measurement: prefix-caching OFF + manual timing (t(max_tokens=N)
- t(max_tokens=1) = pure N-1 decode steps). Avoids any TPOT-metric contamination at long context.
Env: M, REPEAT (prompt length), TP (default 2). No MTP."""
import os
import time

from vllm import LLM, SamplingParams


def main():
    m = os.environ["M"]
    rep = int(os.environ.get("REPEAT", "90"))
    tp = int(os.environ.get("TP", "2"))
    llm = LLM(model=m, enforce_eager=False, max_model_len=20000, gpu_memory_utilization=0.92,
              trust_remote_code=True, max_num_seqs=1, tensor_parallel_size=tp,
              enable_prefix_caching=False)
    para = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
            "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
    prompt = "请仔细阅读以下材料并总结：\n" + para * rep + "\n请用中文系统总结上述材料。"
    spN = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=257, ignore_eos=True)
    sp1 = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1, ignore_eos=True)
    o = llm.generate([prompt], spN)  # warmup
    plen = len(o[0].prompt_token_ids)
    t = time.perf_counter(); llm.generate([prompt], sp1); t1 = time.perf_counter() - t
    ts = []
    for _ in range(2):
        t = time.perf_counter(); o = llm.generate([prompt], spN); ts.append(time.perf_counter() - t)
    tN = min(ts)
    nN = len(o[0].outputs[0].token_ids)
    decode = (nN - 1) / (tN - t1) if tN > t1 else 0.0
    print(f"CLEAN plen={plen} tp={tp} | ttft(prefill+1)={t1*1000:.0f}ms | "
          f"decode={decode:.1f} tok/s ({nN-1} decode-tok in {tN-t1:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
