#!/usr/bin/env python3
"""Compare two quants of the same model: decode tok/s (1000/TPOT), prefill tok/s (long prompt /
TTFT), and MTP accept-len. Env: M, K (default 2). TP2, cudagraph on, long ~3.5k-token prompt so the
prefill (int8-act IMMA on sm_86) shows. One process per quant."""
import os
import time

from vllm import LLM, SamplingParams


def hmean(llm, name):
    for x in llm.get_metrics():
        if getattr(x, "name", "") == name and getattr(x, "count", 0):
            return x.sum / x.count
    return None


def gval(llm, name):
    for x in llm.get_metrics():
        if getattr(x, "name", "") == name:
            return getattr(x, "value", None)
    return None


def main():
    m = os.environ["M"]
    k = int(os.environ.get("K", "2"))
    spec = {"method": "mtp", "num_speculative_tokens": k} if k > 0 else None
    llm = LLM(model=m, enforce_eager=False, max_model_len=8192, gpu_memory_utilization=0.92,
              speculative_config=spec, disable_log_stats=False, trust_remote_code=True,
              max_num_seqs=1, tensor_parallel_size=2)
    para = ("中国历史悠久，文明源远流长。从夏商周的奠基，到秦汉的大一统，再到唐宋的繁荣、"
            "元明清的转型与近现代的变革，每个阶段都有独特的政治制度、经济形态和文化成就。")
    rep = int(os.environ.get("REPEAT", "90"))
    prompt = "请仔细阅读以下材料并完成总结任务：\n" + para * rep + "\n请用中文系统地总结上述材料的核心内容与各阶段特点。"
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, ignore_eos=True)
    o = llm.generate([prompt], sp)  # warmup
    plen = len(o[0].prompt_token_ids)
    t0 = time.perf_counter()
    ntok = 0
    for _ in range(3):
        out = llm.generate([prompt], sp)
        ntok += len(out[0].outputs[0].token_ids)
    dt = time.perf_counter() - t0
    ttft = hmean(llm, "vllm:time_to_first_token_seconds")
    tpot = hmean(llm, "vllm:time_per_output_token_seconds")
    nd = gval(llm, "vllm:spec_decode_num_drafts")
    na = gval(llm, "vllm:spec_decode_num_accepted_tokens")
    acclen = (1 + na / nd) if (k > 0 and nd) else 1.0
    decode = (1.0 / tpot) if tpot else (ntok / dt)
    prefill = (plen / ttft) if ttft else 0.0
    label = os.environ.get("LABEL", m.split('/')[-1])
    print(f"RESULT {label} k={k}: plen={plen} | "
          f"decode={decode:.1f} tok/s (TPOT={tpot*1000 if tpot else 0:.2f}ms) | "
          f"prefill={prefill:.0f} tok/s (TTFT={ttft*1000 if ttft else 0:.0f}ms) | "
          f"accept_len={acclen:.2f}", flush=True)


if __name__ == "__main__":
    main()
