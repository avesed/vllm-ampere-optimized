#!/usr/bin/env python3
"""Single-stream decode/prefill throughput for one (parallel-mode, K) config. Env: M, PARMODE=tp|pp,
K (0=no spec). Reports decode tok/s (1000/TPOT), TTFT/prefill tok/s, and MTP accept-len. Run once per
config on the 2x3090 box. decode=1000/TPOT and prefill=input_len/TTFT per the project's tok/s units."""
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
    par = os.environ["PARMODE"]
    k = int(os.environ.get("K", "2"))
    kw = dict(tensor_parallel_size=2) if par == "tp" else dict(pipeline_parallel_size=2)
    spec = {"method": "mtp", "num_speculative_tokens": k} if k > 0 else None
    llm = LLM(model=m, enforce_eager=False, max_model_len=8192, gpu_memory_utilization=0.92,
              speculative_config=spec, disable_log_stats=False, trust_remote_code=True,
              max_num_seqs=1, **kw)
    prompt = "请详细介绍中国从古至今的历史发展，分朝代阶段讲述，并分析各阶段的政治、经济与文化特点及其影响。"
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512, ignore_eos=True)
    o = llm.generate([prompt], sp)  # warmup
    plen = len(o[0].prompt_token_ids)
    t0 = time.perf_counter()
    ntok = 0
    for _ in range(5):
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
    print(f"RESULT par={par} k={k} plen={plen}: "
          f"decode={decode:.1f} tok/s (TPOT={tpot*1000 if tpot else 0:.2f}ms) | "
          f"TTFT={ttft*1000 if ttft else 0:.0f}ms (prefill={prefill:.0f} tok/s) | "
          f"accept_len={acclen:.2f}", flush=True)


if __name__ == "__main__":
    main()
