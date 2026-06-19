#!/usr/bin/env python3
"""I-6: validate the int8-QK backend fires in ALL workers under TP/PP via the general_plugins
entry-point (multiprocessing ON — the real serving path, NOT VLLM_ENABLE_V1_MULTIPROCESSING=0).

Unlike i4b_e2e.py (in-process monkeypatch), this installs NOTHING in-process: the override is
expected to come from the `vllm.general_plugins` entry-point (register_int8qk, gated by
VLLM_INT8QK=1) which load_general_plugins() runs in the engine core AND every TP/PP worker.

Per-worker fire evidence: each worker process logs "INT8QK FIRED rank=tpX/ppY pid=..." the first
time the int8 path runs in it. Parse the launcher-captured stderr for one line per expected rank.
needle + zh-coherence assert correctness.

NOTE: with multiprocessing ON the engine spawns child processes that re-import this module, so the
engine work MUST live under `if __name__ == "__main__":` (else recursive-spawn RuntimeError).

Env: PARALLEL=tp2|pp2|tp1, TARGET_LEN (default 8192), MODEL (default /model), N_RUNS (default 2),
     GPU_MEM (default 0.85), MAX_BATCHED_TOKENS (0 => single-step), RUN_ZHCOT (default 1).
"""
import os
import statistics
import time


def main() -> None:
    PARALLEL = os.environ.get("PARALLEL", "tp2").lower()
    TARGET = int(os.environ.get("TARGET_LEN", "8192"))
    N_RUNS = int(os.environ.get("N_RUNS", "2"))
    GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
    MODEL = os.environ.get("MODEL", "/model")
    RUN_ZHCOT = os.environ.get("RUN_ZHCOT", "1") == "1"
    MAX_BATCHED = int(os.environ.get("MAX_BATCHED_TOKENS", "0"))

    if PARALLEL == "tp2":
        TP, PP, EXPECT_RANKS = 2, 1, 2
    elif PARALLEL == "pp2":
        TP, PP, EXPECT_RANKS = 1, 2, 2
    else:
        TP, PP, EXPECT_RANKS = 1, 1, 1

    # The whole point: multiprocessing ON (default). Do NOT set
    # VLLM_ENABLE_V1_MULTIPROCESSING=0.
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    print(
        f">>> I6 PARALLEL={PARALLEL} TP={TP} PP={PP} target={TARGET} "
        f"VLLM_INT8QK={os.environ.get('VLLM_INT8QK', '0')} "
        f"mp={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', 'default-on')}",
        flush=True,
    )

    from vllm import LLM, SamplingParams

    mbt = MAX_BATCHED if MAX_BATCHED > 0 else (TARGET + 512)
    chunked = MAX_BATCHED > 0 and MAX_BATCHED < TARGET
    llm = LLM(
        model=MODEL, max_model_len=TARGET + 512, max_num_batched_tokens=mbt,
        enable_chunked_prefill=True, enforce_eager=True, max_num_seqs=1,
        gpu_memory_utilization=GPU_MEM, enable_prefix_caching=False,
        tensor_parallel_size=TP, pipeline_parallel_size=PP,
        limit_mm_per_prompt={"image": 0, "video": 0}, trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    # needle-in-haystack at TARGET
    sent = "The committee reviewed the quarterly logistics report and found nothing unusual. "
    sids = tok.encode(sent, add_special_tokens=False)
    base = (sids * (TARGET // len(sids) + 8))[:TARGET]
    needle = tok.encode(" Remember this: the vault access code is 73914. ",
                        add_special_tokens=False)
    q = tok.encode(
        "\nQuestion: What is the vault access code? Answer with only the number.\n",
        add_special_tokens=False,
    )
    mid = TARGET // 2
    ids = base[:mid] + needle + base[mid:TARGET - len(needle) - len(q)] + q

    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    sp24 = SamplingParams(temperature=0.0, max_tokens=24)

    print(">>> warmup", flush=True)
    _ = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp1)

    walls = []
    for _ in range(N_RUNS):
        t0 = time.time()
        _ = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp1)
        walls.append(time.time() - t0)
    ttft_med = statistics.median(walls)

    out = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp24)
    ans = out[0].outputs[0].text
    needle_hit = "73914" in ans

    zh_ok = None
    zh_text = ""
    if RUN_ZHCOT:
        zh_prompt = ("请一步一步推理：一个水池有两个进水管和一个排水管。"
                     "甲管单独注满需要4小时，乙管单独注满需要6小时，排水管单独排空需要12小时。"
                     "三管同时打开，多少小时能注满水池？请给出推理过程和最终答案。")
        zp = tok.apply_chat_template([{"role": "user", "content": zh_prompt}],
                                     tokenize=False, add_generation_prompt=True)
        zout = llm.generate(
            prompts=[zp],
            sampling_params=SamplingParams(temperature=0.6, top_p=0.95, max_tokens=512),
        )
        zh_text = zout[0].outputs[0].text
        cjk = sum(1 for ch in zh_text if "一" <= ch <= "鿿")
        zh_ok = cjk > 30 and ("3" in zh_text)

    print("\n================ I6 RESULT ================", flush=True)
    print(f"parallel={PARALLEL} tp={TP} pp={PP} len={TARGET} chunked={chunked} "
          f"VLLM_INT8QK={os.environ.get('VLLM_INT8QK', '0')}", flush=True)
    print(f"TTFT walls(s)={['%.3f' % w for w in walls]} median={ttft_med:.3f}s", flush=True)
    print(f"needle_hit={needle_hit} ans={ans!r}", flush=True)
    if RUN_ZHCOT:
        print(f"zh_cot_ok={zh_ok} zh[:200]={zh_text[:200]!r}", flush=True)
    print(f"=== I6_DONE parallel={PARALLEL} ttft_med={ttft_med:.3f} needle={needle_hit} "
          f"zhcot={zh_ok} expect_ranks={EXPECT_RANKS} ===", flush=True)


if __name__ == "__main__":
    main()
