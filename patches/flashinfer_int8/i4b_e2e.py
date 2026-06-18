#!/usr/bin/env python3
"""I-4b e2e PoC harness: measure int8-QK FlashInfer prefill vs fp16-FA prefill on the REAL
Qwen3.5-9B-W4A8 (hd256 hybrid, 8/32 full-attn layers, GQA g4).

Two backends (env TEST_BACKEND):
  * FA    : stock FlashAttention (fp16) baseline — no override.
  * INT8QK: route the eligible pure-fresh hd256 prefill through the int8-QK FlashInfer kernel
            (per-token int8 Q/K + smooth_k, per-tensor int8 V, scale_q/scale_k tensors, fp16 PV).
            Requires the flashinfer package to be PRE-PATCHED (i1_apply -> i4_apply ->
            i4_compute_qk) in this container BEFORE this script runs (do it in a bash step).

Measures: prefill TTFT (median of N runs at TARGET_LEN), needle-retrieval correctness, and a
Chinese chain-of-thought coherence check. Reports the int8-QK fire count (==# full-attn layers
* #seqs that used the int8 path).

Env: TEST_BACKEND, TARGET_LEN, N_RUNS (default 3), GPU_MEM (default 0.60), MODEL (default /model),
     RUN_ZHCOT (default 1), TP (default 1).
"""
import os, time, statistics

BACKEND = os.environ.get("TEST_BACKEND", "FA")
TARGET = int(os.environ.get("TARGET_LEN", "65536"))
N_RUNS = int(os.environ.get("N_RUNS", "3"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.60"))
MODEL = os.environ.get("MODEL", "/model")
RUN_ZHCOT = os.environ.get("RUN_ZHCOT", "1") == "1"
TP = int(os.environ.get("TP", "1"))
# CHUNKED prefill: when MAX_BATCHED_TOKENS < TARGET, vLLM splits the prompt into chunks of
# this many tokens -> chunk-0 is fresh, chunks 1..N have a cached prefix. 0 (default) => single
# step (max_num_batched_tokens = TARGET+512, the old I-4b behavior).
MAX_BATCHED = int(os.environ.get("MAX_BATCHED_TOKENS", "0"))

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

if BACKEND == "INT8QK":
    import vllm.v1.attention.backends.int8qk_backend as I
    from vllm.v1.attention.backends import registry as R
    _PATH = "vllm.v1.attention.backends.int8qk_backend.Int8QKAttentionBackend"
    R._ATTN_OVERRIDES[R.AttentionBackendEnum.FLASH_ATTN] = _PATH
    # name validation: get_name() must return a real enum name or vLLM raises "Unknown backend".
    I.Int8QKAttentionBackend.get_name = staticmethod(lambda: "FLASH_ATTN")
    FIRE = I.INT8QK_FIRE
    print(">>> INT8QK override installed (FLASH_ATTN->Int8QK, name=FLASH_ATTN)", flush=True)
else:
    FIRE = {"calls": 0, "seqs": 0, "fresh": 0, "cached": 0}

from vllm import LLM, SamplingParams

# Chunked vs single-step: max_num_batched_tokens governs the chunk size. CHUNKED requires
# enable_chunked_prefill=True (default-on in v0.23) + a small budget; single-step uses a budget
# >= the whole prompt so it never splits (the original I-4b behavior; MAX_BATCHED_TOKENS=0).
mbt = MAX_BATCHED if MAX_BATCHED > 0 else (TARGET + 512)
chunked = MAX_BATCHED > 0 and MAX_BATCHED < TARGET
print(f">>> loading {MODEL} backend={BACKEND} target={TARGET} tp={TP} "
      f"max_batched_tokens={mbt} chunked={chunked}", flush=True)
llm = LLM(
    model=MODEL, max_model_len=TARGET + 512, max_num_batched_tokens=mbt,
    enable_chunked_prefill=True,
    enforce_eager=True, max_num_seqs=1, gpu_memory_utilization=GPU_MEM,
    enable_prefix_caching=False, tensor_parallel_size=TP,
    limit_mm_per_prompt={"image": 0, "video": 0}, trust_remote_code=True,
)
tok = llm.get_tokenizer()

# ---- needle-in-haystack prompt at TARGET_LEN ----
sent = "The committee reviewed the quarterly logistics report and found nothing unusual. "
sids = tok.encode(sent, add_special_tokens=False)
base = (sids * (TARGET // len(sids) + 8))[:TARGET]
needle = tok.encode(" Remember this: the vault access code is 73914. ", add_special_tokens=False)
q = tok.encode("\nQuestion: What is the vault access code? Answer with only the number.\n",
               add_special_tokens=False)
mid = TARGET // 2
ids = base[:mid] + needle + base[mid:TARGET - len(needle) - len(q)] + q
sp = SamplingParams(temperature=0.0, max_tokens=24)

# ---- warmup (build JIT module / cudagraph; also the first-call JIT compile for INT8QK) ----
print(">>> warmup run (JIT build for INT8QK) ...", flush=True)
fire0 = FIRE["calls"]
fr0, ca0 = FIRE.get("fresh", 0), FIRE.get("cached", 0)
tw = time.time()
_ = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp)
print(f">>> warmup wall={time.time()-tw:.2f}s int8qk_calls_during_warmup={FIRE['calls']-fire0} "
      f"fresh={FIRE.get('fresh',0)-fr0} cached={FIRE.get('cached',0)-ca0}", flush=True)

# ---- timed runs (prefill TTFT = wall of the single prefill+1tok; generate 1 tok to isolate TTFT) ----
sp_ttft = SamplingParams(temperature=0.0, max_tokens=1)
walls = []
fire_before = FIRE["calls"]
fr_before, ca_before = FIRE.get("fresh", 0), FIRE.get("cached", 0)
for r in range(N_RUNS):
    t0 = time.time()
    out = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp_ttft)
    walls.append(time.time() - t0)
fire_ttft = FIRE["calls"] - fire_before
fresh_ttft = FIRE.get("fresh", 0) - fr_before
cached_ttft = FIRE.get("cached", 0) - ca_before
ttft_med = statistics.median(walls)

# ---- needle correctness (full 24-tok answer) ----
out = llm.generate(prompts=[{"prompt_token_ids": ids}], sampling_params=sp)
ans = out[0].outputs[0].text
needle_hit = "73914" in ans
fire_needle = FIRE["calls"]

# ---- zh-CoT coherence check (short context; thinking-safe sampling temp=0.6/top_p=0.95) ----
zh_text = ""
zh_ok = None
if RUN_ZHCOT:
    zh_prompt = ("请一步一步推理：一个水池有两个进水管和一个排水管。"
                 "甲管单独注满需要4小时，乙管单独注满需要6小时，排水管单独排空需要12小时。"
                 "三管同时打开，多少小时能注满水池？请给出推理过程和最终答案。")
    msgs = [{"role": "user", "content": zh_prompt}]
    zp = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    zsp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024)
    zout = llm.generate(prompts=[zp], sampling_params=zsp)
    zh_text = zout[0].outputs[0].text
    # coherence heuristics: contains CJK, no degenerate repetition, mentions the answer 3
    cjk = sum(1 for ch in zh_text if "一" <= ch <= "鿿")
    # crude repetition guard: max run of an identical 8-char window
    rep = False
    if len(zh_text) > 64:
        w = {}
        for i in range(0, len(zh_text) - 8, 8):
            seg = zh_text[i:i + 8]
            w[seg] = w.get(seg, 0) + 1
        rep = max(w.values()) > 12 if w else False
    has_ans = ("3" in zh_text)
    zh_ok = (cjk > 40) and (not rep) and has_ans

print("\n================ I4B RESULT ================", flush=True)
print(f"backend={BACKEND} len={TARGET} tp={TP} prompt_tok={len(ids)} "
      f"max_batched_tokens={mbt} chunked={chunked}", flush=True)
print(f"TTFT walls(s)={['%.3f'%w for w in walls]}  median={ttft_med:.3f}s", flush=True)
print(f"int8qk fire: total_calls={FIRE['calls']} during_{N_RUNS}_ttft_runs={fire_ttft} "
      f"(== {fire_ttft//max(N_RUNS,1)}/run)  during_needle_run(cumulative)={fire_needle}", flush=True)
print(f"int8qk fire SPLIT during_{N_RUNS}_ttft_runs: fresh={fresh_ttft} cached={cached_ttft} "
      f"(per_run fresh={fresh_ttft//max(N_RUNS,1)} cached={cached_ttft//max(N_RUNS,1)}); "
      f"cached>0 PROVES int8 fired on cached-prefix chunks (not just chunk-0)", flush=True)
print(f"needle_hit={needle_hit} ans={ans!r}", flush=True)
if RUN_ZHCOT:
    cjk = sum(1 for ch in zh_text if "一" <= ch <= "鿿")
    print(f"zh_cot_ok={zh_ok} cjk_chars={cjk} len={len(zh_text)}", flush=True)
    print(f"zh_cot_text[:600]={zh_text[:600]!r}", flush=True)
print(f"=== I4B_DONE backend={BACKEND} len={TARGET} ttft_med={ttft_med:.3f} "
      f"needle={needle_hit} zhcot={zh_ok} fire_per_run={fire_ttft//max(N_RUNS,1)}", flush=True)
