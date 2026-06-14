"""Part B: load a quantized model in vLLM (cudagraph ON), confirm the kernel/scheme,
and measure single-stream decode, batch-16 decode (aggregate), and prefill throughput.
Memory-conservative (max_model_len / max_num_seqs / mem-util) because int8 W8A8 +
cudagraph capture can OOM on 24GB. Falls back to enforce_eager if cudagraph OOMs.
Usage: vllm_verify.py <model> <tag> [max_model_len=4096] [max_num_seqs=32]"""
import sys, time


def build(path, mml, mns, eager):
    from vllm import LLM
    return LLM(model=path, tensor_parallel_size=2, max_model_len=mml,
               max_num_seqs=mns, enforce_eager=eager, gpu_memory_utilization=0.90,
               limit_mm_per_prompt={"image": 1, "video": 0})


def main():
    path, tag = sys.argv[1], sys.argv[2]
    mml = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    mns = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    from vllm import SamplingParams
    from transformers import AutoTokenizer

    eager = False
    t0 = time.time()
    try:
        llm = build(path, mml, mns, eager=False)               # cudagraph ON
    except Exception as e:
        print(f"[vv:{tag}] cudagraph build FAILED ({type(e).__name__}: {str(e)[:80]}); retrying enforce_eager", flush=True)
        eager = True
        llm = build(path, mml, mns, eager=True)
    print(f"[vv:{tag}] load {time.time()-t0:.1f}s  cudagraph={'OFF(eager)' if eager else 'ON'}", flush=True)

    tok = AutoTokenizer.from_pretrained(path)
    p = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}],
                                tokenize=False, add_generation_prompt=True)
    dec = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, ignore_eos=True)
    llm.generate([p], SamplingParams(max_tokens=8, temperature=0))  # warmup

    # 1) SINGLE-STREAM decode (batch 1) -> comparable to interactive single-user tok/s
    t = time.time(); o = llm.generate([p], dec); dt = time.time() - t
    n = len(o[0].outputs[0].token_ids)
    print(f"[vv:{tag}] SINGLE-STREAM decode = {n/dt:.1f} tok/s  ({n} tok / {dt:.1f}s, batch 1)", flush=True)
    print(f"[vv:{tag}] sample: {o[0].outputs[0].text[:120]!r}", flush=True)

    # 2) BATCH-16 decode (aggregate throughput)
    t = time.time(); outs = llm.generate([p] * 16, dec); dt = time.time() - t
    n = sum(len(x.outputs[0].token_ids) for x in outs)
    print(f"[vv:{tag}] BATCH16 decode = {n/dt:.1f} tok/s  ({n} tok / {dt:.1f}s, aggregate)", flush=True)

    # 3) PREFILL (compute-bound) -> where int8 should win
    longu = "Read the passage and summarize it.\n\n" + ("The quick brown fox jumps over the lazy dog. " * 200)
    lp = tok.apply_chat_template([{"role": "user", "content": longu}], tokenize=False, add_generation_prompt=True)
    nin = sum(len(tok(x).input_ids) for x in [lp] * 8)
    t = time.time(); llm.generate([lp] * 8, SamplingParams(max_tokens=1, temperature=0)); dt = time.time() - t
    print(f"[vv:{tag}] PREFILL = {nin/dt:.0f} tok/s  ({nin} in-tok / {dt:.1f}s, 8 prompts)", flush=True)
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
