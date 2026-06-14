"""Batch-size sweep: aggregate decode throughput at B=16/64/256 to find where the
compute-bound crossover happens — theory says int8 (W8A8) overtakes int4 (W4A16)
once batch pushes arithmetic intensity past the 3090 roofline ridge (~76 fp16).
Short context + adaptive: a batch that OOMs is caught and reported, not fatal.
Usage: vllm_batch_sweep.py <model> <tag>"""
import sys, time


def main():
    path, tag = sys.argv[1], sys.argv[2]
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    eager = False
    def build(e):
        return LLM(model=path, tensor_parallel_size=2, max_model_len=2048,
                   max_num_seqs=256, gpu_memory_utilization=0.92, enforce_eager=e,
                   # capture ONLY the sizes we bench -> small cudagraph mem (was OOM capturing 0..512)
                   compilation_config={"cudagraph_capture_sizes": [16, 64, 256]},
                   limit_mm_per_prompt={"image": 1, "video": 0})
    try:
        llm = build(False)
    except Exception as ex:
        print(f"[{tag}] cudagraph build failed ({type(ex).__name__}); eager", flush=True)
        eager = True
        llm = build(True)
    print(f"[{tag}] cudagraph={'OFF' if eager else 'ON'}", flush=True)

    tok = AutoTokenizer.from_pretrained(path)
    p = tok.apply_chat_template([{"role": "user", "content": "Tell me a few facts about France."}],
                                tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=128, ignore_eos=True)
    llm.generate([p], SamplingParams(max_tokens=8, temperature=0))  # warmup

    for B in [16, 64, 256]:
        try:
            t = time.time(); outs = llm.generate([p] * B, sp); dt = time.time() - t
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            print(f"[{tag}] BATCH {B:>3}: {n/dt:>7.1f} tok/s aggregate   ({n} tok / {dt:.1f}s)", flush=True)
        except Exception as ex:
            print(f"[{tag}] BATCH {B:>3}: FAILED  {type(ex).__name__}: {str(ex)[:80]}", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
