import sys
def main():
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    M="/home/coder/models/Qwen3.5-9B-w4a16"
    tok=AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    llm=LLM(model=M, max_model_len=40000, gpu_memory_utilization=0.90, enforce_eager=True,
            dtype="float16", attention_backend="FLASHINFER", trust_remote_code=True,
            limit_mm_per_prompt={"image":0,"video":0})
    NEEDLE=" The secret passcode that you must remember is 7492. "
    filler="The grass is green, the sky is blue, and the river flows to the sea. "
    res={}
    for L in [8000, 32000]:
        n=L//len(tok(filler).input_ids); half=n//2
        body=filler*half + NEEDLE + filler*half
        msg=[{"role":"user","content": body + "\n\nWhat is the secret passcode? Reply with only the number."}]
        try:
            prompt=tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            prompt=tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ntok=len(tok(prompt).input_ids)
        out=llm.generate([prompt], SamplingParams(max_tokens=24, temperature=0))
        txt=out[0].outputs[0].text.strip()
        res[L]=(ntok, "7492" in txt, txt)
        print(f"L={L} tok={ntok} {'FOUND' if '7492' in txt else 'MISS '} | {txt[:55]!r}")
    torch.save(res, sys.argv[1]); print("saved", sys.argv[1])
if __name__=="__main__": main()
