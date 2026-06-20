import json
from datasets import load_dataset
g = load_dataset("openai/gsm8k", "main", split="test")
with open("gsm8k.jsonl", "w") as f:
    for r in g:
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
print("gsm8k", len(g), flush=True)
m = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
with open("mmlu_pro.jsonl", "w") as f:
    for r in m:
        f.write(json.dumps({"question": r["question"], "options": r["options"],
                            "answer": r["answer"], "category": r.get("category")}) + "\n")
print("mmlu_pro", len(m), flush=True)
print("LOAD_DONE", flush=True)
