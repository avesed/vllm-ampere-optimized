#!/usr/bin/env python3
"""Upload the MTP-head graft artifacts (mtp.safetensors + index + config) to the HF repos,
overwriting only those 3 small files; the multi-GB quantized weight shard(s) are untouched.

Pre-verified (scripts/inspect_mtp_upload.py): each target's base model.safetensors is byte-size
identical local-vs-HF, so the locally-built index.json exactly describes the HF weights. Run on the
sandbox with HF_HUB_DISABLE_XET=1 (classic LFS; xet hangs on large commits)."""
import os
from huggingface_hub import HfApi

api = HfApi()
base = os.path.expanduser("~/models")

PAIRS = [
    ("Qwen3.6-27B-W4A16", "Avesed/Qwen3.6-27B-INT4-W4A16"),
    ("Qwen3.6-27B-W8A8", "Avesed/Qwen3.6-27B-INT8-W8A8"),
    ("Qwen3.6-35B-A3B-W4A16", "Avesed/Qwen3.6-35B-A3B-INT4-W4A16"),
    ("Qwen3.6-35B-A3B-W8A8", "Avesed/Qwen3.6-35B-A3B-INT8-W8A8"),
]
FILES = ["mtp.safetensors", "model.safetensors.index.json", "config.json"]

for ld, repo in PAIRS:
    print(f"== {repo} ==", flush=True)
    for fn in FILES:
        lp = f"{base}/{ld}/{fn}"
        assert os.path.exists(lp), f"missing {lp}"
        api.upload_file(
            path_or_fileobj=lp, path_in_repo=fn, repo_id=repo, repo_type="model",
            commit_message=f"Add native MTP head ({fn}) for speculative decoding",
        )
        print(f"  uploaded {fn} ({os.path.getsize(lp) / 1e6:.1f} MB)", flush=True)
print("ALL DONE", flush=True)
