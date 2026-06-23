#!/usr/bin/env python3
"""Pre-upload inspection for MTP-grafted quants: verify local base == HF base (size), detect
single-file vs sharded layout (to build the right index), confirm whether a repo already has the
head, and check repo existence. Read-only against HF; prints a report. Run on the sandbox."""
import os, struct, json, urllib.request
from huggingface_hub import HfApi

api = HfApi()
base = os.path.expanduser("~/models")

TARGETS = [
    ("Qwen3.6-27B-W4A16", "Avesed/Qwen3.6-27B-INT4-W4A16"),
    ("Qwen3.6-27B-W8A8", "Avesed/Qwen3.6-27B-INT8-W8A8"),
    ("Qwen3.6-35B-A3B-W4A16", "Avesed/Qwen3.6-35B-A3B-INT4-W4A16"),
    ("Qwen3.6-35B-A3B-W8A8", "Avesed/Qwen3.6-35B-A3B-INT8-W8A8"),
]


def hf_files(repo):
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings}


def header_mtp_keys(repo, fname="model.safetensors"):
    """Range-read the safetensors header from HF and return the mtp.* tensor names."""
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
    req2 = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
    hdr = json.loads(urllib.request.urlopen(req2).read())
    return [k for k in hdr if k.startswith("mtp.")], len(hdr)


print("=== 4 upload targets: base size match + layout ===")
for ld, repo in TARGETS:
    try:
        hf = hf_files(repo)
        main = [k for k in hf if k.endswith(".safetensors") and "mtp" not in k]
        sharded = any("index.json" in k for k in hf)
        hfsum = sum(hf[k] for k in main)
        lm = [f for f in os.listdir(f"{base}/{ld}") if f.endswith(".safetensors") and "mtp" not in f]
        lsum = sum(os.path.getsize(f"{base}/{ld}/{f}") for f in lm)
        tag = "MATCH" if hfsum == lsum else "*** MISMATCH ***"
        print(f"  {repo}")
        print(f"    sharded={sharded} hf_main={len(main)}f={hfsum/1e9:.2f}G "
              f"local={lsum/1e9:.2f}G {tag} mtp_on_hf={'mtp.safetensors' in hf}")
    except Exception as e:
        print(f"  {repo}: ERROR {type(e).__name__}: {e}")

print("=== Qwopus 35B int4-mixed: head baked into single file? (HF header) ===")
repo = "Avesed/Qwopus3.6-35B-A3B-v1-int4-mixed"
try:
    mk, ntot = header_mtp_keys(repo)
    verdict = "HAS HEAD (skip)" if mk else "MISSING HEAD (needs graft)"
    print(f"  {repo}: header_tensors={ntot} mtp_keys={len(mk)} -> {verdict}")
except Exception as e:
    print(f"  header check failed: {type(e).__name__}: {e}")

print("=== W4A8 HF repos exist? ===")
for repo in ["Avesed/Qwen3.6-27B-W4A8", "Avesed/Qwopus3.6-27B-v2-W4A8"]:
    try:
        api.model_info(repo)
        print(f"  {repo}: EXISTS")
    except Exception as e:
        print(f"  {repo}: NOT FOUND ({type(e).__name__})")
