#!/usr/bin/env python3
"""Graft the MTP head into Avesed/Qwopus3.6-35B-A3B-v1-int4-mixed WITHOUT downloading the 24GB
quant: take the base tensor names from the HF safetensors header (range-read), the mtp.* head from
the local same-lineage FP16 base Qwopus3.6-35B-A3B-v1, build mtp.safetensors + index.json + patched
config.json, and upload only those 3 files.

Also reports whether the Qwopus head == the Qwen3.6-35B-A3B head (per the user's check). We always
use the Qwopus head (correct lineage for the Qwopus quant) regardless. Run on the sandbox with
HF_HUB_DISABLE_XET=1."""
import json
import os
import struct
import urllib.request

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

base = os.path.expanduser("~/models")
QWOPUS_SRC = f"{base}/Qwopus3.6-35B-A3B-v1"
QWEN_SRC = f"{base}/Qwen3.6-35B-A3B"
REPO = "Avesed/Qwopus3.6-35B-A3B-v1-int4-mixed"
api = HfApi()


def load_mtp(d):
    out = {}
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        with safe_open(os.path.join(d, f), framework="pt") as h:
            for k in h.keys():
                if k.startswith("mtp."):
                    out[k] = h.get_tensor(k)
    return out


print("=== compare Qwopus head vs Qwen3.6-35B head ===", flush=True)
qwopus_mtp = load_mtp(QWOPUS_SRC)
qwen_mtp = load_mtp(QWEN_SRC)
same_keys = set(qwopus_mtp) == set(qwen_mtp)
maxdiff = 0.0
for k in qwopus_mtp:
    if k in qwen_mtp and qwopus_mtp[k].shape == qwen_mtp[k].shape:
        maxdiff = max(maxdiff, (qwopus_mtp[k].float() - qwen_mtp[k].float()).abs().max().item())
identical = same_keys and maxdiff == 0.0
print(f"  qwopus_keys={len(qwopus_mtp)} qwen_keys={len(qwen_mtp)} same_keys={same_keys} "
      f"maxdiff={maxdiff:.6g} -> {'IDENTICAL' if identical else 'DIFFERENT (using Qwopus head)'}")

# sanity guard: head must be the expected ~19 non-trivial MoE mtp tensors
nparam = sum(t.numel() for t in qwopus_mtp.values())
assert len(qwopus_mtp) >= 15 and nparam > 1e8, f"qwopus head looks wrong: {len(qwopus_mtp)} keys, {nparam} params"
print(f"  using Qwopus head: {len(qwopus_mtp)} tensors, {nparam/1e6:.1f}M params", flush=True)

print("=== read HF base header (no full download) ===", flush=True)
url = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors"
req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
req2 = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"})
hdr = json.loads(urllib.request.urlopen(req2).read())
base_keys = [k for k in hdr if k != "__metadata__"]
base_bytes = max(hdr[k]["data_offsets"][1] for k in base_keys)
print(f"  HF base tensors={len(base_keys)} base_bytes={base_bytes/1e9:.2f}G", flush=True)

work = f"{base}/_qwopus35b_mtp_upload"
os.makedirs(work, exist_ok=True)
save_file(qwopus_mtp, f"{work}/mtp.safetensors", metadata={"format": "pt"})
mtp_bytes = sum(t.numel() * t.element_size() for t in qwopus_mtp.values())
wm = {k: "model.safetensors" for k in base_keys}
wm.update({k: "mtp.safetensors" for k in qwopus_mtp})
index = {"metadata": {"total_size": base_bytes + mtp_bytes}, "weight_map": dict(sorted(wm.items()))}
with open(f"{work}/model.safetensors.index.json", "w") as f:
    json.dump(index, f, indent=2)

cfg_path = hf_hub_download(REPO, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
qc = cfg.get("quantization_config", {})
ig = qc.setdefault("ignore", [])
if not any("mtp" in str(p).lower() for p in ig):
    ig.append("re:.*mtp.*")
with open(f"{work}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  built index ({len(wm)} tensors) + config (ignore mtp). uploading...", flush=True)

for fn in ["mtp.safetensors", "model.safetensors.index.json", "config.json"]:
    api.upload_file(path_or_fileobj=f"{work}/{fn}", path_in_repo=fn, repo_id=REPO,
                    repo_type="model", commit_message=f"Add native MTP head ({fn}) for speculative decoding")
    print(f"  uploaded {fn} ({os.path.getsize(f'{work}/{fn}')/1e6:.1f} MB)", flush=True)
print("QWOPUS 35B DONE", flush=True)
