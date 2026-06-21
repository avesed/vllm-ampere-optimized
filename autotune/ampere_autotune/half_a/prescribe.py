"""HALF-A: classify the bottleneck (roofline + R1-R5) and PRESCRIBE vLLM startup flags.

Recommend-only: emits a flag set + the exact restart command; never applies anything
(engine flags are startup-baked, no hot-reload — see ../docs/RESEARCH-autotune-gpu-oc.md
§5.1). Preferred home is upstream jungledesh/profile (Apache-2.0); this is the thin
fallback. STATUS: scaffold.
"""
from __future__ import annotations


def run(args, matrix) -> int:
    print("[half_a] vLLM flag recommender — scaffold.")
    print(f"[half_a] endpoint: {getattr(args, 'endpoint', None)}  local-gpus: {len(matrix.gpus)}")
    print("[half_a] TODO: scrape /metrics + NVML(read-only) -> roofline -> R1-R5 -> flag set.")
    print("[half_a] See DESIGN.md; prefer contributing to jungledesh/profile.")
    return 0
