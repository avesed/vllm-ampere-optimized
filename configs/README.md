# configs/ — device-tuned kernel configs (new files, not patches)

Drop-in data files that vLLM loads at runtime but does **not** ship for our cards. Unlike
`patches/` (diffs against upstream code), these are **new files** copied verbatim into the vLLM
tree by `scripts/apply_patches.sh` — they never conflict with upstream and, being pure data,
don't trip the native-code guard (they ride the default overlay wheel/image fine).

## fused_moe/

Auto-tuned **fused-MoE Triton tile configs** (`BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`,
`num_stages` per token-count). vLLM looks these up by
`E={experts},N={intermediate},device_name={gpu}[,dtype=...]` under
`vllm/model_executor/layers/fused_moe/configs/`. **If there's no exact match it falls back to a
generic heuristic** (logs `Using default MoE config. Performance might be sub-optimal!`), which is
slower on a 3090. Dropping in a tuned file = faster MoE forward, **zero accuracy change**.

| file | model shape | precision |
|---|---|---|
| `E=256,N=512,device_name=NVIDIA_GeForce_RTX_3090,dtype=int4_w4a16.json` | 256 experts, intermediate 512 (e.g. a 35B-A3B MoE) | W4A16 (int4 weight, fp16 act → `moe_wna16`) |

The `dtype` suffix follows vLLM's `get_config_dtype_str`: none = bf16/fp16, `fp8_w8a8`,
`int8_w8a8`, `int4_w4a16`. The `triton_version` key inside is fine — vLLM pops it before parsing.

## Regenerate / add more

Tune against your own model + GPU and drop the result here:

```bash
python vllm/benchmark/kernels/benchmark_moe.py --model <hf-id> --tp-size 2 --dtype int4_w4a16 --tune
# -> writes E=...,device_name=NVIDIA_GeForce_RTX_3090,dtype=....json ; move it into configs/fused_moe/
```

**No-rebuild alternative:** point vLLM at a folder of these JSONs at runtime via
`VLLM_TUNED_CONFIG_FOLDER=/path/to/folder` — it takes priority over the packaged configs, no
wheel/image rebuild needed (handy for A/B testing a config).
