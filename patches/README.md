# patches/

Ordered, upstream-relative unified diffs applied on top of a clean vLLM release checkout
in CI (`git apply --3way --whitespace=fix patches/*.patch`, lexical order). One `.patch`
per logical change so a drift failure points at exactly what broke.

| patch | what it does |
|---|---|
| `0001-marlin-w4a8-int8-ampere.patch` | Wires int4-weight + int8-activation (W4A8) through the **Marlin** kernel on Ampere — vLLM gates its dedicated W4A8 kernels (Cutlass/Machete) to Hopper. Edits `compressed_tensors_w4a8_int.py` (`act_type=torch.int8`), `mixed_precision/marlin.py` (allow int4 in the 8-bit-act assert, pack signed int4→uint4b8, pass effective `wtype`), `marlin_utils.py` (add `int4` to supported types). Pure-Python (no `.cu`/CMake). Upstream [vllm#38064](https://github.com/vllm-project/vllm/issues/38064)/[#38066](https://github.com/vllm-project/vllm/pull/38066). |
| `0002-cap-fastapi-prometheus-compat.patch` | Caps `fastapi[standard] < 0.137` in `requirements/common.txt`. FastAPI **0.137** introduced the `_IncludedRouter` route type (a `BaseRoute` with no `.path`); `prometheus-fastapi-instrumentator` iterates `app.routes` doing `route.path` (only special-casing `Mount`) → **`AttributeError: '_IncludedRouter' object has no attribute 'path'`** crashes `vllm serve` at startup. Upstream's loose `fastapi[standard]>=0.115.0` (no cap) lets a fresh install pull the broken fastapi — so a fresh `pip install` of the **official** vLLM wheel breaks too. This makes our published wheel install-and-`serve` cleanly. Requirements text only (no native change). Proposed upstream (see [`../docs/UPSTREAM-PR.md`](../docs/UPSTREAM-PR.md)). |

## Two rules

1. **Pure-Python only, or the fast-path breaks.** The wheel fast-path
   (`VLLM_USE_PRECOMPILED=1`) reuses upstream's prebuilt `.so`. That is correct *only* while
   no patch edits native code. `scripts/apply_patches.sh` greps the applied diff for
   `.cu/.cpp/.cuh/CMakeLists/csrc/` and sets `NATIVE_CHANGED=1` to force a from-source build
   if you ever cross that line. Keep native changes out of here unless you mean it.

2. **Anchors drift; refresh against the tag.** These diffs match upstream context lines that
   move between releases. `patch-drift-check.yml` runs `git apply --check` daily and opens an
   issue when the series stops applying.

## Regenerate a patch against a new tag

```bash
git clone --depth 1 --branch <tag> https://github.com/vllm-project/vllm.git vllm
python patches/regenerate.py vllm          # applies the edits in-place (fails loudly on skew)
cd vllm && git diff > ../patches/0001-marlin-w4a8-int8-ampere.patch
```

`regenerate.py` carries the exact string anchors; if an anchor no longer matches it prints
which one, so you update the anchor (and the diff) together. See `../docs/PATCHING.md`.
