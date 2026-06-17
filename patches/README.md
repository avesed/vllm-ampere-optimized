# patches/

Ordered, upstream-relative unified diffs applied on top of a clean vLLM release checkout
in CI (`git apply --3way --whitespace=fix patches/*.patch`, lexical order). One `.patch`
per logical change so a drift failure points at exactly what broke.

| patch | what it does |
|---|---|
| `0001-marlin-w4a8-int8-ampere.patch` | Wires int4-weight + int8-activation (W4A8) through the **Marlin** kernel on Ampere — vLLM gates its dedicated W4A8 kernels (Cutlass/Machete) to Hopper. Edits `compressed_tensors_w4a8_int.py` (`act_type=torch.int8`), `mixed_precision/marlin.py` (allow int4 in the 8-bit-act assert, pack signed int4→uint4b8, pass effective `wtype`), `marlin_utils.py` (add `int4` to supported types). Pure-Python (no `.cu`/CMake). Upstream [vllm#38064](https://github.com/vllm-project/vllm/issues/38064)/[#38066](https://github.com/vllm-project/vllm/pull/38066). |
| `0002-marlin-int8-8row-decode-ampere.patch` | **Native.** Implements + fixes the int8 (`kS8`) `m_block_size_8` **8-row decode tile** in Marlin — upstream gates it to 16-bit activations only (`marlin.cu`: `m_block_size_8 = prob_m<=8 && size_bits==16`). Enables the 8-row tile for W4A8 small-batch (`prob_m<=8`) decode via four transposed-`m16n8k32`-layout fixes, all `is_a_8bit`/`m_block_size_8`-gated (16-bit path untouched): **(1) gather k-order** — `b0=K[tg*4..+3]`, `b1=K[tg*4+16..+19]` (two K-half int4 chunks, not 8 contiguous K); **(2) grouped-fold weight-scale** — warp-shuffle col `gid`'s int16 scale from lane `gid>>1`; **(3) per-row activation scale** — `sh_a_s[tid*2+g%2]`, `[0]`-slot only; **(4) writeback** — `32` cols/warp for int8 (was 16-bit's `64`). Plus `marlin.cu` gate (kS8->8-row) and `generate_kernels.py` (emit `0.5` m-block kernels). **Touches `csrc/` -> trips the native-code guard (rule 1); W4A8 builds from source.** Validated: standalone Marlin GEMM M=4/8 uniform+real `max_diff ~0.003`; 27B W4A8 decode coherent. Zero decode throughput change (weight-bandwidth-bound) — correctness/completeness fix. |

## Two rules

1. **Pure-Python only, or it won't reach the default ship.** The default OVERLAY wheel
   (`scripts/build_wheel_overlay.sh`) applies only `vllm/*` (Python) hunks onto the official
   **released** wheel (`pip download vllm==<tag>`), and the overlay image is
   `FROM vllm/vllm-openai:<tag>` + the same hunks — neither can carry native
   (`.cu/.cpp/.cuh/CMakeLists/csrc/`) changes. `scripts/apply_patches.sh` greps the applied diff for
   those paths and sets `NATIVE_CHANGED=1`; a native patch (e.g. `0002`) then ships **only** via the
   opt-in from-source image (`build_image_ampere.sh`, repo var `BUILD_RUNNER`). Keep native changes
   out of here unless you mean it.

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
