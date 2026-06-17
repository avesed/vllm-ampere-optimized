# Patching runbook

What to do when `patch-drift-check` opens a "patch drift" issue (the series stopped applying to a
new upstream tag).

## Refresh the patch against the new tag

```bash
git clone --depth 1 --branch <tag> https://github.com/vllm-project/vllm.git vllm
python patches/regenerate.py vllm
```

- **Clean apply** (`PATCH_APPLIED_CLEAN`): the string anchors still match; just re-emit the diff:
  ```bash
  cd vllm && git diff > ../patches/0001-marlin-w4a8-int8-ampere.patch
  ```
- **Version skew** (`regenerate.py` exits non-zero and lists the misses): upstream moved the code
  the anchor matched. Open the named file, find the new form of that block, update the
  corresponding `old`/`new` string in `patches/regenerate.py`, and rerun until clean, then emit
  the diff as above.

Commit the refreshed `.patch` (and any `regenerate.py` anchor edits). The next `build.yml` run for
that tag will pass the `git apply --check` gate.

## Why `git apply --3way`, not `git am`

Our patches are unified diffs, **not** mailbox commits (no `From:`/`Subject:` headers), so `git am`
fails on them outright. Plain `git apply` dies on the smallest context drift. `git apply --3way`
falls back to a 3-way merge when context shifts (cross-version resilience) and returns a clean
non-zero exit only when a hunk genuinely can't apply — which is what the drift canary keys on.

## The native-code rule

Keep patches **pure-Python** where you can. The default ship is an OVERLAY: the wheel
(`scripts/build_wheel_overlay.sh`) applies only the `vllm/*` (Python) hunks onto the official
**released** wheel (`pip download vllm==<tag>`), and the image is `FROM vllm/vllm-openai:<tag>` +
the same Python hunks. Neither can carry native (`.cu/.cpp/.cuh/CMakeLists/csrc/`) changes — a native
patch's kernels are simply **absent** from the default overlay wheel/image (the overlay just logs
`(no applicable vllm/ hunks …)` and moves on). A native patch therefore ships **only** via the opt-in
from-source image (`scripts/build_image_ampere.sh`, set repo var `BUILD_RUNNER`), which applies the
full patch and compiles. `scripts/apply_patches.sh` flags native hunks (`NATIVE_CHANGED=1`) so a
release that needs them is known to require the from-source path. (Example: patch 0002, the int8
8-row Marlin tile, is native → from-source image only; the overlay wheel/image carries patch 0001.)

## If upstream fixes it

If vLLM merges the W4A8-Ampere enablement (upstream
[#38066](https://github.com/vllm-project/vllm/pull/38066)) or restructures Marlin so our patch is
redundant, retire it: `git rm patches/0001-marlin-w4a8-int8-ampere.patch`. With the overlay model
that's a one-file delete — no fork revert to untangle.
