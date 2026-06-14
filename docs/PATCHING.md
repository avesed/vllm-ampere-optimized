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

Keep patches **pure-Python**. The wheel fast-path (`VLLM_USE_PRECOMPILED=1`) ships upstream's
prebuilt kernels; if a patch edits `.cu/.cpp/.cuh/CMakeLists/csrc/`, those kernels no longer
reflect the patch — silent wrong results. `scripts/apply_patches.sh` detects this and sets
`NATIVE_CHANGED=1`, which makes `build_wheel_fastpath.sh` refuse to run and forces the from-source
image build. If you *intend* a native change, accept that the fast-path is gone for that release.

## If upstream fixes it

If vLLM merges the W4A8-Ampere enablement (upstream
[#38066](https://github.com/vllm-project/vllm/pull/38066)) or restructures Marlin so our patch is
redundant, retire it: `git rm patches/0001-marlin-w4a8-int8-ampere.patch`. With the overlay model
that's a one-file delete — no fork revert to untangle.
