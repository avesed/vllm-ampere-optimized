# Release flow

The fork is **vendored + built from source, locally**. There is **no CI auto-build**: a from-source
vLLM CUDA build needs a GPU, and a self-hosted GPU runner on a public repo is a security risk
(`self-hosted + public PRs = arbitrary code execution`). So the maintainer builds and pushes the image
by hand. CI is only two **github-hosted** canaries that open issues — they never build or push.

## End to end

```
watch-upstream.yml (cron, every 6h)
  └─ gh api .../releases/latest  ≠  UPSTREAM_VLLM_VERSION ?
        └─ opens a "re-vendor" ISSUE (no build is triggered)

maintainer, locally:
  1. scripts/revendor.sh <vllm_tag> <flashinfer_tag>   # only if upstream bumped; else skip
       └─ clones fresh upstream into vllm/ + flashinfer/, replays the recipe
          (regenerate.py 0001 + git apply 0002 + 0003 + apply_to_source.py); drift FAILS LOUDLY
  2. git diff && git commit                             # review + commit the vendored trees
  3. OWNER=<you> scripts/build_image_source.sh          # from-source sm_80+sm_86 build → push ghcr :<tag>-ampere-<cu> + :latest
  4. (optional) scripts/smoke_test.sh <img> ; scripts/ampere_kernel_ci.sh <img> "$(cat UPSTREAM_VLLM_VERSION)"
       W4A16_CKPT=<w4a16> W4A8_CKPT=<w4a8> scripts/int8_cudagraph_regression.sh <img>   # asserts patch 0003
  5. echo <tag> > UPSTREAM_VLLM_VERSION && git commit   # bump the marker (revendor.sh already does this)

patch-drift-check.yml (cron daily + on patches/** PRs)  # github-hosted; replays the recipe onto the
  └─ git apply --check patches/* onto LATEST upstream    # latest tags in temp checkouts, opens an issue
                                                          # if an anchor drifted. Never builds.
```

No marker auto-bump, no partial-failure logic — the human runs the steps and commits the marker.

## Build any tag / CUDA variant (locally)

```bash
docker login ghcr.io                                              # once; needs a PAT with write:packages
OWNER=<you> scripts/build_image_source.sh                         # cu130 (default), from the vendored source
OWNER=<you> CUDA_VERSION=12.9.1 scripts/build_image_source.sh     # cu129 broad-compat variant
```

`build_image_source.sh` builds vLLM from `vllm/` (two-stage: vLLM image, then the int8-QK FlashInfer
overlay from `flashinfer/`), tags `:<tag>-ampere-<cu>` + `:latest`, and `--push`es. `VLLM_TAG` defaults
to `UPSTREAM_VLLM_VERSION`; `TORCH_CUDA_ARCH_LIST` defaults to `8.0 8.6` (all Ampere). It uses GHA
registry cache only when run inside Actions; locally it uses docker's own layer cache.

## First-time setup

1. **ghcr push** — `docker login ghcr.io -u <you>` with a PAT that has `write:packages`. After the
   first push, make the package public (Packages → settings) for anonymous `docker pull`.
2. **Actions** — only `watch-upstream` + `patch-drift-check` run in CI; both need `issues: write`
   (Settings → Actions → General → Workflow permissions → Read and write). No `packages: write` token,
   no self-hosted runner, no secrets — the build never runs in CI.
3. **Build box** — any Linux host with an NVIDIA GPU + docker buildx. The from-source build is heavy
   (full vLLM CUDA compile for sm_80+sm_86); the 2×3090 dev box is the reference builder.

## Driver / CUDA note

cu130 images need NVIDIA driver **≥ 580.65.06**; cu129 needs ≥ 575. The 2×3090 dev box runs
590.48.01, so cu130 is the default. Build a cu129 variant for hosts on older drivers (A100 clusters,
older rigs) with `CUDA_VERSION=12.9.1 scripts/build_image_source.sh`.

## Install what it produces

```bash
docker run --gpus all -p 8000:8000 \
  ghcr.io/<owner>/vllm-ampere-optimized:v0.23.0-ampere-cu130 \
  --model <hf-id> --max-model-len 8192
```
