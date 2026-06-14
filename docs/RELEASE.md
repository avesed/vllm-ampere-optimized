# Release flow

## End to end

```
watch-upstream.yml (cron, every 6h)
  └─ gh api .../releases/latest  ≠  UPSTREAM_VLLM_VERSION ?
        └─ createWorkflowDispatch → build.yml(vllm_tag)
             ├─ drift-check     : git apply --check patches/*  (gates the rest)
             ├─ wheel-fastpath  : apply patches → VLLM_USE_PRECOMPILED wheel → GitHub Release
             ├─ image           : apply patches → docker build (sm_80+sm_86) → ghcr → smoke test
             └─ bump-marker     : on BOTH succeeding, commit UPSTREAM_VLLM_VERSION = vllm_tag
```

The marker bumps **only when wheel + image both succeed**, so a partial failure re-fires on the
next cron tick instead of being silently skipped.

## Manually build any tag

`Actions → build → Run workflow`, or:

```bash
gh workflow run build.yml -f vllm_tag=v0.23.0                 # cu130 (default)
gh workflow run build.yml -f vllm_tag=v0.23.0 -f cuda_version=12.9.1   # cu129 broad-compat variant
```

## First-time setup

1. Push this repo to GitHub (the `UPSTREAM_VLLM_VERSION` marker ships as `none`, so the first
   `watch-upstream` run builds the current latest).
2. **Runner for the image build** — pick one:
   - *Self-hosted (recommended, fastest):* register a runner on the 2×3090 box and set repo
     variable `BUILD_RUNNER` to its label (e.g. `self-hosted`). Native build, persistent sccache,
     and the smoke test actually runs (needs a GPU). Use only on a **private** repo (self-hosted
     runners + public PRs = arbitrary code execution).
   - *GitHub-hosted:* leave `BUILD_RUNNER` unset → `ubuntu-latest` with a `free-disk-space` step.
     A full CUDA build there is slow and disk-tight even single-arch; consider an 8-core larger
     runner. The smoke test auto-skips (no GPU).
3. No secrets needed: ghcr push uses the built-in `GITHUB_TOKEN` (`packages: write`), Release
   upload uses it too (`contents: write`). Make the ghcr package public in repo/org settings if you
   want anonymous `docker pull`.

## Driver / CUDA note

cu130 images need NVIDIA driver **≥ 580.65.06**; cu129 needs ≥ 575. The 2×3090 dev box runs
590.48.01, so cu130 is the default. Publish a cu129 variant for hosts on older drivers (A100
clusters, older rigs) via the manual `cuda_version=12.9.1` dispatch.

## Install what it produces

```bash
# wheel (any Ampere host with matching torch/CUDA):
pip install https://github.com/<owner>/vllm-ampere-optimized/releases/download/v0.23.0-ampere/<wheel>.whl

# image:
docker run --gpus all -p 8000:8000 \
  ghcr.io/<owner>/vllm-ampere-optimized:v0.23.0-ampere-cu130 \
  --model <hf-id> --max-model-len 8192
```
