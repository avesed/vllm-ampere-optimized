# Release flow

The fork is **vendored + built from source, locally**. There is **no CI auto-build**: a from-source
vLLM CUDA build needs a GPU, and a self-hosted GPU runner on a public repo is a security risk
(`self-hosted + public PRs = arbitrary code execution`). So the maintainer builds and pushes the image
by hand. CI is a single **github-hosted** canary (`watch-upstream`) that opens a one-time reminder
issue per new upstream release — it never builds or pushes.

## End to end

```
watch-upstream.yml (cron, daily)
  └─ gh api .../releases/latest  ≠  UPSTREAM_VLLM_VERSION ?
        └─ opens ONE reminder ISSUE per new tag (deduped incl. closed; no build is triggered)

maintainer, locally:
  1. scripts/revendor.sh <vllm_tag> <flashinfer_tag>   # only if upstream bumped; else skip
       └─ 3-way MERGES the vendored trees onto the new tags in .revendor/ (no patch replay -- the
          tree IS the fork). Conflicts stop it loudly; resolve, git add, then --sync-back.
  1b. scripts/revendor.sh --sync-back                   # copy merged trees in + bump UPSTREAM_VLLM_VERSION
  2. git diff && git commit                             # review + commit the vendored trees
  3. OWNER=<you> scripts/build_image_source.sh          # from-source sm_80+sm_86 build → push ghcr :<tag>-ampere-<cu> + :latest
  4. scripts/release-matrix/run.sh <blocks>             # MANDATORY e2e gate on the new image AND the previous
                                                        # release, same night, same box (see "Release matrix")
  4b. (optional) scripts/smoke_test.sh <img> ; scripts/ampere_kernel_ci.sh <img> "$(cat UPSTREAM_VLLM_VERSION)"
       W4A16_CKPT=<w4a16> W4A8_CKPT=<w4a8> scripts/int8_cudagraph_regression.sh <img>   # asserts the AOT cache-key fix
  5. echo <tag> > UPSTREAM_VLLM_VERSION && git commit   # bump the marker (--sync-back already does this)

```

No marker auto-bump, no partial-failure logic — the human runs the steps and commits the marker.

## Release matrix

`scripts/release-matrix/` is the end-to-end gate: every block serves the image under test over the
OpenAI API and logs pass/fail lines. Run it on an otherwise idle box, once on the candidate image and
once on the previous release, and compare the two log sets block by block.

```bash
export MODELS=/path/to/staged/checkpoints GSM8K_JSONL=/path/to/gsm8k.jsonl
IMG=<candidate> LOGDIR=out/new bash scripts/release-matrix/run.sh t0_smoke t6m_mtp t6_spec t5_quant \
  t7_hybrid t4_gemma t1_famp t2_batch t12_mm t5e_equiv t6g_greedy t10_perf t9_soak
IMG=<previous>  LOGDIR=out/old bash scripts/release-matrix/run.sh ...   # same blocks
```

- Each block serves fixed checkpoint directory names under `MODELS` (see its `serve` lines).
- `run.sh` first scans `MODELS` for lost-write holes (`ckpt_holes.py`, zeroed 4 MiB extents) and refuses
  to run if any checkpoint has them; list a known-holey checkpoint in `CKPT_HOLES_OK` to allow it.
- Containers carry the label `vllm-release-matrix=1`; cleanup only ever selects on that label.
- Compare like with like: acceptance length per category and per sampling regime (`t6g_greedy` for draft
  code changes, `t6_spec`/`t6h_heads` sampled), perf as medians from `t10_perf` on both images.
- Re-run the affected blocks after every fix.

## Build any tag / CUDA variant (locally)

```bash
docker login ghcr.io                                              # once; needs a PAT with write:packages
OWNER=<you> scripts/build_image_source.sh                         # cu130 (default), from the vendored source
OWNER=<you> CUDA_VERSION=12.9.1 scripts/build_image_source.sh     # cu129 broad-compat variant
```

`build_image_source.sh` builds vLLM from `vllm/` (three-stage: vLLM image, the vendored FlashInfer
overlay from `flashinfer/`, then the famp_marlin `.so` from `flashampere/marlin/csrc`), tags
`:<tag>-ampere-<cu>` + `:latest` (fork releases add `:X.Y`, e.g. `:0.3`), and `--push`es. `VLLM_TAG` defaults
to `UPSTREAM_VLLM_VERSION`; `TORCH_CUDA_ARCH_LIST` defaults to `8.0 8.6` (all Ampere). It uses GHA
registry cache only when run inside Actions; locally it uses docker's own layer cache.

## First-time setup

1. **ghcr push** — `docker login ghcr.io -u <you>` with a PAT that has `write:packages`. After the
   first push, make the package public (Packages → settings) for anonymous `docker pull`.
2. **Actions** — only `watch-upstream` runs in CI; it needs `issues: write`
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
  ghcr.io/<owner>/vllm-ampere-optimized:latest \
  --model <hf-id> --max-model-len 8192
```
