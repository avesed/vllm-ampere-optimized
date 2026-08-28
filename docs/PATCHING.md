# Editing the fork

## There is no patch-apply step

`scripts/build_image_source.sh` builds the vendored `vllm/` and `flashinfer/` trees **exactly as they
sit in git**. No Dockerfile, no build stage, nothing applies `patches/*.patch`.

So: **to change fork behaviour, edit `vllm/…`, `flashinfer/…` or `flashampere/…` directly and commit.**
Writing or refreshing a `.patch` file changes nothing.

`patches/` is a *historical record* of what the fork changed and why — worth reading, and worth
appending to when you add something substantial, but it is not executable. Most of it no longer
applies to a current upstream tag, and at least one entry (`0002`) describes a kernel edit that now
lives somewhere else entirely (`flashampere/marlin/`).

`flashampere/` at the project root is the source of truth for the attention backend; its copy under
`vllm/vllm/v1/attention/backends/flashampere/` must be kept in sync by hand.

## Moving to a new upstream tag

Because the tree *is* the fork, an upstream bump is a **3-way merge**, not a replay:

```bash
scripts/revendor.sh v0.28.0 v0.6.16.post3   # merges both trees in .revendor/, stops on conflicts
# ... resolve any conflicts in .revendor/<tree>, `git add` the results (do NOT commit) ...
scripts/revendor.sh --sync-back             # copies the merged trees into the repo, bumps the marker
git diff && git commit
```

What that does per tree: clone upstream blobless, check out the tag we are currently on, rsync our
vendored tree over it and commit (that commit *is* the fork delta), check out the new tag, and
`git cherry-pick` the delta onto it.

**Do not use `git rebase` for this.** vLLM's release tags sit on release branches that carry commits
absent from a newer tag's history, so rebase tries to replay *upstream* commits and produces garbage
conflicts. `cherry-pick` uses the old tag's tree as the merge base, which is what you want.

To see what the fork actually changed at any point, diff the vendored tree against its upstream tag
(`git diff <tag> fork-old` inside the scratch repo) — never read `patches/` for that.

## Reading the merge

Conflicts cluster in a few predictable places. From the v0.25.1 → v0.28.0 merge (80 fork files, 16
conflicted):

- **Both sides added an entry at the same anchor** (`envs.py`, `registry.py`, `arg_utils.py`,
  import lists) — keep both, then check for a duplicate key.
- **Upstream landed our fix** — drop ours. Check before merging: e.g. vllm#50833 reproduced our
  dynamic-INT8-W8A8-MoE fix verbatim, and `VllmConfig.num_lookahead_tokens` absorbed our per-method
  DFlash/DSpark lookahead branches.
- **Upstream moved or deleted the file we edited** — find the new home and port the edit
  (`compressed_tensors_moe_wna16_marlin.py` was consolidated into `..._wna16.py`).
- **Both sides restructured the same function** — reconstruct the region from both originals rather
  than picking hunks, and check who imports what before choosing a side.

## Native code ships from source

The fork carries native (`.cu`/`.cuh`) edits — the fp16-PV FlashInfer prefill kernel, the vendored
famp Marlin GEMM and XQA kernels — which a pip overlay onto an official wheel or image cannot carry.
That is why the whole fork is vendored and built from source (`scripts/build_image_source.sh`, run on
a local GPU box; there is no overlay path and no CI build).
