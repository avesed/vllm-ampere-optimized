# vendor/ — pristine upstream kernel baselines (原装 / rollback reference)

**Never edit files here.** These are untouched upstream copies of the kernels flashampere forks &
Ampere-specializes. They exist so any Ampere kernel change can be **diffed against stock** and
**rolled back** — the safety baseline for kernel R&D with limited testing.

## xqa-upstream/
Pristine FlashInfer XQA, captured verbatim from the image's
`flashinfer/data/csrc/xqa/` (the source famp's `../xqa/csrc/xqa` was vendored from, BEFORE our
Ampere un-gate + hd512 + mixed-batch changes). Includes files famp PRUNED from its build:
`mha_sm90.cu` (Hopper), `mla_sm120.cu`/`mla_sm120.cuh` (the SM120 MLA kernel — see
[[project_mla_ampere_verdict]]; only present here as reference, NOT built), `gmma*.cuh`, `tma.h`,
`tensorMap.*`.

## How to use it
- **See what we changed** vs stock:
  `diff -u vendor/xqa-upstream/mha.cu ../xqa/csrc/xqa/mha.cu`
- **Roll a kernel back to stock** (if an Ampere change misbehaves): copy the upstream file over the
  working one (note: upstream XQA is Hopper/SM120-gated — it won't *run* on Ampere unmodified; the
  working RUNTIME fallback on Ampere is stock FA/FI via decline-to-stock, see ../README.md).
- **Re-pull upstream** later: re-capture from a newer flashinfer to re-baseline.

The RUNTIME fallback (the production safety net) is separate and lives in the backend itself —
see ../README.md "Fallback / rollback".
