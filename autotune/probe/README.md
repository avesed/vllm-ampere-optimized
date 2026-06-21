# probe/ — fine-grained in-engine step probe (designed-only)

A passive `StatLoggerBase` plugin for vLLM that taps the per-step
`SchedulerStats`/`IterationStats`/`SpecDecodingStats` vLLM already materializes on CPU, at
decode-step cadence (5-20 ms) — far finer than the 10 s in-engine logging window or a ~2 s
external `/metrics` scrape. Feeds **offline** analysis + the `../benchmarks/` harness; it is
**never a runtime control input**. See `../docs/RESEARCH-autotune-gpu-oc.md` §5.3.

- `step_probe.py` — `vllm.stat_logger_plugins` entry point, env-gated `VLLM_STEP_PROBE=1`.
  **Cudagraph-safe by construction**: `record()` runs in the frontend process outside the
  captured forward; every field is already a CPU int/float/list. **Hard rule:** NO
  `.item()`/`.tolist()`/`.cpu()` in the path; bounded ring buffer; JSONL off the hot path;
  validate under FULL cudagraph (the int8qk `.tolist()` bug was masked by `enforce_eager`).
- `analyze_step_probe.py` — parse the JSONL into is-this-lever-worth-it verdicts
  (accept-len-vs-batch crossover ≈ the `disable-by-batch-size` threshold; decode roofline),
  mirroring `../benchmarks/analyze_torch_prof.py`.

## Honest caveat (why this is narrow)

Stock vLLM already exposes a lot: Prometheus (incl. `spec_decode_num_accepted_tokens_per_pos`),
`--collect-detailed-traces` (OTLP, per finished request), the torch profiler, and
`SpecDecodingLogging.log()` (mean accept-length + per-position vector + goodput each interval).
For a **one-time spec-K characterization** that already suffices — the fork's measured
K=2 (9B) / K=3 (27B) numbers came from those stock logs. The probe earns its keep ONLY on:
- the **same-step accept-len ↔ batch-size join** (crossover from one mixed-load run, not N sweeps),
- per-step cadence for curve derivation, and
- the **hybrid mamba-state cache pressure stock cannot see** (`kv_cache_usage` counts only the
  full-attn layers; the mamba state cache is what actually OOMs).

The two genuinely net-new engine fixes (a Mamba/GDN `ComponentMetrics` + correct
`num_kv_layers` so the decode roofline isn't ~3× off on the 3:1 hybrid; a mamba-state-cache
occupancy counter) are small `perf.py`-class patches worth more than the probe itself.
`SchedulerStats`/`IterationStats` are explicitly **non-stable** vLLM interfaces → rebase risk.
