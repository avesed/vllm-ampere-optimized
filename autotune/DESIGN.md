# ampere-autotune — Engineering Design

> Standalone **host-side** tuning tool for the vllm-ampere-optimized fork. NOT shipped
> inside any vLLM serving image. Full research backing: `../docs/RESEARCH-autotune-gpu-oc.md`.

## Overview

`ampere-autotune` wrings extra throughput out of an Ampere serving box in **two cleanly
separated halves**, selected by one flag. **The default is `--vllm`.**

- **HALF-A `--vllm` (DEFAULT, no privilege)** — measure → classify → **prescribe** vLLM
  startup flags. Reads vLLM `/metrics` (Prometheus) + NVML read-only, classifies the
  bottleneck (roofline + R1–R5), emits a recommended flag set + the exact restart command.
  It **never mutates anything** — not the engine, not the GPU.
- **HALF-B `--hw` (HOST ROOT, opt-in)** — GPU **silicon** tuning over a tier ladder, gated
  by a hard preflight and a **mandatory `--dry-run`**.

The two halves never share privileged code: a CI test asserts HALF-A imports never pull
`half_b/` or `silicon`.

## Why HALF-A is offline / recommend-only

The eight perf-critical engine flags (`max-num-seqs`, `max-num-batched-tokens`,
`gpu-memory-utilization`, TP/PP, `kv-cache-dtype`, `enable-prefix-caching`, block sizing)
bind at `EngineArgs.create_engine_config()` with **no runtime hot-reload**: the KV-block
count is frozen by startup gpu-memory profiling, cudagraphs are captured against those
values, and TP/PP fix the process group. So HALF-A is necessarily
measure → recommend → **restart** → re-measure (online == canary/blue-green relaunch).
`half_a/` contains zero NVML-write and zero privileged imports. Ideal home for the
classify/prescribe logic is upstream `jungledesh/profile` (Apache-2.0); a thin wrapper
lives here as fallback.

## The tier ladder (HALF-B)

| Tier | Lever | Scope |
|---|---|---|
| **T0** power-limit (`SetPowerManagementLimit`) | safe | ALL Ampere incl. A100 |
| **T1** locked-clocks (`SetGpuLockedClocks`) | safe (factory envelope) | ALL Ampere; may obviate most of T3 |
| **T2** core-offset / **T3** mem-offset (`SetClockOffsets`) | gated OC | consumer GeForce 3090/3080 + workstation A6000/A5000 only |

A40/A10/A2/A100 return `NOT_SUPPORTED` by HW. The +6–8% decode lever lives at the GDDR6X
EDR bandwidth-knee, **just below** corruption.

## Preflight: capability + permission detection (runs first, every invocation)

Four probe groups feed one `CapabilityMatrix`. Each probe returns an **OUTCOME enum + a
`fix` string**, never a crash. All probes are **read-only or no-op writes — nothing moves a
clock**. See `ampere_autotune/preflight/`.

1. **GPU-OC-SUPPORT / SKU-tier** (`sku.py`) — compute-cap + mem-type + brand, plus a SAFE
   read-only `GetClockOffsets` probe. → tier **ceiling** + **gate family**
   (`GOLDEN_EDR_JUNCTION` for GDDR6X / `ECC_GOLDEN` for GDDR6 / `ECC` for HBM·datacenter).
2. **TELEMETRY-READ** (`telemetry_perm.py`) — NVML read (always, no priv) vs BAR0-MMIO
   junction (GDDR6X mem-temp; **four independent prereqs**: root + `/dev/mem` +
   `iomem=relaxed` + Secure-Boot-off + idle sanity-match) vs DCGM-PROF (datacenter only).
3. **OC-PERMISSION** (`oc_perm.py`) — euid==0 + CAP_SYS_ADMIN + container/privileged detect
   + driver ≥ R555.85 + a **SAFE no-clock write-cap probe** (set the offset to its *current*
   value, read back) that separates `NO_PERMISSION` (unprivileged container) from
   `NOT_SUPPORTED` (datacenter HW) **without moving a clock**.
4. **DRIVER/STATE** (`driver_state.py`) — persistence-mode (offsets wiped on reboot), ECC
   mode (selects the gate family), GPU-UUID (profile key).

`matrix.py` aggregates the four into a printed table + `--json` + a **DECISION** line that
unlocks tiers/modes; every refusal carries the exact missing capability + the fix.

### Unlock rule
- `--vllm` (HALF-A) always runs if NVML-read OK or `/metrics` reachable.
- **HALF-B unlocked iff** `priv == HOST_ROOT_PRIVILEGED` AND `write_cap == WRITE_OK` AND
  driver ≥ R555.85; tier ≤ SKU ceiling.
- **T3 sustained mem-OC additionally requires** `junction == JUNCTION_READABLE` (else T3
  REFUSED — no headless silent-overheat guard).
- `ECC_OFF` GDDR6X → golden-token gate MANDATORY; `ECC_ON` → SBE/DBE gate.
- Default `--hw` mode = **monitor-only**; any `--hw` write requires `--dry-run` first.

## How the HALF-B machinery plugs together

- **Correctness gate (`gate.py`)** — on no-ECC GDDR6X, corruption is **SILENT** (no
  Xid/ECC/throttle). THE gate is **exact golden token-id** compare under
  `VLLM_BATCH_INVARIANT=1`; BW+verify GB/s (EDR-knee, GDDR6X only), `mismatch_count`,
  junction temp, ECC deltas are **coverage/health probes**, not the gate. Climb-stop =
  FIRST-of `{EDR knee | mismatch>0 | golden fail | junction ≥ 95 °C}` minus a **layered
  guard band** (not "1 tick").
- **Adaptive search (`search.py`)** — coarse-up 105 MHz (7 ticks) / fine-down 30 MHz
  (2 ticks), 15 MHz KMD snap; objective = max decode tok/s at **zero sustained errors**
  (= the knee); then a **heat-soak re-check** before persisting (GDDR6X heat-soaks to
  ~110 °C invisible to Linux → cold-validated knees drift).
- **Monitor-only default (`monitor.py`)** — watchdog reverts/derates via one NVML write,
  **acts before it logs** (subtract-only dead-man's-switch); raises an offset UP only via a
  drained full gate. Continuous runtime auto-tuning is **explicitly NOT pursued**.
- **Persisted profile (`profile_store.py`)** — per-GPU-UUID JSON at
  `~/.config/ampere-autotune/<uuid>.json`; re-validated on temp-delta / driver-change /
  UUID-mismatch. **Never a default** (silicon lottery — per physical card).

### FOOTGUN (load-bearing): NVML mem-offset value = **half** the GDDR MT/s number.
`silicon.py` and `profile_store.py` always log/convert **both**; mislabeling doubles or
halves every offset and silently corrupts the search.

## The fine-grained step-probe (`probe/`)

A passive `StatLoggerBase` plugin (`vllm.stat_logger_plugins`, env `VLLM_STEP_PROBE=1`) that
taps per-step `SchedulerStats`/`IterationStats`/`SpecDecodingStats` vLLM already materializes
on CPU — **no `.item()/.tolist()/.cpu()` in the forward path; FULL-cudagraph-safe**. Honest
caveat: stock `SpecDecodingLogging.log()` already suffices for one-time spec-K
characterization; the probe earns its keep only on finer cadence/joins (accept-len-vs-batch
crossover) and the **hybrid mamba-state pressure stock cannot see**.

## Packaging & safety posture

Self-contained host package (`pyproject.toml`, console-script `ampere-autotune`, deps
`nvidia-ml-py` + optional `hw` extra), **never inside a vLLM image** — clocks are global host
driver state and the NVIDIA Container Toolkit injects no `/dev/mem` and no SET capability.
Deploy as a **host systemd unit ordered `Before=` the vLLM container**, with
`ampere-autotune preflight --hw --json` as an `ExecStartPre` gate so the monitor refuses to
start on an unsupported/locked host, and persistence re-apply on boot. `--dry-run` is
mandatory for any `--hw` write; the unprivileged-container case is a **first-class REFUSAL
with prescription**, not an opaque error.

## What this directory is NOT

- Not shipped in the fork's serving image / not under the fork `scripts/`.
- Not a runtime auto-tuner (engine flags are restart-bound; see `../docs/RESEARCH-autotune-gpu-oc.md` §5.1/§5.2).
- HALF-A flag-tuning is per-deployment config — it lives here for convenience but should
  defer to / contribute upstream to `jungledesh/profile`.
