# ampere-autotune

A standalone **host-side** two-tier tuner for Ampere vLLM serving. Full design in
[`DESIGN.md`](DESIGN.md); research backing in
[`../docs/RESEARCH-autotune-gpu-oc.md`](../docs/RESEARCH-autotune-gpu-oc.md).

> **Not** shipped inside any vLLM serving image. The `--hw` tier needs host root; the
> `--vllm` tier needs no privilege.

## Two tiers (one flag, default `--vllm`)

| | `--vllm` (default) | `--hw` (opt-in) |
|---|---|---|
| Privilege | none | **host root** |
| Touches | nothing — recommends flags | GPU clocks/power (NVML) |
| Output | flag set + restart command | validated per-GPU clock profile |
| Safety | read-only | preflight + mandatory `--dry-run` + golden-token gate + monitor-only default |

## Quick start

```bash
# always start here — it never changes anything:
ampere-autotune preflight              # print the capability/permission matrix
ampere-autotune preflight --hw --json  # machine-readable; use as a systemd ExecStartPre gate

# HALF-A (default): recommend vLLM flags against a running server (no privilege)
ampere-autotune recommend --endpoint http://localhost:8000

# HALF-B (host root, opt-in): silicon tuning — dry-run is mandatory before any write
sudo ampere-autotune tune --hw --dry-run
sudo ampere-autotune tune --hw --mode characterize   # the real climb (gated)
sudo ampere-autotune monitor --hw                     # monitor-only watchdog (default mode)
sudo ampere-autotune revert --hw                      # back to stock clocks
```

## Do I need root?

```
running a server, want better flags?         -> ampere-autotune recommend   (no root)
want more decode tok/s from the silicon?     -> sudo ampere-autotune tune --hw --dry-run
inside a container?                           -> HALF-B is REFUSED. Run on the HOST
                                                (clocks are global host driver state).
A100 / A40 / A10 / A2 ?                       -> T0/T1 only (offsets locked in HW).
```

## Install

```bash
uv pip install -e .          # HALF-A only (no privilege)
uv pip install -e '.[hw]'    # + HALF-B host silicon tuning deps
```

Preflight runs first on every invocation and **refuses** any action it could not unlock,
printing the exact missing capability and the fix.
