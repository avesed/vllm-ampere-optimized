# Upstream PR: cap `fastapi < 0.137`

Ready-to-submit fix for **vllm-project/vllm**. (We carry it locally as
`patches/0002-cap-fastapi-prometheus-compat.patch`; it should live upstream so every fresh
`pip install vllm` works.)

## Title

```
[Bugfix] Cap fastapi < 0.137 (0.137 _IncludedRouter breaks prometheus-fastapi-instrumentator at serve startup)
```

## The bug

A **fresh** `pip install vllm==0.23.0` (clean venv, latest deps) crashes at `vllm serve` startup:

```
File ".../prometheus_fastapi_instrumentator/routing.py", line 55, in <...>
    route_name = route.path
AttributeError: '_IncludedRouter' object has no attribute 'path'
```

The engine loads fine; the **HTTP API server** dies while mounting Prometheus metrics.

## Root cause

`requirements/common.txt` pins `fastapi[standard] >= 0.115.0` with **no upper bound**, so a fresh
resolve pulls **fastapi 0.137.0**. FastAPI 0.137 introduced a new route type
**`_IncludedRouter`** (`fastapi/routing.py`, a `BaseRoute` subclass with **no `.path`**) that now
appears in `app.routes` for included routers. `prometheus-fastapi-instrumentator` (the dep vLLM
uses in `vllm/entrypoints/serve/instrumentator/metrics.py`) iterates `app.routes` doing
`route.path`, special-casing only `Mount` — so it raises on `_IncludedRouter`. Even the latest
instrumentator (8.0.0) does not handle it yet.

Bisected: fastapi **0.135 / 0.136 are fine**, **0.137 is the first broken** version
(`from fastapi.routing import _IncludedRouter` succeeds only on >= 0.137).

Existing CI / the published Docker image don't catch this because they install from a
build-time-resolved environment that predates fastapi 0.137; only a *fresh* install hits it.

## The fix (one line)

```diff
--- a/requirements/common.txt
+++ b/requirements/common.txt
-fastapi[standard] >= 0.115.0 # Required by FastAPI's form models in the OpenAI API server's audio transcriptions endpoint.
+fastapi[standard] >= 0.115.0, < 0.137  # 0.137 added the _IncludedRouter route type (BaseRoute, no .path), breaking prometheus-fastapi-instrumentator route iteration at serve startup
```

(Lift `patches/0002-cap-fastapi-prometheus-compat.patch` verbatim.)

The cap can be lifted once `prometheus-fastapi-instrumentator` ships a release that tolerates
`_IncludedRouter` (an upstream issue/PR there is the deeper fix — its `routing.py` should skip
routes lacking `.path` the way it already skips/handles `Mount`).

## Minimal repro

```bash
python -m venv /tmp/v && /tmp/v/bin/pip install -q vllm==0.23.0
/tmp/v/bin/python - <<'PY'
import fastapi; print("fastapi", fastapi.__version__)          # -> 0.137.x
from fastapi.routing import _IncludedRouter                    # exists only >= 0.137
PY
/tmp/v/bin/vllm serve <any-model> --tensor-parallel-size 1     # AttributeError: _IncludedRouter ... 'path'
# with `pip install 'fastapi[standard]<0.137'` first: serve starts cleanly.
```
