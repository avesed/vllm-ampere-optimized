# Upstream PR package: cap `fastapi < 0.137`

Ready-to-submit fix for **vllm-project/vllm**. We carry it locally as
`patches/0002-cap-fastapi-prometheus-compat.patch`; it belongs upstream so every fresh
`pip install vllm` can `vllm serve`.

> Status check (2026-06-14): **no existing vLLM issue/PR** for this. The root cause is tracked in
> the instrumentator repo — [trallnag/prometheus-fastapi-instrumentator#370](https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/370)
> (OPEN). FastAPI 0.137's `_IncludedRouter` refactor broke many projects; there is **no released
> instrumentator fix yet** (verified: even instrumentator 8.0.0 still crashes vLLM's app), so a
> fastapi cap is the correct interim mitigation.

## Title (vLLM requires a bracket-tag prefix)

```
[Bugfix][Frontend] Cap fastapi < 0.137 to avoid prometheus-fastapi-instrumentator crash on serve startup
```

## PR body

> ### Purpose
>
> A fresh `pip install vllm` (clean venv, latest deps) crashes immediately at `vllm serve` startup:
>
> ```
> File ".../prometheus_fastapi_instrumentator/routing.py", line 55
>     route_name = route.path
> AttributeError: '_IncludedRouter' object has no attribute 'path'
> ```
>
> The engine initializes fine; the **HTTP API server** dies while mounting Prometheus metrics
> (`vllm/entrypoints/serve/instrumentator/metrics.py` → `Instrumentator().instrument(app)`).
>
> **Root cause:** `requirements/common.txt` pins `fastapi[standard] >= 0.115.0` with **no upper
> bound**, so a fresh resolve pulls **fastapi 0.137.0**. FastAPI 0.137 introduced a new route type
> `_IncludedRouter` (`fastapi/routing.py`, a `BaseRoute` with no `.path`) that now appears in
> `app.routes` for included routers. `prometheus-fastapi-instrumentator` iterates `app.routes`
> doing `route.path`, special-casing only `Mount`, so it raises on `_IncludedRouter`. This is the
> upstream instrumentator bug [trallnag/prometheus-fastapi-instrumentator#370](https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/370)
> (still open; no released fix — instrumentator 8.0.0 does not handle it either).
>
> Bisected: fastapi **0.135 / 0.136 are fine**, **0.137 is the first broken** version
> (`from fastapi.routing import _IncludedRouter` succeeds only on ≥ 0.137).
>
> CI and the published Docker image don't catch this because they install a build-time-resolved
> environment that predates fastapi 0.137; only a *fresh* user install hits it.
>
> **Fix:** cap to `< 0.137` in `requirements/common.txt`. The cap can be lifted once a
> prometheus-fastapi-instrumentator release tolerates `_IncludedRouter`.
>
> ### Test Plan
>
> ```bash
> python -m venv /tmp/v && /tmp/v/bin/pip install vllm            # pulls fastapi 0.137 (broken)
> /tmp/v/bin/vllm serve facebook/opt-125m                         # AttributeError: _IncludedRouter ... 'path'
> # with this PR's cap (or `pip install 'fastapi[standard]<0.137'`):
> /tmp/v/bin/vllm serve facebook/opt-125m                         # serves cleanly
> ```
>
> ### Test Result
>
> Before: `vllm serve` crashes at startup (`AttributeError: '_IncludedRouter' …`).
> After (`fastapi 0.135.4`): server reaches `Application startup complete`, `/v1/completions` +
> `/metrics` respond, `vllm bench serve` runs to completion (256/256 requests, 0 failed).

## The diff

```diff
--- a/requirements/common.txt
+++ b/requirements/common.txt
-fastapi[standard] >= 0.115.0 # Required by FastAPI's form models in the OpenAI API server's audio transcriptions endpoint.
+fastapi[standard] >= 0.115.0, < 0.137  # 0.137 added the _IncludedRouter route type (BaseRoute, no .path), breaking prometheus-fastapi-instrumentator route iteration at serve startup
```

(Identical to `patches/0002-cap-fastapi-prometheus-compat.patch`.)

## How to submit (from your fork)

vLLM enforces **DCO sign-off** and **pre-commit**. A one-line requirements cap needs no new test
(it's a dependency constraint), but run pre-commit so the lint bot is green.

```bash
git clone git@github.com:<you>/vllm.git && cd vllm
git checkout -b bugfix/cap-fastapi-0137
# apply the one line:
sed -i 's|^fastapi\[standard\] >= 0.115.0 .*|fastapi[standard] >= 0.115.0, < 0.137  # 0.137 added the _IncludedRouter route type (BaseRoute, no .path), breaking prometheus-fastapi-instrumentator route iteration at serve startup|' requirements/common.txt
uv pip install "pre-commit>=4.5.1" && pre-commit install && pre-commit run --files requirements/common.txt
git commit -s -am "[Bugfix][Frontend] Cap fastapi < 0.137 to avoid prometheus-fastapi-instrumentator crash on serve startup"
git push origin bugfix/cap-fastapi-0137
# then open the PR on github.com/vllm-project/vllm with the title + body above
```

`git commit -s` adds the required `Signed-off-by:` line. Link instrumentator #370 in the PR so
maintainers can drop the cap when it's fixed upstream.
