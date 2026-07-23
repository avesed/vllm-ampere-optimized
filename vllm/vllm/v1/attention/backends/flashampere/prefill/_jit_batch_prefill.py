"""famp owned FA2 BATCH paged prefill (fp16-PV) — the de-taxed prefill path.

Injection point: flashinfer's standard batch-prefill flow is
    wrapper.plan() -> get_batch_prefill_module(backend, *args)   [functools.cache]
                   -> gen_batch_prefill_module(backend, *args).build_and_load()
We shim `gen_batch_prefill_module` in flashinfer.prefill's namespace with a BY-EXACT-KEY
override: famp's config keys resolve to a JitSpec rebuilt under famp's name with the VENDORED
prefill.cuh first on the include path (FA_PV16 fp16-PV epilogue + the sm86 tile heuristic);
every other key defers to stock flashinfer (e.g. the hd512 bf16 decode wrapper). This keeps the
wrapper's standard plan/run ABI (the `_jit_module` ctor path expects the CUSTOMIZE codegen ABI —
16-arg plan — and is NOT usable for a standard-ABI kernel; measured mismatch 19-vs-16).
Install BEFORE the first plan() for the key (get_batch_prefill_module caches per key)."""
import pathlib

import torch
from flashinfer.jit.attention.modules import gen_batch_prefill_module as _stock_gen
from flashinfer.jit.core import JitSpec, gen_jit_spec

_INC = pathlib.Path(__file__).parent / "include"  # vendored flashinfer/attention/prefill.cuh


def _famp_spec(backend, *args, use_fp16_pv: bool = True) -> JitSpec:
    base = _stock_gen(backend, *args)
    extra_cflags = ["-DFA_USE_FP16_PV=1"] if use_fp16_pv else []
    inc = [str(_INC)] + [str(p) for p in (base.extra_include_dirs or [])]
    return gen_jit_spec(
        "famp_" + base.name,                       # unique name -> fresh build dir, our settings
        base.sources,                              # flashinfer's generated binding (ABI match)
        extra_cflags=base.extra_cflags,
        extra_cuda_cflags=list(base.extra_cuda_cflags) + extra_cflags,
        extra_ldflags=base.extra_ldflags,
        extra_include_paths=inc,                   # OUR prefill.cuh first
        needs_device_linking=base.needs_device_linking,
    )


def install_famp_batch_prefill(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    use_fp16_pv: bool = True,
) -> None:
    """Idempotently route this exact batch-prefill config to the famp spec (in-process only)."""
    import flashinfer.prefill as fip

    key = (
        "fa2", dtype_q, dtype_kv, dtype_o, torch.int32, head_dim, head_dim,
        0,      # PosEncodingMode NONE
        False,  # use_sliding_window
        False,  # use_logits_soft_cap
        False,  # use_fp16_qk_reduction
    )
    cur = fip.gen_batch_prefill_module
    if not getattr(cur, "_famp_shim", False):
        overrides: dict[tuple, JitSpec] = {}

        def _shim(backend, *args):
            spec = overrides.get((backend, *args))
            return spec if spec is not None else _stock_gen(backend, *args)

        _shim._famp_shim = True
        _shim._famp_overrides = overrides
        fip.gen_batch_prefill_module = _shim
        cur = _shim
    if key not in cur._famp_overrides:
        cur._famp_overrides[key] = _famp_spec(*key, use_fp16_pv=use_fp16_pv)
