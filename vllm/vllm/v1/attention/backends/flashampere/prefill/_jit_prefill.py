"""famp owned FA2 prefill (fp16-PV) — drops the patched-flashinfer dependency.

flashinfer's gen_single_prefill_module already renders the per-config binding .cu (the jinja
codegen), so we reuse that, but rebuild the JitSpec under our OWN name with our VENDORED
prefill.cuh on the include path (so `#include <flashinfer/attention/prefill.cuh>` resolves to
OURS, which carries the fp16-PV o_frag). We pass the SAME use_fp16_pv flag the caller will use so
the generated binding's run() ABI matches single_prefill's marshalling (patched flashinfer renders a
different binding for fp16pv=True). On STOCK flashinfer (no fp16pv arg) we add -DFA_USE_FP16_PV=1
ourselves. Either way we OWN the kernel (prefill.cuh) + the patch; flashinfer only supplies the
binding + cutlass/jit infra.
"""
import functools
import pathlib
import torch
from flashinfer.jit.attention.modules import gen_single_prefill_module
from flashinfer.jit.core import gen_jit_spec, JitSpec

_INC = pathlib.Path(__file__).parent / "include"  # vendored flashinfer/attention/prefill.cuh


def gen_famp_prefill_spec(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    use_fp16_pv: bool = True,
) -> JitSpec:
    args = ("fa2", dtype_q, dtype_kv, dtype_o, head_dim_qk, head_dim_vo,
            pos_encoding_mode, use_sliding_window, use_logits_soft_cap, use_fp16_qk_reduction)
    try:
        base = gen_single_prefill_module(*args, use_fp16_pv)  # patched: bakes the cflag + the matching binding
        extra_cflags = []
    except TypeError:
        base = gen_single_prefill_module(*args)               # stock flashinfer
        extra_cflags = ["-DFA_USE_FP16_PV=1"] if use_fp16_pv else []
    inc = [str(_INC)] + [str(p) for p in (base.extra_include_dirs or [])]
    return gen_jit_spec(
        "famp_" + base.name,                       # unique name -> fresh build dir, our settings
        base.sources,                              # reuse flashinfer's generated binding .cu (ABI match)
        extra_cflags=base.extra_cflags,
        extra_cuda_cflags=list(base.extra_cuda_cflags) + extra_cflags,
        extra_ldflags=base.extra_ldflags,
        extra_include_paths=inc,                   # OUR prefill.cuh first
        needs_device_linking=base.needs_device_linking,
    )


@functools.cache
def get_famp_prefill_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    use_fp16_pv: bool = True,
):
    return gen_famp_prefill_spec(
        dtype_q, dtype_kv, dtype_o, head_dim_qk, head_dim_vo,
        pos_encoding_mode, use_sliding_window, use_logits_soft_cap, use_fp16_qk_reduction, use_fp16_pv,
    ).build_and_load()
