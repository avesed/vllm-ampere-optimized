"""Build the standalone fused_silu_int8 extension (torch.ops.fused_silu_int8.*).

Single self-contained .cu (no cross-TU device-symbol refs) -> plain torch.utils.cpp_extension.load
(NO -rdc / -dlink, unlike the marlin build). We still reuse marlin's _ensure_cuda_libs() header-copy
trick (torch's CUDAContextLight.h pulls cusparse.h/cublas which live only in the pip nvidia/* include)
and set CUDA_HOME, and build for BOTH sm_80 and sm_86 (project scope).

Usage:
  from flashampere.fused_silu_int8.build import fused_silu_mul_quant_int8, get_fused_silu_int8
  out_i8, scales = fused_silu_mul_quant_int8(gate_up)   # gate_up: [..., 2N] bf16/fp16
"""
import functools
import os

import torch
from torch.utils.cpp_extension import load

from flashampere.marlin.build import _ensure_cuda_libs

_THIS = os.path.dirname(os.path.abspath(__file__))


def _arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


@functools.cache
def get_fused_silu_int8(arch: str | None = None):
    """Compile (cached) and return the loaded extension module; registers torch.ops.fused_silu_int8."""
    # Fast path: if the .so is already built, load it DIRECTLY (no ninja). vLLM TP workers run with a
    # SANITIZED env (no PATH), so cpp_extension.load() — which invokes ninja even for a cache hit —
    # fails there with "Ninja is required"; torch.ops.load_library does NOT need ninja. Build once in
    # a normal env (or via test/build), then the workers just load the cached .so.
    try:
        from torch.utils.cpp_extension import _get_build_directory
        _so = os.path.join(_get_build_directory("fused_silu_int8", verbose=False),
                           "fused_silu_int8.so")
        if os.path.exists(_so):
            torch.ops.load_library(_so)
            return torch.ops.fused_silu_int8
    except Exception:
        pass
    arch = arch or _arch()
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    _extra_inc = _ensure_cuda_libs()  # missing lib headers staged here if the toolkit isn't writable

    abi = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True))
    cuda_flags = [
        "-std=c++17", "-O3",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    # The .cu registers its op ONLY via TORCH_LIBRARY (no PYBIND11_MODULE / PyInit_*), so this
    # must be is_python_module=False: load() then runs the .so for its registration side-effect
    # and returns None; the op is reached via torch.ops.fused_silu_int8.* (not a returned module).
    load(
        name="fused_silu_int8",
        sources=[os.path.join(_THIS, "silu_and_mul_int8_quant.cu")],
        extra_include_paths=_extra_inc,
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
        is_python_module=False,
        verbose=True,
    )
    return torch.ops.fused_silu_int8


def fused_silu_mul_quant_int8(input: torch.Tensor):
    """Fused SiluAndMul + per-token int8 quant.

    Args:
        input: [..., 2N] bf16 or fp16, last dim = gate(0:N) || up(N:2N).
    Returns:
        (out_int8 [..., N], scales_fp32 [..., 1]) matching SiluAndMul -> per_token_quant_int8.
    """
    get_fused_silu_int8()  # ensure built/registered
    orig = input.shape
    x = input.reshape(-1, orig[-1]).contiguous()
    M, twoN = x.shape
    assert twoN % 2 == 0, "last dim must be even"
    N = twoN // 2
    out = torch.empty((M, N), dtype=torch.int8, device=x.device)
    scales = torch.empty((M,), dtype=torch.float32, device=x.device)
    torch.ops.fused_silu_int8.fused_silu_mul_quant_int8(x, out, scales)
    return out.view(*orig[:-1], N), scales.view(*orig[:-1], 1)
