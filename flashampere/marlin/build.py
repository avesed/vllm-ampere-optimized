"""famp vendored Marlin — build the standalone famp_marlin extension (torch.ops.famp_marlin.*).

Marlin's host (marlin.cu) takes the ADDRESS of Marlin<...> __global__s defined in the per-config
kernel .cu (cross-TU device refs), so it needs RELOCATABLE DEVICE CODE (-rdc) + a device-link step
(nvcc -dlink) — what vLLM's CMake does (CUDA_SEPARABLE_COMPILATION). torch.cpp_extension.load() can't
do the -dlink, so we drive nvcc/g++ manually: -rdc compile each src -> nvcc -dlink -> g++ -shared.

Self-contained (no cutlass). BUILD-TIME compile (~510 kernels), cached after the first build. Usage:
  from flashampere.marlin.build import get_famp_marlin; m = get_famp_marlin()  # -> torch.ops.famp_marlin
"""
import concurrent.futures
import functools
import glob
import os
import subprocess
import sys
import sysconfig

import torch
from torch.utils import cpp_extension as _cppe

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")
_MARLIN = os.path.join(_CSRC, "marlin")
_BUILD = os.path.join(_HERE, "build")


def _arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _ensure_cuda_libs() -> list:
    """torch's CUDAContextLight.h #includes <cusparse.h>/<cublas...> which live ONLY in the pip CUDA
    (nvidia/*/include) here — but that dir is a FULL second CUDA (own crt/cuda_runtime.h) and putting
    it on -I alongside the toolkit gives two host_runtime.h -> the '__cudaLaunch' macro clash. So fill
    just the MISSING library headers (keeping the toolkit's own consistent core). Preferred: copy them
    into the toolkit include. If the toolkit isn't writable (non-root, e.g. serving as the sandbox
    'coder' user), STAGE them into a local writable dir and return it as an extra -I — only lib
    headers are staged, so the crt/cuda_runtime conflict is still avoided. Returns the list of extra
    include dirs to add (empty when the toolkit-copy succeeded)."""
    import shutil
    toolkit = next((t for t in ("/usr/local/cuda/targets/x86_64-linux/include",
                                "/usr/local/cuda/include")
                    if os.path.isfile(os.path.join(t, "cuda_runtime.h"))), None)
    if toolkit is None:
        return []
    sp = sysconfig.get_paths()["purelib"]
    staging = os.path.join(_HERE, "build", "_cuda_inc")
    extra = []
    for inc in sorted(glob.glob(os.path.join(sp, "nvidia", "*", "include"))):
        for h in glob.glob(os.path.join(inc, "*.h")):
            base = os.path.basename(h)
            if os.path.exists(os.path.join(toolkit, base)):
                continue  # already in the toolkit core; never shadow it
            try:
                shutil.copy2(h, os.path.join(toolkit, base))  # preferred: fill the toolkit in place
            except (PermissionError, OSError):
                os.makedirs(staging, exist_ok=True)           # fallback: stage locally + add to -I
                sdst = os.path.join(staging, base)
                if not os.path.exists(sdst):
                    shutil.copy2(h, sdst)
                if staging not in extra:
                    extra.append(staging)
    return extra


def _manual_build(sources, arch):
    """nvcc -rdc compile (parallel) -> nvcc -dlink -> g++ -shared famp_marlin.so."""
    os.makedirs(_BUILD, exist_ok=True)
    cc = arch.replace(".", "")  # "8.6" -> "86"
    abi = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True))
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    def _torch_paths(fn):  # include_paths/library_paths signature varies across torch versions
        for kw in ({"device_type": "cuda"}, {"cuda": True}, {}):
            try:
                return fn(**kw)
            except TypeError:
                continue
        return fn()

    iflags = []
    py_inc = sysconfig.get_paths()["include"]  # Python.h (registration.h needs it; cpp_extension adds it)
    for i in _torch_paths(_cppe.include_paths) + [py_inc, _CSRC, _MARLIN]:
        iflags += ["-I", i]
    common = [
        "-std=c++17", "-O3", "-Xcompiler", "-fPIC",
        "--relocatable-device-code=true", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        f"-gencode=arch=compute_{cc},code=sm_{cc}",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}", "-DTORCH_EXTENSION_NAME=famp_marlin",
        "-DTORCH_API_INCLUDE_EXTENSION_H", "-DNDEBUG",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    ]

    def compile_one(src):
        o = os.path.join(_BUILD, os.path.basename(src) + ".o")
        subprocess.check_call([nvcc, "-c", src, "-o", o] + common + iflags)
        return o

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        objs = list(ex.map(compile_one, sources))

    dlink = os.path.join(_BUILD, "famp_marlin_dlink.o")
    subprocess.check_call([nvcc, "-dlink", f"-gencode=arch=compute_{cc},code=sm_{cc}",
                           "--relocatable-device-code=true", "-Xcompiler", "-fPIC"]
                          + objs + ["-o", dlink])

    so = os.path.join(_BUILD, "famp_marlin.so")
    lib_dirs = list(_torch_paths(_cppe.library_paths))
    for c in (os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "targets/x86_64-linux/lib")):
        if os.path.isdir(c):
            lib_dirs.append(c)
    lflags = []
    for l in lib_dirs:
        lflags += ["-L", l]
    subprocess.check_call(["g++", "-shared", "-o", so] + objs + [dlink] + lflags
                          + ["-lc10", "-ltorch", "-ltorch_cpu", "-lc10_cuda", "-ltorch_cuda", "-lcudart"])
    torch.ops.load_library(so)
    return so


@functools.cache
def get_famp_marlin(arch: str | None = None):
    arch = arch or _arch()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    _ensure_cuda_libs()
    subprocess.check_call([sys.executable, os.path.join(_MARLIN, "generate_kernels.py"), arch], cwd=_MARLIN)
    sources = [
        os.path.join(_MARLIN, "marlin.cu"),
        os.path.join(_MARLIN, "gptq_marlin_repack.cu"),
        os.path.join(_MARLIN, "awq_marlin_repack.cu"),
        os.path.join(_CSRC, "famp_marlin_binding.cu"),
    ] + sorted(glob.glob(os.path.join(_MARLIN, "*kernel_*.cu")))
    _manual_build(sources, arch)
    return torch.ops.famp_marlin


if __name__ == "__main__":
    m = get_famp_marlin(sys.argv[1] if len(sys.argv) > 1 else None)
    ok = [op for op in ("marlin_gemm", "gptq_marlin_repack", "awq_marlin_repack") if hasattr(m, op)]
    print("FAMP_MARLIN_BUILT ops_registered:", ok)
