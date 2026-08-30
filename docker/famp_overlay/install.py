"""Install the flashampere backend into an already-installed vLLM (validation overlay)."""

import pathlib
import shutil
import subprocess
import sys

import vllm

VLLM = pathlib.Path(vllm.__file__).parent
HERE = pathlib.Path(__file__).parent

# 1) the package itself
dst = VLLM / "v1" / "attention" / "backends" / "flashampere"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(HERE.parent / "flashampere", dst)
print(f"[famp-overlay] package -> {dst}")

# 2) the vllm.general_plugins entry point (register_flashampere runs in every worker process)
plugin = pathlib.Path("/tmp/famp/plugin")
(plugin / "famp_plugin").mkdir(parents=True, exist_ok=True)
(plugin / "famp_plugin" / "__init__.py").write_text(
    "from vllm.v1.attention.backends.flashampere.backend import register_flashampere\n"
    '__all__ = ["register_flashampere"]\n'
)
(plugin / "pyproject.toml").write_text(
    '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n'
    '[project]\nname = "famp-plugin"\nversion = "0"\n\n'
    '[project.entry-points."vllm.general_plugins"]\n'
    'flashampere = "famp_plugin:register_flashampere"\n'
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(plugin)],
    check=True,
)
print("[famp-overlay] entry point installed")

# 3) the cuda.py prepend (same block the fork's patch 0008 adds)
cuda_py = VLLM / "platforms" / "cuda.py"
src = cuda_py.read_text()
# Same rewrite the fork's patch 0008 does: turn the Ampere fall-through `return [...]` into a
# named list, prepend CUSTOM when famp registered it, then return. The FLASH_ATTN-first ordering
# makes this unique (the other TURBOQUANT list is the FLASHINFER-first SM100f branch).
anchor = """        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.TURBOQUANT,
            ]
"""
block = """        else:
            backends = [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.TURBOQUANT,
            ]
            # [flashampere] Prepend the unified Ampere backend when registered (its
            # vllm.general_plugins entry-point ran with VLLM_FLASHAMPERE=1) on Ampere (sm_8x).
            # CUSTOM self-fences via validate_configuration, so non-Ampere / fp8-KV / encoder
            # layers fall through to FLASH_ATTN.
            if (
                device_capability.major == 8
                and AttentionBackendEnum.CUSTOM.is_overridden()
            ):
                backends.insert(0, AttentionBackendEnum.CUSTOM)
            return backends
"""
if "flashampere" in src:
    print("[famp-overlay] cuda.py already prepends CUSTOM")
else:
    assert src.count(anchor) == 1, f"cuda.py anchor not found ({src.count(anchor)}x)"
    cuda_py.write_text(src.replace(anchor, block))
    print(f"[famp-overlay] cuda.py prepend -> {cuda_py}")

# 4) Gemma4: skip the TRITON force when famp is on (fork patch; famp owns the hd512 full-attn
#    layers, so there is no mixed-backend divergence to avoid).
cfg_py = VLLM / "model_executor" / "models" / "config.py"
src = cfg_py.read_text()
if "_famp_on" in src:
    print("[famp-overlay] models/config.py already famp-aware")
else:
    a1 = "        if is_fa_version_supported(4) and max_head_dim <= 512:\n"
    a2 = (
        "        elif vllm_config.attention_config.backend is None:\n"
        "            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN\n"
    )
    assert src.count(a1) == 1 and src.count(a2) == 1, (src.count(a1), src.count(a2))
    src = src.replace(
        a1,
        "        import os as _os\n\n"
        '        _famp_on = _os.environ.get("VLLM_FLASHAMPERE", "0") in ("1", "true", "True")\n\n'
        + a1,
    )
    src = src.replace(
        a2,
        "        elif vllm_config.attention_config.backend is None and not _famp_on:\n"
        "            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN\n",
    )
    cfg_py.write_text(src)
    print(f"[famp-overlay] Gemma4 TRITON-force gate -> {cfg_py}")
