"""vllm.general_plugins entry point: make famp own the Marlin kernel for W4A8/W4A16 layers.

The implementation lives in kernel.py next to FampMarlinKernel (so the kernel + its registration are
one unit). This module just re-exports register_fampmarlin so both entry-point paths work:
    flashampere.marlin.plugin:register_fampmarlin   (used by SERVE_BENCH / the .dist-info)
    flashampere.marlin.kernel:register_fampmarlin
"""
from flashampere.marlin.kernel import register_fampmarlin

__all__ = ["register_fampmarlin"]
