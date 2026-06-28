# SPDX-License-Identifier: Apache-2.0
"""famp's OWN vendored XQA spec-decode verify kernel (independent of installed FlashInfer's xqa).
Source copied from flashinfer csrc/xqa (self-contained) + builder/wrapper adapted (csrc->local,
Ampere sm80/86 un-gated). Uses stock FlashInfer only as the JIT toolchain (gen_jit_spec/CompilationContext)."""
from ._xqa import xqa
__all__ = ["xqa"]
