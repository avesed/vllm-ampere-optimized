"""famp owned FA2 prefill (fp16-PV) — vendored prefill.cuh + thin JIT harness + owned marshalling.
See _prefill.single_prefill (the drop-in) and _jit_prefill.get_famp_prefill_module (the builder)."""
from ._jit_prefill import gen_famp_prefill_spec, get_famp_prefill_module  # noqa: F401
from ._prefill import single_prefill  # noqa: F401
