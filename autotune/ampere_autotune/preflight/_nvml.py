"""Thin, defensive NVML adapter.

Everything here degrades gracefully: if ``nvidia-ml-py`` is absent, the GPU is missing,
or an NVML symbol does not exist on this driver, callers get a clean ``(ok, value)`` or a
classified error string instead of an exception. This keeps preflight CI-safe (it runs with
NO GPU and NO root) and makes every probe report an OUTCOME, never a crash.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:  # nvidia-ml-py exposes the module as `pynvml`
    import pynvml as _pynvml  # type: ignore
except Exception:  # pragma: no cover - import guard
    _pynvml = None


def nvml() -> Optional[Any]:
    """Return the pynvml module, or None if unavailable."""
    return _pynvml


def available() -> bool:
    return _pynvml is not None


@dataclass
class Call:
    ok: bool
    value: Any = None
    error: str = ""          # human string
    kind: str = ""           # "NOT_SUPPORTED" | "NO_PERMISSION" | "MISSING_SYMBOL" | "ERROR"


def _classify(exc: Exception) -> str:
    """Map an NVMLError to a stable kind string (version-tolerant)."""
    if _pynvml is None:
        return "ERROR"
    val = getattr(exc, "value", None)
    for kind, const in (
        ("NOT_SUPPORTED", "NVML_ERROR_NOT_SUPPORTED"),
        ("NO_PERMISSION", "NVML_ERROR_NO_PERMISSION"),
    ):
        c = getattr(_pynvml, const, object())
        if val is not None and val == c:
            return kind
    # fall back to message sniffing for odd wrappers
    msg = str(exc).lower()
    if "not supported" in msg:
        return "NOT_SUPPORTED"
    if "permission" in msg or "insufficient" in msg:
        return "NO_PERMISSION"
    return "ERROR"


def call(fn_name: str, *args) -> Call:
    """Invoke an NVML function by name, never raising.

    Returns a Call with ok/value or a classified failure. Unknown symbols (older drivers)
    return kind="MISSING_SYMBOL" so probes can say "this driver is too old" cleanly.
    """
    if _pynvml is None:
        return Call(False, error="nvidia-ml-py not installed", kind="MISSING_SYMBOL")
    fn = getattr(_pynvml, fn_name, None)
    if fn is None:
        return Call(False, error=f"{fn_name} not in this NVML build", kind="MISSING_SYMBOL")
    try:
        return Call(True, value=fn(*args))
    except Exception as exc:  # NVMLError or anything else
        return Call(False, error=str(exc), kind=_classify(exc))


class Session:
    """Context manager around nvmlInit/Shutdown; yields handles or degrades cleanly."""

    def __init__(self) -> None:
        self.ok = False
        self.reason = ""

    def __enter__(self) -> "Session":
        if _pynvml is None:
            self.reason = "nvidia-ml-py not installed"
            return self
        try:
            _pynvml.nvmlInit()
            self.ok = True
        except Exception as exc:
            self.reason = f"nvmlInit failed: {exc} (no driver / no GPU?)"
        return self

    def __exit__(self, *_) -> None:
        if self.ok and _pynvml is not None:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass

    def device_count(self) -> int:
        if not self.ok:
            return 0
        cnt = call("nvmlDeviceGetCount")
        return int(cnt.value) if cnt.ok else 0

    def handle(self, idx: int):
        h = call("nvmlDeviceGetHandleByIndex", idx)
        return h.value if h.ok else None
