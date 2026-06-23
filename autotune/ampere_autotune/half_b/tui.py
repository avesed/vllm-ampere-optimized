"""Live TUI for the HALF-B characterize/soak so the user watches progress in real time.

render_frame() is a PURE text table (unit-tested). Reporter wraps it: it uses rich.Live for a
smooth in-place refresh when `rich` is installed, else falls back to plain reprinted frames — so
the tool degrades gracefully with no hard dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional

_COLS = [("gpu", 12, "GPU"), ("phase", 11, "phase"), ("offset", 8, "offset"),
         ("read", 11, "read GB/s"), ("mismatch", 9, "mismatch"), ("temp", 6, "temp"),
         ("status", 0, "status")]


def _fmt(key: str, val) -> str:
    if val is None:
        return "-"
    if key == "offset":
        return f"+{int(val)}"
    if key == "read":
        return f"{float(val):.0f}"
    if key == "mismatch":
        return str(int(val))
    if key == "temp":
        return f"{int(val)}C"
    return str(val)


def render_frame(rows: List[Dict], *, title: str = "ampere-autotune --hw", footer: str = "") -> str:
    """Pure: render the per-GPU rows into a fixed-width table string."""
    head = "".join((h.ljust(w) if w else h) for _, w, h in _COLS)
    out = [title, head, "-" * max(len(head), len(title))]
    for r in rows:
        line = ""
        for key, w, _ in _COLS:
            cell = _fmt(key, r.get(key))
            line += (cell.ljust(w) if w else cell)
        out.append(line)
    if footer:
        out.append(footer)
    return "\n".join(out)


class Reporter:
    """Live updater keyed by GPU. ``update(gpu, phase=..., offset=..., read=..., mismatch=...,
    temp=..., status=...)`` then refreshes the display. Use as a context manager."""

    def __init__(self, title: str = "ampere-autotune --hw", use_rich: Optional[bool] = None,
                 sink=print):
        self.title = title
        self.rows: Dict[str, Dict] = {}
        self.footer = ""
        self._sink = sink
        self._live = None
        if use_rich is None:
            try:
                import rich  # noqa: F401
                use_rich = True
            except Exception:
                use_rich = False
        self._use_rich = use_rich

    def __enter__(self):  # pragma: no cover - rich/tty path
        if self._use_rich:
            try:
                from rich.live import Live
                from rich.text import Text
                self._Text = Text
                self._live = Live(Text(self.frame()), auto_refresh=False)
                self._live.__enter__()
            except Exception:
                self._use_rich = False
        return self

    def __exit__(self, *exc):  # pragma: no cover - rich/tty path
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
        else:
            self._sink(self.frame())
        return False

    def frame(self) -> str:
        return render_frame(self.rowlist(), title=self.title, footer=self.footer)

    def rowlist(self) -> List[Dict]:
        return [self.rows[g] for g in sorted(self.rows)]

    def update(self, gpu: str, **fields) -> None:
        self.rows.setdefault(gpu, {"gpu": gpu}).update(fields)
        self._refresh()

    def set_footer(self, text: str) -> None:
        self.footer = text
        self._refresh()

    def _refresh(self) -> None:  # pragma: no cover - rich/tty path
        if self._live is not None:
            self._live.update(self._Text(self.frame()), refresh=True)
        elif self._sink is not print:
            return  # silent sink (tests)
        else:
            print(self.frame(), flush=True)
