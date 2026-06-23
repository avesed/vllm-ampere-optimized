"""Where autotune results land. Default = XDG_STATE_HOME/ampere-autotune/results (the conventional
home for run history/results: persistent, not config, not cache). `--output <file|dir>` overrides
(needed in docker, where the default is ephemeral — mount a volume + point here). The human report
is ALWAYS also printed to stdout (so docker-logs / shell redirect capture it regardless).
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Optional


def state_dir() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "ampere-autotune")


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def save_report(mode: str, text: str, data: Optional[dict] = None, output: Optional[str] = None) -> Optional[str]:
    """Write the report (.txt human + optional .json machine). `output` = explicit file or dir; else
    a timestamped file under the state dir. Returns the .txt path, or None if it couldn't write."""
    name = f"{_stamp()}-{mode}.txt"
    if output:
        path = os.path.join(output, name) if (os.path.isdir(output) or output.endswith(os.sep)) else output
    else:
        path = os.path.join(state_dir(), "results", name)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(text.rstrip("\n") + "\n")
        if data is not None:
            with open(os.path.splitext(path)[0] + ".json", "w") as f:
                json.dump(data, f, indent=2, default=str)
        return path
    except OSError:
        return None


def emit(text: str, mode: str, args=None, data: Optional[dict] = None) -> None:
    """Print the human report to stdout (always), then persist it and print the saved path."""
    print(text)
    output = getattr(args, "output", None) if args is not None else None
    path = save_report(mode, text, data=data, output=output)
    if path:
        print(f"\n[saved] {path}")
