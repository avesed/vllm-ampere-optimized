#!/usr/bin/env python3
"""Preprocessor-balance check for the vendored native tree.

The Python side of a re-vendor is covered by compileall + a pyflakes diff against the pristine
upstream tag. The native side had no equivalent, so a merge that left one stray `#endif` in
csrc/libtorch_stable/ops.h was only discovered ~4 hours into a from-source build. Conditional
nesting is an absolute property -- no upstream reference needed -- so this runs in seconds.

    python3 scripts/check_native_guards.py [tree ...]      # default: vllm/csrc flashampere
"""

import pathlib
import re
import sys

EXTS = {".h", ".hpp", ".cuh", ".cu", ".cc", ".cpp", ".cxx"}
OPEN = re.compile(r"^\s*#\s*(if|ifdef|ifndef)\b")
MID = re.compile(r"^\s*#\s*(elif|else)\b")
CLOSE = re.compile(r"^\s*#\s*endif\b")


def check(path: pathlib.Path) -> str | None:
    depth, opened_at = 0, []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if OPEN.match(line):
            depth += 1
            opened_at.append(lineno)
        elif CLOSE.match(line):
            depth -= 1
            if depth < 0:
                return f"{path}:{lineno}: #endif without a matching #if"
            opened_at.pop()
        elif MID.match(line) and depth == 0:
            return f"{path}:{lineno}: #else/#elif outside any #if"
    if depth > 0:
        return f"{path}:{opened_at[0]}: #if opened here is never closed ({depth} unclosed)"
    return None


def main() -> int:
    roots = [pathlib.Path(a) for a in sys.argv[1:]] or [
        pathlib.Path("vllm/csrc"),
        pathlib.Path("flashampere"),
    ]
    files = [p for r in roots if r.exists() for p in r.rglob("*") if p.suffix in EXTS]
    bad = [msg for p in sorted(files) if (msg := check(p))]
    for msg in bad:
        print(msg)
    print(f"checked {len(files)} native files: {len(bad)} unbalanced")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
