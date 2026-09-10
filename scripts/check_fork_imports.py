#!/usr/bin/env python3
"""Verify every `from vllm.X import Y` in the vendored tree actually resolves.

pyflakes catches undefined names *within* a module. It cannot tell you that
`from vllm.model_executor.models.qwen3_dflash import dflash_target_rope_is_neox_style`
names something that no longer exists -- and that is exactly how two release-blocking bugs
reached a GPU this cycle:

  0.28: mm_prefix_ranges          (upstream renamed it; our added lines kept the old name)
  0.29: dflash_target_rope_is_neox_style (upstream deleted it; our load_model still imported it)

Both survived a clean 3-way merge, imported fine, and only failed at runtime -- one of them only
under spec decode. This walks the AST (nothing is executed), resolves each import target against
the defining module's own AST, and reports misses.

Function-local imports are checked too; those are the ones that fail late.

    python3 scripts/check_fork_imports.py [--baseline <pristine-tree>] [tree]

With --baseline, findings that also occur in pristine upstream are dropped, so the output is
fork-introduced breakage only.
"""

import argparse
import ast
import pathlib
import sys


def module_path(root: pathlib.Path, dotted: str) -> pathlib.Path | None:
    rel = dotted.split(".")
    if rel and rel[0] == "vllm":
        rel = rel[1:]
    base = root.joinpath(*rel)
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def exported(path: pathlib.Path) -> set[str] | None:
    """Top-level names a module provides. None => can't tell (star import / __getattr__)."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                return None
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.If):        # TYPE_CHECKING / version blocks
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add(a.asname or a.name.split(".")[0])
    if "__getattr__" in names:
        return None
    return names


def scan(root: pathlib.Path) -> set[str]:
    findings: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if not node.module.startswith("vllm."):
                continue
            target = module_path(root, node.module)
            if target is None:
                continue
            have = exported(target)
            if have is None:
                continue
            for alias in node.names:
                if alias.name == "*" or alias.name in have:
                    continue
                # a submodule import (`from pkg import mod`) is fine
                if module_path(root, f"{node.module}.{alias.name}") is not None:
                    continue
                findings.add(
                    f"{path.relative_to(root.parent)}:{node.lineno}: "
                    f"cannot import '{alias.name}' from '{node.module}'"
                )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", nargs="?", default="vllm/vllm")
    ap.add_argument("--baseline", help="pristine upstream tree to subtract")
    args = ap.parse_args()

    found = scan(pathlib.Path(args.tree))
    if args.baseline:
        base = {f.split(":", 1)[1] for f in scan(pathlib.Path(args.baseline))}
        found = {f for f in found if f.split(":", 1)[1] not in base}
    for f in sorted(found):
        print(f)
    print(f"unresolved fork imports: {len(found)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
