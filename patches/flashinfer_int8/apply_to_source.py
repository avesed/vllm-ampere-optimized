#!/usr/bin/env python3
"""Apply the validated int8-QK edits (i1_apply + i4_apply + i4_compute_qk) to the VENDORED
FlashInfer SOURCE tree at repo `flashinfer/` (v0.6.12), producing the maintainable fork source.

The three apply scripts target the INSTALLED-PACKAGE layout (FI=dirname(flashinfer.__file__);
headers under FI/data/include/...). The source tree uses include/ + csrc/ at the repo root and
the python package at flashinfer/flashinfer/. We bridge the two WITHOUT re-transcribing any edit:
  1. temp symlinks  flashinfer/flashinfer/data/include -> ../../include , data/csrc -> ../../csrc
  2. stub `import flashinfer` so FI = flashinfer/flashinfer (no torch needed on this box)
  3. exec the three scripts unchanged (their strict asserts FAIL LOUDLY if any anchor doesn't
     match the source — same v0.6.12 content as the package, so they match == proof of equivalence)
  4. remove the symlinks + *.orig/*.i4orig/*.i2orig backups so only the edits remain.
Run locally: python3 patches/flashinfer_int8/apply_to_source.py
"""
import sys, os, types, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
# default: the vendored repo clone; override with argv[1] to replay onto a fresh upstream
# checkout (e.g. patch-drift-check / revendor against a new tag in a temp dir).
SRC = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "flashinfer")
PKG = os.path.join(SRC, "flashinfer")           # python package root (== FI for the scripts)
assert os.path.isdir(PKG), f"no flashinfer source package at {PKG}"
assert os.path.isfile(os.path.join(SRC, "include/flashinfer/mma.cuh")), "source headers missing"

links = []
for rel, target in [("data/include", os.path.join(SRC, "include")),
                    ("data/csrc", os.path.join(SRC, "csrc"))]:
    lp = os.path.join(PKG, rel)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    if os.path.islink(lp):
        os.unlink(lp)
    os.symlink(target, lp)
    links.append(lp)

stub = types.ModuleType("flashinfer")
stub.__file__ = os.path.join(PKG, "__init__.py")
sys.modules["flashinfer"] = stub

try:
    for script in ["i1_apply.py", "i4_apply.py", "i4_compute_qk.py"]:
        path = os.path.join(HERE, script)
        print(f"\n=== exec {script} (-> source) ===")
        g = {"__file__": path, "__name__": "__main__"}
        exec(compile(open(path).read(), script, "exec"), g)
finally:
    for lp in links:
        if os.path.islink(lp):
            os.unlink(lp)
    # drop the empty data/ dir the symlinks lived in
    dp = os.path.join(PKG, "data")
    if os.path.isdir(dp) and not os.listdir(dp):
        os.rmdir(dp)
    # remove backups the scripts wrote into the source tree
    n = 0
    for pat in ("*.orig", "*.i4orig", "*.i2orig"):
        for f in glob.glob(os.path.join(SRC, "**", pat), recursive=True):
            os.remove(f); n += 1
    print(f"\ncleaned {n} backup file(s) + temp symlinks")
print("APPLY_TO_SOURCE_DONE")
