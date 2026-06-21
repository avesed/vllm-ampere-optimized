"""CI-safe test setup: the whole suite runs with NO GPU and NO root.

Makes the standalone package importable without installation.
"""
import os
import sys

# autotune/ on sys.path so `import ampere_autotune` works in-tree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
