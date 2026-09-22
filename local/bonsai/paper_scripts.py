"""Run one of the paper's scripts unchanged (apart from constants its docstring says to edit) inside the Bonsai-only sandbox.

The sandbox local/run_bonsai/ has datasets/ and results/Bonsai_2_27B_ternary/ only, so scripts that scan results/ for
"every model" see just Bonsai. Outputs land under local/run_bonsai/ and are collected into results/bonsai/ afterwards.
"""
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SANDBOX = REPO / "local" / "run_bonsai"


def run(script, replacements=None, argv=None):
    """exec scripts/<script> with `replacements` {old: new} applied to its source (each must occur exactly once)."""
    path = REPO / "scripts" / script
    src = path.read_text()
    for old, new in (replacements or {}).items():
        assert src.count(old) == 1, f"{script}: {old!r} occurs {src.count(old)} times"
        src = src.replace(old, new)
    old_cwd, old_argv, old_path = os.getcwd(), sys.argv, list(sys.path)
    os.chdir(SANDBOX)
    sys.argv = [str(path)] + (argv or [])
    try:
        exec(compile(src, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    finally:
        os.chdir(old_cwd); sys.argv = old_argv; sys.path[:] = old_path
