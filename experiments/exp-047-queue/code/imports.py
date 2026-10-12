"""exp-047-queue: import a sibling experiment's module by PATH, not by name.

Five experiments in this repo ship a module called `census`, three ship `serve`,
two ship `segments` and three ship `base`.  `sys.path.insert` plus a plain import
therefore resolves to whichever directory was inserted last.  This loads a named
file under a unique module name, so which file was loaded is a fact about the call.
Copied from `exp-046-surfacefit/code/imports.py`, which this lane must not edit.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(exp: str, module: str, deps: tuple[str, ...] = ()):
    """`exp` is an experiment directory name, `module` a file in its `code/`."""
    path = ROOT / "experiments" / exp / "code" / f"{module}.py"
    key = f"prc47__{exp.replace('-', '_')}__{module}"
    if key in sys.modules:
        return sys.modules[key]
    dirs = [str(path.parent)] + [str(ROOT / "experiments" / e / "code") for e in deps]
    added = [d for d in dirs if d not in sys.path]
    for d in added:
        sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    finally:
        for d in added:
            if d in sys.path:
                sys.path.remove(d)
    return mod
