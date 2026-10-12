"""exp-046-surfacefit: import a sibling experiment's module by PATH, not by name.

Five experiments in this repo ship a module called `census`, three ship `serve` and
two ship `segments`.  `sys.path.insert` plus `import census` therefore resolves to
whichever experiment happened to be inserted last, and the answer changes with import
order -- a bug that runs silently until one of the two modules lacks the attribute
being called.  This loads a named file and gives it a unique module name, so which
file was loaded is a fact about the call and not about the order.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load(exp: str, module: str, deps: tuple[str, ...] = ()):
    """`exp` is an experiment directory name, `module` a file in its `code/`.

    `deps` names further experiments whose `code/` the module imports from by plain
    name (`allsegments` imports `segments` from `exp-043-pilot`).  Those directories
    are on the path only while the module executes.
    """
    path = ROOT / "experiments" / exp / "code" / f"{module}.py"
    key = f"prc46__{exp.replace('-', '_')}__{module}"
    if key in sys.modules:
        return sys.modules[key]
    # the module's own siblings must still resolve, so its directory goes on the
    # path while it is executed and comes off again afterwards
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
