"""exp-147 member drivers: shared plumbing (CLI, imports, hashes, record). No model logic.

Two code roots meet in a member driver:

* `SRC = $PRC_SOURCE_ROOT` holds the member code (`prc/`, `experiments/exp-*/code/`), the
  caches and the raw data. The member is imported from there and called, not paraphrased.
* This worktree holds the split code: `prc/weekpair.py` and `code/pilot.py`.

Both trees have a package named `prc`. `bootstrap()` imports SRC's `prc` as `prc`, then
loads this worktree's `prc/weekpair.py` under the name `prc.weekpair`, so the member's
`prc.*` imports resolve to SRC and every split mask comes from the worktree. It refuses to
run if either binding is wrong.

Import order matters for the thread cap: `parse_args()` sets the BLAS/OpenMP thread
variables, so a driver calls it before anything imports numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../exp-147-foldserve/code/members
CODE = HERE.parent                               # .../exp-147-foldserve/code
WT = HERE.parents[3]                             # the exp-147 worktree root
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PRC_THREADS")
SMOKE_FRACTION = 0.02
SMOKE_ROUNDS = 20
SMOKE_SEED = 20260930


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", help="a pilot.FITS name: outer-k, inner-a-b (a<b) or final")
    g.add_argument("--split", help="alias: the fit that writes this split's artifact")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_FRACTION:.0%} of the label rows, {SMOKE_ROUNDS} rounds, "
                         "written under <out>/smoke/")
    a = ap.parse_args(argv)
    if a.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for v in THREAD_VARS:            # forced, so the declared count is the count that runs
        os.environ[v] = str(a.threads)
    a.out = a.out.resolve()
    a.dest = a.out / "smoke" if a.smoke else a.out
    return a


def resolve_fit(pilot, a) -> tuple[str, list]:
    """(fit name, [(split, owner), ...]) from `--fit`, or from the `--split` alias."""
    name = a.fit or a.split
    if name in pilot.FITS:
        return name, pilot.artifacts_of(name)
    for f in pilot.FITS:
        if f != "final" and any(s.name == name for s, _ in pilot.artifacts_of(f)):
            return f, pilot.artifacts_of(f)
    raise SystemExit(f"{name} is neither a pilot.FITS name nor a split one of them writes")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bootstrap():
    """(SRC, weekpair, pilot): member code from SRC, split code from this worktree."""
    src = Path(os.environ["PRC_SOURCE_ROOT"]).resolve()
    if "prc" in sys.modules:
        raise RuntimeError("prc was imported before bootstrap(); its origin is unknown")
    sys.path.insert(0, str(src))
    import prc                                                   # noqa: E402
    if Path(prc.__file__).resolve().parent != src / "prc":
        raise RuntimeError(f"prc resolved to {prc.__file__}, not {src / 'prc'}")
    wp = _load("prc.weekpair", WT / "prc" / "weekpair.py")
    prc.weekpair = wp
    n_path = len(sys.path)
    pilot = _load("exp147_pilot", CODE / "pilot.py")
    if pilot.WP is not wp:
        raise RuntimeError("pilot.py bound a weekpair other than this worktree's")
    # pilot.py puts the worktree root first on sys.path; nothing else may resolve there
    del sys.path[0:len(sys.path) - n_path]
    return src, wp, pilot


def member_function(path: Path, name: str, namespace: dict):
    """One top-level function of a member file, compiled from that file's own source.

    For a module whose import has side effects (a directory created at import), this runs
    the function's exact text without running the rest of the module.
    """
    import ast
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(fns) != 1:
        raise RuntimeError(f"{path}: {len(fns)} top-level functions named {name}")
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def refuse_existing(dest: Path, stage: str, split) -> None:
    """Fail before the fit, not after it (pilot.write is write-once)."""
    for suf in (".parquet", ".json"):
        p = dest / f"{stage}-{split.name}{suf}"
        if p.exists():
            raise SystemExit(f"{p} exists; write-once")


def smoke_keep(n: int):
    """The smoke row sample: a fixed-seed Bernoulli(SMOKE_FRACTION) draw over n rows."""
    import numpy as np
    return np.random.default_rng(SMOKE_SEED).random(n) < SMOKE_FRACTION


def code_hashes(src: Path, drivers: list[Path], src_files=()) -> dict:
    """sha256 of every imported module file under SRC (plus `src_files`, member files run
    without an import), and of the exp-147 files used."""
    out = {str(Path(f).resolve().relative_to(src)): sha(f) for f in src_files}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.suffix != ".py" or src not in p.parents:
            continue
        rel = p.relative_to(src)
        if rel.parts[0] in ("prc", "experiments"):      # member source; not SRC/.venv
            out[str(rel)] = sha(p)
    wt = {str(p.relative_to(WT)): sha(p) for p in
          [*drivers, HERE / "_wp_common.py", CODE / "pilot.py", WT / "prc" / "weekpair.py"]}
    return {"source_root": dict(sorted(out.items())), "exp147_worktree": wt}


def git_head(root: Path):
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:                                            # noqa: BLE001
        return None


def environment(src: Path, threads: int) -> dict:
    import lightgbm
    import numpy as np
    import pandas as pd
    return {"source_root": str(src), "source_git_head": git_head(src),
            "worktree_git_head": git_head(WT), "host": socket.gethostname(),
            "python": platform.python_version(), "lightgbm": lightgbm.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "threads": threads,
            "thread_env": {v: os.environ.get(v) for v in THREAD_VARS}}
