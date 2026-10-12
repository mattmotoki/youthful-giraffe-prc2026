"""Did adding the optional weight break the UNWEIGHTED path? Refit member 1 and compare bytes.

`covmatch.py` needed a sample weight inside `arms.arm_lgb`, and the way that is normally done
-- an optional argument defaulting to the old behaviour -- is the shape section 45.3 lists as
the source of five silent defects.  Reading the diff is not evidence; refitting is.  This
refits placebo member 1 through the DEFAULT path and requires the three fold vectors to equal
`work/d-bundle-placebo1-255-free.npz` **bit for bit**.  Anything else means the edit moved the
shipped head, and every cached member and every candidate priced against them is suspect.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/wcheck.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ps = argparse.ArgumentParser()
_ps.add_argument("--code", default=None,
                 help="a directory to search BEFORE code/ -- used to run this check against "
                      "the git-HEAD arms.py and chainfold.py, which is what separates 'the "
                      "edit moved the fit' from 'a fresh LightGBM fit is not reproducible'")
_ps.add_argument("--out", default=None)
_ARGS = _ps.parse_args()
sys.path.insert(0, str(HERE))
if _ARGS.code:
    sys.path.insert(0, str(Path(_ARGS.code).resolve()))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402

SEED = 1
CACHE = EXP / "work/d-bundle-placebo1-255-free.npz"
OUT = Path(_ARGS.out) if _ARGS.out else EXP / "analysis/weight-path-check.json"


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    rng = np.random.default_rng(SEED)
    blocks = {}
    for m, b in blk.items():
        pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
        blocks[m] = pb
    real = A.month_block
    A.month_block = (lambda blocks: (
        lambda month, prefix_months, rule=None: blocks[month]))(blocks)
    try:
        pan = A.panel_with_block(BA.load())
    finally:
        A.month_block = real
    setting_was, hold_was = CF.SETTING, CF.HOLD
    CF.SETTING, CF.HOLD = {"num_leaves": 255}, ()
    try:
        d = CF.fit_arm(pan)                      # the DEFAULT path: no wmap, no prov
    finally:
        CF.SETTING, CF.HOLD = setting_was, hold_was

    want = {f: v.astype("float64") for f, v in np.load(CACHE).items()}
    same = {f: bool(np.array_equal(d[f], want[f])) for f in A.FOLDS}
    rec = {"what": __doc__.strip().splitlines()[0], "seed": SEED,
           "code_searched_first": _ARGS.code,
           "arms_module_file": A.__file__, "chainfold_module_file": CF.__file__,
           "cache": str(CACHE.relative_to(ROOT)),
           "identical_by_fold": same, "identical": bool(all(same.values())),
           "max_abs_diff_by_fold": {f: float(np.max(np.abs(d[f] - want[f])))
                                    for f in A.FOLDS},
           "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    for f in A.FOLDS:
        print(f"  {f:10s} identical {same[f]}  max|diff| {rec['max_abs_diff_by_fold'][f]:.3e}")
    try:
        shown = OUT.resolve().relative_to(ROOT)
    except ValueError:
        shown = OUT
    print("written to", shown, f"({rec['seconds']:.0f}s)")
    if not rec["identical"]:
        raise SystemExit("THE UNWEIGHTED PATH MOVED -- the optional weight changed the "
                         "shipped head. Every cached member and every candidate priced "
                         "against them is suspect until this is explained.")


if __name__ == "__main__":
    main()
