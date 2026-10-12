"""The month this lane is working, and the leave-one-out fold that serves it.

§7bk did 2025-11; §7bo does 2025-10.  The producers are unchanged code -- the
month and the fold come from the environment and DEFAULT TO NOVEMBER, so every
§7bk command reproduces byte-for-byte with no variable set.

`prc.data.FOLDS` ships `november_loo` and nothing for October.  The fold is added
**in this process only**, through the same `setdefault` the module documents as
its extension point ("additive, no existing fold changed").  It is NOT written
into `prc/data.py`: surfacefit's bag runs beside this lane, and a shared-code
edit would change a fold table under another desk's running job.  Runtime
injection is the same discipline this lane used for `frames.FAMILY` and
`deliver.THREADS`.
"""
from __future__ import annotations

import os

MONTH = os.environ.get("PRC_LANE_MONTH", "2025-11")
FOLD = os.environ.get("PRC_LANE_FOLD", f"{MONTH[:4]}-{MONTH[5:]}_loo")
_NAMES = {"2025-02": "february_loo", "2025-03": "march_loo",
          "2025-04": "april_loo", "2025-05": "may_loo",
          "2025-06": "june_loo", "2025-08": "august_loo",
          "2025-09": "september_loo", "2025-10": "october_loo",
          "2025-11": "november_loo"}
if FOLD.endswith("-_loo") or FOLD == f"{MONTH[:4]}-{MONTH[5:]}_loo":
    FOLD = _NAMES.get(MONTH, FOLD)


#: True once `install()` has had to ADD the fold -- i.e. `prc/data.py` does not
#: ship it, so every subprocess needs the same injection.  False for November,
#: whose fold is shipped, which keeps §7bk's commands byte identical.
ADDED = False


def install() -> str:
    """Add this lane's leave-one-out fold to `prc.data.FOLDS`, in-process only."""
    global ADDED
    import prc.data as D

    if FOLD not in D.FOLDS:
        ADDED = True
        D.FOLDS.setdefault(FOLD, D.Fold(
            FOLD, tuple(m for m in D.MONTHS if m != MONTH), MONTH))
    f = D.FOLDS[FOLD]
    if f.valid_month != MONTH or MONTH in f.train_months:
        raise SystemExit(f"{FOLD} does not hold {MONTH} out")
    if len(f.train_months) != len(D.MONTHS) - 1:
        raise SystemExit(f"{FOLD} trains on {len(f.train_months)} months, not 11")
    return FOLD


def family(round_budget: str = "m11") -> None:
    """`frames.FAMILY` is a ROUND-BUDGET map and has no entry for a loo fold."""
    import frames as F

    F.FAMILY.setdefault(FOLD, round_budget)
