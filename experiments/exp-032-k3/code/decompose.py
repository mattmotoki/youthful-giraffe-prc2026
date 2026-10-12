"""Where the K3 member's fold movement sits: regime, tail band, airport.

Reads the two committed OOF sets this experiment writes -- `--arm base` and
`--arm window` -- and decomposes the difference. No refit, no data beyond the
member's own predictions and the regime columns exp-026's frames already
address, so it costs seconds and can be re-run after any arm.

    PRC_THREADS=1 python code/decompose.py

The currency is the member's own fold RMSE, not the board: this member reaches
the board through blend-5's 0.2265 weight outside `nm_late` and 0.0000 inside
it, and through the stacker. `code/candidate.py` is what prices it there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = F.FOLDS


def contrib(y, a, b, m) -> float:
    """The signed seconds of fold RMSE the rows in `m` account for.

    `sqrt(mean(e_a^2)) - sqrt(mean(e_a^2 with rows m swapped to e_b^2))`: what
    the fold would gain if ONLY these rows moved from a to b. Positive means
    the arm `b` improves on them.

    These parts are NOT exactly additive, and an earlier version of this
    docstring wrongly said they were (Sol round 1, blocking 2). The MEAN of
    squared error is additive over a partition; the square ROOT taken after
    each swap is not. The residual is written out as its own `ADDS-UP` row per
    fold and per family rather than assumed away -- on the july_loo member arm
    it is between -0.0055 and +0.0016 s against a 2.0476 s total, under 0.3 %,
    which is why the readings drawn from it stand. Read a part as "what this
    slice is worth on its own", never as a term in an exact sum.
    """
    ea = (y - a) ** 2
    eb = (y - b) ** 2
    swap = ea.copy()
    swap[m] = eb[m]
    return float(np.sqrt(ea.mean()) - np.sqrt(swap.mean()))


def main() -> None:
    rows = []
    for fold in FOLDS:
        bp = EXP / "oof" / f"k3-tail-base-{fold}.parquet"
        wp = EXP / "oof" / f"k3-tail-window-{fold}.parquet"
        if not (bp.exists() and wp.exists()):
            print(f"[skip] {fold}: both arms are not on disk yet")
            continue
        base = pd.read_parquet(bp)
        win = pd.read_parquet(wp)
        g = F.gates(fold)[["MVT_ID_mvt", "regime", "airport"]]
        m = base.merge(win[["MVT_ID_mvt", "y_pred"]], on="MVT_ID_mvt",
                       suffixes=("_base", "_win")).merge(g, on="MVT_ID_mvt", how="left")
        if len(m) != len(base):
            raise SystemExit(f"{fold}: join covers {len(m)} of {len(base)}")
        y = m["y_true"].to_numpy("float64")
        a = m["y_pred_base"].to_numpy("float64")
        b = m["y_pred_win"].to_numpy("float64")
        r = lambda p, k=None: float(np.sqrt(np.mean(((y - p) ** 2)[
            np.ones(len(y), bool) if k is None else k])))
        rows.append({"fold": fold, "cut": "ALL", "n": int(len(y)),
                     "base": round(r(a), 4), "window": round(r(b), 4),
                     "gain": round(r(a) - r(b), 4), "share_of_gain": 1.0})
        total = r(a) - r(b)
        # Sol round 1, blocking 2: `frames.gates` returns the BLEND regimes
        # (`nm_bulk`, `nm_late`, `nonm`, `noac`), not the tail router's heads,
        # so the old `regime=nm` cut matched nothing and no NM row was ever
        # written. The labels are read from the data instead of assumed.
        seen = [str(k) for k in pd.unique(m["regime"]) if pd.notna(k)]
        want = ["nm_bulk", "nm_late", "nonm", "noac"]
        regimes = [k for k in want if k in seen] + sorted(set(seen) - set(want))
        if not regimes:
            raise SystemExit(f"{fold}: no regime labels on the joined frame")
        fam = {"regime": [f"regime={k}" for k in regimes],
               "band": ["y<=3600", "y>3600"], "airport": []}
        cuts = {f"regime={k}": (m["regime"].to_numpy() == k) for k in regimes}
        cuts["y<=3600"] = y <= 3600.0
        cuts["y>3600"] = y > 3600.0
        for apt in sorted(pd.unique(m["airport"].astype(str))):
            cuts[f"apt={apt}"] = m["airport"].astype(str).to_numpy() == apt
            fam["airport"].append(f"apt={apt}")
        got = {}
        for name, k in cuts.items():
            if not k.any():
                continue
            c = contrib(y, a, b, k)
            got[name] = c
            rows.append({"fold": fold, "cut": name, "n": int(k.sum()),
                         "base": round(r(a, k), 4), "window": round(r(b, k), 4),
                         "gain": round(c, 4),
                         "share_of_gain": round(c / total, 4) if total else float("nan")})
        # each family partitions the fold, so its parts SHOULD sum to `total`
        # if the decomposition were additive. It is not; the gap is recorded.
        for label, names in fam.items():
            part = sum(got[n] for n in names if n in got)
            rows.append({"fold": fold, "cut": f"ADDS-UP {label}",
                         "n": int(sum(int(cuts[n].sum()) for n in names if n in got)),
                         "base": round(total, 6), "window": round(part, 6),
                         "gain": round(part - total, 6),
                         "share_of_gain": round((part - total) / total, 6) if total else float("nan")})
        moved = np.abs(b - a)
        rows.append({"fold": fold, "cut": "MOVE rms/max/n>1s", "n": int((moved > 1).sum()),
                     "base": round(float(np.sqrt((moved ** 2).mean())), 4),
                     "window": round(float(moved.max()), 4), "gain": np.nan,
                     "share_of_gain": np.nan})
    df = pd.DataFrame(rows)
    out = EXP / "analysis" / "member-decompose.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.to_string(index=False))
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
