"""The day-clustered interval on `envctl.py`'s july_loo difference.

`envctl.py` prints each run's fold RMSE against v6.  The quantity that answers
the question is the difference **between the two runs**, both of them Kaggle,
one with the window block and one without, on the fold whose stacker is
identical in both.  This computes it with exp-028's `panel.boot_delta`, the same
calendar-day-clustered bootstrap every other interval in this lane uses.

    python envdiff.py > ../analysis/envctl.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT))
import panel as P  # noqa: E402
import prc.data as D  # noqa: E402

FOLD = "july_loo"


def main() -> None:
    d = P.load_fold(FOLD)[["MVT_ID_mvt", "airport", "regime", "t"]]
    got = {}
    for tag in ("f1kag", "k2kag"):
        for arm in ("k2-nowin", "k2-hold"):
            got[(tag, arm)] = pd.read_parquet(
                EXP / "oof" / f"exp-027-{tag}-{arm}-{FOLD}.parquet")
    base = got[("f1kag", "k2-hold")]
    m = base.merge(d, on="MVT_ID_mvt", how="inner")
    if len(m) != len(base):
        raise SystemExit("the panel join lost rows")
    y = m["y_true"].to_numpy("float64")
    ids = m["MVT_ID_mvt"].to_numpy("float64")
    ok = ~pd.Series(ids).isin(D.excluded_ids(FOLD)).to_numpy()
    nn = ok & (m["regime"].to_numpy() != "noac")
    cl = P.clusters(m)
    print(f"k2kag minus f1kag on {FOLD}: both member columns fitted on Kaggle, one")
    print("with the transient-window block and one without; December and January")
    print("keep the committed workstation member in both runs, so the stacker for")
    print("this fold is the same model on both sides and the difference is the")
    print("apply-side member column alone.\n")
    for arm in ("k2-nowin", "k2-hold"):
        for col, label in (("y_pred_nomask", "no EHAM mask"), ("y_pred", "v7 EHAM mask")):
            a = got[("f1kag", arm)][col].to_numpy("float64")
            b = got[("k2kag", arm)][col].to_numpy("float64")
            dl, lo, hi = P.boot_delta(y[ok], b[ok], a[ok], pd.factorize(cl[ok])[0])
            dn, lon, hin = P.boot_delta(y[nn], b[nn], a[nn], pd.factorize(cl[nn])[0])
            print(f"  {arm:9s} {label:13s} {-dl:+7.4f} [{-hi:+7.4f}, {-lo:+7.4f}]"
                  f"   no-noac {-dn:+7.4f} [{-hin:+7.4f}, {-lon:+7.4f}]")


if __name__ == "__main__":
    main()
