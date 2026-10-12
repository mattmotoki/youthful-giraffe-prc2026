"""Paired bootstrap on the NM bulk, for reading one climb step against another.

Whole-fold RMSE is the competition metric, but it is a bad instrument for
deciding whether a mechanism works: about 4,000 of july's 190,713 rows carry
four fifths of the squared error, so a 5 s effect on the other 186,000 is
invisible under their noise. The congestion worker hit this directly — four of
its seven feature families appeared to flip sign between folds on whole-fold
RMSE, and all seven improve both folds once scored on the NM bulk.

So mechanisms are decided here and candidates are decided on whole-fold RMSE.
The comparison is paired: both models are scored on the same rows and the
bootstrap resamples rows, so the interval is on the *difference*, which is far
tighter than the interval on either RMSE.

Two populations are reported. `core` is defined from inputs alone — NM join
present, not a noac row — and is the honest one. `nm_bulk` adds `target <= 3600`
to match the congestion worker's tables; it conditions on the outcome, which is
safe for a paired comparison of two similar models but not for comparing a
shrunk predictor against an unshrunk one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D  # noqa: E402

N_BOOT = 400
RNG = np.random.default_rng(7)
CLUSTER = True  # resample airport-days, not rows


def _load(name: str, fold: str) -> pd.DataFrame | None:
    p = HERE / "oof" / f"{name}-{fold}.parquet"
    return pd.read_parquet(p).set_index("MVT_ID_mvt") if p.exists() else None


def compare(a: str, b: str, folds=("july", "december", "january")) -> pd.DataFrame:
    rows = []
    for fold in folds:
        A, B = _load(a, fold), _load(b, fold)
        if A is None or B is None:
            continue
        d = D.load_month(D.FOLDS[fold].valid_month,
                         columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
                                  "ADEP_mvt", "MVT_TIME_UTC_mvt"])
        d = d[d.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
        idx = A.index.intersection(B.index)
        # Errors are correlated within an airport-day -- one disrupted morning at
        # LIRF moves hundreds of rows together -- so resampling rows independently
        # understates the interval. Clusters are (airport, calendar day).
        day = d.MVT_TIME_UTC_mvt.dt.floor("D").astype("int64").reindex(idx).to_numpy()
        apt = d.ADEP_mvt.reindex(idx).astype("string").to_numpy()
        cl = pd.factorize(pd.Series(apt) + "|" + pd.Series(day).astype(str))[0]
        noac = d.AIRCRAFT_TYPE_mvt.isna().reindex(idx).to_numpy()
        nm = d.FLIGHT_ID_mvt.isna().reindex(idx).to_numpy()
        y = A.y_true.reindex(idx).to_numpy()
        ea = (y - A.y_pred.reindex(idx).to_numpy()) ** 2
        eb = (y - B.y_pred.reindex(idx).to_numpy()) ** 2
        for pop, mask in (("all", np.ones(len(idx), bool)),
                          ("core", ~noac & ~nm),
                          ("nm_bulk", ~noac & ~nm & (y <= 3600))):
            sa, sb = ea[mask], eb[mask]
            n = len(sa)
            if not n:
                continue
            obs = np.sqrt(sa.mean()) - np.sqrt(sb.mean())
            if CLUSTER:
                groups = pd.factorize(cl[mask])[0]
                order = np.argsort(groups, kind="stable")
                ga, gb = sa[order], sb[order]
                bounds = np.searchsorted(groups[order], np.arange(groups.max() + 2))
                sizes = np.diff(bounds)
                ca = np.add.reduceat(ga, bounds[:-1]) * (sizes > 0)
                cb = np.add.reduceat(gb, bounds[:-1]) * (sizes > 0)
                G = len(sizes)
                k = RNG.integers(0, G, size=(N_BOOT, G))
                nn = sizes[k].sum(axis=1)
                diffs = np.sqrt(ca[k].sum(axis=1) / nn) - np.sqrt(cb[k].sum(axis=1) / nn)
            else:
                k = RNG.integers(0, n, size=(N_BOOT, n))
                diffs = np.sqrt(sa[k].mean(axis=1)) - np.sqrt(sb[k].mean(axis=1))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append(dict(fold=fold, population=pop, n=n,
                             rmse_a=float(np.sqrt(sa.mean())), rmse_b=float(np.sqrt(sb.mean())),
                             delta=float(obs), lo=float(lo), hi=float(hi),
                             significant="yes" if lo * hi > 0 else "no"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    folds = sys.argv[3].split(",") if len(sys.argv) > 3 else ("july", "december", "january")
    t = compare(a, b, folds)
    pd.set_option("display.width", 220)
    print(f"delta = RMSE({a}) - RMSE({b}); negative means {a} is better\n")
    print(t.round(2).to_string(index=False))
