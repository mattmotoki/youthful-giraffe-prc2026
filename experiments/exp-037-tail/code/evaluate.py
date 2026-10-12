"""Step 4: what the chain adds, measured out of fold.

Two instruments, both on 2025 noac rows only.

**LOMO** (twelve months, 1,507 rows): each arm's own head against the base
arm's own head.  This is what the arm adds to the head, at the highest power
the data has.  It says nothing about the shipped vector.

**Folds** (july_loo / december / january, the shipped instrument): the arm is

    arm = shipped_oof + (y_arm - y_base)     on rows with mvt_sched < BAND_MIN

against the shipped out-of-fold prediction itself (stack-5's, which is what v9
serves on the noac rows).  The restriction was fixed before the fold numbers
were read -- by file time, not by a commit, since this module and its first
output entered the repository in one commit -- and it has a reason: above
`BAND_MIN` the tail MEMBER inside stack-5 is the exp-009 two-point band table
and not the learned mixture, so a difference of two learned mixtures is not a
correction to it there; and exp-033 §8 prices that band's whole move at 1.33
times the movement budget in any case.  It is one row too wide at the top:
above `BAND_TOP` (26 h) `_band_predict` returns NaN and the learned mixture
keeps the row, which is 1 of the 1,507, 1 of the 388 and 0 of the 491 fold
rows, so the restriction costs nothing measurable and errs towards not moving.

Intervals are calendar-day-clustered paired bootstraps on the difference of
RMSE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from prc.tail_model import BAND_MIN  # noqa: E402
from build import LAB  # noqa: E402

OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof"
FOLDS = ("july_loo", "december", "january")
NBOOT = 4000
RNG = np.random.default_rng(20260908)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def day_boot_gain(d: pd.DataFrame, base: str, arm: str, nboot: int = NBOOT) -> dict:
    """Day-clustered paired bootstrap of RMSE(base) - RMSE(arm)."""
    uniq, inv = np.unique(d.day.to_numpy(), return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    y = d.y_true.to_numpy(float)
    b, a = d[base].to_numpy(float), d[arm].to_numpy(float)
    pt = rmse(y, b) - rmse(y, a)
    out = np.empty(nboot)
    for k in range(nboot):
        rows = np.concatenate([idx[i] for i in RNG.integers(0, len(uniq), len(uniq))])
        out[k] = rmse(y[rows], b[rows]) - rmse(y[rows], a[rows])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"gain": pt, "lo": float(lo), "hi": float(hi), "p_worse": float((out < 0).mean())}


def auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


ARM_NAMES = ("chain", "rot")


def lomo_table(v: pd.DataFrame, arms=ARM_NAMES) -> pd.DataFrame:
    rows = []
    for arm in [a for a in arms if f"y_{a}" in v.columns]:
        for scope, d0 in (("all", v), ("below_band", v[v.mvt_sched < BAND_MIN])):
            for lab in ["ALL"] + LAB:
                d = d0 if lab == "ALL" else d0[d0.band == lab]
                if len(d) < 10:
                    continue
                r = {"arm": arm, "scope": scope, "band": lab, "n": len(d),
                     "auc_base": auc(d.ph, d.p_base), "auc_arm": auc(d.ph, d[f"p_{arm}"]),
                     "rmse_base": rmse(d.y_true, d.y_base),
                     "rmse_arm": rmse(d.y_true, d[f"y_{arm}"])}
                r.update(day_boot_gain(d, "y_base", f"y_{arm}"))
                rows.append(r)
    return pd.DataFrame(rows)


def load_shipped() -> pd.DataFrame:
    out = []
    for f in FOLDS:
        d = pd.read_parquet(OOF / f"stack5-{f}.parquet")
        d = d[["MVT_ID_mvt", "y_true", "y_pred"]].copy()
        d["fold"] = f
        out.append(d)
    return pd.concat(out, ignore_index=True)


def fold_table(v: pd.DataFrame, arms=ARM_NAMES) -> tuple:
    ship = load_shipped().rename(columns={"y_pred": "shipped", "y_true": "y_ship"})
    m = v.merge(ship, on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")
    assert m.shipped.notna().all(), "a fold noac row has no shipped out-of-fold prediction"
    assert np.allclose(m.y_true, m.y_ship), "label mismatch against the shipped out-of-fold file"
    live = m.mvt_sched.to_numpy(float) < BAND_MIN
    rows = []
    for arm in [a for a in arms if f"y_{a}" in m.columns]:
        m[f"corr_{arm}"] = np.where(live, m[f"y_{arm}"] - m.y_base, 0.0)
        m[f"arm_{arm}"] = m.shipped + m[f"corr_{arm}"]
        for fold in list(FOLDS) + ["POOLED"]:
            d = m if fold == "POOLED" else m[m.fold == fold]
            for lab in ["ALL"] + LAB:
                s = d if lab == "ALL" else d[d.band == lab]
                if len(s) < 5:
                    continue
                r = {"arm": arm, "fold": fold, "band": lab, "n": len(s),
                     "n_moved": int((s[f"corr_{arm}"] != 0).sum()),
                     "shift_rms": float(np.sqrt(np.mean(s[f"corr_{arm}"] ** 2))),
                     "rmse_shipped": rmse(s.y_true, s.shipped),
                     "rmse_arm": rmse(s.y_true, s[f"arm_{arm}"])}
                r.update(day_boot_gain(s, "shipped", f"arm_{arm}"))
                rows.append(r)
    return pd.DataFrame(rows), m


def main() -> None:
    v = pd.read_parquet(EXP / "work" / "lomo.parquet")
    t = lomo_table(v)
    t.to_csv(EXP / "analysis" / "lomo.csv", index=False)
    with pd.option_context("display.width", 220):
        print("LOMO, twelve 2025 months, each arm's head against the base head")
        print(t.round(3).to_string(index=False))

    f = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    ft, m = fold_table(f)
    ft.to_csv(EXP / "analysis" / "fold_gain.csv", index=False)
    m.to_parquet(EXP / "work" / "fold_eval.parquet")
    with pd.option_context("display.width", 220):
        print()
        print("Folds, the arm on top of the shipped out-of-fold vector")
        print(ft.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
