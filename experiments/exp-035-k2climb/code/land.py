"""Average the seed files, price every arm at the member, write the harness's `oof/`.

The exp-027 candidate harness reads `oof/<name>-{fold}.parquet` through its
`PRC_K2` template, in the `COMMON.md` fold contract (`MVT_ID_mvt`, `fold`,
`y_true`, `y_pred`). This file builds those files for the climbed member and
prints the member-level A/B that the gates are read from.

Two things it refuses to do silently, because either would read as a clean
result:

  * average files whose `MVT_ID_mvt` sets or orders differ -- every seed file is
    joined on the id, never on position, and a row count that moves is fatal;
  * price a fold against a baseline from another machine. Only july_loo has a
    Kaggle F1-fresh baseline (293.9248) and a Kaggle K2 baseline (293.3992);
    december and january are priced against the committed K2 fold files, which
    are Kaggle files too, so those A/Bs are like for like *against K2* even
    though neither has a flag-off Kaggle baseline of its own.

    python land.py                     # price everything that is present
    python land.py --write <name>      # also write oof/<name>-{fold}.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
K2 = str(ROOT / "experiments/exp-027-step/oof/k2-window-{fold}.parquet")
F1 = str(ROOT / "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet")
#: The only like-for-like flag-off baseline that exists on this machine.
KAGGLE_F1 = {"july_loo": 293.9248}

#: arm name -> the seed files that make its member, in order. The K2 file is
#: seed `None`; `k2s1-*` is seed 20 and `k2s2-*` is seed 33.
ARMS = {
    #: the identity arm: the K2 file alone, so every gain must print 0.0000 and
    #: `K2` must print 293.3992 on july_loo. It is the test that this file's
    #: joining, averaging and scoring add nothing of their own.
    "k2": ["<k2>"],
    "seeds": ["<k2>", "k2s1-{fold}", "k2s2-{fold}"],
    "wide": ["k2w-{fold}"],
    #: arm 3: one fit, no averaging, so the "mean" is the file itself.
    "phi": ["k2p-{fold}"],
    "mod": ["k2m-{fold}"],
    "wide-seeds": ["k2w-{fold}", "k2ws1-{fold}", "k2ws2-{fold}"],
}


def _read(name: str, fold: str) -> pd.DataFrame | None:
    p = K2.format(fold=fold) if name == "<k2>" else EXP / "oof" / f"{name.format(fold=fold)}.parquet"
    p = Path(p)
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    need = {"MVT_ID_mvt", "y_true", "y_pred"}
    if not need <= set(d.columns):
        raise SystemExit(f"{p}: columns {sorted(d.columns)} lack {sorted(need)}")
    return d[["MVT_ID_mvt", "y_true", "y_pred"]]


def member(arm: str, fold: str) -> pd.DataFrame | None:
    """The arm's member prediction on one fold: the mean over its seed files."""
    parts = [(n, _read(n, fold)) for n in ARMS[arm]]
    have = [(n, d) for n, d in parts if d is not None]
    if len(have) != len(parts):
        return None
    base = have[0][1].rename(columns={"y_pred": "p0"})
    for i, (n, d) in enumerate(have[1:], start=1):
        d = d.rename(columns={"y_pred": f"p{i}", "y_true": f"y{i}"})
        base = base.merge(d, on="MVT_ID_mvt", how="inner")
        if len(base) != len(have[0][1]):
            raise SystemExit(f"{arm}/{fold}: {n} covers {len(base)} of {len(have[0][1])} rows")
        if not np.allclose(base["y_true"], base[f"y{i}"], equal_nan=True):
            raise SystemExit(f"{arm}/{fold}: {n} disagrees with the first file's y_true")
        base = base.drop(columns=[f"y{i}"])
    cols = [c for c in base.columns if c.startswith("p")]
    out = base[["MVT_ID_mvt", "y_true"]].copy()
    out["fold"] = fold
    out["y_pred"] = base[cols].to_numpy("float64").mean(axis=1)
    out["n_seeds"] = len(cols)
    return out


def rmse(y, p, lo=None):
    y = np.asarray(y, "float64"); p = np.asarray(p, "float64")
    m = np.ones(len(y), bool) if lo is None else (y <= 3600 if lo else y > 3600)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def price(arm: str) -> dict:
    rows = {}
    print(f"\n=== {arm}")
    print(f"{'fold':10s} {'n':>7s} {'seeds':>5s}  {'K2':>10s} {'arm':>10s} {'gain':>8s}  "
          f"{'K2 tail':>9s} {'arm tail':>9s} {'tail':>8s}  {'cleanK2':>9s} {'clean':>9s} {'gain':>8s}")
    for f in FOLDS:
        m = member(arm, f)
        if m is None:
            print(f"{f:10s} pending")
            continue
        k = _read("<k2>", f)
        j = k.rename(columns={"y_pred": "k2"}).merge(
            m[["MVT_ID_mvt", "y_pred", "n_seeds"]], on="MVT_ID_mvt", how="inner")
        if len(j) != len(k):
            raise SystemExit(f"{arm}/{f}: arm covers {len(j)} of {len(k)} K2 rows")
        y = j["y_true"].to_numpy("float64")
        a, b = j["k2"].to_numpy("float64"), j["y_pred"].to_numpy("float64")
        ca = D.rmse_clean(y, a, j["MVT_ID_mvt"], f)
        cb = D.rmse_clean(y, b, j["MVT_ID_mvt"], f)
        r = dict(n=len(j), n_seeds=int(j["n_seeds"].iloc[0]),
                 k2=rmse(y, a), arm=rmse(y, b),
                 k2_lo=rmse(y, a, True), arm_lo=rmse(y, b, True),
                 k2_hi=rmse(y, a, False), arm_hi=rmse(y, b, False),
                 k2_clean=ca, arm_clean=cb)
        r["gain"] = r["k2"] - r["arm"]
        r["gain_lo"] = r["k2_lo"] - r["arm_lo"]
        r["gain_hi"] = r["k2_hi"] - r["arm_hi"]
        r["gain_clean"] = ca - cb
        if f in KAGGLE_F1:
            r["f1_kaggle"] = KAGGLE_F1[f]
            r["gain_vs_f1"] = KAGGLE_F1[f] - r["arm"]
        rows[f] = r
        print(f"{f:10s} {r['n']:7d} {r['n_seeds']:5d}  {r['k2']:10.4f} {r['arm']:10.4f} "
              f"{r['gain']:+8.4f}  {r['k2_hi']:9.2f} {r['arm_hi']:9.2f} {r['gain_hi']:+8.2f}  "
              f"{ca:9.4f} {cb:9.4f} {r['gain_clean']:+8.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    out = {}
    for arm in a.arms:
        out[arm] = price(arm)
    (EXP / "analysis").mkdir(exist_ok=True)
    (EXP / "analysis" / "member.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {(EXP / 'analysis' / 'member.json').relative_to(ROOT)}")
    if a.write:
        arm = a.write
        for f in FOLDS:
            m = member(arm, f)
            if m is None:
                print(f"[write] {arm}/{f}: pending")
                continue
            p = EXP / "oof" / f"{arm}-{f}.parquet"
            m[["MVT_ID_mvt", "fold", "y_true", "y_pred"]].to_parquet(p, index=False)
            print(f"[write] {p.relative_to(ROOT)}  {len(m)} rows, {int(m['n_seeds'].iloc[0])} seeds")


if __name__ == "__main__":
    main()
