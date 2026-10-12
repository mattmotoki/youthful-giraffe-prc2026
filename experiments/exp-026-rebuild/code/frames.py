"""Fold frames for the blend lane, built from committed OOF alone.

`experiments/exp-016-stack/code/members.py` builds the same object, but its
`gates()` reads `artifacts/exp-008/frames/` and `artifacts/exp-014/frames/`,
which are git-ignored and did not survive the workstation.  Every quantity the
blend rule needs is nevertheless inside git: the C2-cat and L2-hub out-of-fold
files carry `noac`, `nm_missing`, `airport`, `d_takeoff_aobt` and
`d_aobt_lobt` beside their predictions, and those five columns are the whole
input to `prc.blend.regime_labels` and to the residual guard.

So this module is the same frame from a different address, and it says so by
checking itself: the regime it derives is compared row by row against the
`regime` column exp-016 froze into `oof/exp-016-oof-{fold}.parquet`, and the
blend rule fitted on these frames is compared against the rotated numbers
`blend-config-4.json` and `blend-config-5.json` published.

What is NOT here: the stacker's gate features (`local_hour`, `queue_ahead_aobt`,
the day-state block).  Those lived only in the lost frames.  The shipped
stacker arm (`a5`, members only) does not read them; the `a2` arm does and
cannot run on this machine.
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
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
#: exp-016 `members.SHARES`: the 2026 row counts of the two ranking months.
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
GUARD = (-20657.0, 9069.0)   # every fold in exp-008-diversity/guard_bounds.json

#: Where the five gate columns come from.  The pinned C2-cat OOF is inside both
#: blend-4's and blend-5's member sets and its bytes are pinned in two places
#: (`pins-members-blend4.json`, exp-012 `MANIFEST.md`), so it is the address
#: least likely to move under anyone.
GATE_SRC = "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet"
GATE_CHECK = "experiments/exp-012-blend2/published/L2-hub8725-{fold}.parquet"
#: exp-016's frozen OOF, used only to check the derived regime and to supply `t`.
X16 = "experiments/exp-016-stack/oof/exp-016-oof-{fold}.parquet"

#: Every member address this experiment can reach, out of fold.
OOF = {
    "T-tailv2":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "T-tailv4":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet",
    "C2-cat":    "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet",
    "L2-hub":    "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
    "L2-hub8725": "experiments/exp-012-blend2/published/L2-hub8725-{fold}.parquet",
    "D1-winter": "experiments/exp-012-blend2/published/D1-winter-deice-v1-{fold}.parquet",
    "F1-fresh":  "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet",
    "V3-allin":  "experiments/exp-012-blend2/published/V3-allin-v1-{fold}.parquet",
    "N1-nmtail": "experiments/exp-012-blend2/published/N1-nmtail-v4-{fold}.parquet",
}
#: Ranking-side addresses that survived in git.  `T-tailv2` and `L2-hub` are
#: absent on purpose: their ranking files lived only in `artifacts/` (hashes
#: `e9d97994…` and `29adfc27…` in `pins-members-blend4.json`) and are gone.
RANKING = {
    "T-tailv4":   "experiments/exp-012-blend2/published/T-tailv4-ranking.parquet",
    "C2-cat":     "experiments/exp-012-blend2/published/C2-cat-ranking.parquet",
    "L2-hub8725": "experiments/exp-012-blend2/published/L2-hub8725-ranking.parquet",
    "D1-winter":  "experiments/exp-012-blend2/published/D1-winter-deice-v1-ranking.parquet",
    "F1-fresh":   "experiments/exp-012-blend2/published/F1-fresh-s1-ranking.parquet",
    "V3-allin":   "experiments/exp-012-blend2/published/V3-allin-v1-ranking.parquet",
    "N1-nmtail":  "experiments/exp-012-blend2/published/N1-nmtail-v4-ranking.parquet",
}
#: A member that is null where it does not predict.  It may carry weight only
#: where it is defined; `prc.blend.build_candidate` enforces the same rule.
PARTIAL = ("N1-nmtail",)
#: blend-5 names the tail and Huber members after their SLOTS, not their
#: lineages, so `blend-config-5.json` says `T-tailv2` and `L2-hub` while the
#: files behind them are the v4 tail and the 8,725-round Huber.  Kept explicit
#: because reading the config without it silently builds blend-4's members.
SLOT = {"T-tailv2": "T-tailv4", "L2-hub": "L2-hub8725"}


def _read(path: str, fold: str, cols: list[str]) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / path.format(fold=fold))
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    d = d[cols].copy()
    d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
    return d


def gates(fold: str, check: bool = True) -> pd.DataFrame:
    """`MVT_ID_mvt`, the five regime/guard inputs, `airport`, `t`, and `regime`.

    `check=True` re-reads the same five columns from a second member's OOF and
    from exp-016's frozen frame, and raises on any disagreement.  The columns
    are written by whichever producer built each member, so agreement across
    two producers is the evidence that they mean the same thing.
    """
    cols = ["MVT_ID_mvt", "noac", "nm_missing", "d_aobt_lobt", "d_takeoff_aobt", "airport"]
    g = _read(GATE_SRC, fold, cols)
    g["regime"] = B.regime_labels(g["noac"], g["nm_missing"], g["d_aobt_lobt"], B.LATE_CUT)
    if check:
        h = _read(GATE_CHECK, fold, cols)
        m = g.merge(h, on="MVT_ID_mvt", suffixes=("", "_b"))
        if len(m) != len(g):
            raise ValueError(f"{fold}: gate cross-check covers {len(m)} of {len(g)}")
        for c in cols[1:]:
            a, b = m[c].to_numpy(), m[c + "_b"].to_numpy()
            bad = ~(pd.isna(a) & pd.isna(b)) & (a != b)
            if bad.any():
                raise ValueError(f"{fold}: gate column {c} disagrees on {int(bad.sum())} rows")
    x = _read(X16, fold, ["MVT_ID_mvt", "regime", "t"])
    m = g.merge(x, on="MVT_ID_mvt", how="left", suffixes=("", "_16"))
    seen = m["regime_16"].notna()
    bad = int((m.loc[seen, "regime"] != m.loc[seen, "regime_16"]).sum())
    if bad:
        raise ValueError(f"{fold}: derived regime differs from exp-016's on {bad} rows")
    g["t"] = m["t"].to_numpy()
    g.attrs["regime_checked_rows"] = int(seen.sum())
    return g


def fold_frame(fold: str, names: tuple[str, ...]) -> pd.DataFrame:
    """Prepared member predictions, the label, and the regime, for one fold.

    Members are prepared exactly as `prc.blend.build_candidate` prepares them --
    floor, then the residual guard -- so a member is the same estimator here and
    at serve time.  Both operations are idempotent.
    """
    base = None
    for n in names:
        d = _read(OOF[n], fold, ["MVT_ID_mvt", "y_true", "y_pred"]).rename(columns={"y_pred": n})
        how = "left" if n in PARTIAL else "inner"
        base = d if base is None else base.merge(d[["MVT_ID_mvt", n]], on="MVT_ID_mvt", how=how)
    g = gates(fold)
    m = base.merge(g, on="MVT_ID_mvt", how="inner")
    if len(m) != len(base):
        raise ValueError(f"{fold}: gates cover {len(m)} of {len(base)} member rows")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    for n in names:
        v = m[n].to_numpy("float64")
        ok = np.isfinite(v)
        v[ok] = np.maximum(v[ok], B.FLOOR)
        v[ok] = B.apply_residual_guard(v[ok], a[ok], *GUARD, B.FLOOR)
        m[n] = v
    m["fold"] = fold
    return m


def board_estimate(per_fold: dict) -> float:
    """The two ranking months in quadrature, weighted by their row counts."""
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


def sha(path: str) -> str:
    return D.sha256(ROOT / path)


if __name__ == "__main__":
    names = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin", "N1-nmtail")
    for f in FOLDS:
        m = fold_frame(f, names)
        cov = {n: float(np.isfinite(m[n]).mean()) for n in names}
        print(f"{f}: {m.shape}  regimes {dict(pd.Series(m['regime']).value_counts())}")
        print("   coverage " + " ".join(f"{n}={c:.3f}" for n, c in cov.items()))
