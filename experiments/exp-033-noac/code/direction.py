"""Step 3: fold-measure a direction on the matched sub-population.

The head is the simplest one the matched rows support: a per-band two-class
placeholder mixture.  For a band it estimates, on the FIT months' noac rows,

    p_hat      P(|mvt_sched - y| < 300 s), shrunk toward the pooled rate
    o_ph       mean (mvt_sched - y) over the placeholder rows of the band
    o_np       mean (mvt_sched - y) over the non-placeholder rows of the band

and predicts the RMSE-optimal expectation `mvt_sched - (p*o_ph + (1-p)*o_np)`,
never an argmax (`prc.tail_model`'s own rule).  Shrinkage is m = 5 pseudo-rows
toward the pooled value, because the small bands hold under 20 fit rows.

Two fit sets, the same head:

    all        every noac row of the fit months
    matched    only the fit-month rows that are a k-nearest neighbour, WITHIN
               BAND, of one of the 388 (k = 5, the row-level space, fixed after
               step 2; the k and space sensitivity is reported)

Two instruments:

    LOMO       leave one 2025 month out, all twelve; 1,507 rows of power.
               Scored on the held-out month's MATCHED rows, which is the
               sub-population the 388 belong to, and on all its noac rows.
    folds      july_loo / december / january against the SHIPPED base, which is
               stack-5's own out-of-fold prediction on those rows
               (`experiments/exp-026-rebuild/oof/stack5-*.parquet`).

Intervals are calendar-day-clustered paired bootstraps on the difference of
RMSE; the noac rows are one airport's feed and are day-concentrated.
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
from describe import LAB  # noqa: E402

M_SHRINK = 5.0
K = 5
SPACE = "rowlevel"
NBOOT = 4000
SHIPPED_FOLDS = ("july_loo", "december", "january")


def matched_mask(a: pd.DataFrame, b: pd.DataFrame, k: int = K, space: str = SPACE) -> np.ndarray:
    """2025 rows that are a within-band k-nearest neighbour of one of the 388."""
    d_ba = np.load(EXP / "work" / f"d_ba_{space}.npy")
    out = np.zeros(len(a), bool)
    for lab in LAB:
        q = np.flatnonzero((b.band == lab).to_numpy())
        p = np.flatnonzero((a.band == lab).to_numpy())
        if not len(q) or not len(p):
            continue
        nb = np.argsort(d_ba[np.ix_(q, p)], axis=1)[:, :min(k, len(p))]
        out[p[np.unique(nb.ravel())]] = True
    return out


def fit_head(fit: pd.DataFrame) -> dict:
    """Per-band (p, o_ph, o_np), shrunk toward the pooled values."""
    o = (fit.mvt_sched - fit.y_true).to_numpy(float)
    ph = fit.ph.to_numpy(float)
    p_pool = float(ph.mean())
    oph_pool = float(o[ph == 1].mean()) if (ph == 1).any() else 0.0
    onp_pool = float(o[ph == 0].mean()) if (ph == 0).any() else 0.0
    band = fit.band.to_numpy()
    out = {"_pool": (p_pool, oph_pool, onp_pool)}
    for lab in LAB:
        m = band == lab
        n = int(m.sum())
        if not n:
            out[lab] = (p_pool, oph_pool, onp_pool)
            continue
        p = (ph[m].sum() + M_SHRINK * p_pool) / (n + M_SHRINK)
        a1, a0 = m & (ph == 1), m & (ph == 0)
        oph = (o[a1].sum() + M_SHRINK * oph_pool) / (a1.sum() + M_SHRINK)
        onp = (o[a0].sum() + M_SHRINK * onp_pool) / (a0.sum() + M_SHRINK)
        out[lab] = (float(p), float(oph), float(onp))
    return out


def apply_head(h: dict, d: pd.DataFrame) -> np.ndarray:
    band = d.band.to_numpy()
    ms = d.mvt_sched.to_numpy(float)
    off = np.empty(len(d))
    for i, lab in enumerate(band):
        p, oph, onp = h.get(lab, h["_pool"])
        off[i] = p * oph + (1 - p) * onp
    return ms - off


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def paired_rmse_diff(y, base, arm, day, nboot=NBOOT, seed=23):
    """95 % interval for RMSE(base) - RMSE(arm); positive means the arm is better."""
    y, base, arm, day = map(np.asarray, (y, base, arm, day))
    days, inv = np.unique(day, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(days))]
    rng = np.random.default_rng(seed)
    out = np.empty(nboot)
    for i in range(nboot):
        sel = np.concatenate([idx[j] for j in rng.integers(0, len(days), len(days))])
        out[i] = rmse(y[sel], base[sel]) - rmse(y[sel], arm[sel])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def lomo(a: pd.DataFrame, mm: np.ndarray) -> pd.DataFrame:
    rows = []
    a = a.assign(_m=mm)
    for m in D.MONTHS:
        va = a[a.month == m]
        fit = a[a.month != m]
        h_all = fit_head(fit)
        h_mat = fit_head(fit[fit._m])
        for scope, sel in (("matched", va[va._m]), ("all", va)):
            if len(sel) < 5:
                continue
            rows.append({"month": m, "scope": scope, "n": len(sel),
                         "rmse_all": rmse(sel.y_true, apply_head(h_all, sel)),
                         "rmse_matched": rmse(sel.y_true, apply_head(h_mat, sel))})
    t = pd.DataFrame(rows)
    return t


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    mm = matched_mask(a, b)
    print(f"matched 2025 rows (within band, k={K}, {SPACE}): {mm.sum()} of {len(a)}")

    # --- instrument 1: LOMO over the twelve months --------------------------
    t = lomo(a, mm)
    t.to_csv(EXP / "analysis" / "lomo_by_month.csv", index=False)
    pooled = []
    a = a.assign(_m=mm)
    for scope in ("matched", "all"):
        yy, p_all, p_mat, dd = [], [], [], []
        for m in D.MONTHS:
            va = a[a.month == m]
            va = va[va._m] if scope == "matched" else va
            fit = a[a.month != m]
            h_all, h_mat = fit_head(fit), fit_head(fit[fit._m])
            yy.append(va.y_true.to_numpy(float))
            p_all.append(apply_head(h_all, va))
            p_mat.append(apply_head(h_mat, va))
            dd.append(va.day.to_numpy())
        yy, p_all, p_mat, dd = (np.concatenate(x) for x in (yy, p_all, p_mat, dd))
        lo, hi = paired_rmse_diff(yy, p_all, p_mat, dd)
        pooled.append({"instrument": "LOMO-12", "scope": scope, "n": len(yy),
                       "rmse_fit_all": rmse(yy, p_all), "rmse_fit_matched": rmse(yy, p_mat),
                       "gain": rmse(yy, p_all) - rmse(yy, p_mat), "lo": lo, "hi": hi})
    print("\n=== instrument 1: LOMO over twelve months, head fitted all vs matched ===")
    print(pd.DataFrame(pooled).round(2).to_string(index=False))
    pd.DataFrame(pooled).to_csv(EXP / "analysis" / "lomo_pooled.csv", index=False)
    print("\nper month (scope=matched):")
    print(t[t.scope == "matched"].assign(gain=lambda d: d.rmse_all - d.rmse_matched)
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
