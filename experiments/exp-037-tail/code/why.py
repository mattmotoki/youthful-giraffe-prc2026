"""Why the arm loses on the shipped vector while the chain wins inside its own family.

Three measurements.

1. The self-fitted fold gain with its day-clustered interval, and LOMO's.
2. The correlation of the correction with the error it is supposed to remove:
   `corr(c, y - y_base)` against `corr(c, y - shipped)`.  If the chain is
   repairing the base head's own weakness rather than carrying information the
   shipped model lacks, the first is positive and the second is not.
3. The chain's MARGINAL information about the placeholder class beyond the base
   classifier, out of fold: a likelihood-ratio test of `logit(ph) ~ logit(p_base)`
   against the same plus the chain columns, with a day-clustered permutation null.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc.tail_model import BAND_MIN  # noqa: E402
from evaluate import day_boot_gain, rmse, FOLDS  # noqa: E402
from frames import CHAIN_FEATS, ROT_FEATS  # noqa: E402

RNG = np.random.default_rng(20260908)
NPERM = 2000
ARMS = {"chain": CHAIN_FEATS, "rot": CHAIN_FEATS + ROT_FEATS}
#: the marginal test is NESTED: each arm's own columns are added to a null that
#: already holds the base classifier's logit AND the arm before it.  Round 3 of
#: the review found the first version testing chain+rot against the logit alone
#: and calling all fifteen columns new.
NULL_COLS = {"chain": (), "rot": CHAIN_FEATS}
NEW_COLS = {"chain": CHAIN_FEATS, "rot": ROT_FEATS}


def ll(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_ll(X, y) -> tuple:
    m = LogisticRegression(max_iter=5000, C=1e6).fit(X, y)
    return ll(y, m.predict_proba(X)[:, 1]), m


def main() -> None:
    rows = []
    m = pd.read_parquet(EXP / "work" / "repair_eval.parquet")
    arms = [a for a in ARMS if f"y_{a}" in m.columns]
    v = pd.read_parquet(EXP / "work" / "lomo.parquet")
    for arm in arms:
        for fold in list(FOLDS) + ["POOLED"]:
            d = m if fold == "POOLED" else m[m.fold == fold]
            r = {"arm": arm, "instrument": "folds, self-fitted", "scope": fold, "n": len(d)}
            r.update(day_boot_gain(d, "y_base", f"y_{arm}"))
            rows.append(r)
        r = {"arm": arm, "instrument": "LOMO, self-fitted", "scope": "12 months", "n": len(v)}
        r.update(day_boot_gain(v, "y_base", f"y_{arm}"))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "why_selffitted.csv", index=False)
    print("each arm against the base inside the self-fitted family")
    print(t.round(2).to_string(index=False))

    # the durable address for §6's three numbers
    lev = []
    for fold in list(FOLDS) + ["POOLED"]:
        d = m if fold == "POOLED" else m[m.fold == fold]
        row = {"fold": fold, "n": len(d),
               "rmse_shipped": rmse(d.y_true, d.shipped),
               "rmse_my_base": rmse(d.y_true, d.y_base),
               "rms_base_minus_shipped": rmse(d.y_base, d.shipped),
               "corr_base_shipped": float(np.corrcoef(d.y_base, d.shipped)[0, 1])}
        for arm in arms:
            row[f"rmse_my_{arm}"] = rmse(d.y_true, d[f"y_{arm}"])
        lev.append(row)
    lv = pd.DataFrame(lev)
    lv.to_csv(EXP / "analysis" / "why_levels.csv", index=False)
    print("\nhow close the self-fitted base head is to the shipped vector")
    print(lv.round(2).to_string(index=False))

    print("\nwhat the correction is correlated with")
    live = m.mvt_sched.to_numpy(float) < BAND_MIN
    e_base = (m.y_true - m.y_base).to_numpy(float)
    e_ship = (m.y_true - m.shipped).to_numpy(float)
    out = []
    for arm in arms:
      c = np.where(live, (m[f"y_{arm}"] - m.y_base).to_numpy(float), 0.0)
      for fold in list(FOLDS) + ["POOLED"]:
        k = np.ones(len(m), bool) if fold == "POOLED" else (m.fold == fold).to_numpy()
        k &= live
        out.append({"arm": arm, "fold": fold, "n": int(k.sum()),
                    "corr_with_base_error": float(np.corrcoef(c[k], e_base[k])[0, 1]),
                    "corr_with_shipped_error": float(np.corrcoef(c[k], e_ship[k])[0, 1]),
                    "rms_correction": float(np.sqrt(np.mean(c[k] ** 2))),
                    "rms_base_error": float(np.sqrt(np.mean(e_base[k] ** 2))),
                    "rms_shipped_error": float(np.sqrt(np.mean(e_ship[k] ** 2))),
                    # the RMSE-optimal scale of the correction against each error
                    "opt_lam_base": float(np.dot(c[k], e_base[k]) / np.dot(c[k], c[k])),
                    "opt_lam_shipped": float(np.dot(c[k], e_ship[k]) / np.dot(c[k], c[k]))})
    o = pd.DataFrame(out)
    o.to_csv(EXP / "analysis" / "why_correlation.csv", index=False)
    print(o.round(3).to_string(index=False))

    print("\neach arm's marginal information about `ph` beyond the base classifier, out of fold")
    n25 = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    cols = sorted(set(sum([list(ARMS[a]) for a in arms], [])))
    v = v.merge(n25[["MVT_ID_mvt"] + cols], on="MVT_ID_mvt", validate="one_to_one")
    p = np.clip(v.p_base.to_numpy(float), 1e-6, 1 - 1e-6)
    v["lg"] = np.log(p / (1 - p))
    d = v.dropna(subset=["lg"]).copy()
    for c_ in cols:
        d[c_] = pd.to_numeric(d[c_], errors="coerce")
    d[cols] = d[cols].fillna(d[cols].median())
    y = d.ph.to_numpy(float)
    days = d.day.to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    marg = []
    idx_by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    for arm in arms:
        null_c, new_c = list(NULL_COLS[arm]), list(NEW_COLS[arm])
        X = np.column_stack([d[["lg"]].to_numpy(float),
                             d[null_c].to_numpy(float) if null_c else np.empty((len(d), 0)),
                             d[new_c].to_numpy(float)])
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        k0 = 1 + len(null_c)
        l0, _ = fit_ll(X[:, :k0], y)
        l1, _ = fit_ll(X, y)
        lr = 2 * (l1 - l0)
        # A calendar-day permutation of the NEW block.  Days hold different
        # numbers of rows, so concatenating the permuted day blocks and writing
        # them back by position does NOT keep a day whole -- round 3 of the
        # review found exactly that.  Instead every row of day i draws its new
        # block from the rows of day perm(i), cycling when the donor day is
        # shorter, so each row's new covariates come from one other calendar day.
        null = np.empty(NPERM)
        for b in range(NPERM):
            perm = RNG.permutation(len(uniq))
            order = np.empty(len(y), dtype="int64")
            for i, donor in enumerate(perm):
                rows_i, rows_s = idx_by_day[i], idx_by_day[donor]
                order[rows_i] = rows_s[np.arange(len(rows_i)) % len(rows_s)]
            Xp = X.copy()
            Xp[:, k0:] = X[order, k0:]
            lp, _ = fit_ll(Xp, y)
            null[b] = 2 * (lp - l0)
        pval = float((null >= lr).mean())
        print(f"  {arm:6s} n={len(d)}  LR={lr:.2f} for {len(new_c)} new columns "
              f"over a null of {k0} (the base logit"
              f"{' plus the chain block' if null_c else ''}), "
              f"day-permutation p={pval:.4f} (null mean {null.mean():.2f}, "
              f"95th {np.quantile(null, 0.95):.2f})", flush=True)
        marg.append({"arm": arm, "n": len(d), "LR": lr, "k_new": len(new_c),
                     "k_null": k0, "p_dayperm": pval,
                     "null_mean": float(null.mean()),
                     "null_q95": float(np.quantile(null, 0.95))})
    pd.DataFrame(marg).to_csv(EXP / "analysis" / "why_marginal.csv", index=False)


if __name__ == "__main__":
    main()
