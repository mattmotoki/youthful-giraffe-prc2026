"""Every number of record in RESULTS.md, from one script with one log.

Opus round 1 finding B4: several figures had no committed producer. Everything
quoted in the report is printed here, tagged `[Rn]` with the finding it belongs
to, and `work/measure.log` is the artifact.

Also answers B1 by measuring the head the delivery path actually uses (the
seed-averaged mixture) in the same harness as the alternatives, and B2 by
testing the arrival-side observable as a RATE.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent))
from threads import THREADS   # caps BLAS at import time; docs/COMPUTE.md compute allocation
import sys
import numpy as np, pandas as pd, lightgbm as lgb
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-009-noac2"))
from lomo import BASE, NEW, CATS_NEW, P, SEEDS, PH_TOL, enc

ART = D.ROOT / "artifacts" / "exp-009-noac2"
EXP = D.ROOT / "experiments" / "exp-009-noac2"
NR, NN, DAY = 344841, 388, 86400.0
FEATS, CATS = BASE + NEW, CATS_NEW
AIR = ("LTFM", "EHAM", "LFPG", "EDDF", "EGLL", "LEMD", "LEBL", "EDDM", "LIRF", "LSZH")
S = lambda a, b: (a - b).dt.total_seconds()


def heads(Xtr, otr, Xte, ms):
    """The three forms, all 5-seed: offset, mixture (the delivered path), hard switch."""
    mk = lambda s, cls=False: (lgb.LGBMClassifier if cls else lgb.LGBMRegressor)(
        **dict(P, seed=s, bagging_seed=s, feature_fraction_seed=s, data_random_seed=s,
               **({"objective": "binary"} if cls else {})))
    off = np.mean([mk(s).fit(Xtr, otr, categorical_feature=CATS).predict(Xte) for s in SEEDS], axis=0)
    ph = (np.abs(otr) < PH_TOL).astype(int)
    p = np.mean([mk(s, True).fit(Xtr, ph, categorical_feature=CATS).predict_proba(Xte)[:, 1] for s in SEEDS], axis=0)
    onp = np.mean([mk(s).fit(Xtr[ph == 0], otr[ph == 0], categorical_feature=CATS).predict(Xte) for s in SEEDS], axis=0)
    return {"offset (5-seed)": ms - off,
            "mixture (5-seed, DELIVERED path)": ms - (1 - p) * onp,
            "hard switch": np.where(p >= 0.5, ms, ms - onp)}, p


def lomo_all(df, extra=()):
    feats = list(FEATS) + list(extra)
    acc, ys, mons, ps = {}, [], [], []
    for m in D.MONTHS:
        te, tr = df[df.month == m], df[df.month != m]
        if not len(te):
            continue
        lev = {c: pd.Index(sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))) for c in CATS}
        Xtr, Xte = enc(tr, feats, CATS, lev), enc(te, feats, CATS, lev)
        otr = tr.mvt_sched.to_numpy(float) - tr.y.to_numpy(float)
        h, p = heads(Xtr, otr, Xte, te.mvt_sched.to_numpy(float))
        for k, v in h.items():
            acc.setdefault(k, []).append(v)
        ys.append(te.y.to_numpy(float)); mons.append(np.repeat(m, len(te))); ps.append(p)
    y, mon, p = np.concatenate(ys), np.concatenate(mons), np.concatenate(ps)
    rows = []
    for k, v in acc.items():
        se = (y - np.concatenate(v)) ** 2
        rows.append(dict(head=k, rmse=np.sqrt(se.mean()),
                         contrib=np.sqrt(NN * se.mean() / NR),
                         worst_month=max(np.sqrt(se[mon == mm].mean()) for mm in np.unique(mon))))
    return pd.DataFrame(rows), y, p, mon


if __name__ == "__main__":
    df = pd.read_parquet(ART / "noac_train.parquet")
    o = df.mvt_sched.to_numpy(float) - df.y.to_numpy(float)
    ph = np.abs(o) < PH_TOL
    print(f"[R0] {len(df)} noac rows, P(placeholder) = {ph.mean():.4f}\n")

    print("[R6/B1] the three head forms, all 5-seed, same harness:")
    res, y, p_hat, mon = lomo_all(df)
    print(res.round(1).to_string(index=False))
    print("\n[R2r2] the same three forms WITH the desk's arrival-day placeholder rate:")
    res2, _, _, _ = lomo_all(df, extra=("arr_day_ph_rate",))
    print(res2.round(1).to_string(index=False))
    res2.to_csv(EXP / "work" / "heads_with_arr_rate.csv", index=False)
    res.to_csv(EXP / "work" / "heads.csv", index=False)
    lab = (np.abs(df.mvt_sched.to_numpy(float) - df.y.to_numpy(float)) < PH_TOL).astype(int)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    order = np.argsort([list(D.MONTHS).index(m) for m in mon], kind="stable")
    print(f"\n[R6] calibrated p (SPEC promised this and round 1 noted it was never measured):"
          f" AUC={roc_auc_score(lab[order], p_hat[order]):.3f}"
          f"  Brier={brier_score_loss(lab[order], p_hat[order]):.3f}  base={lab.mean():.3f}")

    print("\n[R1/B2] the arrival-side observable as a RATE, not a count:")
    ev = pd.read_parquet(EXP / "work" / "arr_evidence.parquet")
    ev["arr_ph_rate"] = ev.arr_ph_n / ev.arr_n
    q = pd.qcut(ev.arr_ph_rate, 5, duplicates="drop")
    t = ev.groupby(q, observed=True).agg(n=("ph", "size"), p_ph=("ph", "mean"))
    print(t.round(3).to_string())
    print(f"  corr(day arrival placeholder rate, departure placeholder flag) = "
          f"{ev.arr_ph_rate.corr(ev.ph.astype(float)):.4f}")
    L = ev[ev.apt == "LIRF"]
    print(f"  within LIRF alone = {L.arr_ph_rate.corr(L.ph.astype(float)):.4f}")
    j = df.merge(ev[["arr_ph_rate"]].set_index(ev.index), left_index=True, right_index=True, how="left") \
        if False else df.copy()
    j["arr_ph_rate"] = ev["arr_ph_rate"].to_numpy()      # same row order, both built from D.MONTHS
    for c in ("noac_day_n", "noac_day_share", "noac_day_count"):
        print(f"  corr(arrival day rate, {c}) = {j.arr_ph_rate.corr(j[c]):.4f}")

    print("\n[R2/B3] decomposition of the oracle-p residual, by taxi band:")
    npx = df[~ph]
    # SSE against the oracle-p predictor's own conditional mean, stated explicitly
    base = npx.y.mean()
    se = (npx.y - base) ** 2
    for lo, hi in [(0, 1200), (1200, 3600), (3600, 15000), (15000, 60000), (60000, 1e9)]:
        k = (npx.y >= lo) & (npx.y < hi)
        print(f"  y [{lo},{hi}): n={int(k.sum()):4d}  SSE share {se[k].sum()/se.sum():.3f}")
    roll = npx.y > 60000
    print(f"  rows with y > 60,000 (a day-early block): {int(roll.sum())}")
    print(f"  rows in 15,000-60,000 (long taxis, NOT rollovers): {int(((npx.y>15000)&(npx.y<=60000)).sum())}")

    print("\n[R1] base rates of |block - schedule| <= 60 s over the ten airports, 12 months:")
    a_n = a_h = d_n = d_h = 0
    for m in D.MONTHS:
        f = D.load_month(m, columns=["PHASE_mvt", "ADEP_mvt", "ADES_mvt",
                                     "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
        f["apt"] = np.where(f.PHASE_mvt == "DEP", f.ADEP_mvt, f.ADES_mvt)
        f = f[f.apt.isin(AIR)]
        b = S(f.BLOCK_TIME_UTC_mvt, f.SCHED_TIME_UTC_mvt).abs() <= 60
        isa = f.PHASE_mvt == "ARR"
        a_n += int(isa.sum()); a_h += int(b[isa].sum())
        d_n += int((~isa).sum()); d_h += int(b[~isa].sum())
    print(f"  arrivals   {a_h}/{a_n} = {a_h/a_n:.4f}")
    print(f"  departures {d_h}/{d_n} = {d_h/d_n:.4f}")
