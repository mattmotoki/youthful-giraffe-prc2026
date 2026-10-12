"""Three p-heads for the noac two-mode mixture, LOMO over twelve months.

  L  log-loss     the delivered design: a binary classifier for p, then
                  pred = ms - (1-p)*off_np
  R  stake-weighted  fit p by the loss we are SCORED on. With taxi_hat = ms -
                  off_np and d = ms - taxi_hat = off_np,
                      (p*ms + (1-p)*taxi_hat - y)^2 == d^2 * (p - r)^2,
                      r = (y - taxi_hat)/d,
                  so it is a weighted least-squares fit of r with weight d^2:
                  r = 1 is a placeholder, r = 0 an ordinary taxi. The weight is
                  the stake, so the head cannot bet big on weak evidence where
                  ms is large -- which is exactly the delivered head's failure.
  C  calibrated   L, then isotonic on p within ms deciles (and per operator
                  group), because the overconfidence runs along ms.

`taxi_hat` and the calibrator are CROSS-FITTED inside the training months; using
in-sample values would bias r toward 0 on the ordinary rows that the offset
regressor fits well, which is the whole population under test.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import prc.data as D
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "exp-009-noac2"))
from threads import THREADS

NR, NN = 344841, 388
PH_TOL, DAY, MIN_D = 300.0, 86400.0, 60.0
SEEDS = (11, 23, 37)
FEATS = ["mvt_sched", "flight_len", "hour", "dow", "minute_of_day",
         "noac_day_count", "nonm_day_count", "n_arr_sched_mvt", "gap_arr_after_sched",
         "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
         "gap_dep_prev_mvt", "gap_dep_next_mvt", "vfr", "noac_day_n",
         "noac_day_share", "dep_day_n", "arr_within_1h_of_sched",
         "arr_within_3h_of_sched", "flight_n_digits", "flight_n_alpha",
         "flight_repeated_prefix", "mvt_sched_log", "stand_busy_frac_6h"]
CATS = ["apt", "runway", "stand", "flight_prefix"]
BASE = dict(learning_rate=0.04, num_leaves=8, n_estimators=300, min_child_samples=30,
            feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
            n_jobs=THREADS, verbose=-1, deterministic=True, force_row_wise=True)


def enc(d, levels):
    X = d[FEATS].copy()
    for c in CATS:
        X[c] = pd.Categorical(d[c].astype(str), categories=levels[c])
    return X


def seeded(**kw):
    return [dict(BASE, seed=s, bagging_seed=s, feature_fraction_seed=s,
                 data_random_seed=s, **kw) for s in SEEDS]


def fit_offset(tr, levels, ph):
    """The ordinary-taxi offset, fitted on NON-placeholder rows only."""
    m = ~ph
    if m.sum() < 20:
        return None
    X, o = enc(tr[m], levels), (tr.mvt_sched[m] - tr.y[m]).to_numpy(float)
    return [lgb.LGBMRegressor(**p).fit(X, o, categorical_feature=CATS) for p in seeded()]


def pred_offset(models, d, levels):
    return np.mean([m.predict(enc(d, levels)) for m in models], axis=0)


def run(df):
    out = []
    for mo in sorted(df.month.unique()):
        te, tr = df[df.month == mo].copy(), df[df.month != mo].copy()
        levels = {c: pd.Index(sorted(set(df[c].astype(str)))) for c in CATS}
        ms_tr, y_tr = tr.mvt_sched.to_numpy(float), tr.y.to_numpy(float)
        ms_te, y_te = te.mvt_sched.to_numpy(float), te.y.to_numpy(float)
        ph_tr = pd.Series(np.abs(y_tr - ms_tr) <= PH_TOL, index=tr.index)
        roll_tr = np.abs(y_tr - DAY) < 3600

        off_models = fit_offset(tr, levels, ph_tr.to_numpy())
        off_te = pred_offset(off_models, te, levels)
        taxi_te = ms_te - off_te
        d_te = np.maximum(off_te, MIN_D)

        # ---- L: log-loss classifier ------------------------------------
        clf = [lgb.LGBMClassifier(**dict(p, objective="binary")).fit(
                   enc(tr, levels), ph_tr.to_numpy().astype(int),
                   categorical_feature=CATS) for p in seeded()]
        p_L = np.mean([c.predict_proba(enc(te, levels))[:, 1] for c in clf], axis=0)

        # ---- cross-fitted taxi_hat and p on the TRAINING months ---------
        oof_off = np.full(len(tr), np.nan)
        oof_p = np.full(len(tr), np.nan)
        for a, b in KFold(5, shuffle=True, random_state=7).split(tr):
            ia, ib = tr.index[a], tr.index[b]
            om = fit_offset(tr.loc[ia], levels, ph_tr.loc[ia].to_numpy())
            if om is None:
                continue
            oof_off[b] = pred_offset(om, tr.loc[ib], levels)
            # seed-averaged like p_L, or the calibrator is fitted on a
            # different estimator from the one it is applied to
            cs = [lgb.LGBMClassifier(**dict(q, objective="binary")).fit(
                      enc(tr.loc[ia], levels), ph_tr.loc[ia].to_numpy().astype(int),
                      categorical_feature=CATS) for q in seeded()]
            oof_p[b] = np.mean([c.predict_proba(enc(tr.loc[ib], levels))[:, 1]
                                for c in cs], axis=0)

        # ---- R: stake-weighted ratio ------------------------------------
        d_tr = np.maximum(oof_off, MIN_D)
        taxi_tr = ms_tr - oof_off
        r_tr = (y_tr - taxi_tr) / d_tr
        ok = np.isfinite(r_tr) & ~roll_tr & (oof_off > MIN_D)
        reg = [lgb.LGBMRegressor(**p).fit(
                   enc(tr[ok], levels), np.clip(r_tr[ok], -1.0, 2.0),
                   sample_weight=(d_tr[ok] ** 2) / np.mean(d_tr[ok] ** 2),
                   categorical_feature=CATS) for p in seeded()]
        p_R = np.clip(np.mean([m.predict(enc(te, levels)) for m in reg], axis=0), 0.0, 1.0)

        # ---- C: isotonic on p within ms deciles -------------------------
        p_C = p_L.copy()
        good = np.isfinite(oof_p)
        if good.sum() > 100:
            qs = np.unique(np.quantile(ms_tr[good], np.linspace(0, 1, 11)))
            qs[0], qs[-1] = -np.inf, np.inf
            for lo, hi in zip(qs[:-1], qs[1:]):
                mtr = good & (ms_tr >= lo) & (ms_tr < hi)
                mte = (ms_te >= lo) & (ms_te < hi)
                if mtr.sum() >= 40 and mte.any() and ph_tr.to_numpy()[mtr].std() > 0:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                    iso.fit(oof_p[mtr], ph_tr.to_numpy()[mtr].astype(float))
                    p_C[mte] = iso.predict(p_L[mte])

        for name, p in (("L log-loss", p_L), ("R stake-weighted", p_R), ("C calibrated", p_C)):
            pred = taxi_te + p * d_te
            out.append(pd.DataFrame(dict(month=mo, head=name, y=y_te, pred=pred,
                                         ms=ms_te, p=p, op=te.op_prefix.to_numpy(),
                                         truth=(np.abs(y_te - ms_te) <= PH_TOL))))
        print(f"  {mo} done (n={len(te)})", flush=True)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_parquet(D.ROOT / "artifacts/exp-009-noac2/noac_train.parquet")
    df = df[df.y.notna()].reset_index(drop=True)
    res = run(df)
    res.to_parquet(D.ROOT / "experiments/exp-023-noac-band/work/heads_lomo.parquet", index=False)
    print("\n=== LOMO over 1,507 noac rows ===")
    for name, g in res.groupby("head"):
        e = g.y - g.pred
        band = (g.ms >= 3600) & (g.ms < 8 * 3600)
        print(f"{name:18s} noac RMSE {np.sqrt((e**2).mean()):8.1f}   "
              f"1-8 h band {np.sqrt((e[band]**2).mean()):8.1f} (n={int(band.sum())})   "
              f"ranking contribution {np.sqrt(NN*(e**2).mean()/NR):6.2f} s")
