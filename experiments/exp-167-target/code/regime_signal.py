"""M3: a placeholder classifier for the LIRF rows without aircraft type, with typed-row encodings.

    .venv/bin/python experiments/exp-167-target/code/regime_signal.py --tag 001

Leave-one-month-out over the twelve 2025 months (the month of a row never fits its own model, and
its encodings come from the other eleven months). Three input sets:

  base    the row's own inputs (MVT - SCHED, UTC date difference, hour, weekday, stand group,
          runway, flight-string shape, counts of rows without type that day)
  typed   base + placeholder rates of the typed LIRF departures (flight, prefix, stand, hour),
          all rows and late rows (|AOBT_3 - SCHED| > 600 s), and the typed lateness around the row
  prior   one number only: logit of the prefix's typed late-row placeholder rate (a Bayes prior)

Mixture: p x (MVT - SCHED) + (1 - p) x m, m = a regression on the real rows of the fit months of
y - 86,400 x (UTC date difference). Read on the January and July rows against v32 (the served
form on these rows): G with the 2026 cell weights, and the same without the largest rows.
Write-once.
"""
from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import common as C

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

SEEDS = (11, 23, 37, 51, 77)
PARAMS = dict(learning_rate=0.04, num_leaves=8, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, n_estimators=300, verbose=-1, n_jobs=4)
BASE = ["ms", "ms_log", "ku", "hour", "dow", "minute_of_day", "flen", "dup", "noflight", "sgrp_code", "rwy_code",
        "noac_day_n", "sched_min5"]
TYPED = ["q_pre_late", "q_pre_all", "n_pre", "q_flight_late", "q_flight_all", "n_flight", "q_stand_late", "q_stand_all",
         "q_ades_late", "nb_late1h_60m", "nb_med_60m", "day_late1h", "arr_ph_day"]
SETS = {"base": BASE, "typed": BASE + TYPED, "prior": ["logit_q_pre_late", "ms_log", "ku"]}


def shrunk(sum_, n, prior, k):
    return (sum_ + k * prior) / (n + k)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = C.raw_2025()
    t = t[t.apt == "LIRF"]
    arr = t[t.PHASE_mvt == "ARR"].copy()
    d = C.add_regime(t[t.PHASE_mvt == "DEP"])
    d[C.ID] = d[C.ID].astype("int64")
    d["month"] = d.MVT_TIME_UTC_mvt.dt.month
    d["day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    d["t"] = (d.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    f = d.FLIGHT_mvt.fillna("")
    d["pre"] = f.str[:3]
    d["flen"] = f.str.len()
    d["noflight"] = (f == "").astype(int)
    d["dup"] = [int(len(s) >= 6 and s[3:] == s[max(0, len(s) - 5):] and s[3:5] in (s[1:3], s[2:3] + s[3:4]) or
                    (len(s) >= 5 and (s[3:5] == s[1:3] or s[3] == s[2]))) for s in f]
    d["stand"] = d.STAND_mvt.fillna("NA")
    d["ades"] = d.ADES_mvt.fillna("NA")
    d["as_"] = C.secs(d.AOBT_3_flt, d.SCHED_TIME_UTC_mvt)
    d["late"] = d.as_.abs() > 600
    ty, x = d[~d.noac].copy(), d[d.noac].copy()

    x["ms_log"] = np.sign(x.ms) * np.log1p(x.ms.abs())
    x["hour"] = x.MVT_TIME_UTC_mvt.dt.hour
    x["dow"] = x.MVT_TIME_UTC_mvt.dt.weekday
    x["minute_of_day"] = x.SCHED_TIME_UTC_mvt.dt.hour * 60 + x.SCHED_TIME_UTC_mvt.dt.minute
    x["sched_min5"] = (x.SCHED_TIME_UTC_mvt.dt.minute % 5 == 0).astype(int)
    x["sgrp_code"] = x.stand.str[:1].map({c: i for i, c in enumerate("123456789HNP")}).fillna(-1)
    x["rwy_code"] = x.RUNWAY_mvt.fillna("NA").map({c: i for i, c in enumerate(["25", "16R", "16L", "34L", "34R", "NA"])}).fillna(-1)
    x["noac_day_n"] = x.day.map(x.groupby("day").size())
    # label-free day and neighbourhood lateness of typed departures, and the arrivals' placeholder share
    x["day_late1h"] = x.day.map(ty.groupby("day").as_.apply(lambda s: (s > 3600).mean())).astype("float64")
    arr["day"] = arr.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    arr["ph"] = C.secs(arr.BLOCK_TIME_UTC_mvt, arr.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL
    x["arr_ph_day"] = x.day.map(arr.groupby("day").ph.mean()).astype("float64")
    tys = ty.sort_values("t")
    tt, tms = tys.t.to_numpy(), tys.ms.to_numpy()
    lo, hi = np.searchsorted(tt, x.t.to_numpy() - 3600), np.searchsorted(tt, x.t.to_numpy() + 3600)
    x["nb_late1h_60m"] = [np.mean(tms[a:b] > 3600) if b > a else np.nan for a, b in zip(lo, hi)]
    x["nb_med_60m"] = [np.median(tms[a:b]) if b > a else np.nan for a, b in zip(lo, hi)]
    return x.reset_index(drop=True), ty


def encodings(x: pd.DataFrame, ty: pd.DataFrame, held: int) -> pd.DataFrame:
    """Typed placeholder rates from the months other than `held`, for every row of x."""
    src = ty[ty.month != held]
    late = src[src.late]
    g_all, g_late = src.ph.mean(), late.ph.mean()
    out = pd.DataFrame(index=x.index)
    for key, name, k in (("pre", "pre", 20.0), ("FLIGHT_mvt", "flight", 10.0), ("stand", "stand", 30.0), ("ades", "ades", 30.0)):
        a, b = src.groupby(key).ph.agg(["sum", "size"]), late.groupby(key).ph.agg(["sum", "size"])
        na = x[key].map(a["size"]).fillna(0.0)
        out[f"q_{name}_all"] = shrunk(x[key].map(a["sum"]).fillna(0.0), na, g_all, k)
        out[f"q_{name}_late"] = shrunk(x[key].map(b["sum"]).fillna(0.0), x[key].map(b["size"]).fillna(0.0), g_late, k)
        out[f"n_{name}"] = na
    out["logit_q_pre_late"] = np.log(out.q_pre_late / (1 - out.q_pre_late))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json, out_pred = C.RESULTS / f"regime-signal-{a.tag}.json", C.SAVED / f"regime-signal-{a.tag}.parquet"
    C.refuse_existing(out_json, out_pred)
    x, ty = build()
    print(f"LIRF rows without type {len(x):,}; placeholder {x.ph.mean():.3f}; typed rows {len(ty):,}", flush=True)
    x["y0"] = x.y - C.DAY * x.ku.clip(lower=0)              # the real label without its date rule
    P = {s: np.full(len(x), np.nan) for s in SETS}
    M = np.full(len(x), np.nan)
    for m in range(1, 13):
        e = encodings(x, ty, m)
        z = pd.concat([x, e], axis=1)
        tr, te = (z.month != m).to_numpy(), (z.month == m).to_numpy()
        for s, cols in SETS.items():
            if s == "prior":
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, max_iter=1000).fit(z.loc[tr, cols], z.ph[tr])
                P[s][te] = lr.predict_proba(z.loc[te, cols])[:, 1]
            else:
                P[s][te] = np.mean([lgb.LGBMClassifier(**dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd))
                                    .fit(z.loc[tr, cols], z.ph[tr].astype(int)).predict_proba(z.loc[te, cols])[:, 1]
                                    for sd in SEEDS], axis=0)
        real = tr & ~z.ph.to_numpy()
        rc = ["ms", "ms_log", "hour", "sgrp_code", "q_pre_late", "noac_day_n"]
        M[te] = np.mean([lgb.LGBMRegressor(**dict(PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, n_estimators=150))
                         .fit(z.loc[real, rc], z.y0[real]).predict(z.loc[te, rc]) for sd in SEEDS], axis=0)
        print("month", m, "done", flush=True)
    x["m_real"] = np.clip(M, C.FLOOR, None) + C.DAY * x.ku.clip(lower=0)
    rec: dict = {"what": "exp-167 M3: placeholder classifier on LIRF rows without type, leave-one-month-out",
                 "rows": int(len(x)), "sets": SETS, "params": PARAMS, "seeds": SEEDS, "classifier": {}}
    jj = x.month.isin((1, 7)).to_numpy()
    for s in SETS:
        x[f"p_{s}"] = P[s]
        pc = np.clip(P[s], 1e-4, 1 - 1e-4)
        rec["classifier"][s] = {"auc_all": float(roc_auc_score(x.ph, pc)), "brier_all": float(brier_score_loss(x.ph, pc)),
                                "logloss_all": float(log_loss(x.ph, pc)), "auc_janjul": float(roc_auc_score(x.ph[jj], pc[jj])),
                                "brier_janjul": float(brier_score_loss(x.ph[jj], pc[jj])),
                                "auc_by_month": {int(m): float(roc_auc_score(x.ph[x.month == m], pc[(x.month == m).to_numpy()])) for m in range(1, 13)}}
        x[f"mix_{s}"] = np.maximum(C.FLOOR, pc * x.ms + (1 - pc) * x.m_real)
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec["classifier"][s].items() if k != "auc_by_month"})
    rec["base_brier"] = float(x.ph.mean() * (1 - x.ph.mean()))
    x["oracle"] = np.maximum(C.FLOOR, np.where(x.ph, x.ms, x.m_real))

    # the fold read against v32 (the served form on these rows)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    h = x[jj].join(r[["v32", "F", "cls", "w26"]], on=C.ID, how="inner")
    rec["fold_rows"] = int(len(h))
    full_ids = r.index.to_numpy("int64")
    y_all = GS.labels(full_ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    pos = r.index.get_indexer(h[C.ID])
    rec["mse_per_noac_row"] = {c: float(((h.y - h[c]) ** 2).mean()) for c in ["v32", "oracle"] + [f"mix_{s}" for s in SETS]}
    rec["G_vs_F"] = {}
    for c in ["oracle"] + [f"mix_{s}" for s in SETS] + ["half_typed"]:
        cand = base.copy()
        val = h[c].to_numpy("float64") if c != "half_typed" else 0.5 * h.v32.to_numpy("float64") + 0.5 * h.mix_typed.to_numpy("float64")
        cand[pos] = np.maximum(C.FLOOR, val)
        sc = GS.score(full_ids, base, cand, y=y_all)
        d = (y_all[pos] - base[pos]) ** 2 - (y_all[pos] - cand[pos]) ** 2
        w = h.w26.to_numpy()
        order = np.argsort(-np.abs(d * w))
        rec["G_vs_F"][c] = {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                            "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                            "G_without_top5_rows": float((d * w).sum() - (d * w)[order[:5]].sum()),
                            "G_without_top20_rows": float((d * w).sum() - (d * w)[order[:20]].sum()),
                            "top5_rows": [{"id": int(h[C.ID].iloc[i]), "y": float(h.y.iloc[i]), "ms": float(h.ms.iloc[i]),
                                           "base": float(base[pos][i]), "cand": float(cand[pos][i]), "d_w": float((d * w)[i])} for i in order[:5]]}
        g = rec["G_vs_F"][c]
        print(f"G({c} against v32 on the fold noac rows) = {g['G']:+.1f} [{g['lo95']:+.1f}, {g['hi95']:+.1f}]  "
              f"without top 5 rows {g['G_without_top5_rows']:+.1f}, top 20 {g['G_without_top20_rows']:+.1f}, months {g['months']}", flush=True)
    C.refuse_existing(out_json, out_pred)
    keep = [C.ID, "month", "y", "ph", "regime", "ms", "ku", "m_real", "oracle"] + [f"p_{s}" for s in SETS] + [f"mix_{s}" for s in SETS]
    x[keep].to_parquet(out_pred, index=False)
    rec["predictions"] = {"file": str(out_pred.relative_to(C.ROOT)), "sha256": C.sha(out_pred)}
    rec["code_sha256"] = {"regime_signal.py": C.sha(__file__), "common.py": C.sha(C.__file__)}
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
