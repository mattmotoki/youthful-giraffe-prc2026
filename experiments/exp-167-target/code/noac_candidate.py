"""M8: the C2 candidate vector for 2026 and its false-correction cost.

    .venv/bin/python experiments/exp-167-target/code/noac_candidate.py --tag 001

The typed-rate mixture of M3 fitted on the twelve 2025 months (classifier of the placeholder flag
and the real-row regression), served on the 2026 LIRF departures without aircraft type:

    candidate = max(60, 0.5 x v47 + 0.5 x mixture)   where ku <= 0  (same UTC date; M3c's rule)
    candidate = v47                                   elsewhere

2026 inputs are label-free: MVT, SCHED, stand, runway, flight string, the typed departures'
MVT - SCHED and AOBT_3 - SCHED, and the arrivals' BLOCK (an input in the ranking file). The
typed-row placeholder rates come from 2025 labels only.

Checks before the vector is written:
  1. the 2026 feature builder, run on the 2025 rows, equals M3's `build()` on every model input;
  2. the forecast formula (two-point error model, truth = the model's p), run on the 2025 fold
     with M3c's held predictions, against M3c's actual G.

Outputs (write-once): `results/noac-candidate-<tag>.json`,
`candidates/c2-noac-half-<tag>.parquet` (the 2026 rows without type: inputs, p, both points, v47,
candidate), and the full entry `data/exp167/c2-noac-half-<tag>-entry.parquet` (v47 with the moved
rows replaced; same schema as v47; not in git, sha256 in the JSON).
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as C
import regime_signal as S

V47 = C.ROOT / "submissions/uploaded/youthful-giraffe_v47.parquet"
V47_SHA = "3d58e932"                      # first 8 hex of the uploaded v47 (docs/LEDGER.md)
JJ = C.SAVED / "regime-jj-001.parquet"
COLS = S.SETS["typed"]
RC = ["ms", "ms_log", "hour", "sgrp_code", "q_pre_late", "noac_day_n"]
W = 0.5
SHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)


def frame_2026() -> pd.DataFrame:
    t = pd.read_parquet(C.RAW / "ranking.parquet")
    t["apt"] = np.where(t.PHASE_mvt == "DEP", t.ADEP_mvt, t.ADES_mvt)
    return t[t.apt == "LIRF"].copy()


def features(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M3's label-free inputs for the LIRF rows without type of any frame (2025 or 2026)."""
    arr = t[t.PHASE_mvt == "ARR"].copy()
    d = t[t.PHASE_mvt == "DEP"].copy()
    d[C.ID] = d[C.ID].astype("int64")
    d["ms"] = C.secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["noac"] = d.AIRCRAFT_TYPE_mvt.isna()
    d["ku"] = ((d.MVT_TIME_UTC_mvt - pd.Timedelta(minutes=15)).dt.normalize() - d.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days
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
    ty, x = d[~d.noac].copy(), d[d.noac].copy()
    x["ms_log"] = np.sign(x.ms) * np.log1p(x.ms.abs())
    x["hour"] = x.MVT_TIME_UTC_mvt.dt.hour
    x["dow"] = x.MVT_TIME_UTC_mvt.dt.weekday
    x["minute_of_day"] = x.SCHED_TIME_UTC_mvt.dt.hour * 60 + x.SCHED_TIME_UTC_mvt.dt.minute
    x["sched_min5"] = (x.SCHED_TIME_UTC_mvt.dt.minute % 5 == 0).astype(int)
    x["sgrp_code"] = x.stand.str[:1].map({c: i for i, c in enumerate("123456789HNP")}).fillna(-1)
    x["rwy_code"] = x.RUNWAY_mvt.fillna("NA").map({c: i for i, c in enumerate(["25", "16R", "16L", "34L", "34R", "NA"])}).fillna(-1)
    x["noac_day_n"] = x.day.map(x.groupby("day").size())
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


def forecast(delta, w_old, w_new, pi, weight) -> float:
    """Expected G of moving the placeholder weight from w_old to w_new when the truth is pi."""
    return float((weight * delta ** 2 * (pi * ((1 - w_old) ** 2 - (1 - w_new) ** 2) + (1 - pi) * (w_old ** 2 - w_new ** 2))).sum())


def scenarios(delta, w_old, w_new, p, weight) -> dict:
    out = {f"pi=p{s:+.1f}": forecast(delta, w_old, w_new, np.clip(p + s, 0, 1), weight) for s in SHIFTS}
    out["pi=served_weight"] = forecast(delta, w_old, w_new, np.clip(w_old, 0, 1), weight)
    out["pi=0 (all real)"] = forecast(delta, w_old, w_new, np.zeros(len(p)), weight)
    out["pi=1 (all placeholder)"] = forecast(delta, w_old, w_new, np.ones(len(p)), weight)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json = C.RESULTS / f"noac-candidate-{a.tag}.json"
    out_rows = C.EXP / f"candidates/c2-noac-half-{a.tag}.parquet"
    out_entry = C.SAVED / f"c2-noac-half-{a.tag}-entry.parquet"
    C.refuse_existing(out_json, out_rows, out_entry)
    if not C.sha(V47).startswith(V47_SHA):
        raise SystemExit("the v47 file is not the uploaded v47")
    rec: dict = {"what": "exp-167 M8: C2 candidate (0.5 v47 + 0.5 typed-rate mixture on same-date LIRF rows without type)",
                 "v47_sha256": C.sha(V47), "weight": W}

    # check 1: the builder equals M3's on 2025
    x25, ty25 = S.build()
    t25 = C.raw_2025()
    xb, _ = features(t25[t25.apt == "LIRF"])
    if len(xb) != len(x25) or not (xb[C.ID].to_numpy() == x25[C.ID].to_numpy()).all():
        raise SystemExit("builder parity: the 2025 rows differ")
    ctx = [c for c in COLS if not c.startswith(("q_", "n_pre", "n_flight"))] + ["pre", "FLIGHT_mvt", "stand", "ades"]
    bad = [c for c in ctx if not ((xb[c] == x25[c]) | (xb[c].isna() & x25[c].isna())).all()]
    if bad:
        raise SystemExit(f"builder parity fails on {bad}")
    rec["builder_parity_2025"] = {"rows": int(len(xb)), "columns_equal": ctx}
    print("builder parity on 2025: pass,", len(ctx), "columns", flush=True)

    # check 2: the forecast formula on the 2025 fold (M3c's held predictions)
    h = pd.read_parquet(JJ)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    h = h.join(r[["v32", "w26"]], on=C.ID)
    k0 = (h.ku <= 0).to_numpy()
    dl = (h.ms - h.m_real).to_numpy()[k0]
    wv = np.divide(h.v32.to_numpy()[k0] - h.m_real.to_numpy()[k0], dl, out=np.zeros(k0.sum()), where=np.abs(dl) > 1)
    wn = (1 - W) * wv + W * h.p.to_numpy()[k0]
    cand = np.maximum(C.FLOOR, (1 - W) * h.v32 + W * h.mix).to_numpy()[k0]
    act = float((h.w26.to_numpy()[k0] * ((h.y.to_numpy()[k0] - h.v32.to_numpy()[k0]) ** 2 - (h.y.to_numpy()[k0] - cand) ** 2)).sum())
    rec["fold_forecast_check"] = {"actual_G": act, "scenarios": scenarios(dl, wv, wn, h.p.to_numpy()[k0], h.w26.to_numpy()[k0]),
                                  "mean_p": float(h.p[k0].mean()), "placeholder_share": float(h.ph[k0].mean()),
                                  "mean_served_weight": float(np.clip(wv, 0, 1).mean())}
    print("2025 fold: actual", round(act, 1), "| forecast", {k: round(v) for k, v in rec["fold_forecast_check"]["scenarios"].items()}, flush=True)

    # the twelve-month fit and the 2026 rows
    x26, _ = features(frame_2026())
    parts = [pd.concat([x25[x25.month == m], S.encodings(x25[x25.month == m], ty25, m)], axis=1) for m in range(1, 13)]
    tr = pd.concat(parts)
    te = pd.concat([x26, S.encodings(x26, ty25, 0)], axis=1)
    kw = lambda sd, **o: dict(S.PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, **o)
    p = np.mean([lgb.LGBMClassifier(**kw(sd)).fit(tr[COLS], tr.ph.astype(int)).predict_proba(te[COLS])[:, 1] for sd in S.SEEDS], axis=0)
    real = ~tr.ph.to_numpy()
    y0 = tr.y - C.DAY * tr.ku.clip(lower=0)
    m_ = np.mean([lgb.LGBMRegressor(**kw(sd, n_estimators=150)).fit(tr.loc[real, RC], y0[real]).predict(te[RC]) for sd in S.SEEDS], axis=0)
    te["p"] = p
    te["m_real"] = np.clip(m_, C.FLOOR, None) + C.DAY * te.ku.clip(lower=0).to_numpy()
    te["mix"] = np.maximum(C.FLOOR, te.p * te.ms + (1 - te.p) * te.m_real)
    v47 = pd.read_parquet(V47)
    v = pd.Series(v47.TAXITIME_SEC_mvt.to_numpy("float64"), index=v47.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    te["v47"] = v.reindex(te[C.ID].to_numpy()).to_numpy()
    if te.v47.isna().any():
        raise SystemExit("a 2026 row without type is not in v47")
    move = (te.ku <= 0).to_numpy()
    te["candidate"] = np.where(move, np.maximum(C.FLOOR, (1 - W) * te.v47 + W * te.mix), te.v47)
    te["moved"] = move
    dl = (te.ms - te.m_real).to_numpy()[move]
    wv = np.divide(te.v47.to_numpy()[move] - te.m_real.to_numpy()[move], dl, out=np.zeros(move.sum()), where=np.abs(dl) > 1)
    wn = (1 - W) * wv + W * te.p.to_numpy()[move]
    wt = np.full(move.sum(), 1.0 / C.N_RANKING)
    rec["rows_2026"] = {"no_type_lirf": int(len(te)), "moved": int(move.sum()),
                        "by_month": {int(m): {"rows": int(len(g)), "moved": int(g.moved.sum()), "mean_p": float(g.p[g.moved].mean()),
                                              "mean_served_weight": float(np.clip(np.divide(g.v47 - g.m_real, g.ms - g.m_real), 0, 1)[g.moved].mean()),
                                              "mean_abs_change_s": float((g.candidate - g.v47).abs()[g.moved].mean()),
                                              "mean_change_s": float((g.candidate - g.v47)[g.moved].mean())}
                                     for m, g in te.groupby("month")}}
    rec["forecast_2026"] = scenarios(dl, wv, wn, te.p.to_numpy()[move], wt)
    for m in (1, 7):
        mm = (te.month.to_numpy() == m)[move]
        rec[f"forecast_2026_month_{m}"] = scenarios(dl[mm], wv[mm], wn[mm], te.p.to_numpy()[move][mm], wt[mm])
    print("2026 rows:", rec["rows_2026"], flush=True)
    print("2026 forecast (s2 per ranking row):", {k: round(v) for k, v in rec["forecast_2026"].items()}, flush=True)
    keep = [C.ID, "month", "ms", "ku", "p", "m_real", "mix", "v47", "candidate", "moved"]
    C.refuse_existing(out_json, out_rows, out_entry)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    te[keep].to_parquet(out_rows, index=False)
    new = v.copy()
    new.loc[te[C.ID].to_numpy()] = te.candidate.to_numpy()
    entry = pd.DataFrame({"MVT_ID_mvt": v47.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new.reindex(v47.MVT_ID_mvt.to_numpy("float64").astype("int64")).to_numpy()})
    if len(entry) != C.N_RANKING or not np.isfinite(entry.TAXITIME_SEC_mvt).all() or (entry.TAXITIME_SEC_mvt < C.FLOOR).any():
        raise SystemExit("the entry is not a full finite vector above the floor")
    changed = int((entry.TAXITIME_SEC_mvt.to_numpy() != v47.TAXITIME_SEC_mvt.to_numpy()).sum())
    entry.to_parquet(out_entry, index=False)
    rec["entry"] = {"file": str(out_entry.relative_to(C.ROOT)), "sha256": C.sha(out_entry), "rows": int(len(entry)), "rows_changed_from_v47": changed,
                    "max_abs_change_s": float(np.abs(entry.TAXITIME_SEC_mvt.to_numpy() - v47.TAXITIME_SEC_mvt.to_numpy()).max())}
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    rec["code_sha256"] = {"noac_candidate.py": C.sha(__file__), "regime_signal.py": C.sha(S.__file__), "common.py": C.sha(C.__file__)}
    C.write_json(out_json, rec)
    print("entry", rec["entry"], flush=True)


if __name__ == "__main__":
    main()
