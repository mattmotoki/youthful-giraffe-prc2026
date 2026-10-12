"""M8-r1: the C2 candidate without the flight-level rates (the correction of `noac_candidate.py`).

    .venv/bin/python experiments/exp-167-target/code/noac_candidate_r1.py --tag r1

Why: candidate 001 used the typed-row placeholder rate of the row's own flight string. In July
2026 only 31 % of the flight strings of the rows without type exist among the 2025 typed rows
(92 % inside 2025), so `n_flight = 0` moved from 8 % to 69 % of the rows and lifted the model's
placeholder chance from 0.30 (July 2025) to 0.55 (July 2026). The flight-level inputs do not
transfer between the years. This script drops them (`q_flight_late`, `q_flight_all`, `n_flight`).

Part A, the two-fold protocol (fit on the ten other 2025 months, read on January and July),
weights 1, 0.5 and 0.25 against v32 on the same-date rows.
Part B, the 2026 vector at weight 0.25 (the one weight that is positive in both months and
without the top rows and days in part A; the weight was read on the fold rows):

    candidate = max(60, 0.75 x v47 + 0.25 x mixture)   where ku <= 0,   v47 elsewhere

with label-free shift diagnostics of the inputs. Write-once.
"""
from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import common as C
import noac_candidate as N
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

DROP = ("q_flight_late", "q_flight_all", "n_flight")
COLS = [c for c in S.SETS["typed"] if c not in DROP]
RC = N.RC
TEN = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
W = 0.25


def fit_score(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    kw = lambda sd, **o: dict(S.PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, **o)
    p = np.mean([lgb.LGBMClassifier(**kw(sd)).fit(tr[COLS], tr.ph.astype(int)).predict_proba(te[COLS])[:, 1] for sd in S.SEEDS], axis=0)
    real = ~tr.ph.to_numpy()
    y0 = tr.y - C.DAY * tr.ku.clip(lower=0)
    m_ = np.mean([lgb.LGBMRegressor(**kw(sd, n_estimators=150)).fit(tr.loc[real, RC], y0[real]).predict(te[RC]) for sd in S.SEEDS], axis=0)
    return p, np.clip(m_, C.FLOOR, None) + C.DAY * te.ku.clip(lower=0).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json = C.RESULTS / f"noac-candidate-{a.tag}.json"
    out_rows = C.EXP / f"candidates/c2-noac-q25-{a.tag}.parquet"
    out_entry = C.SAVED / f"c2-noac-q25-{a.tag}-entry.parquet"
    C.refuse_existing(out_json, out_rows, out_entry)
    if not C.sha(N.V47).startswith(N.V47_SHA):
        raise SystemExit("the v47 file is not the uploaded v47")
    x, ty = S.build()
    rec: dict = {"what": "exp-167 M8-r1: C2 candidate without flight-level rates", "columns": COLS, "dropped": DROP,
                 "weight": W, "v47_sha256": C.sha(N.V47)}

    # part A: two-fold protocol
    tsrc = ty[ty.month.isin(TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    p, m_real = fit_score(tr, te)
    mix = np.maximum(C.FLOOR, p * te.ms + (1 - p) * m_real)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    pos = r.index.get_indexer(te[C.ID])
    v32, w26, k0 = base[pos], r.w26.to_numpy()[pos], (te.ku <= 0).to_numpy()
    jm, mo = {}, te.month.to_numpy()
    for m in (1, 7):
        jm[str(m)] = {"rows": int((mo == m).sum()), "auc": float(roc_auc_score(te.ph[mo == m], p[mo == m])),
                      "brier": float(brier_score_loss(te.ph[mo == m], p[mo == m])), "mean_p": float(p[mo == m].mean()),
                      "P_share": float(te.ph[mo == m].mean()), "share_operator_unknown": float((te.n_pre[mo == m] == 0).mean()),
                      "share_no_flight_string": float(te.noflight[mo == m].mean())}
    rec["two_fold"] = {"classifier": jm, "auc": float(roc_auc_score(te.ph, p)), "G_vs_v32": {}}
    print("two-fold classifier:", jm, flush=True)
    for name, wt in (("w100", 1.0), ("w50", 0.5), ("w25", 0.25)):
        cand = base.copy()
        cand[pos] = np.maximum(C.FLOOR, np.where(k0, wt * mix + (1 - wt) * v32, v32))
        sc = GS.score(ids, base, cand, y=y)
        d = ((y[pos] - base[pos]) ** 2 - (y[pos] - cand[pos]) ** 2) * w26
        o = np.argsort(-np.abs(d))
        ph = te.ph.to_numpy()
        rec["two_fold"]["G_vs_v32"][name] = {
            "G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"],
            "months": sc["views"]["month"], "G_without_top5_rows": float(d.sum() - d[o[:5]].sum()),
            "G_without_top20_rows": float(d.sum() - d[o[:20]].sum()), "G_on_placeholder_rows": float(d[ph].sum()),
            "G_on_real_rows": float(d[~ph].sum()), "rows_better": int((d > 0).sum()), "rows_worse": int((d < 0).sum())}
        g = rec["two_fold"]["G_vs_v32"][name]
        print(f"{name}: G {g['G']:+.1f} [{g['lo95']:+.1f}, {g['hi95']:+.1f}] months {({k: round(v) for k, v in g['months'].items()})} without top days "
              f"{g['G_without_top_days']:+.0f}, top 5 rows {g['G_without_top5_rows']:+.0f}, top 20 {g['G_without_top20_rows']:+.0f}; "
              f"placeholder rows {g['G_on_placeholder_rows']:+.0f}, real rows {g['G_on_real_rows']:+.0f}", flush=True)

    # part B: the twelve-month fit and the 2026 rows
    x26, _ = N.features(N.frame_2026())
    tr12 = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], ty, m)], axis=1) for m in range(1, 13)])
    te26 = pd.concat([x26, S.encodings(x26, ty, 0)], axis=1)
    p26, m26 = fit_score(tr12, te26)
    te26["p"], te26["m_real"] = p26, m26
    te26["mix"] = np.maximum(C.FLOOR, te26.p * te26.ms + (1 - te26.p) * te26.m_real)
    v47 = pd.read_parquet(N.V47)
    v = pd.Series(v47.TAXITIME_SEC_mvt.to_numpy("float64"), index=v47.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    te26["v47"] = v.reindex(te26[C.ID].to_numpy()).to_numpy()
    if te26.v47.isna().any():
        raise SystemExit("a 2026 row without type is not in v47")
    move = (te26.ku <= 0).to_numpy()
    te26["candidate"] = np.where(move, np.maximum(C.FLOOR, (1 - W) * te26.v47 + W * te26.mix), te26.v47)
    te26["moved"] = move
    sw = np.clip((te26.v47 - te26.m_real) / (te26.ms - te26.m_real), 0, 1)
    rec["rows_2026"] = {"no_type_lirf": int(len(te26)), "moved": int(move.sum()), "by_month": {
        int(m): {"rows": int(len(g)), "moved": int(g.moved.sum()), "mean_p": float(g.p[g.moved].mean()),
                 "mean_served_weight": float(sw[g.index][g.moved].mean()), "mean_change_s": float((g.candidate - g.v47)[g.moved].mean()),
                 "mean_abs_change_s": float((g.candidate - g.v47).abs()[g.moved].mean()),
                 "share_operator_unknown": float((g.n_pre == 0).mean()), "share_no_flight_string": float(g.noflight.mean())}
        for m, g in te26.groupby("month")}}
    print("2026:", rec["rows_2026"], flush=True)
    keep = [C.ID, "month", "ms", "ku", "p", "m_real", "mix", "v47", "candidate", "moved"]
    C.refuse_existing(out_json, out_rows, out_entry)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    te26[keep].to_parquet(out_rows, index=False)
    new = v.copy()
    new.loc[te26[C.ID].to_numpy()] = te26.candidate.to_numpy()
    entry = pd.DataFrame({"MVT_ID_mvt": v47.MVT_ID_mvt.to_numpy(),
                          "TAXITIME_SEC_mvt": new.reindex(v47.MVT_ID_mvt.to_numpy("float64").astype("int64")).to_numpy()})
    if len(entry) != C.N_RANKING or not np.isfinite(entry.TAXITIME_SEC_mvt).all() or (entry.TAXITIME_SEC_mvt < C.FLOOR).any():
        raise SystemExit("the entry is not a full finite vector above the floor")
    entry.to_parquet(out_entry, index=False)
    diff = np.abs(entry.TAXITIME_SEC_mvt.to_numpy() - v47.TAXITIME_SEC_mvt.to_numpy())
    rec["entry"] = {"file": str(out_entry.relative_to(C.ROOT)), "sha256": C.sha(out_entry), "rows": int(len(entry)),
                    "rows_changed_from_v47": int((diff > 0).sum()), "max_abs_change_s": float(diff.max())}
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    rec["code_sha256"] = {"noac_candidate_r1.py": C.sha(__file__), "noac_candidate.py": C.sha(N.__file__),
                          "regime_signal.py": C.sha(S.__file__), "common.py": C.sha(C.__file__)}
    C.write_json(out_json, rec)
    print("entry", rec["entry"], flush=True)


if __name__ == "__main__":
    main()
