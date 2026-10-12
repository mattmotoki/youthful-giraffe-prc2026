"""M12 (C31): the late-inbound sign in the class model of the LIRF rows without a plan.

    .venv/bin/python experiments/exp-167-target/code/inbound_sign.py --tag 001

Source: `fresh` (exp-168 `code/lirf_signal_strata.py`).  By the join window a row without a plan
is real without a delay message, or a placeholder with one.  A late inbound aircraft is a sign of
a delay message.  The sign (label-free; arrival rows keep their in-block time in 2026):

    the last arrival at the departure's stand before its take-off, same airline prefix,
    turn (take-off - in-block) from 1,200 s to 24 h;
    in_late = 1 when the in-block is later than SCHED - 30 min, 0 when earlier, missing without a link.

Read: the two-fold protocol of M8-r1 (fit on the ten other 2025 months, read on January and
July) with r1's columns ("r1"), r1 + in_late ("sign"), and r1 + in_late + the in-block lead
in seconds ("sign+lead", my variation), at the weights 0.25 (the registered kill test of
`levers`), 0.5 and 1.  Also the coverage of the sign in 2025 and 2026.  Write-once.
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
import noac_candidate_r1 as R1
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

ARMS = {"r1": [], "sign": ["in_late"], "sign+lead": ["in_late", "in_lead"]}
WEIGHTS = (("w25", 0.25), ("w50", 0.5), ("w100", 1.0))


def inbound_sign(t: pd.DataFrame) -> pd.DataFrame:
    """`in_late` and `in_lead` for the LIRF departures of a frame (2025 or 2026)."""
    L = t[t.apt == "LIRF"]
    dep = L[(L.PHASE_mvt == "DEP") & L.STAND_mvt.notna()].sort_values("MVT_TIME_UTC_mvt")
    arr = L[(L.PHASE_mvt == "ARR") & L.STAND_mvt.notna() & L.BLOCK_TIME_UTC_mvt.notna()].sort_values("BLOCK_TIME_UTC_mvt")
    a = pd.DataFrame({"STAND_mvt": arr.STAND_mvt.to_numpy(), "a_blk": arr.BLOCK_TIME_UTC_mvt.to_numpy(),
                      "a_pre": arr.FLIGHT_mvt.fillna("").str[:3].to_numpy()})
    a["a_blk"] = pd.to_datetime(a.a_blk, utc=True).astype(dep.MVT_TIME_UTC_mvt.dtype)
    m = pd.merge_asof(dep[[C.ID, "STAND_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt"]], a.assign(key=a.a_blk),
                      left_on="MVT_TIME_UTC_mvt", right_on="key", by="STAND_mvt", direction="backward", allow_exact_matches=False)
    turn = C.secs(m.MVT_TIME_UTC_mvt, m.a_blk)
    linked = (m.FLIGHT_mvt.fillna("").str[:3] == m.a_pre) & turn.between(1200, 86400)
    lead = C.secs(m.a_blk, m.SCHED_TIME_UTC_mvt)
    out = pd.DataFrame({C.ID: m[C.ID].astype("int64"), "in_late": np.where(linked, (lead > -1800).astype("float64"), np.nan),
                        "in_lead": np.where(linked, lead, np.nan)})
    return out


def fit_score(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    kw = lambda sd, **o: dict(S.PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, **o)
    p = np.mean([lgb.LGBMClassifier(**kw(sd)).fit(tr[cols], tr.ph.astype(int)).predict_proba(te[cols])[:, 1] for sd in S.SEEDS], axis=0)
    real = ~tr.ph.to_numpy()
    y0 = tr.y - C.DAY * tr.ku.clip(lower=0)
    m_ = np.mean([lgb.LGBMRegressor(**kw(sd, n_estimators=150)).fit(tr.loc[real, R1.RC], y0[real]).predict(te[R1.RC]) for sd in S.SEEDS], axis=0)
    return p, np.clip(m_, C.FLOOR, None) + C.DAY * te.ku.clip(lower=0).to_numpy()


def coverage(x: pd.DataFrame) -> dict:
    return {int(m): {"rows": int(len(g)), "linked": float(g.in_late.notna().mean()), "late_of_linked": float(g.in_late.mean()),
                     "late_of_all": float((g.in_late == 1).mean())} for m, g in x.groupby("month")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json, out_rows = C.RESULTS / f"inbound-sign-{a.tag}.json", C.SAVED / f"inbound-sign-{a.tag}.parquet"
    C.refuse_existing(out_json, out_rows)
    x, ty = S.build()
    sg = inbound_sign(C.raw_2025()).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()
    rec: dict = {"what": "exp-167 M12: C31, the late-inbound sign in the M8-r1 class model (two-fold protocol)", "r1_columns": R1.COLS,
                 "arms": ARMS, "coverage_2025": coverage(x)}
    g = x[x.y < 80000].groupby(x.in_late.map({0.0: "on time", 1.0: "late"}).fillna("no link"))
    rec["placeholder_share_2025_by_sign"] = {k: {"rows": int(len(v)), "placeholder": float(v.ph.mean())} for k, v in g}
    print("2025 placeholder share by sign:", rec["placeholder_share_2025_by_sign"], flush=True)

    # 2026 coverage (no label)
    t26 = N.frame_2026()
    x26, _ = N.features(t26)
    s26 = inbound_sign(t26).set_index(C.ID)
    x26["in_late"] = s26.in_late.reindex(x26[C.ID]).to_numpy()
    rec["coverage_2026"] = coverage(x26)
    print("coverage 2025 (January, July):", {m: rec["coverage_2025"][m] for m in (1, 7)}, "\ncoverage 2026:", rec["coverage_2026"], flush=True)

    tsrc = ty[ty.month.isin(R1.TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in R1.TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    pos = r.index.get_indexer(te[C.ID])
    v32, k0, mo, ph = base[pos], (te.ku <= 0).to_numpy(), te.month.to_numpy(), te.ph.to_numpy()
    cands: dict = {}
    keep = te[[C.ID, "month", "ph", "ms", "ku", "in_late", "in_lead"]].copy()
    rec["arms_read"] = {}
    for arm, extra in ARMS.items():
        cols = R1.COLS + extra
        p, m_real = fit_score(tr, te, cols)
        mix = np.maximum(C.FLOOR, p * te.ms + (1 - p) * m_real)
        keep["p_" + arm], keep["m_real"] = p, m_real
        ar = {"auc": float(roc_auc_score(ph, p)), "brier": float(brier_score_loss(ph, p)),
              "auc_by_month": {int(m): float(roc_auc_score(ph[mo == m], p[mo == m])) for m in (1, 7)},
              "mean_p_by_month": {int(m): float(p[mo == m].mean()) for m in (1, 7)}, "G_vs_v32": {}}
        for name, wt in WEIGHTS:
            cand = base.copy()
            cand[pos] = np.maximum(C.FLOOR, np.where(k0, wt * mix + (1 - wt) * v32, v32))
            cands[(arm, name)] = cand
            sc = GS.score(ids, base, cand, y=y)
            ar["G_vs_v32"][name] = {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                                    "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"]}
        rec["arms_read"][arm] = ar
        print(arm, "AUC", round(ar["auc"], 4), ar["auc_by_month"], {k: round(v["G"], 1) for k, v in ar["G_vs_v32"].items()}, flush=True)
    rec["gain_over_r1"] = {}
    for arm in ("sign", "sign+lead"):
        rec["gain_over_r1"][arm] = {}
        for name, _ in WEIGHTS:
            sc = GS.score(ids, cands[("r1", name)], cands[(arm, name)], y=y)
            d = ((y[pos] - cands[("r1", name)][pos]) ** 2 - (y[pos] - cands[(arm, name)][pos]) ** 2) * r.w26.to_numpy()[pos]
            o = np.argsort(-np.abs(d))
            rec["gain_over_r1"][arm][name] = {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                                              "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                                              "G_without_top5_rows": float(d.sum() - d[o[:5]].sum()), "rows_better": int((d > 0).sum()),
                                              "rows_worse": int((d < 0).sum()), "G_on_placeholder_rows": float(d[ph].sum()),
                                              "G_on_real_rows": float(d[~ph].sum())}
            q = rec["gain_over_r1"][arm][name]
            print(f"{arm} over r1 at {name}: {q['G']:+.1f} [{q['lo95']:+.1f}, {q['hi95']:+.1f}] months {({k: round(v) for k, v in q['months'].items()})} "
                  f"without top days {q['G_without_top_days']:+.0f}, top 5 rows {q['G_without_top5_rows']:+.0f}; better {q['rows_better']}, worse {q['rows_worse']}", flush=True)
    C.refuse_existing(out_json, out_rows)
    keep.to_parquet(out_rows, index=False)
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    rec["code_sha256"] = {"inbound_sign.py": C.sha(__file__), "noac_candidate_r1.py": C.sha(R1.__file__), "noac_candidate.py": C.sha(N.__file__),
                          "regime_signal.py": C.sha(S.__file__), "common.py": C.sha(C.__file__)}
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
