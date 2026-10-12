"""M22-r1 (unit 10b, C39 (b)): the airline recording practice of the month on LIRF rows WITH a plan.

    .venv/bin/python experiments/exp-167-target/code/practice_arm_r1.py --tag 001 --base data/exp167/noplan-form-v2-001-entry.parquet

Correction of `practice_arm.py` (M22): a row whose take-off is at or before SCHED is outside
the rule.  A placeholder block time equals SCHED, so it cannot be after the take-off; M22 (1)
counts 0 placeholders on the 17 such rows of 2025.  Nothing else differs.

A LIRF departure with a plan whose airline prefix has a default share of 0.95 or more on its
LIRF arrivals of the same month (10 arrivals and more; `noplan_form_v2.arrival_share`) takes a
value by the airline's status.  The status comes from labelled months: over the prefix's months
that pass the same arrival criterion, n = departures with a plan, s = their placeholder share.

    confirmed  n >= 10, s >= 0.9        the placeholder point max(60, MVT - SCHED)
    excluded   n >= 10, s <  0.5        the base
    mixed      n >= 10, 0.5 <= s < 0.9  base + s x (point - base)
    new        n <  10                  base + rate x (point - base),
                                        rate = (confirmed airlines + 0.5) / (airlines with n >= 10 + 1)

(1) 2025, each month held in turn (status from the eleven other months): placeholder share by
status; a count, no model.  (2) Fold: the arm on January and July 2025 (status from the ten
other months) against the v47 form and against F.  (3) 2026 on the base entry (a v2 entry of
the no-plan form: its rows have no plan, the arm's rows have one).  The `new` rows are in the
entry only when their share in (1) is 0.75 or more (SPEC unit 10b).  Write-once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import noac_candidate_r1 as R1
import noplan_form_v2 as V2

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

ARR_SHARE, MIN_ARR = 0.95, 10
MIN_DEP, S_CONFIRMED, S_EXCLUDED = 10, 0.9, 0.5
NEW_MIN_SHARE = 0.75
STATUSES = ("confirmed", "mixed", "new", "excluded")
BANDS = [0, 600, 1800, 3900, np.inf]
BAND_NAMES = ["0 to 10 min", "10 to 30 min", "30 to 65 min", "above 65 min"]


def rows(t: pd.DataFrame, labelled: bool) -> pd.DataFrame:
    """LIRF departures with a plan whose prefix passes the arrival criterion in the row's month."""
    share = V2.arrival_share(t)
    d = t[(t.PHASE_mvt == "DEP") & (t.ADEP_mvt == "LIRF") & t.LOBT_flt.notna()].copy()
    d[C.ID] = d[C.ID].astype("int64")
    d["pre"] = d.FLIGHT_mvt.str[:3]
    d["month"] = d.MVT_TIME_UTC_mvt.dt.month
    d["day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    d["ms"] = C.secs(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["exot"] = C.secs(d.MVT_TIME_UTC_mvt, d.AOBT_3_flt)
    key = pd.MultiIndex.from_arrays([d.pre.to_numpy(), d.month.to_numpy()])
    d["arr_share"], d["arr_n"] = share.pre_def.reindex(key).to_numpy(), share.pre_n.reindex(key).to_numpy()
    d = d[(d.arr_share >= ARR_SHARE) & (d.arr_n >= MIN_ARR) & (d.ms > 0)].copy()
    d["point"] = np.maximum(C.FLOOR, d.ms)
    d["band"] = pd.cut(d.ms, BANDS, labels=BAND_NAMES).astype(str)
    if labelled:
        d["y"] = d.TAXITIME_SEC_mvt.astype("float64")
        d["ph"] = C.secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL
    return d.reset_index(drop=True)


def status_table(src: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Status of each prefix from labelled rows of `src`, and the rate for a new prefix."""
    g = src.groupby("pre").ph.agg(n="size", s="mean")
    g = g[g.n >= MIN_DEP]
    g["status"] = np.where(g.s >= S_CONFIRMED, "confirmed", np.where(g.s < S_EXCLUDED, "excluded", "mixed"))
    rate = float(((g.status == "confirmed").sum() + 0.5) / (len(g) + 1.0))
    return g, rate


def apply(d: pd.DataFrame, base: np.ndarray, tab: pd.DataFrame, rate: float, with_new: bool) -> tuple[np.ndarray, np.ndarray]:
    st = d.pre.map(tab.status).fillna("new").to_numpy()
    s = d.pre.map(tab.s).to_numpy("float64")
    pt = d.point.to_numpy("float64")
    w = np.select([st == "confirmed", st == "mixed", (st == "new") & with_new], [1.0, np.nan_to_num(s), rate], 0.0)
    return np.maximum(C.FLOOR, base + w * (pt - base)), st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", required=True, help="a full entry; the v2 entry of the no-plan form on the wanted route")
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"practice-arm-r1-{a.tag}.json", C.RESULTS / f"practice-arm-r1-{a.tag}.md"
    out_rows, out_entry = C.EXP / f"candidates/practice-arm-r1-{a.tag}.parquet", C.SAVED / f"practice-arm-r1-{a.tag}-entry.parquet"
    C.refuse_existing(out_json, out_md, out_rows, out_entry)
    base_path = Path(a.base)
    rec: dict = {"what": "exp-167 M22-r1: the airline recording practice of the month on LIRF rows with a plan after SCHED (C39 (b))", "base_entry": str(base_path), "base_sha256": C.sha(base_path),
                 "rule": {"arrival_share_at_least": ARR_SHARE, "min_arrivals": MIN_ARR, "min_departures": MIN_DEP, "confirmed_at_least": S_CONFIRMED, "excluded_under": S_EXCLUDED,
                          "new_rows_need_share": NEW_MIN_SHARE}}
    doc = ["# M22-r1: the airline recording practice of the month on LIRF rows with a plan, take-off after SCHED (C39 (b))", ""]
    t25 = C.raw_2025()
    d25 = rows(t25, labelled=True)

    # ------------------------------------------------------------ (1) 2025, each month held in turn
    parts = []
    for m in range(1, 13):
        tab, rate = status_table(d25[d25.month != m])
        g = d25[d25.month == m].copy()
        g["status"] = g.pre.map(tab.status).fillna("new").to_numpy()
        g["status_n"], g["status_s"], g["rate_new"] = g.pre.map(tab.n).to_numpy(), g.pre.map(tab.s).to_numpy(), rate
        parts.append(g)
    lo = pd.concat(parts, ignore_index=True)
    by_status = lo.groupby("status").ph.agg(rows="size", placeholders="sum", share="mean")
    by_status["airlines"] = lo.groupby("status").pre.nunique()
    by_status["prefix_months"] = lo.groupby("status").apply(lambda s_: len(s_[["pre", "month"]].drop_duplicates()), include_groups=False)
    by_band = lo.groupby(["status", "band"]).ph.agg(rows="size", share="mean").unstack(0)
    by_air = lo.groupby(["status", "pre"]).ph.agg(rows="size", share="mean", months=lambda s_: 0).reset_index()
    by_air["months"] = lo.groupby(["status", "pre"]).month.nunique().to_numpy()
    by_pm = lo.groupby(["status", "pre", "month"]).ph.agg(rows="size", share="mean").reset_index()
    weak = by_pm[(by_pm.status == "confirmed") & (by_pm.share < 0.9) & (by_pm.rows >= 3)]
    new_share = float(by_status.share.get("new", np.nan))
    with_new = bool(np.isfinite(new_share) and new_share >= NEW_MIN_SHARE)
    rec["held_month_2025"] = {"rows": int(len(lo)), "by_status": by_status.to_dict("index"), "by_status_and_band": {str(k): v for k, v in by_band.to_dict().items()},
                              "by_airline": by_air.to_dict("records"), "confirmed_prefix_months_under_0.9": weak.to_dict("records"),
                              "new_rows_by_prefix_month": by_pm[by_pm.status == "new"].to_dict("records"), "rate_new_by_held_month": {int(m): float(g.rate_new.iloc[0]) for m, g in lo.groupby("month")},
                              "new_rows_in_the_entry": with_new}
    print("(1) 2025, each month held in turn: eligible rows with a plan", len(lo), "\n", by_status.round(3).to_string(), "\nby band (share):\n", by_band.round(3).to_string(),
          "\nby airline:\n", by_air.sort_values(["status", "rows"], ascending=[True, False]).round(3).to_string(index=False),
          "\nconfirmed prefix-months under 0.9 (3 rows and more):\n", weak.round(2).to_string(index=False),
          "\nnew rows by prefix and month:\n", by_pm[by_pm.status == "new"].round(2).to_string(index=False), "\nnew rows in the entry:", with_new, flush=True)
    doc += ["## (1) 2025, each month held in turn (status from the eleven other months)", "", C.md(by_status, digits=3), "",
            "By airline:", "", C.md(by_air.sort_values(["status", "rows"], ascending=[True, False]), index=False, digits=3), ""]

    # ------------------------------------------------------------ (2) fold
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    tab10, rate10 = status_table(d25[d25.month.isin(R1.TEN)])
    f = d25[d25[C.ID].isin(r.index)].reset_index(drop=True)
    pos = r.index.get_indexer(f[C.ID])
    w26 = r.w26.to_numpy()[pos]
    fold = {"status_table_ten_months": tab10.reset_index().to_dict("records"), "rate_new_ten_months": rate10, "rows": int(len(f)), "class_of_rows": r.cls.astype(str).to_numpy()[pos].tolist().count("clock")}
    for bk in ("v47", "F"):
        base = np.maximum(C.FLOOR, r[bk].to_numpy("float64"))
        for nm, wn in (("as_served", with_new), ("without_new", False), ("with_new", True)):
            pred, st = apply(f, base[pos], tab10, rate10, wn)
            cand = base.copy()
            cand[pos] = pred
            sc = GS.score(ids, base, cand, y=y)
            g = ((y[pos] - base[pos]) ** 2 - (y[pos] - pred) ** 2) * w26
            fold[f"{bk}_{nm}"] = {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                                  "rows_moved": int((pred != base[pos]).sum()), "rows_gain": int((g > 0).sum()), "rows_lose": int((g < 0).sum()),
                                  "by_status": {k: {"rows": int((st == k).sum()), "placeholders": int(f.ph[st == k].sum()), "gain": float(g[st == k].sum()),
                                                    "mean_abs_move_s": float(np.abs(pred - base[pos])[st == k].mean()) if (st == k).any() else 0.0} for k in STATUSES},
                                  "by_airline": {k: float(v) for k, v in pd.Series(g).groupby(f.pre.to_numpy()).sum().items()}}
            q = fold[f"{bk}_{nm}"]
            print(f"(2) fold, base {bk}, {nm}: G {q['G']:+.1f} [{q['lo95']:+.1f}, {q['hi95']:+.1f}] months {({k: round(v, 1) for k, v in q['months'].items()})}; without top days {q['G_without_top_days']:+.1f}; "
                  f"moved {q['rows_moved']}, gain {q['rows_gain']}, lose {q['rows_lose']}\n    by status {({k: (v['rows'], v['placeholders'], round(v['gain'], 1)) for k, v in q['by_status'].items()})}\n"
                  f"    by airline {({k: round(v, 1) for k, v in q['by_airline'].items() if abs(v) >= 0.05})}", flush=True)
    rec["fold"] = fold
    doc += ["## (2) Fold (status from the ten other months; G in s² per ranking row)", "",
            C.md(pd.DataFrame([{"base": k.split("_")[0], "arm": "_".join(k.split("_")[1:]), "G": v["G"], "lo95": v["lo95"], "hi95": v["hi95"], "January": v["months"].get("1", v["months"].get(1)),
                                "July": v["months"].get("7", v["months"].get(7)), "rows moved": v["rows_moved"], "rows that gain": v["rows_gain"], "rows that lose": v["rows_lose"]}
                               for k, v in fold.items() if isinstance(v, dict) and "G" in v]), index=False), ""]

    # ------------------------------------------------------------ (3) 2026
    t26 = pd.read_parquet(C.RAW / "ranking.parquet")
    d26 = rows(t26, labelled=False)
    tab12, rate12 = status_table(d25)
    b = pd.read_parquet(base_path)
    if len(b) != C.N_RANKING:
        raise SystemExit("the base is not a full entry")
    v = pd.Series(b.TAXITIME_SEC_mvt.to_numpy("float64"), index=b.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    d26["base"] = v.reindex(d26[C.ID]).to_numpy()
    if d26.base.isna().any():
        raise SystemExit("a 2026 row is not in the base entry")
    d26["candidate"], d26["status"] = apply(d26, d26.base.to_numpy(), tab12, rate12, with_new)
    d26["full_point_distance"] = (d26.point - d26.base) ** 2 / C.N_RANKING
    d26["change"] = d26.candidate - d26.base
    # forecast without a label: a row is a placeholder with the (1) share of its status, else it sits at a real point
    real = d25[~d25.ph & (d25.y < 80000)]
    off = float((real.y - real.exot).median()) if len(real) else 0.0
    all_real = t25[(t25.PHASE_mvt == "DEP") & (t25.ADEP_mvt == "LIRF") & t25.LOBT_flt.notna()]
    yr, er = all_real.TAXITIME_SEC_mvt.astype("float64"), C.secs(all_real.MVT_TIME_UTC_mvt, all_real.AOBT_3_flt)
    okr = (C.secs(all_real.BLOCK_TIME_UTC_mvt, all_real.SCHED_TIME_UTC_mvt).abs() > C.PH_TOL) & (yr < 3600) & er.notna()
    off_all, pt_all = float((yr - er)[okr].median()), float(yr[okr].median())
    d26["real_point"] = np.where(d26.exot.notna(), d26.exot + off_all, pt_all)
    pshare = {k: float(by_status.share.get(k, np.nan)) for k in STATUSES}
    p_row = d26.status.map(pshare).to_numpy("float64")
    bb, cc, pp, rr = d26.base.to_numpy(), d26.candidate.to_numpy(), d26.point.to_numpy(), d26.real_point.to_numpy()
    d26["forecast"] = (p_row * ((pp - bb) ** 2 - (pp - cc) ** 2) + (1 - p_row) * ((rr - bb) ** 2 - (rr - cc) ** 2)) / C.N_RANKING
    g26 = d26.groupby("status")
    t26s = pd.DataFrame({"rows": g26.size(), "airlines": g26.pre.nunique(), "rows changed": g26.change.apply(lambda s_: int((s_ != 0).sum())), "mean change, s": g26.change.mean(),
                         "distance from the base": g26.change.apply(lambda s_: (s_ ** 2).sum() / C.N_RANKING), "distance of the base to the point": g26.full_point_distance.sum(),
                         "placeholder share of the status, 2025": pd.Series(pshare), "forecast without a label": g26.forecast.sum()})
    t26a = d26.groupby(["status", "pre", "month"]).agg(rows=("change", "size"), arr_share=("arr_share", "first"), arr_n=("arr_n", "first"), mean_base=("base", "mean"), mean_point=("point", "mean"),
                                                         distance=("change", lambda s_: (s_ ** 2).sum() / C.N_RANKING), to_point=("full_point_distance", "sum"), forecast=("forecast", "sum")).reset_index()
    rec["rows_2026"] = {"rows": int(len(d26)), "status_table_twelve_months": tab12.reset_index().to_dict("records"), "rate_new_twelve_months": rate12, "by_status": t26s.to_dict("index"),
                        "by_airline_month": t26a.to_dict("records"), "distance_from_base": float((d26.change ** 2).sum() / C.N_RANKING), "forecast_without_a_label": float(d26.forecast.sum()),
                        "real_point_offset_s": off_all, "real_point_without_aobt_s": pt_all, "real_offset_of_eligible_rows_s": off, "new_rows_in_the_entry": with_new}
    print("(3) 2026 rows with a plan in eligible prefix-months:", len(d26), "\n", t26s.round(2).to_string(), "\n", t26a.round(2).to_string(index=False),
          "\nstatus table (twelve months):\n", tab12.round(3).to_string(), "\nrate for a new prefix:", round(rate12, 3),
          "\ndistance from the base", round(rec["rows_2026"]["distance_from_base"], 1), "; forecast", round(rec["rows_2026"]["forecast_without_a_label"], 1), flush=True)
    doc += [f"## (3) 2026 on `{base_path.name}` (sha256 {rec['base_sha256'][:12]}…); distances and forecast in s² per ranking row", "", C.md(t26s, digits=2), "",
            C.md(t26a, index=False, digits=2), ""]
    C.refuse_existing(out_json, out_md, out_rows, out_entry)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    d26[[C.ID, "month", "FLIGHT_mvt", "pre", "ms", "arr_share", "arr_n", "status", "base", "point", "candidate"]].to_parquet(out_rows, index=False)
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    new = v.copy()
    new.loc[d26[C.ID].to_numpy()] = d26.candidate.to_numpy()
    order = b.MVT_ID_mvt.to_numpy("float64").astype("int64")
    e = pd.DataFrame({"MVT_ID_mvt": b.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new.reindex(order).to_numpy()})
    if len(e) != C.N_RANKING or not np.isfinite(e.TAXITIME_SEC_mvt).all() or (e.TAXITIME_SEC_mvt < C.FLOOR).any():
        raise SystemExit("the entry is not a full finite vector above the floor")
    e.to_parquet(out_entry, index=False)
    diff = np.abs(e.TAXITIME_SEC_mvt.to_numpy() - b.TAXITIME_SEC_mvt.to_numpy("float64"))
    rec["entry"] = {"file": str(out_entry.relative_to(C.ROOT)), "sha256": C.sha(out_entry), "rows": int(len(e)), "rows_changed_from_base": int((diff > 0).sum()),
                    "max_abs_change_s": float(diff.max()), "distance_from_base": float((diff ** 2).sum() / C.N_RANKING)}
    print("entry", rec["entry"], flush=True)
    rec["code_sha256"] = {"practice_arm_r1.py": C.sha(__file__), "noplan_form_v2.py": C.sha(V2.__file__), "common.py": C.sha(C.__file__)}
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
