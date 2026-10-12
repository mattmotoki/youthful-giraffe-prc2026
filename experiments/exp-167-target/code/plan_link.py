"""M11 (C30): the plan of a LIRF row without a plan, read from the arrival row of the same flight.

    .venv/bin/python experiments/exp-167-target/code/plan_link.py --tag 001

Source: `fresh` (exp-168 `ANGLES.md` Part 3, `code/link_rule.py`, `code/fold_reads.py` (g)).  The
flight columns are joined only when |BLOCK - LOBT| <= 3,606 s.  A departure without a plan that
lands at one of the ten airports has an arrival row with LOBT and AOBT_3.  This file is my own
code of the link and of rule 2 (a check of `fresh`'s link), and the route read of the desk:
G against F with `prc/gscore.py` on the January and July 2025 rows, by month, day and state.

States (d1 = AOBT_3 - LOBT, s1 = SCHED - LOBT, both from the arrival row and the departure):
    real         |d1| > 3,900 and |s1| <= 3,606   -> MVT - AOBT_3, at most MVT - LOBT - 3,606 when d1 > 0
    placeholder  |d1| < 2,700 and |s1| > 3,606    -> MVT - SCHED
    wrong-hour   |d1| < 2,700 and |s1| <= 3,606   -> keep the entry
    edge         2,700 <= |d1| <= 3,900           -> as real
    open         |d1| > 3,900 and |s1| > 3,606    -> 0.89 real + 0.11 placeholder
Write-once.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import common as C

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

WINDOW = 3606.0
OPEN_REAL = 0.89            # `fresh`: the real share of the open state, all 2025 months (the one in-sample constant)
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AIRCRAFT_TYPE_mvt", "LOBT_flt", "AOBT_3_flt", "CALLSIGN_flt", "ADEP_flt"]


def link(t: pd.DataFrame, airport: str | None = "LIRF") -> pd.DataFrame:
    """One arrival row per departure without a plan: same callsign, origin and destination."""
    dep = t[(t.PHASE_mvt == "DEP") & t.LOBT_flt.isna()]
    if airport:
        dep = dep[dep.ADEP_mvt == airport]
    arr = t[(t.PHASE_mvt == "ARR") & t.LOBT_flt.notna() & t.ADEP_flt.isin(C.AIRPORTS) & t.CALLSIGN_flt.notna()]
    a = arr[["ADEP_flt", "ADES_mvt", "CALLSIGN_flt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]]
    a.columns = ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "arr_t", "L", "A3"]
    c = dep.drop(columns=["LOBT_flt", "AOBT_3_flt"]).merge(a, on=["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"], how="inner")
    c["exot"] = C.secs(c.MVT_TIME_UTC_mvt, c.A3)
    c = c[c.exot.between(60, 3600) & (c.arr_t > c.MVT_TIME_UTC_mvt)].copy()
    c["rank"] = (c.exot - 900).abs()
    c = c.sort_values([C.ID, "rank"]).drop_duplicates(C.ID)
    c[C.ID] = c[C.ID].astype("int64")
    return c.reset_index(drop=True)


def states(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["d1"] = C.secs(c.A3, c.L)
    c["s1"] = C.secs(c.SCHED_TIME_UTC_mvt, c.L)
    c["M_L"] = C.secs(c.MVT_TIME_UTC_mvt, c.L)
    c["ms"] = C.secs(c.MVT_TIME_UTC_mvt, c.SCHED_TIME_UTC_mvt)
    out, inn, far = c.d1.abs() > 3900, c.d1.abs() < 2700, c.s1.abs() > WINDOW
    c["state"] = np.select([out & ~far, inn & far, inn & ~far, out & far], ["real", "placeholder", "wrong-hour", "open"], "edge")
    c["est"] = np.where(c.d1 > 3900, np.minimum(c.exot, c.M_L - WINDOW), c.exot)
    c["rule2"] = np.select([c.state == "placeholder", c.state.isin(["real", "edge"]), c.state == "open"],
                           [c.ms, c.est, OPEN_REAL * c.est + (1 - OPEN_REAL) * c.ms], np.nan)
    return c


def label_class(c: pd.DataFrame) -> np.ndarray:
    y = c.TAXITIME_SEC_mvt.astype("float64")
    return np.where(y > 80000, "roll", np.where(C.secs(c.BLOCK_TIME_UTC_mvt, c.SCHED_TIME_UTC_mvt).abs() <= C.PH_TOL, "ph", "real"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"plan-link-{a.tag}.json", C.RESULTS / f"plan-link-{a.tag}.md"
    C.refuse_existing(out_json, out_md)
    t = C.raw_2025(columns=COLS)
    rec: dict = {"what": "exp-167 M11: C30 route read of rule 2 on the fold rows (own link code)", "window_s": WINDOW, "open_real_share": OPEN_REAL}
    doc = ["# M11: C30, the plan of a LIRF row without a plan from the arrival row of the same flight", ""]

    # the link on all ten airports and twelve months, against `fresh`'s counts
    call = states(link(t, airport=None))
    call["kl"] = label_class(call)
    tab = call.groupby(["ADEP_mvt", "state"]).size().unstack(fill_value=0)
    lirf12 = call[call.ADEP_mvt == "LIRF"]
    tkl = lirf12.groupby(["state", "kl"]).size().unstack(fill_value=0)
    rec["links_2025"] = int(len(call))
    rec["links_2025_by_airport_state"] = tab.to_dict("index")
    rec["lirf_2025_state_by_label_class"] = tkl.to_dict("index")
    doc += [f"## The link, all of 2025: {len(call):,} linked departures without a plan (`fresh`: 1,440)", "", C.md(tab), "",
            "LIRF, state against the class of the label:", "", C.md(tkl), ""]
    print(tab.to_string(), tkl.to_string(), sep="\n", flush=True)

    # the route read on the fold rows
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    c = lirf12[lirf12[C.ID].isin(r.index) & lirf12.rule2.notna()].copy()
    pos = r.index.get_indexer(c[C.ID])
    cand = base.copy()
    cand[pos] = np.maximum(C.FLOOR, c.rule2.to_numpy())
    sc = GS.score(ids, base, cand, y=y)
    c["y"], c["F"], c["new"], c["w26"] = y[pos], base[pos], cand[pos], r.w26.to_numpy()[pos]
    c["gain"] = ((c.y - c.F) ** 2 - (c.y - c.new) ** 2) * c.w26
    c["gain_unit"] = ((c.y - c.F) ** 2 - (c.y - c.new) ** 2) / len(r)
    c["day"] = c.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    byd = c.groupby("day").gain.sum().sort_values(ascending=False)
    o = c.gain.abs().sort_values(ascending=False).index
    wh = lirf12[lirf12[C.ID].isin(r.index) & lirf12.rule2.isna()]
    rec["fold"] = {"rows_linked": int(len(c) + len(wh)), "rows_moved": int(len(c)), "rows_kept_wrong_hour": int(len(wh)),
                   "G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                   "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                   "G_unit_weights": float(c.gain_unit.sum()), "sum_of_row_gains": float(c.gain.sum()),
                   "rows_gain": int((c.gain > 0).sum()), "rows_lose": int((c.gain < 0).sum()),
                   "largest_row_gain": float(c.gain.max()), "largest_row_loss": float(c.gain.min()),
                   "G_without_top5_rows": float(c.gain.sum() - c.gain[o[:5]].sum()),
                   "days": int(len(byd)), "largest_day": str(byd.index[0]), "largest_day_gain": float(byd.iloc[0]),
                   "largest_day_share": float(byd.iloc[0] / c.gain.sum()), "top3_days_share": float(byd.iloc[:3].sum() / c.gain.sum()),
                   "days_gain": int((byd > 0).sum()), "days_lose": int((byd < 0).sum()),
                   "by_state": c.groupby("state").gain.agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "gain"}).to_dict("index"),
                   "state_by_label_class": c.assign(kl=label_class(c)).groupby(["state", "kl"]).size().unstack(fill_value=0).to_dict("index")}
    f = rec["fold"]
    print(f"G {f['G']:+.1f} [{f['lo95']:+.1f}, {f['hi95']:+.1f}] months {({k: round(v) for k, v in f['months'].items()})}; unit weights {f['G_unit_weights']:+.1f}; "
          f"without top days {f['G_without_top_days']:+.0f}; without top 5 rows {f['G_without_top5_rows']:+.0f}; rows {f['rows_gain']} gain, {f['rows_lose']} lose; "
          f"largest day {f['largest_day']} share {f['largest_day_share']:.3f}; top 3 days {f['top3_days_share']:.3f}", flush=True)
    td = c.groupby("day").agg(rows=("gain", "size"), gain=("gain", "sum")).sort_values("gain", ascending=False)
    td["share"] = td.gain / c.gain.sum()
    tr = c.assign(kl=label_class(c))[["day", "FLIGHT_mvt", "state", "kl", "ms", "exot", "M_L", "y", "F", "new", "gain"]].sort_values("gain", ascending=False)
    rec["fold_days"] = td.reset_index().to_dict("records")
    rec["fold_rows"] = tr.to_dict("records")
    doc += ["## The route read on the fold rows (G against F, 2026 cell weights)", "",
            f"G **{f['G']:+,.0f} [{f['lo95']:+,.0f}, {f['hi95']:+,.0f}]**; months {({k: round(v) for k, v in f['months'].items()})}; "
            f"{f['G_unit_weights']:+,.0f} at unit weights; {f['G_without_top_days']:+,.0f} without the top days; "
            f"{f['G_without_top5_rows']:+,.0f} without the 5 largest rows; {f['rows_moved']} rows moved, {f['rows_gain']} gain, {f['rows_lose']} lose; "
            f"{f['days']} days, the largest ({f['largest_day']}) holds {100 * f['largest_day_share']:.1f} % of the gain, the top three {100 * f['top3_days_share']:.1f} %.", "",
            "By state:", "", C.md(c.groupby("state").gain.agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "gain"})), "",
            "By day:", "", C.md(td, digits=3), "", "By row:", "", C.md(tr, index=False), ""]
    print(td.round(3).to_string(), tr.round(0).to_string(), sep="\n", flush=True)
    rec["code_sha256"] = {"plan_link.py": C.sha(__file__), "common.py": C.sha(C.__file__)}
    C.refuse_existing(out_json, out_md)
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
