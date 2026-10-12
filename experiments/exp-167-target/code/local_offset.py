"""M13 (C32): LIRF block times that are early by the local UTC offset.

    .venv/bin/python experiments/exp-167-target/code/local_offset.py --tag 001

Source: `fresh` (exp-168 `code/lirf_local_time.py`).  A LIRF row without a plan and
0 < MVT - SCHED < 3,900 s cannot have a normal block time (it would be joined).  Its label is the
placeholder MVT - SCHED, or a taxi time plus the local UTC offset (the block time was entered
1 h early in winter time, 2 h in summer time).  Two-point rule:

    pred = q x (offset + 900) + (1 - q) x (MVT - SCHED)
    q = (shifted rows of the airline prefix + 2 g) / (placeholder and shifted rows of the prefix + 2)

with the counts and the overall share g from other months.  Read: the two-fold protocol (tables
from the ten other months of 2025), G against F on the January and July rows.  Then the 2026
rows with the twelve-month table (no label).  Write-once.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import common as C
import noac_candidate as N

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

TEN = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
BAND = (0.0, 3900.0)
TAXI = 900.0
PRIOR = 2.0


def rows(t: pd.DataFrame) -> pd.DataFrame:
    """LIRF departures without a plan in the band, with the local offset and the airline prefix."""
    x = t[(t.PHASE_mvt == "DEP") & (t.ADEP_mvt == "LIRF") & t.LOBT_flt.isna()].copy()
    x[C.ID] = x[C.ID].astype("int64")
    loc = x.MVT_TIME_UTC_mvt.dt.tz_convert("Europe/Rome")
    x["off"] = (loc.dt.tz_localize(None) - x.MVT_TIME_UTC_mvt.dt.tz_localize(None)).dt.total_seconds()
    x["month"] = x.MVT_TIME_UTC_mvt.dt.month
    x["ms"] = C.secs(x.MVT_TIME_UTC_mvt, x.SCHED_TIME_UTC_mvt)
    x["pre"] = x.FLIGHT_mvt.fillna("").str[:3]
    return x[(x.ms > BAND[0]) & (x.ms < BAND[1])].reset_index(drop=True)


def kinds(x: pd.DataFrame) -> np.ndarray:
    y = x.TAXITIME_SEC_mvt.astype("float64")
    bs = C.secs(x.BLOCK_TIME_UTC_mvt, x.SCHED_TIME_UTC_mvt)
    return np.select([y > 80000, bs.abs() <= C.PH_TOL, (y > x.off + 240) & (y < x.off + 2400) & (bs < 3000 - x.off)], ["roll", "ph", "shifted"], "real")


def table(src: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    s = src[src.kind.isin(["ph", "shifted"])]
    return s.groupby("pre").kind.agg(lambda k: (k == "shifted").sum()), s.groupby("pre").size(), float((s.kind == "shifted").mean())


def weight(x: pd.DataFrame, tab) -> tuple[np.ndarray, np.ndarray]:
    a, n, g = tab
    n_ = x.pre.map(n).fillna(0).to_numpy("float64")
    return (x.pre.map(a).fillna(0).to_numpy("float64") + PRIOR * g) / (n_ + PRIOR), n_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"local-offset-{a.tag}.json", C.RESULTS / f"local-offset-{a.tag}.md"
    C.refuse_existing(out_json, out_md)
    x = rows(C.raw_2025())
    x["y"] = x.TAXITIME_SEC_mvt.astype("float64")
    x["kind"] = kinds(x)
    rec: dict = {"what": "exp-167 M13: C32, the two-point prefix rule on LIRF rows without a plan and 0 < MVT - SCHED < 3,900 s",
                 "rows_2025": int(len(x)), "kinds_2025": x.kind.value_counts().to_dict(),
                 "kinds_2025_by_season": x.groupby(["off", "kind"]).size().unstack(fill_value=0).to_dict("index")}
    tp = x.groupby(["pre", "kind"]).size().unstack(fill_value=0)
    tp["n"] = tp.sum(axis=1)
    tp = tp[tp.n >= 3].sort_values("n", ascending=False)
    doc = ["# M13: C32, LIRF block times early by the local UTC offset", "", f"2025: {len(x)} rows in the band; kinds {rec['kinds_2025']}.", "",
           "By airline prefix (3 rows and more; the empty prefix is a row without flight id):", "", C.md(tp), ""]
    print(rec["kinds_2025"], "\n", tp.to_string(), flush=True)

    # two-fold read
    tab10 = table(x[x.month.isin(TEN)])
    te = x[x.month.isin((1, 7))].copy()
    te["q"], te["n10"] = weight(te, tab10)
    te["rule"] = te.q * (te.off + TAXI) + (1 - te.q) * te.ms
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    te = te[te[C.ID].isin(r.index)].copy()
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    pos = r.index.get_indexer(te[C.ID])
    cand = base.copy()
    cand[pos] = np.maximum(C.FLOOR, te.rule.to_numpy())
    sc = GS.score(ids, base, cand, y=y)
    te["F"], te["w26"] = base[pos], r.w26.to_numpy()[pos]
    te["gain"] = ((te.y - te.F) ** 2 - (te.y - te.rule) ** 2) * te.w26
    known = np.where(te.kind == "shifted", te.off + TAXI, np.where(te.kind == "ph", te.ms, te.F))
    o = te.gain.abs().sort_values(ascending=False).index
    rec["fold"] = {"rows": int(len(te)), "overall_shifted_share_ten_months": tab10[2], "G": sc["G"], "lo95": sc["interval"]["lo95"],
                   "hi95": sc["interval"]["hi95"], "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"],
                   "cell_in_F": float((((te.y - te.F) ** 2) * te.w26).sum()), "cell_in_rule": float((((te.y - te.rule) ** 2) * te.w26).sum()),
                   "cell_with_known_class": float((((te.y - known) ** 2) * te.w26).sum()),
                   "rows_gain": int((te.gain > 0).sum()), "rows_lose": int((te.gain < 0).sum()),
                   "G_without_top5_rows": float(te.gain.sum() - te.gain[o[:5]].sum()),
                   "gain_by_prefix": te.groupby("pre").gain.agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "gain"}).to_dict("index"),
                   "gain_by_kind": te.groupby("kind").gain.agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "gain"}).to_dict("index")}
    f = rec["fold"]
    tr = te[["FLIGHT_mvt", "month", "kind", "ms", "off", "q", "n10", "y", "F", "rule", "gain"]].sort_values("gain", ascending=False)
    rec["fold_rows"] = tr.to_dict("records")
    print(f"fold: {f['rows']} rows; G {f['G']:+.1f} [{f['lo95']:+.1f}, {f['hi95']:+.1f}] months {({k: round(v) for k, v in f['months'].items()})}; "
          f"cell in F {f['cell_in_F']:.0f}, in the rule {f['cell_in_rule']:.0f}, known class {f['cell_with_known_class']:.0f}; "
          f"without top days {f['G_without_top_days']:+.0f}; without top 5 rows {f['G_without_top5_rows']:+.0f}; {f['rows_gain']} gain, {f['rows_lose']} lose", flush=True)
    print(tr.round(2).to_string(), flush=True)
    doc += ["## Two-fold read (tables from the ten other months), G against F", "",
            f"{f['rows']} rows; G **{f['G']:+,.0f} [{f['lo95']:+,.0f}, {f['hi95']:+,.0f}]**; months {({k: round(v) for k, v in f['months'].items()})}; "
            f"the cell is {f['cell_in_F']:,.0f} in F, {f['cell_in_rule']:,.0f} in the rule, {f['cell_with_known_class']:,.0f} with the known class; "
            f"{f['G_without_top_days']:+,.0f} without the top days; {f['G_without_top5_rows']:+,.0f} without the 5 largest rows; "
            f"{f['rows_gain']} rows gain, {f['rows_lose']} lose.", "", C.md(tr, index=False, digits=2), ""]

    # 2026 rows with the twelve-month table
    t26 = N.frame_2026()
    z = rows(t26)
    z["q"], z["n25"] = weight(z, table(x))
    z["rule"] = z.q * (z.off + TAXI) + (1 - z.q) * z.ms
    v = pd.read_parquet(N.V47)
    v = pd.Series(v.TAXITIME_SEC_mvt.to_numpy("float64"), index=v.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    z["v47"] = v.reindex(z[C.ID]).to_numpy()
    z["t"] = ((z.v47 - z.ms) / (z.off + TAXI - z.ms)).clip(0, 1)
    rec["rows_2026"] = {"rows": int(len(z)), "by_month": z.groupby("month").size().to_dict(),
                        "distance_v47_from_rule": float(((z.v47 - z.rule) ** 2).sum() / C.N_RANKING),
                        "rows_with_prefix_unknown_in_2025": int((z.n25 == 0).sum()),
                        "mean_q": float(z.q.mean()), "mean_v47_implied_weight": float(z.t.mean())}
    tz = z[["FLIGHT_mvt", "month", "ms", "off", "q", "n25", "v47", "t", "rule"]].sort_values(["month", "ms"])
    rec["rows_2026_table"] = tz.to_dict("records")
    print("2026:", rec["rows_2026"], "\n", tz.round(2).to_string(), flush=True)
    doc += [f"## 2026 (no label): {len(z)} rows; distance of v47 from the rule {rec['rows_2026']['distance_v47_from_rule']:.0f} s2 per ranking row", "",
            C.md(tz, index=False, digits=2), ""]
    rec["code_sha256"] = {"local_offset.py": C.sha(__file__), "noac_candidate.py": C.sha(N.__file__), "common.py": C.sha(C.__file__)}
    C.refuse_existing(out_json, out_md)
    out_md.write_text("\n".join(doc) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
