"""Q1/Q2/Q5: structure of BLOCK - AOBT and the other anchors; label quality."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

COLS = ["MVT_ID_mvt","FLIGHT_ID_mvt","CALLSIGN_flt","PHASE_mvt","ADEP_mvt","ADES_mvt","STAND_mvt",
        "RUNWAY_mvt","AIRCRAFT_TYPE_mvt","WK_TBL_CAT_flt","AIRCRAFT_OPERATOR_flt","MARKET_SEGMENT_flt",
        "FLIGHT_TYPE_flt","FLIGHT_RULE_mvt","TAXITIME_SEC_mvt","month","is_ranking","has_nm",
        "nm_taxi","blk_aobt","blk_lobt","blk_iobt","blk_eobt1","blk_sched","aobt_lobt","aobt_sched",
        "lobt_iobt","lobt_eobt1","lobt_sched","mvt_sched","hour","dow","doy",
        "MVT_TIME_UTC_mvt","BLOCK_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","AOBT_3_flt","LOBT_flt"]

def stand_base(s: pd.Series) -> pd.Series:
    """Strip a MARS sub-stand suffix: 519L/519R -> 519, G11L -> G11."""
    return s.astype(str).str.replace(r"(?<=[0-9])[LRC]$", "", regex=True)

def main():
    df = C.load_all(columns=COLS)
    df["apt"] = C.airport(df)
    dep = df[df.PHASE_mvt.astype(str) == "DEP"].copy()
    dep["stand"] = dep["STAND_mvt"].astype(str)
    dep["stand_b"] = stand_base(dep["stand"])
    dep["stand_pfx"] = dep["stand_b"].str.extract(r"^([A-Za-z]+|[0-9])", expand=False).fillna("?")
    dep["apt_stand"] = dep["apt"] + "|" + dep["stand_b"]
    dep["apt_pfx"] = dep["apt"] + "|" + dep["stand_pfx"]
    dep["is25"] = ~dep["is_ranking"]
    dep.to_parquet(d.ROOT / "artifacts" / "exp-003-anatomy" / "dep_all.parquet", index=False)
    tr25 = dep[dep.is25]
    out = []
    W = out.append

    # ---------- global ----------
    W("## Global distribution of the anchors (2025 departures with an NM join)\n")
    nm = tr25[tr25.has_nm]
    rows = {}
    for c in ["blk_aobt","blk_lobt","blk_iobt","blk_eobt1","blk_sched","nm_taxi","TAXITIME_SEC_mvt",
              "aobt_lobt","aobt_sched","lobt_eobt1","mvt_sched"]:
        rows[c] = C.describe(nm[c])
    W(C.md_table(pd.DataFrame(rows).T.rename_axis("quantity")))
    W("")
    W(f"2025 departures: {len(tr25):,}; with NM join {tr25.has_nm.mean():.4%}; "
      f"bulk (y<=3600 and NM) {((tr25.TAXITIME_SEC_mvt<=3600)&tr25.has_nm).mean():.4%}\n")

    # ---------- per airport ----------
    W("## BLOCK - AOBT_3 by airport (2025, NM join)\n")
    t = nm.groupby("apt", observed=True)["blk_aobt"].apply(lambda s: pd.Series(C.describe(s))).unstack()
    t["|d|<=60s"] = nm.groupby("apt", observed=True)["blk_aobt"].apply(lambda s: (s.abs() <= 60).mean() * 100)
    t["|d|<=300s"] = nm.groupby("apt", observed=True)["blk_aobt"].apply(lambda s: (s.abs() <= 300).mean() * 100)
    W(C.md_table(t.rename_axis("airport")))
    W("")

    # ---------- drift by month ----------
    W("## Median BLOCK - AOBT_3 by airport and month (2025 + the two ranking months)\n")
    piv = dep[dep.has_nm].pivot_table(index="apt", columns="month", values="blk_aobt",
                                      aggfunc="median", observed=True)
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.0f}"))
    W("")
    W("## Same, but P90 of |BLOCK - AOBT_3| (dispersion drift)\n")
    piv2 = dep[dep.has_nm].pivot_table(index="apt", columns="month", values="blk_aobt",
                                       aggfunc=lambda s: np.percentile(s.abs(), 90), observed=True)
    W(C.md_table(piv2.rename_axis("airport"), floatfmt="{:.0f}"))
    W("")

    # ---------- by hour ----------
    W("## BLOCK - AOBT_3 by take-off hour (2025, NM)\n")
    th = nm.groupby("hour", observed=True)["blk_aobt"].agg(["size","median","std",
        lambda s: np.percentile(s,10), lambda s: np.percentile(s,90)])
    th.columns = ["n","median","std","p10","p90"]
    W(C.md_table(th.rename_axis("hour"), floatfmt="{:.0f}"))
    W("")

    # ---------- by wake / segment / flight type ----------
    for g in ["WK_TBL_CAT_flt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt","FLIGHT_RULE_mvt"]:
        W(f"## BLOCK - AOBT_3 by {g} (2025, NM)\n")
        tg = nm.groupby(g, observed=True)["blk_aobt"].agg(["size","median","std"])
        W(C.md_table(tg.rename_axis(g), floatfmt="{:.0f}"))
        W("")

    C.write("q1_offset_structure.md", "\n".join(out))

if __name__ == "__main__":
    main()
