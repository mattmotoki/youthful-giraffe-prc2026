"""Q2 (which plan time anchors the block time) and Q5 (label quality)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

TR = [f"2025-{m:02d}" for m in range(1, 7)]
VA = "2025-07"


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    extra = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/mvt_all.parquet",
                            columns=["MVT_ID_mvt", "PHASE_mvt", "IOBT_flt", "EOBT_1_flt"])
    extra = extra[extra["PHASE_mvt"].astype(str) == "DEP"].drop(columns=["PHASE_mvt"])
    dep = dep.merge(extra, on="MVT_ID_mvt", how="left", validate="one_to_one")
    dep["has_lobt"] = dep["LOBT_flt"].notna()
    tr25 = dep[~dep.is_ranking]
    out = []; W = out.append

    W("# Q2 — which plan time anchors the block time; Q5 — label quality\n")

    # ---------------- Q2 ----------------
    W("## Availability of each anchor (share of departures with the column non-null)\n")
    av = []
    for frame, name in [(tr25, "2025 (all 12 months)"), (dep[dep.is_ranking], "ranking (2026)")]:
        r = {"frame": name, "n": len(frame)}
        for c, col in [("AOBT_3", "AOBT_3_flt"), ("LOBT", "LOBT_flt"), ("SCHED", "SCHED_TIME_UTC_mvt"),
                       ("MVT (take-off)", "MVT_TIME_UTC_mvt")]:
            r[c] = 100 * frame[col].notna().mean()
        av.append(r)
    W(C.md_table(pd.DataFrame(av).set_index("frame"), floatfmt="{:.3f}"))
    W("\nAOBT and LOBT are null on exactly the same rows "
      f"({(tr25.AOBT_3_flt.isna() == tr25.LOBT_flt.isna()).mean():.6f} agreement in 2025): the NM join is "
      "all-or-nothing, so **when AOBT is absent the only anchor left is `SCHED_TIME_UTC_mvt`**.\n")

    # anchor RMSE, fitted offset on the january-june months, scored on july
    W("## Anchor comparison: taxi_hat = (MVT - anchor) - median(BLOCK - anchor), fitted on 2025-01..06, scored on 2025-07\n")
    tr = dep[dep.month.isin(TR)]
    va = dep[dep.month == VA]
    segs = {
        "all departures": lambda f: pd.Series(True, index=f.index),
        "NM join": lambda f: f.has_nm,
        "NM join, y<=3600 (bulk)": lambda f: f.has_nm & f.TAXITIME_SEC_mvt.le(3600),
        "NM join, y>3600": lambda f: f.has_nm & f.TAXITIME_SEC_mvt.gt(3600),
        "NM join, AOBT-LOBT>3600": lambda f: f.has_nm & f.aobt_lobt.gt(3600),
        "no NM join": lambda f: ~f.has_nm,
        "noac (AIRCRAFT_TYPE_mvt null)": lambda f: f.AIRCRAFT_TYPE_mvt.isna(),
    }
    anchors = {"AOBT_3": ("blk_aobt", "AOBT_3_flt"), "LOBT": ("blk_lobt", "LOBT_flt"),
               "IOBT": ("blk_iobt", "IOBT_flt"), "EOBT_1": ("blk_eobt1", "EOBT_1_flt"),
               "SCHED": ("blk_sched", "SCHED_TIME_UTC_mvt")}
    rows = []
    for sname, sfun in segs.items():
        mt, mv = sfun(tr), sfun(va)
        r = {"segment": sname, "n_july": int(mv.sum())}
        y = va.loc[mv, "TAXITIME_SEC_mvt"].to_numpy(dtype=float)
        for aname, (offcol, tcol) in anchors.items():
            off = tr.loc[mt, offcol].median()
            pred = ((va.loc[mv, "MVT_TIME_UTC_mvt"] - va.loc[mv, tcol]).dt.total_seconds() - off).to_numpy()
            ok = np.isfinite(pred)
            r[aname] = np.sqrt(np.nanmean((y[ok] - pred[ok]) ** 2)) if ok.any() else np.nan
            r[aname + " cov%"] = 100 * ok.mean() if len(ok) else np.nan
        med = tr.loc[mt, "TAXITIME_SEC_mvt"].median()
        r["const median"] = float(np.sqrt(np.mean((y - med) ** 2)))
        rows.append(r)
    T = pd.DataFrame(rows).set_index("segment")
    W(C.md_table(T[["n_july"] + list(anchors) + ["const median"]], floatfmt="{:.1f}"))
    W("")
    W("Coverage of each anchor within the segment (%):\n")
    W(C.md_table(T[[a + " cov%" for a in anchors]], floatfmt="{:.1f}"))
    W("")

    # LOBT updates
    W("## Do LOBT updates say anything about the actual off-block?\n")
    nm = tr25[tr25.has_nm]
    nm = nm.assign(upd=pd.cut(nm.lobt_eobt1, [-np.inf, -1800, -600, -1, 0, 600, 1800, np.inf],
                              labels=["<-30m", "-30..-10m", "-10m..0", "no change",
                                      "0..+10m", "+10..30m", ">+30m"]))
    t = nm.groupby("upd", observed=True).agg(
        n=("blk_aobt", "size"), share_pct=("blk_aobt", lambda s: 100 * len(s) / len(nm)),
        med_blk_aobt=("blk_aobt", "median"), std_blk_aobt=("blk_aobt", "std"),
        med_blk_lobt=("blk_lobt", "median"), std_blk_lobt=("blk_lobt", "std"),
        med_aobt_lobt=("aobt_lobt", "median"), med_taxi=("TAXITIME_SEC_mvt", "median"))
    W(C.md_table(t.rename_axis("LOBT - EOBT_1"), floatfmt="{:.1f}"))
    W("")
    W("## `AOBT_3 - LOBT` bands: how the block time moves off AOBT and onto LOBT\n")
    nm2 = tr25[tr25.has_nm].assign(
        band=pd.cut(tr25.loc[tr25.has_nm, "aobt_lobt"],
                    [-np.inf, -600, 0, 600, 1800, 3600, 7200, np.inf],
                    labels=["<-10m", "-10..0", "0..10m", "10..30m", "30..60m", "1..2h", ">2h"]))
    t = nm2.groupby("band", observed=True).agg(
        n=("blk_aobt", "size"), share_pct=("blk_aobt", lambda s: 100 * len(s) / len(nm2)),
        rmse_about_aobt=("blk_aobt", lambda s: float(np.sqrt(np.mean((s - s.median()) ** 2)))),
        rmse_about_lobt=("blk_lobt", lambda s: float(np.sqrt(np.mean((s - s.median()) ** 2)))),
        med_blk_aobt=("blk_aobt", "median"), med_blk_lobt=("blk_lobt", "median"),
        p_tail=("TAXITIME_SEC_mvt", lambda s: 100 * (s > 3600).mean()))
    W(C.md_table(t.rename_axis("AOBT_3 - LOBT"), floatfmt="{:.1f}"))
    W("")

    # ---------------- Q5 ----------------
    W("## Q5 — label quality on the 2025 training months\n")
    n = len(tr25)
    flags = {
        "taxi <= 0": tr25.TAXITIME_SEC_mvt <= 0,
        "taxi < 0 (block after take-off)": tr25.TAXITIME_SEC_mvt < 0,
        "taxi == 0 exactly": tr25.TAXITIME_SEC_mvt == 0,
        "taxi < 120 s": tr25.TAXITIME_SEC_mvt < 120,
        "taxi > 3600 s": tr25.TAXITIME_SEC_mvt > 3600,
        "taxi > 4 h": tr25.TAXITIME_SEC_mvt > 14400,
        "|BLOCK - SCHED| <= 1 s": tr25.blk_sched.abs() <= 1,
        "|BLOCK - SCHED| <= 60 s": tr25.blk_sched.abs() <= 60,
        "BLOCK == SCHED to the second": tr25.blk_sched == 0,
        "|BLOCK - LOBT| <= 1 s": tr25.blk_lobt.abs() <= 1,
        "|BLOCK - AOBT_3| <= 1 s": tr25.blk_aobt.abs() <= 1,
        "AIRCRAFT_TYPE_mvt null (noac)": tr25.AIRCRAFT_TYPE_mvt.isna(),
        "no NM join": ~tr25.has_nm,
        "STAND_mvt null/nan": tr25.stand.isin(["nan", "None", ""]),
    }
    rows = []
    for k, m in flags.items():
        y = tr25.loc[m, "TAXITIME_SEC_mvt"]
        rows.append({"flag": k, "n": int(m.sum()), "pct": 100 * m.mean(),
                     "median_taxi": float(y.median()) if len(y) else np.nan,
                     "mean_taxi": float(y.mean()) if len(y) else np.nan,
                     "p_gt3600_pct": 100 * float((y > 3600).mean()) if len(y) else np.nan,
                     "share_of_SSE_pct": 100 * float(((y - tr25.TAXITIME_SEC_mvt.mean()) ** 2).sum()
                                                     / ((tr25.TAXITIME_SEC_mvt - tr25.TAXITIME_SEC_mvt.mean()) ** 2).sum())})
    W(C.md_table(pd.DataFrame(rows).set_index("flag"), floatfmt="{:.3f}"))
    W("")

    # duplicates
    W("### Duplicates\n")
    key1 = ["ADEP_mvt", "MVT_TIME_UTC_mvt", "stand"]
    key2 = ["FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt"]
    key3 = ["CALLSIGN_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]
    for name, k in [("airport+take-off+stand", key1), ("FLIGHT_ID+take-off", key2),
                    ("callsign+take-off+airport", key3)]:
        dupm = tr25.duplicated(subset=k, keep=False) & tr25[k].notna().all(axis=1)
        sub = tr25[dupm]
        W(f"- **{name}**: {int(dupm.sum())} rows ({100*dupm.mean():.4f} %) in "
          f"{sub.groupby(k, observed=True).ngroups if len(sub) else 0} groups; "
          f"median taxi {sub.TAXITIME_SEC_mvt.median() if len(sub) else float('nan'):.0f} s, "
          f"P(taxi>3600) {100*(sub.TAXITIME_SEC_mvt>3600).mean() if len(sub) else float('nan'):.2f} %")
    W("")
    W("### Is the placeholder detectable from the ranking-frame columns?\n")
    ph = tr25.blk_sched.abs() <= 60
    feats = {"noac": tr25.AIRCRAFT_TYPE_mvt.isna(), "no NM join": ~tr25.has_nm,
             "MVT-SCHED > 2 h": tr25.mvt_sched > 7200,
             "MVT-SCHED > 4 h": tr25.mvt_sched > 14400,
             "noac AND MVT-SCHED > 2 h": tr25.AIRCRAFT_TYPE_mvt.isna() & tr25.mvt_sched.gt(7200),
             "airport is LIRF": tr25.apt == "LIRF"}
    rows = []
    for k, m in feats.items():
        rows.append({"screen": k, "n": int(m.sum()),
                     "P(placeholder|screen) %": 100 * float(ph[m].mean()) if m.any() else np.nan,
                     "recall of placeholders %": 100 * float((m & ph).sum() / ph.sum()),
                     "lift": float(ph[m].mean() / ph.mean()) if m.any() else np.nan})
    W(C.md_table(pd.DataFrame(rows).set_index("screen"), floatfmt="{:.3f}"))
    W(f"\nBase rate of `|BLOCK - SCHED| <= 60 s` over all 2025 departures: {100*ph.mean():.3f} %.\n")

    C.write("q2q5.md", "\n".join(out))


if __name__ == "__main__":
    main()
