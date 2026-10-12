"""Q4: what changed between 2025 and the two ranking months."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d
from scipy.stats import wasserstein_distance

M25 = list(d.MONTHS)
M26 = ["2026-01", "2026-07"]


def tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two categorical share vectors."""
    idx = a.index.union(b.index)
    return float(0.5 * (a.reindex(idx).fillna(0) - b.reindex(idx).fillna(0)).abs().sum())


def shares(df, col):
    return df[col].astype(str).value_counts(normalize=True)


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    extra = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/mvt_all.parquet",
                            columns=["MVT_ID_mvt", "PHASE_mvt", "ADES_FILED_flt"])
    extra = extra[extra["PHASE_mvt"].astype(str) == "DEP"].drop(columns=["PHASE_mvt"])
    dep = dep.merge(extra, on="MVT_ID_mvt", how="left", validate="one_to_one")
    dep["stand"] = dep["stand"].astype(str)
    out = []
    W = out.append

    W("# The 2026 shift (Q4)\n")
    W("Departures only. 2025 = twelve training files; 2026-01 and 2026-07 = the live\n"
      "ranking file (344,841 departures, SHA-256 `9db75e3a…`), split by take-off time.\n")

    # ---- volumes and NM rate
    W("## Volume, NM-join rate, noac rate\n")
    t = dep.groupby("month", observed=True).agg(
        n=("MVT_ID_mvt", "size"),
        nm_rate=("has_nm", "mean"),
    )
    t["noac_rate_pct"] = dep.assign(noac=dep["AIRCRAFT_TYPE_mvt"].isna()).groupby(
        "month", observed=True)["noac"].mean() * 100
    t["nm_rate_pct"] = t.pop("nm_rate") * 100
    t["dep_per_day"] = t["n"] / dep.groupby("month", observed=True)["doy"].nunique()
    W(C.md_table(t[["n", "dep_per_day", "nm_rate_pct", "noac_rate_pct"]].rename_axis("month"),
                 floatfmt="{:.3f}"))
    W("")

    # ---- airport mix
    W("## Airport share of departures (%)\n")
    piv = (dep.pivot_table(index="apt", columns="month", values="MVT_ID_mvt",
                           aggfunc="size", observed=True)
           .pipe(lambda x: x / x.sum() * 100))
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.2f}"))
    W("")

    # ---- categorical drift: TVD of each 2025 month vs each ranking month
    W("## Distribution distance to the ranking months (total variation, 0 = identical)\n")
    cats = {"airport": "apt", "stand (apt|stand)": "apt_stand", "runway": None,
            "operator": "AIRCRAFT_OPERATOR_flt", "aircraft type": "AIRCRAFT_TYPE_mvt",
            "wake": "WK_TBL_CAT_flt", "segment": "MARKET_SEGMENT_flt",
            "destination": "ADES_mvt", "take-off hour": "hour"}
    dep["apt_runway"] = dep["apt"] + "|" + dep["RUNWAY_mvt"].astype(str)
    cats["runway"] = "apt_runway"
    rows = []
    ref = {m26: {k: shares(dep[dep.month == m26], c) for k, c in cats.items()} for m26 in M26}
    for m in M25:
        sub = dep[dep.month == m]
        r = {"month": m}
        for m26 in M26:
            for k, c in cats.items():
                r[f"{m26} {k}"] = tvd(shares(sub, c), ref[m26][k])
        rows.append(r)
    T = pd.DataFrame(rows).set_index("month")
    for m26 in M26:
        cols = [c for c in T.columns if c.startswith(m26)]
        sub = T[cols].copy()
        sub.columns = [c.replace(m26 + " ", "") for c in cols]
        sub["MEAN"] = sub.mean(axis=1)
        W(f"### against {m26}\n")
        W(C.md_table(sub.rename_axis("2025 month"), floatfmt="{:.4f}"))
        best = sub["MEAN"].nsmallest(3)
        W(f"\nclosest 2025 months by mean TVD: {', '.join(f'{i} ({v:.4f})' for i, v in best.items())}\n")

    # ---- numeric drift
    W("## Numeric distributions: Wasserstein distance (seconds) to the ranking months\n")
    nums = ["nm_taxi", "mvt_sched", "aobt_lobt", "aobt_sched", "lobt_sched"]
    rows = []
    ref_n = {m26: {c: dep.loc[(dep.month == m26) & dep.has_nm, c].dropna().to_numpy() for c in nums}
             for m26 in M26}
    for m in M25:
        sub = dep[(dep.month == m) & dep.has_nm]
        r = {"month": m}
        for m26 in M26:
            for c in nums:
                x = sub[c].dropna().to_numpy()
                r[f"{m26} {c}"] = wasserstein_distance(x, ref_n[m26][c])
        rows.append(r)
    T = pd.DataFrame(rows).set_index("month")
    for m26 in M26:
        cols = [c for c in T.columns if c.startswith(m26)]
        sub = T[cols].copy(); sub.columns = [c.replace(m26 + " ", "") for c in cols]
        W(f"### against {m26}\n")
        W(C.md_table(sub.rename_axis("2025 month"), floatfmt="{:.1f}"))
        W("")

    # ---- take-off minus AOBT summary per month
    W("## take-off minus AOBT_3 (`nm_taxi`) by month, NM rows\n")
    t = dep[dep.has_nm].groupby("month", observed=True)["nm_taxi"].agg(
        ["size", "mean", "median", "std",
         lambda s: np.percentile(s, 10), lambda s: np.percentile(s, 90)])
    t.columns = ["n", "mean", "median", "std", "p10", "p90"]
    W(C.md_table(t.rename_axis("month"), floatfmt="{:.1f}"))
    W("")

    # ---- hour profile
    W("## Take-off hour profile (% of the month's departures)\n")
    piv = (dep.pivot_table(index="hour", columns="month", values="MVT_ID_mvt",
                           aggfunc="size", observed=True).fillna(0))
    piv = piv / piv.sum() * 100
    W(C.md_table(piv.rename_axis("hour"), floatfmt="{:.2f}"))
    W("")

    # ---- unseen levels
    W("## Category levels present in the ranking months and absent from 2025\n")
    # every categorical any exp-003 model uses, at the exact key the model uses
    dep["stand_raw"] = dep["stand"].astype(str)
    dep["key_raw"] = dep["apt"] + "|" + dep["stand_raw"]
    dep["CALLSIGN_PFX"] = dep["CALLSIGN_flt"].astype("string").str[:3].astype(str)
    inventory = ["apt", "key_raw", "apt_stand", "stand_raw", "stand_b", "stand_pfx",
                 "apt_pfx", "apt_runway", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                 "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
                 "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "ADES_mvt", "ADES_FILED_flt",
                 "CALLSIGN_PFX"]
    seen_all = {c: set(dep.loc[dep.month.isin(M25), c].astype(str)) for c in inventory}
    lines = []
    for m26 in M26 + ["2026 both"]:
        sub = dep[dep.month.isin(M26)] if m26 == "2026 both" else dep[dep.month == m26]
        for c, seen in seen_all.items():
            v = sub[c].astype(str)
            new = ~v.isin(seen)
            lines.append({"frame": m26, "column": c, "new_levels": int(v[new].nunique()),
                          "rows": int(new.sum()), "row_pct": 100 * new.mean()})
    W(C.md_table(pd.DataFrame(lines).set_index("frame"), floatfmt="{:.4f}"))
    W("")
    W("### the new levels themselves (2026 rows, both months)\n")
    W("The assignment asks for *every* level and its row count, so the complete "
      "lists are written to `q4_new_levels.md` next to this file; the head of each "
      "is repeated here.\n")
    sub = dep[dep.month.isin(M26)]
    full = ["# Every category level present in the ranking months and absent from 2025\n",
            "One section per column; every level, with the number of ranking rows that "
            "carry it, and the split between the two ranking months. Generated by "
            "`experiments/exp-003-anatomy/code/q4_shift.py`.\n"]
    for c in seen_all:
        v = sub[c].astype(str)
        new = v[~v.isin(seen_all[c])]
        vc = new.value_counts()
        W(f"- **{c}**: {len(vc)} new levels on {len(new)} rows. "
          f"Top: {', '.join(f'{k} ({n})' for k, n in vc.head(15).items()) if len(vc) else 'none'}")
        full.append(f"\n## {c} — {len(vc)} new levels, {len(new)} rows\n")
        if not len(vc):
            full.append("(none)\n")
            continue
        jan = sub.loc[sub.month == "2026-01", c].astype(str).value_counts()
        jul = sub.loc[sub.month == "2026-07", c].astype(str).value_counts()
        full.append("| level | rows | 2026-01 | 2026-07 |")
        full.append("|---|---|---|---|")
        for k, n in vc.items():
            full.append(f"| `{k}` | {n} | {int(jan.get(k, 0))} | {int(jul.get(k, 0))} |")
    C.write("q4_new_levels.md", "\n".join(full))
    W("")
    W("### 2025 levels that vanished in 2026 (retired stands/runways with >=200 2025 rows)\n")
    for c in ["apt_stand", "apt_runway"]:
        v25 = dep.loc[dep.month.isin(M25), c].astype(str).value_counts()
        s26 = set(sub[c].astype(str))
        gone = v25[(~v25.index.isin(s26)) & (v25 >= 200)]
        W(f"- **{c}**: {len(gone)} retired levels. "
          f"Top: {', '.join(f'{k} ({n})' for k, n in gone.head(15).items()) if len(gone) else 'none'}")
    # ---- every numeric feature the models propose, 2025 against each ranking month
    W("## Every numeric model feature: 2025 against the two ranking months\n")
    W("Mean and standard deviation of every numeric column of the model feature frame "
      "(`artifacts/exp-003-anatomy/feat/`), and the shift in units of the 2025 standard "
      "deviation. Anything past |0.2| is worth a look.\n")
    from features import ART, BASE_NUM, OCC, NBR, CSN
    num = [c for c in BASE_NUM + OCC + [c + "_b" for c in OCC] + NBR + CSN]
    tr = pd.concat([pd.read_parquet(ART / f"{m}.parquet", columns=num) for m in d.MONTHS],
                   ignore_index=True)
    rk = pd.read_parquet(ART / "ranking.parquet", columns=num + ["MVT_TIME_UTC_mvt"])
    rk["month"] = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    rows = []
    for c in num:
        m25, s25 = tr[c].mean(), tr[c].std()
        r = {"feature": c, "2025 mean": m25, "2025 sd": s25,
             "null % 2025": 100 * tr[c].isna().mean()}
        for m26 in M26:
            v = rk.loc[rk.month == m26, c]
            r[f"{m26} mean"] = v.mean()
            r[f"{m26} shift/sd"] = (v.mean() - m25) / s25 if s25 and s25 == s25 else np.nan
            r[f"null % {m26}"] = 100 * v.isna().mean()
        rows.append(r)
    T = pd.DataFrame(rows).set_index("feature")
    T = T.reindex(T[[f"{m}  shift/sd".replace("  ", " ") for m in M26]].abs().max(axis=1)
                  .sort_values(ascending=False).index)
    W(C.md_table(T, floatfmt="{:.3f}"))
    W("")
    C.write("q4_shift.md", "\n".join(out))


if __name__ == "__main__":
    main()
