"""Q4 supplement: per-airport shift between 2025 and the ranking months."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

M26 = ["2026-01", "2026-07"]


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    dep["apt_stand"] = dep["apt"].astype(str) + "|" + dep["stand_b"].astype(str)
    out = []; W = out.append
    W("# Q4 supplement: per-airport shift\n")

    W("## Mean take-off minus AOBT_3 by airport and month (s, NM rows)\n")
    piv = dep[dep.has_nm].pivot_table(index="apt", columns="month", values="nm_taxi",
                                      aggfunc="mean", observed=True)
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.0f}"))
    W("")
    W("## Std of take-off minus AOBT_3 by airport and month (s, NM rows)\n")
    piv = dep[dep.has_nm].pivot_table(index="apt", columns="month", values="nm_taxi",
                                      aggfunc="std", observed=True)
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.0f}"))
    W("")
    W("## Mean taxi time by airport and month (2025 only; the label is hidden in 2026)\n")
    piv = dep[~dep.is_ranking].pivot_table(index="apt", columns="month",
                                           values="TAXITIME_SEC_mvt", aggfunc="mean", observed=True)
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.0f}"))
    W("")

    W("## NM-join rate by airport and month (%)\n")
    piv = dep.pivot_table(index="apt", columns="month", values="has_nm", aggfunc="mean",
                          observed=True) * 100
    W(C.md_table(piv.rename_axis("airport"), floatfmt="{:.2f}"))
    W("")

    W("## Stand-mix distance (TVD) per airport: each 2025 month vs the ranking month\n")
    rows = []
    for a, ga in dep.groupby("apt", observed=True):
        r = {"airport": a}
        for m26 in M26:
            ref = ga.loc[ga.month == m26, "apt_stand"].value_counts(normalize=True)
            for m in d.MONTHS:
                s = ga.loc[ga.month == m, "apt_stand"].value_counts(normalize=True)
                idx = ref.index.union(s.index)
                r[f"{m26}|{m}"] = 0.5 * (ref.reindex(idx).fillna(0) - s.reindex(idx).fillna(0)).abs().sum()
        rows.append(r)
    T = pd.DataFrame(rows).set_index("airport")
    for m26 in M26:
        sub = T[[c for c in T.columns if c.startswith(m26)]].copy()
        sub.columns = [c.split("|")[1] for c in sub.columns]
        sub["closest"] = sub.idxmin(axis=1)
        W(f"### stand mix vs {m26}\n")
        W(C.md_table(sub.rename_axis("airport"), floatfmt="{:.3f}"))
        W("")

    W("## Runway mix: share of the airport's departures, 2025 total vs each ranking month\n")
    for a in sorted(dep.apt.unique()):
        ga = dep[dep.apt == a]
        s25 = ga.loc[~ga.is_ranking, "RUNWAY_mvt"].astype(str).value_counts(normalize=True) * 100
        s1 = ga.loc[ga.month == "2026-01", "RUNWAY_mvt"].astype(str).value_counts(normalize=True) * 100
        s7 = ga.loc[ga.month == "2026-07", "RUNWAY_mvt"].astype(str).value_counts(normalize=True) * 100
        t = pd.DataFrame({"2025": s25, "2026-01": s1, "2026-07": s7}).fillna(0)
        t = t[(t.max(axis=1) >= 1.0)].sort_values("2025", ascending=False)
        W(f"**{a}**  " + "; ".join(f"{i}: {r['2025']:.1f}/{r['2026-01']:.1f}/{r['2026-07']:.1f}"
                                   for i, r in t.iterrows()))
    W("")
    C.write("q4b_airport_shift.md", "\n".join(out))


if __name__ == "__main__":
    main()
