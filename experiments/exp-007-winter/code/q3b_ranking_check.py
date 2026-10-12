"""Does the day-state block actually fire on January 2026, and is it clean there?

The case for shipping these columns is a ranking month no fold can price
(`docs/shift-2026.md`: 27 airport-days above the entire 2025 range). This file
transforms the live ranking file with a DayState fitted on all twelve 2025 months
and asks (a) whether every ranking departure gets a value, (b) how far January
2026's days sit above the 2025 reference, and (c) whether July 2026 looks
ordinary, which is the control.

Features are built one ranking month at a time: the file holds January and July
2026 as isolated months and nothing may join across the gap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import prc.data as D  # noqa: E402
import prc.day_state_features as DS  # noqa: E402

OUT = HERE.parent / "analysis" / "q3b_ranking.md"
Z = ("ds_sched_med_z", "ds_gap_p90_z", "ds_span_med_z", "ds_span_std_z",
     "ds_share_span1800_z", "ds_nm_rate_z", "ds_taxiin_p90_z")


def main():
    train = D.load_months(D.MONTHS)
    ds = DS.DayState().fit(train)
    rk = D.load_ranking()
    parts = []
    for m in sorted(rk["month"].unique()):
        sub = rk[rk["month"] == m]
        F = ds.transform(sub)
        F = F.assign(month=m, apt=sub.loc[F.index, "ADEP_mvt"].astype(str).to_numpy(),
                     day=sub.loc[F.index, "MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy())
        parts.append(F)
    R = pd.concat(parts)
    R.to_parquet(HERE.parent / "work" / "ranking_daystate.parquet")

    P = ["# exp-007 — the day-state block on the live ranking file\n",
         "`DayState` fitted on all twelve 2025 months, transformed one ranking month at a time.\n",
         "## Coverage\n",
         f"{len(R):,} ranking departures (expected 344,841: "
         f"{'match' if len(R) == 344841 else 'MISMATCH'}).\n",
         "| column | non-null | min | median | max |", "|---|---|---|---|---|"]
    for c in DS.COLUMNS:
        s = R[c]
        P.append(f"| `{c}` | {100*s.notna().mean():.2f} % | {s.min():,.3f} | {s.median():,.3f} | {s.max():,.3f} |")
    P.append("")
    P.append("## How far the ranking days sit above the 2025 reference\n")
    P.append("One row per airport-day. `z` is (the day's statistic − the airport's 2025 median day) "
             "divided by the airport's 2025 interquartile range of that statistic, so z = 1 means "
             "the day is one 2025 IQR above a normal day at that airport.\n")
    days = R.groupby(["month", "apt", "day"])[list(Z)].first().reset_index()
    P.append("| month | airport-days | " + " | ".join(f"median `{c}`" for c in Z) + " |")
    P.append("|---" * (2 + len(Z)) + "|")
    for m, g in days.groupby("month"):
        P.append(f"| {m} | {len(g)} | " + " | ".join(f"{g[c].median():+.2f}" for c in Z) + " |")
    P.append("")
    P.append("| month | airport-days with `ds_span_std_z` > 2 | > 3 | with `ds_share_span1800_z` > 2 | "
             "with `ds_nm_rate_z` < −3 |")
    P.append("|---|---|---|---|---|")
    for m, g in days.groupby("month"):
        P.append(f"| {m} | {int((g['ds_span_std_z'] > 2).sum())} | {int((g['ds_span_std_z'] > 3).sum())} | "
                 f"{int((g['ds_share_span1800_z'] > 2).sum())} | {int((g['ds_nm_rate_z'] < -3).sum())} |")
    P.append("")
    P.append("## The January 2026 days the block flags hardest\n")
    j = days[days["month"] == "2026-01"].sort_values("ds_span_std_z", ascending=False).head(15)
    P.append("| airport | day | " + " | ".join(f"`{c}`" for c in Z) + " |")
    P.append("|---" * (2 + len(Z)) + "|")
    for _, x in j.iterrows():
        P.append(f"| {x['apt']} | {x['day']:%Y-%m-%d} | " + " | ".join(f"{x[c]:+.2f}" for c in Z) + " |")
    OUT.write_text("\n".join(P) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
