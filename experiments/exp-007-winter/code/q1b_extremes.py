"""How common is a label like LFPG 2025-01-19's 84,240 s, and what marks it?

Scans all twelve 2025 training months for departures whose recorded taxi time
is physically impossible-to-implausible, and asks (a) what their other columns
look like, (b) how much of a month's SSE budget one of them is worth, (c)
whether anything in the *masked* frame separates them.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import prc.data as D  # noqa: E402

CUTS = (3600, 7200, 10800, 21600, 43200)
rows = []
detail = []
for m in D.MONTHS:
    df = D.load_month(m)
    dep = df[(df["PHASE_mvt"] == "DEP") & df[D.TARGET].notna()].copy()
    dep["apt"] = dep["ADEP_mvt"].astype(str)
    y = dep[D.TARGET].astype(float)
    no_nm = dep["FLIGHT_ID_mvt"].isna()
    noac = dep["AIRCRAFT_TYPE_mvt"].isna()
    rec = {"month": m, "n": len(dep), "n_nonlirf": int((dep["apt"] != "LIRF").sum())}
    for c in CUTS:
        rec[f"n>{c}"] = int((y > c).sum())
        rec[f"n>{c}_nonlirf"] = int(((y > c) & (dep["apt"] != "LIRF")).sum())
    rec["max"] = float(y.max())
    rows.append(rec)
    big = dep[y > 21600]
    for _, r in big.iterrows():
        detail.append({
            "month": m, "apt": r["apt"], "flight": r["FLIGHT_mvt"],
            "y": float(r[D.TARGET]),
            "mvt": r["MVT_TIME_UTC_mvt"], "block": r["BLOCK_TIME_UTC_mvt"],
            "sched": r["SCHED_TIME_UTC_mvt"], "lobt": r["LOBT_flt"], "aobt": r["AOBT_3_flt"],
            "no_nm": bool(pd.isna(r["FLIGHT_ID_mvt"])), "noac": bool(pd.isna(r["AIRCRAFT_TYPE_mvt"])),
            "mvt_minus_sched": (r["MVT_TIME_UTC_mvt"] - r["SCHED_TIME_UTC_mvt"]).total_seconds(),
            "block_minus_sched": (r["BLOCK_TIME_UTC_mvt"] - r["SCHED_TIME_UTC_mvt"]).total_seconds(),
            "stand": r["STAND_mvt"], "rwy": r["RUNWAY_mvt"], "ades": r["ADES_mvt"],
        })
    print(m, rec["n"], {k: v for k, v in rec.items() if k.startswith("n>")}, flush=True)

S = pd.DataFrame(rows)
T = pd.DataFrame(detail)
S.to_csv(HERE.parent / "work" / "extremes_by_month.csv", index=False)
T.to_csv(HERE.parent / "work" / "extremes_detail.csv", index=False)
print(S.to_string())
print(T.to_string())
