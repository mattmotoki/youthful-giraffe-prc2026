"""Is the EJU rule a detector, or a rule fitted to two rows?

gpt-5.6-sol round 1, blocking finding 1: my claim that no observable column
separates the two unreachable rows is false. The conjunction

    ADEP = LFPG  and  no NM join  and  aircraft type present
    and  FLIGHT_mvt starts with 'EJU'  and  |take-off - schedule| <= 1 h

selects exactly `EJU983W` and `EJU42AY` among January 2025's LFPG no-NM rows.
That disproves "by construction unreachable". The desk's question is the next
one: a conjunction that isolates two rows post hoc is not yet a detector, so
measure its precision out of sample and price the hedge it would license.

The hedge: on a firing row, add `p * R` seconds, where `p` is the rule's
precision estimated on the fold's TRAINING months and `R` is the mean residual of
an event. That is the RMSE-optimal additive correction under the rule's own
model. A firing row that is not an event costs `(p*R)^2`; one that is recovers
`2*p*R*residual - (p*R)^2`.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import common as C  # noqa: E402
import q1c_trimmed as q1c  # noqa: E402

OUT = HERE.parent / "analysis" / "q1_detector.md"
EVENT_CUT = 21600.0


def fires(d: pd.DataFrame) -> pd.Series:
    ms = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return (d["ADEP_mvt"].eq("LFPG") & d["FLIGHT_ID_mvt"].isna()
            & d["AIRCRAFT_TYPE_mvt"].notna()
            & d["FLIGHT_mvt"].astype(str).str.startswith("EJU") & ms.abs().le(3600))


def month_table() -> pd.DataFrame:
    rows = []
    for m in D.MONTHS:
        d = D.load_month(m)
        d = d[d["PHASE_mvt"].eq("DEP") & d[D.TARGET].notna()]
        f = fires(d)
        ev = f & d[D.TARGET].astype(float).gt(EVENT_CUT)
        rows.append({"month": m, "fires": int(f.sum()), "events": int(ev.sum()),
                     "resid_sum": float(d.loc[ev, D.TARGET].astype(float).sum())})
    return pd.DataFrame(rows).set_index("month")


def main():
    T = month_table()
    P = ["# exp-007 — the EJU rule priced as a detector\n", __doc__.strip().split("\n\n", 1)[1], "",
         "## Where the rule fires\n",
         "| month | rows it fires on | of those, rollover events |", "|---|---|---|"]
    for m, r in T.iterrows():
        if r["fires"]:
            P.append(f"| {m} | {int(r['fires'])} | {int(r['events'])} |")
    P.append(f"| **2025 total** | **{int(T['fires'].sum())}** | **{int(T['events'].sum())}** |")
    P.append("")
    rk = D.load_ranking()
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    fr = fires(rk)
    P.append(f"In the live ranking file the rule fires on **{int(fr.sum())}** rows, "
             + ", ".join(f"{n} in {m}" for m, n in rk.loc[fr, "month"].value_counts().items())
             + ". Their labels are hidden, so the rule cannot be checked there.\n")
    P.append("## Precision, in sample and out\n")
    ins = T["events"].sum() / T["fires"].sum()
    oos = T.drop("2025-01")["events"].sum() / T.drop("2025-01")["fires"].sum()
    P.append(f"The rule was found *on* January 2025's two events, so its all-2025 precision of "
             f"{T['events'].sum()}/{T['fires'].sum()} = **{100*ins:.1f} %** is in-sample. Removing the "
             f"month it was found on leaves {T.drop('2025-01')['events'].sum()}/"
             f"{int(T.drop('2025-01')['fires'].sum())} = **{100*oos:.1f} %**: "
             f"**out of sample the rule has fired {int(T.drop('2025-01')['fires'].sum())} times and been "
             f"right {int(T.drop('2025-01')['events'].sum())} times.**\n")
    P.append("## What the hedge it licenses does to each fold\n")
    P.append("`p` is estimated on the fold's own training months, `R` is the mean residual of an "
             "event under the baseline line (83,335 and 57,070 s, mean 70,203). Both are then "
             "applied to the validation month, which is the only honest order.\n")
    P.append("| fold | training fires / events | p | hedge p·R | validation fires | of those events | "
             "RMSE before | after | Δ |")
    P.append("|---|---|---|---|---|---|---|---|---|")
    R_MEAN = 70203.0
    for fold in ("july", "december", "january"):
        f = D.FOLDS[fold]
        tr = T.loc[list(f.train_months)]
        p = tr["events"].sum() / max(tr["fires"].sum(), 1)
        hedge = p * R_MEAN
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        full = D.load_month(f.valid_month)
        full = full[full["PHASE_mvt"].eq("DEP") & full[D.TARGET].notna()]
        fire_ids = set(full.loc[fires(full), D.ID])
        m = rows[D.ID].isin(fire_ids).to_numpy()
        ev = m & rows["y"].gt(EVENT_CUT).to_numpy()
        pred2 = rows["pred"].to_numpy() + m * hedge
        before, after = C.rmse(rows["err"]), C.rmse(rows["y"].to_numpy() - pred2)
        P.append(f"| {fold} | {int(tr['fires'].sum())} / {int(tr['events'].sum())} | {p:.3f} | "
                 f"{hedge:,.0f} s | {int(m.sum())} | {int(ev.sum())} | {before:.2f} | {after:.2f} | "
                 f"**{after-before:+.2f} s** |")
    P.append("")
    P.append("And the same with the two unreachable rows removed from the validation month, i.e. "
             "the hedge's cost when it is right about nothing:\n")
    P.append("| fold | validation fires (clean) | RMSE before | after | Δ |")
    P.append("|---|---|---|---|---|")
    for fold in ("july", "december", "january"):
        f = D.FOLDS[fold]
        tr = T.loc[list(f.train_months)]
        hedge = (tr["events"].sum() / max(tr["fires"].sum(), 1)) * R_MEAN
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        rows = rows[~q1c.rollover_mask(rows)]
        full = D.load_month(f.valid_month)
        full = full[full["PHASE_mvt"].eq("DEP") & full[D.TARGET].notna()]
        fire_ids = set(full.loc[fires(full), D.ID])
        m = rows[D.ID].isin(fire_ids).to_numpy()
        pred2 = rows["pred"].to_numpy() + m * hedge
        before, after = C.rmse(rows["err"]), C.rmse(rows["y"].to_numpy() - pred2)
        P.append(f"| {fold} | {int(m.sum())} | {before:.2f} | {after:.2f} | **{after-before:+.2f} s** |")
    P.append("")
    OUT.write_text("\n".join(P))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
