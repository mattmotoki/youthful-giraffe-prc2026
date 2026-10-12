"""Census of date-rollover rows in the 2025 months, split by whether any input can reach them.

A rollover row is one whose recorded block time sits on an earlier calendar day
than it should, so the label `MVT − BLOCK` is about 86,400 s plus an ordinary
taxi. The class matters far past its size: `exp-007 rollover` measured **two**
such rows carrying 55.3 % of the whole january fold's squared error.

The split this file makes is the one that decides whether a fold number means
anything:

  DETECTABLE    some input carries the same shift -- `MVT − SCHED`, `MVT − AOBT_3`
                or `MVT − LOBT` is itself about a day -- so a model can in
                principle route the row. `prc.tail_model` 2.2.0's noac head v4
                does exactly this above 8 h of `MVT − SCHED` (`exp-009 final`).
  UNREACHED     the label is a day out and every one of those inputs is
                ordinary. No feature built from the timing columns can separate
                the row from a normal departure.

**"Unreached", not "undetectable", and the distinction is exp-007's own.** Its
review round 1 corrected the claim that no observable separates the two january
rows: the `FLIGHT_mvt` prefix `EJU` crossed with |take-off − schedule| <= 1 h
selects exactly them among LFPG's no-NM january rows. That is a real separator on
two rows found after the fact, which is not the same as a feature that would
generalise, so this census reports the class as unreached and says what would be
needed to call it unreachable.

No fits. Reads the twelve labelled 2025 months.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = EXP / "analysis"

#: The class needs BOTH conditions and two earlier versions of this file got it
#: wrong in opposite directions.
#:
#:   * A cut on the label alone (40,000 s) has no gap to sit in -- 16 rows land in
#:     40-50 ks, 7 in 50-60 ks, 2 in 60-70 ks, 0 in 70-80 ks, 16 above 80 ks -- so
#:     it either splits the class or absorbs long genuine taxis.
#:   * The date lag alone catches **1,696** rows, 90 % of which are ordinary
#:     departures that pushed back before midnight and took off after it. Their
#:     median label is 1,253 s. They are not rollovers.
#:
#: So: a rollover row has the block time on an EARLIER CALENDAR DAY **and** an
#: abnormal label. The joint table (`analysis/rollover_joint.csv`) shows the two
#: populations separating cleanly at 10,000 s: at lag 1 there are 1,531 rows at or
#: below 3,600 s, 105 between 3.6 and 10 ks, then 24 / 20 / 16 in the bands above.
#: All 16 rows above 80,000 s have lag 1, which reproduces `exp-002 rollover`'s
#: "16 rows a year" exactly.
DATE_LAG_MIN = 1
ABNORMAL = 10_000.0
#: Sensitivity: the same census at these cuts is reported so the choice is visible.
ABNORMAL_ALT = (40_000.0, 80_000.0)

OBSERVABLES = {
    "mvt_sched": ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
    "mvt_aobt": ("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
    "mvt_lobt": ("MVT_TIME_UTC_mvt", "LOBT_flt"),
    "mvt_iobt": ("MVT_TIME_UTC_mvt", "IOBT_flt"),
    "mvt_eobt1": ("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
}


def month_frame(m: str) -> pd.DataFrame:
    d = D.load_month(m)
    d = d[(d["PHASE_mvt"] == "DEP") & d[D.TARGET].notna()].copy()
    d["y"] = d[D.TARGET].astype("float64")
    d["airport"] = d["ADEP_mvt"].astype(str)
    d["month"] = m
    d["regime"] = B.regimes_from_raw(d)
    for k, (a, b) in OBSERVABLES.items():
        d[k] = (d[a] - d[b]).dt.total_seconds()
    d["block_date_lag_days"] = (d["MVT_TIME_UTC_mvt"].dt.floor("D")
                                - d["BLOCK_TIME_UTC_mvt"].dt.floor("D")).dt.days
    return d


def main() -> None:
    frames = {m: month_frame(m) for m in D.MONTHS}
    allrows = pd.concat(frames.values(), ignore_index=True)
    n = len(allrows)
    print(f"labelled 2025 departures: {n:,}", flush=True)

    # ---- the threshold is not a free parameter: show the gap --------------
    y = np.sort(allrows["y"].to_numpy("float64"))
    band = [(lo, int(((y >= lo) & (y < hi)).sum())) for lo, hi in
            [(10_000, 20_000), (20_000, 30_000), (30_000, 40_000), (40_000, 50_000),
             (50_000, 60_000), (60_000, 70_000), (70_000, 80_000), (80_000, 90_000),
             (90_000, 1e9)]]
    pd.DataFrame(band, columns=["lo", "n"]).to_csv(OUT / "rollover_threshold.csv", index=False)
    print("\n=== the label above 10,000 s, by band (the threshold sits in the gap) ===")
    for lo, c in band:
        print(f"  {lo:>7,.0f} - {lo+10_000 if lo < 90_000 else 999_999:>7,.0f}  {c:>4}")

    # ---- the two populations, shown rather than asserted --------------------
    bins = [-1e9, 3600, 10_000, 40_000, 70_000, 80_000, 1e9]
    labs = ["<=3600", "3.6-10k", "10-40k", "40-70k", "70-80k", ">80k"]
    J = pd.crosstab(allrows["block_date_lag_days"],
                    pd.cut(allrows["y"], bins, labels=labs))
    J.to_csv(OUT / "rollover_joint.csv")
    print("\n=== date lag (rows) x label band (cols) ===")
    print(J.to_string())

    def classify(cut: float) -> pd.DataFrame:
        r = allrows[(allrows["block_date_lag_days"] >= DATE_LAG_MIN)
                    & (allrows["y"] > cut)].copy()
        obs = r[list(OBSERVABLES)].to_numpy("float64")
        # An input CARRIES the shift when the input is ITSELF abnormal -- the day
        # is visible in it. An earlier version asked whether the input was merely
        # close to the label, which is true of every ordinary row and made the
        # split meaningless.
        carries = np.where(np.isnan(obs), False, obs > cut)
        r["n_inputs_carrying"] = carries.sum(axis=1)
        r["reachable"] = r["n_inputs_carrying"] > 0
        r["n_inputs_present"] = np.isfinite(obs).sum(axis=1)
        r["carried_by"] = [",".join([k for j, k in enumerate(OBSERVABLES) if carries[i, j]]) or "-"
                           for i in range(len(r))]
        return r

    print("\n=== sensitivity of the split to the abnormal-label cut ===")
    for cut in (ABNORMAL,) + ABNORMAL_ALT:
        r = classify(cut)
        print(f"  cut {cut:>8,.0f} s: {len(r):>3} rollover rows, "
              f"{int(r['reachable'].sum()):>3} detectable, "
              f"{int((~r['reachable']).sum()):>3} unreached")

    roll = classify(ABNORMAL)
    cols = ["MVT_ID_mvt", "month", "airport", "regime", "y", "block_date_lag_days", "reachable",
            "n_inputs_carrying", "carried_by", "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_TYPE_mvt"] + list(OBSERVABLES)
    roll[cols].sort_values(["reachable", "month"]).to_csv(OUT / "rollover_census.csv", index=False)

    print(f"\n=== the {len(roll)} date-lagged rows, split ===")
    print(f"  DETECTABLE (an input carries the same shift): {int(roll['reachable'].sum())}")
    print(f"  UNREACHED  (every input ordinary):            {int((~roll['reachable']).sum())}")
    print("\nby regime:")
    print(pd.crosstab(roll["regime"], roll["reachable"]).to_string())

    un = roll[~roll["reachable"]]
    print(f"\n=== the {len(un)} unreached rows, in full ===")
    show = ["MVT_ID_mvt", "month", "airport", "regime", "y", "block_date_lag_days",
            "FLIGHT_mvt", "mvt_sched", "mvt_aobt", "mvt_lobt"]
    print(un[show].round(0).to_string(index=False), flush=True)

    print("\n=== the detectable ones, by month and what carries them ===")
    de = roll[roll["reachable"]]
    print(de.groupby(["month", "carried_by"]).agg(
        n=("y", "size"), y_med=("y", "median")).round(0).to_string())

    # ---- rate, for the Poisson arithmetic in docs/board-reading.md --------
    per_month = roll.groupby("month")["reachable"].agg(
        n_total="size", n_unreached=lambda s: int((~s).sum())).reindex(D.MONTHS).fillna(0)
    per_month.to_csv(OUT / "rollover_per_month.csv")
    per_apt = un.groupby("airport").size().rename("n_unreached")
    per_apt.to_csv(OUT / "rollover_per_airport.csv")
    print("\n=== per month ===")
    print(per_month.astype(int).to_string())
    print("\n=== unreached, per airport ===")
    print(per_apt.to_string() if len(per_apt) else "  (none)")

    rate = len(un) / n
    print(f"\nunreached rate over 2025: {len(un)} / {n:,} = {rate:.3e} per departure")
    for label, nn in (("the 344,841 ranking departures", 344_841),
                      ("January 2026 (152,719)", 152_719),
                      ("July 2026 (192,122)", 192_122)):
        print(f"  Poisson mean in {label}: {rate*nn:.3f}")


if __name__ == "__main__":
    main()
