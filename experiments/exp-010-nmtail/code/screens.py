"""The screen table in RESULTS.md Q2, as a committed script (review round 1, N6)."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")   # for `features`
sys.path.insert(0, "experiments/exp-010-nmtail/code")    # must win for `common`
import common as C
import prc.data as d
from run_folds import frame

MIN_GAP = 600.0   # strictly greater than: AOBT_3 - LOBT > 600 s


def main():
    f = pd.concat([frame(m) for m in d.MONTHS], ignore_index=True)
    y = f.TAXITIME_SEC_mvt.astype(float)
    f["true_phi"] = np.where(f.gap.abs() > 1e-9, (f.nm_taxi - y) / f.gap, np.nan)
    f["lobt_true"] = np.abs(y - f.mvt_lobt) < np.abs(y - f.nm_taxi)
    f["apt_hour"] = f.apt.astype(str) + "|" + f.MVT_TIME_UTC_mvt.dt.floor("h").astype(str)
    g = f.groupby("apt_hour", observed=True)
    f["lagfrac"] = ((g["gap"].transform(lambda s: (s < -1800).sum()) - (f.gap < -1800))
                    / (g["gap"].transform("size") - 1)).astype(float)
    sub = f[f.has_nm & (f.gap < -MIN_GAP)]
    rows = []

    def add(name, mask):
        s = sub[mask]
        rows.append({"screen": name, "rows": len(s), "share %": 100 * len(s) / len(sub),
                     "P(block-on-LOBT) %": 100 * s.lobt_true.mean(),
                     "P(taxi>3600) %": 100 * (s.TAXITIME_SEC_mvt > 3600).mean(),
                     "mean phi": float(np.nanmean(s.true_phi))})

    add("(base: all rows with AOBT_3 - LOBT > 600 s)", np.ones(len(sub), bool))
    add("an arrival parked in (LOBT, AOBT)", sub.reocc_lobt_aobt >= 1)
    add("... of a different aircraft type", sub.reocc_lobt_aobt_diff >= 1)
    add("... of the same type only", (sub.reocc_lobt_aobt >= 1) & (sub.reocc_lobt_aobt_diff == 0))
    add("AOBT - LOBT > 1 h", sub.gap < -3600)
    add("both: > 1 h and a different-type re-occupation",
        (sub.gap < -3600) & (sub.reocc_lobt_aobt_diff >= 1))
    add("LOBT within 60 s of SCHED", sub.lobt_sched.abs() <= 60)
    for thr in (0.05, 0.10, 0.20):
        add(f"airport-hour NM-lag fraction > {thr}", sub.lagfrac > thr)
    C.write("q2_screens.md",
            "# Screens for block-on-LOBT (all twelve 2025 months)\n\n"
            f"Population: NM-joined departures with `AOBT_3 - LOBT > {MIN_GAP:.0f} s` "
            f"strictly, {len(sub):,} rows.\n\n"
            + C.md_table(pd.DataFrame(rows).set_index("screen"), floatfmt="{:.2f}") + "\n")


if __name__ == "__main__":
    main()
