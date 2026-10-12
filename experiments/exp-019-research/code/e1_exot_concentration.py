"""E1: is `MVT − AOBT_3` an EXOT lookup, or a taxi time against a rounded clock?

The tail worker refuted the first derivation on seconds: `AOBT_3` is on `:00`
98.45 % of the time, so it cannot be the airport's second-resolution take-off
minus a whole-minute EXOT -- that would inherit the take-off's seconds. The
reconciliation to test is that ETFMS uses its OWN minute-resolution take-off
(radar / M3 ATOT):

    AOBT_3 = ATOT_nm(minute)  −  EXOT(minute)          -> lands on :00
    MVT − AOBT_3 = EXOT + (MVT − ATOT_nm),  |MVT − ATOT_nm| < 1 min

Under that hypothesis `round(MVT − AOBT_3)` to the minute is **the airport's EXOT
lookup value**, which is a table: few distinct values per (stand, runway) cell,
because §4.7.21.1(3) of the DPI guide says the taxi-time "shall at least depend
on gate/stand/parking position and departure runway" and §4.7.21.2(2) says it is
only resent when it moves by 5 minutes or more.

Under the competing hypothesis -- `AOBT_3` is simply a minute-rounded REPORTED
off-block -- `MVT − AOBT_3` is the true taxi time plus rounding, and it should be
no more concentrated than the true taxi time itself.

**So the control is the discriminator, not the concentration.** The same
statistics are computed on `TAXITIME_SEC_mvt` over the same cells and the same
rows. A lookup table is much more concentrated than the physical quantity it
estimates; a rounded clock is not.

Gate, frozen in docs/external-sweep-2026-09-05.md before any number was read:
at least one airport shows >= 30 % of low-delay rows on <= 10 distinct
whole-minute values per (stand-prefix, runway) cell.

No fits. Reads the twelve labelled 2025 months, because the control needs labels.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

OUT = EXP / "analysis"
#: `prc.stand_features` convention: leading letters, or a single leading digit.
_MARS = re.compile(r"(?<=\d)[LRC]$|(?<=[A-Za-z]\d)[LRC]$")
_PFX = re.compile(r"^([A-Za-z]+|[0-9])")
LOW_DELAY = 300.0     # |AOBT - LOBT| <= 5 min: the standard, un-disrupted case
MIN_CELL = 50         # rows per (airport, stand-prefix, runway) cell


def load() -> pd.DataFrame:
    cols = ["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "TAXITIME_SEC_mvt"]
    d = pd.concat([D.load_month(m, columns=cols) for m in D.MONTHS], ignore_index=True)
    d = d[(d["PHASE_mvt"] == "DEP") & d["AOBT_3_flt"].notna()
          & d["TAXITIME_SEC_mvt"].notna()].copy()
    d["apt"] = d["ADEP_mvt"].astype(str)
    st = d["STAND_mvt"].astype("string").str.strip()
    base = st.str.replace(_MARS, "", regex=True)
    d["pfx"] = base.str.extract(_PFX, expand=False).fillna("?")
    d["rwy"] = d["RUNWAY_mvt"].astype(str)
    d["gap"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d["aobt_lobt"] = (d["AOBT_3_flt"] - d["LOBT_flt"]).dt.total_seconds()
    d["taxi"] = d["TAXITIME_SEC_mvt"].astype("float64")
    return d


def stats(v: np.ndarray) -> tuple[int, float, float]:
    """(distinct whole-minute values, share on the mode, share on the top 3)."""
    m = np.round(v / 60.0).astype("int64")
    vc = pd.Series(m).value_counts()
    n = len(m)
    return len(vc), float(vc.iloc[0]) / n, float(vc.iloc[:3].sum()) / n


def main() -> None:
    d = load()
    low = d[np.abs(d["aobt_lobt"]) <= LOW_DELAY]
    print(f"labelled 2025 departures with an NM join: {len(d):,}", flush=True)
    print(f"low-delay (|AOBT-LOBT| <= {LOW_DELAY:.0f} s): {len(low):,} "
          f"({100*len(low)/len(d):.1f} %)", flush=True)

    # ---- the seconds check the tail worker's argument turns on ------------
    print("\n=== does MVT - AOBT_3 inherit MVT's seconds? ===")
    for lab, s in (("AOBT_3_flt on :00", (d["AOBT_3_flt"].dt.second == 0)),
                   ("MVT on :00", (d["MVT_TIME_UTC_mvt"].dt.second == 0)),
                   ("BLOCK on :00", (d["BLOCK_TIME_UTC_mvt"].dt.second == 0))):
        print(f"  {lab:22s} {s.mean():.4f}")
    sec_gap = (d["gap"].to_numpy() % 60).astype("int64")
    h = np.bincount(sec_gap, minlength=60)
    print("  (MVT - AOBT_3) mod 60, share in each 10 s band:")
    for i in range(0, 60, 10):
        print(f"    {i:>2}-{i+9:<2}s  {h[i:i+10].sum()/len(sec_gap):.4f}")
    print("  a rounded REPORTED off-block predicts uniform (0.1667 per band);")
    print("  an NM take-off that TRUNCATES to the minute predicts the same, so this")
    print("  separates only if NM rounds. Reported for completeness, not as the test.")

    # ---- the cell concentration test, with the taxi-time control ----------
    rows = []
    g = low.groupby(["apt", "pfx", "rwy"], observed=True)
    for (apt, pfx, rwy), sub in g:
        if len(sub) < MIN_CELL:
            continue
        n_gap, mode_gap, top3_gap = stats(sub["gap"].to_numpy())
        n_tax, mode_tax, top3_tax = stats(sub["taxi"].to_numpy())
        rows.append(dict(apt=apt, pfx=pfx, rwy=rwy, n=len(sub),
                         gap_distinct=n_gap, gap_mode=mode_gap, gap_top3=top3_gap,
                         taxi_distinct=n_tax, taxi_mode=mode_tax, taxi_top3=top3_tax,
                         gap_med=float(np.median(sub["gap"])),
                         taxi_med=float(np.median(sub["taxi"]))))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "e1_cell_concentration.csv", index=False)
    print(f"\n=== {len(C)} (airport, stand-prefix, runway) cells with >= {MIN_CELL} "
          f"low-delay rows ===")

    w = C["n"].to_numpy("float64")
    def wm(c): return float(np.average(C[c], weights=w))
    print(f"  row-weighted mean distinct whole-minute values per cell:  "
          f"MVT-AOBT {wm('gap_distinct'):.1f}   true taxi {wm('taxi_distinct'):.1f}")
    print(f"  row-weighted mean share on the modal minute:              "
          f"MVT-AOBT {wm('gap_mode'):.3f}   true taxi {wm('taxi_mode'):.3f}")
    print(f"  row-weighted mean share on the top 3 minutes:             "
          f"MVT-AOBT {wm('gap_top3'):.3f}   true taxi {wm('taxi_top3'):.3f}")

    # ---- the frozen gate --------------------------------------------------
    print(f"\n=== the frozen gate ===")
    print("  'at least one airport shows >= 30 % of low-delay rows on <= 10 distinct")
    print("   whole-minute values per (stand-prefix, runway) cell'")
    per_apt = []
    for apt, sub in C.groupby("apt"):
        ok = sub["gap_distinct"] <= 10
        share = float(sub.loc[ok, "n"].sum()) / float(sub["n"].sum())
        okt = sub["taxi_distinct"] <= 10
        share_t = float(sub.loc[okt, "n"].sum()) / float(sub["n"].sum())
        per_apt.append(dict(apt=apt, cells=len(sub), rows=int(sub["n"].sum()),
                            share_gap_le10=share, share_taxi_le10=share_t,
                            gap_distinct_med=float(sub["gap_distinct"].median()),
                            taxi_distinct_med=float(sub["taxi_distinct"].median())))
    A = pd.DataFrame(per_apt).sort_values("share_gap_le10", ascending=False)
    A.to_csv(OUT / "e1_by_airport.csv", index=False)
    print(A.round(3).to_string(index=False))
    passed = bool((A["share_gap_le10"] >= 0.30).any())
    print(f"\n  GATE {'PASSES' if passed else 'FAILS'}: "
          f"best airport {A.iloc[0]['apt']} at {A.iloc[0]['share_gap_le10']:.3f} "
          f"(its taxi-time control {A.iloc[0]['share_taxi_le10']:.3f})")

    # ---- does the gap depend on stand and runway the way EXOT must? -------
    print("\n=== does MVT - AOBT_3 vary with runway WITHIN a stand prefix? ===")
    print("  (EXOT must, per DPI guide 4.7.21.1(3); a taxi time does too, so this")
    print("   is reported beside the same statistic for the true taxi time)")
    rr = []
    for (apt, pfx), sub in C.groupby(["apt", "pfx"]):
        if len(sub) < 2:
            continue
        rr.append(dict(apt=apt, pfx=pfx, runways=len(sub),
                       gap_spread=float(sub["gap_med"].max() - sub["gap_med"].min()),
                       taxi_spread=float(sub["taxi_med"].max() - sub["taxi_med"].min())))
    R = pd.DataFrame(rr)
    R.to_csv(OUT / "e1_runway_spread.csv", index=False)
    print(f"  {len(R)} (airport, stand-prefix) groups with 2+ runways")
    print(f"  median spread of the cell median across runways: "
          f"MVT-AOBT {R['gap_spread'].median():.0f} s   true taxi {R['taxi_spread'].median():.0f} s")


if __name__ == "__main__":
    main()
