"""exp-025-cloud: a synthetic PRC dataset with the real schema.

Why this exists: the competition files and the S3 key are not on this laptop
yet, and the Kaggle port must be proved before they land. This writes files
that carry the same columns, dtypes and *classes* as the real ones -- ordinary
departures, the SCHED-anchor class, the no-NM class, the null-aircraft-type
"ghost" class, and a heavy right tail -- so every branch of
`experiments/exp-018-fresh/code/` executes. The NUMBERS it produces are
meaningless; only the code path, the runtime and the memory are.

  .venv/bin/python experiments/exp-025-cloud/code/make_synthetic.py \
      --out /tmp/prc-synth --scale smoke

Scales: `smoke` is 1/50 of the real row counts (seconds, for a smoke test);
`full` reproduces the real per-month departure counts (2,085,047 training
departures, 344,841 ranking departures) so a Kaggle runtime measured on it
transfers to the real run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Real per-month departure counts where the record has them (january 153,706,
# july 190,713, december 165,677 from experiments/exp-018-fresh/results/
# fold4-final.json) and a seasonal interpolation elsewhere summing to the
# 2,085,047 of exp-018-fresh RESULTS.md 1.1.
DEP_PER_MONTH = {
    "2025-01": 153_706, "2025-02": 148_000, "2025-03": 165_000, "2025-04": 175_000,
    "2025-05": 182_000, "2025-06": 188_000, "2025-07": 190_713, "2025-08": 189_000,
    "2025-09": 183_000, "2025-10": 178_000, "2025-11": 166_951, "2025-12": 165_677,
}
# The ranking file is 56 % july and 44 % january by row count (exp-018-fresh
# RESULTS.md 3), 344,841 departures in total.
RANK_PER_MONTH = {"2026-01": 152_000, "2026-07": 192_841}

#: Ten departure airports, as in the ranking file. The identities are real ICAO
#: codes so that per-airport code paths look natural; every NUMBER attached to
#: them here is invented.
#: The set is `prc.tail_features.RANKING_AIRPORTS`, not a plausible-looking ten:
#: `tail_signals` keeps only departures from those airports, so a wrong tenth
#: airport silently deletes 7 % of the ranking rows from the tail lineage
#: (generator v2 used LOWW and T-tailv2 wrote 320,797 rows instead of 344,841).
AIRPORTS = ["EDDF", "EHAM", "EGLL", "LFPG", "LIRF", "LEBL", "EDDM", "LSZH", "LTFM", "LEMD"]
#: share of departures, and the airport-specific BLOCK - AOBT_3 offset in
#: seconds (median and scale), loosely after docs/input-rule.md.
APT = {
    "EDDF": (0.14, 40, 260), "EHAM": (0.13, 114, 300), "EGLL": (0.11, 20, 587),
    "LFPG": (0.13, 10, 330), "LIRF": (0.08, -179, 771), "LEBL": (0.09, 30, 300),
    "EDDM": (0.09, 25, 280), "LSZH": (0.07, 5, 282), "LTFM": (0.09, 234, 420),
    "LEMD": (0.07, 15, 300),
}
RUNWAYS = {a: [f"{n:02d}{s}" for n, s in [(7, "L"), (7, "R"), (25, "L"), (25, "C")]]
           for a in AIRPORTS}
TYPES = ["A320", "A321", "A20N", "B738", "B38M", "A319", "B77W", "A333", "E195", "CRJ9",
         "B739", "A359", "B788", "AT76", "DH8D"]
OPERATORS = ["DLH", "KLM", "BAW", "AFR", "ITY", "VLG", "RYR", "EZY", "SWR", "THY", "AUA",
             "EJU", "WZZ", "TAP", "IBE", "UAE", "DAL", "UAL", "AAL", "SAS"]
WTC = ["M", "H", "L", "J"]
SEG = ["Traditional Scheduled", "Lowcost", "Charter", "Business Aviation", "All-Cargo"]
FTYPE = ["S", "N", "X"]
FRULE = ["I", "V"]
DESTS = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LIRF", "KJFK", "OMDB", "LTFM", "LSZH",
         "EDDM", "LEBL", "EIDW", "ESSA", "LOWW", "EKCH", "LPPT", "LGAV", "EPWA", "LHBP"]

#: Column order copied from the real `ranking.parquet`, not invented. A frame
#: with the right names in the wrong order is still a different file.
COLUMNS = [
    "MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt",
    "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "TAXITIME_SEC_mvt", "LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt",
    "ADES_FILED_flt", "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt",
    "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt",
    "ARVT_3_flt",
]


def _ts(base: np.ndarray, offset_s: np.ndarray) -> np.ndarray:
    """UTC datetime64[ns] from epoch seconds, NaT where the offset is NaN."""
    out = np.where(np.isfinite(offset_s), (base + offset_s) * 1e9, np.nan)
    return pd.to_datetime(pd.Series(out), unit="ns", utc=True).to_numpy()


def _month_bounds(month: str) -> tuple[float, float]:
    s = pd.Timestamp(month + "-01", tz="UTC")
    e = s + pd.offsets.MonthBegin(1)
    return s.value / 1e9, e.value / 1e9


def build_month(month: str, n_dep: int, rng: np.random.Generator,
                id0: int, blank_dep: bool, keep_inside: bool = False) -> pd.DataFrame:
    """One month: `n_dep` departures and about as many arrivals.

    `blank_dep` reproduces the ranking file: departure BLOCK_TIME_UTC_mvt and
    TAXITIME_SEC_mvt are null.

    `keep_inside` drops departures whose TAKE-OFF falls outside the month. The
    ranking months need it and the training months do not, and the difference is
    not cosmetic: `prc.data.load_ranking` derives `month` from
    `MVT_TIME_UTC_mvt`, and `exp-018-fresh/code/dataset.py` then selects
    `rk[rk["month"] == month]`. A generated departure whose long taxi pushes its
    take-off past midnight on the last day is therefore invisible to the
    feature build while still sitting in `submitting.parquet`, and
    `prc.data.write_submission` rejects the result -- which is exactly what the
    first full-scale run hit, on 81 of 344,841 rows. The real ranking file
    cannot contain such a row, because its two months are defined by take-off
    time, so this is a defect of the generator and not of the code under test.
    """
    t0, t1 = _month_bounds(month)
    n_arr = n_dep
    n = n_dep + n_arr
    phase = np.array(["DEP"] * n_dep + ["ARR"] * n_arr, dtype=object)

    p = np.array([APT[a][0] for a in AIRPORTS], dtype="float64")
    p = p / p.sum()
    apt_i = rng.choice(len(AIRPORTS), size=n, p=p)
    apt = np.array(AIRPORTS, dtype=object)[apt_i]
    offset_med = np.array([APT[a][1] for a in AIRPORTS], dtype="float64")[apt_i]
    offset_scale = np.array([APT[a][2] for a in AIRPORTS], dtype="float64")[apt_i]

    # a diurnal schedule: two peaks, tapering overnight
    day = rng.integers(0, int((t1 - t0) // 86400), size=n)
    hour = np.clip(rng.normal(np.where(rng.random(n) < 0.5, 8.5, 16.5), 3.2, n), 0, 23.99)
    sched = t0 + day * 86400.0 + hour * 3600.0 + rng.integers(0, 60, n)

    stand_n = rng.integers(1, 120, size=n)
    stand_pfx = np.array(list("ABCDEFGHJK"), dtype=object)[rng.integers(0, 10, size=n)]
    stand = np.char.add(stand_pfx.astype(str), stand_n.astype(str)).astype(object)
    rwy = np.empty(n, dtype=object)
    for i, a in enumerate(AIRPORTS):
        m = apt_i == i
        rwy[m] = np.array(RUNWAYS[a], dtype=object)[rng.integers(0, len(RUNWAYS[a]), m.sum())]

    typ = np.array(TYPES, dtype=object)[rng.integers(0, len(TYPES), n)]
    oper = np.array(OPERATORS, dtype=object)[rng.integers(0, len(OPERATORS), n)]
    dest = np.array(DESTS, dtype=object)[rng.integers(0, len(DESTS), n)]
    flight = np.char.add(oper.astype(str), rng.integers(100, 9999, n).astype(str)).astype(object)

    is_dep = phase == "DEP"

    # ---------------------------------------------------------------- classes
    u = rng.random(n)
    no_nm = is_dep & (u < 0.0150)                       # 1.5 % of ranking departures
    ghost = is_dep & (u >= 0.0150) & (u < 0.0161)       # 0.11 %, mostly LIRF
    ghost &= (apt == "LIRF") | (rng.random(n) < 0.02)
    exact = is_dep & (u >= 0.05) & (u < 0.0966)         # 4.66 % SCHED-anchor class

    # ------------------------------------------------------- departure timing
    # off-block delay against the schedule, then the airport clock offset, then
    # the taxi itself with a lognormal right tail.
    dly = np.maximum(rng.gamma(1.6, 260.0, n) - 120.0, -600.0)
    aobt = sched + dly
    offs = offset_med + rng.standard_t(2.5, n) * offset_scale * 0.5
    block = aobt + offs
    block = np.where(exact, sched + rng.integers(-6, 7, n), block)

    base_taxi = np.exp(rng.normal(6.75, 0.42, n))       # median ~ 855 s
    load = 1.0 + 0.55 * np.clip(rng.normal(0, 1, n), -1.2, 4.0) ** 2 * 0.25
    taxi = np.clip(base_taxi * load, 45, None)
    heavy = rng.random(n) < 0.0034                      # 0.34 % above 3600 s
    taxi = np.where(heavy, taxi + rng.gamma(2.0, 2400.0, n), taxi)
    # ghosts: about half carry BLOCK == SCHED, so their label is MVT - SCHED,
    # which after a long ground hold is tens of thousands of seconds.
    gclass = ghost & (rng.random(n) < 0.53)
    block = np.where(gclass, sched, block)
    taxi = np.where(gclass, taxi + rng.gamma(1.4, 9000.0, n), taxi)
    mvt_dep = block + taxi

    # ---------------------------------------------------------- arrival timing
    land = sched                                        # ARR: MVT is the landing
    taxi_in = np.clip(np.exp(rng.normal(6.1, 0.45, n)), 40, None)
    inblock = land + taxi_in

    mvt = np.where(is_dep, mvt_dep, land)
    blk = np.where(is_dep, block, inblock)
    tax = np.where(is_dep, taxi, taxi_in)

    lobt = sched - rng.gamma(2.0, 300.0, n)
    iobt = lobt + rng.normal(0, 30, n)
    eobt1 = lobt + rng.gamma(1.5, 200.0, n)
    plan_ft = rng.gamma(3.0, 1800.0, n) + 1200.0
    arvt1 = eobt1 + plan_ft
    arvt3 = mvt + plan_ft * rng.normal(1.0, 0.06, n)

    nan = np.full(n, np.nan)
    df = pd.DataFrame({
        "MVT_ID_mvt": np.arange(id0, id0 + n, dtype="float64"),
        "FLIGHT_ID_mvt": np.arange(id0, id0 + n, dtype="float64") + 5e8,
        "PHASE_mvt": phase,
        "ADEP_mvt": np.where(is_dep, apt, dest),
        "ADES_mvt": np.where(is_dep, dest, apt),
        "FLIGHT_mvt": flight,
        "FLIGHT_RULE_mvt": np.array(FRULE, dtype=object)[(rng.random(n) > 0.02).astype(int) * 0],
        "RUNWAY_mvt": rwy,
        "STAND_mvt": stand,
        "AIRCRAFT_TYPE_mvt": np.where(ghost, None, typ),
        "MVT_TIME_UTC_mvt": _ts(np.zeros(n), mvt),
        "BLOCK_TIME_UTC_mvt": _ts(np.zeros(n), np.where(blank_dep & is_dep, np.nan, blk)),
        "SCHED_TIME_UTC_mvt": _ts(np.zeros(n), sched),
        "TAXITIME_SEC_mvt": np.where(blank_dep & is_dep, nan, np.round(tax)),
        "CALLSIGN_flt": flight,
        "AIRCRAFT_TYPE_flt": np.where(ghost, None, typ),
        "AIRCRAFT_OPERATOR_flt": oper,
        "WK_TBL_CAT_flt": np.array(WTC, dtype=object)[rng.integers(0, len(WTC), n)],
        "MARKET_SEGMENT_flt": np.array(SEG, dtype=object)[rng.integers(0, len(SEG), n)],
        "FLIGHT_TYPE_flt": np.array(FTYPE, dtype=object)[rng.integers(0, len(FTYPE), n)],
        "FLIGHT_RULE_flt": "I",
        "ADEP_flt": np.where(is_dep, apt, dest),
        "ADES_flt": np.where(is_dep, dest, apt),
        "ADES_FILED_flt": np.where(is_dep, dest, apt),
        "LOBT_flt": _ts(np.zeros(n), lobt),
        "IOBT_flt": _ts(np.zeros(n), iobt),
        "EOBT_1_flt": _ts(np.zeros(n), eobt1),
        "AOBT_3_flt": _ts(np.zeros(n), aobt),
        "ARVT_1_flt": _ts(np.zeros(n), arvt1),
        "ARVT_3_flt": _ts(np.zeros(n), arvt3),
    })[COLUMNS]

    # the no-NM class: every _flt column null, as in the real file
    flt_cols = [c for c in COLUMNS if c.endswith("_flt")]
    df.loc[no_nm, flt_cols] = None
    for c in ("LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt",
              "ARVT_1_flt", "ARVT_3_flt"):
        df.loc[no_nm, c] = pd.NaT
    if keep_inside:
        mvt_s = df["MVT_TIME_UTC_mvt"].astype("int64") / 1e9
        inside = (mvt_s >= t0) & (mvt_s < t1)
        df = df[inside]
    df = df.sort_values("MVT_TIME_UTC_mvt", kind="stable").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--seed", type=int, default=20260907)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    div = 50 if a.scale == "smoke" else 1
    rng = np.random.default_rng(a.seed)

    # The tail lineage keeps only departures from RANKING_AIRPORTS, so an
    # airport set that merely looks right deletes rows without saying so.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from prc.tail_features import RANKING_AIRPORTS
    except Exception:                                       # noqa: BLE001
        print("prc.tail_features not importable; airport set unchecked", flush=True)
    else:
        if set(AIRPORTS) != set(RANKING_AIRPORTS):
            raise SystemExit(f"AIRPORTS != prc RANKING_AIRPORTS: "
                             f"extra {sorted(set(AIRPORTS) - set(RANKING_AIRPORTS))}, "
                             f"missing {sorted(set(RANKING_AIRPORTS) - set(AIRPORTS))}")

    manifest = {"generator_version": 4, "scale": a.scale, "seed": a.seed,
                "files": {}}   # v2: ranking take-offs are kept inside their month
    idc = 180_000_000
    for month, n in DEP_PER_MONTH.items():
        df = build_month(month, max(n // div, 500), rng, idc, blank_dep=False)
        idc += len(df)
        s = pd.Timestamp(month + "-01")
        e = s + pd.offsets.MonthBegin(1)
        p = out / f"training_{s:%Y-%m-%d}_{e:%Y-%m-%d}.parquet"
        df.to_parquet(p, index=False)
        manifest["files"][p.name] = {"rows": len(df), "dep": int((df.PHASE_mvt == "DEP").sum()),
                                     "bytes": p.stat().st_size}
        print(f"{p.name}  {len(df):>9,} rows  {p.stat().st_size/2**20:6.1f} MB", flush=True)

    rk = []
    for month, n in RANK_PER_MONTH.items():
        target = max(n // div, 500)
        # oversample, drop the take-offs that leave the month, then trim to the
        # exact count, so the ranking file holds precisely the real row count
        df = build_month(month, int(target * 1.02) + 50, rng, idc, blank_dep=True,
                         keep_inside=True)
        dep_idx = df.index[df["PHASE_mvt"] == "DEP"][:target]
        arr_idx = df.index[df["PHASE_mvt"] == "ARR"][:target]
        df = df.loc[sorted(set(dep_idx) | set(arr_idx))].reset_index(drop=True)
        assert int((df["PHASE_mvt"] == "DEP").sum()) == target, month
        idc += len(df)
        rk.append(df)
    rank = pd.concat(rk, ignore_index=True)
    m = rank.loc[rank["PHASE_mvt"] == "DEP", "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    bad = int((~m.isin(RANK_PER_MONTH)).sum())
    if bad:
        raise RuntimeError(f"{bad} ranking departures take off outside their month")
    p = out / "ranking.parquet"
    rank.to_parquet(p, index=False)
    # If the real file is on this machine, the generated schema must equal it
    # column for column. Generator v3 shipped without ADEP_flt and ADES_flt and
    # with a spurious EOBT_flt, which nothing in `prc/` reads -- so no run ever
    # complained, and the gap was found by reading the real file, not by running.
    real = Path(__file__).resolve().parents[3] / "data" / "raw" / "ranking.parquet"
    if real.exists():
        want = list(pd.read_parquet(real, columns=None).head(0).columns)
        got = list(rank.columns)
        if want != got:
            raise RuntimeError(
                f"schema differs from the real ranking file: "
                f"missing {[c for c in want if c not in got]}, "
                f"extra {[c for c in got if c not in want]}, "
                f"order_ok={sorted(want) == sorted(got) and want == got}")
        print("schema matches data/raw/ranking.parquet", flush=True)
    manifest["files"]["ranking.parquet"] = {
        "rows": len(rank), "dep": int((rank.PHASE_mvt == "DEP").sum()),
        "bytes": p.stat().st_size}
    print(f"ranking.parquet  {len(rank):>9,} rows  {p.stat().st_size/2**20:6.1f} MB", flush=True)

    sub = rank.loc[rank.PHASE_mvt == "DEP", ["MVT_ID_mvt"]].copy()
    sub["TAXITIME_SEC_mvt"] = np.nan
    p = out / "submitting.parquet"
    sub.to_parquet(p, index=False)
    manifest["files"]["submitting.parquet"] = {"rows": len(sub), "bytes": p.stat().st_size}
    print(f"submitting.parquet {len(sub):>7,} rows", flush=True)

    (out / "SYNTHETIC.json").write_text(json.dumps(manifest, indent=1))
    print("\nSYNTHETIC -- schema only. Every number here is invented; no result "
          "computed on it may enter the ledger.")


if __name__ == "__main__":
    main()
