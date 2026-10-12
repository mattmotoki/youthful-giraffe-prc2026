"""exp-045-harvest: the receipt rate by airport, day and month.

The one number the desk stops the programme on: the share of an airport's departures
whose broadcast callsign is actually seen on the ground inside the airport box.  It is
an upper bound on what a feature can use, not a match rate — the pilot's matcher lost a
further few per cent to its take-off window and to join ambiguity.

Definition.  A departure counts as received when its callsign appears in the ground
callsign table of its airport with a last ground point inside `[MVT_TIME - 6 h,
MVT_TIME + 1 h]`, where `MVT_TIME_UTC_mvt` is the actual take-off.  The window is used
rather than the pilot's "seen at all that day" because the day of a 2026 departure
cannot be taken from `BLOCK_TIME_UTC_mvt` (blanked in the ranking file) and a
same-day rule would then turn a midnight departure into a false miss.  The pilot's
same-day rate is reported beside it for the days the two overlap.

The callsign of a departure with no `CALLSIGN_flt` is reconstructed with the airline
prefix map, learned on the 2025 training months other than the one being censused, as
in `exp-043-pilot/code/census.py`.

    .venv/bin/python experiments/exp-045-harvest/code/census.py [YYYY-MM ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ART = ROOT / "artifacts/exp-045"
AIRPORTS = ["EHAM", "LFPG", "EGLL", "LTFM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]
BEFORE_S, AFTER_S = 6 * 3600.0, 1 * 3600.0


def training() -> pd.DataFrame:
    frames = []
    for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        t = pd.read_parquet(p, columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
                                        "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
        frames.append(t[t.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt"))
    t = pd.concat(frames, ignore_index=True)
    t["month"] = t.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    return t


def ranking() -> pd.DataFrame:
    r = pd.read_parquet(ROOT / "data/raw/ranking.parquet",
                        columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt",
                                 "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    r = r[r.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    r["month"] = r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    return r


def prefix_map(train: pd.DataFrame, exclude: str) -> pd.Series:
    """FLIGHT_mvt letter prefix -> modal CALLSIGN_flt letter prefix, off `exclude`."""
    t = train[train.month != exclude].dropna(subset=["FLIGHT_mvt", "CALLSIGN_flt"])
    fp = t.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    cp = t.CALLSIGN_flt.str.extract(r"^([A-Z]+)")[0]
    ok = fp.notna() & cp.notna()
    return pd.DataFrame({"f": fp[ok], "c": cp[ok]}).groupby("f")["c"].agg(
        lambda s: s.mode().iat[0])


def cohort(month: str, train: pd.DataFrame, rank: pd.DataFrame) -> pd.DataFrame:
    src = rank if month.startswith("2026") else train
    d = src[(src.month == month) & (src.ADEP_mvt.isin(AIRPORTS))].copy()
    pm = prefix_map(train, month)
    fpre = d.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    fsuf = d.FLIGHT_mvt.str.replace(r"^[A-Z]+", "", regex=True)
    recon = (fpre.map(pm) + fsuf.fillna("")).where(fpre.map(pm).notna())
    d["callsign"] = d.CALLSIGN_flt.fillna(recon).str.strip()
    d["callsign_source"] = np.where(d.CALLSIGN_flt.notna(), "flt",
                                    np.where(recon.notna(), "prefix", "none"))
    d["has_nm"] = d.AOBT_3_flt.notna()
    # unit-safe: the parquet columns are microsecond datetimes, so astype("int64")
    # would be 1000x off and the window test would miss every row
    d["mvt"] = (d.MVT_TIME_UTC_mvt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d["day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    return d


def ground(days: list[str]) -> pd.DataFrame:
    """The ground callsign tables of `days`, plus the day before and after each."""
    want = set(days)
    for d in days:
        ts = pd.Timestamp(d)
        want |= {str((ts - pd.Timedelta(days=1)).date()), str((ts + pd.Timedelta(days=1)).date())}
    got = [p for d in sorted(want) if (p := ART / f"callsigns/callsigns_{d}.parquet").exists()]
    if not got:
        return pd.DataFrame(columns=["airport", "callsign", "t_first", "t_last", "day"])
    out = []
    for p in got:
        g = pd.read_parquet(p)
        g["day"] = p.name[10:20]
        out.append(g)
    g = pd.concat(out, ignore_index=True)
    g["callsign"] = g.callsign.str.strip()
    return g


def rate(coh: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """`received`, `received_anyday` and `received_sameday` per departure.

    The window test is applied to each ground appearance separately.  Aggregating
    `min(t_first)` and `max(t_last)` over the month first -- which this function did
    until 2026-09-08 -- turns the test into "the callsign's month-long span overlaps
    the window", and a callsign flown daily then counts as received on every day of
    the month.  It priced EGLL at 83.2 % in March 2025 where the stated definition
    gives 37.0 %, because Heathrow's flight numbers repeat and its reception does not.
    The aggregate answer is kept as `received_anyday`: the gap between the two is the
    share of departures whose callsign the network hears on ANOTHER day, which is a
    fact about intermittency rather than about the airline.
    """
    m = coh.reset_index(drop=True).copy()
    m["_i"] = np.arange(len(m))
    slim = m[["_i", "ADEP_mvt", "callsign", "mvt"]]
    j = slim.merge(g, left_on=["ADEP_mvt", "callsign"], right_on=["airport", "callsign"],
                   how="inner")
    hit = (j.t_last >= j.mvt - BEFORE_S) & (j.t_first <= j.mvt + AFTER_S)
    m["received"] = m._i.isin(set(j.loc[hit, "_i"]))

    key = g.groupby(["airport", "callsign"], observed=True).agg(
        t_first=("t_first", "min"), t_last=("t_last", "max")).reset_index()
    a = m[["_i", "ADEP_mvt", "callsign", "mvt"]].merge(
        key, left_on=["ADEP_mvt", "callsign"], right_on=["airport", "callsign"], how="left")
    m["received_anyday"] = ((a.t_last >= a.mvt - BEFORE_S)
                            & (a.t_first <= a.mvt + AFTER_S)).fillna(False).to_numpy()

    sd = set(zip(g.airport.astype(str), g.callsign, g.day))
    m["received_sameday"] = [k in sd for k in zip(m.ADEP_mvt, m.callsign, m.day)]
    return m.drop(columns=["_i"])


def main(months: list[str]) -> None:
    train, rank = training(), ranking()
    have = {p.name[10:20] for p in (ART / "callsigns").glob("callsigns_*.parquet")}
    out = json.loads((EXP / "analysis/receipt.json").read_text()) if (
        EXP / "analysis/receipt.json").exists() else {}
    for month in months:
        coh = cohort(month, train, rank)
        days = sorted(d for d in coh.day.unique() if d in have)
        if not days:
            print(f"{month}: no ground callsign tables on disk yet")
            continue
        coh = coh[coh.day.isin(days)]
        m = rate(coh, ground(days))
        by = m.groupby("ADEP_mvt", observed=True).agg(
            departures=("received", "size"), received=("received", "sum"),
            sameday=("received_sameday", "sum"), anyday=("received_anyday", "sum"),
            no_nm=("has_nm", lambda s: (~s).sum()))
        by["rate"] = (by.received / by.departures).round(4)
        by["rate_sameday"] = (by.sameday / by.departures).round(4)
        by["rate_anyday"] = (by.anyday / by.departures).round(4)
        nm = m[m.has_nm].groupby("ADEP_mvt", observed=True)["received"].mean().round(4)
        nn = m[~m.has_nm].groupby("ADEP_mvt", observed=True)["received"].mean().round(4)
        by["rate_nm"], by["rate_no_nm"] = nm, nn
        out[month] = {
            "days_measured": days,
            "departures": int(len(m)),
            "receipt_rate": round(float(m.received.mean()), 4),
            # the pre-2026-09-08 answer: the callsign's whole-month span overlaps the
            # window.  Kept as the diagnostic it is -- see rate().
            "receipt_rate_anyday": round(float(m.received_anyday.mean()), 4),
            "receipt_rate_nm": round(float(m[m.has_nm].received.mean()), 4),
            "receipt_rate_no_nm": (round(float(m[~m.has_nm].received.mean()), 4)
                                   if (~m.has_nm).any() else None),
            "callsign_source": m.callsign_source.value_counts().to_dict(),
            "by_airport": {a: {k: (None if pd.isna(v) else (float(v) if isinstance(v, float)
                                                            else int(v)))
                               for k, v in r.items()}
                           for a, r in by.iterrows()},
            # per airport AND day: the month rate hides a receiver outage inside it,
            # and the two gate days turned out to be the worst two of January at three
            # airports.  A month number alone would have kept both invisible.
            "by_airport_day": {
                str(a): {str(d): round(float(v), 4) for d, v in col.dropna().items()}
                for a, col in (m.groupby(["day", "ADEP_mvt"], observed=True)["received"]
                               .mean().unstack()).items()},
            "by_day": {d: round(float(x), 4) for d, x in
                       m.groupby("day", observed=True)["received"].mean().items()},
        }
        print(f"\n{month}: {len(days)} days, {len(m)} departures, "
              f"receipt {out[month]['receipt_rate']:.1%} "
              f"(NM {out[month]['receipt_rate_nm']:.1%}, "
              f"no-NM {out[month]['receipt_rate_no_nm']})")
        print(by[["departures", "rate", "rate_sameday", "rate_nm", "rate_no_nm", "no_nm"]]
              .sort_values("rate").to_string())
    (EXP / "analysis/receipt.json").write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(EXP / "code"))
    from days import ORDER
    main(sys.argv[1:] or ORDER)
