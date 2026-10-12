"""exp-043-pilot step 1: the receipt census, and the 40 % gate.

Reads the kernel slices in `artifacts/exp-043/` and the week's cohort, and answers
the four questions `assignments/pilot.md` step 1 asks, per day:

  distinct ICAO24 on the ground, points per aircraft-day on the ground, the median
  inter-point interval on the ground, and the share of the day's EHAM departures
  whose callsign appears on the ground at all.

The callsign of a no-NM row (no `CALLSIGN_flt`) is reconstructed with the airline
prefix map, which is learned on the ELEVEN 2025 months that are not July, so the
map is never fitted on the week it is scored on.

  .venv/bin/python experiments/exp-043-pilot/code/census.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ART = ROOT / "artifacts/exp-043"
sys.path.insert(0, str(EXP / "code"))


def prefix_map() -> pd.Series:
    """FLIGHT_mvt letter prefix -> modal CALLSIGN_flt letter prefix, learned off July."""
    frames = []
    for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        if "2025-07-01" in p.name:
            continue
        t = pd.read_parquet(p, columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
        frames.append(t[t.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt"))
    t = pd.concat(frames, ignore_index=True).dropna(subset=["FLIGHT_mvt", "CALLSIGN_flt"])
    t["fpre"] = t.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    t["cpre"] = t.CALLSIGN_flt.str.extract(r"^([A-Z]+)")[0]
    t = t.dropna(subset=["fpre", "cpre"])
    return t.groupby("fpre")["cpre"].agg(lambda s: s.mode().iat[0])


def with_callsign(coh: pd.DataFrame) -> pd.DataFrame:
    """`callsign` (given or reconstructed) and how it was obtained."""
    pm = prefix_map()
    fpre = coh.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    fsuf = coh.FLIGHT_mvt.str.replace(r"^[A-Z]+", "", regex=True)
    recon = fpre.map(pm).fillna("") + fsuf.fillna("")
    recon = recon.where(fpre.map(pm).notna())
    coh = coh.copy()
    coh["callsign_recon"] = recon
    coh["callsign"] = coh.CALLSIGN_flt.fillna(recon)
    coh["callsign_source"] = np.where(coh.CALLSIGN_flt.notna(), "flt",
                                      np.where(recon.notna(), "prefix", "none"))
    return coh


#: the columns `segments.py` and this census actually read.  The laptop runs with
#: its swap nearly full, so the unused ones (alt_ft, track, flags, src, idx) are
#: left in the file rather than concatenated across seven days.
COLS = ["icao", "reg", "type", "t", "lat", "lon", "on_ground", "in_box",
        "first_airborne", "alt_ft", "gs_kt", "callsign"]


def load_slices(days: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # the days are read in calendar order whatever order the caller passes, because
    # the de-duplication below keeps the FIRST copy of a repeated point and that
    # must not depend on an argument list (round 4's determinism finding, extended
    # to every step that reads order rather than just the sort in `segments.py`)
    days = sorted(days)
    pts = pd.concat([pd.read_parquet(ART / f"points_{d}.parquet", columns=COLS)
                     for d in days], ignore_index=True)
    # a day's trace file carries a few points from the previous UTC day, so two
    # neighbouring day files can hold the same point twice
    pts = pts.drop_duplicates(["icao", "t"], ignore_index=True)
    cen = pd.concat([pd.read_parquet(ART / f"census_{d}.parquet").assign(day=d) for d in days],
                    ignore_index=True)
    return pts, cen


def main(days: list[str]) -> None:
    pts, cen = load_slices(days)
    coh = with_callsign(pd.read_parquet(EXP / "work/cohort.parquet"))
    coh = coh[coh.day.dt.strftime("%Y-%m-%d").isin(days)]

    g = pts[pts.on_ground & pts.in_box].copy()
    g["utc"] = pd.to_datetime(g.t, unit="s", utc=True)
    g["gday"] = g.utc.dt.strftime("%Y-%m-%d")

    per_day = {}
    for d in days:
        gd = g[g.gday == d]
        cd = coh[coh.day.dt.strftime("%Y-%m-%d") == d]
        seen = set(gd.callsign.dropna().str.strip().unique())
        hit = cd.callsign.isin(seen)
        dt = (gd.sort_values("t").groupby("icao")["t"].diff().dropna())
        dt = dt[(dt > 0) & (dt <= 900)]
        per_day[d] = {
            "departures": int(len(cd)),
            "icao24_on_ground": int(gd.icao.nunique()),
            "ground_points": int(len(gd)),
            "points_per_aircraft_day": {
                "median": float(gd.groupby("icao").size().median()),
                "mean": float(gd.groupby("icao").size().mean()),
                "p10": float(gd.groupby("icao").size().quantile(.1)),
                "p90": float(gd.groupby("icao").size().quantile(.9))},
            "inter_point_interval_s": {
                "median": float(dt.median()), "p10": float(dt.quantile(.1)),
                "p90": float(dt.quantile(.9)),
                "share_at_or_under_35s": float((dt <= 35).mean())},
            "distinct_callsigns_on_ground": len(seen),
            "departures_callsign_seen_on_ground": int(hit.sum()),
            "departure_receipt_rate": round(float(hit.mean()), 4),
            "receipt_rate_nm_rows": round(float(hit[cd.has_nm].mean()), 4),
            "receipt_rate_no_nm_rows": (round(float(hit[~cd.has_nm].mean()), 4)
                                        if (~cd.has_nm).any() else None),
            "no_nm_rows": int((~cd.has_nm).sum()),
            "aircraft_in_box_any": int(cen[cen.day == d].shape[0]),
            "aircraft_in_box_with_ground": int((cen[cen.day == d].n_in_box_ground > 0).sum()),
        }
        print(f"{d}: {per_day[d]['departure_receipt_rate']:.1%} of {per_day[d]['departures']} "
              f"departures seen on the ground; {per_day[d]['icao24_on_ground']} ICAO24 on the "
              f"ground; median interval {per_day[d]['inter_point_interval_s']['median']:.0f} s")

    # pooled receipt is the SUM OF THE DAILY hits, never a week-wide callsign
    # union: a callsign seen on Sunday must not mark Monday's departure as seen
    hit = pd.Series(False, index=coh.index)
    for d in days:
        seen = set(g.loc[g.gday == d, "callsign"].dropna().str.strip().unique())
        on = coh.day.dt.strftime("%Y-%m-%d") == d
        hit.loc[on] = coh.loc[on, "callsign"].isin(seen)
    res = {
        "days": days, "airport": "EHAM", "box": {"lat": [52.25, 52.40], "lon": [4.65, 4.85]},
        "per_day": per_day,
        "pooled": {
            "departures": int(len(coh)),
            "receipt_rate": round(float(hit.mean()), 4),
            "receipt_rate_nm_rows": round(float(hit[coh.has_nm].mean()), 4),
            "receipt_rate_no_nm_rows": (round(float(hit[~coh.has_nm].mean()), 4)
                                        if (~coh.has_nm).any() else None),
            "callsign_source": coh.callsign_source.value_counts().to_dict()},
        "gate": {"threshold": 0.40, "measured": round(float(hit.mean()), 4),
                 "passes": bool(hit.mean() >= 0.40)},
    }
    (EXP / "analysis/census.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print(json.dumps(res["pooled"] | res["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:] or ["2025-07-07", "2025-07-08"])
