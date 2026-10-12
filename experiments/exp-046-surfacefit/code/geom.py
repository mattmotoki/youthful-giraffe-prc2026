"""The observation's own next features: the wheels-off geometry and the start of motion.

The desk, 2026-09-08 18:10 UTC: "the observed runway and hold-short geometry, the taxi
route length from the run itself, the run's start-of-motion gap -- as arms over
`c_lgb_shipped`".

**What the summaries can and cannot give.**  A day's summary is one row per ground
segment: endpoints and aggregates, not a path.  So of the three the desk named:

  runway geometry      YES.  `wheels_lat`/`wheels_lon` is the point where the aircraft
                       left the ground.  Its bearing and distance from the stand, and its
                       bearing and distance from the airport's own reference point, say
                       which runway end the run used and from which side of the field.
  taxi route length    PARTLY.  Only the CHORD is available -- stand to wheels-off in a
                       straight line -- not the path length.  The chord and the implied
                       speed `chord / obs_taxi` are what the run itself can support, and
                       they are named `chord`, not `route`, for that reason.
  hold-short geometry  NO.  A hold-short point needs the trajectory between the stand and
                       the runway and the summaries do not carry it.  It is not
                       approximated here: `seg_gap_max_s` (already a feature) is a
                       reception gap, not a hold, and calling it one would put a name on
                       the block that the data does not support.  Recovering it means
                       re-running the kernel to emit a path summary, which is a harvest
                       change and not this stage's.
  start-of-motion gap  YES, and three ways: acquisition to first motion, first motion to
                       the 250 m displacement onset, and the strict and loose off-block
                       rules against the frozen one.

`code/block.py` is NOT touched (desk, 2026-09-08 17:55: it is the sha both lanes pin and
it moves only in one coordinated rebuild after v14).  These columns are computed here and
joined to the block by the segment's own identity -- `(airport, icao, obs_wheels_off)` --
so the matcher is unchanged and the boundary cannot move.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/geom.py [month]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                                  # noqa: E402

VERSION = "geom-1"
R_EARTH_KM = 6371.0088

GEOM_FEATURES = [
    # the runway end the run used
    "rwy_lat", "rwy_lon", "rwy_from_apt_km", "rwy_from_apt_sin", "rwy_from_apt_cos",
    "rwy_gs_kt", "rwy_airborne_gap_s", "rwy_air_alt_ft",
    # the taxi chord, and what it implies about the run
    "taxi_chord_km", "taxi_chord_speed_kt", "taxi_chord_bearing_sin",
    "taxi_chord_bearing_cos", "stand_from_apt_km",
    # the start of motion
    "acquired_before_motion_s", "motion_to_disp_s",
    "offblock_strict_minus_disp_s", "offblock_loose_minus_disp_s",
    # what the run's own quality flags say
    "seg_starts_fast", "seg_ends_stationary", "seg_ends_airborne", "seg_airborne_alt_ok",
]


#: **The lane rule, written rather than inherited** (desk, 2026-09-08 19:05 UTC, after
#: the `c_geom_rel` reading): the airport-RELATIVE forms only; a raw coordinate never
#: enters a fitting column list here.  `block.py` computes `stand_lat`/`stand_lon` and
#: every fitting site in this lane drops them for `stand_dist_km`/`stand_sin`/`stand_cos`;
#: the same holds for `rwy_lat`/`rwy_lon`.  They stay in the parquet because a diagnosis
#: needs them -- `analysis/newarms.json` shows the raw coordinates buying +0.14 s of board
#: and costing 9.8 s of july tail, which is the measurement that settled the rule -- and
#: `SERVED_GEOM` is the list an arm may fit on.
ABSOLUTE_COORDINATES = ["rwy_lat", "rwy_lon"]
SERVED_GEOM = [c for c in GEOM_FEATURES if c not in ABSOLUTE_COORDINATES]


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def month_geom(month: str, rule: str | None = None) -> pd.DataFrame:
    """`MVT_ID_mvt` -> the geometry columns, for one month.  Cached beside the block."""
    if rule is None:
        rule, _ = A.serving_rule()
    key = A.block_cache_key(month, rule)
    p = A.WORK / f"geom-{month}-{VERSION}-{key}.parquet"
    if p.exists():
        return pd.read_parquet(p)

    blk = A.month_block(month, A.prefix_months_for(month), rule=None)
    dep = pd.read_parquet(D.month_file(month),
                          columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"])
    dep = dep[dep.PHASE_mvt == "DEP"][["MVT_ID_mvt", "ADEP_mvt"]]
    b = blk.merge(dep, on="MVT_ID_mvt", how="left", validate="one_to_one")
    b["airport"] = b.ADEP_mvt.astype(str)

    s = B.load_summaries(B.AIRPORTS, A.month_days(month), require_rule=rule)
    keep = ["airport", "icao", "obs_wheels_off", "wheels_lat", "wheels_lon",
            "wheels_gs_kt", "t_airborne", "air_alt_ft", "first_motion", "seg_start",
            "obs_offblock_disp", "obs_offblock_strict", "obs_offblock_loose",
            "starts_fast", "ends_stationary", "ends_airborne", "airborne_alt_ok"]
    s = s[[c for c in keep if c in s.columns]].copy()
    s = s[s.obs_wheels_off.notna()]
    # the segment's own identity: one airport, one aircraft, one instant of leaving the
    # ground.  A duplicate here would make the join ambiguous, so it is dropped and counted
    dup = int(s.duplicated(["airport", "icao", "obs_wheels_off"]).sum())
    s = s.drop_duplicates(["airport", "icao", "obs_wheels_off"])

    # MANY-to-one: the block's unmatched rows all carry a null icao and a null
    # wheels-off, so the left key is not unique; the RIGHT key is, which is what makes
    # the join well defined, and a null key matches nothing
    j = b.merge(s, how="left", left_on=["airport", "obs_icao", "obs_wheels_off"],
                right_on=["airport", "icao", "obs_wheels_off"], validate="many_to_one")

    # the airport's own reference point, taken from the stands this month's own runs saw:
    # label-free, and it needs no external aerodrome table
    ref = (j.dropna(subset=["stand_lat", "stand_lon"])
             .groupby("airport")[["stand_lat", "stand_lon"]].median())
    rlat = j.airport.map(ref.stand_lat).to_numpy("float64")
    rlon = j.airport.map(ref.stand_lon).to_numpy("float64")

    out = pd.DataFrame({"MVT_ID_mvt": j.MVT_ID_mvt.to_numpy()})
    wl, wo = j.wheels_lat.to_numpy("float64"), j.wheels_lon.to_numpy("float64")
    sl, so = j.stand_lat.to_numpy("float64"), j.stand_lon.to_numpy("float64")
    out["rwy_lat"], out["rwy_lon"] = wl, wo
    out["rwy_from_apt_km"] = _hav(rlat, rlon, wl, wo)
    ba = _bearing(rlat, rlon, wl, wo)
    out["rwy_from_apt_sin"], out["rwy_from_apt_cos"] = np.sin(ba), np.cos(ba)
    out["rwy_gs_kt"] = j.wheels_gs_kt.to_numpy("float64")
    out["rwy_airborne_gap_s"] = (j.t_airborne.to_numpy("float64")
                                 - j.obs_wheels_off.to_numpy("float64"))
    out["rwy_air_alt_ft"] = j.air_alt_ft.to_numpy("float64")

    chord = _hav(sl, so, wl, wo)
    out["taxi_chord_km"] = chord
    taxi = j.obs_taxi.to_numpy("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["taxi_chord_speed_kt"] = np.where(taxi > 0, chord / (taxi / 3600.0) / 1.852,
                                              np.nan)
    bc = _bearing(sl, so, wl, wo)
    out["taxi_chord_bearing_sin"], out["taxi_chord_bearing_cos"] = np.sin(bc), np.cos(bc)
    out["stand_from_apt_km"] = _hav(rlat, rlon, sl, so)

    fm = j.first_motion.to_numpy("float64")
    disp = j.obs_offblock_disp.to_numpy("float64")
    out["acquired_before_motion_s"] = fm - j.seg_start.to_numpy("float64")
    out["motion_to_disp_s"] = disp - fm
    out["offblock_strict_minus_disp_s"] = j.obs_offblock_strict.to_numpy("float64") - disp
    out["offblock_loose_minus_disp_s"] = j.obs_offblock_loose.to_numpy("float64") - disp
    for src, dst in (("starts_fast", "seg_starts_fast"),
                     ("ends_stationary", "seg_ends_stationary"),
                     ("ends_airborne", "seg_ends_airborne"),
                     ("airborne_alt_ok", "seg_airborne_alt_ok")):
        out[dst] = (j[src].astype("boolean").astype("float64").to_numpy()
                    if src in j else np.nan)

    out.attrs["report"] = {
        "month": month, "version": VERSION, "rule_sha256": rule,
        "block_cache_key": key, "rows": int(len(out)),
        "segments_with_a_wheels_off": int(len(s)),
        "duplicate_segment_identities_dropped": dup,
        "rows_joined_to_a_segment": int(j.wheels_lat.notna().sum()),
        "rows_with_a_clock": int(blk.obs_has_clock.sum()),
        "coverage_of_the_with_clock_rows": float(
            (j.wheels_lat.notna() & blk.obs_has_clock.to_numpy(bool)).sum()
            / max(int(blk.obs_has_clock.sum()), 1)),
        "airport_reference_points": {a: [float(r.stand_lat), float(r.stand_lon)]
                                     for a, r in ref.iterrows()},
    }
    out.to_parquet(p, index=False)
    (p.with_suffix(".json")).write_text(json.dumps(out.attrs["report"], indent=2,
                                                   sort_keys=True))
    return out


def main() -> None:
    months = sys.argv[1:] or list(A.FOLD_MONTH.values())
    rep = {}
    for m in months:
        g = month_geom(m)
        r = g.attrs.get("report") or json.loads(
            (A.WORK / f"geom-{m}-{VERSION}-"
             f"{A.block_cache_key(m, A.serving_rule()[0])}.json").read_text())
        rep[m] = r
        nn = {c: float(g[c].notna().mean()) for c in GEOM_FEATURES}
        rep[m]["not_null_share"] = nn
        print(f"{m}: {r['rows']} rows, joined {r['rows_joined_to_a_segment']}, "
              f"with-clock coverage {r['coverage_of_the_with_clock_rows']:.4f}, "
              f"duplicates dropped {r['duplicate_segment_identities_dropped']}")
    (EXP / "analysis/geom.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print("written to analysis/geom.json")


if __name__ == "__main__":
    main()
