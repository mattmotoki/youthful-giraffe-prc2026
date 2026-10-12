"""Track inputs of the head, from the adsb.lol points of the row's own ground segment.

    nice -n 19 taskset -c 0-3,64-67 $PY code/track.py --procs 8

For every clock row: find the harvest segment whose 250 m onset is the row's clock
(`onset = MVT − obs_obs_taxi`, same airport, within 0.51 s; the wheels-off time must agree when
the row has one), read the segment's ground points, and compute

- the implied off-block clock of exp-127 unit 2 (first ground point after the last turnaround
  stop, its distance to the surveyed stand, its speed, 3 kt when the speed is missing or below
  1 kt), for every month instead of the three fold months;
- pushback-phase inputs from the motion before the onset: motion against the reported heading
  (pushback), its duration and length, the stop after it, the lead of the first motion to the
  onset, the speed at the onset, the time from the onset to 10 kt, a taxi-out (remote) stand flag.

Every output is a duration, a distance, a speed or a flag of the aircraft's own motion; no point
count, sampling interval or gap is written. Reads no label. Writes `inputs/track.parquet` and
`inputs/track.json` (write-once). Segment rule constants and the anchor are those of
`exp-127 rederive/kaggle/coverage-kernel/rederive_clock.py` (frozen from exp-043/045).
"""
from __future__ import annotations

import argparse
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

KINPUT = C.W148 / "kinput"
GEOMETRY = Path(os.environ.get("EXP159_GEOMETRY", KINPUT / "prc-exp134-input/stand_geometry.csv"))
#: exp-186 (final file): the folders that hold the `prc-harvest-*` outputs, separated by ":"; the default is
#: KINPUT alone. A final run adds the February and June 2026 harvest folder; a day in two folders stops the run.
HARVEST_DIRS = [Path(d) for d in os.environ.get("EXP159_HARVEST_DIRS", str(KINPUT)).split(":") if d]
MOVE_KT, TURNAROUND = 3.0, 600.0
TOL_S = 0.51
PUSH_SPEED_KT = 3.0
STEP_M = 3.0            # a displacement step shorter than this is position jitter
PUSH_MIN_M = 15.0       # reverse motion over at least this length is a pushback
AFTER_S = 600.0         # look for 10 kt up to this long after the onset
LFPG = "LFPG"           # its surveyed stand names failed exp-127's position check
OUT_COLS = ["trk_matched", "trk_implied_taxi", "trk_implied_lead_s", "trk_d_first_m", "trk_v_first_kt",
            "trk_survey_to_runway_km", "trk_survey_to_obs_stand_m", "trk_survey_is_osm",
            "trk_push_detected", "trk_push_duration_s", "trk_push_length_m", "trk_stop_after_push_s",
            "trk_push_start_lead_s", "trk_move_start_lead_s", "trk_onset_speed_kt", "trk_onset_to_10kt_s",
            "trk_remote_proxy", "trk_heading_change_deg"]


# ------------------------------------------------------------------ frozen segment rule
def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if not len(mask):
        return []
    cut = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    edges = [0, *cut.tolist(), len(mask)]
    return [(a, b, bool(mask[a])) for a, b in zip(edges[:-1], edges[1:])]


def _moving(gs: np.ndarray, lat: np.ndarray, lon: np.ndarray, t: np.ndarray) -> np.ndarray:
    mv = gs > MOVE_KT
    miss = np.isnan(gs)
    if miss.any() and len(t) > 1:
        dx = np.gradient(lon) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
        dy = np.gradient(lat) * 110540.0
        dt = np.gradient(t)
        with np.errstate(invalid="ignore", divide="ignore"):
            spd = np.hypot(dx, dy) / np.where(dt == 0, np.nan, dt) / 0.5144
        mv[miss] = spd[miss] > MOVE_KT
    return np.nan_to_num(mv, nan=0.0).astype(bool)


def _span(t: np.ndarray, a: int, b: int) -> float:
    if b - a >= 2:
        return float(t[b - 1] - t[a])
    return float(t[b] - t[a]) if b < len(t) else 0.0


def departure_anchor(ts: np.ndarray, runs: list[tuple[int, int, bool]]) -> int:
    turns = [b for a, b, is_moving in runs if not is_moving and _span(ts, a, b) >= TURNAROUND]
    return turns[-1] - 1 if turns else 0


def haversine_m(lat, lon, lat0: float, lon0: float):
    earth_m = 6_371_008.8
    p1, p2 = np.radians(lat), np.radians(lat0)
    dl = np.radians(np.asarray(lon, dtype="float64") - lon0)
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * earth_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def angdiff(a, b):
    return np.abs((np.asarray(a, "float64") - np.asarray(b, "float64") + 180.0) % 360.0 - 180.0)


# ------------------------------------------------------------------ surveyed stands (exp-127 rule)
def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def stand_tables(geometry: pd.DataFrame):
    exact: dict[tuple[str, str], tuple[float, float, str]] = {}
    for (airport, name), rows in geometry.groupby(["airport", "stand_name_mvt"], sort=True):
        values = rows[["stand_latitude", "stand_longitude", "stand_source"]].drop_duplicates()
        if pd.notna(name) and len(values) == 1:
            r = values.iloc[0]
            exact[(str(airport), str(name))] = (float(r.stand_latitude), float(r.stand_longitude), str(r.stand_source))
    norm: dict[tuple[str, str], list] = {}
    for (airport, name), value in exact.items():
        norm.setdefault((airport, normalize(name)), []).append((name, value))
    unique = {k: v[0] for k, v in norm.items() if k[1] and len(v) == 1}
    runway = {(str(a), str(s), str(r)): float(d) for a, s, r, d in
              geometry[["airport", "stand_name_mvt", "runway_end", "distance_to_runway_end_m"]].itertuples(index=False)}
    return exact, unique, runway


def add_survey(rows: pd.DataFrame, exact, unique, runway) -> pd.DataFrame:
    lat, lon, osm, rwy = [], [], [], []
    for airport, stand, end in zip(rows.airport.astype(str), rows.STAND_mvt, rows.RUNWAY_mvt):
        hit, name = None, None
        if airport != LFPG and pd.notna(stand):
            hit, name = exact.get((airport, str(stand))), str(stand)
            if hit is None:
                item = unique.get((airport, normalize(stand)))
                hit, name = (item[1], item[0]) if item is not None else (None, None)
        if hit is None:
            lat.append(np.nan); lon.append(np.nan); osm.append(np.nan); rwy.append(np.nan)
        else:
            lat.append(hit[0]); lon.append(hit[1]); osm.append(float(hit[2] != "xplane_gateway"))
            rwy.append(runway.get((airport, name, str(end)), np.nan) / 1000.0)
    out = rows.copy()
    out["survey_lat"], out["survey_lon"] = lat, lon
    out["trk_survey_is_osm"], out["trk_survey_to_runway_km"] = osm, rwy
    return out


# ------------------------------------------------------------------ one segment
def features(ts, lat, lon, gs, trk, onset_s: float, mvt_s: float, survey, obs_stand) -> dict:
    out: dict[str, float] = {}
    moving = _moving(gs.copy(), lat, lon, ts)
    anchor = departure_anchor(ts, _runs(moving))
    if anchor >= len(ts):
        return out
    # --- implied off-block clock (exp-127 unit 2)
    v_first = float(gs[anchor]) if np.isfinite(gs[anchor]) else np.nan
    out["trk_v_first_kt"] = v_first
    if np.isfinite(survey[0]):
        d_first = float(haversine_m(lat[anchor], lon[anchor], survey[0], survey[1]))
        v_eff = PUSH_SPEED_KT if (not np.isfinite(v_first) or v_first < 1.0) else v_first
        implied = float(ts[anchor] - d_first / (v_eff * 0.514444))
        out["trk_d_first_m"] = d_first
        out["trk_implied_lead_s"] = float(onset_s - implied)
        out["trk_implied_taxi"] = float(mvt_s - implied)
        if np.isfinite(obs_stand[0]):
            out["trk_survey_to_obs_stand_m"] = float(haversine_m(obs_stand[0], obs_stand[1], survey[0], survey[1]))
    # --- motion before the onset
    k_on = int(np.searchsorted(ts, onset_s - 1e-6))          # first point at or after the onset
    if k_on < len(ts):
        out["trk_onset_speed_kt"] = float(gs[k_on]) if np.isfinite(gs[k_on]) else np.nan
        after = np.flatnonzero((gs[k_on:] >= 10.0) & (ts[k_on:] <= onset_s + AFTER_S))
        if len(after):
            out["trk_onset_to_10kt_s"] = float(ts[k_on + after[0]] - onset_s)
    pre = slice(anchor, min(k_on, len(ts) - 1) + 1)
    t, la, lo = ts[pre], lat[pre], lon[pre]
    if len(t) < 2:
        return out
    dx = np.diff(lo) * 111320.0 * np.cos(np.radians(la[0]))
    dy = np.diff(la) * 110540.0
    step = np.hypot(dx, dy)
    real = step >= STEP_M
    if not real.any():
        return out
    first = int(np.flatnonzero(real)[0])
    out["trk_move_start_lead_s"] = float(onset_s - t[first])   # the last point before the first real step
    head = pd.Series(trk[pre]).ffill().bfill().to_numpy()       # the heading of each point
    if not np.isfinite(head).any():
        return out
    brg = np.degrees(np.arctan2(dx, dy)) % 360.0
    off = angdiff(brg, head[1:])
    rev = real & (off > 120.0)
    fwd = real & (off < 60.0)
    rev_len, fwd_len = float(step[rev].sum()), float(step[fwd].sum())
    out["trk_heading_change_deg"] = float(angdiff(head[first], head[-1]))
    if rev_len >= PUSH_MIN_M:
        idx = np.flatnonzero(rev)
        t_start, t_end = float(t[idx[0]]), float(t[idx[-1] + 1])
        out["trk_push_detected"] = 1.0
        out["trk_push_length_m"] = rev_len
        out["trk_push_duration_s"] = t_end - t_start
        out["trk_push_start_lead_s"] = float(onset_s - t_start)
        later = np.flatnonzero(fwd[idx[-1] + 1:])
        if len(later):
            out["trk_stop_after_push_s"] = float(t[idx[-1] + 1 + later[0]] - t_end)
        out["trk_remote_proxy"] = 0.0
    elif rev_len + fwd_len >= PUSH_MIN_M:
        out["trk_push_detected"] = 0.0
        out["trk_push_length_m"] = 0.0
        at_stand = np.isfinite(obs_stand[0]) and float(haversine_m(la[0], lo[0], obs_stand[0], obs_stand[1])) <= 30.0
        out["trk_remote_proxy"] = float(bool(at_stand) and fwd_len >= 30.0 and not moving[anchor])
    return out


# ------------------------------------------------------------------ one month
def day_files() -> dict[str, Path]:
    files = {}
    for p in (q for d in HARVEST_DIRS for q in d.glob("prc-harvest-*/points_*.parquet")):
        day = p.name[len("points_"):len("points_") + 10]
        if day in files:
            raise RuntimeError(f"{day}: points in two harvest outputs")
        files[day] = p
    return files


def month_job(args) -> pd.DataFrame:
    month, rows, files = args
    rows = rows.copy()
    rows["onset_s"] = rows.mvt_s - rows.obs_obs_taxi
    rows["wheels_s"] = rows.onset_s + rows.obs_obs_seg_taxi
    rows["day"] = pd.to_datetime(rows.onset_s, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    recs = []
    stats = {"rows": len(rows), "no_day_file": 0, "no_segment": 0, "ambiguous": 0, "matched": 0}
    for day, grp in rows.groupby("day", sort=True):
        if day not in files:
            stats["no_day_file"] += len(grp)
            continue
        seg = pd.read_parquet(files[day].with_name(f"segments_{day}.parquet"),
                              columns=["airport", "icao", "seg_start", "seg_end", "obs_offblock",
                                       "obs_wheels_off", "stand_lat", "stand_lon"])
        seg = seg[seg.obs_offblock.notna()].reset_index(drop=True)
        pts = pd.read_parquet(files[day], columns=["airport", "icao", "t", "lat", "lon", "gs_kt", "track", "on_ground"])
        pts = pts[pts.on_ground].sort_values("t", kind="stable")
        by_ac = {k: v for k, v in pts.groupby(["airport", "icao"], sort=False, observed=True)}
        for airport, g in grp.groupby("airport", sort=False):
            s = seg[seg.airport == airport].sort_values("obs_offblock", kind="stable")
            times = s.obs_offblock.to_numpy("float64")
            for r in g.itertuples(index=False):
                lo_, hi_ = np.searchsorted(times, [r.onset_s - TOL_S, r.onset_s + TOL_S])
                cand = s.iloc[lo_:hi_]
                if len(cand) > 1:                           # the wheels-off time separates two aircraft
                    w = cand.obs_wheels_off.to_numpy("float64")
                    ok = (np.isfinite(w) == np.isfinite(r.wheels_s)) & ~(np.abs(w - r.wheels_s) > TOL_S)
                    cand = cand[ok]
                if len(cand) != 1:
                    stats["no_segment" if len(cand) == 0 else "ambiguous"] += 1
                    continue
                c = cand.iloc[0]
                p = by_ac.get((airport, c.icao))
                if p is None:
                    stats["no_segment"] += 1
                    continue
                tt = p.t.to_numpy("float64")
                a, b = np.searchsorted(tt, [c.seg_start - 1e-6, c.seg_end + 1e-6])
                if b - a < 1:
                    stats["no_segment"] += 1
                    continue
                f = features(tt[a:b], p.lat.to_numpy("float64")[a:b], p.lon.to_numpy("float64")[a:b],
                             p.gs_kt.to_numpy("float64")[a:b], p.track.to_numpy("float64")[a:b],
                             float(r.onset_s), float(r.mvt_s), (r.survey_lat, r.survey_lon),
                             (float(c.stand_lat), float(c.stand_lon)))
                f[C.ID] = r.MVT_ID_mvt
                f["trk_matched"] = 1.0
                recs.append(f)
                stats["matched"] += 1
    out = pd.DataFrame(recs, columns=[C.ID] + OUT_COLS) if recs else pd.DataFrame(columns=[C.ID] + OUT_COLS)
    out.attrs["stats"] = stats
    print(month, stats, flush=True)
    return out, month, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=C.TRACK)
    a = ap.parse_args()
    rec_path = a.out.with_suffix(".json")
    C.refuse_existing(a.out, rec_path)
    t0 = time.time()
    cols = [C.ID, "month", "airport", "mvt_s", "obs_obs_taxi", "obs_obs_seg_taxi", "STAND_mvt", "RUNWAY_mvt",
            "v23_implied_offblock_to_current250_s", "v23_survey_to_obs_stand_m", "v23_survey_to_runway_km"]
    frame = pd.read_parquet(C.FRAME, columns=cols)
    if a.months:
        frame = frame[frame.month.isin(a.months)]
    geometry = pd.read_csv(GEOMETRY)
    frame = add_survey(frame, *stand_tables(geometry))
    files = day_files()
    jobs = [(m, g, files) for m, g in frame.groupby("month", sort=True)]
    with Pool(min(a.procs, len(jobs))) as pool:
        results = pool.map(month_job, jobs, chunksize=1)
    track = pd.concat([r[0] for r in results], ignore_index=True)
    track = frame[[C.ID, "trk_survey_is_osm", "trk_survey_to_runway_km"]].merge(
        track.drop(columns=["trk_survey_is_osm", "trk_survey_to_runway_km"]), on=C.ID, how="left",
        validate="one_to_one")
    track["trk_matched"] = track.trk_matched.fillna(0.0)
    track = track[[C.ID] + OUT_COLS]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    track.to_parquet(a.out, index=False)

    j = frame.merge(track, on=C.ID, suffixes=("", "_t"))
    by = j.groupby("month")
    cover = {c: by[c].apply(lambda s: float(s.notna().mean())).round(4).to_dict() for c in OUT_COLS[1:]}
    cover["trk_matched"] = by["trk_matched"].mean().round(4).to_dict()
    both = j.v23_implied_offblock_to_current250_s.notna() & j.trk_implied_lead_s.notna()
    diff = (j.trk_implied_lead_s - j.v23_implied_offblock_to_current250_s)[both]
    parity = {"rows_v23": int(j.v23_implied_offblock_to_current250_s.notna().sum()),
              "rows_both": int(both.sum()),
              "rows_v23_only": int((j.v23_implied_offblock_to_current250_s.notna() & j.trk_implied_lead_s.isna()).sum()),
              "share_within_0.01s": float((diff.abs() <= 0.01).mean()) if len(diff) else None,
              "share_within_1s": float((diff.abs() <= 1.0).mean()) if len(diff) else None,
              "abs_diff_p99": float(diff.abs().quantile(0.99)) if len(diff) else None}
    rec = {"what": "exp-159 track inputs of the clock rows (label-free)", "rows": int(len(track)),
           "months": sorted(frame.month.unique().tolist()),
           "match": {m: s for _, m, s in results}, "coverage_by_month": cover,
           "parity_with_exp127_v23_implied": parity,
           "inputs_sha256": {"frame": C.sha(C.FRAME), "stand_geometry": C.sha(GEOMETRY)},
           "point_files": len(files), "output_sha256": C.sha(a.out), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"track.py": C.sha(__file__), "common.py": C.sha(C.__file__)}}
    C.write_json(rec_path, rec)
    print(pd.DataFrame(cover).round(3).to_string())
    print("parity with exp-127:", parity)


if __name__ == "__main__":
    main()
