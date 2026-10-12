"""exp-047-queue: the runway-state block, one row per departure.

`state.py` holds the engine and the argument for what is new about it.  This file
builds the three event sources it reads and the reference instant it reads them at,
and writes the block as a parquet keyed by `MVT_ID_mvt` so `exp-046-surfacefit` can
take it as an input (`assignments/queue.md` step 3).

**The reference instant.**  `assignments/queue.md`: "the observed off-block for a
matched row; for an unmatched row the instant the serving path may read under
`docs/input-rule.md`".  The choice, and the readings rejected, are:

    t_ref = obs_offblock            where the harvest heard the aircraft push back
          | AOBT_3_flt              else, where NM saw it (98.5 % of ranking rows)
          | MVT_TIME - m(apt, hour) else; `m` is the median of `MVT_TIME - AOBT_3`
                                    over the OTHER rows of the same airport and hour
                                    of the same masked month -- no label anywhere

rejected: (a) the take-off instant alone, which is the instant the shipped block
already reads and would re-measure exp-041's negative -- it is kept, but as the
CONTROL arm `F@to`, never as the block's own anchor; (b) `LOBT_flt`/`EOBT_1_flt`, the
planned off-block, because the queue is a realised state and the realised clock is
available on 98.5 % of rows; (c) dropping the rows with no clock of their own, which
are 37 % of the ranking file and the population this lane exists for.

**No column is a function of `MVT_TIME - t_ref`.**  For a matched row that difference
IS exp-046's observed taxi, so a block carrying it would measure that lane's feature
and not the runway state.  `--audit` asserts it column by column.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import state as S                                            # noqa: E402
from imports import load as _load                            # noqa: E402

B46 = _load("exp-046-surfacefit", "block")                   # the repo's one matcher
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

AIRPORTS = ["EHAM", "LFPG", "EGLL", "LTFM", "LEMD", "EDDF", "LEBL", "LIRF",
            "EDDM", "LSZH"]
RUNWAYS = ROOT / "experiments/exp-045-harvest/analysis/runways-10.csv"
FRAME_COLS = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
              "RUNWAY_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
              "TAXITIME_SEC_mvt", "AOBT_3_flt", "CALLSIGN_flt",
              # the DEPARTURE side of the turnaround chain (desk wiring rule,
              # 2026-09-09): `B46.build` reads the candidate departure's stand, hashed
              # operator and aircraft type, so the frame has to carry them.  Nothing in
              # this lane reads either column; they pass through to the one matcher.
              "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
              # the CELL's regulation signature (desk, 2026-09-09): `regnet_block` reads
              # `AOBT_3_flt - EOBT_1_flt` over the cell, the operator-day and the
              # airport-day.  `EOBT_1_flt` is UNMASKED in the ranking file, so the family
              # is servable; `mask_like_ranking` blanks only the block time and the target.
              "EOBT_1_flt"]
SOURCES = ("F", "O", "H")
GEO_COLS = ["route_km", "route_km_rel", "stand_dist_km"]

#: THE MATCHER THIS LANE IS BUILT AGAINST (desk MATCHER_SHA line, 2026-09-09).  It is
#: DECLARED, not discovered: every record still records the sha it hashed at run time, but
#: a matcher that moves under a running lane now stops the run instead of quietly writing
#: rows from a different matching function.  `MATCHER_SHA_BEFORE_THE_REBUILD` is what v14
#: was built and scored on, kept so a record can say which side of the rebuild it is from.
MATCHER_SHA = "122c7d84571b53e02de26e53add8e783bc95be9f0eaea5b6ef6c155280aad780"
MATCHER_SHA_BEFORE_THE_REBUILD = "695af973"
#: what v17 was built and scored on, before the turnaround pass (desk MATCHER_SHA line,
#: 2026-09-09, `exp-046-surfacefit` at `d92aec3`)
MATCHER_SHA_BEFORE_THE_CHAIN = (
    "6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b")

#: THE WIRING RULE (desk, 2026-09-09).  `B46.build` runs the turnaround pass ONLY when it
#: is handed the month's ARR rows; without them it is a silent no-op -- no error, and 3,179
#: clocks simply missing.  So the arrival side is not optional here and the count is
#: asserted against `exp-046-surfacefit`'s own reference, per month, on every build.
#: Ranking months: 1,382 (2026-01) and 1,797 (2026-07).  Fold months: 1,170 / 861 / 731,
#: given by the desk as a set without naming which month is which, so the assertion is on
#: the SET and the mapping this lane measures is recorded beside it.
MODE3_REFERENCE = {"2026-01": 1382, "2026-07": 1797}
MODE3_REFERENCE_FOLD_SET = (1170, 861, 731)
#: NO BLOCK CACHE LIVES IN THIS LANE -- every block is rebuilt from the month and the
#: summaries on each run, which is why the trap surfacefit hit (a cache keyed on the
#: POPULATION alone returns pre-rebuild rows when only the matching function moved) cannot
#: bite here.  If one is ever added: the matcher sha goes in the cache FILENAME, and never
#: into the population key the pins record.
BLOCK_CACHE = None


def check_matcher(strict: bool = True) -> str:
    """Hash the matcher on disk and stop the run if it is not the declared one."""
    got = sha_of(ROOT / "experiments/exp-046-surfacefit/code/block.py")
    if strict and got != MATCHER_SHA:
        raise SystemExit(
            f"the matcher on disk is {got[:12]}… and this lane declares "
            f"{MATCHER_SHA[:12]}… (code/block.py MATCHER_SHA); re-read the desk's "
            "MATCHER_SHA line before building anything")
    return got


def _epoch(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, utc=True) - pd.Timestamp(0, tz="UTC")
            ).dt.total_seconds().to_numpy("float64")


def sha_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ the keys

def keys(frame: pd.DataFrame) -> dict:
    """Airport and airport|runway integer codes, and the runway table `state` needs."""
    apt = pd.Categorical(frame["ADEP_mvt"].astype(str), categories=AIRPORTS)
    home = np.where(frame["PHASE_mvt"].to_numpy() == "DEP",
                    frame["ADEP_mvt"].astype(str).to_numpy(),
                    frame["ADES_mvt"].astype(str).to_numpy())
    hcode = pd.Categorical(home, categories=AIRPORTS).codes.astype("int64")
    rk = pd.Series(home).astype(str) + "|" + frame["RUNWAY_mvt"].astype(str).to_numpy()
    rcat = pd.Categorical(rk)
    rcode = rcat.codes.astype("int64")
    # airport code -> its runway codes, padded with -1
    tab = pd.DataFrame({"a": hcode, "r": rcode}).drop_duplicates()
    tab = tab[(tab.a >= 0) & (tab.r >= 0)]
    width = int(tab.groupby("a").size().max()) if len(tab) else 1
    m = np.full((len(AIRPORTS), width), -1, "int64")
    for a, g in tab.groupby("a"):
        v = g.r.to_numpy()
        m[a, :len(v)] = v
    # an extra runway code per airport for observed segments whose runway is unknown
    unk = {a: len(rcat.categories) + a for a in range(len(AIRPORTS))}
    del apt
    return {"apt": hcode, "rwy": rcode, "rwy_of_apt": m,
            "rwy_categories": list(rcat.categories), "unknown_rwy": unk}


def runway_geometry() -> pd.DataFrame:
    """Departure threshold and heading per `airport|ident`, from OurAirports."""
    r = pd.read_csv(RUNWAYS)
    rows = []
    for side, other in (("le", "he"), ("he", "le")):
        rows.append(pd.DataFrame({
            "airport": r.airport_ident, "ident": r[f"{side}_ident"].astype(str),
            "thr_lat": r[f"{side}_latitude_deg"], "thr_lon": r[f"{side}_longitude_deg"],
            "end_lat": r[f"{other}_latitude_deg"], "end_lon": r[f"{other}_longitude_deg"],
            "hdg": r[f"{side}_heading_degT"], "closed": r.closed}))
    g = pd.concat(rows, ignore_index=True)
    g = g[(g.closed == 0) & g.thr_lat.notna()].copy()
    g["key"] = g.airport + "|" + g.ident
    return g.drop_duplicates("key").set_index("key")


def km(lat1, lon1, lat2, lon2):
    dy = (np.asarray(lat2, "float64") - np.asarray(lat1, "float64")) * 110.574
    dx = ((np.asarray(lon2, "float64") - np.asarray(lon1, "float64")) * 111.320
          * np.cos(np.radians((np.asarray(lat1, "float64")
                               + np.asarray(lat2, "float64")) / 2)))
    return np.hypot(dx, dy)


# --------------------------------------------------------------- the sources

def file_events(frame: pd.DataFrame, k: dict) -> S.Events:
    """Source F: the clocks the ranking file itself carries.

    Departures: `AOBT_3_flt` -> `MVT_TIME_UTC_mvt`.  Arrivals: `MVT_TIME_UTC_mvt`
    (touchdown) -> `BLOCK_TIME_UTC_mvt` (on-block), which arrivals keep in the ranking
    file, with `TAXITIME_SEC_mvt` as the observed taxi-in.
    """
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    mvt = _epoch(frame["MVT_TIME_UTC_mvt"])
    blk = _epoch(frame["BLOCK_TIME_UTC_mvt"])
    aobt = _epoch(frame["AOBT_3_flt"])
    taxi = frame["TAXITIME_SEC_mvt"].to_numpy("float64")
    return S.Events(k["apt"][isdep], k["rwy"][isdep], aobt[isdep], mvt[isdep],
                    k["apt"][~isdep], k["rwy"][~isdep], mvt[~isdep], blk[~isdep],
                    taxi[~isdep], k)


def observed_events(frame: pd.DataFrame, k: dict, seg: pd.DataFrame,
                    matched_seg: pd.DataFrame, hybrid: bool) -> tuple[S.Events, dict]:
    """Sources O and H: the departure side from the adsb.lol ground observations.

    The ARRIVAL side is the file's in both, because an arrival keeps every column in
    the ranking file -- its landing time, its on-block time and its taxi-in are all
    given, so there is nothing for an observation to add.  Only the departure side
    changes.

    O: every received ground run that is not an arrival and carries an off-block
       instant, `obs_offblock` -> `obs_wheels_off` (or `t_airborne`, or `seg_end`).
       Its runway is the matched departure's `RUNWAY_mvt` where the matcher found one,
       and an airport-specific UNKNOWN code otherwise, so an unidentified aircraft
       still counts in the airport queue and never in a runway queue.
    H: O, plus every file departure the network did not hear, on its file clock.
    """
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    mvt = _epoch(frame["MVT_TIME_UTC_mvt"])
    blk = _epoch(frame["BLOCK_TIME_UTC_mvt"])
    aobt = _epoch(frame["AOBT_3_flt"])
    taxi = frame["TAXITIME_SEC_mvt"].to_numpy("float64")

    d = seg[seg.kind.ne("arrival") & seg.obs_offblock.notna()].copy()
    end = d.obs_wheels_off.where(d.obs_wheels_off.notna(), d.t_airborne)
    d["ev_end"] = end.where(end.notna(), d.seg_end)
    # the runway of an observed run: the matched departure's, else UNKNOWN.  The
    # join key is `(airport, icao, obs_offblock rounded to the second)`, which is the
    # identity of a ground run and is what `exp-046`'s matcher hands back through
    # `obs_icao` and `obs_taxi` without needing that lane's internal segment index.
    acode = pd.Categorical(d.airport.astype(str), categories=AIRPORTS).codes.astype("int64")
    d["ev_key"] = (d.airport.astype(str) + "|" + d.icao.astype(str) + "|"
                   + np.rint(d.obs_offblock.to_numpy("float64")).astype("int64").astype(str))
    rcode = np.array([k["unknown_rwy"][a] if a >= 0 else -1 for a in acode], "int64")
    n_known = 0
    if len(matched_seg):
        mm = (matched_seg.dropna(subset=["ev_key"]).drop_duplicates("ev_key")
              .set_index("ev_key")["rwy_code"])
        got = pd.to_numeric(mm.reindex(d.ev_key.to_numpy()), errors="coerce").to_numpy()
        ok = np.isfinite(got)
        rcode = np.where(ok, np.nan_to_num(got, nan=-1).astype("int64"), rcode)
        n_known = int(ok.sum())
    dep_apt = acode
    dep_rwy = rcode
    dep_s = d.obs_offblock.to_numpy("float64")
    dep_e = d.ev_end.to_numpy("float64")
    ekeys = list(d.ev_key.to_numpy())
    n_obs = len(d)
    n_file_added = 0
    if hybrid:
        # "heard" is `obs_has_clock`, not `obs_matched`: the O event set holds the
        # ground runs that carry an off-block instant, so a row matched to a run with
        # no clock is NOT in it and the hybrid must still add its file event.  Keying
        # this on membership of `matched_seg` made every row heard and H identical to
        # O on the first run.
        heard = np.zeros(int(isdep.sum()), bool)
        if len(matched_seg):
            hid = set(matched_seg.loc[matched_seg.obs_has_clock
                                      & matched_seg.ev_key.notna(), "MVT_ID_mvt"])
            heard = frame.loc[isdep, "MVT_ID_mvt"].isin(hid).to_numpy()
        add = ~heard
        n_file_added = int(add.sum())
        dep_apt = np.concatenate([dep_apt, k["apt"][isdep][add]])
        dep_rwy = np.concatenate([dep_rwy, k["rwy"][isdep][add]])
        dep_s = np.concatenate([dep_s, aobt[isdep][add]])
        dep_e = np.concatenate([dep_e, mvt[isdep][add]])
        ekeys += ["file|" + str(int(v)) for v in
                  frame.loc[isdep, "MVT_ID_mvt"].to_numpy()[add]]
    ev = S.Events(dep_apt, dep_rwy, dep_s, dep_e,
                  k["apt"][~isdep], k["rwy"][~isdep], mvt[~isdep], blk[~isdep],
                  taxi[~isdep], k)
    #: the identity of every departure event, so a query row can find its OWN and
    #: subtract it: the ground-run key for an observed run, `file|<MVT_ID>` for a
    #: file run the hybrid appended.
    ev.ev_keys = np.asarray(ekeys, dtype=object)
    return ev, {"observed_runs": n_obs, "runway_known": n_known,
                "file_runs_added": n_file_added}


# ---------------------------------------------------------------- the anchor

def anchor(frame: pd.DataFrame, obs_offblock: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """`t_ref` per departure row, and the census of where each row's clock came from.

    Nothing here reads a label: `AOBT_3_flt` and `MVT_TIME_UTC_mvt` are ranking
    columns, `obs_offblock` is an observation, and the airport-hour fallback offset is
    a median of `MVT_TIME - AOBT_3` over the OTHER rows of the same masked month.
    """
    dep = frame[frame["PHASE_mvt"] == "DEP"]
    mvt = _epoch(dep["MVT_TIME_UTC_mvt"])
    aobt = _epoch(dep["AOBT_3_flt"])
    apt = dep["ADEP_mvt"].astype(str).to_numpy()
    hour = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.hour.to_numpy()
    diff = mvt - aobt
    kdf = pd.DataFrame({"apt": apt, "hour": hour, "diff": diff})
    # leave-one-out median is not defined cheaply; the median of a group of thousands
    # is insensitive to one row, and the fallback only ever applies to rows whose own
    # `diff` is NaN and so are not in the group at all
    med = kdf.dropna(subset=["diff"]).groupby(["apt", "hour"])["diff"].median()
    ameda = kdf.dropna(subset=["diff"]).groupby("apt")["diff"].median()
    fb = pd.MultiIndex.from_arrays([apt, hour])
    off = med.reindex(fb).to_numpy("float64")
    off = np.where(np.isfinite(off), off, pd.Series(apt).map(ameda).to_numpy("float64"))
    off = np.where(np.isfinite(off), off, float(np.nanmedian(diff)))
    t = np.where(np.isfinite(aobt), aobt, mvt - off)
    src = np.where(np.isfinite(aobt), "nm", "median")
    if obs_offblock is not None:
        use = np.isfinite(obs_offblock)
        t = np.where(use, obs_offblock, t)
        src = np.where(use, "observed", src)
    return t, {"rows": int(len(t)),
               **{f"anchor_{k}": int((src == k).sum())
                  for k in ("observed", "nm", "median")}}


# --------------------------------------------------------------- the matcher

def _sibling_arms():
    """`exp-046-surfacefit/code/arms.py`, loaded against ITS OWN `base` and `block`.

    Both lanes ship a `base` and a `block`, and by the time the arrival side is wanted
    this lane's are already in `sys.modules` under those plain names -- so a plain load of
    the sibling's `arms` binds it to THIS lane's `base` and dies on `V12_BOARD`.  The
    plain names are stashed across the load and put back after it; the sibling's module
    keeps its own references, so one load is enough and `imports.load` caches it.
    """
    keys = ("base", "block", "state", "population", "pin", "frames")
    stash = {k: sys.modules.pop(k) for k in keys if k in sys.modules}
    try:
        return _load("exp-046-surfacefit", "arms")
    finally:
        for k in keys:
            sys.modules.pop(k, None)
        sys.modules.update(stash)


def months_of(frame: pd.DataFrame) -> list[str]:
    """The calendar months of `frame`'s DEPARTURES, in order."""
    m = frame.loc[frame["PHASE_mvt"] == "DEP", "MVT_TIME_UTC_mvt"]
    return sorted(set(m.dt.strftime("%Y-%m").dropna()))


def arrivals_for(frame: pd.DataFrame) -> pd.DataFrame | None:
    """The ARR rows the turnaround pass chains through, for `frame`'s own months.

    ONE function per side, not a re-implementation (desk wiring rule, 2026-09-09):

      * a 2025 FOLD month is `exp-046-surfacefit/code/arms.py`'s `month_arrivals`, which
        masks like the ranking file and asserts the arrival in-block survives the mask;
      * a 2026 RANKING month has no movement file, so it is the ARR slice of the ranking
        file for that month -- the same expression `exp-046-surfacefit/code/serve.py`
        takes at its own build (its lines 193-195), so the fold side and both lanes'
        serves compute one and the same function.

    The stand chain reads `STAND_mvt`, `AIRCRAFT_OPERATOR_flt`, `AIRCRAFT_TYPE_mvt` and
    the ARRIVAL's `BLOCK_TIME_UTC_mvt`.  `FRAME_COLS` gained the operator and the type
    for the DEPARTURE side of the same chain; the arrival side is still loaded here rather
    than sliced out of `frame`, because a fold month's arrivals come from the sibling's
    own function and a ranking month's from the ranking file.
    """
    months = months_of(frame)
    if not months:
        return None
    fold = [m for m in months if not m.startswith("2026-")]
    rank = [m for m in months if m.startswith("2026-")]
    parts = []
    if fold:
        parts += [_sibling_arms().month_arrivals(m) for m in fold]
    if rank:
        rk = D.load_ranking()
        parts += [rk[rk.PHASE_mvt.eq("ARR")
                     & rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").isin(rank)]]
    arr = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    if not arr["BLOCK_TIME_UTC_mvt"].notna().any():
        raise SystemExit(f"{months}: the arrival in-block is blank; the pass cannot chain")
    return arr.reset_index(drop=True)


def check_mode3(b: pd.DataFrame, months: list[str]) -> dict:
    """The turnaround pass ran, and placed what the sibling says it places.

    Fail-closed, because the failure this guards is SILENT: a build handed no arrivals
    returns a complete block with 3,179 clocks missing and no error anywhere.
    """
    mode = b["match_mode"].to_numpy("float64") if "match_mode" in b else np.array([])
    n = int(np.nansum(mode == 3.0))
    rec = {"months": months, "mode3_rows": n,
           "attrs": {k: v for k, v in b.attrs.get("turnaround", {}).items()}}
    if len(months) == 1 and months[0] in MODE3_REFERENCE:
        want = MODE3_REFERENCE[months[0]]
        rec["reference"] = want
        if n != want:
            raise SystemExit(
                f"{months[0]}: the turnaround pass placed {n} rows, the desk's reference "
                f"is {want}. A build handed no arrivals is a SILENT no-op -- check "
                "`arrivals_for` reached `B46.build` before reading any number off this run")
    elif len(months) == 1:
        rec["reference_set"] = list(MODE3_REFERENCE_FOLD_SET)
        rec["in_the_fold_reference_set"] = n in MODE3_REFERENCE_FOLD_SET
        if n == 0:
            raise SystemExit(
                f"{months[0]}: the turnaround pass placed nothing. Without `arrivals=` it "
                "is a silent no-op; with them a fold month places one of "
                f"{MODE3_REFERENCE_FOLD_SET}")
    return rec


def matched(frame: pd.DataFrame, train_for_prefix: pd.DataFrame, k: dict,
            require_rule: str | None, wide: bool = True) -> tuple[pd.DataFrame, dict]:
    """`exp-046-surfacefit/code/block.py`'s match, re-keyed for the event join.

    The repo has one matcher and this lane does not write a second one; it is imported
    read-only and its file's SHA-256 is recorded beside every number, because the
    sibling worker may edit it while this lane runs.
    """
    arr = arrivals_for(frame)
    b = B46.build(frame, train_for_prefix, airports=AIRPORTS, wide=wide,
                  require_rule=require_rule, arrivals=arr)
    turnaround = check_mode3(b, months_of(frame))
    dep = frame[frame["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    if not np.array_equal(b["MVT_ID_mvt"].to_numpy("float64"),
                          dep["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the matcher returned rows in another order")
    mvt = _epoch(dep["MVT_TIME_UTC_mvt"])
    ob = mvt - b["obs_taxi"].to_numpy("float64")
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    out = pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy("float64"),
        "airport": dep["ADEP_mvt"].astype(str).to_numpy(),
        "day": dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
        "icao": b["obs_icao"].astype(str).to_numpy(),
        "obs_offblock": ob,
        "rwy_code": k["rwy"][isdep],
        "stand_lat": b["stand_lat"].to_numpy("float64"),
        "stand_lon": b["stand_lon"].to_numpy("float64"),
        "obs_matched": b["obs_matched"].to_numpy(bool),
        "obs_has_clock": b["obs_has_clock"].to_numpy(bool),
        "reach_class": b["reach_class"].astype(str).to_numpy()})
    out["ev_key"] = np.where(
        np.isfinite(ob),
        out.airport + "|" + out.icao + "|"
        + np.rint(np.nan_to_num(ob, nan=0.0)).astype("int64").astype(str), None)
    rep = {"matcher_sha256": sha_of(ROOT / "experiments/exp-046-surfacefit/code/block.py"),
           "rows": int(len(out)), "matched": int(out.obs_matched.sum()),
           "with_clock": int(out.obs_has_clock.sum()),
           "turnaround": turnaround,
           "days_present": b.attrs.get("days_present"),
           "days_wrong_rule": len(b.attrs.get("days_wrong_rule", [])),
           "rules_seen": b.attrs.get("rules_seen"),
           "by_airport": {a: {"rows": int(len(g)), "clock": int(g.obs_has_clock.sum())}
                          for a, g in out.groupby("airport")}}
    return out, rep


# ------------------------------------------------- outages in the reception

#: The desk, 2026-09-08 14:45 UTC, from the harvest: *"an EDDF feeder OUTAGE on Jan
#: 27-31 that must not be read as normal reception: treat those airport-days as
#: no-clock for the class boundary and the receipt scaling"*.
#:
#: **This lane keeps no table of its own** (desk, 15:02 UTC).  `exp-046-surfacefit`'s
#: `block.py` now holds those airport-days inside `load_summaries` itself, at matcher
#: sha `a0652dff…`: the segments never load, so no row matches on them, no segment of
#: theirs feeds a neighbour, and the receipt denominator never counts them.  Every one
#: of the three things this lane's own table used to do is therefore done upstream, by
#: the ONE matcher both lanes import -- which is the point, because a second list here
#: could drift from that one and put the two lanes' class boundaries out of step.  The
#: set is read from `B46.HELD_AIRPORT_DAYS` and what survives here is the AUDIT LABEL
#: (`reach_class == "outage"`, so a consumer can tell a held row from an ordinarily
#: unmatched one) and the ASSERTION that the upstream hold really did leave no clock.
#:
#: The warning below is this lane's own and stays: `outage_report` lists every
#: airport-day that looks anomalous and is NOT held, in every build record, so the next
#: outage cannot pass silently.  It is deliberately not a rule -- a threshold of the
#: form "below half the airport's own median" also flags LFPG, whose 24 % declining to
#: 5 % is that airport's normal reception and not an outage.
#: an undeclared airport-day is WARNED about below this fraction of the airport's own
#: month median; it is never excluded on this alone
OUTAGE_WARN_FRAC = 0.5
#: an airport-day with fewer than this many departures is too small to read a receipt
#: rate off, and falls back to the airport's rate over its harvested days
MIN_DAY_ROWS = 30
RECEIPT_JSON = ROOT / "experiments/exp-045-harvest/analysis/receipt.json"


def outage_set(months: list[str]) -> set[tuple[str, str]]:
    """The `(airport, day)` pairs `exp-046-surfacefit`'s matcher holds, over `months`."""
    held = getattr(B46, "HELD_AIRPORT_DAYS", {}) or {}
    return {(a, d) for (a, d) in held if d[:7] in months}


def outage_report(months: list[str], declared: set[tuple[str, str]]) -> dict:
    """The declared set, and every undeclared airport-day that looks like one."""
    rep = {"declared": sorted(f"{a}|{d}" for a, d in declared),
           "warn_frac": OUTAGE_WARN_FRAC, "undeclared_warnings": [],
           "source": "exp-046-surfacefit/code/block.py HELD_AIRPORT_DAYS "
                     "(desk 2026-09-08 14:45 and 15:02 UTC); the rates are "
                     "exp-045-harvest/analysis/receipt.json by_airport_day",
           "held_reason": getattr(B46, "HELD_REASON", None)}
    if not RECEIPT_JSON.exists():
        rep["receipt_json"] = "missing"
        return rep
    rec = json.loads(RECEIPT_JSON.read_text())
    for m in months:
        by = (rec.get(m) or {}).get("by_airport_day") or {}
        for a, series in by.items():
            vals = [v for v in series.values() if v is not None]
            if not vals:
                continue
            med = float(np.median(vals))
            if med <= 0:
                continue
            for d, v in series.items():
                if v is not None and v < OUTAGE_WARN_FRAC * med \
                        and (a, d) not in declared:
                    rep["undeclared_warnings"].append(
                        {"airport": a, "day": d, "rate": v, "airport_median": round(med, 4)})
    return rep


# ---------------------------------------------------------------- the block

def receipt_rates(mat: pd.DataFrame,
                  present: set | None = None) -> tuple[pd.Series, dict]:
    """Share of an airport's departures whose ground run was identified.

    Label-free: it is a property of the match, not of the target.  A count divided by
    it reads on the scale the airport would show at full receipt, so a 16 % airport
    and a 98 % one enter the same column comparably.

    **The denominator is the HARVESTED airport-days, not the month** (FALLBACK review
    round 1, blocking 2).  A row on a day the harvest has not delivered has no clock
    by construction, so counting it drags the rate down and multiplies every `_sc`
    column by the wrong constant: on the 2025-07 emit, EHAM read 0.2153 (4,841 clocks
    over the whole month) where its harvested week is heard at 0.94, so `_sc` was the
    raw count times 4.65.  Worse, the constant then differs between two months with
    different coverage, which is exactly what `_sc` exists to remove.  Rows on
    unharvested days keep a rate of NaN, so their `_sc` columns are NaN rather than a
    count scaled by a rate measured somewhere else.

    **The grain is the AIRPORT-DAY** (desk, 2026-09-08 15:05 UTC: *"receipt scaling is
    per airport-day, so dips carry fewer neighbours and nothing else"*).  That is what
    makes a one-day reception dip need no special handling: EGLL on 2026-01-01 is
    divided by EGLL's rate ON 2026-01-01, so the column already reads on the scale that
    day was heard at, and the only thing a dip costs is the neighbours it did not
    deliver.  A day too small to estimate on falls back to the airport's own rate over
    its harvested days, and the record says which rows took the fallback.  The returned
    Series is indexed by `(airport, day)`; `airport_rate` beside it is the fallback.
    """
    if present is None:
        r = mat.groupby("airport")["obs_has_clock"].mean()
        return r, {"denominator": "whole_month", "grain": "airport",
                   "rate": {a: round(float(v), 4) for a, v in r.items()}}
    key = pd.Series(list(zip(mat["airport"], mat["day"])), index=mat.index)
    onday = key.isin(present).to_numpy()
    sub = mat[onday]
    apt_rate = (sub.groupby("airport")["obs_has_clock"].mean() if len(sub)
                else pd.Series(dtype="float64"))
    if len(sub):
        g = sub.groupby(["airport", "day"])["obs_has_clock"].agg(["mean", "size"])
        small = g["size"] < MIN_DAY_ROWS
        day_rate = g["mean"].where(~small,
                                   g.index.get_level_values(0).map(apt_rate))
    else:
        day_rate = pd.Series(dtype="float64")
        small = pd.Series(dtype=bool)
    rep = {"denominator": "harvested_airport_days", "grain": "airport_day",
           "min_day_rows": MIN_DAY_ROWS,
           "rows_on_harvested_days": int(onday.sum()),
           "rows_off_harvested_days": int((~onday).sum()),
           "airport_days_below_min_rows": int(small.sum()) if len(small) else 0,
           "rate": {a: round(float(v), 4) for a, v in apt_rate.items()},
           "rate_by_airport_day_spread": {
               a: {"min": round(float(v.min()), 4), "median": round(float(v.median()), 4),
                   "max": round(float(v.max()), 4), "days": int(len(v))}
               for a, v in (day_rate.groupby(level=0) if len(day_rate)
                            else pd.Series(dtype="float64").groupby(level=0))},
           "days_by_airport": {a: int(len(set(d for aa, d in present if aa == a)))
                               for a in sorted(set(mat["airport"]))}}
    return day_rate, rep


def geometry(mat: pd.DataFrame, dep: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """The row's stand-to-runway route, from OTHER aircraft's observed stands.

    The stand position is the centroid of the observed stand positions of the rows
    that share this row's `STAND_mvt`, **with this row's own contribution removed**, so
    no column reads the row's own segment.  The runway threshold is open OurAirports
    geometry (`exp-045-harvest/analysis/boxes.json` cites the same source and hashes).
    """
    key = (dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str)).to_numpy()
    lat = mat["stand_lat"].to_numpy("float64")
    lon = mat["stand_lon"].to_numpy("float64")
    ok = np.isfinite(lat) & np.isfinite(lon)
    code, n = pd.factorize(key)
    n = len(n)
    slat = np.bincount(code, weights=np.where(ok, lat, 0.0), minlength=n)
    slon = np.bincount(code, weights=np.where(ok, lon, 0.0), minlength=n)
    cnt = np.bincount(code, weights=ok.astype("float64"), minlength=n)
    num_lat = slat[code] - np.where(ok, lat, 0.0)
    num_lon = slon[code] - np.where(ok, lon, 0.0)
    den = cnt[code] - ok.astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        clat = np.where(den > 0, num_lat / den, np.nan)
        clon = np.where(den > 0, num_lon / den, np.nan)
    rk = (dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
    thr = geo.reindex(rk)
    route = km(clat, clon, thr["thr_lat"].to_numpy("float64"),
               thr["thr_lon"].to_numpy("float64"))
    apt = dep["ADEP_mvt"].astype(str).to_numpy()
    med = pd.Series(route).groupby(apt).transform("median").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = route / np.where(med > 0, med, np.nan)
    return pd.DataFrame({"route_km": route, "route_km_rel": rel,
                         "stand_dist_km": km(clat, clon, thr["end_lat"].to_numpy("float64"),
                                             thr["end_lon"].to_numpy("float64"))})


def build(frame: pd.DataFrame, train_for_prefix: pd.DataFrame | None = None,
          require_rule: str | None = None, sources: tuple[str, ...] = ("F",),
          shift_s: float = 0.0, permute: int = 0, thin: float = 1.0,
          pred: pd.Series | None = None,
          split: int = 0, outage: bool = True,
          permute_file: bool = False) -> tuple[pd.DataFrame, dict]:
    # `outage` now only controls the AUDIT LABEL; the hold itself is upstream
    """The runway-state block for every departure of `frame`.

    `sources` selects which event sets are read; `O` and `H` need the harvest
    summaries and `exp-046`'s matcher, `F` needs neither.  `shift_s` moves every
    SOURCE event (never an anchor) by that many seconds: the placebo.
    """
    frame = frame[frame["ADEP_mvt"].isin(AIRPORTS)
                  | frame["ADES_mvt"].isin(AIRPORTS)].reset_index(drop=True)
    k = keys(frame)
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    dep = frame[isdep].reset_index(drop=True)
    mvt = _epoch(dep["MVT_TIME_UTC_mvt"])
    rep: dict = {"rows": int(len(dep)), "shift_s": shift_s, "permute_seed": permute,
                 "sources": list(sources),
                 # DECLARED, and checked before a single row is built (desk, 2026-09-09)
                 "matcher_sha256_declared": MATCHER_SHA,
                 "matcher_sha256": check_matcher()}

    mat = seg = None
    if any(s in sources for s in ("O", "H")) or "geo" in sources or "No" in sources:
        mat, mrep = matched(frame, train_for_prefix, k, require_rule)
        rep["match"] = mrep
        want_days = sorted(dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").unique())
        seg = B46.load_summaries(AIRPORTS, want_days, require_rule=require_rule)
        rep["summary_days"] = len(seg.attrs.get("days_present", []))
        # the POPULATION itself, not only its size, so `pin.py` can key on it
        rep["days_present_list"] = [list(x) for x in seg.attrs.get("days_present", [])]
        if "kind" not in seg.columns:
            # a month the harvest has not delivered returns an EMPTY frame, and every
            # observed column below would fail on a missing attribute several frames
            # deep.  Say what is missing instead.
            raise SystemExit(
                f"no summaries for {want_days[0]}..{want_days[-1]} at rule "
                f"{(require_rule or 'any')[:12]}…: {len(seg.attrs.get('days_missing', []))} "
                f"airport-days missing, {len(seg.attrs.get('days_wrong_rule', []))} on "
                "another rule.  This month is not harvested yet; an observed arm cannot "
                "be built on it.")
    if mat is not None:
        # THE DESK'S OUTAGE RULE, now upstream (2026-09-08 14:45 and 15:02 UTC).
        # `exp-046-surfacefit`'s `load_summaries` holds the degraded feeder days, so by
        # the time `mat` exists no row of one carries a clock.  What is left here is the
        # ASSERTION that this is so -- if it ever is not, the two lanes' class
        # boundaries have drifted and the desk's disjoint composition is unsafe -- and
        # the AUDIT LABEL, so a consumer can tell a held row from an ordinarily
        # unmatched one.  No clock is nulled here any more; nothing is.
        months = sorted(set(dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").unique()))
        declared = outage_set(months)
        orep = outage_report(months, declared)
        hit = np.zeros(len(mat), bool)
        if declared:
            key = list(zip(mat["airport"].astype(str), mat["day"].astype(str)))
            hit = np.array([k in declared for k in key])
            still = int((hit & mat["obs_has_clock"].to_numpy(bool)).sum())
            orep["rows_on_held_days"] = int(hit.sum())
            orep["rows_on_held_days_still_clocked"] = still
            if still:
                raise SystemExit(
                    f"{still} rows on a held airport-day still carry a clock under "
                    f"matcher {sha_of(ROOT / 'experiments/exp-046-surfacefit/code/block.py')[:12]}"
                    "…: the two lanes' class boundaries have drifted")
            if hit.any() and outage:
                mat = mat.copy()
                mat.loc[hit, "reach_class"] = "outage"
        orep["months"] = months
        orep["label_applied"] = bool(outage)
        orep["matcher_sha256"] = sha_of(ROOT / "experiments/exp-046-surfacefit/code/block.py")
        rep["outage"] = orep
    if mat is not None and thin < 1.0:
        # THE RECEIPT-RESPONSE EXPERIMENT.  The neighbourhood estimate depends on the
        # NEIGHBOURS having a clock, not on this row having one, so what an airport
        # yields is a function of its receipt rate.  EHAM is heard at 96 %; thinning
        # its clocks to a fraction simulates the density of LFPG (16 %), EGLL (50 %)
        # and EDDF (69 %) on a week where the truth is known, and gives the curve the
        # other airports' readings can be checked against when they land.
        rng = np.random.default_rng(1000 + int(round(thin * 1000)))
        drop = rng.random(len(mat)) >= thin
        mat = mat.copy()
        mat.loc[drop, ["obs_offblock", "ev_key"]] = np.nan
        mat.loc[drop, "obs_has_clock"] = False
        mat.loc[drop, "reach_class"] = "unmatched"
        rep["thin"] = {"fraction": thin, "clocks_dropped": int(drop.sum()),
                       "clocks_left": int(mat.obs_has_clock.sum())}
    t_file, cen_file = anchor(frame, None)
    t_obs, cen_obs = anchor(frame, mat["obs_offblock"].to_numpy("float64")
                            if mat is not None else None)
    rep["anchor_file"], rep["anchor_obs"] = cen_file, cen_obs

    out = pd.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy("float64")})
    qa, qr = k["apt"][isdep], k["rwy"][isdep]
    self_idx = np.arange(int(isdep.sum()))

    if "F" in sources:
        ev = file_events(frame, k)
        if shift_s:
            ev.dep_start += shift_s; ev.dep_end += shift_s
            ev.arr_start += shift_s; ev.arr_end += shift_s
        out = pd.concat([out, S.state(ev, qa, qr, mvt, self_idx, "Fto_")], axis=1)
        out = pd.concat([out, S.state(ev, qa, qr, t_file, self_idx, "Fob_")], axis=1)
    if "O" in sources or "H" in sources:
        segd = seg.copy()
        if shift_s:
            for c in ("obs_offblock", "obs_wheels_off", "t_airborne", "seg_end",
                      "seg_start", "first_motion"):
                segd[c] = segd[c] + shift_s
        for src, hyb, pre in (("O", False, "Oob_"), ("H", True, "Hob_")):
            if src not in sources:
                continue
            ev, erep = observed_events(frame, k, segd, mat, hybrid=hyb)
            if shift_s:
                ev.arr_start += shift_s; ev.arr_end += shift_s
            rep[f"events_{src}"] = erep
            # self-exclusion: the row's own OBSERVED run, found by its event key
            own = _own_index(ev, mat, hyb)
            # ...and the check that it WAS found (FALLBACK round 1, non-blocking).
            # Audit A tests source F by re-running with the exclusion off; it cannot
            # see an O or H row whose key simply failed to join, which would leave the
            # row inside its own counts silently.  The join is asserted here, on every
            # build, and the count is in every record.
            clk = mat["obs_has_clock"].to_numpy(bool)
            want = np.ones(len(own), bool) if hyb else clk
            miss = int((want & (own < 0)).sum())
            rep[f"own_index_{src}"] = {"found": int((own >= 0).sum()),
                                       "required": int(want.sum()), "missing": miss}
            if miss and not shift_s:
                raise SystemExit(
                    f"{pre}: {miss} rows that must have an own event did not join one; "
                    "self-exclusion would not run on them")
            if shift_s:
                # the segment clocks move and `mat`'s do not, so the key cannot join:
                # the placebo is only ever run on source F, and this says why
                rep[f"own_index_{src}"]["note"] = "shifted: keys cannot join, F only"
            out = pd.concat([out, S.state(ev, qa, qr, t_obs, own, pre)], axis=1)
    if "N" in sources:
        ob = None
        if mat is not None:
            ob = mat["obs_offblock"].to_numpy("float64")
            if shift_s:
                ob = ob + shift_s
        coin = None
        if split:
            coin = np.random.default_rng(split).integers(0, 2, size=len(dep))
            rep["split_seed"] = int(split)
            rep["split_counts"] = [int((coin == 0).sum()), int((coin == 1).sum())]
        rep["permute_file"] = bool(permute_file)
        nb = neighbour_block(frame, k, pred, ob, permute, coin, permute_file)
        out = pd.concat([out, nb], axis=1)
        if coin is not None:
            out["split_coin"] = coin.astype("int64")
        rep["neighbour_columns"] = [c for c in nb.columns]
    if "regnet" in sources:
        # families Xr and Xn, both from the file alone -- no matcher, no summary, no
        # label -- so they are built whatever `mat` is and they reach every dark cell
        rn = regnet_block(frame, k)
        out = pd.concat([out, rn], axis=1)
        rep["regnet_columns"] = [c for c in rn.columns]
    if "geo" in sources:
        g = geometry(mat, dep, runway_geometry())
        out = pd.concat([out, g], axis=1)
    if mat is not None:
        # `exp-046-surfacefit`'s own feature, carried here ONLY so this lane can ask
        # what the runway state adds BEYOND the row's own clock.  It is the one column
        # in the file that reads the row's own ground run, it is prefixed `own_` and it
        # is never part of a runway-state arm.
        own = _epoch(dep["MVT_TIME_UTC_mvt"]) - mat["obs_offblock"].to_numpy("float64")
        own = np.where((own > 0) & (own < 20000), own, np.nan)
        out["own_obs_taxi"] = own
        if pred is not None:
            pv = pd.Series(dep["MVT_ID_mvt"].to_numpy("float64")).map(pred
                                                                      ).to_numpy("float64")
            out["own_obs_taxi_minus_p11"] = own - pv
        out["obs_has_clock"] = mat["obs_has_clock"].to_numpy()
        out["obs_matched"] = mat["obs_matched"].to_numpy()
        out["reach_class"] = mat["reach_class"].to_numpy()
        # the harvested airport-days, from the summary frame the block actually read
        dp = seg.attrs.get("days_present") if seg is not None else None
        present = set(map(tuple, dp)) if isinstance(dp, (list, tuple)) and dp else None
        if present is not None:
            # `load_summaries` loads the day either side of the span, so `days_present`
            # carries days no row of this frame sits on and the delivered-day count read
            # one too many (round 2 non-blocking C).  The receipt denominator is the
            # harvested days THIS FRAME HAS ROWS ON.
            own_days = set(zip(mat["airport"].astype(str), mat["day"].astype(str)))
            present = present & own_days
            rep["days_present_outside_frame"] = len(set(map(tuple, dp))) - len(present)
        # a held airport-day is not in `days_present` at all, because the matcher never
        # loaded its segments; nothing is subtracted here
        rr, rrep = receipt_rates(mat, present)
        if present is None:
            rate = dep["ADEP_mvt"].astype(str).map(rr).to_numpy("float64")
        else:
            # per airport-DAY (desk 15:05 UTC); a row off a harvested day gets NaN,
            # not another day's rate
            idx = pd.MultiIndex.from_arrays([mat["airport"].astype(str).to_numpy(),
                                             mat["day"].astype(str).to_numpy()])
            rate = rr.reindex(idx).to_numpy("float64")
            onday = pd.Series(list(zip(mat["airport"], mat["day"]))).isin(present)
            rate = np.where(onday.to_numpy(), rate, np.nan)
        rep["receipt"] = rrep
        rep["receipt_rate"] = rrep["rate"]
        # The airport-day's RECEIPT, carried as a column.  It is a property of the
        # INSTRUMENT, never of the flight: it must not enter a head as a feature (that
        # is the desk's 16:50 UTC rule), and it is here so a consumer can DROP the rows
        # whose class is an instrument artefact.  The 2025 feeder map is not the 2026
        # one -- EDDM 0 % over 2025-07-01..15, EGLL 12 % rising to 65, LEMD 55 % against
        # 99.4 in january -- so a 2025 no-clock row at those airport-days is a normal
        # flight the network did not hear, and fitting a no-clock head on it teaches the
        # head about the receiver.
        out["receipt_rate"] = rate
        for c in [c for c in out.columns if c.startswith(("Oob_", "Hob_"))
                  and c.split("_", 1)[1] in S.COLS_COUNT]:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[c + "_sc"] = out[c].to_numpy("float64") / np.where(rate > 0, rate, np.nan)
    out.attrs.update(rep)
    return out, rep


def _own_index(ev, mat, hybrid) -> np.ndarray:
    """Index of each departure row's OWN event in `ev`'s departure arrays, else -1.

    Under O the row's own event is its observed ground run where the matcher found
    one; under H a row the network never heard was appended on its file clock, and
    that appended event is its own too.  Every count in `state()` subtracts it.
    """
    lut = pd.Series(np.arange(len(ev.ev_keys)), index=pd.Index(ev.ev_keys))
    lut = lut[~lut.index.duplicated(keep="first")]
    want = mat["ev_key"].to_numpy(dtype=object)
    if hybrid:
        fk = np.array(["file|" + str(int(v)) for v in mat["MVT_ID_mvt"].to_numpy()],
                      dtype=object)
        want = np.where(pd.isna(want), fk, want)
    got = pd.to_numeric(lut.reindex(pd.Index(want)).to_numpy(), errors="coerce")
    return np.where(np.isfinite(got), np.nan_to_num(got, nan=-1), -1).astype("int64")


# --------------------------------------------- the neighbourhood, per source

def neighbour_block(frame: pd.DataFrame, k: dict, pred: pd.Series | None,
                    obs_offblock: np.ndarray | None,
                    permute: int = 0,
                    coin: np.ndarray | None = None,
                    permute_file: bool = False) -> pd.DataFrame:
    """Family N: what the OTHER departures' observed taxi says about this cell.

    Two clocks, priced apart, exactly as the state block prices its two:

        Nf_*   the neighbour's taxi read as `MVT_TIME - AOBT_3_flt` -- the file's own
               proxy, 350-466 s from the truth (`docs/input-rule.md`, exp-043 §4)
        No_*   the neighbour's taxi read as `MVT_TIME - obs_offblock` -- the adsb.lol
               observation, 143 s from the truth on the pilot's week

    `pred` is v11's prediction for every departure of the month, joined by
    `MVT_ID_mvt`.  It is our own model output, not a label, and it exists for the
    ranking months because it is the file we serve.  Where it is absent the residual
    form is dropped and only the anomaly form (observed taxi minus the airport-hour
    median observed taxi of the same masked month) is built.

    **`coin` is the split half** (FALLBACK review round 1, blocking 1).  The share
    statistic `1 - after/before` reads the self-excluded cross-product over the pairs
    of a cell, and the correction for row `j` is built from the OTHER rows' residuals
    of that same cell -- row `i`'s among them.  So `E[r_i h_j]` carries a term in
    `E[r_i r_i]`, which is idiosyncratic variance and not shared component, and the
    statistic over-credits this family.  With a coin, row `i`'s neighbourhood is built
    from the rows of the OTHER coin only; the caller then reads the cross-product
    inside `(cell, coin)`, where no pair's correction has seen either member.  The
    price is half the neighbours, so the split-half share is an attenuated, honest
    reading of the same quantity, not a second estimate of a different one.
    """
    dep = frame[frame["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    mvt = _epoch(dep["MVT_TIME_UTC_mvt"])
    aobt = _epoch(dep["AOBT_3_flt"])
    apt = k["apt"][isdep]
    rwy = k["rwy"][isdep]
    cell = np.floor(mvt / 600.0).astype("int64")
    cell = np.char.add(np.char.add(apt.astype(str), "|"), cell.astype(str))
    hour = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.hour.to_numpy()
    p = (pd.Series(dep["MVT_ID_mvt"].to_numpy("float64")).map(pred).to_numpy("float64")
         if pred is not None else np.full(len(dep), np.nan))
    if permute and obs_offblock is not None:
        # THE PLACEBO FOR FAMILY N.  A +2 h shift of every clock is not one here: it
        # moves each neighbour's taxi by the same constant, which a linear head
        # absorbs into its intercept and which leaves the cell-to-cell structure
        # untouched.  The test that does bite is a PERMUTATION -- inside each airport
        # and UTC day, the observed off-block instants are dealt out to different
        # aircraft, so the marginal distribution of the clock survives and the pairing
        # of a clock to its own flight does not.  Under it the neighbourhood estimate
        # must fall to the noise floor.
        rng = np.random.default_rng(permute)
        day = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
        grp = (dep["ADEP_mvt"].astype(str) + "|" + day).to_numpy()
        order = {g: np.flatnonzero(grp == g) for g in np.unique(grp)}

        def deal(v):
            out = np.asarray(v, "float64").copy()
            for g, idx0 in order.items():
                idx = idx0[np.isfinite(out[idx0])]
                if len(idx) > 1:
                    out[idx] = out[rng.permutation(idx)]
            return out

        obs_offblock = deal(obs_offblock)
        if permute_file:
            # THE PLACEBO THAT BITES FAMILY Nf.  Dealing only the OBSERVED clocks
            # leaves `Nf_*` untouched, so its reading under `--permute` is a control
            # and not a test.  December's no-clock rows put +510 s^2 on `Nf` alone,
            # which is a claim about the file clock and needs its own placebo: the
            # `AOBT_3_flt` off-blocks dealt out to other aircraft inside the same
            # airport and UTC day, exactly as the observed ones are.
            aobt = deal(aobt)
    outs = []
    for tag, ob in (("Nf_", aobt), ("No_", obs_offblock)):
        if ob is None:
            continue
        taxi = mvt - np.asarray(ob, "float64")
        taxi = np.where((taxi > 0) & (taxi < 20000), taxi, np.nan)
        med = pd.DataFrame({"a": dep["ADEP_mvt"].astype(str).to_numpy(),
                            "h": hour, "v": taxi}).dropna().groupby(["a", "h"])["v"].median()
        base = med.reindex(pd.MultiIndex.from_arrays(
            [dep["ADEP_mvt"].astype(str).to_numpy(), hour])).to_numpy("float64")
        heard = np.isfinite(taxi)
        if coin is None:
            outs.append(S.neighbourhood(apt, rwy, cell, mvt, taxi - p, taxi - base,
                                        heard, tag))
            continue
        c = np.asarray(coin).astype("int64")
        halves = [S.neighbourhood(apt, rwy, cell, mvt, taxi - p, taxi - base,
                                  heard & (c == 1 - g), tag) for g in (0, 1)]
        pick = halves[0].copy()
        take = (c == 1)
        for col in pick.columns:
            pick[col] = np.where(take, halves[1][col].to_numpy("float64"),
                                 halves[0][col].to_numpy("float64"))
        outs.append(pick)
    return pd.concat(outs, axis=1) if outs else pd.DataFrame(index=range(len(dep)))


# --------------------------------------------- families Xr and Xn (desk, 2026-09-09)

REG_S = 900.0          # a departure whose actual off-block is 15 min past its estimated
SLIP_CLIP = (-1800.0, 7200.0)


def _loo_mean(v: np.ndarray, ok: np.ndarray, grp: np.ndarray) -> tuple:
    """Leave-one-out group mean of `v` over `grp`, and the count it is taken over.

    A row never reads its own value, so nothing here is a self-read: the base head
    already carries the ROW's own `aobt_eobt`, and what these families add is what the
    OTHER departures around it are doing.  Rows whose own value is missing still get the
    group's full mean, and a group with no other finite row returns NaN, never zero --
    a zero would say "nobody slipped" where the truth is "nobody was measured".
    """
    codes, uniq = pd.factorize(pd.Series(grp))
    n = len(uniq)
    w = np.where(ok, 1.0, 0.0)
    s = np.bincount(codes, weights=np.where(ok, v, 0.0), minlength=n)
    c = np.bincount(codes, weights=w, minlength=n)
    num = s[codes] - np.where(ok, v, 0.0)
    den = c[codes] - w
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), den


def regnet_block(frame: pd.DataFrame, k: dict) -> pd.DataFrame:
    """Families `Xr` (the CELL's regulation signature) and `Xn` (the NETWORK state).

    Both are RESULTS §7ab's first two design changes, and both are built from the FILE
    ALONE -- no observation, no label -- so they reach `LTFM` and every other dark cell,
    which is the whole reason the desk ordered them.

        Xr_*   the share of the airport x 10-minute cell whose `AOBT_3 - EOBT_1` is past
               15 minutes, leave-one-out, with the WITHIN-OPERATOR share beside it and
               the airport-day share under it.  An ATFM regulation slips a bank, not a
               flight; a carrier's own delay propagation slips that carrier.  The two
               shares in one ridge are exactly §7ab's kill test: if the cell share only
               works because it is a carrier fingerprint, the operator share takes it.
        Xn_*   the same UTC hour at the OTHER nine airports, self-airport excluded.
               Every other feature in both lanes is inside one airport; a frontal system
               or a European regulation is not.

    **What §7ab measured and this does NOT build.**  §7ab's network read is the
    leave-one-airport-out mean of `r12 = y_true - p12`.  That is a LABEL, and the ranking
    months have none: the same-hour rows at the other nine airports of a served month are
    themselves ranking rows, so the column cannot exist at serving time.  §7ab's +0.0683 s
    is therefore an information bound on the network channel, not a feature.  What is
    built here is the servable read of the same channel: the leave-one-airport-out mean of
    the FILE-CLOCK taxi anomaly and of the SLIP anomaly, both of which the ranking file
    carries (`MVT_TIME_UTC_mvt`, `AOBT_3_flt` and `EOBT_1_flt` are unmasked).
    """
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    dep = frame[isdep].reset_index(drop=True)
    apt = k["apt"][isdep]                       # -1 at an airport outside the ten
    home = apt >= 0
    t = _epoch(dep["MVT_TIME_UTC_mvt"])
    aobt = _epoch(dep["AOBT_3_flt"])
    eobt = _epoch(dep["EOBT_1_flt"]) if "EOBT_1_flt" in dep.columns \
        else np.full(len(dep), np.nan)
    hod = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.hour.to_numpy()
    a = apt.astype(str)
    cell = np.char.add(np.char.add(a, "|"), np.floor(t / 600.0).astype("int64").astype(str))
    day = np.floor(t / 86400.0).astype("int64")
    aday = np.char.add(np.char.add(a, "|"), day.astype(str))
    op = dep["AIRCRAFT_OPERATOR_flt"].astype(str).to_numpy()
    oday = np.char.add(np.char.add(aday, "|"), op)
    hbin = np.floor(t / 3600.0).astype("int64")

    out = pd.DataFrame(index=range(len(dep)))

    # ---- Xr: the regulation signature
    slip = aobt - eobt
    slip = np.where(np.isfinite(slip) & (slip > -21600) & (slip < 86400), slip, np.nan)
    ok = np.isfinite(slip) & home
    reg = np.where(ok, (slip > REG_S).astype("float64"), 0.0)
    sw = np.clip(slip, *SLIP_CLIP)
    for tag, grp in (("cell", cell), ("op", oday), ("day", aday)):
        share, n = _loo_mean(reg, ok, grp)
        out[f"Xr_{tag}_reg_share"] = np.where(home, share, np.nan)
        out[f"Xr_{tag}_reg_n"] = np.where(home, n, np.nan)
        if tag != "day":
            mean, _ = _loo_mean(np.where(ok, sw, 0.0), ok, grp)
            out[f"Xr_{tag}_slip_mean_s"] = np.where(home, mean, np.nan)
    # the two contrasts the kill test needs: a regulation is the cell ABOVE its carrier
    # and above its day, a carrier fingerprint is the cell equal to its carrier
    out["Xr_cell_minus_op"] = (out["Xr_cell_reg_share"].to_numpy("float64")
                               - out["Xr_op_reg_share"].to_numpy("float64"))
    out["Xr_cell_minus_day"] = (out["Xr_cell_reg_share"].to_numpy("float64")
                                - out["Xr_day_reg_share"].to_numpy("float64"))

    # ---- Xn: the network state, self-AIRPORT excluded
    ftaxi = t - aobt
    ftaxi = np.where((ftaxi > 0) & (ftaxi < 20000), ftaxi, np.nan)
    for tag, v in (("ftaxi", ftaxi), ("slip", np.where(np.isfinite(slip), sw, np.nan))):
        okv = np.isfinite(v) & home
        # the anomaly, against this month's own airport x hour-of-day median, so the
        # network term carries the DEVIATION of an airport from its own normal hour and
        # not the level differences between airports
        med = pd.DataFrame({"a": a, "h": hod, "v": np.where(okv, v, np.nan)}).dropna() \
            .groupby(["a", "h"])["v"].median()
        base = med.reindex(pd.MultiIndex.from_arrays([a, hod])).to_numpy("float64")
        an = v - base
        oka = np.isfinite(an) & home
        codes, uniq = pd.factorize(pd.Series(hbin))
        n = len(uniq)
        w = np.where(oka, 1.0, 0.0)
        hs = np.bincount(codes, weights=np.where(oka, an, 0.0), minlength=n)[codes]
        hc = np.bincount(codes, weights=w, minlength=n)[codes]
        ah = np.char.add(np.char.add(a, "|"), hbin.astype(str))
        own_mean, own_n = _loo_mean(an, oka, ah)          # this airport, self excluded
        acodes, auniq = pd.factorize(pd.Series(ah))
        an_s = np.bincount(acodes, weights=np.where(oka, an, 0.0),
                           minlength=len(auniq))[acodes]
        an_c = np.bincount(acodes, weights=w, minlength=len(auniq))[acodes]
        num, den = hs - an_s, hc - an_c
        net = np.where((den > 0) & home, num / np.maximum(den, 1e-9), np.nan)
        out[f"Xn_net_{tag}"] = net
        out[f"Xn_net_{tag}_n"] = np.where(home, den, np.nan)
        out[f"Xn_own_{tag}"] = np.where(home, own_mean, np.nan)
        out[f"Xn_net_minus_own_{tag}"] = net - np.where(home, own_mean, np.nan)
    return out
