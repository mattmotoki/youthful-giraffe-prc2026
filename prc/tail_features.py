"""Per-departure tail signals for PRC 2026 taxi-out prediction.

`tail_signals(frame)` takes one month's *full* movement frame — departures and
arrivals together, masked exactly like the ranking file (departure
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` blanked) — and returns one row per
departure with the signals that separate the extreme rows from the bulk.

Everything here is computable from the ranking file. Nothing reads a
departure's own block time or taxi time, and nothing reads another
*departure's* block time; arrival block times (on-block) are legal inputs and
are used for the stand-occupancy signals (see `docs/input-rule.md`).

Findings behind the signals are in `experiments/exp-002-tail/RESULTS.md`:

- `noac` (`AIRCRAFT_TYPE_mvt` null) marks 0.07 % of departures that carry
  94 % of the rows above 15,000 s. On roughly half of them the recorded block
  time is a placeholder equal to `SCHED_TIME_UTC_mvt`, so `mvt_sched` is the
  target itself; `mvt_sched` also orders the placeholder probability.
- `aobt_lobt` (`AOBT_3_flt - LOBT_flt`) above ~3600 s marks the NM-joined tail
  at 99x lift; on those rows `mvt_lobt` is the right block-time proxy.
- Stand occupancy: the last arrival to park at the same stand before this
  departure's take-off bounds the hidden block time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 1.2.0 — exp-009 added ten observables for p(placeholder) on the noac rows,
#: plus the desk's airport-day arrival placeholder rate (round 2, R2) and
#: the schedule-date-cross test for the noac two-point mixture (1.3.0).
VERSION = "1.3.0"

RANKING_AIRPORTS = ("LTFM", "EHAM", "LFPG", "EDDF", "EGLL", "LEMD", "LEBL", "EDDM", "LIRF", "LSZH")

#: columns returned by :func:`tail_signals`, in order
SIGNALS = (
    "apt", "stand", "runway", "hour", "dow", "minute_of_day",
    "noac", "nonm", "vfr", "flight_len",
    "mvt_sched", "mvt_lobt", "mvt_iobt", "mvt_eobt", "mvt_aobt",
    "aobt_lobt", "lobt_sched", "eobt_sched", "lobt_iobt", "aobt_sched",
    "noac_day_count", "nonm_day_count",
    "apt_day_med_mvt_aobt", "apt_day_p90_mvt_aobt", "apt_day_long_frac",
    "apt_hour_med_mvt_aobt", "mvt_aobt_vs_day_med",
    "gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h",
    "n_arr_sched_mvt", "gap_arr_after_sched",
    "gap_dep_prev_mvt", "gap_dep_prev_aobt", "gap_dep_next_mvt",
    "stand_busy_frac_6h", "mvt_arvt1", "mvt_arvt3", "arvt3_arvt1",
    # exp-009: candidate observables for p(placeholder) on the noac rows
    "noac_day_n", "noac_day_share", "dep_day_n", "arr_within_1h_of_sched",
    "arr_within_3h_of_sched", "flight_n_digits", "flight_n_alpha",
    "flight_repeated_prefix", "flight_prefix", "mvt_sched_log",
    "arr_day_ph_rate", "sched_date_cross",
)


def _sec(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def _epoch(s: pd.Series) -> pd.Series:
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def airport_of(frame: pd.DataFrame) -> pd.Series:
    """The observed airport: ADEP for departures, ADES for arrivals."""
    return pd.Series(
        np.where(frame["PHASE_mvt"].to_numpy() == "DEP", frame["ADEP_mvt"], frame["ADES_mvt"]),
        index=frame.index,
    )


def _stand_events(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(arrivals, departures) with an airport|stand key and epoch-second times."""
    f = frame.copy()
    f["apt"] = airport_of(f)
    f["_key"] = f["apt"].astype(str) + "|" + f["STAND_mvt"].astype(str)
    arr = f.loc[f["PHASE_mvt"] == "ARR", ["_key", "BLOCK_TIME_UTC_mvt"]].copy()
    arr["a_on"] = _epoch(arr["BLOCK_TIME_UTC_mvt"])
    arr = arr.dropna(subset=["a_on"]).sort_values("a_on", kind="mergesort")
    dep = f.loc[f["PHASE_mvt"] == "DEP"].copy()
    dep["t_mvt"] = _epoch(dep["MVT_TIME_UTC_mvt"])
    dep["t_aobt"] = _epoch(dep["AOBT_3_flt"]) if "AOBT_3_flt" in dep else np.nan
    return arr, dep


def _stand_signals(frame: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    """Stand-occupancy signals for every departure, from arrival on-block times.

    An arrival's on-block time is a legal input. The aircraft flying this
    departure occupied the stand until its block-off, so an arrival that
    parked at the same stand *after* that block-off but before the take-off
    puts a floor under the taxi time; an arrival before the block-off puts a
    ceiling on it. Which of the two a given neighbour is cannot be decided
    from the file, so the raw gaps are handed to the model rather than
    applied as hard bounds (a FIFO pairing was tried and desynchronises; see
    RESULTS.md finding 7).
    """
    arr, _ = _stand_events(frame)
    out = pd.DataFrame(index=dep.index)
    if arr.empty:
        for c in ("gap_arr1", "gap_arr2", "gap_arr3", "gap_arr_next", "n_arr_6h", "stand_busy_frac_6h"):
            out[c] = np.nan
        return out

    a = arr.reset_index(drop=True)
    a["p1"] = a.groupby("_key", sort=False)["a_on"].shift(1)
    a["p2"] = a.groupby("_key", sort=False)["a_on"].shift(2)
    left = dep[["_key", "t_mvt"]].reset_index().sort_values("t_mvt", kind="mergesort")
    back = pd.merge_asof(
        left, a[["_key", "a_on", "p1", "p2"]], left_on="t_mvt", right_on="a_on",
        by="_key", direction="backward", allow_exact_matches=True,
    ).set_index("index")
    fwd = pd.merge_asof(
        left, a[["_key", "a_on"]].rename(columns={"a_on": "a_next"}), left_on="t_mvt", right_on="a_next",
        by="_key", direction="forward", allow_exact_matches=False,
    ).set_index("index")

    t = dep["t_mvt"]
    out["gap_arr1"] = t - back["a_on"].reindex(dep.index)
    out["gap_arr2"] = t - back["p1"].reindex(dep.index)
    out["gap_arr3"] = t - back["p2"].reindex(dep.index)
    out["gap_arr_next"] = fwd["a_next"].reindex(dep.index) - t
    # arrivals at this stand in the 6 h before take-off
    a = a.assign(_rank=a.groupby("_key", sort=False).cumcount() + 1.0)

    def rank_at(times: pd.Series) -> pd.Series:
        """How many arrivals had parked at the stand at or before `times`."""
        q = pd.DataFrame({"_key": dep["_key"].to_numpy(), "t": times.to_numpy()}, index=dep.index)
        q = q.reset_index().dropna(subset=["t"]).sort_values("t", kind="mergesort")
        r = pd.merge_asof(q, a[["_key", "a_on", "_rank"]], left_on="t", right_on="a_on",
                          by="_key", direction="backward").set_index("index")
        return r["_rank"].reindex(dep.index).fillna(0.0)

    t_sched = _epoch(dep["SCHED_TIME_UTC_mvt"])
    r_mvt = rank_at(dep["t_mvt"])
    out["n_arr_6h"] = (r_mvt - rank_at(dep["t_mvt"] - 21600.0)).clip(lower=0)
    out["stand_busy_frac_6h"] = out["n_arr_6h"] / 6.0
    # Arrivals that parked at this stand between the scheduled off-block and the
    # take-off. If the recorded block time is a placeholder equal to the
    # schedule, the stand was free from then on and such an arrival is expected;
    # if the aircraft was still on stand, one cannot occur.
    out["n_arr_sched_mvt"] = (r_mvt - rank_at(t_sched)).clip(lower=0)
    nxt_after_sched = pd.merge_asof(
        pd.DataFrame({"_key": dep["_key"].to_numpy(), "t": t_sched.to_numpy()}, index=dep.index)
          .reset_index().dropna(subset=["t"]).sort_values("t", kind="mergesort"),
        a[["_key", "a_on"]].rename(columns={"a_on": "a_after"}), left_on="t", right_on="a_after",
        by="_key", direction="forward", allow_exact_matches=False,
    ).set_index("index")["a_after"].reindex(dep.index)
    out["gap_arr_after_sched"] = np.where(nxt_after_sched <= dep["t_mvt"], nxt_after_sched - t_sched, np.nan)
    return out


def _dep_neighbour_signals(dep: pd.DataFrame) -> pd.DataFrame:
    """Previous and next departure from the same stand (take-off and NM off-block).

    Only take-off times and NM off-block times of other departures are used;
    their block times are blanked in the ranking file and are never read.
    """
    out = pd.DataFrame(index=dep.index)
    d = dep[["_key", "t_mvt", "t_aobt"]].reset_index().sort_values("t_mvt", kind="mergesort")
    d["prev_mvt"] = d.groupby("_key", sort=False)["t_mvt"].shift(1)
    d["prev_aobt"] = d.groupby("_key", sort=False)["t_aobt"].shift(1)
    d["next_mvt"] = d.groupby("_key", sort=False)["t_mvt"].shift(-1)
    d = d.set_index("index")
    out["gap_dep_prev_mvt"] = dep["t_mvt"] - d["prev_mvt"].reindex(dep.index)
    out["gap_dep_prev_aobt"] = dep["t_mvt"] - d["prev_aobt"].reindex(dep.index)
    out["gap_dep_next_mvt"] = d["next_mvt"].reindex(dep.index) - dep["t_mvt"]
    return out


def tail_signals(frame: pd.DataFrame, airports: tuple[str, ...] | None = RANKING_AIRPORTS) -> pd.DataFrame:
    """Per-departure tail signals for one month frame.

    Parameters
    ----------
    frame:
        One month of movements, departures and arrivals together, masked like
        the ranking file. Timestamps must be tz-aware (``prc.data.load_month``
        and ``prc.data.fold_frames`` already parse them).
    airports:
        Keep only departures from these airports. ``None`` keeps all.

    Returns
    -------
    DataFrame indexed like the departure rows of ``frame``, with the columns
    in :data:`SIGNALS`.
    """
    arr, dep = _stand_events(frame)
    if airports is not None:
        dep = dep[dep["apt"].isin(airports)]

    out = pd.DataFrame(index=dep.index)
    out["apt"] = dep["apt"].astype("category")
    out["stand"] = (dep["apt"].astype(str) + "|" + dep["STAND_mvt"].astype(str)).astype("category")
    out["runway"] = (dep["apt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str)).astype("category")
    mvt = dep["MVT_TIME_UTC_mvt"]
    out["hour"] = mvt.dt.hour.astype("int16")
    out["dow"] = mvt.dt.dayofweek.astype("int16")
    out["minute_of_day"] = (mvt.dt.hour * 60 + mvt.dt.minute).astype("int16")

    # --- finding 1: the extreme-tail flag -------------------------------
    out["noac"] = dep["AIRCRAFT_TYPE_mvt"].isna().astype("int8")
    out["nonm"] = dep["LOBT_flt"].isna().astype("int8")
    out["vfr"] = (dep["FLIGHT_RULE_mvt"].astype(str) == "V").astype("int8")
    out["flight_len"] = dep["FLIGHT_mvt"].astype(str).str.len().astype("int16")

    # --- own-flight time deltas ----------------------------------------
    out["mvt_sched"] = _sec(mvt, dep["SCHED_TIME_UTC_mvt"])
    out["mvt_lobt"] = _sec(mvt, dep["LOBT_flt"])
    out["mvt_iobt"] = _sec(mvt, dep["IOBT_flt"])
    out["mvt_eobt"] = _sec(mvt, dep["EOBT_1_flt"])
    out["mvt_aobt"] = _sec(mvt, dep["AOBT_3_flt"])
    out["aobt_lobt"] = _sec(dep["AOBT_3_flt"], dep["LOBT_flt"])
    out["lobt_sched"] = _sec(dep["LOBT_flt"], dep["SCHED_TIME_UTC_mvt"])
    out["eobt_sched"] = _sec(dep["EOBT_1_flt"], dep["SCHED_TIME_UTC_mvt"])
    out["lobt_iobt"] = _sec(dep["LOBT_flt"], dep["IOBT_flt"])
    out["aobt_sched"] = _sec(dep["AOBT_3_flt"], dep["SCHED_TIME_UTC_mvt"])

    # --- finding 2: the two failure modes are separated by daily volume --
    day = mvt.dt.floor("D")
    grp = pd.DataFrame({"apt": dep["apt"].to_numpy(), "day": day.to_numpy(),
                        "noac": out["noac"].to_numpy(), "nonm": out["nonm"].to_numpy()},
                       index=dep.index)
    for c in ("noac", "nonm"):
        out[f"{c}_day_count"] = grp.groupby(["apt", "day"], observed=True)[c].transform("sum").astype("float64")

    # --- finding 12: the tail clusters by airport-day, and the day's own
    # take-off-minus-NM-off-block distribution says which days those are.
    # `mvt_aobt` is observable for every ranking row, so these aggregates are
    # legal and they are the only signal that reaches the winter tail, which
    # the noac machinery does not touch.
    grp["ma"] = out["mvt_aobt"].to_numpy()
    grp["hour"] = out["hour"].to_numpy()
    gd = grp.groupby(["apt", "day"], observed=True)["ma"]
    out["apt_day_med_mvt_aobt"] = gd.transform("median")
    out["apt_day_p90_mvt_aobt"] = gd.transform(lambda s: s.quantile(0.9))
    grp["long"] = (grp["ma"] > 2400).astype("float64")
    out["apt_day_long_frac"] = grp.groupby(["apt", "day"], observed=True)["long"].transform("mean")
    out["apt_hour_med_mvt_aobt"] = grp.groupby(["apt", "day", "hour"], observed=True)["ma"].transform("median")
    out["mvt_aobt_vs_day_med"] = out["mvt_aobt"] - out["apt_day_med_mvt_aobt"]

    # --- finding 7: stand occupancy -------------------------------------
    out = out.join(_stand_signals(frame, dep))
    out = out.join(_dep_neighbour_signals(dep))
    out = out.join(_return_signals(frame, dep))
    out = out.join(_noac_signals(dep, out))
    out = out.join(_arrival_day_rate(frame, dep))
    return out.loc[:, list(SIGNALS)]


def _noac_signals(dep: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """exp-009 candidates for p(placeholder). All label-free.

    The per-day noac count is the one with a measured relation: P(placeholder)
    runs 0.69 on an airport-day with at most two noac rows down to 0.24 above
    twenty (exp-002 finding 2), i.e. an isolated unenriched record is likely a
    placeholder while a bulk outage is not. The stand's next arrival relative to
    SCHED is the one observable that can be right without resolving the
    ambiguity: if the stand was re-occupied soon after the scheduled off-block
    then the aircraft did leave near SCHED either way.
    """
    o = pd.DataFrame(index=dep.index)
    day = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    g = pd.DataFrame({"apt": dep["ADEP_mvt"].astype(str).to_numpy(), "day": day.to_numpy(),
                      "noac": out["noac"].to_numpy()}, index=dep.index)
    grp = g.groupby(["apt", "day"], observed=True)["noac"]
    o["noac_day_n"] = grp.transform("sum").astype(float)
    o["noac_day_share"] = grp.transform("mean").astype(float)
    o["dep_day_n"] = grp.transform("size").astype(float)
    o["arr_within_1h_of_sched"] = (out["gap_arr_after_sched"] < 3600).astype(float)
    o["arr_within_3h_of_sched"] = (out["gap_arr_after_sched"] < 10800).astype(float)
    f = dep["FLIGHT_mvt"].astype(str)
    o["flight_n_digits"] = f.str.count(r"\d").astype(float)
    o["flight_n_alpha"] = f.str.count(r"[A-Za-z]").astype(float)
    # `\1?` made the backreference optional, so this matched any string starting
    # with three capitals and was 1.0 on 95 % of rows -- a dead feature (Opus r1).
    o["flight_repeated_prefix"] = f.str.match(r"^([A-Z]{3})\1").fillna(False).astype(float)
    o["flight_prefix"] = f.str[:3].astype("category")
    ms = out["mvt_sched"]
    o["mvt_sched_log"] = np.sign(ms) * np.log1p(ms.abs())
    return o


#: local zone per ranking airport, for the schedule-date-cross test
_TZ = {"LIRF": "Europe/Rome", "LTFM": "Europe/Istanbul", "LSZH": "Europe/Zurich",
       "EHAM": "Europe/Amsterdam", "LFPG": "Europe/Paris", "EDDF": "Europe/Berlin",
       "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "LEMD": "Europe/Madrid",
       "LEBL": "Europe/Madrid"}
#: mean taxi on the rollover rows, used to place the off-block without a label
ROLLOVER_TAXI = 1136.0


def _sched_date_cross(dep: pd.DataFrame) -> pd.Series:
    """Local calendar days between the scheduled off-block and the actual one.

    A rollover block time carries the right time of day and the wrong date, which
    a system produces by stamping the *scheduled* date onto the actual off-block.
    That can only happen when the actual off-block falls on a different local day
    from the schedule. Measured on 2025: all 11 rollovers above 8 h cross the
    date and none of the 12 non-crossing rows is a rollover -- necessary, not
    sufficient, since 27 placeholders cross too.

    Label-free: the off-block is placed at `MVT_TIME - ROLLOVER_TAXI`.
    """
    off = dep["MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=ROLLOVER_TAXI)
    out = np.full(len(dep), np.nan)
    apt = dep["ADEP_mvt"].astype(str).to_numpy()
    for a in pd.unique(apt):
        m = apt == a
        tz = _TZ.get(a, "UTC")
        d1 = off[m].dt.tz_convert(tz).dt.normalize()
        d0 = dep["SCHED_TIME_UTC_mvt"][m].dt.tz_convert(tz).dt.normalize()
        out[m] = (d1 - d0).dt.days.to_numpy()
    return pd.Series(out, index=dep.index)


def _arrival_day_rate(frame: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    """Share of the airport-day's ARRIVALS whose on-block equals their schedule.

    The desk's observable. Arrivals keep `BLOCK_TIME_UTC_mvt` in the ranking
    file, so this is label-free. As a *count* it is useless (tens to hundreds
    every day); as a per-day **rate** it is monotone across quintiles, 0.387 to
    0.611, correlation +0.178 overall and +0.157 within LIRF.

    It is 0.48 anti-correlated with the per-day noac count, which is why it
    measured as redundant on the offset head. On the head that actually ships
    -- the seed-averaged mixture -- it is worth 11 to 13 s of noac LOMO.
    """
    f = frame.copy()
    f["apt"] = airport_of(f)
    f["day"] = f["MVT_TIME_UTC_mvt"].dt.floor("D")
    arr = f[f["PHASE_mvt"] == "ARR"]
    ph = (_sec(arr["BLOCK_TIME_UTC_mvt"], arr["SCHED_TIME_UTC_mvt"]).abs() <= 60.0)
    rate = ph.groupby([arr["apt"], arr["day"]], observed=True).mean().rename("arr_day_ph_rate")
    key = pd.MultiIndex.from_arrays([dep["ADEP_mvt"].astype(str),
                                     dep["MVT_TIME_UTC_mvt"].dt.floor("D")])
    out = pd.DataFrame(index=dep.index)
    out["arr_day_ph_rate"] = rate.reindex(key).to_numpy()
    out["sched_date_cross"] = _sched_date_cross(dep)
    return out


def _return_signals(frame: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    """Take-off timed against the flight's own arrival at its destination.

    `ARVT_1_flt` (estimated) and `ARVT_3_flt` (actual) are the flight's arrival
    at ADES and are present in the ranking file. A departure whose recorded
    take-off is *after* its own arrival at the far end is internally
    inconsistent, and the inconsistency is almost always in the movement record
    whose block time defines the target. On July 2025 the 537 rows with
    `MVT - ARVT_1 > 0` carry a 9.3 % tail rate (27x the base rate) and the 4 rows
    with `MVT - ARVT_3 > 0` carry 75 % (218x). This is the observable form of the
    "an arrival with the same identifier sits between block time and take-off"
    check: the matching arrival is the flight's own landing at the destination,
    not another movement at this airport.
    """
    out = pd.DataFrame(index=dep.index)
    for col, name in (("ARVT_1_flt", "mvt_arvt1"), ("ARVT_3_flt", "mvt_arvt3")):
        out[name] = _sec(dep["MVT_TIME_UTC_mvt"], dep[col]) if col in dep else np.nan
    out["arvt3_arvt1"] = (_sec(dep["ARVT_3_flt"], dep["ARVT_1_flt"])
                          if "ARVT_3_flt" in dep and "ARVT_1_flt" in dep else np.nan)
    return out
