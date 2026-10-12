"""Candidate off-block proxies for the no-NM segment (assignment question 1).

Design rule from `RESULTS.md` §1: a proxy is fed in **taxi units**
(`take-off − proxy`), never as a raw offset beside `MVT − SCHED`. Handing a
tree the two large timestamps and expecting it to find their difference costs
about 200 s even when the difference is the exact answer.

Families
--------
``base``    schedule and movement columns only; no `_flt` column exists on
            these rows, in training or at prediction.
``op``      the operator, recovered from the `FLIGHT_mvt` callsign prefix.
            `AIRCRAFT_OPERATOR_flt` is null on exactly this segment, and
            exp-002 finding 5 shows the segment is dominated by particular
            carriers, so the prefix is the only route to that signal.
``stand``   `prc.stand_features.occupancy_features` — the previous and next
            arrival at the stand as taxi bounds, turnaround, neighbour gaps.
``surface`` `prc.surface_features` neighbour and pressure columns: what
            take-out is actually running at that airport and runway right now,
            measured on flights that *do* have an AOBT.
``bracket`` the nearest same-runway departures with a real AOBT, and their
            observed spans — the assignment's "bracketed queue entry".
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import prc.data as D
import prc.stand_features as SF
import prc.surface_features as S

CATS = ["apt", "apt_stand", "apt_rwy", "ac_type", "ades", "flight_rule", "op3"]

SURFACE_COLS = (["nbr_span_apt", "nbr_span_rwy", "nbr_taxiin_apt", "nbr_taxiin_rwy"]
                + [f"dep_{a}_{b}{w}" for w in (900, 1800, 3600)
                   for a in ("rwy", "apt") for b in ("pre", "post")]
                + [f"arr_{a}_{b}1800" for a in ("rwy", "apt") for b in ("pre", "post")]
                + ["gap_prev_rwy", "gap_next_rwy", "n10_rwy", "rank10_rwy",
                   "taxiing_at_off", "arr_taxiing_at_off"])
# stand columns are listed explicitly in FAMILIES below
BRACKET_COLS = ["brk_prev_span", "brk_next_span", "brk_prev_gap", "brk_next_gap",
                "brk_span_mean", "brk_n"]


def _op3(flight: pd.Series) -> np.ndarray:
    """ICAO operator prefix of the callsign: leading letters, up to three."""
    s = flight.astype(str).str.upper()
    pref = s.str.extract(r"^([A-Z]{1,3})", expand=False).fillna("NA")
    return pref.to_numpy().astype(str)


def bracket_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Spans of the nearest same-runway departures that do have an AOBT.

    If a flight took off between two neighbours whose queue times are known,
    its own queue entry is bracketed by theirs. Returned in taxi units.
    """
    phase = frame["PHASE_mvt"].to_numpy()
    dep_i = np.flatnonzero(phase == "DEP")
    apt = frame["ADEP_mvt"].astype(str).to_numpy()[dep_i]
    rwy = frame["RUNWAY_mvt"].astype(str).to_numpy()[dep_i]
    key = np.char.add(np.char.add(apt, ":"), rwy)
    off = S.epoch(frame["MVT_TIME_UTC_mvt"])[dep_i]
    aobt = S.epoch(frame["AOBT_3_flt"])[dep_i]
    span = off - aobt
    have = np.isfinite(span)
    out = pd.DataFrame(index=frame.index[dep_i])
    prev_s = np.full(len(dep_i), np.nan); next_s = np.full(len(dep_i), np.nan)
    prev_g = np.full(len(dep_i), np.nan); next_g = np.full(len(dep_i), np.nan)
    mean_s = np.full(len(dep_i), np.nan); nnb = np.zeros(len(dep_i))
    for _, idx in S._groups(key):
        src = idx[have[idx]]
        if len(src) == 0:
            continue
        order = np.argsort(off[src], kind="stable")
        st, sv = off[src][order], span[src][order]
        q = off[idx]
        # `left` for the previous side and `right` for the next, so a row that
        # is itself in the source set (any departure with an AOBT) reads
        # neither side as itself. With `left` on both, `brk_next_span` was the
        # row's OWN span on 100 % of AOBT rows and a genuine neighbour on the
        # segment: one column meaning two different things in the two
        # populations, which cost the segment 102 s.
        lo = np.searchsorted(st, q, "left")
        hi = np.searchsorted(st, q, "right")
        ok_p = lo > 0
        ok_n = hi < len(st)
        pi = np.clip(lo - 1, 0, len(st) - 1)
        ni = np.clip(hi, 0, len(st) - 1)
        prev_s[idx] = np.where(ok_p, sv[pi], np.nan)
        next_s[idx] = np.where(ok_n, sv[ni], np.nan)
        prev_g[idx] = np.where(ok_p, q - st[pi], np.nan)
        next_g[idx] = np.where(ok_n, st[ni] - q, np.nan)
        both = ok_p & ok_n
        mean_s[idx] = np.where(both, (np.where(ok_p, sv[pi], np.nan)
                                      + np.where(ok_n, sv[ni], np.nan)) / 2.0,
                               np.where(ok_p, sv[pi], np.where(ok_n, sv[ni], np.nan)))
        nnb[idx] = len(st)
    out["brk_prev_span"] = prev_s
    out["brk_next_span"] = next_s
    out["brk_prev_gap"] = prev_g
    out["brk_next_gap"] = next_g
    out["brk_span_mean"] = mean_s
    out["brk_n"] = nnb
    return out


def month_features(month: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(X, y, segment mask) for one month, every family built at once."""
    raw = D.load_month(month)
    masked, _ = D.mask_like_ranking(raw)
    dep_pos = masked["PHASE_mvt"].to_numpy() == "DEP"
    dep = masked[dep_pos]
    mvt, sch = dep["MVT_TIME_UTC_mvt"], dep["SCHED_TIME_UTC_mvt"]
    apt = dep["ADEP_mvt"].astype(str)
    X = pd.DataFrame(index=dep.index)
    X["mvt_sched"] = (mvt - sch).dt.total_seconds().to_numpy()
    X["hour"] = mvt.dt.hour.to_numpy() + mvt.dt.minute.to_numpy() / 60.0
    X["sched_hour"] = sch.dt.hour.to_numpy() + sch.dt.minute.to_numpy() / 60.0
    X["dow"] = mvt.dt.dayofweek.to_numpy()
    X["flight_len"] = dep["FLIGHT_mvt"].astype(str).str.len().to_numpy()
    X["apt"] = apt.to_numpy()
    X["apt_stand"] = (apt + ":" + dep["STAND_mvt"].astype(str)).to_numpy()
    X["apt_rwy"] = (apt + ":" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
    X["ac_type"] = dep["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    X["ades"] = dep["ADES_mvt"].astype(str).to_numpy()
    X["flight_rule"] = dep["FLIGHT_RULE_mvt"].astype(str).to_numpy()
    X["op3"] = _op3(dep["FLIGHT_mvt"])
    st = SF.occupancy_features(masked)
    for c in st.columns:
        X[c] = st.reindex(X.index)[c].to_numpy()
    sur = S.build_features(masked, None)
    for c in SURFACE_COLS:
        if c in sur.columns:
            X[c] = sur.reindex(X.index)[c].to_numpy()
    brk = bracket_features(masked)
    for c in BRACKET_COLS:
        X[c] = brk.reindex(X.index)[c].to_numpy()
    y = raw.loc[dep.index, D.TARGET].to_numpy(float)
    seg = (dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].notna()).to_numpy()
    return X.reset_index(drop=True), y, seg


BASE = ["mvt_sched", "hour", "sched_hour", "dow", "flight_len", "apt", "apt_stand",
        "apt_rwy", "ac_type", "ades", "flight_rule"]
FAMILIES = {
    "op": ["op3"],
    "stand": [
        "taxi_ub_pm", "taxi_lb_nm", "taxi_ub_pa", "taxi_lb_na", "off_lb_pm", "off_lb_pa",
        "off_ub_na", "off_ub_nm", "arr_taxi_pa", "arr_taxi_na", "arr_turnaround_pa",
        "arr_td_gap_pa", "same_type_pa", "same_type_na", "dep_prev_aobt_gap",
        "dep_prev_mvt_gap", "dep_next_aobt_gap", "dep_next_mvt_gap"],
    "surface": SURFACE_COLS,
    "bracket": BRACKET_COLS,
}
