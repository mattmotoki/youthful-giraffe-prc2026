"""LIRF-specific signals for the taxi-out task (exp-006).

Rome Fiumicino behaves differently from the other nine airports in two ways that
these features are built for.

1. **The block clock sits between the two off-block times.** At LIRF the airport
   block time is neither the filed off-block (`LOBT_flt`) nor Network Manager's
   actual off-block (`AOBT_3_flt`) but a point between them: the least-squares
   fraction phi in `BLOCK - AOBT = phi * (LOBT - AOBT)` is 0.34 to 0.42 in every
   month of 2025, against 0.06 at EDDF and -0.07 at EHAM. The spread
   `AOBT - LOBT` that phi multiplies grows from a 120 s median in January to
   600 s in July, which is the whole of Rome's summer difficulty. The
   `al_*` block below measures that spread and how far it has drifted right now.

2. **A missing NM join and a missing aircraft type are the same event.** 1,486
   of the 1,487 LIRF rows in 2025 with no `LOBT_flt` also have no
   `AIRCRAFT_TYPE_mvt`, while at the other nine airports the two are disjoint
   (EHAM 515 no-NM, 0 noac). Rome's own movement feed drops the whole record,
   which is why the block time is often left at the schedule. On such a row only
   `FLIGHT_mvt`, `FLIGHT_RULE_mvt`, `ADES_mvt`, `SCHED_TIME_UTC_mvt`,
   `RUNWAY_mvt` and `STAND_mvt` survive; the `noac_*` and `stand_*` blocks below
   are the substitutes for everything else.

Everything is computed from the single month frame handed in, because the
ranking file holds January and July 2026 as two isolated months. Departure
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked on entry, so no branch
can read them; arrival block time and taxi-in are kept, exactly as the ranking
file keeps them (`docs/input-rule.md`).

Usage:

    from prc.lirf_features import lirf_signals
    S = lirf_signals(month_frame)          # indexed like the departure rows

`lirf_block(X)` is the cheap subset that only needs the columns exp-001's
`build_features` already produces, for callers that have the feature frame but
not the raw month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["lirf_signals", "lirf_block", "LIRF_CAT_FEATURES", "PHI_BY_AIRPORT",
           "PhiEncoder", "VERSION"]

#: Module version. Bump on any change to a default or to the meaning of a column,
#: and name the consumers in the same report (desk rule, 2026-09-05).
#:
#: 2  2026-09-05  `PhiEncoder` SUPERSEDED by `prc.phi_features.PhiModel`, which
#:                emits a documented column schema and supports cross-fitting. It
#:                is left in place and unchanged for any caller that still holds a
#:                reference; nothing in the repo uses it. No `lirf_signals` column
#:                was added, renamed or redefined.
#:                Consumers at the time of the bump:
#:                `experiments/exp-001-retro-v1/lirf_step.py`,
#:                `experiments/exp-006-lirf/work/cache_lirf_signals.py`.
#: 1  2026-09-05  first version.
VERSION = 2

#: Least-squares phi in BLOCK - AOBT = phi * (LOBT - AOBT), fitted per airport
#: over all twelve 2025 months (`experiments/exp-006-lirf/work/q2b_phi_airports.py`).
#: Recorded for interpretation; the models fit their own.
PHI_BY_AIRPORT = {
    "LIRF": 0.38, "LTFM": 0.37, "EGLL": 0.28, "EDDM": 0.24, "LFPG": 0.20,
    "LEBL": 0.16, "LSZH": 0.15, "LEMD": 0.15, "EDDF": 0.08, "EHAM": -0.05,
}

LIRF_CAT_FEATURES = ["flight_prefix", "stand_group"]

_BIG = 1 << 34


def _epoch(s: pd.Series) -> np.ndarray:
    v = s.astype("datetime64[ns, UTC]").astype("int64").to_numpy().astype("float64")
    v[s.isna().to_numpy()] = np.nan
    return v / 1e9


def _airport(df: pd.DataFrame) -> pd.Series:
    return df["ADEP_mvt"].where(df["PHASE_mvt"].eq("DEP"), df["ADES_mvt"])


def _packed(group: np.ndarray, t: np.ndarray) -> np.ndarray:
    ok = np.isfinite(t)
    out = np.full(t.shape, -1, dtype="int64")
    out[ok] = group[ok].astype("int64") * _BIG + t[ok].astype("int64")
    return out


def _count_between(ref_group, ref_t, q_group, q_lo, q_hi):
    """Reference events of the same group with time in [lo, hi]."""
    ref = np.sort(_packed(ref_group, ref_t)[_packed(ref_group, ref_t) >= 0])
    ok = np.isfinite(q_lo) & np.isfinite(q_hi) & (q_hi >= q_lo)
    base = np.where(ok, q_group.astype("int64") * _BIG, 0)
    lo = base + np.where(ok, q_lo, 0).astype("int64")
    hi = base + np.where(ok, q_hi, 0).astype("int64")
    cnt = np.searchsorted(ref, hi, "right") - np.searchsorted(ref, lo, "left")
    return np.where(ok, cnt.astype("float64"), np.nan)


def _nearest(ref_group, ref_t, q_group, q_t, side):
    """Index of the nearest same-group reference event before/after each query."""
    ref = _packed(ref_group, ref_t)
    order = np.argsort(ref, kind="stable")
    ref_sorted, idx_sorted = ref[order], order
    good = ref_sorted >= 0
    ref_sorted, idx_sorted = ref_sorted[good], idx_sorted[good]
    q = _packed(q_group, q_t)
    pos = np.searchsorted(ref_sorted, q, "left" if side == "prev" else "right")
    if side == "prev":
        pos = pos - 1
        ok = (pos >= 0) & (q >= 0)
    else:
        ok = (pos < len(ref_sorted)) & (q >= 0)
    pos = np.clip(pos, 0, max(len(ref_sorted) - 1, 0))
    if len(ref_sorted) == 0:
        return np.zeros(len(q), dtype="int64"), np.zeros(len(q), dtype=bool)
    same = ref_sorted[pos] // _BIG == np.where(q >= 0, q // _BIG, -1)
    return idx_sorted[pos], ok & same


def _take(values, idx, ok):
    out = np.full(len(idx), np.nan)
    if len(values):
        out[ok] = values[idx[ok]]
    return out


def _rolling_median_before(group, ev_t, ev_v, q_group, q_t, window):
    """Median of `ev_v` over same-group events with time in [t - window, t).

    Implemented as a per-group sort plus a Python loop over groups; the ten
    airports keep this cheap even on a full month.
    """
    out = np.full(len(q_t), np.nan)
    df_ev = pd.DataFrame({"g": group, "t": ev_t, "v": ev_v}).dropna()
    if df_ev.empty:
        return out
    q = pd.DataFrame({"g": q_group, "t": q_t})
    for g, ev in df_ev.groupby("g", sort=False):
        qi = np.flatnonzero((q.g.to_numpy() == g) & np.isfinite(q.t.to_numpy()))
        if not len(qi):
            continue
        ev = ev.sort_values("t")
        et, evv = ev.t.to_numpy(), ev.v.to_numpy()
        qt = q.t.to_numpy()[qi]
        hi = np.searchsorted(et, qt, "left")
        lo = np.searchsorted(et, qt - window, "left")
        for k in range(len(qi)):
            if hi[k] > lo[k]:
                out[qi[k]] = np.median(evv[lo[k]:hi[k]])
    return out


def _stand_group(stand: pd.Series) -> pd.Series:
    n = pd.to_numeric(stand.astype("string").str.extract(r"^(\d+)")[0], errors="coerce")
    g = (n // 100).astype("Float64")
    return g.astype("object").where(n.notna(), None)


def lirf_signals(frame: pd.DataFrame, airports: tuple[str, ...] | None = None) -> pd.DataFrame:
    """LIRF-motivated departure signals for one month frame.

    `airports` restricts the *output rows* to those departure airports; the
    aggregates are always computed over the whole frame, because the ranking file
    is handed over whole. None (the default) returns every departure row.
    """
    df = frame.copy()
    is_dep = df["PHASE_mvt"].eq("DEP").to_numpy()
    if "BLOCK_TIME_UTC_mvt" in df.columns:
        df.loc[is_dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    if "TAXITIME_SEC_mvt" in df.columns:
        df["TAXITIME_SEC_mvt"] = df["TAXITIME_SEC_mvt"].astype("float64")
        df.loc[is_dep, "TAXITIME_SEC_mvt"] = np.nan

    apt_all = _airport(df)
    apt_code_all = pd.factorize(apt_all.astype("string").fillna("\x00NA"))[0]
    t_all = _epoch(df["MVT_TIME_UTC_mvt"])
    dep = df.loc[is_dep]
    apt_code = apt_code_all[is_dep]
    t_dep = t_all[is_dep]
    aobt = _epoch(dep["AOBT_3_flt"])
    lobt = _epoch(dep["LOBT_flt"])
    sched = _epoch(dep["SCHED_TIME_UTC_mvt"])
    al = aobt - lobt                     # AOBT - LOBT, the spread phi multiplies
    day = np.floor(t_dep / 86400.0)
    daykey = pd.Series(apt_code.astype("int64") * 100000 + day.astype("int64"))

    out = pd.DataFrame(index=dep.index)
    out["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy()

    # --- 1. the block clock: anchors between AOBT and LOBT --------------------
    # BLOCK - AOBT = phi * (LOBT - AOBT) with an airport-specific phi, so an
    # anchor at a fixed phi is a candidate block time. A tree with constant
    # leaves cannot form these from `d_takeoff_aobt` and `d_aobt_lobt` by itself.
    for phi in (0.25, 0.375, 0.5, 0.75):
        out[f"mvt_anchor_{int(phi*1000):03d}"] = (t_dep - aobt) + phi * al
    out["al_gap"] = al
    out["al_gap_slog"] = np.sign(al) * np.log1p(np.abs(al))
    out["al_gap_pos"] = np.maximum(al, 0.0)

    # --- 2. how far the clock has drifted right now --------------------------
    for w, tag in ((3600.0, "60m"), (10800.0, "180m")):
        med = _rolling_median_before(apt_code, aobt, al, apt_code, aobt, w)
        out[f"al_med_{tag}"] = med
        out[f"al_vs_{tag}"] = al - med
    gd = pd.Series(al).groupby(daykey)
    out["al_day_med"] = daykey.map(gd.median()).to_numpy("float64")
    out["al_day_p90"] = daykey.map(gd.quantile(0.9)).to_numpy("float64")
    out["al_vs_day"] = al - out["al_day_med"].to_numpy()

    # --- 3. the pushback queue this flight sat in ----------------------------
    # departures whose NM off-block falls between this flight's filed and actual
    # off-block: how many aircraft left the stand while this one was still on it.
    out["dep_off_between_lobt_aobt"] = _count_between(apt_code, aobt, apt_code, lobt, aobt)
    out["dep_off_between_sched_aobt"] = _count_between(apt_code, aobt, apt_code, sched, aobt)

    # --- 4. the feed state of the day, measured on the other movements -------
    noac = dep["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    nonm = dep["LOBT_flt"].isna().to_numpy()
    n_day = daykey.map(daykey.value_counts()).to_numpy("float64")
    out["day_n_dep"] = n_day
    out["day_noac_n"] = daykey.map(daykey[noac].value_counts()).fillna(0.0).to_numpy("float64")
    out["day_noac_frac"] = out["day_noac_n"].to_numpy() / n_day
    out["day_nonm_frac"] = (daykey.map(daykey[nonm].value_counts()).fillna(0.0).to_numpy("float64") / n_day)
    # neighbours in time rather than in the calendar day
    hourkey = apt_code.astype("int64") * 1000000 + np.floor(t_dep / 3600.0).astype("int64")
    hk = pd.Series(hourkey)
    out["hour_noac_n"] = hk.map(hk[noac].value_counts()).fillna(0.0).to_numpy("float64")
    idx, ok = _nearest(apt_code[noac], t_dep[noac], apt_code, t_dep, "prev")
    out["noac_prev_gap"] = t_dep - _take(t_dep[noac], idx, ok)
    idx, ok = _nearest(apt_code[noac], t_dep[noac], apt_code, t_dep, "next")
    out["noac_next_gap"] = _take(t_dep[noac], idx, ok) - t_dep
    out["noac_neigh_gap"] = np.fmin(out["noac_prev_gap"].to_numpy(), out["noac_next_gap"].to_numpy())

    # --- 5. what the stand says, for rows where the flight plan says nothing --
    arr = ~is_dep
    stand_all = (apt_all.astype("string") + "|" + df["STAND_mvt"].astype("string")).fillna("\x00NA")
    stand_code_all = pd.factorize(stand_all)[0]
    arr_land = t_all[arr]
    arr_block = _epoch(df.loc[arr, "BLOCK_TIME_UTC_mvt"])
    arr_taxiin = df.loc[arr, "TAXITIME_SEC_mvt"].to_numpy("float64")
    arr_noac = df.loc[arr, "AIRCRAFT_TYPE_mvt"].isna().to_numpy().astype("float64")

    # the arrival that last parked here: it brought the airframe, and its
    # on-block is the earliest time this departure could have been at the stand.
    idx, ok = _nearest(stand_code_all[arr], arr_block, stand_code_all[is_dep], t_dep, "prev")
    prev_block = _take(arr_block, idx, ok)
    out["stand_in_block_gap"] = t_dep - prev_block
    out["stand_in_taxiin"] = _take(arr_taxiin, idx, ok)
    out["stand_in_turnaround"] = sched - prev_block
    out["stand_in_land_gap"] = t_dep - _take(arr_land, idx, ok)
    # the inbound aircraft's own record was stripped too: the outage is at the
    # stand, not only on this row.
    out["stand_in_noac"] = _take(arr_noac, idx, ok)
    # the next aircraft to park here: this departure must have left the stand
    # before that on-block, which is a genuine upper bound on the block time.
    idx, ok = _nearest(stand_code_all[arr], arr_block, stand_code_all[is_dep], t_dep, "next")
    out["stand_next_block_gap"] = _take(arr_block, idx, ok) - t_dep
    idx, ok = _nearest(stand_code_all[is_dep], t_dep, stand_code_all[is_dep], t_dep, "prev")
    out["stand_prev_dep_gap"] = t_dep - _take(t_dep, idx, ok)
    idx, ok = _nearest(stand_code_all[is_dep], t_dep, stand_code_all[is_dep], t_dep, "next")
    out["stand_next_dep_gap"] = _take(t_dep, idx, ok) - t_dep

    # --- 6. identity substitutes for a stripped record ----------------------
    flight = dep["FLIGHT_mvt"].astype("string")
    out["flight_prefix"] = flight.str.extract(r"^([A-Za-z]{2,3})")[0].to_numpy()
    out["flight_len"] = flight.str.len().to_numpy("float64")
    out["ades_is_adep"] = (dep["ADES_mvt"].astype("string").fillna("\x00A")
                           == dep["ADEP_mvt"].astype("string").fillna("\x00B")
                           ).to_numpy().astype("float64")
    out["stand_group"] = _stand_group(dep["STAND_mvt"]).to_numpy()
    stand_num = pd.to_numeric(
        dep["STAND_mvt"].astype("string").str.extract(r"^(\d+)")[0], errors="coerce")
    out["stand_num"] = stand_num.to_numpy("float64")
    out["stand_retired"] = stand_num.between(501, 507).fillna(False).to_numpy().astype("float64")

    if airports is not None:
        out = out[dep["ADEP_mvt"].isin(airports).to_numpy()]
    return out


def lirf_block(X: pd.DataFrame) -> pd.DataFrame:
    """The subset of block-clock anchors that needs only exp-001 feature columns.

    `X` must carry `d_takeoff_aobt` and `d_aobt_lobt`. Returns `X` with the
    anchors appended (a copy).
    """
    out = X.copy()
    al = out["d_aobt_lobt"].to_numpy("float64")
    base = out["d_takeoff_aobt"].to_numpy("float64")
    for phi in (0.25, 0.375, 0.5, 0.75):
        out[f"mvt_anchor_{int(phi*1000):03d}"] = base + phi * al
    out["al_gap_slog"] = np.sign(al) * np.log1p(np.abs(al))
    out["al_gap_pos"] = np.maximum(al, 0.0)
    return out


class PhiEncoder:
    """SUPERSEDED by `prc.phi_features.PhiModel`; kept working for old callers.

    Per-airport and per-airport-operator block-clock fraction, fitted on labelled
    months. `PhiModel` does the same thing with a documented column schema, a
    row-count-weighted estimator that bounds one flight's leverage, cross-fitting
    (`PhiCV`) and an analytic leave-one-row-out. Prefer it.

    `BLOCK - AOBT = phi * (LOBT - AOBT)` holds with an airport-specific phi that
    is stable over the year (LIRF 0.34-0.42 in every 2025 month) and an
    operator-specific refinement that is stable too: over the 58 LIRF operators
    with at least 60 departures in each half of 2025, phi ranges from 0.09 to
    1.21 and the correlation between the two halves is **0.948**. Different
    handling agents record the off-block at different moments; the airport's
    block time and Network Manager's differ by that convention.

    Fitted on training months only (it reads the departure block time, which
    exists only there) and applied to a validation or ranking frame. The estimate
    is a shrunk least-squares coefficient, so a group's own row moves it by
    O(1/n) rather than by O(1) as a small-group target mean would - which is why
    this encoding does not have the train-versus-serve noise mismatch that sank
    exp-001's `c9` group means (`experiments/exp-001-retro-v1/RESULTS.md`).

        enc = PhiEncoder().fit(train_frame)
        X = enc.transform(X, valid_frame)      # adds phi_apt, phi_op, anchor_*
    """

    def __init__(self, shrink: float = 2.0e5, min_rows: int = 100):
        #: shrink is in units of sum((LOBT-AOBT)^2); 2e5 is ~ 450 s of spread on
        #: one flight, so a group needs a few hundred flights to move off the
        #: airport prior.
        self.shrink = float(shrink)
        self.min_rows = int(min_rows)
        self.phi_global = 0.0
        self.phi_apt: dict[str, float] = {}
        self.phi_op: dict[tuple[str, str], float] = {}

    @staticmethod
    def _parts(frame: pd.DataFrame):
        dep = frame[frame["PHASE_mvt"].eq("DEP")]
        gap = (dep["LOBT_flt"] - dep["AOBT_3_flt"]).dt.total_seconds().to_numpy("float64")
        r = (dep["BLOCK_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds().to_numpy("float64")
        ok = np.isfinite(gap) & np.isfinite(r)
        return dep, gap, r, ok

    def fit(self, frame: pd.DataFrame) -> "PhiEncoder":
        dep, gap, r, ok = self._parts(frame)
        apt = dep["ADEP_mvt"].astype("string").fillna("\x00NA").to_numpy()
        op = dep["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("\x00NA").to_numpy()
        d = pd.DataFrame(dict(apt=apt, op=op, gap=gap, r=r))[ok]
        self.phi_global = float((d.r * d.gap).sum() / max((d.gap ** 2).sum(), 1e-9))
        for a, s in d.groupby("apt"):
            self.phi_apt[a] = float((s.r * s.gap).sum() / max((s.gap ** 2).sum(), 1e-9))
            prior = self.phi_apt[a]
            for o, t in s.groupby("op"):
                if len(t) < self.min_rows:
                    continue
                num = float((t.r * t.gap).sum()) + self.shrink * prior
                den = float((t.gap ** 2).sum()) + self.shrink
                self.phi_op[(a, o)] = num / den
        return self

    def values(self, frame: pd.DataFrame) -> pd.DataFrame:
        """phi_apt / phi_op and the block-time anchors they imply, per departure row."""
        dep = frame[frame["PHASE_mvt"].eq("DEP")]
        apt = dep["ADEP_mvt"].astype("string").fillna("\x00NA")
        op = dep["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("\x00NA")
        pa = apt.map(self.phi_apt).astype("float64").fillna(self.phi_global).to_numpy()
        keys = pd.Series(list(zip(apt.to_numpy(), op.to_numpy())), index=dep.index)
        po = keys.map(self.phi_op).astype("float64").to_numpy()
        po = np.where(np.isfinite(po), po, pa)
        t = _epoch(dep["MVT_TIME_UTC_mvt"])
        aobt = _epoch(dep["AOBT_3_flt"])
        gap = _epoch(dep["LOBT_flt"]) - aobt
        out = pd.DataFrame(index=dep.index)
        out["phi_apt"] = pa
        out["phi_op"] = po
        out["phi_op_vs_apt"] = po - pa
        out["anchor_phi_apt"] = (t - aobt) - pa * gap
        out["anchor_phi_op"] = (t - aobt) - po * gap
        return out

    def transform(self, X: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        """Append the phi columns to a departure feature frame `X` built from `frame`."""
        v = self.values(frame).reset_index(drop=True)
        out = X.reset_index(drop=True).copy()
        assert len(v) == len(out), (len(v), len(out))
        for c in v.columns:
            out[c] = v[c].to_numpy()
        return out
