"""Airport-day state features for PRC 2026 taxi-out prediction.

A departure's taxi time depends on the state of the airport *that day* — a
de-icing morning, an ATC-restricted afternoon, a snow day — and the ranking
frame shows that state without showing any departure's label. Every movement's
take-off or landing time is present, every departure's Network Manager off-block
(`AOBT_3_flt`, 98.5 % of rows) and filed off-block (`LOBT_flt`) are present, and
every ARRIVAL's block time and taxi time are present. Only the departure's own
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are blanked.

`prc.surface_features` already turns that picture into 30-minute rolling
neighbourhood statistics. This module works one level coarser, at the
**airport-day**, and adds the two things a rolling median cannot see: the whole
day's dispersion and tail share, and the day placed against what that airport
normally looks like.

Why the reference must come from the training months. `docs/shift-2026.md` shows
January 2026 running outside the entire 2025 range of daily take-off-minus-AOBT
on 27 airport-days. A day statistic normalised *within its own month* would show
those days as ordinary, because the whole month moved. So `fit` stores each
airport's reference distribution over the training months' days and `transform`
places every day against it. `fit` reads no label of any kind — the day
statistics are built from timestamps, the NM join flag and arrival taxi times —
so no leave-one-month-out construction is needed.

Leakage contract (`docs/input-rule.md`): a feature may read any movement's
take-off/landing time, any departure's `AOBT_3_flt` / `LOBT_flt` /
`SCHED_TIME_UTC_mvt`, and any ARRIVAL's block time and taxi time. It may never
read a departure's `BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt`. `transform`
blanks those two columns on departure rows inside its own working arrays, so the
guarantee does not depend on the caller passing a masked frame.

Usage::

    ds = DayState().fit(train_frame)      # training months, no labels used
    F  = ds.transform(month_frame)        # -> one row per departure, same index

`transform` is a groupby over airport-days plus one sorted-array scan for the
hourly proxy; a month frame takes a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Behaviour version. Bump on any change that moves a produced value, keep the
#: previous behaviour reachable through explicit constructor arguments, and
#: report consumers in the experiment that made the change.
VERSION = 1

DEP_LABEL_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")

#: seconds of take-off-minus-AOBT above which a departure counts as a long taxi.
#: 1800 s is the assignment's de-icing threshold; it is a constructor argument.
DEICE_THRESHOLD = 1800.0
#: half-width of the "surrounding hour" for the row-level de-icing proxy
DEICE_HALF_WINDOW = 1800.0
#: an airport-day with fewer departures than this gets NaN day statistics
MIN_DEPARTURES = 20
#: floor on the reference spread, so a stable airport does not produce huge z
MIN_IQR = {"sched": 60.0, "gap": 60.0, "span": 30.0, "std": 30.0, "share": 0.01,
           "rate": 0.005, "taxiin": 30.0, "count": 5.0}

#: the day statistics, and which MIN_IQR floor each one uses
DAY_STATS = (
    ("ds_sched_med", "sched"), ("ds_sched_p90", "sched"),
    ("ds_gap_med", "gap"), ("ds_gap_p90", "gap"),
    ("ds_span_med", "span"), ("ds_span_p90", "span"), ("ds_span_std", "std"),
    ("ds_share_span1800", "share"),
    ("ds_nm_rate", "rate"),
    ("ds_taxiin_med", "taxiin"), ("ds_taxiin_p90", "taxiin"),
    ("ds_n_dep", "count"), ("ds_mvt_total", "count"),
)
#: day statistics that also get a z column against the training reference
Z_STATS = ("ds_sched_med", "ds_sched_p90", "ds_gap_p90", "ds_span_med", "ds_span_p90",
           "ds_span_std", "ds_share_span1800", "ds_nm_rate", "ds_taxiin_p90")

ROW_COLUMNS = ("ds_deice_hr", "ds_deice_day", "ds_deice_hr_n")

#: A seven-column subset, offered for the same reason exp-005 offers WINTER_CORE:
#: the full block is 25 columns against a measured day-effect ceiling of 2.4 s
#: (exp-007 finding 5), so a consumer may want the referenced columns the
#: label-free day screen ranked highest (exp-007 finding 6) without the raw
#: levels the model can already reconstruct from the row's own values. Every
#: member is a `_z` column or the row-level proxy, i.e. carries the day's
#: position against that airport's own training-month reference.
CORE = [
    "ds_sched_med_z",          # the day's delay level, the best single marker
    "ds_gap_p90_z",            # the day's plan gap at the tail
    "ds_span_std_z",           # the day's dispersion of take-off - AOBT
    "ds_nm_rate_z",            # the NM join failing, EHAM January 2026's signature
    "ds_share_span1800_z",     # the de-icing proxy, referenced
    "ds_taxiin_p90_z",         # the arrivals' side of the same surface
    "ds_deice_hr",             # the row's own surrounding hour, itself excluded
]

#: The single column that beat its own shuffled control on all three folds
#: (exp-007 finding 11): the share of same-airport departures in the surrounding
#: hour whose take-off minus AOBT exceeds the threshold, the row itself excluded.
#: It is the assignment's de-icing proxy and it is bounded in [0, 1] by
#: construction, so it cannot leave its training range the way the `_z` columns
#: can (exp-007 finding 13).
DEICE = ["ds_deice_hr"]

COLUMNS = ([c for c, _ in DAY_STATS] + [f"{c}_z" for c in Z_STATS] + list(ROW_COLUMNS))

#: no categorical outputs
CAT_FEATURES: tuple[str, ...] = ()


def _epoch(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype="datetime64[ns]")
    out = v.astype("int64").astype("float64")
    out[np.isnat(v)] = np.nan
    return out / 1e9


def assert_no_departure_labels(frame: pd.DataFrame) -> None:
    """Raise if a departure row carries a block time or taxi time.

    Callers that build features from an unmasked training month legitimately
    hold those columns; this helper exists for tests and for a caller that wants
    the check.
    """
    dep = frame["PHASE_mvt"].eq("DEP").to_numpy()
    for c in DEP_LABEL_COLUMNS:
        if c in frame.columns and frame.loc[dep, c].notna().any():
            raise AssertionError(f"{c} is present on departure rows")


def _p90(s: pd.Series) -> float:
    return s.quantile(0.90)


@dataclass
class DayState:
    """Airport-day state, with the reference distribution fitted on training months."""

    deice_threshold: float = DEICE_THRESHOLD
    deice_half_window: float = DEICE_HALF_WINDOW
    min_departures: int = MIN_DEPARTURES

    def __post_init__(self) -> None:
        self.reference_: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame) -> "DayState":
        """Store each airport's reference distribution of every day statistic.

        `train` is one or more concatenated month frames. No label is read: the
        day statistics come from timestamps, the NM join flag and arrival taxi
        times, and the departure label columns are blanked before use.
        """
        days = self._day_table(train)
        ref = days.groupby("apt").agg(
            **{f"{c}__med": (c, "median") for c, _ in DAY_STATS},
            **{f"{c}__lo": (c, lambda s: s.quantile(0.25)) for c, _ in DAY_STATS},
            **{f"{c}__hi": (c, lambda s: s.quantile(0.75)) for c, _ in DAY_STATS},
        )
        self.reference_ = ref
        return self

    @property
    def fitted(self) -> bool:
        return self.reference_ is not None

    @property
    def version(self) -> int:
        return VERSION

    # ------------------------------------------------------------ internals
    def _arrays(self, frame: pd.DataFrame) -> dict:
        """Legal columns only, with the departure labels blanked in local copies."""
        phase = frame["PHASE_mvt"].to_numpy()
        is_dep = phase == "DEP"
        apt = np.where(is_dep, frame["ADEP_mvt"].astype(str), frame["ADES_mvt"].astype(str))
        t = _epoch(frame["MVT_TIME_UTC_mvt"])
        taxi = np.array(frame["TAXITIME_SEC_mvt"], dtype="float64")
        taxi[is_dep] = np.nan                       # hard leakage guard
        day = np.floor(t / 86400.0)                 # UTC day index; NaN-safe
        return {"is_dep": is_dep, "apt": apt, "t": t, "taxi": taxi, "day": day,
                "aobt": _epoch(frame["AOBT_3_flt"]), "lobt": _epoch(frame["LOBT_flt"]),
                "sched": _epoch(frame["SCHED_TIME_UTC_mvt"]),
                "nm": frame["FLIGHT_ID_mvt"].notna().to_numpy().astype("float64")}

    def _day_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per airport-day, the raw statistics."""
        a = self._arrays(frame)
        d, r = a["is_dep"], ~a["is_dep"]
        dep = pd.DataFrame({
            "apt": a["apt"][d], "day": a["day"][d],
            "span": a["t"][d] - a["aobt"][d],
            "gap": a["aobt"][d] - a["lobt"][d],
            "sched": a["t"][d] - a["sched"][d],
            "nm": a["nm"][d],
        })
        arr = pd.DataFrame({"apt": a["apt"][r], "day": a["day"][r], "taxiin": a["taxi"][r]})
        g = dep.groupby(["apt", "day"], sort=True)
        out = pd.DataFrame({
            "ds_sched_med": g["sched"].median(),
            "ds_sched_p90": g["sched"].agg(_p90),
            "ds_gap_med": g["gap"].median(),
            "ds_gap_p90": g["gap"].agg(_p90),
            "ds_span_med": g["span"].median(),
            "ds_span_p90": g["span"].agg(_p90),
            "ds_span_std": g["span"].std(),
            # KNOWN DIVERGENCE (winter, 2026-09-05), queued for VERSION 2: this
            # denominator is every departure of the day, so a no-NM row -- which
            # has no observable span at all -- counts as "not long" and biases the
            # share down by the day's no-NM rate (1-2 %). `_deice`'s hourly column
            # divides by the departures with a FINITE span instead, which is the
            # better definition. `ds_deice_day` below is built to match THIS one,
            # so the two day-level columns are self-consistent and only the hourly
            # column uses the other convention. Not changed here because a
            # measurement was in flight against these bytes.
            "ds_share_span1800": g["span"].agg(lambda s: float((s > self.deice_threshold).mean())),
            "ds_nm_rate": g["nm"].mean(),
            "ds_n_dep": g.size().astype("float64"),
        })
        ag = arr.groupby(["apt", "day"], sort=True)
        out = out.join(pd.DataFrame({
            "ds_taxiin_med": ag["taxiin"].median(),
            "ds_taxiin_p90": ag["taxiin"].agg(_p90),
            "ds_n_arr": ag.size().astype("float64"),
        }), how="left")
        out["ds_mvt_total"] = out["ds_n_dep"] + out["ds_n_arr"].fillna(0.0)
        thin = out["ds_n_dep"] < self.min_departures
        out.loc[thin, [c for c, _ in DAY_STATS]] = np.nan
        return out.drop(columns=["ds_n_arr"]).reset_index()

    def _deice(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Row-level share of long-taxi neighbours, the row itself excluded.

        For each departure: among the same airport's departures whose take-off is
        within `deice_half_window` seconds, the share whose take-off-minus-AOBT
        exceeds `deice_threshold`. The row's own contribution is removed from
        both numerator and denominator, so the column is a statement about the
        neighbourhood and not a monotone function of the row's own span on a thin
        hour.
        """
        a = self._arrays(frame)
        d = a["is_dep"]
        apt, t = a["apt"][d], a["t"][d]
        span = t - a["aobt"][d]
        long = (span > self.deice_threshold).astype("float64")
        long_known = np.isfinite(span).astype("float64")     # no-NM rows have no span
        idx = frame.index[d]
        n = np.full(len(t), np.nan)
        s = np.full(len(t), np.nan)
        w = self.deice_half_window
        for key in np.unique(apt):
            m = np.flatnonzero(apt == key)
            tt = t[m]
            order = np.argsort(tt, kind="stable")
            m, tt = m[order], tt[order]
            ck = np.r_[0.0, np.cumsum(long_known[m])]
            cl = np.r_[0.0, np.cumsum(np.nan_to_num(long[m]) * long_known[m])]
            lo = np.searchsorted(tt, tt - w, side="left")
            hi = np.searchsorted(tt, tt + w, side="right")
            n[m] = (ck[hi] - ck[lo]) - long_known[m]
            s[m] = (cl[hi] - cl[lo]) - np.nan_to_num(long[m]) * long_known[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            hr = np.where(n > 0, s / n, np.nan)
        return pd.DataFrame({"ds_deice_hr": hr, "ds_deice_hr_n": n}, index=idx)

    # ------------------------------------------------------------ transform
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Day-state columns for every departure row of `frame`, in its own index."""
        a = self._arrays(frame)
        d = a["is_dep"]
        days = self._day_table(frame).set_index(["apt", "day"])
        key = pd.MultiIndex.from_arrays([a["apt"][d], a["day"][d]])
        F = days.reindex(key)
        F.index = frame.index[d]

        if self.fitted:
            ref = self.reference_.reindex(a["apt"][d])
            ref.index = frame.index[d]
            for c in Z_STATS:
                floor = MIN_IQR[dict(DAY_STATS)[c]]
                spread = (ref[f"{c}__hi"] - ref[f"{c}__lo"]).clip(lower=floor)
                F[f"{c}_z"] = (F[c] - ref[f"{c}__med"]) / spread
        else:
            for c in Z_STATS:
                F[f"{c}_z"] = np.nan

        F = F.join(self._deice(frame))
        # the day-level version of the same proxy, the row's own span removed
        share, ndep = F["ds_share_span1800"], F["ds_n_dep"]
        own = ((a["t"][d] - a["aobt"][d]) > self.deice_threshold).astype("float64")
        own_known = np.isfinite(a["t"][d] - a["aobt"][d]).astype("float64")
        num = share * ndep - pd.Series(own * own_known, index=F.index)
        den = ndep - pd.Series(own_known, index=F.index)
        F["ds_deice_day"] = np.where(den > 0, num / den, np.nan)
        return F[list(COLUMNS)].astype("float32")
