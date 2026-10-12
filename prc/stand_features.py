"""Stand-level and stand-neighbour features for the PRC 2026 taxi-out task.

Two ideas, both legal under `docs/input-rule.md`:

1. **Stand occupancy bounds.** In the ranking frame arrivals keep their
   on-block time (`BLOCK_TIME_UTC_mvt`); only *departure* block times are
   blanked. A stand holds one aircraft at a time, so for a departure `d` from
   stand `S` with hidden off-block `B`, the arrival that brought the aircraft
   bounds `B` from below and the next aircraft to park at `S` bounds it from
   above. In taxi terms, with `T = MVT_TIME_UTC_mvt` (given),

       T - A_next  <  taxi = T - B  <  T - A_prev.

   The construction problem is that "next arrival" must be defined without
   knowing `B`. Several anchors are offered here; `validate_bounds` measures
   each one's coverage, violation rate and tightness on a month where `B` is
   known, and the caller decides which to clip with and which to feed as
   features. See `experiments/exp-003-anatomy/analysis/q6_bounds.md`.

2. **Per-stand offset statistics.** `BLOCK_TIME - AOBT_3` has a stand-level
   location (pushback convention, contact versus remote stand). `StandStats`
   fits those on training months only and transforms any frame.

3. **Displacement bounds.** `displacement_bounds` counts arrivals between two
   departures at a stand; see the verdict below before using it.

Nothing here reads a departure's own `BLOCK_TIME_UTC_mvt` or
`TAXITIME_SEC_mvt`; `StandStats.fit` reads them from training months, which is
what a fold's training months are for.

**Measured verdicts** (exp-003, fixed LightGBM, bulk = NM join and taxi <= 3600 s,
july / december 2025 folds; `experiments/exp-003-anatomy/RESULTS.md`):

- `occupancy_features(key="key_raw")` — **use these**: -5.2 s / -4.5 s of bulk
  RMSE added to a 40-feature base set, of which the previous-arrival half carries
  -4.9 s. Always with the raw stand name.
- `occupancy_features(key="key_base")` — **do not add on top of the raw-key set**:
  0.0 s. The MARS parent name also breaks the occupancy logic (arrival/departure
  alternation holds on 88 % of stand intervals with the raw name and 83 % with
  the parent).
- `StandStats` — **not in the recommended feature set**: +0.01 s of bulk RMSE
  added to a 40-feature base set, and -0.16 s (july) / +0.06 s (december) when
  removed from the full set, i.e. indistinguishable from zero in both
  directions. LightGBM's native categorical handling of the 2,341-level stand
  name already holds this information. Kept for callers without categorical
  support and for the EDDM 103-108 apron, which is 1,177 July-2026 rows with
  no 2025 history and therefore no 2025 slice on which to settle the question.
- `displacement_bounds` — **feature only, never a clip**: valid on 74 % of the
  rows it fires on, because a stand is not a single-aircraft position.
- Clipping predictions: `max(pred, 0)` is worth 24 s on a naive predictor and
  1.1 s on the model. The occupancy bounds add 0.3 s on top of that and are not
  worth the code path as clips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Module version. Desk rule (2026-09-05): never change a default in `prc/`
#: silently. Bump this, keep the previous behaviour reachable by argument, and
#: report the consumers. Consumers should pin the arguments they depend on;
#: the current pins are `occupancy_features(key="key_raw")` and
#: `StandStats(min_count=30)`.
#:
#: 1.2.0 (2026-09-05, exp-003): adds `airport_neighbour_features`, the second of
#: the two families that move the bulk metric, so callers get both from one
#: module. Purely additive; no existing behaviour changes.
#: 1.1.0 (2026-09-05, exp-003, gpt-5.6-sol round 3): `StandStats.transform` now
#: falls back to the fitted *global* statistic for each column when a row's group
#: is unseen at every level. Before, `st_best_mean` was filled with the global
#: median and `st_best_std`, `st_best_p25`, `st_best_p75` and `st_best_n` were
#: left NaN. Pass `global_fallback=False` for the old behaviour. Consumers:
#: `experiments/exp-003-anatomy/code/ablate.py` only, which does not pin the
#: argument and therefore takes the new default; the `stats` block measures zero
#: on the bulk either way.
#: 1.0.0 (2026-09-05, exp-003): first version consumed outside its own
#: experiment. `occupancy_features` matches arrivals on their **on-block** time
#: (`BLOCK_TIME_UTC_mvt`), not their touchdown; `displacement_bounds` counts
#: arrivals in a 12-hour window bounded below by the previous departure at the
#: stand and returns nothing for the first departure of a stand in the frame.
VERSION = "1.2.0"

__all__ = [
    "VERSION",
    "normalize_stand",
    "movement_airport",
    "add_stand_keys",
    "occupancy_features",
    "StandStats",
    "validate_bounds",
    "displacement_bounds",
    "airport_neighbour_features",
]

_MARS = re.compile(r"(?<=[0-9])[LRC]$")
_PREFIX = re.compile(r"^([A-Za-z]+|[0-9])")


def normalize_stand(stand: pd.Series) -> pd.Series:
    """Collapse a MARS sub-stand onto its parent: 519L/519R -> 519, G11L -> G11.

    Large stands are split into two narrow-body positions with an L/R (and at a
    few airports C) suffix. Two aircraft can occupy the sub-stands at once, so
    the parent key is the right unit for *soft* neighbour statistics and the
    raw name is the right unit for a *hard* occupancy bound.
    """
    return stand.astype("string").str.strip().str.replace(_MARS, "", regex=True)


def movement_airport(df: pd.DataFrame) -> pd.Series:
    """The airport a movement happens at: ADEP for departures, ADES for arrivals."""
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    return pd.Series(
        np.where(is_dep, df["ADEP_mvt"].astype("string"), df["ADES_mvt"].astype("string")),
        index=df.index, dtype="string",
    )


def add_stand_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add `apt`, `stand_raw`, `stand_base`, `stand_pfx`, `key_raw`, `key_base`."""
    out = df.copy()
    out["apt"] = movement_airport(out)
    out["stand_raw"] = out["STAND_mvt"].astype("string").str.strip()
    out["stand_base"] = normalize_stand(out["stand_raw"])
    out["stand_pfx"] = out["stand_base"].str.extract(_PREFIX, expand=False).fillna("?")
    out["key_raw"] = out["apt"] + "|" + out["stand_raw"]
    out["key_base"] = out["apt"] + "|" + out["stand_base"]
    return out


# --------------------------------------------------------------------------
# 1. occupancy neighbours
# --------------------------------------------------------------------------

def _asof(left: pd.DataFrame, on_left: str, right: pd.DataFrame, on_right: str,
          by: str, direction: str, suffix: str, cols: dict[str, str],
          allow_exact: bool = True) -> pd.DataFrame:
    """merge_asof on possibly differently named time columns.

    `on_left` is the departure-side anchor, `on_right` the neighbour-side event
    time (for an arrival that is its **on-block** time, not its touchdown).
    Returns `cols` renamed with `suffix`, indexed like `left`.
    """
    l = left[[on_left, by]].rename(columns={on_left: "_t"})
    l = l.reset_index().rename(columns={"index": "_i"})
    l = l[l["_t"].notna()].sort_values("_t", kind="mergesort")
    keep = [c for c in cols if c not in (on_right, by)]
    r = right[[on_right, by] + keep].rename(columns={on_right: "_t"})
    if on_right in cols:
        r[on_right] = right[on_right]
    r = r[r["_t"].notna()].sort_values("_t", kind="mergesort")
    m = pd.merge_asof(l, r, on="_t", by=by, direction=direction,
                      allow_exact_matches=allow_exact)
    m = m.set_index("_i")
    return m[list(cols)].set_axis([f"{v}_{suffix}" for v in cols.values()], axis=1)


def occupancy_features(mvt: pd.DataFrame, key: str = "key_raw") -> pd.DataFrame:
    """Stand-neighbour features for every departure row of one month's frame.

    `mvt` is a single month of movements as the ranking file gives it: arrivals
    with every column, departures with `BLOCK_TIME_UTC_mvt` and
    `TAXITIME_SEC_mvt` blank. Returns a frame indexed like `mvt`'s departure
    rows. All times are seconds; `taxi_lb_*` is a lower bound on taxi time,
    `taxi_ub_*` an upper bound.
    """
    df = mvt if "key_raw" in mvt.columns else add_stand_keys(mvt)
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep].copy()
    arr = df.loc[~is_dep].copy()
    arr = arr.assign(
        arr_on=arr["BLOCK_TIME_UTC_mvt"],
        arr_td=arr["MVT_TIME_UTC_mvt"],
        arr_taxi=arr["TAXITIME_SEC_mvt"].astype("float64"),
        arr_type=arr["AIRCRAFT_TYPE_mvt"].astype("string"),
    )
    acols = {"arr_on": "arr_on", "arr_td": "arr_td", "arr_taxi": "arr_taxi", "arr_type": "arr_type"}
    dcols = {"AOBT_3_flt": "d_aobt", "MVT_TIME_UTC_mvt": "d_mvt"}

    out = {}
    out["pm"] = _asof(dep, "MVT_TIME_UTC_mvt", arr, "arr_on", key, "backward", "pm", acols)
    out["nm"] = _asof(dep, "MVT_TIME_UTC_mvt", arr, "arr_on", key, "forward", "nm", acols)
    out["pa"] = _asof(dep, "AOBT_3_flt", arr, "arr_on", key, "backward", "pa", acols)
    out["na"] = _asof(dep, "AOBT_3_flt", arr, "arr_on", key, "forward", "na", acols)
    dep_r = dep[[key, "MVT_TIME_UTC_mvt", "AOBT_3_flt"]]
    out["pd"] = _asof(dep, "MVT_TIME_UTC_mvt", dep_r, "MVT_TIME_UTC_mvt", key,
                      "backward", "pd", dcols, allow_exact=False)
    out["nd"] = _asof(dep, "MVT_TIME_UTC_mvt", dep_r, "MVT_TIME_UTC_mvt", key,
                      "forward", "nd", dcols, allow_exact=False)

    F = pd.concat(out.values(), axis=1).reindex(dep.index)
    T = dep["MVT_TIME_UTC_mvt"]
    A = dep["AOBT_3_flt"]
    secs = lambda x: x.dt.total_seconds()

    res = pd.DataFrame(index=dep.index)
    res["taxi_ub_pm"] = secs(T - F["arr_on_pm"])
    res["taxi_lb_nm"] = secs(T - F["arr_on_nm"])
    res["taxi_ub_pa"] = secs(T - F["arr_on_pa"])
    res["taxi_lb_na"] = secs(T - F["arr_on_na"])
    res["off_lb_pm"] = secs(F["arr_on_pm"] - A)
    res["off_lb_pa"] = secs(F["arr_on_pa"] - A)
    res["off_ub_na"] = secs(F["arr_on_na"] - A)
    res["off_ub_nm"] = secs(F["arr_on_nm"] - A)
    res["arr_taxi_pa"] = F["arr_taxi_pa"]
    res["arr_taxi_na"] = F["arr_taxi_na"]
    res["arr_turnaround_pa"] = secs(A - F["arr_on_pa"])
    res["arr_td_gap_pa"] = secs(A - F["arr_td_pa"])
    res["same_type_pa"] = (F["arr_type_pa"] == dep["AIRCRAFT_TYPE_mvt"].astype("string")).astype("float64")
    res["same_type_na"] = (F["arr_type_na"] == dep["AIRCRAFT_TYPE_mvt"].astype("string")).astype("float64")
    res["dep_prev_aobt_gap"] = secs(A - F["d_aobt_pd"])
    res["dep_prev_mvt_gap"] = secs(T - F["d_mvt_pd"])
    res["dep_next_aobt_gap"] = secs(F["d_aobt_nd"] - A)
    res["dep_next_mvt_gap"] = secs(F["d_mvt_nd"] - T)
    return res


# --------------------------------------------------------------------------
# 2. per-stand offset statistics
# --------------------------------------------------------------------------

OFFSET = "blk_aobt"


@dataclass
class StandStats:
    """Group location/scale of `BLOCK_TIME - AOBT_3`, fitted on training months.

    `min_count` levels seen fewer times fall back to the parent key, then to the
    airport, then to the global value, so a stand that appears only in the
    ranking months still gets a sensible number.
    """

    min_count: int = 30
    keys: tuple[str, ...] = ("key_raw", "key_base", "apt_pfx", "apt")
    tables: dict = field(default_factory=dict)
    global_: dict = field(default_factory=dict)

    @staticmethod
    def _prepare(dep: pd.DataFrame) -> pd.DataFrame:
        d = dep if "key_raw" in dep.columns else add_stand_keys(dep)
        if "apt_pfx" not in d.columns:
            d = d.assign(apt_pfx=d["apt"] + "|" + d["stand_pfx"])
        if OFFSET not in d.columns:
            d = d.assign(**{OFFSET: (d["BLOCK_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()})
        return d

    def fit(self, dep_train: pd.DataFrame) -> "StandStats":
        d = self._prepare(dep_train)
        d = d[d[OFFSET].notna()]
        self.global_ = {"med": float(d[OFFSET].median()), "mean": float(d[OFFSET].mean()),
                        "std": float(d[OFFSET].std()), "n": int(len(d)),
                        "p25": float(d[OFFSET].quantile(0.25)),
                        "p75": float(d[OFFSET].quantile(0.75))}
        for k in self.keys:
            g = d.groupby(k, observed=True)[OFFSET].agg(["size", "median", "mean", "std",
                                                         lambda s: s.quantile(0.25),
                                                         lambda s: s.quantile(0.75)])
            g.columns = ["n", "med", "mean", "std", "p25", "p75"]
            self.tables[k] = g[g["n"] >= self.min_count]
        return self

    def transform(self, dep: pd.DataFrame, prefix: str = "st",
                  global_fallback: bool = True) -> pd.DataFrame:
        """`global_fallback=False` restores the pre-1.1.0 behaviour, in which a row
        whose group was unseen at every level got the global *median* as its mean
        and NaN for scale and count."""
        d = self._prepare(dep)
        out = pd.DataFrame(index=d.index)
        filled = {c: pd.Series(np.nan, index=d.index) for c in ("med", "mean", "std", "p25", "p75", "n")}
        for k in self.keys:  # most specific first
            t = self.tables.get(k)
            if t is None or not len(t):
                continue
            j = d[k].map(t["med"]) if False else t.reindex(d[k].to_numpy())
            j.index = d.index
            for c in filled:
                filled[c] = filled[c].where(filled[c].notna(), j[c])
            out[f"{prefix}_{k}_med"] = j["med"].to_numpy()
            out[f"{prefix}_{k}_n"] = j["n"].to_numpy()
        for c, s in filled.items():
            if global_fallback:
                fill = 0.0 if c == "n" else self.global_[c]
            else:
                fill = self.global_["med"] if c in ("med", "mean") else np.nan
            out[f"{prefix}_best_{c}"] = s.fillna(fill)
        return out


# --------------------------------------------------------------------------
# 3. bound validation (needs a month with true block times)
# --------------------------------------------------------------------------

def validate_bounds(mvt_full: pd.DataFrame, key: str = "key_raw") -> pd.DataFrame:
    """Coverage / validity / tightness of each occupancy bound on a labelled month.

    `mvt_full` must carry true departure block times. Returns one row per bound
    with the share of departures it is defined on, the share of *those* it does
    not violate, and quantiles of the slack (bound minus truth, seconds).
    """
    df = add_stand_keys(mvt_full)
    masked = df.copy()
    is_dep = masked["PHASE_mvt"].astype("string") == "DEP"
    truth = masked.loc[is_dep, "BLOCK_TIME_UTC_mvt"].copy()
    masked.loc[is_dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    masked.loc[is_dep, "TAXITIME_SEC_mvt"] = np.nan
    feats = occupancy_features(masked, key=key)
    T = df.loc[is_dep, "MVT_TIME_UTC_mvt"]
    taxi = (T - truth).dt.total_seconds()
    n = len(taxi)
    rows = []
    for name, sign in [("taxi_ub_pm", +1), ("taxi_ub_pa", +1), ("taxi_lb_nm", -1), ("taxi_lb_na", -1)]:
        b = feats[name]
        ok = b.notna()
        slack = (b - taxi) if sign > 0 else (taxi - b)   # >= 0 when the bound holds
        rows.append({
            "bound": name, "coverage": ok.mean(),
            "valid": float((slack[ok] >= 0).mean()),
            "slack_p10": float(np.nanpercentile(slack[ok], 10)),
            "slack_p50": float(np.nanpercentile(slack[ok], 50)),
            "slack_p90": float(np.nanpercentile(slack[ok], 90)),
            "n": int(ok.sum()), "n_dep": n,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. displacement bounds: an arrival that parked while we were "still there"
# --------------------------------------------------------------------------

def displacement_bounds(mvt: pd.DataFrame, key: str = "key_raw", jmax: int = 3,
                        window: int = 43200) -> pd.DataFrame:
    """Hard upper bounds on the hidden off-block from *counting* arrivals.

    At a stand the occupancy alternates arrival -> departure, so between one
    departure's off-block and the next there is exactly one arrival. If `k`
    arrivals landed on the stand between the previous departure's take-off and
    ours, then at most one of them is the aircraft we are flying, so the latest
    `k - 1` of them parked *after* our off-block `B`. Each gives `B < A`, i.e.
    a lower bound on the taxi time `T - B`.

    This is the construction that catches the leave-early rows: the aircraft
    blocked off at 13:51, another aircraft parked on the stand at 14:30, and the
    take-off was at 16:08, so the taxi time is at least 98 minutes even though
    the NM off-block says otherwise. Anchoring on `AOBT_3` cannot see it.

    Returns, for each departure, `disp_lb_j` for j = 1..jmax: the lower bound on
    the taxi time implied by the j-th arrival back from our take-off, defined
    only when at least `j + 1` arrivals fall in the window.
    """
    df = mvt if "key_raw" in mvt.columns else add_stand_keys(mvt)
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep]
    arr = df.loc[~is_dep]
    arr = arr[arr["BLOCK_TIME_UTC_mvt"].notna()]

    def ns(x):  # datetime64 -> int64 nanoseconds, whatever the stored unit is
        return x.dt.as_unit("ns").astype("int64").to_numpy()
    res = pd.DataFrame(index=dep.index, columns=[f"disp_lb_{j}" for j in range(1, jmax + 1)],
                       dtype="float64")
    res["disp_k"] = np.nan

    a_key = arr[key].astype(object).fillna("?").astype(str).to_numpy()
    a_t = ns(arr["BLOCK_TIME_UTC_mvt"])
    o = np.argsort(a_key, kind="mergesort")
    a_key, a_t = a_key[o], a_t[o]
    a_starts = {k: (s, e) for k, s, e in zip(*_runs(a_key))}

    dep_key = dep[key].astype(object).fillna("?").astype(str)
    for k, idx in dep_key.groupby(dep_key).indices.items():
        se = a_starts.get(k)
        if se is None:
            continue
        s, e = se
        at = np.sort(a_t[s:e])
        sub = dep.iloc[idx]
        T = ns(sub["MVT_TIME_UTC_mvt"])
        order = np.argsort(T, kind="mergesort")
        Ts = T[order]
        # previous departure's take-off at the same stand; the count is undefined
        # for the first departure in the frame (no lower edge to count from) and
        # is capped at `window` so that a stand that sat empty for a week does
        # not accumulate a spurious count.
        if len(Ts) < 2:
            continue
        t_prev = np.concatenate([[np.iinfo(np.int64).min // 2], Ts[:-1]])
        t_prev = np.maximum(t_prev, Ts - window * 1_000_000_000)
        first = np.zeros(len(Ts), dtype=bool)
        first[0] = True
        hi = np.searchsorted(at, Ts, "left")          # arrivals strictly before our take-off
        lo = np.searchsorted(at, t_prev, "right")     # arrivals after the previous take-off
        kcount = hi - lo
        out = np.full((len(Ts), jmax), np.nan)
        for j in range(1, jmax + 1):
            ok = (kcount >= j + 1) & (hi - j >= 0) & ~first
            if ok.any():
                out[ok, j - 1] = (Ts[ok] - at[hi[ok] - j]) / 1e9
        inv = np.empty(len(order), dtype=int)
        inv[order] = np.arange(len(order))
        res.iloc[idx, :jmax] = out[inv]
        res.iloc[idx, jmax] = kcount[inv]
    return res


def _runs(sorted_keys):
    """Start/end offsets of each run of equal values in a sorted array."""
    if len(sorted_keys) == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)
    change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(sorted_keys)]])
    return sorted_keys[starts], starts, ends

# --------------------------------------------------------------------------
# 5. airport-neighbour departures (the other family that pays)
# --------------------------------------------------------------------------

def airport_neighbour_features(mvt: pd.DataFrame) -> pd.DataFrame:
    """What the other departures around this one were doing.

    Every input is a column the ranking file supplies for other departures --
    their take-off time and their NM off-block -- never their block or taxi
    time. Together with `occupancy_features` these are the only two families
    that moved the bulk metric in exp-003 (-3.4 s and -4.9 s added to a
    40-feature base set; -3.0 s and -0.9 s in each other's presence).

    Returns a frame indexed like `mvt`'s departure rows:

    - `nbr_apt_30m` departures at the same airport taking off within +/- 30
      minutes, excluding this one;
    - `nbr_apt_prev10_nmtaxi`, `nbr_apt_next10_nmtaxi` mean take-off minus
      AOBT_3 over the ten departures before / after this one at the airport;
    - `nbr_rwy_prev10_nmtaxi` the same over the ten before it on its runway;
    - `nbr_stand_prev5_nmtaxi` the same over the five before it at its stand;
    - `nbr_apt_prev10_mvtsched` mean take-off minus schedule over the ten
      before it at the airport.
    """
    df = mvt if "key_raw" in mvt.columns else add_stand_keys(mvt)
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep].copy()
    dep["nm_taxi"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["mvt_sched"] = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep = dep.sort_values("MVT_TIME_UTC_mvt", kind="mergesort")
    out = pd.DataFrame(index=dep.index)
    prev = lambda g, k: g.transform(lambda x: x.shift(1).rolling(k, min_periods=1).mean())
    out["nbr_apt_prev10_nmtaxi"] = prev(dep.groupby("apt", observed=True)["nm_taxi"], 10).astype("float32")
    out["nbr_apt_next10_nmtaxi"] = dep.groupby("apt", observed=True)["nm_taxi"].transform(
        lambda x: x[::-1].shift(1).rolling(10, min_periods=1).mean()[::-1]).astype("float32")
    out["nbr_rwy_prev10_nmtaxi"] = prev(
        dep.groupby(["apt", "RUNWAY_mvt"], observed=True)["nm_taxi"], 10).astype("float32")
    out["nbr_apt_prev10_mvtsched"] = prev(
        dep.groupby("apt", observed=True)["mvt_sched"], 10).astype("float32")
    out["nbr_stand_prev5_nmtaxi"] = prev(
        dep.groupby("key_raw", observed=True)["nm_taxi"], 5).astype("float32")
    # seconds, whatever unit the parquet stores: `.astype("int64")` on a
    # datetime64[us] column gives microseconds, and dividing by 1e9 turned the
    # +/-30-minute window into +/-21 days in the first version of this code.
    ts = dep["MVT_TIME_UTC_mvt"].dt.as_unit("s").astype("int64")
    parts = []
    for _, idx in dep.groupby("apt", observed=True).indices.items():
        v = ts.to_numpy()[idx]
        parts.append(pd.Series(
            np.searchsorted(v, v + 1800, "right") - np.searchsorted(v, v - 1800, "left") - 1,
            index=dep.index[idx]))
    out["nbr_apt_30m"] = pd.concat(parts).reindex(dep.index).astype("float32")
    return out.reindex(df.loc[is_dep].index)
