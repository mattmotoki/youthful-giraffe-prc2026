"""Surface-state features for PRC 2026 taxi-out prediction.

The ranking frame is a near-complete picture of the airport surface: every
movement's take-off or landing time, every departure's Network Manager
off-block (`AOBT_3_flt`, 98.5 % of rows), and every arrival's on-block time and
taxi-in time. Only the departure's own `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` are blanked. This module turns that picture into
per-departure features.

Leakage contract (`docs/input-rule.md`): a feature may read any other
movement's take-off/landing time, any departure's `AOBT_3_flt`, and any
arrival's block time and taxi time. It may never read a departure's
`BLOCK_TIME_UTC_mvt` or `TAXITIME_SEC_mvt` -- not its own and not a
neighbour's. `build_features` blanks those two columns on departure rows
inside its own working arrays, so the guarantee does not depend on the caller
passing a masked frame.

Usage::

    sf = SurfaceFeatures().fit(train_frame)   # training months, labels used
    F  = sf.transform(month_frame)            # any month -> per-departure rows

`transform` is vectorised (sorted arrays + ``searchsorted``, 30-minute bucket
aggregates, bounded look-ahead for the dominance counts); one month frame takes
a few seconds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Behaviour version. Bump on any change that moves a produced value, record
#: the change in `docs/surface-features-changelog.md`, and keep the previous
#: behaviour reachable through `SurfaceFeatures.v1()` (or explicit arguments).
#: v1 -> v2 (2026-09-05): query-anchored rolling window, restricted overtaking
#: scan, 60-minute runway-configuration window.
#: v2 -> v3 (2026-09-05): `dominance` defaults to a genuinely exact overtaking
#: count. What v2 called "exact" only counted pairs whose push falls inside
#: [push_i, off_i], which drops real pairs on the 0.14 % of departures with a
#: negative span; it is still reachable as `dominance="push_window"`.
#: See the changelog for row impacts.
VERSION = 3

DEP_LABEL_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")

# Defaults for VERSION 2. Every one of these is also a constructor argument on
# `SurfaceFeatures`, so a consumer can pin the value its numbers were produced
# with rather than inheriting a change. `WINDOWS` additionally decides column
# names, so a consumer changing it must use `family_columns(windows)`.
WINDOWS = (300, 600, 900, 1800, 3600)  # runway-pressure windows, seconds
BUCKET = 1200.0                        # runway-configuration bucket width, seconds
CFG_NOW = (-1, 0, 1)                   # three 20-min buckets = the surrounding hour
CFG_PREV = (-4, -3, -2)                # the hour before that
NBR_WINDOW = "1800s"                   # neighbour rolling window
UNIMPEDED_Q = 0.20                     # taxi-time quantile used as unimpeded estimate
SHRINK_K = 20.0                        # pseudo-count for hierarchical shrinkage
ROLLING_ANCHOR = "query"               # v1 used "event"
DOMINANCE = "exact"                    # v1 "heuristic", v2 "push_window"

# VERSION 1 values, for reproducing a number measured before 2026-09-05.
V1_BUCKET = 1800.0
V1_CFG_NOW = (-1, 0, 1)
V1_CFG_PREV = (-3, -2)

LEVELS = [
    ("L0", ["apt"]),
    ("L1", ["apt", "rwy"]),
    ("L2", ["apt", "term", "rwy"]),
    ("L3", ["apt", "stand"]),
    ("L4", ["apt", "stand", "rwy"]),
]
PARENT = {"L1": "L0", "L2": "L1", "L3": "L0", "L4": "L2"}
LOOKUP_ORDER = ["L4", "L2", "L3", "L1", "L0"]

# categorical outputs of transform()
CAT_FEATURES = ["cfg_dep", "cfg_arr"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def epoch(s: pd.Series) -> np.ndarray:
    """tz-aware datetime column -> float64 seconds since epoch, NaN for NaT."""
    v = s.to_numpy(dtype="datetime64[ns]")
    out = v.astype("int64").astype("float64")
    out[np.isnat(v)] = np.nan
    return out / 1e9


def _cat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(a, dtype=str), ":"), np.asarray(b, dtype=str))


def _groups(key: np.ndarray):
    """Yield (group value, row indices) for a 1-D key array."""
    order = np.argsort(key, kind="stable")
    ks = key[order]
    if len(ks) == 0:
        return
    change = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    edges = np.r_[change, len(ks)]
    for i in range(len(edges) - 1):
        yield ks[edges[i]], order[edges[i]:edges[i + 1]]


def _sorted_events(key: np.ndarray, value: np.ndarray) -> dict:
    """{group: sorted finite event times}."""
    ok = np.isfinite(value)
    k, v = key[ok], value[ok]
    return {g: np.sort(v[idx]) for g, idx in _groups(k)}


def _count_between(events: dict, qkey: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Events strictly after `lo` and strictly before `hi`; NaN if either bound is NaN.

    The strict upper bound excludes the query row's own event when hi is its
    own timestamp.
    """
    out = np.zeros(len(qkey), dtype="float64")
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        out[idx] = (np.searchsorted(arr, hi[idx], "left")
                    - np.searchsorted(arr, lo[idx], "right"))
    out[~(np.isfinite(lo) & np.isfinite(hi))] = np.nan
    return out


def _count_le(events: dict, qkey: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.zeros(len(qkey), dtype="float64")
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        out[idx] = np.searchsorted(arr, t[idx], "right")
    out[~np.isfinite(t)] = np.nan
    return out


def _neighbour_gaps(events: dict, qkey: np.ndarray, t: np.ndarray):
    """Seconds since the previous event and until the next, excluding self."""
    prev = np.full(len(qkey), np.nan)
    nxt = np.full(len(qkey), np.nan)
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        if len(arr) == 0:
            continue
        tq = t[idx]
        lo = np.searchsorted(arr, tq, "left")   # events strictly before tq
        hi = np.searchsorted(arr, tq, "right")  # events at or before tq
        p = np.where(lo > 0, arr[np.clip(lo - 1, 0, len(arr) - 1)], np.nan)
        n = np.where(hi < len(arr), arr[np.clip(hi, 0, len(arr) - 1)], np.nan)
        prev[idx] = tq - p
        nxt[idx] = n - tq
    bad = ~np.isfinite(t)
    prev[bad] = np.nan
    nxt[bad] = np.nan
    return prev, nxt


def _rolling_stat_at(ev_t, ev_v, q_t, window=NBR_WINDOW, stat="median",
                     anchor: str = ROLLING_ANCHOR) -> np.ndarray:
    """`stat` over the events in the half-open window (q - window, q), per query.

    The query times are merged into the event series as NaN-valued rows so that
    the window is anchored on the query and not on the last preceding event: a
    read at the last event would use `(e - window, e]`, which after a quiet
    period returns a statistic from outside the window entirely. Events at
    exactly `q` are excluded, so a row never reads its own value.
    """
    out = np.full(len(q_t), np.nan)
    ok = np.isfinite(ev_t) & np.isfinite(ev_v)
    qok = np.isfinite(q_t)
    if not ok.any() or not qok.any():
        return out
    et, evv = ev_t[ok], ev_v[ok]
    if anchor == "event":  # VERSION 1: read the event-anchored window at the
        order = np.argsort(et, kind="stable")   # last event before the query
        et_v1, ev_v1 = et[order], evv[order]
        s1 = pd.Series(ev_v1, index=pd.to_datetime(et_v1, unit="s"))
        roll1 = getattr(s1.rolling(window), stat)().to_numpy()
        pos = np.searchsorted(et_v1, q_t, "left") - 1
        good = np.isfinite(q_t) & (pos >= 0)
        out[good] = roll1[pos[good]]
        return out
    qt = q_t[qok]
    times = np.concatenate([et, qt])
    vals = np.concatenate([evv, np.full(len(qt), np.nan)])
    order = np.argsort(times, kind="stable")
    s = pd.Series(vals[order], index=pd.to_datetime(times[order], unit="s"))
    roll = getattr(s.rolling(window, closed="left"), stat)().to_numpy()
    is_q = order >= len(et)
    res = np.full(len(qt), np.nan)
    res[order[is_q] - len(et)] = roll[is_q]
    out[qok] = res
    return out


def _grouped_rolling_stat(gkey_ev, ev_t, ev_v, gkey_q, q_t,
                          window=NBR_WINDOW, stat="median",
                          anchor: str = ROLLING_ANCHOR) -> np.ndarray:
    out = np.full(len(q_t), np.nan)
    ev_by_g = {g: idx for g, idx in _groups(gkey_ev)}
    for g, qidx in _groups(gkey_q):
        eidx = ev_by_g.get(g)
        if eidx is None:
            continue
        out[qidx] = _rolling_stat_at(ev_t[eidx], ev_v[eidx], q_t[qidx], window, stat, anchor)
    return out


def _dominance(push: np.ndarray, off: np.ndarray, gkey: np.ndarray,
               mode: str = DOMINANCE, kmax: int = 400, stop_after: int = 25):
    """(overtaken, overtook) per departure, inside each group.

    overtaken_i = #{j: push_j > push_i and off_j < off_i}
    overtook_i  = #{j: push_j < push_i and off_j > off_i}

    An overtaking pair must have `push_j` inside `[push_i, off_i]`, so with the
    group sorted by push time the scan over offsets `s` may stop as soon as no
    `i` satisfies `push_{i+s} < off_i`. Because `push_{i+s}` is non-decreasing
    in `s` for every `i`, that condition never becomes true again: the stop is
    exact, not a heuristic cut-off.
    """
    n = len(push)
    overtaken = np.full(n, np.nan)
    overtook = np.full(n, np.nan)
    for _, idx in _groups(gkey):
        ok = np.isfinite(push[idx]) & np.isfinite(off[idx])
        sub = idx[ok]
        if len(sub) < 2:
            if len(sub):
                overtaken[sub] = 0.0
                overtook[sub] = 0.0
            continue
        p, o = push[sub], off[sub]
        order = np.argsort(p, kind="stable")
        p, o = p[order], o[order]
        m = len(p)
        a = np.zeros(m, dtype="float64")
        b = np.zeros(m, dtype="float64")
        if mode == "heuristic":  # VERSION 1: stop after `stop_after` empty offsets
            dry = 0
            for s in range(1, min(kmax, m - 1) + 1):
                hit = (p[s:] > p[:-s]) & (o[s:] < o[:-s])
                if hit.any():
                    a[:-s] += hit
                    b[s:] += hit
                    dry = 0
                else:
                    dry += 1
                    if dry >= stop_after:
                        break
        elif mode == "push_window":  # VERSION 2
            # Counts only pairs whose push falls inside [push_i, off_i]. That is
            # the whole definition when every span is non-negative, but 0.14 % of
            # departures have off < push (a take-off stamped before the NM
            # off-block), and for those the restriction silently drops real pairs.
            for s in range(1, m):
                reachable = p[s:] < o[:-s]
                if not reachable.any():
                    break
                hit = reachable & (p[s:] > p[:-s]) & (o[s:] < o[:-s])
                if hit.any():
                    a[:-s] += hit
                    b[s:] += hit
        else:  # "exact", VERSION 3 default
            # The forward scan's stopping rule needs off_j >= push_j, so it runs
            # on the rows that satisfy it; every pair involving a negative-span
            # row is then counted directly. Negative spans are rare, so the
            # direct part is cheap and the result is the definition above with
            # no side condition.
            okspan = o >= p
            idx_n = np.flatnonzero(okspan)
            if len(idx_n) > 1:
                pn, on = p[idx_n], o[idx_n]
                an = np.zeros(len(pn), dtype="float64")
                bn = np.zeros(len(pn), dtype="float64")
                for s in range(1, len(pn)):
                    if not (pn[s:] < on[:-s]).any():
                        break
                    hit = (pn[s:] > pn[:-s]) & (on[s:] < on[:-s])
                    if hit.any():
                        an[:-s] += hit
                        bn[s:] += hit
                a[idx_n] = an
                b[idx_n] = bn
            idx_x = np.flatnonzero(~okspan)
            for lo in range(0, len(idx_x), 256):
                xi = idx_x[lo:lo + 256]
                px, ox = p[xi], o[xi]
                a += ((px[None, :] > p[:, None]) & (ox[None, :] < o[:, None])).sum(1)
                b += ((px[None, :] < p[:, None]) & (ox[None, :] > o[:, None])).sum(1)
            for lo in range(0, len(idx_x), 256):
                xi = idx_x[lo:lo + 256]
                px, ox = p[xi], o[xi]
                a[xi] = ((p[None, :] > px[:, None]) & (o[None, :] < ox[:, None])).sum(1)
                b[xi] = ((p[None, :] < px[:, None]) & (o[None, :] > ox[:, None])).sum(1)
        overtaken[sub[order]] = a
        overtook[sub[order]] = b
    return overtaken, overtook


def _kth_after(events: dict, qkey: np.ndarray, q: np.ndarray, ks) -> dict:
    """Time of the k-th event at or after `q`, for each k in `ks`."""
    out = {k: np.full(len(qkey), np.nan) for k in ks}
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        if len(arr) == 0:
            continue
        lo = np.searchsorted(arr, q[idx], "left")
        for k in ks:
            pos = lo + k - 1
            ok = pos < len(arr)
            v = np.full(len(idx), np.nan)
            v[ok] = arr[pos[ok]]
            out[k][idx] = v
    for k in ks:
        out[k][~np.isfinite(q)] = np.nan
    return out


def _kth_before(events: dict, qkey: np.ndarray, q: np.ndarray, ks) -> dict:
    """Time of the k-th event strictly before `q`, for each k in `ks`."""
    out = {k: np.full(len(qkey), np.nan) for k in ks}
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        if len(arr) == 0:
            continue
        hi = np.searchsorted(arr, q[idx], "left")
        for k in ks:
            pos = hi - k
            ok = pos >= 0
            v = np.full(len(idx), np.nan)
            v[ok] = arr[pos[ok]]
            out[k][idx] = v
    for k in ks:
        out[k][~np.isfinite(q)] = np.nan
    return out


def _next_prev_at(events: dict, qkey: np.ndarray, q: np.ndarray,
                  exclude: np.ndarray | None = None):
    """First event at or after `q` and last event strictly before `q`.

    When `exclude` is given, an event exactly equal to the row's own value is
    stepped over (the row's own movement lives in the same event series).
    """
    nxt = np.full(len(qkey), np.nan)
    prv = np.full(len(qkey), np.nan)
    empty = np.empty(0)
    for g, idx in _groups(qkey):
        arr = events.get(g, empty)
        if len(arr) == 0:
            continue
        tq = q[idx]
        lo = np.searchsorted(arr, tq, "left")
        if exclude is not None:
            ex = exclude[idx]
            first = np.where(lo < len(arr), arr[np.clip(lo, 0, len(arr) - 1)], np.nan)
            step = np.isfinite(first) & (first == ex)
            lo = lo + step
        nxt[idx] = np.where(lo < len(arr), arr[np.clip(lo, 0, len(arr) - 1)], np.nan)
        p = np.searchsorted(arr, tq, "left")
        prv[idx] = np.where(p > 0, arr[np.clip(p - 1, 0, len(arr) - 1)], np.nan)
    bad = ~np.isfinite(q)
    nxt[bad] = np.nan
    prv[bad] = np.nan
    return nxt, prv


def _terminal(stand: np.ndarray) -> np.ndarray:
    """Stand prefix: leading letters if any, else the first character."""
    s = pd.Series(stand, dtype="string").fillna("NA")
    pref = s.str.extract(r"^([A-Za-z]+)", expand=False)
    return pref.fillna(s.str.slice(0, 1)).fillna("NA").to_numpy().astype(str)


def _popcount(x: np.ndarray) -> np.ndarray:
    v = x.astype("uint64").copy()
    c = np.zeros(len(v), dtype="int16")
    while v.any():
        c += (v & np.uint64(1)).astype("int16")
        v >>= np.uint64(1)
    return c


# --------------------------------------------------------------------------
# fitted state
# --------------------------------------------------------------------------


@dataclass
class SurfaceFeatures:
    """Learned pieces plus every constant a produced value depends on.

    The behaviour constants are constructor arguments so that a consumer can
    pin what its numbers were produced with instead of inheriting a change to
    the module (desk rule, 2026-09-05). Defaults are VERSION 2;
    `SurfaceFeatures.v1()` returns an instance configured the pre-2026-09-05
    way. `docs/surface-features-changelog.md` lists what moved.
    """

    bucket: float = BUCKET
    cfg_now: tuple = CFG_NOW
    cfg_prev: tuple = CFG_PREV
    windows: tuple = WINDOWS
    nbr_window: str = NBR_WINDOW
    unimpeded_q: float = UNIMPEDED_Q
    shrink_k: float = SHRINK_K
    rolling_anchor: str = ROLLING_ANCHOR
    dominance: str = DOMINANCE
    unimpeded: dict = field(default_factory=dict)
    throughput: pd.DataFrame | None = None
    fitted: bool = False

    @property
    def version(self) -> int:
        """2 for the defaults; 1 when configured the pre-2026-09-05 way."""
        if (self.bucket == V1_BUCKET and tuple(self.cfg_prev) == V1_CFG_PREV
                and self.rolling_anchor == "event" and self.dominance == "heuristic"):
            return 1
        return 2 if self.dominance == "push_window" else VERSION

    def __setstate__(self, state: dict) -> None:
        """Unpickle an instance saved before the behaviour fields existed.

        Such a pickle carries only the fitted state, so the behaviour fields
        would be missing entirely rather than defaulted. They were saved under
        VERSION 1, so that is what they are restored as.
        """
        restored = {f.name: (f.default_factory() if f.default_factory is not dataclasses.MISSING
                             else f.default)
                    for f in dataclasses.fields(self)}
        if "bucket" not in state:
            restored.update(bucket=V1_BUCKET, cfg_now=V1_CFG_NOW, cfg_prev=V1_CFG_PREV,
                            rolling_anchor="event", dominance="heuristic")
        restored.update(state)
        self.__dict__.update(restored)

    @classmethod
    def v1(cls, **kw) -> "SurfaceFeatures":
        """The behaviour in force before 2026-09-05, for reproducing old numbers."""
        return cls(bucket=V1_BUCKET, cfg_now=V1_CFG_NOW, cfg_prev=V1_CFG_PREV,
                   rolling_anchor="event", dominance="heuristic", **kw)

    @classmethod
    def v2(cls, **kw) -> "SurfaceFeatures":
        """VERSION 2: every exp-004 number in the record was produced with this."""
        return cls(dominance="push_window", **kw)

    def fit(self, train: pd.DataFrame) -> "SurfaceFeatures":
        d = train[train["PHASE_mvt"] == "DEP"]
        y = d["TAXITIME_SEC_mvt"].to_numpy(float)
        keep = np.isfinite(y) & (y > 0) & (y <= 7200)
        d, y = d[keep], y[keep]
        stand = d["STAND_mvt"].astype(str).to_numpy()
        tbl = pd.DataFrame({
            "apt": d["ADEP_mvt"].astype(str).to_numpy(),
            "rwy": d["RUNWAY_mvt"].astype(str).to_numpy(),
            "stand": stand,
            "term": _terminal(stand),
            "y": y,
        })
        stats = {}
        for name, keys in LEVELS:
            g = tbl.groupby(keys)["y"]
            stats[name] = pd.DataFrame({"n": g.size(), "q": g.quantile(self.unimpeded_q),
                                        "med": g.median()})
        glob_q, glob_m = float(np.quantile(y, self.unimpeded_q)), float(np.median(y))
        k = self.shrink_k
        shrunk = {"L0": stats["L0"].assign(
            sq=lambda f: (f["n"] * f["q"] + k * glob_q) / (f["n"] + k),
            sm=lambda f: (f["n"] * f["med"] + k * glob_m) / (f["n"] + k))}
        for name in ["L1", "L2", "L3", "L4"]:
            pname = PARENT[name]
            keys = dict(LEVELS)[name]
            s = stats[name].reset_index()
            pkeys = list(shrunk[pname].index.names)
            if "term" in pkeys and "term" not in s.columns:
                s["term"] = _terminal(s["stand"].to_numpy())
            p = shrunk[pname][["sq", "sm"]].rename(columns={"sq": "pq", "sm": "pm"}).reset_index()
            s = s.merge(p, on=pkeys, how="left")
            s["pq"] = s["pq"].fillna(glob_q)
            s["pm"] = s["pm"].fillna(glob_m)
            s["sq"] = (s["n"] * s["q"] + k * s["pq"]) / (s["n"] + k)
            s["sm"] = (s["n"] * s["med"] + k * s["pm"]) / (s["n"] + k)
            shrunk[name] = s.set_index(keys)
        self.unimpeded = {"levels": shrunk, "glob_q": glob_q, "glob_m": glob_m}
        self.throughput = _peak_throughput(train)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-departure features for one isolated month frame."""
        return build_features(frame, self)

    def transform_by_month(self, frame: pd.DataFrame, month_col: str = "month") -> pd.DataFrame:
        """Transform a multi-month frame one month at a time.

        The ranking file holds January and July 2026 as two isolated months, so
        a nearest-event or rolling feature must never reach across the gap
        between them. On the live ranking file this changes 0.4 % of rows for
        the nearest-event columns and 2.7 % for `nbr_taxiin_rwy` against a
        whole-file transform.
        """
        parts = [build_features(g, self) for _, g in frame.groupby(month_col, sort=True)]
        return pd.concat(parts).reindex(frame.index[frame["PHASE_mvt"] == "DEP"])


def _peak_throughput(frame: pd.DataFrame) -> pd.DataFrame:
    """Per airport: a robust ceiling on hourly departures and hourly movements."""
    isdep = frame["PHASE_mvt"].to_numpy() == "DEP"
    apt = np.where(isdep, frame["ADEP_mvt"].astype(str), frame["ADES_mvt"].astype(str))
    t = epoch(frame["MVT_TIME_UTC_mvt"])
    ok = np.isfinite(t)
    g = pd.DataFrame({"apt": apt[ok], "hour": np.floor(t[ok] / 3600.0),
                      "dep": isdep[ok].astype(int)}).groupby(["apt", "hour"]).agg(
        dep=("dep", "sum"), mvt=("dep", "size"))
    return g.groupby("apt").quantile(0.995).rename(
        columns={"dep": "peak_dep_hr", "mvt": "peak_mvt_hr"})


def assert_no_departure_labels(frame: pd.DataFrame) -> None:
    """Fail loudly if a departure label survived into a frame used for features."""
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    for c in DEP_LABEL_COLUMNS:
        if c in frame.columns and frame.loc[dep, c].notna().any():
            raise ValueError(f"{c} is populated on departure rows")


# --------------------------------------------------------------------------
# the feature build
# --------------------------------------------------------------------------

FAMILY_COLUMNS = {
    "queue": ["push_src", "span", "span_lobt", "q_rwy", "q_apt", "q_rwy_lobt", "q_apt_lobt",
              "q_rwy_rate", "q_apt_rate", "q_arr_rwy", "q_arr_apt"],
    "pressure": ([f"{a}_{b}{w}" for w in WINDOWS
                  for a in ("dep_rwy", "dep_apt", "arr_rwy", "arr_apt")
                  for b in ("pre", "post")]
                 + ["mixed_mode", "gap_prev_rwy", "gap_next_rwy", "gap_prev_apt",
                    "gap_next_apt", "rank10_rwy", "n10_rwy"]),
    "occupancy": ["taxiing_at_push", "taxiing_at_off", "arr_taxiing_at_push",
                  "arr_taxiing_at_off", "overtaken", "overtook"],
    "config": ["cfg_dep", "cfg_arr", "n_dep_rwy", "n_arr_rwy", "cfg_changed", "rwy_share"],
    "flow": ["dep_rate_hr", "mvt_rate_hr", "sat_dep", "sat_mvt",
             "nbr_span_apt", "nbr_span_rwy", "nbr_taxiin_apt", "nbr_taxiin_rwy"],
    "geometry": ["unimpeded_q", "stand_rwy_med", "own_excess", "nbr_excess_apt", "nbr_excess_rwy"],
    "stand": ["stand_nxt_arr_dt", "stand_prv_arr_dt", "stand_nxt_dep_off_dt", "stand_prv_dep_off_dt",
              "stand_nxt_dep_push_dt", "stand_ub_offset", "stand_lb_taxi", "stand_arr_in_span",
              "stand_mvt_2h", "stand_turnaround", "stand_lb_taxi_pos", "stand_ub_offset_clip",
              "stand_bound_active"],
}
FAMILIES = tuple(FAMILY_COLUMNS)

# A finer encoding of the same queue information: instead of counting take-offs
# between the push and the take-off, hand the model the actual service times of
# the runway relative to its push. Kept out of FAMILY_COLUMNS so that
# `all families` keeps meaning the six families of the assignment plus `stand`.
KTH_AFTER = (1, 2, 3, 5, 8)
KTH_BEFORE = (1, 2, 3)
EXTRA_FAMILY_COLUMNS = {
    "queue2": ([f"trwy_after{k}" for k in KTH_AFTER] + [f"tapt_after{k}" for k in KTH_AFTER]
               + [f"trwy_before{k}" for k in KTH_BEFORE]
               + ["push_demand_apt30", "push_demand_rwy30", "push_rank_apt30"]),
}


def build_features(frame: pd.DataFrame, sf: "SurfaceFeatures | None" = None) -> pd.DataFrame:
    """Per-departure surface-state features from one month frame.

    Behaviour constants come from `sf`; with `sf=None` the VERSION 2 defaults
    apply and the fitted families (geometry, saturation) are skipped.
    """
    par = sf if sf is not None else SurfaceFeatures()
    phase = frame["PHASE_mvt"].to_numpy()
    is_dep = phase == "DEP"
    is_arr = phase == "ARR"
    apt_all = np.where(is_dep, frame["ADEP_mvt"].astype(str), frame["ADES_mvt"].astype(str))
    rwy_all = frame["RUNWAY_mvt"].astype(str).to_numpy()
    t_all = epoch(frame["MVT_TIME_UTC_mvt"])
    blk_all = epoch(frame["BLOCK_TIME_UTC_mvt"])
    # copy explicitly: to_numpy(dtype=...) returns a view when the column is
    # already float64 (which it is after prc.data.mask_like_ranking), and the
    # blanking below would then write into the caller's frame.
    taxi_all = np.array(frame["TAXITIME_SEC_mvt"], dtype="float64")
    # hard leakage guard: departure labels are never readable here
    blk_all[is_dep] = np.nan
    taxi_all[is_dep] = np.nan
    aobt_all = epoch(frame["AOBT_3_flt"])
    lobt_all = epoch(frame["LOBT_flt"])
    sched_all = epoch(frame["SCHED_TIME_UTC_mvt"])

    dep_i = np.flatnonzero(is_dep)
    arr_i = np.flatnonzero(is_arr)

    dep_apt, dep_rwy = apt_all[dep_i], rwy_all[dep_i]
    dep_ar = _cat(dep_apt, dep_rwy)
    dep_off = t_all[dep_i]
    dep_aobt, dep_lobt, dep_sched = aobt_all[dep_i], lobt_all[dep_i], sched_all[dep_i]
    dep_stand = frame["STAND_mvt"].astype(str).to_numpy()[dep_i]
    dep_push = np.where(np.isfinite(dep_aobt), dep_aobt,
                        np.where(np.isfinite(dep_lobt), dep_lobt, dep_sched))
    push_src = np.where(np.isfinite(dep_aobt), 0, np.where(np.isfinite(dep_lobt), 1, 2))
    dep_lobt_eff = np.where(np.isfinite(dep_lobt), dep_lobt, dep_sched)

    arr_apt, arr_rwy = apt_all[arr_i], rwy_all[arr_i]
    arr_ar = _cat(arr_apt, arr_rwy)
    arr_land, arr_on, arr_taxiin = t_all[arr_i], blk_all[arr_i], taxi_all[arr_i]
    arr_stand = frame["STAND_mvt"].astype(str).to_numpy()[arr_i]
    dep_stand_key = _cat(dep_apt, dep_stand)

    E = {
        "dep_off_apt": _sorted_events(dep_apt, dep_off),
        "dep_off_ar": _sorted_events(dep_ar, dep_off),
        "dep_push_apt": _sorted_events(dep_apt, dep_push),
        "arr_land_apt": _sorted_events(arr_apt, arr_land),
        "arr_land_ar": _sorted_events(arr_ar, arr_land),
        "arr_on_apt": _sorted_events(arr_apt, arr_on),
    }

    F = pd.DataFrame(index=frame.index[dep_i])
    span = dep_off - dep_push

    if True:  # family: queue
        F["push_src"] = push_src.astype(np.int8)
        F["span"] = span
        F["span_lobt"] = dep_off - dep_lobt_eff
        q_rwy = _count_between(E["dep_off_ar"], dep_ar, dep_push, dep_off)
        q_apt = _count_between(E["dep_off_apt"], dep_apt, dep_push, dep_off)
        F["q_rwy"] = q_rwy
        F["q_apt"] = q_apt
        F["q_rwy_lobt"] = _count_between(E["dep_off_ar"], dep_ar, dep_lobt_eff, dep_off)
        F["q_apt_lobt"] = _count_between(E["dep_off_apt"], dep_apt, dep_lobt_eff, dep_off)
        with np.errstate(invalid="ignore", divide="ignore"):
            F["q_rwy_rate"] = q_rwy / np.maximum(span, 60.0) * 600.0
            F["q_apt_rate"] = q_apt / np.maximum(span, 60.0) * 600.0
        F["q_arr_rwy"] = _count_between(E["arr_land_ar"], dep_ar, dep_push, dep_off)
        F["q_arr_apt"] = _count_between(E["arr_land_apt"], dep_apt, dep_push, dep_off)

    if True:  # family: pressure (also feeds flow)
        for w in par.windows:
            F[f"dep_rwy_pre{w}"] = _count_between(E["dep_off_ar"], dep_ar, dep_off - w, dep_off)
            F[f"dep_rwy_post{w}"] = _count_between(E["dep_off_ar"], dep_ar, dep_off, dep_off + w)
            F[f"dep_apt_pre{w}"] = _count_between(E["dep_off_apt"], dep_apt, dep_off - w, dep_off)
            F[f"dep_apt_post{w}"] = _count_between(E["dep_off_apt"], dep_apt, dep_off, dep_off + w)
            F[f"arr_rwy_pre{w}"] = _count_between(E["arr_land_ar"], dep_ar, dep_off - w, dep_off)
            F[f"arr_rwy_post{w}"] = _count_between(E["arr_land_ar"], dep_ar, dep_off, dep_off + w)
            F[f"arr_apt_pre{w}"] = _count_between(E["arr_land_apt"], dep_apt, dep_off - w, dep_off)
            F[f"arr_apt_post{w}"] = _count_between(E["arr_land_apt"], dep_apt, dep_off, dep_off + w)
        F["mixed_mode"] = ((F["arr_rwy_pre900"].fillna(0) + F["arr_rwy_post900"].fillna(0)) > 0).astype(np.int8)
        prev_r, next_r = _neighbour_gaps(E["dep_off_ar"], dep_ar, dep_off)
        prev_a, next_a = _neighbour_gaps(E["dep_off_apt"], dep_apt, dep_off)
        F["gap_prev_rwy"], F["gap_next_rwy"] = prev_r, next_r
        F["gap_prev_apt"], F["gap_next_apt"] = prev_a, next_a
        dfb = pd.DataFrame({"b": _cat(dep_ar, np.floor(dep_off / 600.0).astype("int64")), "t": dep_off})
        F["rank10_rwy"] = dfb.groupby("b")["t"].rank(method="first").to_numpy()
        F["n10_rwy"] = dfb.groupby("b")["t"].transform("size").to_numpy()

    if True:  # family: occupancy
        F["taxiing_at_push"] = (_count_le(E["dep_push_apt"], dep_apt, dep_push)
                                - _count_le(E["dep_off_apt"], dep_apt, dep_push) - 1.0)
        F["taxiing_at_off"] = (_count_le(E["dep_push_apt"], dep_apt, dep_off)
                               - _count_le(E["dep_off_apt"], dep_apt, dep_off))
        F["arr_taxiing_at_push"] = (_count_le(E["arr_land_apt"], dep_apt, dep_push)
                                    - _count_le(E["arr_on_apt"], dep_apt, dep_push))
        F["arr_taxiing_at_off"] = (_count_le(E["arr_land_apt"], dep_apt, dep_off)
                                   - _count_le(E["arr_on_apt"], dep_apt, dep_off))
        overtaken, overtook = _dominance(dep_push, dep_off, dep_apt, par.dominance)
        F["overtaken"], F["overtook"] = overtaken, overtook

    if True:  # family: config
        for k, v in _configuration(dep_apt, dep_rwy, dep_off, arr_apt, arr_rwy, arr_land,
                                   par.bucket, par.cfg_now, par.cfg_prev).items():
            F[k] = v

    if True:  # family: flow
        dep_hr = F["dep_apt_pre1800"].to_numpy() + F["dep_apt_post1800"].to_numpy()
        arr_hr = F["arr_apt_pre1800"].to_numpy() + F["arr_apt_post1800"].to_numpy()
        F["dep_rate_hr"] = dep_hr
        F["mvt_rate_hr"] = dep_hr + arr_hr
        if sf is not None and sf.throughput is not None:
            peak = sf.throughput.reindex(dep_apt)
            F["sat_dep"] = dep_hr / peak["peak_dep_hr"].to_numpy()
            F["sat_mvt"] = (dep_hr + arr_hr) / peak["peak_mvt_hr"].to_numpy()
        real = np.isfinite(dep_aobt)
        F["nbr_span_apt"] = _grouped_rolling_stat(dep_apt[real], dep_off[real], span[real], dep_apt, dep_off,
                                                  par.nbr_window, "median", par.rolling_anchor)
        F["nbr_span_rwy"] = _grouped_rolling_stat(dep_ar[real], dep_off[real], span[real], dep_ar, dep_off,
                                                  par.nbr_window, "median", par.rolling_anchor)
        F["nbr_taxiin_apt"] = _grouped_rolling_stat(arr_apt, arr_on, arr_taxiin, dep_apt, dep_off,
                                                    par.nbr_window, "median", par.rolling_anchor)
        F["nbr_taxiin_rwy"] = _grouped_rolling_stat(arr_ar, arr_on, arr_taxiin, dep_ar, dep_off,
                                                    par.nbr_window, "median", par.rolling_anchor)
    # extra family: queue2, a finer encoding of the queue than a count
    ka = _kth_after(E["dep_off_ar"], dep_ar, dep_push, KTH_AFTER)
    kb = _kth_before(E["dep_off_ar"], dep_ar, dep_off, KTH_BEFORE)
    kaa = _kth_after(E["dep_off_apt"], dep_apt, dep_push, KTH_AFTER)
    for k in KTH_AFTER:
        F[f"trwy_after{k}"] = ka[k] - dep_push
        F[f"tapt_after{k}"] = kaa[k] - dep_push
    for k in KTH_BEFORE:
        F[f"trwy_before{k}"] = dep_off - kb[k]
    F["push_demand_apt30"] = _count_between(E["dep_push_apt"], dep_apt, dep_push - 1800, dep_push)
    F["push_demand_rwy30"] = _count_between(_sorted_events(dep_ar, dep_push), dep_ar,
                                            dep_push - 1800, dep_push)
    F["push_rank_apt30"] = _count_between(E["dep_push_apt"], dep_apt, dep_push, dep_push + 1800)

    # family: stand occupancy. A stand holds one aircraft, so the first movement
    # that takes the stand at or after our push bounds our block time from
    # above: block <= that time, hence taxi >= take-off - that time.
    Est = {
        "arr_on_stand": _sorted_events(_cat(arr_apt, arr_stand), arr_on),
        "dep_off_stand": _sorted_events(dep_stand_key, dep_off),
        "dep_push_stand": _sorted_events(dep_stand_key, dep_push),
    }
    nxt_arr, prv_arr = _next_prev_at(Est["arr_on_stand"], dep_stand_key, dep_push)
    nxt_doff, prv_doff = _next_prev_at(Est["dep_off_stand"], dep_stand_key, dep_push, exclude=dep_off)
    nxt_dpush, _ = _next_prev_at(Est["dep_push_stand"], dep_stand_key, dep_push, exclude=dep_push)
    F["stand_nxt_arr_dt"] = nxt_arr - dep_push
    F["stand_prv_arr_dt"] = dep_push - prv_arr
    F["stand_nxt_dep_off_dt"] = nxt_doff - dep_push
    F["stand_prv_dep_off_dt"] = dep_push - prv_doff
    F["stand_nxt_dep_push_dt"] = nxt_dpush - dep_push
    occ = np.fmin(np.fmin(nxt_arr, nxt_dpush), nxt_doff)
    F["stand_ub_offset"] = occ - dep_push          # block - push <= this
    F["stand_lb_taxi"] = dep_off - occ             # taxi >= this
    F["stand_arr_in_span"] = _count_between(Est["arr_on_stand"], dep_stand_key, dep_push, dep_off)
    F["stand_mvt_2h"] = (_count_between(Est["arr_on_stand"], dep_stand_key, dep_off - 3600, dep_off + 3600)
                         + _count_between(Est["dep_off_stand"], dep_stand_key, dep_off - 3600, dep_off + 3600))
    F["stand_turnaround"] = dep_push - prv_arr     # our probable inbound on-block
    # the bound only bites when the stand is taken over during our taxi (11 % of
    # rows); clipped forms make that regime a single split for the tree
    F["stand_lb_taxi_pos"] = np.where(np.isfinite(F["stand_lb_taxi"]),
                                      np.maximum(F["stand_lb_taxi"], 0.0), np.nan)
    F["stand_ub_offset_clip"] = np.minimum(F["stand_ub_offset"], 7200.0)
    F["stand_bound_active"] = (F["stand_lb_taxi"].to_numpy() > 0).astype(np.int8)

    if sf is not None and sf.fitted:  # family: geometry
        uq, um = _unimpeded_lookup(sf, dep_apt, dep_rwy, dep_stand)
        F["unimpeded_q"] = uq
        F["stand_rwy_med"] = um
        real = np.isfinite(dep_aobt)
        excess = np.where(real, span - uq, np.nan)
        F["own_excess"] = excess
        F["nbr_excess_apt"] = _grouped_rolling_stat(dep_apt[real], dep_off[real], excess[real],
                                                    dep_apt, dep_off, par.nbr_window,
                                                    "median", par.rolling_anchor)
        F["nbr_excess_rwy"] = _grouped_rolling_stat(dep_ar[real], dep_off[real], excess[real],
                                                    dep_ar, dep_off, par.nbr_window,
                                                    "median", par.rolling_anchor)
    return F.copy()  # defragment: the frame was built column by column


def _unimpeded_lookup(sf: SurfaceFeatures, apt, rwy, stand):
    lv = sf.unimpeded["levels"]
    gq, gm = sf.unimpeded["glob_q"], sf.unimpeded["glob_m"]
    term = _terminal(stand)
    q = np.full(len(apt), np.nan)
    m = np.full(len(apt), np.nan)
    cols = {"apt": apt, "rwy": rwy, "stand": stand, "term": term}
    for name in LOOKUP_ORDER:
        keys = dict(LEVELS)[name]
        tbl = lv[name]
        if len(keys) > 1:
            idx = pd.MultiIndex.from_arrays([cols[k] for k in keys])
        else:
            idx = pd.Index(cols[keys[0]])
        hit = tbl.reindex(idx)
        need = ~np.isfinite(q)
        q = np.where(need, hit["sq"].to_numpy(), q)
        m = np.where(need, hit["sm"].to_numpy(), m)
    return np.where(np.isfinite(q), q, gq), np.where(np.isfinite(m), m, gm)


def _configuration(dep_apt, dep_rwy, dep_off, arr_apt, arr_rwy, arr_land,
                   bucket: float = BUCKET, now=CFG_NOW, prev=CFG_PREV) -> dict:
    """Runway configuration in the surrounding hour, as a stable name-set key.

    `cfg_changed` compares that hour's runway set with the hour before it.
    """
    pair = pd.DataFrame({"apt": np.concatenate([dep_apt, arr_apt]),
                         "rwy": np.concatenate([dep_rwy, arr_rwy])}).drop_duplicates()
    pair = pair.sort_values(["apt", "rwy"]).reset_index(drop=True)
    pair["bit"] = pair.groupby("apt").cumcount()
    pair = pair[pair["bit"] < 62]
    bitmap = {(a, r): int(b) for a, r, b in pair[["apt", "rwy", "bit"]].itertuples(index=False)}
    names = {}
    for a, r, b in pair[["apt", "rwy", "bit"]].itertuples(index=False):
        names.setdefault(a, {})[int(b)] = r

    def bucket_masks(apt, rwy, t):
        bit = np.array([bitmap.get((a, r), 62) for a, r in zip(apt, rwy)], dtype="int64")
        good = np.isfinite(t)
        df = pd.DataFrame({"apt": apt[good], "b": np.floor(t[good] / bucket).astype("int64"),
                           "m": (np.int64(1) << bit[good])})
        agg = df.groupby(["apt", "b"])["m"].agg(lambda s: int(np.bitwise_or.reduce(s.to_numpy())))
        return agg, df.groupby(["apt", "b"]).size()

    dmask, dcnt = bucket_masks(dep_apt, dep_rwy, dep_off)
    amask, acnt = bucket_masks(arr_apt, arr_rwy, arr_land)
    good = np.isfinite(dep_off)
    rcnt = pd.DataFrame({"apt": dep_apt[good], "rwy": dep_rwy[good],
                         "b": np.floor(dep_off[good] / bucket).astype("int64")}
                        ).groupby(["apt", "rwy", "b"]).size()
    b0 = np.floor(np.nan_to_num(dep_off, nan=0.0) / bucket).astype("int64")

    def gather_or(agg, offsets):
        acc = np.zeros(len(dep_apt), dtype="int64")
        for off in offsets:
            v = agg.reindex(pd.MultiIndex.from_arrays([dep_apt, b0 + off])).fillna(0).to_numpy().astype("int64")
            acc |= v
        return acc

    def gather_sum(agg, offsets, keys):
        acc = np.zeros(len(dep_apt), dtype="float64")
        for off in offsets:
            acc += agg.reindex(pd.MultiIndex.from_arrays(keys + [b0 + off])).fillna(0).to_numpy().astype("float64")
        return acc

    cur_d = gather_or(dmask, now)
    cur_a = gather_or(amask, now)
    prev_d = gather_or(dmask, prev)

    def decode(apt, mask):
        uniq = pd.unique(pd.Series(list(zip(apt, mask))))
        table = {}
        for a, m in uniq:
            nm = names.get(a, {})
            rs = [nm[b] for b in sorted(nm) if (m >> b) & 1]
            table[(a, m)] = a + "#" + "|".join(rs)
        return np.array([table[(a, m)] for a, m in zip(apt, mask)], dtype=object).astype(str)

    out = {
        "cfg_dep": decode(dep_apt, cur_d),
        "cfg_arr": decode(dep_apt, cur_a),
        "n_dep_rwy": _popcount(cur_d),
        "n_arr_rwy": _popcount(cur_a),
        "cfg_changed": ((prev_d != 0) & (prev_d != cur_d)).astype(np.int8),
    }
    tot = gather_sum(dcnt, now, [dep_apt])
    mine = gather_sum(rcnt, now, [dep_apt, dep_rwy])
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rwy_share"] = np.where(tot > 0, mine / tot, np.nan)
    return out
