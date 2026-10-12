"""exp-047-queue: the runway state at an instant, from other aircraft only.

exp-041 measured that the error v10 shares inside an airport x 10-minute window sits
on the RUNWAY (68.6 / 81.8 / 70.2 % of the shared component) and that the cell's own
observables reach under a tenth of it.  Its close-out names the revival: an observable
that is a runway STATE and is not already in the shipped block.

**What the shipped block already has, and what this file changes.**
`exp-018-fresh/code/features.py` builds `apt_queue`, `rwy_queue`, `apt_pushed_30m`,
`rwy_pushed_30m`, `rwy_share_1h`, `rwy_arrivals_1h` and the `w*_rwy_*` configuration
columns -- every one of them evaluated at the row's own **take-off** instant, and every
one of them built from NM's `AOBT_3_flt`.  This file changes two things and only two:

    the ANCHOR   the state is read at the row's OFF-BLOCK instant, where the aircraft
                 actually joins the queue, not at the instant it leaves it;
    the CLOCK    the events may come from the adsb.lol ground observations
                 (`exp-045-harvest` summaries) instead of, or beside, NM's off-block.

Three event sources are therefore built and priced separately, because a positive
reading on the observations is only worth a harvest if it beats the file:

    F  file      other departures' `AOBT_3_flt` -> `MVT_TIME_UTC_mvt`; arrivals'
                 `MVT_TIME_UTC_mvt` -> `BLOCK_TIME_UTC_mvt` (arrivals keep every
                 column in the ranking file).  Ten airports, every month, no harvest.
    O  observed  every received aircraft's `obs_offblock` -> `obs_wheels_off`, and
                 arrival ground runs `seg_start` -> `seg_end`.  Includes aircraft with
                 no NM row at all; excludes every aircraft the network did not hear.
    H  hybrid    the observed clock where the aircraft was heard, the file clock
                 otherwise -- the serving path.

**No column reads the row's own segment, its own future, or its label.**  Every count
is self-excluded: the row's own event is subtracted from the count it would otherwise
join.  No column is a function of `MVT_TIME_UTC_mvt - t_ref`, the row's own taxi span,
because for a matched row that span IS `exp-046-surfacefit`'s observed taxi and a
block that carried it would measure that lane's feature rather than the runway state.

The counting is a grouped `searchsorted`: events are sorted once per (group key, time)
and every window count is a difference of two insertion points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHIFT = 1 << 32          # composite (group, second) key; times are < 2**31 s


# ---------------------------------------------------------------- the engine

def _code(key: np.ndarray, t: np.ndarray) -> np.ndarray:
    """One int64 that sorts by group then by time, at 1-second resolution."""
    return key.astype("int64") * SHIFT + np.rint(np.asarray(t, "float64")).astype("int64")


def window_counts(src_key: np.ndarray, src_t: np.ndarray,
                  qry_key: np.ndarray, qry_t: np.ndarray,
                  wins: list[tuple[float, float]]) -> list[np.ndarray]:
    """For each query row, the number of source events in `[t+lo, t+hi)` in its group."""
    ok = np.isfinite(src_t)
    s = np.sort(_code(src_key[ok], src_t[ok]))
    qt = np.where(np.isfinite(qry_t), qry_t, 0.0)
    q = _code(qry_key, qt)
    out = []
    for lo, hi in wins:
        a = np.searchsorted(s, q + int(round(lo)), side="left")
        b = np.searchsorted(s, q + int(round(hi)), side="left")
        c = (b - a).astype("float64")
        c[~np.isfinite(qry_t)] = np.nan
        out.append(c)
    return out


def window_sums(src_key: np.ndarray, src_t: np.ndarray, src_v: np.ndarray,
                qry_key: np.ndarray, qry_t: np.ndarray,
                wins: list[tuple[float, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """The SUM and the COUNT of `src_v` over the source events in each window.

    Returned unreduced so the caller can subtract the query row's own event exactly
    before dividing; a mean cannot be self-excluded after the fact.
    """
    ok = np.isfinite(src_t) & np.isfinite(src_v)
    o = np.argsort(_code(src_key[ok], src_t[ok]), kind="stable")
    s = _code(src_key[ok], src_t[ok])[o]
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(src_v, "float64")[ok][o])])
    out = []
    qt = np.where(np.isfinite(qry_t), qry_t, 0.0)
    q = _code(qry_key, qt)
    for lo, hi in wins:
        a = np.searchsorted(s, q + int(round(lo)), side="left")
        b = np.searchsorted(s, q + int(round(hi)), side="left")
        tot = cum[b] - cum[a]
        n = (b - a).astype("float64")
        tot[~np.isfinite(qry_t)] = np.nan
        n[~np.isfinite(qry_t)] = np.nan
        out.append((tot, n))
    return out


def window_means(src_key: np.ndarray, src_t: np.ndarray, src_v: np.ndarray,
                 qry_key: np.ndarray, qry_t: np.ndarray,
                 wins: list[tuple[float, float]]) -> list[np.ndarray]:
    """The MEAN of `src_v` over the source events in each window (NaN when empty)."""
    ok = np.isfinite(src_t) & np.isfinite(src_v)
    o = np.argsort(_code(src_key[ok], src_t[ok]), kind="stable")
    s = _code(src_key[ok], src_t[ok])[o]
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(src_v, "float64")[ok][o])])
    out = []
    qt = np.where(np.isfinite(qry_t), qry_t, 0.0)
    for lo, hi in wins:
        a = np.searchsorted(s, _code(qry_key, qt) + int(round(lo)), side="left")
        b = np.searchsorted(s, _code(qry_key, qt) + int(round(hi)), side="left")
        n = (b - a).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (cum[b] - cum[a]) / n
        m[n <= 0] = np.nan
        m[~np.isfinite(qry_t)] = np.nan
        out.append(m)
    return out


def on_surface(src_key: np.ndarray, start: np.ndarray, end: np.ndarray,
               qry_key: np.ndarray, qry_t: np.ndarray) -> np.ndarray:
    """Events with `start <= t` and `end > t` in the query's group.

    `start <= t` uses `side="right"` so an event beginning exactly at `t` counts, and
    `end > t` uses `side="right"` on the ends so an event ending exactly at `t` does
    not.  A source event with a start and no end is treated as still on the surface.
    """
    #  An event whose end precedes its start is a clock inversion in the source
    #  (`MVT_TIME_UTC_mvt` before `AOBT_3_flt` happens: `docs/input-rule.md` puts the
    #  1st percentile of that difference at -1616 s).  Left alone it would subtract a
    #  phantom aircraft from every count between the two instants, which is how this
    #  function first returned counts of -1.  Such an event is treated as
    #  instantaneous for occupancy -- it enters and leaves at its own start -- and its
    #  real end is still what the `gone` windows count.
    fin_s = np.isfinite(start)
    end_s = np.where(np.isfinite(end), np.maximum(end, start), end)
    ss = np.sort(_code(src_key[fin_s], start[fin_s]))
    fin_e = np.isfinite(end_s) & fin_s
    se = np.sort(_code(src_key[fin_e], end_s[fin_e]))
    qt = np.where(np.isfinite(qry_t), qry_t, 0.0)
    q = _code(qry_key, qt)
    n = (np.searchsorted(ss, q, side="right")
         - np.searchsorted(se, q, side="right")).astype("float64")
    n[~np.isfinite(qry_t)] = np.nan
    return n


# ------------------------------------------------------------- the event set

class Events:
    """Departure and arrival surface events at ten airports, one month.

    `dep_start` is the off-block instant, `dep_end` the take-off, `dep_apt`/`dep_rwy`
    the group keys.  `arr_start` is the touchdown, `arr_end` the on-block.  Rows whose
    clock is missing carry NaN and are simply absent from every count.
    """

    def __init__(self, dep_apt, dep_rwy, dep_start, dep_end,
                 arr_apt, arr_rwy, arr_start, arr_end, arr_taxi, apt_codes):
        self.dep_apt = dep_apt.astype("int64")
        self.dep_rwy = dep_rwy.astype("int64")
        self.dep_start = np.asarray(dep_start, "float64")
        self.dep_end = np.asarray(dep_end, "float64")
        self.arr_apt = arr_apt.astype("int64")
        self.arr_rwy = arr_rwy.astype("int64")
        self.arr_start = np.asarray(arr_start, "float64")
        self.arr_end = np.asarray(arr_end, "float64")
        self.arr_taxi = np.asarray(arr_taxi, "float64")
        self.apt_codes = apt_codes
        #: the event's own duration -- for F the NM off-block to take-off difference,
        #: for O the observed ground run.  The mean of it over the departures that
        #: went from this runway in the last half hour is the REALISED taxi on this
        #: runway now, which the shipped block only has in its 466 s NM proxy form.
        with np.errstate(invalid="ignore"):
            self.dep_taxi = np.where(self.dep_end >= self.dep_start,
                                     self.dep_end - self.dep_start, np.nan)


#: the windows every source is read over, in seconds before the anchor
W10, W30, W60 = 600.0, 1800.0, 3600.0
COLS_COUNT = ["q_taxi_apt", "q_taxi_rwy", "q_taxi_oth", "q_pushed_apt_10m",
              "q_pushed_apt_30m", "q_pushed_rwy_30m", "q_gone_apt_10m",
              "q_gone_apt_30m", "q_gone_rwy_10m", "q_gone_rwy_30m",
              "a_land_apt_10m", "a_land_apt_30m", "a_land_rwy_30m", "a_taxiing_apt"]
COLS_RATE = ["q_ratio_rwy", "q_ratio_apt", "q_served_rwy_60m", "a_taxiin_30m",
             "s_taxi_apt_10m", "s_taxi_apt_30m", "s_taxi_rwy_30m", "s_taxi_rwy_60m",
             "s_taxi_rwy_rel", "s_taxi_n_rwy_30m",
             "cfg_n_rwy_30m", "cfg_rwy_share_30m", "cfg_rwy_active",
             "cfg_n_arr_rwy_30m", "cfg_arr_rwy_share_30m", "cfg_mixed",
             "cfg_changed_30m"]
COLS = COLS_COUNT + COLS_RATE


def state(ev: Events, apt: np.ndarray, rwy: np.ndarray, t: np.ndarray,
          self_dep: np.ndarray | None = None,
          prefix: str = "") -> pd.DataFrame:
    """The runway state at `t`, for query rows keyed by `(apt, rwy)`.

    `self_dep` is the index into `ev`'s departure arrays of the query row's OWN event,
    or -1 where the row has none.  Every count that would include it has it subtracted:
    that is the whole of the self-exclusion, and it is exact rather than approximate
    because the index is known.
    """
    n = len(apt)
    apt = apt.astype("int64")
    rwy = rwy.astype("int64")
    #  `_code` puts every instant on a 1-second grid, so the SELF-EXCLUSION has to
    #  compare on that same grid.  Comparing raw floats against a rounded engine makes
    #  a row that the engine has already counted as gone be subtracted a second time,
    #  and the count goes to -1.  It only bites where an instant is not a whole second,
    #  which is exactly the no-NM anchor `MVT_TIME - median(...)`: a median over an
    #  even-sized group is a half-second.  Found on the 2026-01 ranking census.
    t = np.rint(np.asarray(t, "float64"))
    out = {}

    # --- who is on the surface now
    out["q_taxi_apt"] = on_surface(ev.dep_apt, ev.dep_start, ev.dep_end, apt, t)
    out["q_taxi_rwy"] = on_surface(ev.dep_rwy, ev.dep_start, ev.dep_end, rwy, t)
    out["a_taxiing_apt"] = on_surface(ev.arr_apt, ev.arr_start, ev.arr_end, apt, t)

    # --- who pushed, and who went, in the last 10 / 30 / 60 minutes
    c = window_counts(ev.dep_apt, ev.dep_start, apt, t, [(-W10, 0.0), (-W30, 0.0)])
    out["q_pushed_apt_10m"], out["q_pushed_apt_30m"] = c
    out["q_pushed_rwy_30m"] = window_counts(ev.dep_rwy, ev.dep_start, rwy, t,
                                            [(-W30, 0.0)])[0]
    c = window_counts(ev.dep_apt, ev.dep_end, apt, t, [(-W10, 0.0), (-W30, 0.0)])
    out["q_gone_apt_10m"], out["q_gone_apt_30m"] = c
    c = window_counts(ev.dep_rwy, ev.dep_end, rwy, t,
                      [(-W10, 0.0), (-W30, 0.0), (-W60, 0.0)])
    out["q_gone_rwy_10m"], out["q_gone_rwy_30m"], served60 = c
    out["q_served_rwy_60m"] = served60

    # --- arrivals
    c = window_counts(ev.arr_apt, ev.arr_start, apt, t, [(-W10, 0.0), (-W30, 0.0)])
    out["a_land_apt_10m"], out["a_land_apt_30m"] = c
    out["a_land_rwy_30m"] = window_counts(ev.arr_rwy, ev.arr_start, rwy, t,
                                          [(-W30, 0.0)])[0]
    out["a_taxiin_30m"] = window_means(ev.arr_apt, ev.arr_end, ev.arr_taxi,
                                       apt, t, [(-W30, 0.0)])[0]

    # --- the realised taxi on this runway now, self-excluded below
    sums = {}
    for name, key, wins in (("apt", apt, [(-W10, 0.0), (-W30, 0.0)]),
                            ("rwy", rwy, [(-W30, 0.0), (-W60, 0.0)])):
        src = ev.dep_apt if name == "apt" else ev.dep_rwy
        sums[name] = window_sums(src, ev.dep_end, ev.dep_taxi, key, t, wins)

    # --- the self-exclusion, exact
    if self_dep is not None:
        has = self_dep >= 0
        idx = np.where(has, self_dep, 0)
        s0 = np.rint(ev.dep_start[idx])
        e0 = np.rint(ev.dep_end[idx])
        same_rwy = has & (ev.dep_rwy[idx] == rwy)
        on = has & np.isfinite(s0) & (s0 <= t) & (~np.isfinite(e0) | (e0 > t))
        out["q_taxi_apt"] -= on.astype("float64")
        out["q_taxi_rwy"] -= (on & same_rwy).astype("float64")
        for col, lo, key_same in (("q_pushed_apt_10m", -W10, True),
                                  ("q_pushed_apt_30m", -W30, True),
                                  ("q_pushed_rwy_30m", -W30, False)):
            hit = has & np.isfinite(s0) & (s0 >= t + lo) & (s0 < t)
            if not key_same:
                hit &= same_rwy
            out[col] -= hit.astype("float64")
        for col, lo, key_same in (("q_gone_apt_10m", -W10, True),
                                  ("q_gone_apt_30m", -W30, True),
                                  ("q_gone_rwy_10m", -W10, False),
                                  ("q_gone_rwy_30m", -W30, False),
                                  ("q_served_rwy_60m", -W60, False)):
            hit = has & np.isfinite(e0) & (e0 >= t + lo) & (e0 < t)
            if not key_same:
                hit &= same_rwy
            out[col] -= hit.astype("float64")

        # the realised-taxi means lose the row's own completed run where it falls in
        # the window; a mean cannot be self-excluded after it is divided, which is why
        # `window_sums` returns the sum and the count
        for name, wins in (("apt", (-W10, -W30)), ("rwy", (-W30, -W60))):
            for j, lo in enumerate(wins):
                tot, cnt = sums[name][j]
                hit = has & np.isfinite(e0) & (e0 >= t + lo) & (e0 < t)
                if name == "rwy":
                    hit &= same_rwy
                own = np.where(hit & np.isfinite(ev.dep_taxi[idx]),
                               np.nan_to_num(ev.dep_taxi[idx]), 0.0)
                sums[name][j] = (tot - own,
                                 cnt - (hit & np.isfinite(ev.dep_taxi[idx])
                                        ).astype("float64"))

    with np.errstate(invalid="ignore", divide="ignore"):
        for key, name, j in (("s_taxi_apt_10m", "apt", 0), ("s_taxi_apt_30m", "apt", 1),
                             ("s_taxi_rwy_30m", "rwy", 0), ("s_taxi_rwy_60m", "rwy", 1)):
            tot, cnt = sums[name][j]
            v = np.where(cnt > 0, tot / np.maximum(cnt, 1e-9), np.nan)
            out[key] = v
        out["s_taxi_n_rwy_30m"] = sums["rwy"][0][1]
        med = pd.Series(out["s_taxi_apt_30m"]).groupby(apt).transform("median").to_numpy()
        out["s_taxi_rwy_rel"] = out["s_taxi_rwy_30m"] / np.where(med > 0, med, np.nan)

    out["q_taxi_oth"] = out["q_taxi_apt"] - out["q_taxi_rwy"]
    for c in COLS_COUNT:
        v = out[c]
        bad = np.nanmin(v) if np.isfinite(v).any() else 0.0
        if bad < 0:
            k = int(np.nansum(v < 0))
            raise SystemExit(f"state(): {c} went negative ({bad}) on {k} of {n} rows; "
                             "a count cannot. The self-exclusion and the engine "
                             "disagree about an instant.")

    # --- the queue over the service rate: Little's law, and not a function of the
    #     row's own span
    with np.errstate(invalid="ignore", divide="ignore"):
        out["q_ratio_rwy"] = out["q_taxi_rwy"] / np.maximum(out["q_gone_rwy_30m"], 1.0)
        out["q_ratio_apt"] = out["q_taxi_apt"] / np.maximum(out["q_gone_apt_30m"], 1.0)

    # --- the configuration in force
    rwy_of_apt = ev.apt_codes["rwy_of_apt"]
    dep_rwy_30 = _distinct_and_share(ev.dep_apt, ev.dep_rwy, ev.dep_end, apt, rwy, t,
                                     -W30, 0.0, rwy_of_apt)
    out["cfg_n_rwy_30m"], out["cfg_rwy_share_30m"], out["cfg_rwy_active"] = dep_rwy_30
    arr_rwy_30 = _distinct_and_share(ev.arr_apt, ev.arr_rwy, ev.arr_start, apt, rwy, t,
                                     -W30, 0.0, rwy_of_apt)
    out["cfg_n_arr_rwy_30m"], out["cfg_arr_rwy_share_30m"], arr_active = arr_rwy_30
    out["cfg_mixed"] = np.minimum(out["cfg_rwy_active"], arr_active)
    prev = _distinct_and_share(ev.dep_apt, ev.dep_rwy, ev.dep_end, apt, rwy, t,
                               -2 * W30, -W30, rwy_of_apt)
    out["cfg_changed_30m"] = ((np.abs(out["cfg_n_rwy_30m"] - prev[0]) > 0
                               ).astype("float64")
                              + np.abs(out["cfg_rwy_share_30m"] - prev[1]))

    f = pd.DataFrame({f"{prefix}{k}": v for k, v in out.items()})
    if len(f) != n:
        raise SystemExit("state(): row count changed")
    return f[[f"{prefix}{c}" for c in COLS]]


def _distinct_and_share(src_apt, src_rwy, src_t, apt, rwy, t, lo, hi, rwy_of_apt):
    """Distinct runways used in the window, this runway's share, and whether this
    runway was used at all.

    `rwy_of_apt[a]` lists the runway codes of airport `a`, padded with -1; the ten
    airports carry at most six each, so the loop is short and every step is one
    grouped `searchsorted`.
    """
    tot = window_counts(src_apt, src_t, apt, t, [(lo, hi)])[0]
    mine = window_counts(src_rwy, src_t, rwy, t, [(lo, hi)])[0]
    ndist = np.zeros(len(apt), "float64")
    for j in range(rwy_of_apt.shape[1]):
        col = rwy_of_apt[apt, j]
        ok = col >= 0
        if not ok.any():
            continue
        cj = np.zeros(len(apt), "float64")
        cj[ok] = window_counts(src_rwy, src_t, col[ok], t[ok], [(lo, hi)])[0]
        ndist += (cj > 0).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = mine / np.maximum(tot, 1.0)
    bad = ~np.isfinite(t)
    ndist[bad] = np.nan
    share[bad] = np.nan
    return ndist, share, np.where(bad, np.nan, (mine > 0).astype("float64"))


# ------------------------------------------------- the neighbourhood estimate

NCOLS = ["n_res_cell", "n_res_rwy_cell", "n_res_apt_30m", "n_res_rwy_30m",
         "n_res_rwy_60m", "n_anom_cell", "n_anom_rwy_cell", "n_anom_rwy_30m",
         "n_n_cell", "n_n_rwy_cell", "n_n_rwy_30m", "n_res_rwy_minus_apt"]


def neighbourhood(apt, rwy, cell, t_take, val_res, val_anom, self_ok,
                  prefix: str = "") -> pd.DataFrame:
    """What the OTHER departures nearby say about this cell's common error.

    `val_res[j] = observed taxi of j - v11's prediction for j` and `val_anom[j] =
    observed taxi of j - the airport-hour median observed taxi`.  Neither reads a
    label: the observed taxi is `MVT_TIME_UTC_mvt - obs_offblock`, a ranking column
    minus an adsb.lol observation, and both are available for the ranking months.

    exp-041 measured that the residual v11 SHARES inside an airport x 10-minute window
    is 8,157 s^2 on july and that it sits on the runway.  The self-excluded mean of
    `val_res` over that same window is a direct, oracle-free estimator of it: under
    `r_j = mu_c + eps_j` and `obs_taxi_j = y_j + delta_j`, the mean over the other rows
    of the cell estimates `mu_c` with noise `(sd(eps) + sd(delta)) / sqrt(n)`.  With
    NM's off-block, `sd(delta)` is 350-466 s (`docs/input-rule.md`, exp-043) and the
    estimator is swamped; with the observed off-block the pilot measured 143 s.  That
    difference -- and nothing else -- is what the file arm and the observed arm of this
    experiment separate.

    Every mean is self-excluded exactly: row `i` never enters its own neighbourhood.
    `self_ok` says whether row `i`'s own value is finite and so was in the sum at all.
    """
    out = {}
    n = len(apt)
    rwy_cell = np.char.add(np.char.add(np.asarray(rwy).astype(str), "|"),
                           np.asarray(cell).astype(str))
    for tag, key in (("cell", np.asarray(cell).astype(str)),
                     ("rwy_cell", rwy_cell)):
        for what, v in (("res", val_res), ("anom", val_anom)):
            code, uniq = pd.factorize(key)
            k = int(len(uniq))
            ok = np.isfinite(v) & self_ok
            s = np.bincount(code, weights=np.where(ok, v, 0.0), minlength=k)
            c = np.bincount(code, weights=ok.astype("float64"), minlength=k)
            num = s[code] - np.where(ok, v, 0.0)
            den = c[code] - ok.astype("float64")
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"n_{what}_{tag}"] = np.where(den > 0, num / np.maximum(den, 1e-9),
                                                  np.nan)
            if what == "res":
                # the count is "how many OTHER aircraft in this cell were HEARD", not
                # "how many have a finite residual": on a ranking month the residual
                # form needs a prediction vector and the count must not vanish with it
                okc = self_ok
                cs = np.bincount(code, weights=okc.astype("float64"), minlength=k)
                out[f"n_n_{tag}"] = cs[code] - okc.astype("float64")
    # symmetric time windows, which the retrospective task allows: every other
    # movement's actual timings are inputs (`docs/input-rule.md`), and the shipped
    # block already reads forward windows (`apt_to_15m_after`)
    for tag, key, wins in (("apt_30m", apt, [(-W30, W30)]),
                           ("rwy_30m", rwy, [(-W30, W30)]),
                           ("rwy_60m", rwy, [(-W60, W60)])):
        okv = np.isfinite(val_res) & self_ok
        tot, cnt = window_sums(key[okv], t_take[okv], val_res[okv], key, t_take,
                               wins)[0]
        own = np.where(okv, val_res, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"n_res_{tag}"] = np.where(cnt - okv > 0,
                                           (tot - own) / np.maximum(cnt - okv, 1e-9),
                                           np.nan)
        if tag == "rwy_30m":
            _, hc = window_sums(key[self_ok], t_take[self_ok],
                                np.ones(int(self_ok.sum())), key, t_take, wins)[0]
            out["n_n_rwy_30m"] = hc - self_ok.astype("float64")
            okv2 = np.isfinite(val_anom) & self_ok
            t2, c2 = window_sums(key[okv2], t_take[okv2], val_anom[okv2], key, t_take,
                                 wins)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                out["n_anom_rwy_30m"] = np.where(
                    c2 - okv2 > 0, (t2 - np.where(okv2, val_anom, 0.0))
                    / np.maximum(c2 - okv2, 1e-9), np.nan)
    out["n_res_rwy_minus_apt"] = out["n_res_rwy_30m"] - out["n_res_apt_30m"]
    f = pd.DataFrame({f"{prefix}{k}": v for k, v in out.items()})
    if len(f) != n:
        raise SystemExit("neighbourhood(): row count changed")
    return f[[f"{prefix}{c}" for c in NCOLS]]
