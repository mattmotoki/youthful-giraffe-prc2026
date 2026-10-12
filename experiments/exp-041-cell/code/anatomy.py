"""Step 1 -- the anatomy of v10's shared within-cell error, label side.

Q04 measured `mean(r_i * rbar_i)` with `rbar_i` the SELF-EXCLUDED cell mean of
the other departures' residuals: under `r_i = mu_c + eps_i` that is an unbiased
estimate of `Var(mu_c)`.  This file takes that one number apart.  Every headline
number is built the same way -- a ratio of a pair sum to a pair count, restricted
to a relation -- because a plug-in cell mean carries `Var(eps)/n_c` and the
cross-product does not.

`assignments/cell.md` names three statistics in words: the cell-mean residual's
VARIANCE, the AUTOCORRELATION of the cell mean at lags 1-6, and the cross-fitted
R^2 of the cell mean.  Sol round 1, blocking finding 1: naming a better estimator
is not the same as running the one the assignment asked for.  All three are now
computed and reported, each beside the unbiased reading and with the bias term
that separates them -- `plugin()`, `acf()` and `observables`' two R^2 columns.

    T1  cohort and coverage
    T2  the headline, by cell-size cut, and Q04's exact support replicated
    T3  by airport, with the queue's coverage rule checked per airport
    T4  by hour, by cell size, by regime (each regime on its own cells)
    T5  persistence: the same statistic at lags 0..6 bins, same airport
    T6  the runway x operator crossover -- the four relations and the variance
        components they identify
    T7  what cell-level OBSERVABLES already reach: a cross-fitted cell model
        priced on the contract (2A - B), and the shared component that survives it

    python anatomy.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B  # noqa: E402

sys.path.insert(0, str(B.S39))
import q04 as Q4      # noqa: E402  (arrival loader only; no nuisance model here)
import rowfeat as R   # noqa: E402

EXP = B.EXP
FOLDS = B.FOLDS
D = B.D

# ---- predeclared -------------------------------------------------------------
SIZE_CUTS = (2, 4, 6, 10)          # minimum eligible departures in the cell
LAGS = (0, 1, 2, 3, 4, 5, 6)
HOUR_BANDS = ((0, 6), (6, 10), (10, 14), (14, 18), (18, 24))
SIZE_STRATA = ((2, 3), (4, 5), (6, 9), (10, 19), (20, 10_000))
PAIRS_PER_RELATION = 200           # queue Q10's coverage rule, per airport
DATES_PER_RELATION = 20
WIDE_BINS = 4                      # the wide window, in 10-minute bins
TREES, DEPTH, LR = 300, 3, 0.05


# ---- the relation estimator --------------------------------------------------
def relation(r: np.ndarray, other_sum: np.ndarray, other_cnt: np.ndarray,
             days: np.ndarray, rng, keep: np.ndarray | None = None) -> dict:
    """`sum_i r_i * (sum of the relation's residuals) / sum_i (its count)`.

    Unbiased for `E[r_i r_j]` over the pairs the relation names; each unordered
    pair enters twice, once from each end.  The interval is a day-clustered
    bootstrap of the ratio (both rows of a pair share a cell and so a day).
    """
    if keep is None:
        keep = np.ones(len(r), bool)
    num = np.where(keep, r * other_sum, 0.0)
    den = np.where(keep, other_cnt, 0.0)
    n_pairs = float(den.sum()) / 2.0
    if den.sum() <= 0:
        return {"pairs": 0, "value": None, "ci": [None, None], "rows": 0, "dates": 0}
    pt, lo, hi = B.boot_ratio(num, den, days, rng)
    d = np.unique(days[keep & (other_cnt > 0)])
    return {"pairs": int(round(n_pairs)), "value": round(pt, 1),
            "ci": [round(lo, 1), round(hi, 1)],
            "rows": int((keep & (other_cnt > 0)).sum()), "dates": int(len(d))}


def self_excluded(code: np.ndarray, ncode: int, r: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of the OTHER rows in the same code, per row."""
    s, c = B.group_sums(code, ncode, r)
    return s - r, c - 1.0


def plugin(code: np.ndarray, ncode: int, r: np.ndarray, days: np.ndarray, rng,
           keep: np.ndarray | None = None) -> dict:
    """The assignment's own statistic: the VARIANCE OF THE PLUG-IN CELL MEAN.

    `assignments/cell.md` step 1 names "the cell-mean residual's variance".  A
    plug-in cell mean carries the noise of its own rows, so this number is
    `Var(mu_c) + E[Var(eps)/n_c]` and not the shared component; it is reported
    here beside its own bias estimate and the debiased difference, so the record
    carries the named statistic AND the reason the cross-product is the one the
    verdict rests on.  The two agree when the bias is removed, which is the check.

    Row-weighted, so it is on the same footing as the cross-product: every
    eligible row contributes once, through its own cell.
    """
    if keep is None:
        keep = np.ones(len(r), bool)
    s, c = B.group_sums(code, ncode, r)
    n2 = np.maximum(c, 1.0)
    bar = s / n2
    two = keep & (c >= 2)
    if two.sum() < 50:
        return {"rows": int(two.sum()), "plugin_var_s2": None,
                "noise_term_s2": None, "debiased_s2": None}
    w = two.astype("float64")
    mu = float((bar * w).sum() / w.sum())
    var_num = w * (bar - mu) ** 2
    pt, lo, hi = B.boot_ratio(var_num, w, days, rng)
    # within-cell sample variance, per cell, gathered back on the rows
    ss, _ = B.group_sums(code, ncode, r * r)
    s2 = np.maximum((ss - c * bar ** 2) / np.maximum(c - 1.0, 1.0), 0.0)
    noise = float((w * s2 / n2).sum() / w.sum())
    return {"rows": int(two.sum()), "plugin_var_s2": round(pt, 1),
            "ci": [round(lo, 1), round(hi, 1)],
            "noise_term_s2": round(noise, 1),
            "debiased_s2": round(pt - noise, 1),
            "note": "Var(plug-in cell mean) = Var(mu_c) + E[Var(eps)/n_c]; the "
                    "third number is the first minus the second and is what the "
                    "self-excluded cross-product estimates directly"}


def cell_mean_acf(apt: np.ndarray, bins: np.ndarray, r: np.ndarray,
                  lags=LAGS) -> dict:
    """The AUTOCORRELATION of the cell mean, on MATCHED cell pairs.

    Sol round 2, blocking finding 1: the first version divided a pair-weighted
    lag-L cross-product by a differently weighted lag-0 value.  Those are two
    ratios over two populations, so their quotient is not a correlation.  This
    version is the named statistic: for each lag L, take the cells (a, t) and
    (a, t + L) that are BOTH non-empty, and correlate their plug-in means over
    that one population, centred on that population's own mean.

    Two readings, because the plug-in mean carries `Var(eps)/n_c`:
      `rho_plugin`   Pearson correlation of the plug-in means -- the assignment's
                     statistic, attenuated by the noise in each mean;
      `rho_debiased` the same covariance over the DEBIASED variance of the mean
                     on the same cells, which is the correlation of `mu_c`
                     itself.  Noise is independent across bins, so the numerator
                     needs no correction and only the denominator changes.
    """
    df = pd.DataFrame({"apt": apt, "bin": bins, "r": r})
    g = df.groupby(["apt", "bin"])["r"]
    bar = g.mean()
    cnt = g.size().astype("float64")
    # E[Var(eps)/n_c] per cell, for the debiased denominator
    var_c = g.var(ddof=1).fillna(0.0)
    noise_c = (var_c / cnt).rename("noise")
    tab = pd.DataFrame({"bar": bar, "n": cnt, "noise": noise_c}).reset_index()
    out = {}
    for L in lags:
        right = tab.copy()
        right["bin"] = right["bin"] - L
        j = tab.merge(right, on=["apt", "bin"], suffixes=("_a", "_b"))
        if L == 0:
            j = tab.assign(bar_a=tab["bar"], bar_b=tab["bar"],
                           noise_a=tab["noise"], noise_b=tab["noise"],
                           n_a=tab["n"], n_b=tab["n"])
        if len(j) < 50:
            out[f"lag_{L}"] = {"cell_pairs": int(len(j)), "rho_plugin": None,
                               "rho_debiased": None}
            continue
        a = j["bar_a"].to_numpy("float64")
        b = j["bar_b"].to_numpy("float64")
        na_ = j["n_a"].to_numpy("float64") if "n_a" in j else j["n"].to_numpy("float64")
        nb_ = j["n_b"].to_numpy("float64") if "n_b" in j else j["n"].to_numpy("float64")
        rows = {}
        for wname, w in (("per_cell", np.ones(len(a))), ("per_pair", na_ * nb_)):
            sw = w.sum()
            mu = float((w * a).sum() + (w * b).sum()) / (2 * sw)
            cov = float((w * (a - mu) * (b - mu)).sum() / sw)
            va = float((w * (a - mu) ** 2).sum() / sw)
            vb = float((w * (b - mu) ** 2).sum() / sw)
            nsa = float((w * j["noise_a"].to_numpy("float64")).sum() / sw)
            nsb = float((w * j["noise_b"].to_numpy("float64")).sum() / sw)
            den = np.sqrt(max(va, 1e-12) * max(vb, 1e-12))
            dend = np.sqrt(max(va - nsa, 1e-12) * max(vb - nsb, 1e-12))
            cov_d = (va - nsa) if L == 0 else cov
            rows[wname] = {
                "rho_plugin": round(cov / den, 4),
                "rho_debiased": round(min(cov_d / dend, 9.99), 4),
                "cov_s2": round(cov, 1),
                "var_plugin_s2": round(va, 1),
                "var_debiased_s2": round(va - nsa, 1)}
        out[f"lag_{L}"] = {"cell_pairs": int(len(j)), **rows["per_pair"],
                           "per_cell": rows["per_cell"]}
    out["note"] = ("matched cell pairs at each lag.  The headline numbers weight "
                   "each cell pair by n_a * n_b -- the number of row pairs it "
                   "carries, the same weight the cross-product table uses -- and "
                   "`per_cell` repeats them with one weight per cell.  "
                   "`rho_plugin` is the assignment's statistic on the plug-in "
                   "mean; `rho_debiased` removes the within-cell noise from the "
                   "denominator only, because noise in two different bins is "
                   "independent")
    return out


# ---- the queue's predeclared Q10 cell rule -----------------------------------
MIN_INTERSECTION = 3    # QUEUE.md Q10: "at least three rows per intersection"


def q10_subgrid(ccode: np.ndarray, rwy: np.ndarray, ope: np.ndarray,
                min_rows: int = MIN_INTERSECTION) -> tuple[np.ndarray, dict]:
    """`QUEUE.md` Q10's predeclared support, applied BEFORE any covariance.

    Sol round 2, blocking finding 2: the queue predeclares "two operators across
    two runways, at least three rows per intersection", and asks for the
    unsupported cells to be reported.  The first version kept a row when all four
    relations were non-empty, which is a weaker, row-level condition.

    A cell is supported when some pair of runways and some pair of operators have
    at least `min_rows` rows in each of the four intersections; the chosen 2x2 is
    the one with the largest total count, ties broken by sorted labels.  The mask
    returned is the rows INSIDE that 2x2, so the four relations are matched by
    construction and not only non-empty.
    """
    from itertools import combinations
    df = pd.DataFrame({"c": ccode, "r": rwy, "o": ope})
    cnt = (df.groupby(["c", "r", "o"]).size().rename("n").reset_index())
    cnt = cnt[cnt["n"] >= min_rows]
    keep = np.zeros(len(ccode), bool)
    n_cells_tested = int(pd.unique(ccode).size)
    n_supported = 0
    pick_r: dict[int, tuple] = {}
    pick_o: dict[int, tuple] = {}
    for c, d in cnt.groupby("c", sort=False):
        by_op: dict[object, dict] = {}
        for o, dd in d.groupby("o", sort=True):
            by_op[o] = dict(zip(dd["r"], dd["n"]))
        ops = sorted(by_op)
        if len(ops) < 2:
            continue
        best = None
        for o1, o2 in combinations(ops, 2):
            shared = sorted(set(by_op[o1]) & set(by_op[o2]))
            if len(shared) < 2:
                continue
            for r1, r2 in combinations(shared, 2):
                tot = (by_op[o1][r1] + by_op[o1][r2]
                       + by_op[o2][r1] + by_op[o2][r2])
                cand = (-tot, str(o1), str(o2), str(r1), str(r2))
                if best is None or cand < best[0]:
                    best = (cand, (r1, r2), (o1, o2))
        if best is None:
            continue
        n_supported += 1
        pick_r[c] = best[1]
        pick_o[c] = best[2]
    if pick_r:
        want_r = pd.Series({c: v for c, v in pick_r.items()})
        want_o = pd.Series({c: v for c, v in pick_o.items()})
        rr = want_r.reindex(ccode)
        oo = want_o.reindex(ccode)
        keep = np.array([(isinstance(a, tuple) and rw in a
                          and isinstance(b, tuple) and op in b)
                         for a, b, rw, op in zip(rr, oo, rwy, ope)], bool)
    info = {"min_rows_per_intersection": min_rows,
            "cells": n_cells_tested,
            "cells_supported": n_supported,
            "cells_unsupported": n_cells_tested - n_supported,
            "cells_supported_pct": round(100 * n_supported / max(n_cells_tested, 1), 2),
            "rows_in_the_chosen_2x2": int(keep.sum()),
            "rows_pct_of_eligible": round(100 * float(keep.mean()), 2),
            "note": "the queue's rule is on the CELL and is applied before any "
                    "covariance; the population read is the chosen 2x2 subgrid, "
                    "so every relation has at least three rows at each corner"}
    return keep, info


# ---- arrivals, counts only ---------------------------------------------------
def arrival_cells(month: str) -> pd.DataFrame:
    """Per airport x bin: arrival count and mean taxi-in.  Inputs, not labels:
    `TAXITIME_SEC_mvt` on an ARR row is the taxi-IN time and the ranking mask
    does not blank it (`docs/input-rule.md`)."""
    a = Q4.arrivals(month)
    b = np.floor(a["t"].to_numpy() / B.BIN).astype("int64")
    df = pd.DataFrame({"apt": a["apt"].to_numpy(), "bin": b,
                       "taxi_in": a["taxi_in"].to_numpy(),
                       "late": a["t"].to_numpy() - a["sched"].to_numpy()})
    g = df.groupby(["apt", "bin"]).agg(n_arr=("taxi_in", "size"),
                                       arr_taxi=("taxi_in", "mean"),
                                       arr_late=("late", "mean"))
    return g


def wide(series: pd.Series, counts: pd.Series, bins: int) -> pd.Series:
    """Count-weighted rolling mean over `bins` CONSECUTIVE bins on a full grid."""
    out = []
    for apt, d in series.groupby(level=0):
        idx = d.index.get_level_values(1)
        grid = pd.RangeIndex(int(idx.min()), int(idx.max()) + 1)
        v = d.reset_index(level=0, drop=True).reindex(grid).fillna(0.0)
        n = counts.xs(apt, level=0).reindex(grid).fillna(0.0)
        num = (v * n).rolling(bins, min_periods=1).sum()
        den = n.rolling(bins, min_periods=1).sum()
        w = (num / den.replace(0, np.nan))
        w.index = pd.MultiIndex.from_product([[apt], grid], names=["apt", "bin"])
        out.append(w)
    return pd.concat(out)


def wide_count(counts: pd.Series, bins: int) -> pd.Series:
    out = []
    for apt, d in counts.groupby(level=0):
        idx = d.index.get_level_values(1)
        grid = pd.RangeIndex(int(idx.min()), int(idx.max()) + 1)
        n = d.reset_index(level=0, drop=True).reindex(grid).fillna(0.0)
        w = n.rolling(bins, min_periods=1).sum()
        w.index = pd.MultiIndex.from_product([[apt], grid], names=["apt", "bin"])
        out.append(w)
    return pd.concat(out)


def run_fold(fold: str, pan: pd.DataFrame, rng) -> dict:
    month = R.month_of(fold)
    m = B.panel(fold, pan)
    r = m["r"].to_numpy("float64")
    days = m["day"].to_numpy()
    apt = m["apt"].to_numpy()
    bins = m["bin"].to_numpy()
    elig = m["elig"].to_numpy()
    ei = np.flatnonzero(elig)

    # cells over the ELIGIBLE cohort only: the relation is between rows the
    # screen could ever update together
    ccode, ncell = B.codes_of(apt[ei], bins[ei])
    n_in = np.bincount(ccode, minlength=ncell).astype("float64")
    re = r[ei]
    de = days[ei]
    os_all, oc_all = self_excluded(ccode, ncell, re)
    size = n_in[ccode]

    out = {"fold": fold, "month": month, "n_rows": int(len(m)),
           "n_eligible": int(elig.sum()), "n_cells": int(ncell),
           "median_cell_size": float(np.median(size)),
           "cell_size_p90": float(np.percentile(size, 90))}

    # ---- T2 headline by cell-size cut --------------------------------------
    out["headline"] = {}
    for k in SIZE_CUTS:
        keep = size >= k
        out["headline"][f"cell_ge_{k}"] = relation(re, os_all, oc_all, de, rng, keep)

    # Q04's own support: >= 6 eligible departures AND >= 6 arrivals in the cell
    ac = arrival_cells(month)
    key_e = pd.MultiIndex.from_arrays([apt[ei], bins[ei]], names=["apt", "bin"])
    n_arr = ac["n_arr"].reindex(key_e).fillna(0).to_numpy()
    keep_q04 = (size >= 6) & (n_arr >= 6)
    out["headline"]["q04_support_dep6_arr6"] = relation(re, os_all, oc_all, de,
                                                        rng, keep_q04)
    #  the assignment's named statistic, beside the estimator this file uses
    out["plugin_cell_mean"] = {f"cell_ge_{k}": plugin(ccode, ncell, re, de, rng,
                                                      size >= k)
                               for k in SIZE_CUTS}
    out["headline"]["q04_support_note"] = (
        "exp-039 analysis/q04.json cross_product_r_rbar_self_excluded on the same "
        "support; assignments/cell.md quotes a different triple (see RESULTS 1)")

    # ---- T3 by airport, with the coverage rule ------------------------------
    out["by_airport"] = {}
    out["by_airport_plugin"] = {}
    for a in sorted(set(apt[ei])):
        keep = (apt[ei] == a) & (size >= 2)
        out["by_airport"][a] = relation(re, os_all, oc_all, de, rng, keep)
        out["by_airport_plugin"][a] = plugin(ccode, ncell, re, de, rng, keep)

    # ---- T4 by hour band, cell size, regime ---------------------------------
    hour = np.mod(m["t"].to_numpy("float64"), 86400) / 3600.0
    out["by_hour"] = {}
    for lo, hi in HOUR_BANDS:
        keep = (hour[ei] >= lo) & (hour[ei] < hi) & (size >= 2)
        out["by_hour"][f"{lo:02d}-{hi:02d}"] = relation(re, os_all, oc_all, de,
                                                        rng, keep)
        out.setdefault("by_hour_plugin", {})[f"{lo:02d}-{hi:02d}"] = plugin(
            ccode, ncell, re, de, rng, keep)
    out["by_cell_size"] = {}
    out["by_cell_size_plugin"] = {}
    for lo, hi in SIZE_STRATA:
        keep = (size >= lo) & (size <= hi)
        nm = f"{lo}-{hi if hi < 10000 else '+'}"
        out["by_cell_size"][nm] = relation(re, os_all, oc_all, de, rng, keep)
        out["by_cell_size_plugin"][nm] = plugin(ccode, ncell, re, de, rng, keep)

    # every regime on its OWN cells, including the two held airports as a row
    out["by_regime"] = {}
    for name, sel in (("nm_bulk (eligible)", elig),
                      ("nm_bulk at EHAM/EDDF (held)",
                       (m["regime"].to_numpy() == "nm_bulk")
                       & np.isin(apt, B.HOLD_AIRPORTS)),
                      ("nm_late", m["regime"].to_numpy() == "nm_late"),
                      ("nonm", m["regime"].to_numpy() == "nonm"),
                      ("noac", m["regime"].to_numpy() == "noac")):
        j = np.flatnonzero(sel)
        if len(j) < 50:
            out["by_regime"][name] = {"pairs": 0, "value": None,
                                      "ci": [None, None], "rows": int(len(j)),
                                      "dates": 0}
            continue
        cc, nc = B.codes_of(apt[j], bins[j])
        os_, oc_ = self_excluded(cc, nc, r[j])
        out["by_regime"][name] = relation(r[j], os_, oc_, days[j], rng)
        out.setdefault("by_regime_plugin", {})[name] = plugin(
            cc, nc, r[j], days[j], rng)

    # ---- T5 persistence across adjacent bins, same airport ------------------
    #  lag L: pairs (i, j) with apt(i) = apt(j) and bin(j) = bin(i) + L.
    gg = pd.Series(re).groupby([apt[ei], bins[ei]])
    sums = gg.sum()
    cnts = gg.size().astype("float64")     # ROWS per cell, not one per cell
    out["persistence"] = {}
    for L in LAGS:
        if L == 0:
            out["persistence"]["lag_0"] = relation(re, os_all, oc_all, de, rng)
            continue
        k = pd.MultiIndex.from_arrays([apt[ei], bins[ei] + L])
        s = sums.reindex(k).fillna(0.0).to_numpy()
        c = cnts.reindex(k).fillna(0.0).to_numpy()
        out["persistence"][f"lag_{L}"] = relation(re, s, c, de, rng)
    out["autocorrelation"] = cell_mean_acf(apt[ei], bins[ei], re)

    # ---- T6 the runway x operator crossover ---------------------------------
    rwy = m["rwy"].to_numpy()[ei]
    ope = m["op"].to_numpy()[ei]
    cR, nR_ = B.codes_of(apt[ei], bins[ei], rwy)
    cO, nO_ = B.codes_of(apt[ei], bins[ei], ope)
    cRO, nRO_ = B.codes_of(apt[ei], bins[ei], rwy, ope)
    S, N = B.group_sums(ccode, ncell, re)
    SR, NR = B.group_sums(cR, nR_, re)
    SO, NO = B.group_sums(cO, nO_, re)
    SRO, NRO = B.group_sums(cRO, nRO_, re)
    rel = {
        "same_rwy_same_op": (SRO - re, NRO - 1.0),
        "same_rwy_diff_op": (SR - SRO, NR - NRO),
        "diff_rwy_same_op": (SO - SRO, NO - NRO),
        "diff_rwy_diff_op": (S - SR - SO + SRO, N - NR - NO + NRO),
    }
    #  MATCHED POPULATION (Sol round 1, blocking finding 2).  On the whole
    #  eligible cohort the four relations are NOT estimated on the same cells: a
    #  single-runway cell contributes only same-runway pairs and a single-operator
    #  cell only same-operator pairs, so a difference between two relations mixes
    #  a difference between relations with a difference between cell populations.
    #  The matched population is the rows for which ALL FOUR relations are
    #  non-empty -- every such row contributes to every relation -- and it is what
    #  the variance components and the coverage rule are read on.  The unmatched
    #  reading stays in the record beside it.
    all_four = np.ones(len(re), bool)
    for _, (_s, _c) in rel.items():
        all_four &= _c > 0
    n_rwy = pd.Series(rwy).groupby(ccode).nunique().reindex(
        range(ncell)).to_numpy()[ccode]
    n_ope = pd.Series(ope).groupby(ccode).nunique().reindex(
        range(ncell)).to_numpy()[ccode]
    cell_rule = (n_rwy >= 2) & (n_ope >= 2)
    out["crossover_population"] = {
        "eligible_rows": int(len(re)),
        "rows_in_cells_with_2_runways_and_2_operators": int(cell_rule.sum()),
        "rows_with_all_four_relations_non_empty": int(all_four.sum()),
        "matched_pct_of_eligible": round(100 * float(all_four.mean()), 2),
        "cells_touched_by_matched_rows": int(len(np.unique(ccode[all_four]))),
        "unmatched_rows_by_reason": {
            "single_runway_cell": int((n_rwy < 2).sum()),
            "single_operator_cell": int((n_ope < 2).sum()),
            "2x2_cell_but_a_relation_empty":
                int((cell_rule & ~all_four).sum())},
        "note": "the matched rule is row-level: a row is kept when all four "
                "relations have at least one counterpart, so every relation is "
                "estimated on the same rows and the same days"}

    #  the queue's PREDECLARED cell rule (Sol round 2, blocking finding 2).
    #  At the 10-minute bin the rule is UNSATISFIABLE -- a cell would need four
    #  runway x operator corners of three rows each, and the largest corner in
    #  july is nine rows in a cell of at most seventeen -- so the rule is scanned
    #  over wider windows as well and the supported grain is the one the
    #  components are read on.  "Report unsupported cells" is the queue's own
    #  instruction and zero support is the honest answer at 10 minutes.
    q10_keep, q10_info = q10_subgrid(ccode, rwy, ope)
    out["crossover_population"]["q10_predeclared_rule"] = q10_info
    scan = {}
    for k in (1, 3, 6, 9):
        if k == 1:
            kk, info_k = q10_keep, dict(q10_info)
        else:
            cc_k, nc_k = B.codes_of(apt[ei], bins[ei] // k)
            kk, info_k = q10_subgrid(cc_k, rwy, ope)
        info_k["window_minutes"] = 10 * k
        scan[f"{10 * k}min"] = info_k
        if kk.sum() > 50 and k > 1:
            sgk = np.flatnonzero(kk)
            bk = bins[ei] // k
            c1, n1 = B.codes_of(apt[ei][sgk], bk[sgk])
            c2, n2_ = B.codes_of(apt[ei][sgk], bk[sgk], rwy[sgk])
            c3, n3_ = B.codes_of(apt[ei][sgk], bk[sgk], ope[sgk])
            c4, n4_ = B.codes_of(apt[ei][sgk], bk[sgk], rwy[sgk], ope[sgk])
            rk = re[sgk]
            S1, N1 = B.group_sums(c1, n1, rk)
            S2, N2 = B.group_sums(c2, n2_, rk)
            S3, N3 = B.group_sums(c3, n3_, rk)
            S4, N4 = B.group_sums(c4, n4_, rk)
            relk = {"same_rwy_same_op": (S4 - rk, N4 - 1.0),
                    "same_rwy_diff_op": (S2 - S4, N2 - N4),
                    "diff_rwy_same_op": (S3 - S4, N3 - N4),
                    "diff_rwy_diff_op": (S1 - S2 - S3 + S4, N1 - N2 - N3 + N4)}
            rr_ = {kn: relation(rk, s, c, de[sgk], rng) for kn, (s, c) in relk.items()}
            bv = rr_["diff_rwy_diff_op"]["value"]
            os_k, oc_k = self_excluded(c1, n1, rk)
            sh = relation(rk, os_k, oc_k, de[sgk], rng)["value"]
            comp = None
            if bv is not None:
                comp = {"airport_window_s2": bv,
                        "runway_given_window_s2": round(
                            rr_["same_rwy_diff_op"]["value"] - bv, 1),
                        "operator_given_window_s2": round(
                            rr_["diff_rwy_same_op"]["value"] - bv, 1),
                        "additivity_residual_s2": round(
                            rr_["same_rwy_same_op"]["value"]
                            - rr_["same_rwy_diff_op"]["value"]
                            - rr_["diff_rwy_same_op"]["value"] + bv, 1),
                        "shared_component_on_these_rows_s2": sh}
                tt = sum(v for kn, v in comp.items()
                         if kn.endswith("_s2") and kn != "shared_component_on_these_rows_s2")
                comp["sum_of_components_s2"] = round(tt, 1)
                comp["fractions_of_the_sum"] = {
                    "airport_shared_pct": round(100 * comp["airport_window_s2"] / tt, 1),
                    "runway_shared_pct": round(100 * comp["runway_given_window_s2"] / tt, 1),
                    "operator_shared_pct": round(100 * comp["operator_given_window_s2"] / tt, 1),
                    "additivity_pct": round(100 * comp["additivity_residual_s2"] / tt, 1)}
                if sh:
                    comp["fractions_of_the_shared_component"] = {
                        "denominator_s2": sh,
                        "airport_shared_pct": round(100 * comp["airport_window_s2"] / sh, 1),
                        "runway_shared_pct": round(100 * comp["runway_given_window_s2"] / sh, 1),
                        "operator_shared_pct": round(100 * comp["operator_given_window_s2"] / sh, 1),
                        "additivity_pct": round(100 * comp["additivity_residual_s2"] / sh, 1)}
            cov_k = {}
            for a in sorted(set(apt[ei][sgk])):
                ka = apt[ei][sgk] == a
                cov_k[a] = {}
                for kn, (s, c) in relk.items():
                    pr = float(np.where(ka, c, 0.0).sum()) / 2.0
                    dts = int(len(np.unique(de[sgk][ka & (c > 0)])))
                    cov_k[a][kn] = {"pairs": int(round(pr)), "dates": dts,
                                    "met": bool(pr >= PAIRS_PER_RELATION
                                                and dts >= DATES_PER_RELATION)}
            scan[f"{10 * k}min"]["relations"] = rr_
            scan[f"{10 * k}min"]["components"] = comp
            scan[f"{10 * k}min"]["coverage"] = cov_k
            scan[f"{10 * k}min"]["airports_supported"] = sorted(
                a for a, v in cov_k.items() if all(x["met"] for x in v.values()))
    out["q10_rule_grain_scan"] = scan
    #  on the Q10 subgrid the relations must be recomputed inside the subgrid, or
    #  a "different runway" counterpart could be a third runway the rule never
    #  supported.
    sg = np.flatnonzero(q10_keep)
    rel_q10 = {}
    if len(sg) > 50:
        c_s, n_s = B.codes_of(apt[ei][sg], bins[ei][sg])
        cR_s, nR_s = B.codes_of(apt[ei][sg], bins[ei][sg], rwy[sg])
        cO_s, nO_s = B.codes_of(apt[ei][sg], bins[ei][sg], ope[sg])
        cRO_s, nRO_s = B.codes_of(apt[ei][sg], bins[ei][sg], rwy[sg], ope[sg])
        rs = re[sg]
        S_s, N_s = B.group_sums(c_s, n_s, rs)
        SR_s, NR_s = B.group_sums(cR_s, nR_s, rs)
        SO_s, NO_s = B.group_sums(cO_s, nO_s, rs)
        SRO_s, NRO_s = B.group_sums(cRO_s, nRO_s, rs)
        rel_q10 = {
            "same_rwy_same_op": (SRO_s - rs, NRO_s - 1.0),
            "same_rwy_diff_op": (SR_s - SRO_s, NR_s - NRO_s),
            "diff_rwy_same_op": (SO_s - SRO_s, NO_s - NRO_s),
            "diff_rwy_diff_op": (S_s - SR_s - SO_s + SRO_s,
                                 N_s - NR_s - NO_s + NRO_s),
        }
        os_sg, oc_sg = self_excluded(c_s, n_s, rs)
        out["crossover_q10"] = {k: relation(rs, s, c, de[sg], rng)
                                for k, (s, c) in rel_q10.items()}
        bv = out["crossover_q10"]["diff_rwy_diff_op"]["value"]
        if bv is not None:
            vq = {"airport_window_s2": bv,
                  "runway_given_window_s2": round(
                      out["crossover_q10"]["same_rwy_diff_op"]["value"] - bv, 1),
                  "operator_given_window_s2": round(
                      out["crossover_q10"]["diff_rwy_same_op"]["value"] - bv, 1),
                  "additivity_residual_s2": round(
                      out["crossover_q10"]["same_rwy_same_op"]["value"]
                      - out["crossover_q10"]["same_rwy_diff_op"]["value"]
                      - out["crossover_q10"]["diff_rwy_same_op"]["value"] + bv, 1)}
            sh = relation(rs, os_sg, oc_sg, de[sg], rng)["value"]
            tt = sum(v for k, v in vq.items() if k.endswith("_s2"))
            vq["shared_component_on_these_rows_s2"] = sh
            vq["sum_of_components_s2"] = round(tt, 1)
            vq["fractions_of_the_sum"] = {
                "airport_shared_pct": round(100 * vq["airport_window_s2"] / tt, 1),
                "runway_shared_pct": round(100 * vq["runway_given_window_s2"] / tt, 1),
                "operator_shared_pct": round(100 * vq["operator_given_window_s2"] / tt, 1),
                "additivity_pct": round(100 * vq["additivity_residual_s2"] / tt, 1)}
            if sh:
                vq["fractions_of_the_shared_component"] = {
                    "denominator_s2": sh,
                    "airport_shared_pct": round(100 * vq["airport_window_s2"] / sh, 1),
                    "runway_shared_pct": round(100 * vq["runway_given_window_s2"] / sh, 1),
                    "operator_shared_pct": round(100 * vq["operator_given_window_s2"] / sh, 1),
                    "additivity_pct": round(100 * vq["additivity_residual_s2"] / sh, 1),
                    "note": "the assignment's own denominator: the shared "
                            "component read on the SAME rows.  The four shares do "
                            "not add to 100 % -- the components are conditional "
                            "variances that sum to the same-runway-same-operator "
                            "relation, while the shared component is the "
                            "pair-weighted average over all four relations"}
            out["variance_components_q10"] = vq
        cov_q = {}
        for a in sorted(set(apt[ei][sg])):
            ka = apt[ei][sg] == a
            cov_q[a] = {}
            for k, (s, c) in rel_q10.items():
                pr = float(np.where(ka, c, 0.0).sum()) / 2.0
                dts = int(len(np.unique(de[sg][ka & (c > 0)])))
                cov_q[a][k] = {"pairs": int(round(pr)), "dates": dts,
                               "met": bool(pr >= PAIRS_PER_RELATION
                                           and dts >= DATES_PER_RELATION)}
        out["crossover_q10_coverage"] = cov_q
        out["crossover_q10_airports_supported"] = sorted(
            a for a, v in cov_q.items() if all(x["met"] for x in v.values()))

    for tag, keep_ in (("crossover", None), ("crossover_matched", all_four)):
        out[tag] = {k: relation(re, s, c, de, rng, keep_)
                    for k, (s, c) in rel.items()}
        base_v = out[tag]["diff_rwy_diff_op"]["value"]
        if base_v is None:
            continue
        vc = {
            "airport_window_s2": base_v,
            "runway_given_window_s2": round(
                out[tag]["same_rwy_diff_op"]["value"] - base_v, 1),
            "operator_given_window_s2": round(
                out[tag]["diff_rwy_same_op"]["value"] - base_v, 1),
            "additivity_residual_s2": round(
                out[tag]["same_rwy_same_op"]["value"]
                - out[tag]["same_rwy_diff_op"]["value"]
                - out[tag]["diff_rwy_same_op"]["value"] + base_v, 1),
            "note": "cov(diff,diff) = Var(window); each same-X row minus it is "
                    "Var(X | window); the fourth cell is the additivity check, "
                    "not an assumption"}
        #  Sol round 2, blocking finding 1: this comment used to claim the
        #  denominator was the shared component while the code divided by the sum
        #  of the components.  Both denominators are now reported and named.  The
        #  assignment asks for the fraction of the SHARED COMPONENT, which is the
        #  second block; the first is the share of the four components' sum, which
        #  is what a variance decomposition adds up to.
        shared_here = relation(re, os_all, oc_all, de, rng, keep_)["value"]
        tot = (vc["airport_window_s2"] + vc["runway_given_window_s2"]
               + vc["operator_given_window_s2"] + vc["additivity_residual_s2"])
        vc["shared_component_on_these_rows_s2"] = shared_here
        vc["sum_of_components_s2"] = round(tot, 1)
        if tot:
            vc["fractions_of_the_sum"] = {
                "airport_shared_pct": round(100 * vc["airport_window_s2"] / tot, 1),
                "runway_shared_pct": round(
                    100 * vc["runway_given_window_s2"] / tot, 1),
                "operator_shared_pct": round(
                    100 * vc["operator_given_window_s2"] / tot, 1),
                "additivity_pct": round(
                    100 * vc["additivity_residual_s2"] / tot, 1),
                "denominator_s2": round(tot, 1)}
        if shared_here:
            vc["fractions_of_the_shared_component"] = {
                "denominator_s2": shared_here,
                "airport_shared_pct": round(
                    100 * vc["airport_window_s2"] / shared_here, 1),
                "runway_shared_pct": round(
                    100 * vc["runway_given_window_s2"] / shared_here, 1),
                "operator_shared_pct": round(
                    100 * vc["operator_given_window_s2"] / shared_here, 1),
                "additivity_pct": round(
                    100 * vc["additivity_residual_s2"] / shared_here, 1),
                "note": "the assignment's own denominator: the shared component "
                        "read on the SAME rows.  These four do not add to 100 % "
                        "-- the components are conditional variances whose sum is "
                        "the same-runway-same-operator relation, while the shared "
                        "component is the pair-weighted average over all four"}
        out["variance_components" if tag == "crossover"
            else "variance_components_matched"] = vc

    # the coverage rule, per airport and relation, on the MATCHED rows
    cov = {}
    for a in sorted(set(apt[ei])):
        ka = (apt[ei] == a) & all_four
        cov[a] = {}
        for k, (s, c) in rel.items():
            pr = float(np.where(ka, c, 0.0).sum()) / 2.0
            dts = int(len(np.unique(de[ka & (c > 0)])))
            cov[a][k] = {"pairs": int(round(pr)), "dates": dts,
                         "met": bool(pr >= PAIRS_PER_RELATION
                                     and dts >= DATES_PER_RELATION)}
    out["crossover_coverage"] = cov
    out["crossover_coverage_note"] = (
        "counted on the matched rows only (all four relations non-empty), so an "
        "airport that meets the rule meets it on the population the components "
        "are read on")
    out["crossover_airports_supported"] = sorted(
        a for a, v in cov.items() if all(x["met"] for x in v.values()))
    out["crossover_matched_supported_airports"] = {
        k: relation(re, s, c, de, rng,
                    all_four & np.isin(apt[ei], out["crossover_airports_supported"]))
        for k, (s, c) in rel.items()}

    # ---- T7 what cell-level observables already reach -----------------------
    #  The row set aggregated to the cell is the SHIPPED current feature block
    #  (exp-039's Sol round 2, blocking finding b1: a matched base for these
    #  screens is the shipped block, not a hand-built substitute) TOGETHER with
    #  exp-039's 202-column row feature set, so the upper bound this table
    #  reports is the strongest one available at this cost.
    F = R.features(R.departures(month))
    ids_e = m["MVT_ID_mvt"].to_numpy("float64")[ei]
    Xr = F.reindex(ids_e).to_numpy("float64")
    fcols = list(F.columns)
    del F
    Xs, scols = B.shipped(month, ids_e)
    Xrow = np.column_stack([Xr, np.nan_to_num(Xs, nan=0.0)])
    fcols = fcols + scols
    del Xr, Xs
    out["observables"] = observables(m, ei, ccode, ncell, re, de, ac, rng,
                                     Xrow, fcols)

    # ---- T8 the same two readings at the RUNWAY grain -----------------------
    #  T6 says the shared component sits on the runway inside the window.  A
    #  correction that is constant inside an airport x bin cell cannot reach a
    #  runway-within-window component by construction, so T7's small number is
    #  not yet a statement about reachability.  This block repeats the shared
    #  statistic and the observables model on airport x bin x RUNWAY cells,
    #  where the runway component IS inside the cell.
    ccode2, ncell2 = B.codes_of(apt[ei], bins[ei], rwy)
    n2 = np.bincount(ccode2, minlength=ncell2).astype("float64")[ccode2]
    os2, oc2 = self_excluded(ccode2, ncell2, re)
    rg = {"n_cells": int(ncell2), "median_cell_size": float(np.median(n2)),
          "shared": {}}
    for k in SIZE_CUTS:
        rg["shared"][f"cell_ge_{k}"] = relation(re, os2, oc2, de, rng, n2 >= k)
    gg2 = pd.Series(re).groupby(ccode2)
    sums2 = gg2.sum()
    cnts2 = gg2.size().astype("float64")   # ROWS per runway-cell
    rg["persistence"] = {}
    bin_of = pd.Series(bins[ei]).groupby(ccode2).first().reindex(range(ncell2))
    rwy_of = pd.Series(rwy).groupby(ccode2).first().reindex(range(ncell2))
    apt_of = pd.Series(apt[ei]).groupby(ccode2).first().reindex(range(ncell2))
    lookup = pd.Series(np.arange(ncell2), index=pd.MultiIndex.from_arrays(
        [apt_of.to_numpy(), bin_of.to_numpy(), rwy_of.to_numpy()]))
    for L in LAGS:
        if L == 0:
            rg["persistence"]["lag_0"] = relation(re, os2, oc2, de, rng)
            continue
        k = pd.MultiIndex.from_arrays([apt[ei], bins[ei] + L, rwy])
        j = lookup.reindex(k).to_numpy()
        ok = np.isfinite(j)
        s_ = np.zeros(len(re)); c_ = np.zeros(len(re))
        ji = j[ok].astype("int64")
        s_[ok] = sums2.reindex(ji).fillna(0.0).to_numpy()
        c_[ok] = cnts2.reindex(ji).fillna(0.0).to_numpy()
        rg["persistence"][f"lag_{L}"] = relation(re, s_, c_, de, rng)
    rg["autocorrelation"] = cell_mean_acf(
        np.char.add(np.char.add(apt[ei].astype(str), "|"), rwy.astype(str)),
        bins[ei], re)
    rg["plugin_cell_mean"] = {f"cell_ge_{k}": plugin(ccode2, ncell2, re, de, rng,
                                                     n2 >= k)
                              for k in SIZE_CUTS}
    rg["observables"] = observables(m, ei, ccode2, ncell2, re, de, ac, rng,
                                    Xrow, fcols, extra_key=rwy)
    out["runway_grain"] = rg
    return out


def observables(m, ei, ccode, ncell, re, de, ac, rng, X, cols,
                extra_key: np.ndarray | None = None) -> dict:
    """A cross-fitted model on CELL-LEVEL observables only, priced on the contract.

    The features are constant inside a cell, so the fitted `h` IS a servable
    cell correction and `2A - B` on held-out day blocks is exactly what it would
    be worth.  This is also the repair clause's oracle-free upper bound: what
    survives in the shared cross-product after `h` is removed is the part of the
    shared component no cell-level observable in this set reaches.
    """
    apt = m["apt"].to_numpy()[ei]
    bins = m["bin"].to_numpy()[ei]
    t = m["t"].to_numpy("float64")[ei]

    # cell aggregates of the shipped row feature set
    agg = np.zeros((ncell, X.shape[1]))
    cnt = np.bincount(ccode, minlength=ncell).astype("float64")
    for j in range(X.shape[1]):
        agg[:, j] = np.bincount(ccode, weights=X[:, j], minlength=ncell) / cnt

    # the count / window block, on the same cells
    key = pd.MultiIndex.from_arrays([apt, bins], names=["apt", "bin"])
    fr = {"apt": apt, "bin": bins, "code": ccode}
    if extra_key is not None:
        fr["extra"] = extra_key
    first = pd.DataFrame(fr).drop_duplicates("code").sort_values("code")
    ck = pd.MultiIndex.from_arrays([first["apt"].to_numpy(),
                                    first["bin"].to_numpy()], names=["apt", "bin"])
    dep_cnt = pd.Series(1.0, index=key).groupby(level=[0, 1]).sum()
    n_arr = ac["n_arr"].reindex(ck).fillna(0.0).to_numpy()
    arr_taxi = ac["arr_taxi"].reindex(ck).fillna(-1.0).to_numpy()
    arr_late = ac["arr_late"].reindex(ck).fillna(-1e5).to_numpy()
    dep_wide = wide_count(dep_cnt, WIDE_BINS).reindex(ck).fillna(0.0).to_numpy()
    arr_wide = wide_count(ac["n_arr"], WIDE_BINS).reindex(ck).fillna(0.0).to_numpy()
    taxi_wide = wide(ac["arr_taxi"], ac["n_arr"], WIDE_BINS
                     ).reindex(ck).fillna(-1.0).to_numpy()
    tc = np.bincount(ccode, weights=t, minlength=ncell) / cnt
    n_win = dep_cnt.reindex(ck).fillna(0.0).to_numpy()   # departures in the WINDOW
    extra = [cnt, n_win, n_arr, dep_wide, arr_wide, arr_taxi, arr_late, taxi_wide,
             np.mod(tc, 86400) / 3600.0, np.floor(tc / 86400) % 7,
             pd.factorize(first["apt"].to_numpy())[0].astype("float64")]
    names = cols + ["n_dep_cell", "n_dep_window", "n_arr_cell", "n_dep_wide",
                    "n_arr_wide", "arr_taxi_cell", "arr_late_cell",
                    "arr_taxi_wide", "hour", "dow", "airport"]
    if extra_key is not None:
        extra.append(pd.factorize(first["extra"].to_numpy())[0].astype("float64"))
        names = names + ["cell_runway"]
    C = np.column_stack(extra)
    C = np.column_stack([agg, C])

    # cross-fit by contiguous day blocks
    cell_day = pd.Series(de).groupby(ccode).first().reindex(range(ncell)).to_numpy()
    blk_cell = B.blocks_of(cell_day)
    ybar = np.bincount(ccode, weights=re, minlength=ncell) / cnt
    h_cell = np.zeros(ncell)
    for b in range(B.BLOCKS):
        fit = blk_cell != b
        g = HistGradientBoostingRegressor(max_iter=TREES, max_depth=DEPTH,
                                          learning_rate=LR, random_state=B.SEED)
        g.fit(C[fit], ybar[fit], sample_weight=cnt[fit])
        h_cell[~fit] = g.predict(C[~fit])
    h = h_cell[ccode]

    #  the assignment's repair clause names the CROSS-FITTED R^2 OF THE CELL MEAN
    #  on all cell-level observables at once.  `h_cell` is exactly that
    #  prediction, out of block, so the R^2 is read here on the cells, weighted
    #  by cell size -- the weight the fit used and the weight that makes it
    #  comparable to the row-level statistics.
    #
    #  Two versions, because the plug-in cell mean is not the shared component:
    #  the plain R^2 has `Var(mu_c) + E[Var(eps)/n_c]` in its denominator and no
    #  model can reach the second term, so it understates the model; the shared
    #  R^2 removes that noise term from both sides and is the assignment's
    #  oracle-free upper bound on what the cell's own observables reach.
    two_c = cnt >= 2
    wc = np.where(two_c, cnt, 0.0)
    ssq = np.bincount(ccode, weights=re * re, minlength=ncell)
    s2c = np.maximum((ssq - cnt * ybar ** 2) / np.maximum(cnt - 1.0, 1.0), 0.0)
    mu_w = float((ybar * wc).sum() / wc.sum())
    m2_tot = float((wc * (ybar - mu_w) ** 2).sum() / wc.sum())
    m2_res = float((wc * (ybar - h_cell) ** 2).sum() / wc.sum())
    noise = float((wc * s2c / np.maximum(cnt, 1.0)).sum() / wc.sum())
    r2_cell = 1.0 - m2_res / m2_tot if m2_tot > 0 else None
    shared_before_pl = m2_tot - noise
    shared_after_pl = m2_res - noise
    r2_shared = (None if shared_before_pl <= 0 else
                 1.0 - shared_after_pl / shared_before_pl)
    A = float((re * h).mean()); Bq = float((h ** 2).mean())
    v = 2 * re * h - h ** 2
    pt, lo, hi = B.boot_ratio(v, np.ones(len(v)), de, rng)
    os_all, oc_all = self_excluded(ccode, ncell, re)
    rr = re - h
    os_r, oc_r = self_excluded(ccode, ncell, rr)
    before = relation(re, os_all, oc_all, de, rng)
    after = relation(rr, os_r, oc_r, de, rng)
    frac = (None if not before["value"] else
            round(1.0 - after["value"] / before["value"], 4))
    return {"n_cells": int(ncell), "n_features": int(C.shape[1]),
            "cross_fitted_r2_cell_mean": (None if r2_cell is None
                                          else round(r2_cell, 4)),
            "cross_fitted_r2_shared_part": (None if r2_shared is None
                                            else round(r2_shared, 4)),
            "cross_fitted_r2_note":
                "the first is the count-weighted out-of-block R^2 of the PLUG-IN "
                "cell mean, whose denominator carries the within-cell noise "
                "E[Var(eps)/n_c] that no model can reach; the second removes "
                "that noise from both sides and is the assignment's oracle-free "
                "upper bound",
            "plugin_second_moment_s2": round(m2_tot, 1),
            "plugin_noise_term_s2": round(noise, 1),
            "plugin_debiased_s2": round(shared_before_pl, 1),
            "plugin_debiased_after_s2": round(shared_after_pl, 1),
            "feature_note": "cell means of the exp-039 row feature set "
                            f"({len(cols)} columns) plus the count/window block; "
                            "NOT the literal v10 member frame, which is not "
                            "reproducible at this cost",
            "A_mean_r_h": round(A, 3), "B_mean_h2": round(Bq, 3),
            "mse_gain_s2": round(pt, 2), "ci": [round(lo, 2), round(hi, 2)],
            "unrestricted_scale": round(A / Bq, 4) if Bq > 0 else None,
            "shared_before": before, "shared_after": after,
            "fraction_of_shared_explained": frac,
            "top_features": [names[j] for j in
                             np.argsort(-np.abs(np.corrcoef(
                                 np.column_stack([C, ybar]), rowvar=False)[-1, :-1]
                                 ))[:8]]}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(B.SEED)
    pan = B.V.load()
    recs = {}
    for f in FOLDS:
        recs[f] = run_fold(f, pan[f], rng)
        c = recs[f]
        print(f"[{f}] {c['n_eligible']:,} eligible in {c['n_cells']:,} cells, "
              f"median size {c['median_cell_size']}", flush=True)
        for k, v in c["headline"].items():
            if isinstance(v, dict):
                print(f"    shared {k:24s} {v['value']} {v['ci']} "
                      f"on {v['pairs']:,} pairs")
        pl = c["plugin_cell_mean"]["cell_ge_2"]
        print(f"    plug-in Var(cell mean) {pl['plugin_var_s2']} "
              f"= shared {pl['debiased_s2']} + noise {pl['noise_term_s2']}")
        print("    acf(cell mean): " + "; ".join(
            f"L{k.split('_')[1]} {v['rho_plugin']}/{v['rho_debiased']}"
            for k, v in c["autocorrelation"].items() if k.startswith("lag_")))
        cp = c["crossover_population"]
        print(f"    matched rows (all four relations non-empty) "
              f"{cp['rows_with_all_four_relations_non_empty']:,} of "
              f"{cp['eligible_rows']:,} ({cp['matched_pct_of_eligible']} %)")
        vm = c.get("variance_components_matched", {})
        if vm:
            print(f"    MATCHED components: window {vm['airport_window_s2']}, "
                  f"runway {vm['runway_given_window_s2']}, operator "
                  f"{vm['operator_given_window_s2']}, additivity "
                  f"{vm['additivity_residual_s2']}; shares "
                  f"{vm.get('fractions_of_the_sum')}")
        print("    crossover: " + "; ".join(
            f"{k} {v['value']} {v['ci']} ({v['pairs']:,} pairs)"
            for k, v in c["crossover"].items()), flush=True)
        if "variance_components" in c:
            vc = c["variance_components"]
            print(f"    components: window {vc['airport_window_s2']}, "
                  f"runway {vc['runway_given_window_s2']}, "
                  f"operator {vc['operator_given_window_s2']}, "
                  f"additivity {vc['additivity_residual_s2']}")
        o = c["observables"]
        print(f"    observables (apt x bin): 2A-B {o['mse_gain_s2']} {o['ci']} "
              f"s^2; shared {o['shared_before']['value']} -> "
              f"{o['shared_after']['value']} "
              f"({o['fraction_of_shared_explained']}); cross-fitted R2 "
              f"{o['cross_fitted_r2_cell_mean']} (plug-in) / "
              f"{o['cross_fitted_r2_shared_part']} (shared part)", flush=True)
        rg = c["runway_grain"]
        o2 = rg["observables"]
        print(f"    RUNWAY grain: {rg['n_cells']:,} cells, median size "
              f"{rg['median_cell_size']}; shared "
              f"{rg['shared']['cell_ge_2']['value']} "
              f"{rg['shared']['cell_ge_2']['ci']}; observables 2A-B "
              f"{o2['mse_gain_s2']} {o2['ci']} s^2; shared "
              f"{o2['shared_before']['value']} -> {o2['shared_after']['value']} "
              f"({o2['fraction_of_shared_explained']})", flush=True)
        print("    persistence (runway grain): " + "; ".join(
            f"L{k.split('_')[1]} {v['value']}"
            for k, v in rg["persistence"].items()), flush=True)
    out = {"screen": "anatomy",
           "predeclared": {"bin_s": B.BIN, "size_cuts": list(SIZE_CUTS),
                           "lags": list(LAGS), "blocks": B.BLOCKS,
                           "seed": B.SEED, "trees": TREES, "depth": DEPTH,
                           "lr": LR, "pairs_per_relation": PAIRS_PER_RELATION,
                           "dates_per_relation": DATES_PER_RELATION,
                           "eligible_regime": B.ELIGIBLE_REGIME,
                           "hold_airports": list(B.HOLD_AIRPORTS)},
           "folds": recs, "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "anatomy.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
