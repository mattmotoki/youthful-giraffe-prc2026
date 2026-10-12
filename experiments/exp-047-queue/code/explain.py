"""exp-047-queue step 1: does the runway-state block explain the shared component?

exp-041 §0 asks exactly one question of any candidate for the revival it names:

    cross-fitted by day, does the block explain the error v11 SHARES inside an
    airport x 10-minute window?

The estimator is exp-041's, imported and not rewritten: the self-excluded
cross-product `sum_i r_i * (sum of the other rows' residuals in the cell) / sum_i
(their count)`, which is unbiased for `E[r_i r_j]` and carries no `Var(eps)/n_c`
term.  The block's fitted correction `h` is subtracted from the residual and the same
statistic is read again; the share removed is `1 - after / before`.

`assignments/queue.md`: "the block earns step 2 if it reaches a third on any airport
with a day-clustered interval clear of a tenth".  The interval is therefore on the
SHARE, not on either end of it, so the bootstrap resamples days once per draw and
rebuilds both ends of the ratio from the same resampled days.

Arms, all on the same rows and the same cross fit, so their difference is the thing
each one changes and nothing else:

    Fto   the file's clocks read at the TAKE-OFF instant  -- the control: this is
          where the shipped block already reads them, so a gain here would mean the
          learner, not the anchor
    Fob   the file's clocks read at the OFF-BLOCK instant -- the new anchor, no harvest
    Oob   the observed clocks at the observed off-block   -- the harvest's own
    Hob   observed where heard, file otherwise            -- the serving path
    F     Fto + Fob together
    all   every block column present

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/explain.py --fold july_loo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
import state as S                                            # noqa: E402
import pin as PIN                                            # noqa: E402
#: a RECORDED re-price sets this (or passes `admit_population=True`) and moves the pins;
#: every other run is fail-closed against the population its fold was priced on
ADMIT_POPULATION = bool(os.environ.get("QUEUE_ADMIT_POPULATION"))
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

ARMS = {"Fto": ("Fto_",), "Fob": ("Fob_",), "F": ("Fto_", "Fob_"),
        "Oob": ("Oob_",), "Hob": ("Hob_",), "geo": ("route_", "stand_dist"),
        "Nf": ("Nf_",), "No": ("No_",), "N": ("Nf_", "No_"),
        "own": ("own_",), "own_No": ("own_", "No_"),
        "own_No_Hob": ("own_", "No_", "Hob_"),
        "F_Nf": ("Fto_", "Fob_", "Nf_"),
        # what this lane SERVES on the no-clock class (desk, 2026-09-08 16:25 UTC):
        # the file-clock neighbourhood and the geometry, both label-free at serving
        # time and both standing under --permute-file.
        # THE COMBINED ARM (desk, 2026-09-09).  Arm `N` plus RESULTS §7ab's first two
        # design changes: `Xr_*` the cell's regulation signature with the within-operator
        # share beside it, `Xn_*` the network state at the other nine airports.  Both are
        # built from the file alone (`block.regnet_block`), so the arm is servable on a
        # ranking month; §7ab's LABEL network read is a bound, not a column.
        "reg": ("Xr_",), "net": ("Xn_",), "regnet": ("Xr_", "Xn_"),
        "N_reg": ("Nf_", "No_", "Xr_"), "N_net": ("Nf_", "No_", "Xn_"),
        "N_regnet": ("Nf_", "No_", "Xr_", "Xn_"),
        "Nf_geo": ("Nf_", "route_", "stand_dist"),
        "N_geo": ("Nf_", "No_", "route_", "stand_dist"),
        "Hob_geo": ("Hob_", "route_", "stand_dist"),
        "all": ("Fto_", "Fob_", "Oob_", "Hob_", "Nf_", "No_", "route_",
                "stand_dist"),
        "all_own": ("Fto_", "Fob_", "Oob_", "Hob_", "Nf_", "No_", "own_", "route_",
                    "stand_dist")}
MIN_AIRPORT_ROWS = 2000
RIDGE_ALPHA = 10.0


def block_of(fold: str, sources, require_rule=None, shift_s=0.0, pred=None,
             permute=0, thin=1.0, split=0, permute_file=False,
             admit_population: bool = False):
    """The block for a fold's validation month, from the RANKING-MASKED frame."""
    month = BS.MONTH_OF[fold]
    raw = D.load_month(month, columns=BK.FRAME_COLS)
    masked, _ = D.mask_like_ranking(raw)
    del raw
    train = None
    if any(s in sources for s in ("O", "H", "No")) or "geo" in sources:
        # ONE MAP IN BOTH LANES (desk, 2026-09-09): `base.prefix_frame` is the nine
        # non-fold 2025 months, departures only, identical to surfacefit's.
        train = BS.prefix_frame()
    b, rep = BK.build(masked, train, require_rule=require_rule, sources=sources,
                      shift_s=shift_s, permute=permute, thin=thin, pred=pred,
                      split=split, permute_file=permute_file)
    # THE POPULATION PIN (desk, 2026-09-09).  A fold is priced on the airport-days its
    # pin names; a delivery enters only at a RECORDED re-price, which passes `admit`.
    # An unpinned fold is reported as unpinned and built, so a diagnostic written before
    # the pins existed still runs; a PINNED fold that has moved stops the run.
    if rep.get("days_present_list") is not None:
        pop = PIN.population(rep["days_present_list"], BK.B46._summary_path)
        rep["pin"] = PIN.check(fold, pop, require_rule,
                               admit=admit_population or ADMIT_POPULATION)
        rep["population_key"] = rep["pin"]["population_key"]
    return b, rep


def crossfit(X: np.ndarray, y: np.ndarray, days: np.ndarray,
             learner: str = "gbm") -> np.ndarray:
    """`h`, out of contiguous day block.

    `gbm` is exp-041 T7's learner and budget, unchanged, so the two experiments'
    numbers are comparable.  `ridge` exists because a 300-tree fit on the 5,131 rows
    of one harvested airport-week is not a measurement of anything: with five day
    blocks it has one day of held-out rows per fold and it reads NEGATIVE on arms whose
    signal is real.  A linear fit on standardised columns, with a missing indicator per
    column, has the variance a small covered set can afford, and it is also the form
    the shrinkage argument in `state.neighbourhood` predicts is optimal.
    """
    blk = BS.blocks_of(days)
    h = np.zeros(len(y))
    if learner == "ridge":
        miss = ~np.isfinite(X)
        mu = np.nanmean(np.where(miss, np.nan, X), axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.nanstd(np.where(miss, np.nan, X), axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Z = np.where(miss, 0.0, (X - mu) / sd)
        X = np.column_stack([Z, miss.astype("float64")])
    for b in range(BS.BLOCKS):
        fit = blk != b
        if learner == "ridge":
            g = Ridge(alpha=RIDGE_ALPHA, random_state=None)
        else:
            g = HistGradientBoostingRegressor(max_iter=BS.TREES, max_depth=BS.DEPTH,
                                              learning_rate=BS.LR,
                                              random_state=BS.SEED)
        g.fit(X[fit], y[fit])
        h[~fit] = g.predict(X[~fit])
    return h


def share_removed(r: np.ndarray, h: np.ndarray, code: np.ndarray, ncode: int,
                  days: np.ndarray, rng, keep: np.ndarray | None = None) -> dict:
    """`1 - after/before` on the self-excluded cross-product, with its own interval.

    Both ends are ratios of a pair sum to a pair count over the SAME rows and the same
    days, so one draw of days gives one draw of the share.  Reporting an interval on
    each end separately would not gate the quantity the assignment names.
    """
    if keep is None:
        keep = np.ones(len(r), bool)
    rr = r - h
    os0, oc0 = BS.self_excluded(code, ncode, r)
    os1, oc1 = BS.self_excluded(code, ncode, rr)
    n0 = np.where(keep, r * os0, 0.0)
    n1 = np.where(keep, rr * os1, 0.0)
    dd = np.where(keep, oc0, 0.0)
    codes, uniq = pd.factorize(pd.Series(days))
    k = len(uniq)
    a0 = np.bincount(codes, weights=n0, minlength=k)
    a1 = np.bincount(codes, weights=n1, minlength=k)
    de = np.bincount(codes, weights=dd, minlength=k)
    if de.sum() <= 0 or a0.sum() == 0:
        return {"pairs": 0, "before": None, "after": None, "share": None,
                "ci": [None, None]}
    draws = rng.integers(0, k, size=(BS.BOOT, k))
    b0 = a0[draws].sum(1)
    b1 = a1[draws].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(b0 != 0, 1.0 - b1 / b0, np.nan)
    lo, hi = np.nanpercentile(sh, [2.5, 97.5])
    return {"pairs": int(round(float(dd.sum()) / 2.0)),
            "rows": int((keep & (oc0 > 0)).sum()),
            "before_s2": round(float(a0.sum() / de.sum()), 1),
            "after_s2": round(float(a1.sum() / de.sum()), 1),
            "share": round(float(1.0 - a1.sum() / a0.sum()), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)]}


def annotate_shares(rec: dict) -> dict:
    """`(2A - B) / before` beside `1 - after/before`, on every population.

    THE CONTRACT-CONSISTENT SHARE (FALLBACK review round 1, blocking 1).  A correction
    that predicted a cell's common error exactly would take `2A - B` per row to the
    shared component itself, so `(2A - B) / before` measures the same quantity the
    assignment's gate names -- but through the estimator the fold table is scored on,
    and with none of `1 - after/before`'s over-credit for a family whose correction is
    built from the other rows of the same cell.  Both are reported on every row; the
    share is never replaced.

    It is arithmetic on numbers the record already carries, so it can be applied to a
    record written before this function existed without refitting anything.
    """
    for tag in ("all", "matched", "unmatched", "exp041_eligible"):
        pop = rec.get(tag)
        con = rec.get("contract" if tag in ("all", "exp041_eligible")
                      else f"contract_{tag}")
        if not pop or not con or not pop.get("before_s2"):
            continue
        b0 = float(pop["before_s2"])
        pop["gain_over_shared"] = round(con["mse_gain_s2"] / b0, 4)
        ci = con.get("ci95")
        if ci:
            pop["gain_over_shared_ci"] = [round(ci[0] / b0, 4), round(ci[1] / b0, 4)]
    return rec


def covered(fold: str, require_rule: str | None) -> pd.DataFrame:
    """`(MVT_ID_mvt, covered)` -- is this row's own airport-day harvested?

    A row on a day the harvest has not delivered carries an all-NaN observed block, and
    a fit over a month of such rows would be a fit on the covered days padded with
    noise.  The observed arms are therefore read on the covered rows alone, and the
    record says how many they are.
    """
    month = BS.MONTH_OF[fold]
    raw = D.load_month(month, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                       "MVT_TIME_UTC_mvt"])
    dep = raw[(raw.PHASE_mvt == "DEP") & raw.ADEP_mvt.isin(BK.AIRPORTS)]
    day = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy()
    apt = dep.ADEP_mvt.astype(str).to_numpy()
    rules = BK.B46.rule_by_day()
    have = set()
    for a in BK.AIRPORTS:
        for d in sorted(set(day)):
            pth = BK.B46._summary_path(a, d)
            if pth is None:
                continue
            if require_rule is not None and rules.get((str(pth.parents[1]), d)) != require_rule:
                continue
            have.add((a, d))
    ok = np.array([(a, d) in have for a, d in zip(apt, day)])
    return pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy("float64"),
                         "covered": ok})


def rmse4(y: np.ndarray, p: np.ndarray, nonm: np.ndarray) -> dict:
    """`assignments/COMMON.md`'s four-way RMSE: all, bulk (y <= 3600), tail, no-NM."""
    def r(mask):
        if mask.sum() == 0:
            return None
        return round(float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2))), 4)
    on = np.ones(len(y), bool)
    return {"all": r(on), "bulk_le3600": r(y <= 3600), "tail_gt3600": r(y > 3600),
            "no_nm": r(nonm), "rows": int(len(y)), "rows_no_nm": int(nonm.sum())}


def run_fold(fold: str, arms: list[str], sources, require_rule=None,
             shift_s=0.0, cover: bool = False, learner: str = "gbm",
             permute: int = 0, thin: float = 1.0, permute_file: bool = False,
             by_reach: bool = False, single: bool = False,
             split: int = 0) -> dict:
    rng = np.random.default_rng(BS.SEED)
    pan = BS.load()[fold]
    pred = pd.Series(pan["p11"].to_numpy("float64"),
                     index=pan["MVT_ID_mvt"].to_numpy("float64"))
    blk, brep = block_of(fold, sources, require_rule, shift_s, pred, permute, thin,
                         split, permute_file)
    m = pan.merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if len(m) != len(pan):
        raise SystemExit(f"{fold}: the block join changed the row count")
    cov = None
    if cover:
        cov = covered(fold, require_rule)
        m = m.merge(cov, on="MVT_ID_mvt", how="left", validate="one_to_one")
        m = m[m.covered.fillna(False)].reset_index(drop=True)
        if len(m) < 500:
            raise SystemExit(f"{fold}: only {len(m)} covered rows; nothing to fit")
    r = m["r11"].to_numpy("float64")
    days = m["day"].to_numpy()
    if split and "split_coin" in m.columns:
        # THE CLEAN GATE STATISTIC (FALLBACK round 1, blocking 1): pairs are read
        # inside (cell, coin), and each row's correction was built from the OTHER
        # coin's clocks, so no pair's correction has seen either of its members.
        code, ncode = BS.cellcode(m["apt"].to_numpy(), m["bin"].to_numpy(),
                                  m["split_coin"].fillna(-1).astype("int64").to_numpy())
    else:
        code, ncode = BS.cellcode(m["apt"].to_numpy(), m["bin"].to_numpy())
    out = {"fold": fold, "rows": int(len(m)), "cells": ncode, "block": brep,
           "shift_s": shift_s, "cover_only": bool(cover),
           "learner": learner, "permute_seed": permute, "thin": thin,
           "by_reach": bool(by_reach), "single_column": bool(single),
           "split_seed": int(split), "permute_file": bool(permute_file),
           "days": sorted(set(m["day"].tolist())),
           "airports": sorted(set(m["apt"].tolist())), "arms": {}}
    base_shared = BS.shared(r, code, ncode, days, rng)
    out["shared_component_s2"] = base_shared
    jobs = []
    for arm in arms:
        pre = ARMS[arm]
        got = [c for c in blk.columns
               if c.startswith(pre) and blk[c].dtype.kind in "fiub"]
        if not got:
            continue
        jobs += [(c, [c]) for c in got] if single else [(arm, got)]
    for arm, cols in jobs:
        X = m[cols].to_numpy("float64")
        t0 = time.time()
        if by_reach and "obs_has_clock" in blk.columns:
            # THE REPAIR for the reach-class finding: one head per class instead of
            # one head with a missing indicator.  A missing indicator can only move an
            # unseen row's level, and T3 measures that moving it costs -588 s^2 on the
            # rows with no clock; a head fitted on those rows alone cannot spend the
            # own-clock column at all and has to earn its gain from the neighbourhood.
            hc0 = m["obs_has_clock"].fillna(False).to_numpy(bool)
            h = np.zeros(len(r))
            for sel in (hc0, ~hc0):
                if sel.sum() < 200:
                    continue
                h[sel] = crossfit(X[sel], r[sel], days[sel], learner)
        else:
            h = crossfit(X, r, days, learner)
        y = m["y_true"].to_numpy("float64")
        p11 = m["p11"].to_numpy("float64")
        nonm = m["nm_missing"].to_numpy(bool) if "nm_missing" in m.columns \
            else np.zeros(len(m), bool)
        rec = {"features": len(cols), "seconds": round(time.time() - t0, 1),
               "contract": BS.paired(r, h, days, rng),
               "rmse_v11": rmse4(y, p11, nonm),
               "rmse_v11_plus_h": rmse4(y, p11 + h, nonm),
               "all": share_removed(r, h, code, ncode, days, rng)}
        # exp-041's own cohort, so T7's 8.1 / 3.4 / 2.7 % and this share are read on
        # the same rows: the `nm_bulk` ordinary joined regime outside EHAM and EDDF
        elig = ((m["regime"].to_numpy() == "nm_bulk")
                & ~np.isin(m["apt"].to_numpy(), ("EHAM", "EDDF")))
        if elig.sum() > 500:
            rec["exp041_eligible"] = share_removed(r, h, code, ncode, days, rng, elig)
        if "obs_has_clock" in blk.columns:
            hc = m["obs_has_clock"].fillna(False).to_numpy(bool)
            rec["matched"] = share_removed(r, h, code, ncode, days, rng, hc)
            rec["unmatched"] = share_removed(r, h, code, ncode, days, rng, ~hc)
            # the contract number the fold table has to price separately: `2A - B` per
            # row OF THAT POPULATION, so an arm that helps one and hurts the other
            # cannot hide inside an all-row mean
            for tag, msk in (("matched", hc), ("unmatched", ~hc)):
                if msk.sum() > 100:
                    rec[f"contract_{tag}"] = BS.paired(r[msk], h[msk], days[msk], rng)
                    rec[f"rmse_{tag}"] = {
                        "v11": rmse4(y[msk], p11[msk], nonm[msk]),
                        "v11_plus_h": rmse4(y[msk], (p11 + h)[msk], nonm[msk])}
        by = {}
        for a, idx in m.groupby("apt").groups.items():
            keep = np.zeros(len(m), bool)
            keep[m.index.get_indexer(idx)] = True
            if keep.sum() < MIN_AIRPORT_ROWS:
                continue
            by[a] = share_removed(r, h, code, ncode, days, rng, keep)
        rec["by_airport"] = by
        annotate_shares(rec)
        out["arms"][arm] = rec
        print(f"  {arm:8s} {len(cols):3d} cols  share {rec['all']['share']} "
              f"{rec['all']['ci']}  2A-B {rec['contract']['mse_gain_s2']:.1f}",
              flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="july_loo")
    ap.add_argument("--arms", default="Fto,Fob,F")
    ap.add_argument("--sources", default="F")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--shift-hours", type=float, default=0.0)
    ap.add_argument("--learner", default="gbm", choices=["gbm", "ridge"])
    ap.add_argument("--thin", type=float, default=1.0,
                    help="keep this fraction of the observed clocks, to simulate a "
                         "lower-receipt airport on a week where the truth is known")
    ap.add_argument("--by-reach", action="store_true",
                    help="fit one head per reach class instead of one head with a "
                         "missing indicator")
    ap.add_argument("--single", action="store_true",
                    help="price every column of the arm's prefixes on its own")
    ap.add_argument("--permute", type=int, default=0,
                    help="placebo: deal the observed off-block instants out to other "
                         "aircraft inside each airport-day, with this seed")
    ap.add_argument("--permute-file", action="store_true",
                    help="deal the FILE clocks out too, so the placebo bites family Nf")
    ap.add_argument("--split", type=int, default=0,
                    help="split-half: build each row's neighbourhood from the OTHER "
                         "half's clocks and read the cross-product inside (cell, half)")
    ap.add_argument("--cover", action="store_true",
                    help="keep only rows whose own airport-day the harvest delivered")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = a.arms.split(",")
    srcs = tuple(a.sources.split(","))
    print(f"[{a.fold}] arms {arms} sources {srcs} shift {a.shift_hours} h", flush=True)
    rec = run_fold(a.fold, arms, srcs, a.rule, a.shift_hours * 3600.0, a.cover,
                   a.learner, a.permute, a.thin, a.permute_file, a.by_reach,
                   a.single, a.split)
    tag = "-placebo" if a.shift_hours else ("-permuted" if a.permute else "")
    name = a.out or f"explain-{a.fold}{tag}.json"
    (EXP / "analysis" / name).write_text(json.dumps(rec, indent=2, sort_keys=True,
                                                    default=str))
    print("wrote", EXP / "analysis" / name)


if __name__ == "__main__":
    main()
