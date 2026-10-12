"""exp-047-queue step 2: the fold side on the contract, with a CROSS-MONTH rotation.

`assignments/queue.md` step 2: *"the block as an input to a correction of the fold-side
v11 (`r = y - p_v11`, `d` = the correction; `2A - B` all-row at the ranking shares and
clean; december separate; day-clustered intervals; fitted scale), priced separately on
matched rows and on unmatched rows at airports with receipt, and with surfacefit's
observed-off-block feature present in the base where its block exists."*

The difference from `explain.py`, which answers step 1, is the **rotation**: step 1
cross-fits inside the scored month by day block, which exp-041 §5 measured as an upper
bound on what the same correction serves (0.28 to 0.68 of it survives a month change,
per fold 1.23 / 0.47 / 0.11).  Here the head is fitted on a month it does not score.

**The head is v12** (desk, 2026-09-08 14:50 UTC; tail rule): v11 with only its K2 member
replaced, board 262.7418.  `--head v11` still prices against the old head, so the two
readings can be put side by side, but the default and the served endpoint are v12.
`base.v12_vectors` builds its fold side by `exp-035` §26a's own two lines and gates it
three ways, including that the SAME two lines rebuild v11.

**The rotation, and what each fold's head is fitted on.**  `r11` exists only on the
three fold panels -- each panel is one validation month with an out-of-fold v11 -- so a
"fit on the 2025 months other than the scored fold" needs the block on those months.
The harvest delivered 2025-01 at 15:5x UTC on 2026-09-08, so ALL THREE folds now
rotate and the assignment's "every fold interval clear of zero" clause is evaluated on
three folds of three.  The panels are not the same size, and the record has to say so
(FALLBACK round 3, C3): **july_loo and december are each scored by a head fitted on ONE
panel** (december and july_loo respectively), **january by a head fitted on TWO**, and
the SERVED head in `code/serve.py` is fitted on all three.  So the fold table is a
lower bound on the served head's data, not a like-for-like reading of it.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/arms.py \
        --arms No,own,own_No --rule <sha>
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
import explain as EX                                         # noqa: E402
import extend as XT                                          # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

#: scored fold -> the fold panels its head is fitted on.  Only months the harvest has
#: delivered may appear on the right; `main()` checks and says so.
ROTATION = {"july_loo": ["december"], "december": ["july_loo"],
            "january": ["july_loo", "december"]}
SHARES = BS.SHARES


def weighted_moments(X, miss, w):
    """Per-column weighted mean and standard deviation over the NON-missing entries."""
    W = np.where(miss, 0.0, w[:, None])
    tot = W.sum(axis=0)
    Xz = np.where(miss, 0.0, X)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (W * Xz).sum(axis=0) / tot
        var = (W * (Xz - mu) ** 2).sum(axis=0) / tot
    return mu, np.sqrt(var)


def fit_predict(Xf, yf, Xs, learner: str, miss_indicator: bool = False,
                wf=None) -> np.ndarray:
    """One head, fitted on the fitting months and applied to the scored one.

    The ridge's standardisation is fitted on the FITTING rows only and applied to the
    scored rows, so nothing about the scored month reaches the head -- not even its
    column means.

    **An absent observation is never a feature** (desk, 2026-09-08 16:50 UTC).  The
    ridge used to append one indicator column per feature saying "this value is
    missing", which is exactly that: the 2025 feeder network is a different instrument
    (EDDM 0 % for July 1-15, LEMD 55 %, EGLL 12 % rising), so an indicator lets the
    head read "no receiver at EDDM in early July" as a property of EDDM's flights and
    carry it to a ranking month where the receiver exists.  The columns are now
    mean-imputed and nothing else; `miss_indicator=True` restores the old design for a
    side-by-side reading only.  The GBM path never had indicators -- its native NaN
    handling splits on missingness, so it is refused unless the flag is on.
    """
    if learner == "ridge":
        miss_f, miss_s = ~np.isfinite(Xf), ~np.isfinite(Xs)
        if wf is None:
            mu = np.nanmean(np.where(miss_f, np.nan, Xf), axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.nanstd(np.where(miss_f, np.nan, Xf), axis=0)
        else:
            # RESULTS SS7cf: the fitting population is WEIGHTED, so its column means and
            # spreads are the weighted ones -- the imputed value and the scale a row is
            # standardised by are those of the population the fit is told it sees.
            mu, sd = weighted_moments(Xf, miss_f, np.asarray(wf, "float64"))
            mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Zf = np.where(miss_f, 0.0, (Xf - mu) / sd)
        Zs = np.where(miss_s, 0.0, (Xs - mu) / sd)
        if miss_indicator:
            Zf = np.column_stack([Zf, miss_f.astype("float64")])
            Zs = np.column_stack([Zs, miss_s.astype("float64")])
        g = Ridge(alpha=EX.RIDGE_ALPHA)
        if wf is None:
            g.fit(Zf, yf)
        else:
            g.fit(Zf, yf, sample_weight=np.asarray(wf, "float64"))
        return g.predict(Zs)
    if wf is not None:
        raise SystemExit("arms: a fitting weight is defined for the ridge only")
    if not miss_indicator:
        raise SystemExit(
            "arms: the GBM splits on missingness natively, which the desk's 16:50 UTC "
            "rule forbids ('an absent observation is never a feature'). Use --learner "
            "ridge, or pass --miss-indicator to take the old reading knowingly.")
    g = HistGradientBoostingRegressor(max_iter=BS.TREES, max_depth=BS.DEPTH,
                                      learning_rate=BS.LR, random_state=BS.SEED)
    g.fit(Xf, yf)
    return g.predict(Xs)


#: the ranking file's own reach, from `analysis/census-ranking.json` (2026-01, the one
#: ranking month the harvest has delivered).  `project()` uses it as the WEIGHT on each
#: airport's fold gain.
#: one census per RANKING MONTH.  The desk's 17:05 UTC message: both scored months are
#: measured now and the feeder map differs between them -- LIRF is DARK for all 31 days
#: of 2026-07 and healthy in 2026-01 -- so *"the projection of a fold gain to the board
#: is per airport-MONTH, not per airport, and LIRF's with-clock rows count only in
#: January"*.
CENSUS = {"2026-01": EXP / "analysis" / "census-ranking.json",
          "2026-07": EXP / "analysis" / "census-ranking-2026-07.json"}
RANK_BY_APT_MONTH = EXP / "analysis" / "ranking-rows-by-airport-month.json"
RANK_ROWS = 344841
BLOCK = ROOT / "artifacts/exp-047/block"

#: THE REACH GRAIN (FALLBACK round 3, blocking 1).  A no-clock row is *reached* when the
#: network heard at least one OTHER aircraft in its airport-hour cell, so the `No_*`
#: columns carry an observation rather than the fit rows' imputed mean.  Round 3
#: measured that the fold rows and the served rows are different kinds of rows at the
#: two airports that carry most of the projection: EDDF's fold gains are read at reach
#: 0.99 / 0.97 / 0.96 and applied to 2026 rows at 0.64 and 0.82, and LIRF's july gain,
#: read at 0.93, is applied to 15,827 rows at 0.002 in 2026-07.  Projecting the pooled
#: per-airport gain therefore extrapolates a reached-population gain onto an unreached
#: population.  The projection now reads each airport's fold gain SEPARATELY on the two
#: reach classes and applies each to the ranking month's own count of that class.
REACH_COL = "No_n_n_cell"
#: ONE FLOOR FOR ONE MEASUREMENT (desk, 2026-09-09).  A fold cell (airport x reach
#: class) read on fewer rows than this is not priced: its ranking rows carry NO
#: projection and its weight is reported as UNPRICED, not as a gain of zero, and the
#: sensitivity below says what the total becomes if the class-pooled gain is substituted
#: there instead.  Claiming an unmeasured cell's gain is what round 3 blocked; claiming
#: zero is the conservative reading of the same silence, and it is not conservative when
#: the substitute would be negative, which is why both are reported.
#:
#: THE DEFECT THIS REPAIRS.  Until 2026-09-09 this floor was 100 and the SERVING floor
#: (`MIN_FOLD_ROWS`) was 500, so the projection the gate reads priced cells the hold
#: refuses to serve.  On july the whole gate then rested on ONE such cell:
#: `LIRF/unreached`, 141 fold rows extrapolated 125-fold onto 17,651 ranking rows,
#: reading +8,765 s2 per row under arm `N` and -9,175 under the file-clock family --
#: the same 141 rows, opposite signs, and neither number a measurement.  It flattered
#: arm `N`'s july projection by +449 s2 and damned the per-cell form's by -470.  One
#: floor, and the gate can no longer rest on a cell nothing may serve.
MIN_FOLD_ROWS = 500
MIN_CELL = MIN_FOLD_ROWS

#: THE PER-CELL FORM (desk, 2026-09-09), registered as a RULE and not as a pick.  The
#: programme's own rule is that AN ABSENT OBSERVATION IS NEVER A FEATURE (desk,
#: 2026-09-08 16:50 UTC).  On an UNREACHED row every `No_*` column is exactly that: the
#: network heard no other aircraft in the cell, so the value the head reads is the
#: fitting rows' imputed mean, and standardised beside columns that do vary it dilutes
#: the correction.  T7j measured the dilution -- at `LTFM/unreached` the file-clock
#: family alone reads +986 / +524 / +355 s2 per row against arm `N`'s +756 / +232 /
#: +264 on the same rows.  So the arm FOLLOWS THE REACH CLASS: an unreached cell carries
#: `Nf_*` only, a reached cell carries both families.  The form was chosen after reading
#: T7j on these same three folds and is therefore a SECOND selection on them, which
#: RESULTS SS7g states and counts.
#: the desk's lit-fold refinement of the every-fold clause (2026-09-09).  It is a RULE,
#: set here and read by `arms.py` and `serve.py` alike, never a per-run flag.
LIT_RULE = True
#: True prices the desk's literal Rome window instead of the harvest's arrival (§7r)
DESK_WINDOW = bool(os.environ.get("QUEUE_DESK_WINDOW"))

SPLIT_ARM = "N_reach"
SPLIT_FORM = {"reached": "N", "unreached": "Nf"}
#: THE SHRUNK VARIANT (desk, 2026-09-09).  The per-cell form replaces `N` with `Nf` on
#: an unreached row, and its increment over `N` was a MAYBE: positive on every fold and
#: not clear of zero on january (SS7h, T7l).  Replacing is the largest possible step in
#: that direction; a HALF step keeps the same linear gain and pays a quarter of the
#: quadratic penalty, which is what a MAYBE with a positive mean and a wide interval
#: calls for.  **The weight is 0.5 and it was fixed BEFORE the read** -- the desk's
#: order names it, no search was run, and no other weight is priced anywhere in this
#: record.  A third selection on these folds would be a weight chosen after seeing one.
SHRUNK_ARM = "N_reach_half"
SHRUNK_WEIGHT = 0.5
#: arm -> reach class -> {sub-arm: weight}.  A plain arm is not in here at all.
FORMS = {
    SPLIT_ARM: {"reached": {"N": 1.0}, "unreached": {"Nf": 1.0}},
    SHRUNK_ARM: {"reached": {"N": 1.0},
                 "unreached": {"Nf": SHRUNK_WEIGHT, "N": 1.0 - SHRUNK_WEIGHT}},
    # THE REPAIR OF THE COMBINED ARM, DIAGNOSTIC ONLY (RESULTS SS7ac).  `N_regnet` loses
    # to `N` on every fold, and the per-cell reads say where: the two new families are
    # neutral-to-positive on the two REACHED cells and destroy the gain on
    # `LTFM/unreached`, which is the cell they were built for.  These forms give the new
    # families only the reached cells.  They are a THIRD selection on these same three
    # folds -- the form is chosen after reading the per-cell table on them -- so they are
    # a diagnosis of where the families live, never a serving candidate.
    "N_reg_reach": {"reached": {"N_reg": 1.0}, "unreached": {"N": 1.0}},
    "N_regnet_reach": {"reached": {"N_regnet": 1.0}, "unreached": {"N": 1.0}},
}


def arm_columns(arm: str, pool) -> dict[str, list[str]]:
    """The block columns each reach class carries under `arm`."""
    def cols(name):
        pre = EX.ARMS[name]
        return [c for c in pool.columns
                if c.startswith(pre) and pool[c].dtype.kind in "fiub"]
    if arm not in FORMS:
        c = cols(arm)
        return {"reached": c, "unreached": c}
    return {cls: sorted({c for sub in w for c in cols(sub)})
            for cls, w in FORMS[arm].items()}


def arm_predict(arm: str, fit, m, rcol: str, learner: str, pool=None,
                miss_indicator: bool = False, fit_weight=None):
    """`d` for every row of `m` under `arm`, and the form it was built with.

    Under the split form there are two heads, one per reach class of the SCORED rows,
    and BOTH are fitted on the same fitting rows -- the registered `--fit-rows all`
    population.  The reading rejected: fitting each head only on the rows of its own
    reach class, which would change the columns and the fitting population at once and
    halve each head's data, while `Nf_*` is defined identically on both classes and a
    head fitted on every row estimates the same coefficients from twice as many.
    """
    pool = pool if pool is not None else fit
    cols = arm_columns(arm, pool)
    rep = {"arm": arm, "form": {}}
    if arm not in FORMS:
        c = cols["reached"]
        rep["form"] = {"every row": {"arm": arm, "columns": len(c)}}
        rep["columns"] = c
        return fit_predict(fit[c].to_numpy("float64"), fit[rcol].to_numpy("float64"),
                           m[c].to_numpy("float64"), learner,
                           miss_indicator=miss_indicator, wf=fit_weight), rep
    # one head per SUB-ARM, fitted once on the fitting rows and predicted on EVERY row
    # before any selection: the same head on the same rows must give the same bytes
    # here as it does under its own arm, and a matrix of a different shape can differ
    # from it in the last bit.  On the reached cells this is what makes the per-cell
    # forms' files identical to arm `N`'s, which is the check that they move only the
    # unreached rows.
    need = sorted({sub for w in FORMS[arm].values() for sub in w})
    head = {}
    for sub in need:
        c = [x for x in pool.columns
             if x.startswith(EX.ARMS[sub]) and pool[x].dtype.kind in "fiub"]
        head[sub] = fit_predict(fit[c].to_numpy("float64"),
                                fit[rcol].to_numpy("float64"),
                                m[c].to_numpy("float64"), learner,
                                miss_indicator=miss_indicator, wf=fit_weight)
    rch = reached(m)
    d = np.zeros(len(m), "float64")
    for cls, sel in (("reached", rch), ("unreached", ~rch)):
        w = FORMS[arm][cls]
        rep["form"][cls] = {"arms": w, "columns": len(cols[cls]),
                            "rows": int(sel.sum())}
        if not sel.any():
            continue
        d[sel] = sum(wt * head[sub][sel] for sub, wt in w.items())
    rep["columns"] = sorted(set(cols["reached"]) | set(cols["unreached"]))
    return d, rep


def reached(m: pd.DataFrame) -> np.ndarray:
    """True where the network heard at least one other aircraft in the row's cell."""
    # FALLBACK round 5, item 5: an absent count column used to read as "no row is
    # reached", which silently moves every row of an airport into the unreached cell.
    if REACH_COL not in m.columns:
        raise SystemExit(f"reached: the frame carries no {REACH_COL}; every row would "
                         "read as unreached and the reach grain would be a fiction")
    return np.nan_to_num(m[REACH_COL].to_numpy("float64"), nan=0.0) > 0


#: THE SHARED CLASS VECTOR (desk, 2026-09-09).  `obs_has_clock` decides which rows this
#: lane serves and which `exp-046-surfacefit` serves, so the two lanes must read ONE
#: vector or their updates overlap.  It is surfacefit's, published as
#: `artifacts/exp-046/class/<month>.parquet` with `analysis/class-manifest.json`.  After
#: this lane took the shared prefix map the two vectors differ on 20 ranking rows only,
#: all of them class-RULE differences (13 `ambiguous_strict`, 7 `matched_no_clock`) and
#: none of them matcher differences; taking surfacefit's removes those 20 as well.  The
#: `noac` rows (exp-033) are excluded from the served class in both lanes, so they are
#: excluded from the weight too: the weight has to be the population the file moves.
CLASS = ROOT / "artifacts/exp-046/class"


def rank_reach(rows: str) -> tuple[dict, dict]:
    """Per airport-MONTH: the class's ranking rows, and how many of them are reached.

    The CLASS comes from the shared vector, the REACH from this lane's emitted ranking
    block, which holds every ranking row of the month -- `emit.py` slices the ranking
    file itself -- so no row is lost the way reading the census `rows` as a denominator
    lost LIRF's dark July rows.  The matched count is cross-checked against the census
    and against `ranking-rows-by-airport-month.json`, and this lane's own
    `obs_has_clock` against the shared one.
    """
    rk = D.load_ranking(columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk[["MVT_ID_mvt", "ADEP_mvt"]]
    out, chk = {}, {}
    for month in CENSUS:
        f = BLOCK / f"{month}.parquet"
        if not f.exists():
            raise SystemExit(f"rank_reach: {f} is missing; run code/emit.py {month}")
        b = pd.read_parquet(f, columns=["MVT_ID_mvt", "obs_has_clock", REACH_COL])
        cf = CLASS / f"{month}.parquet"
        if not cf.exists():
            raise SystemExit(f"rank_reach: the shared class vector {cf} is missing")
        cl = pd.read_parquet(cf, columns=["MVT_ID_mvt", "obs_has_clock", "noac"])
        own = b["obs_has_clock"].fillna(False).to_numpy(bool)
        b = b.drop(columns=["obs_has_clock"]).merge(
            cl, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        b = b.merge(rk, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        hc = b["obs_has_clock"].fillna(False).to_numpy(bool)
        noac = b["noac"].fillna(False).to_numpy(bool)
        rr = reached(b)
        keep = {"matched": hc & ~noac, "unmatched": ~hc & ~noac}.get(rows, ~noac)
        apt = b["ADEP_mvt"].to_numpy()
        for a in np.unique(apt):
            sel = (apt == a)
            n = int((sel & keep).sum())
            out.setdefault(a, {})[month] = {
                "class_rows": n,
                "reached": int((sel & keep & rr).sum()),
                "reached_share": round(float((rr[sel & keep]).mean()), 4) if n else 0.0,
                "month_rows": int(sel.sum()),
                "matched": int((sel & hc).sum())}
        chk[month] = {"block_rows": int(len(b)),
                      "class_vector": str(cf.relative_to(ROOT)),
                      "class_sha256": D.sha256(cf),
                      "own_obs_has_clock_disagrees": int((own != hc).sum()),
                      "noac_excluded": int(noac.sum())}
    return out, chk


def rank_weights(rows: str) -> tuple[dict, dict]:
    """Per-airport weight, SPLIT BY REACH: `{airport: {"reached": w, "unreached": w}}`.

    The desk's 2026-09-08 16:50 UTC rule: *"the fold gain is read per airport and
    PROJECTED to the ranking months' own reach ... so a 2025 airport with no receiver is
    not read as a property of its flights"*.  The 2025 feeder network is a different
    instrument -- EDDM 0 % for July 1-15, LEMD 55 %, EGLL 12 % rising to 65 -- so the
    POOLED fold gain on a class mixes airports whose class membership is an artefact of
    the 2025 receiver map.  Reading the gain per airport and weighting by the RANKING
    month's own class shares removes that: EDDM's July no-clock rows price EDDM's gain
    per no-clock row, and that gain is then applied to the few no-clock rows EDDM
    actually has in 2026-01.  Per airport-MONTH since the desk's 17:05 UTC message: the
    feeder map differs between the two scored months, LIRF being dark for all 31 days of
    2026-07 and healthy in 2026-01.

    **And per REACH CLASS** (FALLBACK round 3, blocking 1).  Airport is not a fine
    enough grain: the 2025 receiver map decides not only WHICH rows are no-clock but
    also whether a no-clock row has a heard NEIGHBOUR, and the correction is a different
    object in the two cases -- with a heard neighbour the `No_*` columns carry an
    observation, without one they carry the fit rows' imputed mean and the correction
    is file-clock work.  EDDF is read at reach 0.99 on the folds and served at 0.64 and
    0.82; LIRF is read at 0.93 on july and served at 0.002 in 2026-07.  So the weight is
    split: an airport-month's class rows times its own reached share go to the
    airport's REACHED fold gain, and the rest to its UNREACHED fold gain.

    This is exp-043 step 4's arithmetic (`extrapolate.py`: per-served-row gain times
    served rows over ranking rows), with the airport and the reach dimensions added.
    """
    ranks = json.loads(RANK_BY_APT_MONTH.read_text())
    seen = {m: json.loads(pth.read_text())["by_airport"]
            for m, pth in CENSUS.items() if pth.exists()}
    miss = [m for m in CENSUS if m not in seen]
    if miss:
        raise SystemExit(f"rank_weights: no census for {miss}; run code/census.py "
                         f"--month {miss[0]} --out census-ranking-{miss[0]}.json")
    reach, _chk = rank_reach(rows)
    w, det, tot, disagree = {}, {}, 0, {}
    for apt, by_month in ranks.items():
        for month, n_rows in by_month.items():
            # the class comes from the SHARED vector and the reached share from this
            # lane's emitted ranking block, which holds every ranking row of the month.
            # This lane's own census `own_clock` and the ranking-rows file are read as a
            # CROSS-CHECK.  A row count that disagrees is a defect; a MATCHED count that
            # disagrees is the class RULE differing between the lanes (13
            # `ambiguous_strict` this lane calls with-clock, 7 `matched_no_clock` it
            # calls no-clock), which is why the shared vector exists, so it is recorded
            # per airport-month rather than absorbed.
            rb = reach.get(apt, {}).get(month)
            if rb is None:
                raise SystemExit(f"rank_weights: no block reading for {apt} {month}")
            n = int(rb["class_rows"])
            matched_census = int(seen.get(month, {}).get(apt, {}).get("own_clock", 0))
            if rb["matched"] != matched_census or rb["month_rows"] != n_rows:
                disagree[f"{apt} {month}"] = {
                    "block_matched": rb["matched"], "census_own_clock": matched_census,
                    "block_month_rows": rb["month_rows"], "ranking_json_rows": n_rows}
            q = float(rb["reached_share"])
            cell = w.setdefault(apt, {"reached": 0.0, "unreached": 0.0})
            cell["reached"] += n * q / RANK_ROWS
            cell["unreached"] += n * (1.0 - q) / RANK_ROWS
            tot += n
            det.setdefault(apt, {})[month] = {
                "ranking_rows": n_rows, "matched": matched_census, "class_rows": n,
                "class_share": round(n / max(n_rows, 1), 4),
                "reached_share": round(q, 4),
                "class_rows_reached": int(rb["reached"]),
                "class_rows_unreached": int(n - rb["reached"]),
                "census_covered_rows": int(seen.get(month, {}).get(apt, {})
                                           .get("rows", 0))}
    rep = {"census": {m: str(pth.relative_to(EXP)) for m, pth in CENSUS.items()},
           "ranking_rows_by_airport_month": str(RANK_BY_APT_MONTH.relative_to(EXP)),
           "reach_read_from": str(BLOCK.relative_to(ROOT)) + "/<month>.parquet",
           "reach_column": REACH_COL,
           "rank_rows": RANK_ROWS, "class": rows, "rows_carried": int(tot),
           "share_of_ranking": round(tot / RANK_ROWS, 4),
           "weight_reached": round(sum(v["reached"] for v in w.values()), 5),
           "weight_unreached": round(sum(v["unreached"] for v in w.values()), 5),
           "cross_check_vs_own_census": disagree,
           "class_vector": _chk,
           "by_airport_month": det,
           "note": "per airport-MONTH (desk 17:05 UTC) and per REACH CLASS (FALLBACK "
                   "round 3, blocking 1): the ranking block gives the rows, the class "
                   "and the reached share, so an airport dark in one month (LIRF in "
                   "2026-07, LTFM in both) carries all of that month's rows in the "
                   "no-clock class, none in the with-clock class, and almost all of "
                   "them in the UNREACHED half, which is priced on the fold's own "
                   "unreached rows."}
    return w, rep


def _pct(x: np.ndarray) -> list:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


#: THE CELL HOLD (desk, 2026-09-09), the rule `exp-046-surfacefit` applied to LFPG:
#: *"a served airport-month cell whose fold-side read has no support of its own -- its
#: own day-clustered fold interval crosses zero, or it is extrapolated from under 500
#: fold rows as the 141-row cell is, 125-fold onto 15,551 dark LIRF rows -- is held at
#: the base, with the projected gain forgone stated per cell"*.
#:
#: The test is applied to EVERY fold that prices the cell, exactly as the assignment's
#: advance clause is applied to every fold of the arm: a correction served to a
#: population has to be supported on each fold, not on the average of three.  A fold in
#: which the cell is unpriced at all (under `MIN_CELL` rows) fails it.  And "clear of
#: zero" means clear ON THE POSITIVE SIDE: a cell whose interval sits entirely below
#: zero is the strongest reason to hold it, not a reason to serve it.  The rejected
#: reading, recorded because it changes the candidate: pooling the cell's three folds
#: into ONE interval would keep LIRF/unreached, whose 141 july rows are the very cell
#: the desk named, so the singular "its own fold interval" is read per fold.
#: THE INSTRUMENT WINDOWS (desk, 2026-09-09, from exp-045-harvest's dated events).
#: A cell cannot be asked to clear a fold in which its airport's feeder did not exist.
#: The harvest finds exactly TWO airports that begin inside the training year on a dated
#: day -- *"Madrid switches on on 2025-03-04"* and *"Rome switches on on 2025-05-24"*
#: (exp-045-harvest/RESULTS.md) -- so those two arrivals, and nothing else, make a fold
#: month dark.  A DECLINE is not darkness: EGLL bottoms at 8.7 % and recovers, EDDM has
#: a fortnight's outage inside 2025-07, LFPG never rises above 9.2 % and LTFM reads 0.0 %
#: in every month, and none of them is a dated arrival.  Holding LTFM for its receipt
#: would drop the served LTFM/unreached cell, whose neighbour columns come off the FILE
#: clock (`Nf_*`), not off the feeder at all -- which is why the test is the arrival and
#: not the coverage rate.
INSTRUMENT_ARRIVAL = {"LEMD": "2025-03-04", "LIRF": "2025-05-24"}

#: THE ONE CONFLICT, RAISED AND SETTLED (RESULTS §7r).  The desk's first line read the
#: Rome window as
#: "2025-05-24..08-28", which would make december dark as well.  The harvest's own
#: series does not support the closing date as darkness: Rome falls from 93 % to the
#: thirties on 2025-08-28 and then CLIMBS BACK -- 18.4 % Sep, 38.2 Oct, **62.4 % in the
#: december fold** and 66.0 % in the 2026-01 ranking month -- so december reads HIGHER
#: than EGLL's 57.0 %, which the same line treats as lit.  Rome is dark in the scored
#: 2026-07 (0.1 %), which is a ranking month and not a fold.  The record keeps both: the
#: arrival reading is the one served, and `_desk_window` prices the literal clause
#: beside it so the record can say what it would have cost.  **The desk withdrew the
#: closing date on 2026-09-09** (*"your reading is the right one and replaces mine --
#: LIT means the instrument has ARRIVED at the airport; a decline afterwards is lit,
#: darkness is only the time before arrival"*), so the arrival reading is the rule and the
#: window below is kept only as that sensitivity.
#:
#: THE PARENTHETICAL, in the desk's own final wording (2026-09-09, after FALLBACK round 5
#: item 2 found this file and RESULTS quoting the clause two ways): darkness is the time
#: before arrival *"or a month in which the cell's arm reads a surface observation that is
#: 0 % by instrument"*.  The lit rule reaches a cell only through the features its arm
#: READS.  `LTFM/unreached` is served on the FILE-clock family alone -- `Nf_*`, with no
#: observed neighbour anywhere in it -- so it has no surface instrument that can be dark:
#: its instrument is the ranking file, present in every month.  It therefore has three lit
#: folds and stays served, even though LTFM's feeder reads 0.0 % in all thirteen months.
#: Read the parenthetical as a test on the ARM'S INPUTS, never on the airport's receipt.
DESK_WINDOWS = {"LEMD": ("2025-03-04", None), "LIRF": ("2025-05-24", "2025-08-28")}

#: the per-fold receipt behind all of this, from exp-045-harvest/analysis/receipt.json
#: (`by_airport` -> `rate`), recorded here so the rule can be read without the harvest:
#: EDDF 81.2/82.9/92.6, EDDM 97.1/36.9/97.1, EGLL 57.1/15.1/57.0, EHAM 98.0/96.2/96.5,
#: LEBL 97.7/89.2/95.9, LEMD 0.0/28.8/96.5, LFPG 0.0/4.8/9.2, LIRF 0.0/90.6/62.4,
#: LSZH 93.3/88.5/95.9, LTFM 0.0/0.0/0.0 (january / july_loo / december).
RECEIPT_BY_FOLD = {
    "EDDF": (0.812, 0.829, 0.926), "EDDM": (0.971, 0.369, 0.971),
    "EGLL": (0.571, 0.151, 0.570), "EHAM": (0.980, 0.962, 0.965),
    "LEBL": (0.977, 0.892, 0.959), "LEMD": (0.000, 0.288, 0.965),
    "LFPG": (0.000, 0.048, 0.092), "LIRF": (0.000, 0.906, 0.624),
    "LSZH": (0.933, 0.885, 0.959), "LTFM": (0.000, 0.000, 0.000)}

#: a winter RANKING month is served (2026-01), so a served cell has to carry a winter fold
WINTER_FOLD = "december"
MIN_LIT_FOLDS = 2


def lit_folds(airport: str, folds, desk_window: bool = False) -> list[str]:
    """The folds in which this airport's instrument existed, in the record's order."""
    out = []
    for f in folds:
        m = BS.MONTH_OF.get(f)
        if m is None:
            continue
        first, last = f"{m}-01", f"{m}-31"
        if desk_window:
            win = DESK_WINDOWS.get(airport)
            if win is None:
                out.append(f)
                continue
            start, end = win
            if (start is None or start <= last) and (end is None or end >= first):
                out.append(f)
            continue
        arrived = INSTRUMENT_ARRIVAL.get(airport)
        if arrived is None or arrived <= last:
            out.append(f)
    return out


def cell_supported(per_fold: dict, airport: str | None = None,
                   lit_rule: bool = False, desk_window: bool = False,
                   min_lit: int | None = None,
                   require_winter: bool = True) -> tuple[bool, dict]:
    """`per_fold` is `{fold: the cell's record in by_airport_reach}` -> served?, why.

    `lit_rule` reads the desk's refinement of 2026-09-09: the every-fold clause means
    every fold **in which the cell's airport instrument existed**.  A cell dark by
    instrument in a fold is priced on its LIT folds -- 500+ rows and an interval clear of
    zero above in EACH of them, at least two of them, and december among them because a
    winter ranking month is served.  Without `lit_rule` the older reading applies: every
    fold, lit or not, which holds a cell for the instrument's absence and not the arm's.
    """
    if lit_rule:
        lit = lit_folds(airport or "", sorted(per_fold), desk_window=desk_window)
        dark = [f for f in sorted(per_fold) if f not in lit]
        ok, why = True, {f: "dark by instrument, not priced here" for f in dark}
        need = MIN_LIT_FOLDS if min_lit is None else min_lit
        if len(lit) < need:
            return False, {**why, "lit": f"{len(lit)} lit folds, under {need}"}
        if require_winter and WINTER_FOLD not in lit:
            return False, {**why,
                           "lit": f"{WINTER_FOLD} is dark and a winter ranking month is served"}
        sub, subwhy = cell_supported({f: per_fold[f] for f in lit})
        return sub, {**why, **subwhy, "lit_folds": ", ".join(lit)}
    ok, why = True, {}
    for f, v in sorted(per_fold.items()):
        if v is None:
            ok, why[f] = False, "absent from this fold"
            continue
        n = int(v.get("fold_rows", 0))
        if not v.get("priced"):
            ok, why[f] = False, f"unpriced, {n} fold rows"
            continue
        bad = []
        if n < MIN_FOLD_ROWS:
            bad.append(f"{n} fold rows, under {MIN_FOLD_ROWS}")
        if v["gain_ci95_s2"][0] <= 0:
            bad.append(f"interval {v['gain_ci95_s2']} is not clear of zero above")
        why[f] = "; ".join(bad) or "supported"
        ok &= not bad
    return ok, why


def project(fold_rec: dict, w: dict, rng) -> dict:
    """The projected board gain: each airport's fold gain, per reach class, at the
    RANKING month's own class count and reached share.

    One bootstrap draw resamples the scored fold's DAYS once and rebuilds every
    airport-and-reach-class gain from the resampled rows, so the interval carries the
    correlation between airports on a day -- and the leave-one-airport-out interval is
    read off the SAME draws, which is what makes "still clear with EDDF dropped" a
    measurement rather than a point value (FALLBACK round 3, blocking 1).
    """
    out, keep_draws = {}, {f: {} for f in fold_rec}
    for f, v in fold_rec.items():
        codes, uniq = pd.factorize(pd.Series(v["days"]))
        k = len(uniq)
        draws = rng.integers(0, k, size=(BS.BOOT, k))
        e = v["e0"] - v["e1"]
        rch = v["reached"]

        def cell(sel):
            s = np.bincount(codes[sel], weights=e[sel], minlength=k)
            c = np.bincount(codes[sel], minlength=k).astype("float64")
            num, den = s[draws].sum(1), c[draws].sum(1)
            return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)

        # the class-pooled gain across airports, used ONLY in the sensitivity for cells
        # the fold does not price
        pooled = {}
        for cls, cm in (("reached", rch), ("unreached", ~rch)):
            pooled[cls] = (cell(cm) if cm.sum() else np.zeros(BS.BOOT),
                           float(e[cm].mean()) if cm.sum() else 0.0, int(cm.sum()))

        contrib, sens, per, byapt, unpriced, cellc = {}, {}, {}, {}, {}, {}
        pt, sens_pt, unpriced_w = 0.0, 0.0, 0.0
        for apt, am in v["by_apt"].items():
            if apt not in w:
                continue
            ca, sa, pa, spa, rows = (np.zeros(BS.BOOT), np.zeros(BS.BOOT), 0.0, 0.0, 0)
            for cls, cm in (("reached", rch), ("unreached", ~rch)):
                wt = float(w[apt][cls])
                sel = am & cm
                n = int(sel.sum())
                rows += n
                if wt <= 0.0 and n == 0:
                    continue
                # THE NUMBER THAT MATTERS FOR THIS FINDING: how far the cell is
                # extrapolated.  A gain read on 141 fold rows and applied to 17,650
                # ranking rows is a measurement of the 141, whatever its interval says.
                nrank = int(round(wt * RANK_ROWS))
                rec = {"fold_rows": n, "weight": round(wt, 5),
                       "ranking_rows": nrank,
                       "extrapolation": round(nrank / max(n, 1), 1)}
                if n < MIN_CELL:
                    rec["priced"] = False
                    rec["contribution_s2"] = 0.0
                    rec["sensitivity_gain_per_row_s2"] = round(pooled[cls][1], 1)
                    unpriced[f"{apt}/{cls}"] = {"weight": round(wt, 5),
                                                "fold_rows": n}
                    unpriced_w += wt
                    sa += wt * pooled[cls][0]
                    spa += wt * pooled[cls][1]
                else:
                    g = cell(sel)
                    gp = float(e[sel].mean())
                    ca += wt * g
                    pa += wt * gp
                    sa += wt * g
                    spa += wt * gp
                    rec.update(priced=True, gain_per_row_s2=round(gp, 1),
                               gain_ci95_s2=_pct(g),
                               contribution_s2=round(wt * gp, 2),
                               contribution_ci95_s2=_pct(wt * g))
                    cellc[f"{apt}/{cls}"] = (wt * g, wt * gp)
                    keep_draws[f][f"{apt}/{cls}"] = (wt * g, wt * gp)
                per[f"{apt}/{cls}"] = rec
            contrib[apt] = ca
            sens[apt] = sa
            pt += pa
            sens_pt += spa
            byapt[apt] = {"rows": rows, "weight": round(sum(w[apt].values()), 5),
                          "contribution_s2": round(pa, 2),
                          "contribution_ci95_s2": _pct(ca),
                          "gain_per_row_s2": round(pa / max(sum(w[apt].values()), 1e-9), 1)}
        tot = sum(contrib.values()) if contrib else np.zeros(BS.BOOT)
        stot = sum(sens.values()) if sens else np.zeros(BS.BOOT)
        # NOTE on the board column below: it converts an s^2 drop at v12's CLEAN board,
        # which is the smaller of the two bases, so it is the larger of the two possible
        # second-counts.  The gate is the s^2 interval; `code/serve.py` measures the
        # served file's board directly instead of converting.  An airport with few rows
        # on a resampled day can give an empty denominator, and such a draw contributes
        # a gain of zero, which pulls an interval toward zero rather than away from it.
        # ONE AIRPORT CAN CARRY THE WHOLE PROJECTION, so the leave-one-out is reported
        # WITH ITS INTERVAL, off the same draws.
        loo = {a: {"pooled_mse_reduction_s2": round(pt - byapt[a]["contribution_s2"], 2),
                   "ci95_s2": _pct(tot - c)} for a, c in contrib.items()}
        # and per CELL, because an airport is not the unit the projection rests on
        # either: july's total is carried by LIRF's UNREACHED cell, priced on 141 fold
        # rows.  A reader who wants to know what the number would be without any one
        # cell should not have to take it on trust.
        cloo = {k: {"pooled_mse_reduction_s2": round(pt - v[1], 2),
                    "ci95_s2": _pct(tot - v[0])} for k, v in cellc.items()}
        mse = BS.V12_BOARD_CLEAN ** 2
        lo, hi = np.percentile(tot, [2.5, 97.5])
        out[f] = {"pooled_mse_reduction_s2": round(float(pt), 2),
                  "ci95_s2": [round(float(lo), 2), round(float(hi), 2)],
                  "board_gain_s": round(float(mse ** 0.5 - max(mse - pt, 0.0) ** 0.5), 4),
                  "board_gain_ci95_s": [
                      round(float(mse ** 0.5 - max(mse - lo, 0.0) ** 0.5), 4),
                      round(float(mse ** 0.5 - max(mse - hi, 0.0) ** 0.5), 4)],
                  "leave_one_airport_out": loo,
                  "leave_one_cell_out": cloo,
                  "leave_one_airport_out_s2": {a: v["pooled_mse_reduction_s2"]
                                               for a, v in loo.items()},
                  "largest_contributor": (max(byapt.items(),
                                              key=lambda kv: kv[1]["contribution_s2"])[0]
                                          if byapt else None),
                  "unpriced_weight": round(unpriced_w, 5),
                  "unpriced_cells": unpriced,
                  "sensitivity_unpriced_filled": {
                      "pooled_mse_reduction_s2": round(float(sens_pt), 2),
                      "ci95_s2": _pct(stot),
                      "note": "the same total with every unpriced cell filled by the "
                              "fold's CLASS-POOLED gain across airports. The headline "
                              "gives an unpriced cell a gain of zero; this says what it "
                              "becomes if the class average is assumed there instead."},
                  "reach_class_pooled_gain_per_row_s2": {
                      c: {"fold_rows": pooled[c][2],
                          "gain_per_row_s2": round(pooled[c][1], 1)} for c in pooled},
                  "by_airport": byapt,
                  "by_airport_reach": per}
    # SECOND PASS: the projection restricted to the cells the SERVING rule keeps, with
    # its interval off the same draws.  This is the number that gates the file actually
    # written; the headline above is the number for the whole class, which the candidate
    # no longer serves.
    #
    # THIRD PASS: the hold CROSS-FITTED (desk rule, 2026-09-09, after BOARD READING 15:
    # any increment whose FORM was selected on the folds is priced select-on-two,
    # price-on-the-third).  Each fold's served set is chosen from the OTHER folds by the
    # same rule and the gain is read on the held-out fold.  A cell the others select but
    # this fold does not price contributes nothing and is named, exactly as the one-floor
    # rule requires.  Under the lit rule a two-lit-fold cell has only ONE lit fold left
    # in any selection set, so the cross-fit there asks that one lit fold to clear
    # (`min_lit=1`); it is still out of sample for the fold being priced, but it is a
    # weaker selection than a three-lit cell's, and `selected_on_lit` says which.
    #
    # BOTH READINGS OF THE ROME WINDOW are priced in the same run (§7r): the served set
    # is the arrival reading, and `*_desk_window` repeats the two passes under the desk's
    # literal 2025-05-24..08-28 clause, which differs at LIRF and nowhere else.
    cells = sorted({c for f in out for c in out[f]["by_airport_reach"]})
    mse = BS.V12_BOARD_CLEAN ** 2

    def _pass(dw: bool, suffix: str) -> list[str]:
        served, why = {}, {}
        for c in cells:
            served[c], why[c] = cell_supported(
                {f: out[f]["by_airport_reach"].get(c) for f in out},
                airport=c.split("/")[0], lit_rule=LIT_RULE, desk_window=dw)
        keep = sorted(c for c in cells if served[c])
        for f in out:
            arrs = [keep_draws[f][c][0] for c in keep if c in keep_draws[f]]
            pts = [keep_draws[f][c][1] for c in keep if c in keep_draws[f]]
            tot = sum(arrs) if arrs else np.zeros(BS.BOOT)
            pt = float(sum(pts))
            lo, hi = np.percentile(tot, [2.5, 97.5])
            out[f]["served_cells_only" + suffix] = {
                "cells": keep,
                "pooled_mse_reduction_s2": round(pt, 2),
                "ci95_s2": [round(float(lo), 2), round(float(hi), 2)],
                "board_gain_s": round(float(mse ** 0.5 - max(mse - pt, 0.0) ** 0.5), 4),
                "forgone_s2": round(out[f]["pooled_mse_reduction_s2"] - pt, 2),
                "forgone_by_cell": {c: out[f]["by_airport_reach"][c].get("contribution_s2")
                                    for c in cells
                                    if not served[c] and c in out[f]["by_airport_reach"]},
                "hold_reason": {c: why[c] for c in cells if not served[c]},
                "lit_rule": LIT_RULE, "desk_window": dw,
                "lit_folds": {c: lit_folds(c.split("/")[0], sorted(out), desk_window=dw)
                              for c in cells},
                "min_fold_rows": MIN_FOLD_ROWS,
                "note": "the desk's 2026-09-09 cell hold under the lit-fold refinement: "
                        "a cell is served when it clears in EVERY fold in which its "
                        f"airport instrument existed -- {MIN_FOLD_ROWS}+ rows and a "
                        "day-clustered interval clear of zero above in each -- on at "
                        f"least {MIN_LIT_FOLDS} lit folds with {WINTER_FOLD} among them, "
                        "because a winter ranking month is served."}
        for f in out:
            others = [g for g in out if g != f]
            sel, swhy, sellit = {}, {}, {}
            for c in cells:
                sellit[c] = lit_folds(c.split("/")[0], sorted(others), desk_window=dw)
                sel[c], swhy[c] = cell_supported(
                    {g: out[g]["by_airport_reach"].get(c) for g in others},
                    airport=c.split("/")[0], lit_rule=LIT_RULE, desk_window=dw,
                    min_lit=1, require_winter=False)
            keep_x = sorted(c for c in cells if sel[c])
            priced = [c for c in keep_x if c in keep_draws[f]]
            arrs = [keep_draws[f][c][0] for c in priced]
            pts = [keep_draws[f][c][1] for c in priced]
            tot = sum(arrs) if arrs else np.zeros(BS.BOOT)
            pt = float(sum(pts))
            lo, hi = np.percentile(tot, [2.5, 97.5])
            out[f]["served_cells_crossfit" + suffix] = {
                "selected_on": others, "cells": keep_x,
                "selected_on_lit": {c: sellit[c] for c in keep_x},
                "one_lit_fold_selection": [c for c in keep_x if len(sellit[c]) < 2],
                "cells_unpriced_on_this_fold": [c for c in keep_x
                                                if c not in keep_draws[f]],
                "pooled_mse_reduction_s2": round(pt, 2),
                "ci95_s2": [round(float(lo), 2), round(float(hi), 2)],
                "board_gain_s": round(float(mse ** 0.5 - max(mse - pt, 0.0) ** 0.5), 4),
                "same_set_as_the_in_sample_hold": keep_x == keep,
                "note": "the served set for this fold is chosen on the other folds by "
                        f"the same rule (>= {MIN_FOLD_ROWS} rows and an interval clear "
                        "of zero above in each LIT fold among them) and the gain is read "
                        "here, so the selection is paid for."}
        return keep

    _pass(DESK_WINDOW, "")
    if LIT_RULE and not DESK_WINDOW:
        _pass(True, "_desk_window")
    return out



def has_clock(m: pd.DataFrame) -> np.ndarray:
    """The matched mask, false where the block did not reach the row at all."""
    if "obs_has_clock" not in m.columns:
        return np.zeros(len(m), bool)
    return m["obs_has_clock"].fillna(False).to_numpy(bool)


def drop_dark_fit_rows(fit: pd.DataFrame, floor: float) -> tuple[pd.DataFrame, dict]:
    """Fit rows whose airport-day receipt is below `floor` are not fitted on.

    Diagnosed here, on the first cross-month read: `No` on the no-clock class scores
    -2,959 s^2 per row at EGLL in december, fitted on july_loo, and july_loo is exactly
    the month in which EGLL's feeder ran at 12 % rising to 65.  A july EGLL no-clock row
    is mostly a normal flight the receiver missed, so it teaches the head the receiver's
    map and not the airport's traffic.  The desk's 16:50 UTC projection rule repairs the
    SCORED side of that; this repairs the FITTED side.  `floor = 0` keeps every row and
    reproduces the unfiltered reading exactly.
    """
    if floor <= 0 or "receipt_rate" not in fit.columns:
        return fit, {"floor": floor, "applied": False}
    r = fit["receipt_rate"].to_numpy("float64")
    keep = ~(np.isfinite(r) & (r < floor))
    rep = {"floor": floor, "applied": True, "rows_before": int(len(fit)),
           "rows_kept": int(keep.sum()), "rows_dropped": int((~keep).sum()),
           "rows_with_no_rate": int((~np.isfinite(r)).sum()),
           "note": "a row with NO rate is off a harvested airport-day entirely and is "
                   "KEPT: its class is not an instrument artefact of a day that was "
                   "measured, and dropping it would delete the population the no-clock "
                   "head exists to serve."}
    return fit[keep].reset_index(drop=True), rep


REACH = EXP / "analysis" / "december-reconcile.json"


def instrument_gap() -> dict:
    """|reach(airport, fold month) - reach(airport, the ranking file)| per airport-month.

    `reach` is the share of an airport's departures the matcher clocks, read from the
    five emitted blocks.  It is a property of the RECEIVER MAP and of nothing else, so
    selecting fitting rows on it selects on the instrument rather than on the outcome --
    which is where `--fit-min-receipt` went wrong: a receipt floor keeps the no-clock
    rows a HEALTHY receiver still missed, an unrepresentative tail of the served class.
    Dropping a whole airport-month keeps the class intact wherever it is kept.
    """
    d = json.loads(REACH.read_text())["reach_by_airport_month"]
    rank, gap = {}, {}
    for a, v in d.items():
        num = sum(v[m]["clocked"] for m in ("2026-01", "2026-07") if m in v)
        den = sum(v[m]["rows"] for m in ("2026-01", "2026-07") if m in v)
        rank[a] = num / den if den else 0.0
        for m, x in v.items():
            if m.startswith("2025"):
                gap[(a, m)] = round(abs(x["reach"] - rank[a]), 4)
    return {"ranking_reach": {a: round(r, 4) for a, r in rank.items()}, "gap": gap}


def drop_far_instrument(fit: pd.DataFrame, g: float) -> tuple[pd.DataFrame, dict]:
    """Fitting rows at an airport-MONTH whose reach is far from the ranking file's."""
    if g <= 0 or "month" not in fit.columns:
        return fit, {"max_gap": g, "applied": False}
    ig = instrument_gap()
    key = list(zip(fit["apt"].astype(str), fit["month"].astype(str)))
    drop = np.array([ig["gap"].get(k, 0.0) > g for k in key])
    rep = {"max_gap": g, "applied": True, "rows_before": int(len(fit)),
           "rows_kept": int((~drop).sum()), "rows_dropped": int(drop.sum()),
           "airport_months_dropped": sorted(
               {f"{a} {m}" for (a, m), v in ig["gap"].items() if v > g}),
           "ranking_reach": ig["ranking_reach"]}
    return fit[~drop].reset_index(drop=True), rep


def keep_rows(m: pd.DataFrame, rows: str) -> pd.DataFrame:
    """The population a head is fitted and priced on (desk, 2026-09-08 16:50 UTC).

    Fitting the with-clock head on all rows and reading it on the matched ones is not
    the same head: the unmatched rows carry an imputed observation column, so they pull
    its coefficient toward the imputed value.  `--rows matched` fits it where the
    observation exists, which is where it serves.
    """
    if rows == "all":
        return m
    hc = has_clock(m)
    sel = hc if rows == "matched" else ~hc
    return m[sel].reset_index(drop=True)


def fitted_scale(r: np.ndarray, d: np.ndarray) -> float:
    """The amplitude that maximises `2A - B`: `mean(r*d)/mean(d*d)`, 0 where undefined.

    Reported beside the full-amplitude number, never substituted for it: the
    assignment's advance clause is "every fold interval clear of zero AT FULL
    AMPLITUDE", and a scale fitted on the scored month would not be servable anyway.
    """
    b = float(np.mean(d * d))
    return float(np.mean(r * d) / b) if b > 0 else 0.0


def oof_amplitude(arm: str, fit: pd.DataFrame, xmonths, rcol: str, learner: str,
                  pool, miss_indicator: bool) -> tuple[float, dict]:
    """`mean(r*d)/mean(d*d)`, leave-one-month-out on the EXTENSION months only.

    `fitted_scale` is this same ratio read on the scored fold, which is why the record
    has always reported it beside the full-amplitude number and never substituted it:
    *"a scale fitted on the scored month would not be servable anyway"*.  This estimator
    is the servable one.  Every row it reads is an extension month's row; no fold is
    scored on an extension month, so for each of the three folds the number is out of
    fold by construction and not by argument.  Each month's `d` comes from a head fitted
    on the same population the served head is fitted on MINUS that month, through the
    same `arm_predict` call, so the instrument is not a second instrument.

    RESULTS SS7cd registers it before the run.  It is not clipped: a value above 1 would
    say the folds want MORE amplitude than full, and hiding that would make the number a
    decision rather than a measurement.
    """
    mcol = fit["month"].astype(str).to_numpy()
    num = den = 0.0
    per = {}
    for mm in xmonths:
        sel = mcol == mm
        n = int(sel.sum())
        if n < 100:
            per[mm] = {"rows": n, "used": False}
            continue
        dm, _ = arm_predict(arm, fit[~sel], fit[sel], rcol, learner, pool=pool,
                            miss_indicator=miss_indicator)
        rm = fit.loc[sel, rcol].to_numpy("float64")
        a_, b_ = float(np.sum(rm * dm)), float(np.sum(dm * dm))
        num += a_
        den += b_
        per[mm] = {"rows": n, "used": True,
                   "scale_on_this_month": round(a_ / b_, 4) if b_ > 0 else None,
                   "mean_d2": round(b_ / n, 2)}
    amp = float(num / den) if den > 0 else 1.0
    rep = {"how": "leave-one-month-out on the extension months (RESULTS SS7cd)",
           "amplitude": round(amp, 4), "months": per,
           "pooled_num": round(num, 1), "pooled_den": round(den, 1),
           "clipped": False,
           "note": "no row of the scored fold enters this number; the months it is read "
                   "on are fitting months that no fold is ever scored on"}
    print(f"    amplitude(oof) {amp:.4f} from {sum(1 for v in per.values() if v['used'])}"
          f" extension months", flush=True)
    return amp, rep


def board_interval(rec: dict, rng) -> dict:
    """A day-clustered interval on the BOARD gain, resampling both share folds at once.

    `paired()` gives an interval per fold, which is not the endpoint the advance clause
    gates.  The board is `sqrt(sum share * MSE)`, so one draw has to resample the days
    of both ranking-share folds and rebuild the endpoint from the resampled MSEs.
    """
    if not all(f in rec for f in SHARES):
        return {}
    num0, num1, cnt = {}, {}, {}
    for f in SHARES:
        v = rec[f]
        codes, uniq = pd.factorize(pd.Series(v["days"]))
        k = len(uniq)
        s0 = np.bincount(codes, weights=v["e0"], minlength=k)
        s1 = np.bincount(codes, weights=v["e1"], minlength=k)
        c = np.bincount(codes, minlength=k).astype("float64")
        dr = rng.integers(0, k, size=(BS.BOOT, k))
        num0[f], num1[f], cnt[f] = s0[dr].sum(1), s1[dr].sum(1), c[dr].sum(1)
    m0 = sum(SHARES[f] * num0[f] / cnt[f] for f in SHARES)
    m1 = sum(SHARES[f] * num1[f] / cnt[f] for f in SHARES)
    g = np.sqrt(m0) - np.sqrt(m1)
    pt = (np.sqrt(sum(SHARES[f] * rec[f]["e0"].mean() for f in SHARES))
          - np.sqrt(sum(SHARES[f] * rec[f]["e1"].mean() for f in SHARES)))
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {"board_gain_s": round(float(pt), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="No,own,own_No")
    ap.add_argument("--sources", default="F,O,H,N,geo")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--learner", default="ridge", choices=["gbm", "ridge"])
    ap.add_argument("--folds", default="july_loo,december")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--head", default="v12", choices=["v11", "v12"],
                    help="the board head the correction is priced against; v12 is the "
                         "head from the desk's 2026-09-08 14:50 UTC message (D20)")
    ap.add_argument("--rows", default="all", choices=["all", "matched", "unmatched"],
                    help="the population the head is FITTED and priced on. The desk's "
                         "16:50 UTC rule: the with-clock head is fitted and priced on "
                         "matched rows only (--rows matched); this lane's serving "
                         "endpoint is --rows unmatched.")
    ap.add_argument("--miss-indicator", action="store_true",
                    help="restore the missingness indicator columns (forbidden by the "
                         "desk's 16:50 UTC rule; for a side-by-side reading only)")
    ap.add_argument("--fit-rows", default=None,
                    choices=["all", "matched", "unmatched"],
                    help="the population the head is FITTED on, when it differs from "
                         "the one it is priced on. `--rows unmatched --fit-rows all` is "
                         "this lane's strongest servable reading: `Nf_*` is defined "
                         "identically on both classes (it needs no observation), so a "
                         "head fitted on every row estimates the same coefficients from "
                         "twice the data and is then priced only where it serves. The "
                         "desk's 16:50 UTC rule constrains the WITH-CLOCK head only.")
    ap.add_argument("--fit-min-receipt", type=float, default=0.0,
                    help="drop FITTING rows on airport-days whose harvest receipt is "
                         "below this, so a 2025 airport with no receiver does not teach "
                         "the no-clock head about the receiver (0 = keep every row)")
    ap.add_argument("--fit-max-instrument-gap", type=float, default=0.0,
                    help="drop FITTING rows at an airport-month whose matcher reach is "
                         "further than this from the ranking file's reach at that "
                         "airport; selects on the instrument, not on the outcome "
                         "(0 = keep every row)")
    ap.add_argument("--fit-months", default="",
                    help="EXTENSION MONTHS added to every fold's FITTING panel "
                         "(RESULTS SS7ca iv, the design's one change). Comma separated, "
                         "e.g. 2025-02,2025-03,2025-04,2025-06,2025-08,2025-10,2025-11. "
                         "Their target is the six-member `base6` residual and their "
                         "block is rebuilt in memory at float64 by `code/extend.py`, "
                         "which checks it against the registered parquet. They are "
                         "NEVER scored: they enter `fit` and nothing else. Empty (the "
                         "default) is v17's panel exactly and takes the same path.")
    ap.add_argument("--assert-cells", action="store_true",
                    help="the PRE-FIT ASSERTION of RESULTS SS7ca ii v: stop before "
                         "pricing anything unless the scored fold shares ZERO cells "
                         "with its fitting rows and every fitted column is one of the "
                         "registered 24. Off by default so an arm outside lever A's "
                         "24-column instrument still runs.")
    ap.add_argument("--amplitude", default="1.0",
                    help="the scale `d` is multiplied by before it is priced. `1.0` (the "
                         "default) is the registered full amplitude and multiplies "
                         "nothing. `oof` estimates it LEAVE-ONE-MONTH-OUT on the "
                         "extension months -- rows no fold is ever scored on -- so no "
                         "part of it is read off the fold it is applied to (RESULTS "
                         "SS7cd). It requires --fit-months.")
    ap.add_argument("--fold-share", type=float, default=0.0,
                    help="RESULTS SS7cf: the share of the FITTING weight the fold rows "
                         "carry when --fit-months adds extension months. 0 (the default) "
                         "weights nothing and every row counts once, which is what every "
                         "run so far did. 0.5 makes the fold rows as a block weigh as much "
                         "as all the extension months together. Total weight is kept at "
                         "the row count so the ridge penalty's relative strength does not "
                         "move.")
    ap.add_argument("--out", default="arms.json")
    a = ap.parse_args()
    if a.fold_share and not a.fit_months:
        raise SystemExit("--fold-share weights the fold rows against the extension "
                         "months and there are none: pass --fit-months too")
    if not 0.0 <= a.fold_share < 1.0:
        raise SystemExit("--fold-share is a share of the weight: 0 <= S < 1")
    if a.amplitude == "oof" and not a.fit_months:
        raise SystemExit("--amplitude oof is estimated on the extension months and "
                         "there are none: pass --fit-months too")
    PCOL = "p11" if a.head == "v11" else "p12"
    RCOL = "r11" if a.head == "v11" else "r12"
    arms = a.arms.split(",")
    srcs = tuple(a.sources.split(","))
    folds = a.folds.split(",")
    rng = np.random.default_rng(BS.SEED)
    W, wrep = rank_weights(a.rows)

    pan = BS.load()
    blocks, covers = {}, {}
    need = sorted({f for s in folds for f in ROTATION[s]} | set(folds))
    for f in need:
        t0 = time.time()
        blk, brep = EX.block_of(f, srcs, a.rule,
                                pred=pd.Series(pan[f][PCOL].to_numpy("float64"),
                                               index=pan[f]["MVT_ID_mvt"].to_numpy("float64")))
        blocks[f] = blk
        covers[f] = EX.covered(f, a.rule) if a.cover else None
        print(f"[block] {f}: {len(blk):,} rows, {round(time.time() - t0, 1)} s, "
              f"summary days {brep.get('summary_days')}", flush=True)

    # THE EXTENSION PANEL (RESULTS SS7ca iv).  It does not depend on which fold is
    # scored -- no extension month is ever scored -- so it is built once and appended to
    # every fold's fitting rows.  `--fit-months ""` skips this block entirely.
    ext, extrep = None, None
    if a.fit_months:
        xmonths = [m.strip() for m in a.fit_months.split(",") if m.strip()]
        xcols = sorted({c for arm in arms
                        for v in arm_columns(arm, blocks[folds[0]]).values()
                        for c in v})
        ext, extrep = XT.panel(xmonths, srcs, a.rule, xcols, rcol=RCOL)
        extrep["columns"] = xcols
        extrep["panel_digest"] = XT.digest(extrep)
        extrep["note"] = (
            "every extension month is a FITTING month only: it is concatenated onto "
            "each fold's fitting rows and appears in no scored population, no cell "
            "table and no board endpoint. Its target is the base6 residual, which is "
            "the same head the block's `*_n_res_*` columns were built against.")
        print(f"[panel] {len(xmonths)} extension months, {len(ext):,} rows, "
              f"digest {extrep['panel_digest'][:12]}", flush=True)

    out = {"rotation": {f: ROTATION[f] for f in folds}, "learner": a.learner,
           "arms": {}, "cover_only": bool(a.cover), "rule": a.rule,
           "rows": a.rows, "fit_rows": a.fit_rows or a.rows,
           "miss_indicator": bool(a.miss_indicator),
           "fit_min_receipt": a.fit_min_receipt,
           "fit_max_instrument_gap": a.fit_max_instrument_gap,
           "projection_weights": wrep,
           "head": a.head, "head_record": pan[folds[0]].attrs.get("v12"),
           "matcher_sha256": EX.BK.sha_of(
               ROOT / "experiments/exp-046-surfacefit/code/block.py"),
           "note": __doc__.split("**The rotation")[1].split("\n\n")[0].strip()}
    # STRICTLY ADDITIVE: at the defaults neither key is written, so a run without the
    # two new options produces the same JSON keys it produced before them.
    if extrep is not None:
        out["fit_months"] = extrep["months_named"]
        out["fit_panel"] = extrep
    if a.assert_cells:
        out["pre_fit_assertion"] = {}
    for arm in arms:
        rec, per = {}, {}
        for f in folds:
            m = pan[f].merge(blocks[f], on="MVT_ID_mvt", how="left",
                             validate="one_to_one")
            if covers[f] is not None:
                m = m.merge(covers[f], on="MVT_ID_mvt", how="left")
                m = m[m.covered.fillna(False)].reset_index(drop=True)
            fits = []
            for g in ROTATION[f]:
                mg = pan[g].merge(blocks[g], on="MVT_ID_mvt", how="left",
                                  validate="one_to_one")
                if covers[g] is not None:
                    mg = mg.merge(covers[g], on="MVT_ID_mvt", how="left")
                    mg = mg[mg.covered.fillna(False)].reset_index(drop=True)
                fits.append(mg)
            fit = pd.concat(fits, ignore_index=True)
            fold_fit_rows = int(len(fit))
            if ext is not None:
                # trim the fold side to the columns a FIT reads before concatenating.
                # A fold fitting frame is the panel plus a 186-column block; the
                # extension frame is 35 columns; the outer join of the two untrimmed is
                # about 2.7 GB of float64 that no line below ever reads.  Nothing is
                # dropped that `keep_rows`, the two filters, the assertion or
                # `arm_predict` can touch.
                keepc = (list(xcols) + [RCOL, PCOL, "y_true", "MVT_ID_mvt", "apt",
                                        "month", "day", "t", "obs_has_clock",
                                        "obs_matched", "receipt_rate", REACH_COL])
                fit = fit[[c for c in dict.fromkeys(keepc) if c in fit.columns]]
                fit = pd.concat([fit, ext[[c for c in fit.columns
                                           if c in ext.columns]]], ignore_index=True)
            m = keep_rows(m, a.rows)
            fit = keep_rows(fit, a.fit_rows or a.rows)
            fit, drep = drop_dark_fit_rows(fit, a.fit_min_receipt)
            fit, irep = drop_far_instrument(fit, a.fit_max_instrument_gap)
            if len(m) < 100 or len(fit) < 100:
                raise SystemExit(f"{arm}/{f}: --rows {a.rows} leaves "
                                 f"{len(m)} scored / {len(fit)} fitting rows")
            cols = sorted({c for v in arm_columns(arm, blocks[f]).values()
                           for c in v})
            miss = [c for c in cols if c not in fit.columns]
            if miss:
                raise SystemExit(f"{arm}/{f}: the fitting months lack {miss[:3]}")
            amp, amprep = 1.0, None
            if a.amplitude == "oof":
                amp, amprep = oof_amplitude(arm, fit, xmonths, RCOL, a.learner,
                                            blocks[f], a.miss_indicator)
            elif float(a.amplitude) != 1.0:
                amp = float(a.amplitude)
                amprep = {"how": "given on the command line", "amplitude": amp}
            if a.assert_cells:
                # RESULTS SS7ca ii v, verbatim: "the arm run asserts, before it prices
                # anything, that the held-out fold shares zero cells with its fitting
                # folds and that every fitted column is one of the registered 24. A run
                # that cannot assert both stops and reports rather than serving."
                out["pre_fit_assertion"][f"{arm}/{f}"] = {
                    "cells": XT.assert_disjoint(m, fit, f),
                    "columns": XT.assert_columns(cols, XT.REGISTERED_24)}
            fw, fwrep = None, None
            if a.fold_share:
                is_ext = fit["month"].astype(str).isin(xmonths).to_numpy()
                n_ext, n_fold = int(is_ext.sum()), int((~is_ext).sum())
                ntot = n_ext + n_fold
                w_fold = a.fold_share * ntot / n_fold
                w_ext = (1.0 - a.fold_share) * ntot / n_ext
                fw = np.where(is_ext, w_ext, w_fold)
                fwrep = {"fold_share": a.fold_share, "rows_fold": n_fold,
                         "rows_extension": n_ext, "weight_per_fold_row": round(w_fold, 4),
                         "weight_per_extension_row": round(w_ext, 4),
                         "total_weight": round(float(fw.sum()), 1),
                         "natural_fold_share": round(n_fold / ntot, 4)}
                print(f"    fold-share {a.fold_share}: fold rows x{w_fold:.3f}, "
                      f"extension rows x{w_ext:.3f} (natural share "
                      f"{n_fold / ntot:.3f})", flush=True)
            d, arep = arm_predict(arm, fit, m, RCOL, a.learner, pool=blocks[f],
                                  miss_indicator=a.miss_indicator, fit_weight=fw)
            if amp != 1.0:
                d = amp * d
            r = m[RCOL].to_numpy("float64")
            y = m["y_true"].to_numpy("float64")
            p11 = m[PCOL].to_numpy("float64")
            days = m["day"].to_numpy()
            nonm = m["nm_missing"].to_numpy(bool)
            hc = has_clock(m)
            rch = reached(m)
            v = {"rows": int(len(m)), "fit_rows": int(len(fit)), "fit_filter": drep,
                 "instrument_filter": irep,
                 "features": len(cols), "form": arep["form"],
                 "all": BS.paired(r, d, days, rng),
                 "clean": BS.paired(r[m["clean"].to_numpy(bool)],
                                    d[m["clean"].to_numpy(bool)],
                                    days[m["clean"].to_numpy(bool)], rng),
                 "matched": BS.paired(r[hc], d[hc], days[hc], rng) if hc.sum() > 100 else None,
                 "unmatched": BS.paired(r[~hc], d[~hc], days[~hc], rng) if (~hc).sum() > 100 else None,
                 "fitted_scale": round(fitted_scale(r, d), 4),
                 "rmse_head": EX.rmse4(y, p11, nonm),
                 "rmse_head_plus_d": EX.rmse4(y, p11 + d, nonm),
                 "by_airport": {}, "by_airport_no_clock": {},
                 "by_airport_reach": {},
                 "reach": {"column": REACH_COL, "rows_reached": int(rch.sum()),
                           "rows_unreached": int((~rch).sum()),
                           "share": round(float(rch.mean()), 4),
                           "by_airport": {}}}
            for apt, idx in m.groupby("apt").groups.items():
                sel = np.zeros(len(m), bool)
                sel[m.index.get_indexer(idx)] = True
                if sel.sum() >= EX.MIN_AIRPORT_ROWS:
                    v["by_airport"][apt] = BS.paired(r[sel], d[sel], days[sel], rng)
                # the desk's 14:10 UTC division: the NO-CLOCK class is this lane's
                # primary endpoint, so it is priced per airport as well as pooled.
                # NO 2,000-ROW FLOOR HERE (FALLBACK round 3, blocking 1): EDDF has
                # 1,604 no-clock rows in december and is the largest contributor to
                # december's projection, so the floor left the one airport the reader
                # most needs an interval for without one.
                nc = sel & ~hc
                if nc.sum() >= 100:
                    v["by_airport_no_clock"][apt] = BS.paired(r[nc], d[nc], days[nc], rng)
                v["reach"]["by_airport"][apt] = round(float(rch[sel].mean()), 4)
                # and the same reading SPLIT BY REACH, which is the grain the
                # projection now works at
                for cls, cm in (("reached", rch), ("unreached", ~rch)):
                    cs = sel & cm
                    if cs.sum() >= MIN_CELL:
                        v["by_airport_reach"][f"{apt}/{cls}"] = BS.paired(
                            r[cs], d[cs], days[cs], rng)
            if ext is not None:
                v["fit_rows_folds_only"] = fold_fit_rows
                v["fit_rows_extension"] = int(len(fit)) - fold_fit_rows
            if amprep is not None:
                v["amplitude"] = amprep
            if fwrep is not None:
                v["fit_weight"] = fwrep
            per[f] = v
            by_apt = {apt: (m["apt"].to_numpy() == apt) for apt in m["apt"].unique()}
            rec[f] = {"days": days, "e0": (y - p11) ** 2, "e1": (y - p11 - d) ** 2,
                      "by_apt": by_apt, "reached": rch}
            nc = v.get("unmatched") or {}
            print(f"  {arm:10s} {f:9s} rows {len(m):6,d} fit {len(fit):6,d} "
                  f"2A-B {v['all']['mse_gain_s2']:+9.1f} {v['all'].get('ci95')} "
                  f"| NO-CLOCK n={nc.get('rows', 0):6,d} "
                  f"{nc.get('mse_gain_s2', float('nan')):+9.1f} {nc.get('ci95')} "
                  f"| scale {v['fitted_scale']}", flush=True)
        per["board"] = board_interval(rec, rng)
        per["projected"] = project(rec, W, rng)
        out["arms"][arm] = per
        if per["board"]:
            print(f"  {arm:10s} BOARD {per['board']}", flush=True)
        for f, pv in per["projected"].items():
            print(f"  {arm:10s} PROJECTED {f:9s} {pv['pooled_mse_reduction_s2']:+9.2f} s2 "
                  f"{pv['ci95_s2']} -> board {pv['board_gain_s']:+.4f} s "
                  f"{pv['board_gain_ci95_s']} | drop {pv['largest_contributor']} -> "
                  f"{pv['leave_one_airport_out_s2'].get(pv['largest_contributor'])} "
                  f"{pv['leave_one_airport_out'].get(pv['largest_contributor'], {}).get('ci95_s2')}"
                  f" | unpriced w {pv['unpriced_weight']}",
                  flush=True)
    (EXP / "analysis" / a.out).write_text(json.dumps(out, indent=2, sort_keys=True,
                                                     default=str))
    print("wrote", EXP / "analysis" / a.out)


if __name__ == "__main__":
    main()
