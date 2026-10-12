"""The ungated chain rows, priced on the folds (desk, 2026-09-09: price the trade).

Section 27 measured the ungated pool on the RANKING side: gate (i) AOBT consistency and
(ii) runway agreement keeps 2,493 rows at an estimated precision of 0.896, and about a
tenth of them are on the wrong flight.  The desk's ruling is that the trade is priced, not
argued -- so this runs the ungated pass on the three FOLD months, refits the head on the
enlarged class, and prices the increment over the current head per fold with intervals,
raw and clean.  **The wrong placements are inside that number**: a row on the wrong flight
carries a fabricated clock into the fit and into the score, and the fold table is what says
whether the real rows pay for them.

One build per fold month serves every threshold.  `analysis/chain-check/block_ungated.py`
admits every unique pair the +-60 s clause refuses as `match_mode` 4 and carries the three
check values out with the block (`chain_gap_s`, `chain_abs_dt_offblock`,
`chain_runway_agrees`); this file decides which of them count, so no threshold is baked into
a cached block and the tightening curve costs fits rather than rebuilds.

**Selection.** Any threshold picked on the folds is a selection, so the pick is cross-fitted
exactly as section 12's capacity step was: for each scored fold the threshold is chosen on
the OTHER TWO and priced on the held-out one, and the in-fold and cross-fitted readings are
reported side by side with the premium between them.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/chainfold.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import prc.data as D                                                  # noqa: E402
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import guard as G                                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "block_ungated", EXP / "analysis/chain-check/block_ungated.py")
BU = importlib.util.module_from_spec(_spec)
sys.modules["block_ungated"] = BU
_spec.loader.exec_module(BU)

ARM = "c_lgb_shipped@leaves_255x300_lr050+guard995"
SETTING = {"num_leaves": 255}
HOLD = ("LFPG",)
HEAD_NPZ = EXP / "work/d-guard-leaves_255x300_lr050-guard995.npz"
#: the shipped mode-3 counts; the ungated pass must not disturb them
MODE3_REFERENCE = {"2025-07": 1170, "2025-12": 861, "2025-01": 731}
#: |P - AOBT| in seconds.  1299 is the shipped p95 of section 27; the rest are the
#: desk's tightening ladder.
THRESHOLDS = [1299.0, 900.0, 600.0, 300.0]
#: step 0, from section 27.1: the p95 of the SHIPPED rows' `t_airborne - obs_wheels_off`.
#: Set on the ranking side and carried here unchanged, so no fold chooses it.
GAP_CUT_S = 62.405
LEVEL = 246.6571
N_RANKING = 344841
OUT = EXP / "analysis/chain-fold.json"
CACHE = EXP / "work"


def build_month(month: str, prefix_months: list[str], rule: str | None = None):
    """The ungated block for one fold month, cached under its own name."""
    r = rule or A.serving_rule()[0]
    p = CACHE / f"block-{month}-{A.block_cache_key(month, r)}-ungated.parquet"
    if p.exists():
        return pd.read_parquet(p)
    dep, _ = D.mask_like_ranking(D.load_month(month))
    arr = dep[dep.PHASE_mvt == "ARR"].copy()
    dp = dep[dep.PHASE_mvt == "DEP"].copy()
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in prefix_months], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    BU.PROBE.clear()
    with BU.pinned(BU.pin_for_month(month)):
        blk = BU.build(dp, tr, require_rule=r, arrivals=arr)
    blk.to_parquet(p, index=False)
    return blk


def restrict(blk: pd.DataFrame, thr: float, gap_cut: float,
             use_runway: bool = True) -> pd.DataFrame:
    """Blank the mode-4 rows the gate refuses, so they read exactly as unmatched.

    A refused row is not "matched but unusable": passes 1 and 2 never claimed it, so the
    honest state is the state the shipped matcher leaves it in.
    """
    b = blk.copy()
    m4 = b.match_mode.to_numpy("float64") == 4.0
    keep = (m4
            & (b.chain_gap_s.to_numpy("float64") > gap_cut)          # step 0
            & (b.chain_abs_dt_offblock.to_numpy("float64") <= thr))   # check (i)
    if use_runway:
        keep &= b.chain_runway_agrees.fillna(0).to_numpy("float64") == 1.0
    drop = m4 & ~keep
    cols = [c for c in (["obs_icao", "obs_wheels_off", *BU.FEATURES]) if c in b.columns]
    b.loc[drop, cols] = np.nan
    b.loc[drop, "obs_matched"] = False
    b.loc[drop, "obs_has_clock"] = False
    b.loc[drop, "reach_class"] = "unmatched"
    return b


def panel_for(blocks: dict, thr: float, gaps: dict) -> dict:
    real = A.month_block

    def mb(month, prefix_months, rule=None):
        return restrict(blocks[month], thr, gaps[month])
    A.month_block = mb
    try:
        return A.panel_with_block(BA.load())
    finally:
        A.month_block = real


def fit_arm(pan: dict, wmap: dict | None = None, prov: dict | None = None,
            frac: float | None = None, sub_seed: int = 0,
            extra: "pd.DataFrame | dict | None" = None,
            extra_share: float | None = None,
            folds: "tuple | None" = None) -> dict:
    """The shipped head.  `wmap` is RESULTS section 47.9's composition weight, one number
    per airport; None is the shipped path and reproduces its bytes.

    **An optional argument that preserves the default is the shape that produced the five
    silent defaults of section 45.3**, so the weighted path is made to PROVE it ran: pass a
    dict as `prov` and every fold records the sha256 of the weight vector actually handed to
    LightGBM, its extremes, its Kish effective sample fraction and how many weighted fits
    consumed it.  A weight vector that is constant -- the shape an unweighted fall-through
    would take -- is FATAL here rather than a quiet +0.00 read downstream.

    `extra_share` (RESULTS section 47.28) down-weights the `extra` rows so they carry that
    FRACTION OF THE TOTAL FITTING WEIGHT in every fold.  The desk specifies a share, so the
    per-fold weight is DERIVED here -- `w = s*n_fold / ((1-s)*n_extra)` -- rather than
    transcribed by hand into three numbers that could each be wrong.  It goes through the
    same proving machinery as `wmap`: sha, Kish, and fatal on a constant vector.

    `extra` (RESULTS section 47.49) is EITHER one frame used by every fold -- the shipped path,
    unchanged -- OR a dict keyed by fold, which is what section 47.46's SEASONAL HEAD needs:
    the july fit carries the months near 2026-07 and the january fit carries the months near
    2026-01, because the served months are antipodal and no one set serves both.  A dict must
    name EVERY fold: a fold missing from it would silently fit with no extension at all, which
    is section 45.3's shape exactly, so it is fatal here.  The provenance records the row count
    each fold actually carried, so a per-fold set cannot be claimed without being shown."""
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in BU.FEATURES if c not in ("stand_lat", "stand_lon")]
    # SECTION 47.58.  `folds` fits a SUBSET.  It exists because 47.57's ruling changes the
    # July head and nothing else: january and december keep v11's month set, so refitting
    # them would spend two thirds of the run reproducing vectors that already exist -- and
    # would reproduce them only to the ULP class (47.50), which is exactly the difference
    # that would make an increment "not quite zero" on a fold whose head did not change.
    # The caller MERGES the folds it did not fit from the members it is reusing, and the
    # provenance names which folds this call actually fitted.
    todo = tuple(A.FOLDS) if folds is None else tuple(folds)
    unknown = [f for f in todo if f not in A.FOLDS]
    if unknown:
        raise SystemExit(f"fit_arm was asked for folds {unknown}, which are not folds")
    d_by_fold = {}
    for f in todo:
        others = [g for g in A.FOLDS if g != f]
        # RESULTS section 47.15's learning curve keeps `frac` of the fitting rows.  The draw
        # is made INSIDE each source fold month, so the two months keep their proportions and
        # the curve measures rows and not a shift in which month dominates.  Like `wmap` this
        # is an optional argument that preserves the default, so it must PROVE it ran: a draw
        # that removes nothing below 1.0 is fatal here, never a quiet full-strength number.
        parts, drawn = [], {}
        for g in others:
            sub = pan[g][pan[g].usable]
            if frac is not None and frac < 1.0:
                rng = np.random.default_rng(sub_seed * 1000 + A.FOLDS.index(g))
                take = int(round(frac * len(sub)))
                sub = sub.iloc[np.sort(rng.choice(len(sub), size=take, replace=False))]
                if len(sub) >= int((pan[g].usable).sum()):
                    raise SystemExit(f"{f}/{g}: the draw at frac={frac} removed nothing -- "
                                     f"this level would price as the full fit while being "
                                     f"labelled a subsample")
            elif frac is not None and len(sub) != int(pan[g].usable.sum()):
                raise SystemExit(f"{f}/{g}: frac={frac} must keep every row, kept "
                                 f"{len(sub)} of {int(pan[g].usable.sum())}")
            drawn[g] = int(len(sub))
            parts.append(sub)
        fit = pd.concat(parts, ignore_index=True)
        if frac is not None and prov is not None:
            prov.setdefault(f, {}).update(
                {"subsample_frac": float(frac), "subsample_seed": int(sub_seed),
                 "fit_rows": int(len(fit)), "fit_rows_by_source_fold": drawn})
        sco = pan[f]
        fit = fit.join(pd.concat([ship[g] for g in others])
                       .reindex(fit.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
        sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                       .reset_index(drop=True))
        # RESULTS section 47.17: extra fitting rows from a month OUTSIDE the panel (2025-11).
        # They are fitted and never scored, so they act only through the shared tree.  Same
        # discipline as `wmap` and `frac`: None is the shipped path, and a frame that adds no
        # rows is FATAL here rather than a quiet "the extension does nothing".
        n_before_extra = len(fit)
        # section 47.49: one frame for every fold, or one frame PER fold
        if isinstance(extra, dict):
            missing = [g for g in A.FOLDS if g not in extra]
            if missing:
                raise SystemExit(
                    f"`extra` is a per-fold dict but names no frame for {missing}. Those folds "
                    f"would fit with NO extension while the run is labelled an extended one -- "
                    f"the section 45.3 shape, and fatal here rather than a quiet read")
            extra_f = extra[f]
            if extra_f is None or not len(extra_f):
                raise SystemExit(f"{f}: the per-fold `extra` frame is empty; see below")
        else:
            extra_f = extra
        if extra_f is not None:
            # Only the columns that actually reach LightGBM are required: the features, the
            # target, the day key the OOF split uses and the airport the amplitude is fitted
            # per.  The panel's bookkeeping columns (p10..p12, clean, regime, r) are never
            # read for a fitting row, so demanding them would refuse a valid extension --
            # but a MISSING FEATURE must still be fatal, because it would be fitted as NaN
            # on the extra rows and the arm would learn "this month is the NaN month".
            used = set(list(obs_cols) + scols) | {"rbase", "day", "airport"}
            need = sorted(c for c in used if c not in extra_f.columns)
            if need:
                raise SystemExit(f"{f}: the extra rows are missing {len(need)} column(s) the "
                                 f"fit USES: {need[:8]} -- they would be fitted as NaN and "
                                 f"the tree would learn the month, not the signal")
            nan_filled = sorted(c for c in fit.columns if c not in extra_f.columns)
            fit = pd.concat([fit, extra_f.reindex(columns=fit.columns)], ignore_index=True)
            if len(fit) <= n_before_extra:
                raise SystemExit(f"{f}: `extra` added {len(fit)-n_before_extra} rows. That is "
                                 f"what an empty extension looks like and it would price as "
                                 f"+0.00 and read as 'the extra month does nothing'.")
            if prov is not None:
                prov.setdefault(f, {}).update(
                    {"extra_rows": int(len(fit) - n_before_extra),
                     "panel_rows": int(n_before_extra), "total_fit_rows": int(len(fit)),
                     "extra_multiple": float(len(fit) / n_before_extra),
                     "columns_nan_filled_and_never_used_by_the_fit": nan_filled,
                     "n_features_checked_present": len(used),
                     "extra_is_per_fold": isinstance(extra, dict),
                     "folds_this_call_fitted": list(todo)})
        w = None
        if wmap is not None:
            w = fit.airport.astype(str).map(wmap).fillna(1.0).to_numpy("float64")
            if not np.isfinite(w).all():
                raise SystemExit(f"{f}: the composition weight is not finite -- the map is "
                                 f"{wmap!r} and the fitting rows carry airports it does "
                                 f"not name")
        if extra_share is not None:
            if extra_f is None:
                raise SystemExit(f"{f}: extra_share={extra_share} was given with no `extra` "
                                 f"rows to weight -- there is nothing to down-weight and the "
                                 f"run would price as an ordinary unweighted fit")
            n_ex = len(fit) - n_before_extra
            if not 0.0 < extra_share < 1.0:
                raise SystemExit(f"extra_share must be in (0,1), got {extra_share}")
            wx = (extra_share * n_before_extra) / ((1.0 - extra_share) * n_ex)
            ew = np.ones(len(fit), dtype="float64")
            ew[n_before_extra:] = wx
            w = ew if w is None else w * ew
        if w is not None:
            w = w / w.mean()                     # normalised to mean 1 on the fitting rows
            if float(w.max() - w.min()) < 1e-9:
                raise SystemExit(
                    f"{f}: the fitting weight is CONSTANT ({float(w[0]):.6f}) on "
                    f"{len(w)} fitting rows. That is what an unweighted fall-through looks "
                    f"like, and it would price as +0.00 and read as 'the arm does nothing'. "
                    f"wmap={wmap!r}, extra_share={extra_share!r}. Fix it -- do NOT let this "
                    f"run report a number.")
            if extra_share is not None:
                got = float(w[n_before_extra:].sum() / w.sum())
                if abs(got - extra_share) > 1e-9:
                    raise SystemExit(
                        f"{f}: the extension carries {got:.6f} of the fitting weight, not the "
                        f"{extra_share} asked for -- the derivation is wrong and every number "
                        f"downstream would be at an unknown weighting")
                if prov is not None:
                    prov.setdefault(f, {}).update({
                        "extra_share_asked": float(extra_share),
                        "extra_share_measured": got,
                        "extra_row_weight_asked": float(wx),
                        "extra_row_weight": float(w[-1]),
                        "fold_row_weight": float(w[0]),
                        "weight_ratio_extra_over_fold": float(w[-1] / w[0])})
        d, info = A.arm_lgb(fit, sco, list(obs_cols) + scols, params=SETTING, w=w)
        if w is not None:
            if not info.get("sample_weighted") or not info.get("n_weighted_lgb_fits"):
                raise SystemExit(f"{f}: weights were built but arm_lgb reports "
                                 f"sample_weighted={info.get('sample_weighted')!r} and "
                                 f"{info.get('n_weighted_lgb_fits')!r} weighted fits -- the "
                                 f"weight did not reach LightGBM")
            if prov is not None:
                prov.setdefault(f, {}).update({
                    "rows": int(len(w)),
                    "weight_sha256": hashlib.sha256(
                        np.ascontiguousarray(w, dtype="float64").tobytes()).hexdigest(),
                    "min": float(w.min()), "max": float(w.max()),
                    "kish_effective_fraction": float(w.sum() ** 2
                                                     / (len(w) * float((w ** 2).sum()))),
                    "n_weighted_lgb_fits": int(info["n_weighted_lgb_fits"]),
                    "distinct_weights": int(np.unique(np.round(w, 12)).size),
                })
        d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
        assert not np.any(d[~pan[f].usable.to_numpy(bool)]), f"{f}: a row outside the class moved"
        d_by_fold[f] = d
    # SECTION 47.59.6.  These two loops walked A.FOLDS while the FITTING loop above walks
    # `todo`, so a subset call raised KeyError on the first unfitted fold.  47.58 added the
    # subset and nothing exercised it until now -- the seasonal5 stack that motivated it is
    # registered and unrun.  The guard cut is unchanged: `G.cut(pan, others)` reads the OTHER
    # folds' panels, which are built for every fold whether or not that fold is fitted, so a
    # subset fold's cut is identical to the one a full run would give it.
    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in todo}
    out = {}
    for f in todo:
        v = d_by_fold[f].astype("float64").copy()
        v[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, cuts[f])] = 0.0
        v[np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))] = 0.0
        out[f] = v
    return out


def price_increment(pan: dict, d_new: dict, d_head: dict, rng) -> dict:
    """The increment over the head: d_new - d_head against a base of pbase + d_head."""
    ph, inc = {}, {}
    for f in A.FOLDS:
        p = pan[f].copy()
        p["rbase"] = p.rbase.to_numpy("float64") - d_head[f]
        p["pbase"] = p.pbase.to_numpy("float64") + d_head[f]
        ph[f] = p
        inc[f] = d_new[f] - d_head[f]
    rec = BA.price(ph, inc, rng)
    return {"board_all_row": rec["board_all_row"], "board_clean": rec["board_clean"],
            "every_fold_interval_clear": bool(rec["every_fold_interval_clear"]),
            "every_moving_fold_interval_clear": bool(rec["every_moving_fold_interval_clear"]),
            "folds_with_no_moved_row": rec["folds_with_no_moved_row"],
            "by_fold": {f: rec["by_fold"][f]["all_row"] for f in A.FOLDS},
            "by_fold_clean": {f: rec["by_fold"][f]["clean"] for f in A.FOLDS},
            "rows_moved": {f: int((inc[f] != 0).sum()) for f in A.FOLDS},
            "tail": {f: A.tail_clauses(pan[f], d_new[f])["registered_clause_y_over_3600s"]
                     for f in A.FOLDS}}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(A.SEED)
    rule = A.serving_rule()[0]
    out = {"what": __doc__.strip().splitlines()[0], "arm": ARM, "thresholds": THRESHOLDS,
           "probe": "analysis/chain-check/block_ungated.py", "hold": list(HOLD)}

    blocks, gaps, census = {}, {}, {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = build_month(m, A.prefix_months_for(m), rule)
        n3 = int((b.match_mode.fillna(-1) == 3).sum())
        if n3 != MODE3_REFERENCE[m]:
            raise SystemExit(f"{m}: mode-3 rows {n3} != {MODE3_REFERENCE[m]} -- the "
                             "ungated pass disturbed the shipped placements")
        # STEP 0's threshold is GAP_CUT_S, the p95 of `t_airborne - obs_wheels_off` on
        # the SHIPPED ranking rows (section 27.1).  Taking it there and not here is the
        # point: the fold months never see it chosen, so it cannot be a fold selection.
        gaps[m] = GAP_CUT_S
        blocks[m] = b
        n4 = int((b.match_mode.fillna(-1) == 4).sum())
        keep = ((b.match_mode.fillna(-1) == 4)
                & (b.chain_gap_s > gaps[m])
                & (b.chain_abs_dt_offblock <= THRESHOLDS[0])
                & (b.chain_runway_agrees == 1.0)).sum()
        census[m] = {"mode3": n3, "mode4_admitted": n4,
                     "kept_at_the_loosest_gate": int(keep)}
        print(f"{m}: mode3 {n3} (reference ok), mode4 admitted {n4}, "
              f"kept at 1299 s {int(keep)}  ({time.time()-t0:.0f}s)", flush=True)
    out["census_by_month"] = census
    out["step_0_gap_cut_s"] = gaps
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    zh = np.load(HEAD_NPZ)
    d_head = {f: zh[f].astype("float64") for f in A.FOLDS}
    out["head_npz"] = str(HEAD_NPZ.relative_to(ROOT))

    out["by_threshold"] = {}
    for thr in THRESHOLDS:
        pan = panel_for(blocks, thr, gaps)
        added = {f: int(((pan[f].match_mode.fillna(-1) == 4)
                         & pan[f].usable.to_numpy(bool)).sum()) for f in A.FOLDS}
        d_new = fit_arm(pan)
        rec = price_increment(pan, d_new, d_head, np.random.default_rng(A.SEED))
        rec["rows_added_to_the_class"] = added
        out["by_threshold"][str(int(thr))] = rec
        ci = rec["board_all_row"]["interval"]["ci95"]
        print(f"\nthreshold {int(thr):>4} s  added {added}  "
              f"increment {rec['board_all_row']['gain_s']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
              f"  clean {rec['board_clean']['gain_s']:+.4f}  every-fold-clear "
              f"{rec['every_fold_interval_clear']}", flush=True)
        for f in A.FOLDS:
            r = rec["by_fold"][f]
            print(f"   {f:10s} moved {rec['rows_moved'][f]:>6} s2/row {r['mse_gain_s2']:>+9.2f}"
                  f"  ci [{r['ci95'][0]:>+8.2f},{r['ci95'][1]:>+8.2f}]", flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    print("written to", OUT.relative_to(ROOT), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
