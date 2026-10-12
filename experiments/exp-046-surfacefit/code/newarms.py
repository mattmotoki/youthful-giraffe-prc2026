"""Arms over `c_lgb_shipped`: the observation's own next features, and the `_nbr` repair.

The desk, 2026-09-08 18:10 UTC, fold side only, nothing served.  Every arm here is
`c_lgb_shipped` -- the served head -- with one group of columns added, on the SAME
rotation, the same seed, the same cross-fit day blocks, the same per-airport amplitude,
the same floor and the same clauses.  The predeclared LightGBM is unchanged; capacity is
`code/capacity.py`'s question and is not mixed into this one.

    c_geom            all of `geom.GEOM_FEATURES`
    c_geom_rwy        the wheels-off geometry alone (which runway end, from where)
    c_geom_chord      the taxi chord and the speed it implies
    c_geom_motion     the three start-of-motion gaps and the run's own quality flags
    c_nbr             the twelve `No_*` neighbour columns (the arm that fails july's tail)
    c_nbr_<repair>    the same columns under a LABEL-FREE restriction, `code/nbrtail.py`

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/newarms.py [arm ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402
import geom as G                                                      # noqa: E402

RWY = ["rwy_lat", "rwy_lon", "rwy_from_apt_km", "rwy_from_apt_sin", "rwy_from_apt_cos",
       "rwy_gs_kt", "rwy_airborne_gap_s", "rwy_air_alt_ft"]
CHORD = ["taxi_chord_km", "taxi_chord_speed_kt", "taxi_chord_bearing_sin",
         "taxi_chord_bearing_cos", "stand_from_apt_km"]
MOTION = ["acquired_before_motion_s", "motion_to_disp_s", "offblock_strict_minus_disp_s",
          "offblock_loose_minus_disp_s", "seg_starts_fast", "seg_ends_stationary",
          "seg_ends_airborne", "seg_airborne_alt_ok"]
#: ABSOLUTE position is identity; RELATIVE position is geometry.  This lane already
#: settled that once: `block.py` computes `stand_lat`/`stand_lon` and every fitting site
#: drops them, keeping the airport-relative `stand_dist_km`/`stand_sin`/`stand_cos`
#: instead (`arms.py:782` and four other files).  `geom.py` carries `rwy_lat`/`rwy_lon`
#: for diagnosis and the same rule applies to them: a raw coordinate lets the model name
#: the airport and the runway end, which is exactly the identity a fold cannot transfer.
#: The `_rel` arms are the same columns with the two raw coordinates removed.
ABSOLUTE = G.ABSOLUTE_COORDINATES        # the lane rule lives in geom.py
GEOM_REL = G.SERVED_GEOM
RWY_REL = [c for c in RWY if c not in ABSOLUTE]

GROUPS = {"c_geom": G.GEOM_FEATURES, "c_geom_rwy": RWY, "c_geom_chord": CHORD,
          "c_geom_rel": GEOM_REL, "c_geom_rwy_rel": RWY_REL,
          "c_geom_motion": MOTION, "c_nbr": A.NBR_COLS,
          "c_nbr_centred": A.NBR_COLS, "c_nbr_winsor": A.NBR_COLS,
          "c_geom_nbr_centred": G.GEOM_FEATURES + A.NBR_COLS}
DEFAULT = ["c_geom", "c_geom_rwy", "c_geom_chord", "c_geom_motion"]

#: The `No_*` columns the july tail failure runs through (`analysis/nbr-tail.json`).
#: `No_n_res_*` and `No_n_anom_*` carry an AIRPORT-LEVEL offset: LIRF's whole with-clock
#: population sits at a neighbour residual of about -860 s where every other airport sits
#: at about -380, so a model fitted mostly elsewhere reads LIRF's ordinary rows as extreme
#: and answers with a large negative correction.  159 of the 173 damaged july tail rows
#: are LIRF's.  The `No_n_n_*` counts show no separation at all and are left alone.
NBR_LEVEL = [c for c in A.NBR_COLS if c.startswith(("No_n_res", "No_n_anom"))]


def centre_by_airport(fit: pd.DataFrame, sco: pd.DataFrame,
                      cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Subtract each airport's own median, learned on the FITTING rows alone.

    Label-free and out of fold: the median is a statistic of the neighbour column, not of
    the target, and it is computed on the two fitting months and applied to the scored one.
    An airport absent from the fitting rows keeps its raw value and is reported.
    """
    for c in [x for x in NBR_LEVEL if x in fit.columns]:
        med = fit.groupby("airport", observed=True)[c].median()
        fit[c] = fit[c] - fit.airport.map(med).astype("float64")
        sco[c] = sco[c] - sco.airport.map(med).astype("float64")
    return fit, sco, cols


def winsorise(fit: pd.DataFrame, sco: pd.DataFrame,
              cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Clip each `No_*` column at the fitting rows' 1st and 99th percentile.

    The alternative repair: keep the airport offset but refuse the extremes the model has
    no support for.  Also label-free and out of fold.
    """
    for c in [x for x in A.NBR_COLS if x in fit.columns]:
        lo, hi = np.nanpercentile(fit[c].to_numpy("float64"), [1, 99])
        fit[c] = fit[c].clip(lo, hi)
        sco[c] = sco[c].clip(lo, hi)
    return fit, sco, cols


TRANSFORM = {"c_nbr_centred": centre_by_airport, "c_nbr_winsor": winsorise,
             "c_geom_nbr_centred": centre_by_airport}
OUT = EXP / "analysis/newarms.json"


def panel() -> tuple[dict, dict, list[str], list[str]]:
    pan = A.panel_with_block(BA.load())
    for f in A.FOLDS:
        g = G.month_geom(A.FOLD_MONTH[f])
        pan[f] = pan[f].merge(g, on="MVT_ID_mvt", how="left", validate="one_to_one")
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    return pan, ship, scols, obs_cols


def price_arm(name: str, extra: list[str], pan: dict, ship: dict, scols: list[str],
              obs_cols: list[str], rng) -> dict:
    t0 = time.time()
    d_by_fold, detail = {}, {}
    for f in A.FOLDS:
        others = [g for g in A.FOLDS if g != f]
        fit = pd.concat([pan[g][pan[g].usable] for g in others], ignore_index=True)
        sco = pan[f]
        fit = fit.join(pd.concat([ship[g] for g in others])
                       .reindex(fit.MVT_ID_mvt.to_numpy("float64"))
                       .reset_index(drop=True))
        sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                       .reset_index(drop=True))
        cols = list(obs_cols) + [c for c in extra if c in fit.columns] + scols
        tf = TRANSFORM.get(name)
        if tf is not None:
            fit, sco, cols = tf(fit.copy(), sco.copy(), cols)
        d, info = A.arm_lgb(fit, sco, cols)
        d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
        assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{name} {f}: a noac row moved"
        assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
            f"{name} {f}: a row outside the with-clock class moved"
        d_by_fold[f] = d
        detail[f] = {k: info[k] for k in ("weights", "n_features", "oof_rmse_vs_r11",
                                          "r11_rmse_on_fit_rows")}
    rec = BA.price(pan, d_by_fold, rng)
    rec["detail"] = detail
    rec["columns_added"] = [c for c in extra if c in pan[A.FOLDS[0]].columns]
    rec["transform"] = (TRANSFORM[name].__name__ if name in TRANSFORM else None)
    rec["transform_applies_to"] = (NBR_LEVEL if name in ("c_nbr_centred",
                                                         "c_geom_nbr_centred")
                                   else A.NBR_COLS if name == "c_nbr_winsor" else [])
    rec["columns_asked_for_but_absent"] = [c for c in extra
                                           if c not in pan[A.FOLDS[0]].columns]
    rec["tail_clauses"] = {f: A.tail_clauses(pan[f], d_by_fold[f]) for f in A.FOLDS}
    rec["by_airport_with_clock"] = {
        f: A.by_group(pan[f][pan[f].usable.to_numpy(bool)],
                      d_by_fold[f][pan[f].usable.to_numpy(bool)], "airport", rng)
        for f in A.FOLDS}
    rec["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(EXP / f"work/d-{name}.npz", **d_by_fold)
    return rec


def line(name: str, rec: dict) -> str:
    tc = rec["tail_clauses"]
    return (f"{name:16s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
            f"[{rec['board_all_row']['interval']['ci95'][0]:+7.3f},"
            f"{rec['board_all_row']['interval']['ci95'][1]:+7.3f}] "
            f"clean {rec['board_clean']['gain_s']:+8.4f} "
            f"dec {rec['december_rmse']['gain_s']:+8.4f} "
            f"clear {str(rec['every_fold_interval_clear']):5s} tail "
            + " ".join(
                f"{f[:3]} {tc[f]['registered_clause_y_over_3600s']['rmse_gain_s']:+7.3f}"
                for f in A.FOLDS)
            + ("  ALL PASS" if all(
                tc[f]["registered_clause_y_over_3600s"]["passes_the_clause"]
                for f in A.FOLDS) else "  TAIL FAILS")
            + f"  {rec['seconds']:.0f}s")


def main() -> None:
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT
    rng = np.random.default_rng(A.SEED)
    pan, ship, scols, obs_cols = panel()
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "what": ("arms over c_lgb_shipped on the same rotation and clauses; nothing "
                 "here is served"),
        "reference": "c_lgb_shipped +8.1410 all-row, +9.8561 clean, +15.4488 december",
        "arms": {}}
    rule, _ = A.serving_rule()
    out["rule"] = {"sha256": rule,
                   "block_cache_keys": {m: A.block_cache_key(m, rule)
                                        for m in A.FOLD_MONTH.values()}}
    out["geom_version"] = G.VERSION
    for name in want:
        extra = GROUPS.get(name)
        if extra is None:
            raise SystemExit(f"no such arm {name!r}; known: {sorted(GROUPS)}")
        rec = price_arm(name, extra, pan, ship, scols, obs_cols, rng)
        out["arms"][name] = rec
        print(line(name, rec), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
