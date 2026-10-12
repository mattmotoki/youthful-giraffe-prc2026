"""exp-046-surfacefit: a machinery check on the arms, before the fold months land.

**This is not a measurement and no number here enters the ledger.**  The harvest has
delivered one airport-week (EHAM, 2025-07-07 to 07-13) and two gate days of 2026-01,
so the rotation of `code/arms.py` has nothing to fit on: every arm reads +0.0000.
This script exercises the same three arm functions on the only data that exists — a
day split inside the pilot's EHAM week, fit on days 1-4 and score days 5-7 — so that
the fits, the cross-fitting, the per-airport amplitude and the two structural checks
are proved to run and to produce sane numbers before the real table is asked for.

What it can say: whether an arm runs, whether its amplitude is in a plausible band,
and how the three compare on one airport with 2,700 fitting rows.  What it cannot
say: anything about the other nine airports, about december or january, or about the
board.  Three days of one airport is a smoke test, not evidence.

Two further checks ride along, both on the same week (desk, 2026-09-08 13:40 and 13:56):

`handover` joins `exp-047-queue`'s interface block for the month and checks that the two
lanes cut the month at the same place -- same rows, complementary classes -- rather than
trusting that they do because that lane imports this one's matcher.

`cross_class` is the counterfactual this lane does NOT serve: one head fitted and served
across BOTH reach classes, priced on the two populations apart.  exp-047 measured
+6,884 s2 on the rows with a clock and -588 s2 on the rows without one; this reproduces
the shape on this lane's own instrument, so the boundary is a measured choice here and
not an inherited one.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import arms as A                                             # noqa: E402
import base as BA                                            # noqa: E402
import block as B                                            # noqa: E402

FIT_DAYS = ["2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10"]
SCO_DAYS = ["2025-07-11", "2025-07-12", "2025-07-13"]


def report(p: pd.DataFrame, d: np.ndarray) -> dict:
    r = p.rbase.to_numpy("float64")
    v = 2 * r * d - d * d
    m = d != 0
    return {"rows": int(len(p)), "rows_moved": int(m.sum()),
            "mse_gain_s2_all_rows": float(np.mean(v)),
            "mse_gain_s2_per_moved_row": float(np.mean(v[m])) if m.any() else 0.0,
            "rmse_base_moved": float(np.sqrt(np.mean(r[m] ** 2))) if m.any() else None,
            "rmse_arm_moved": float(np.sqrt(np.mean((r[m] - d[m]) ** 2))) if m.any() else None,
            "rms_update_s": float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 0.0}


IFACE = ROOT / "artifacts/exp-047/block"


def handover(pan: pd.DataFrame, month: str = "2025-07") -> dict:
    """Do the two lanes cut the month at the same place?

    `exp-047-queue` writes `reach_class` from THIS lane's matcher into its interface
    block, so the check is an equality on every row, not a comparison of totals.  The
    one difference allowed is this lane's `noac` hold, which is applied in the panel and
    not in `block.py`, so a `noac` row can carry a class there and `noac_hold` here.
    """
    f = IFACE / f"{month}.parquet"
    if not f.exists():
        return {"skipped": f"{f.relative_to(ROOT)} does not exist yet"}
    o = pd.read_parquet(f, columns=["MVT_ID_mvt", "reach_class", "obs_has_clock"])
    mine = pan[["MVT_ID_mvt", "reach_class", "usable", "noac"]].copy()
    j = mine.merge(o, on="MVT_ID_mvt", how="outer", indicator=True,
                   suffixes=("_mine", "_047"), validate="one_to_one")
    both = j._merge.eq("both")
    noac = j.noac.fillna(False).astype(bool)
    agree = j.reach_class_mine.eq(j.reach_class_047)
    theirs_clock = j.obs_has_clock.fillna(False).astype(bool)
    mine_serves = j.usable.fillna(False).astype(bool)
    return {
        "file": str(f.relative_to(ROOT)),
        "rows_mine": int(len(mine)), "rows_047": int(len(o)),
        "rows_in_both": int(both.sum()),
        "rows_only_mine": int(j._merge.eq("left_only").sum()),
        "rows_only_047": int(j._merge.eq("right_only").sum()),
        "reach_class_agrees": int((both & agree).sum()),
        "reach_class_differs": int((both & ~agree).sum()),
        "reach_class_differs_on_a_noac_row": int((both & ~agree & noac).sum()),
        "reach_class_differs_elsewhere": int((both & ~agree & ~noac).sum()),
        "i_serve": int(mine_serves.sum()),
        "they_carry_a_clock": int(theirs_clock.sum()),
        "i_serve_but_they_call_it_no_clock": int((mine_serves & ~theirs_clock).sum()),
        "they_carry_a_clock_but_i_hold_it": int((~mine_serves & theirs_clock).sum()),
        "of_those_noac": int((~mine_serves & theirs_clock & noac).sum()),
        "left_to_exp_047": int((~mine_serves).sum()),
        "note": ("the two populations are complementary by construction: exp-047 imports "
                 "this lane's code/block.py as the matcher and keys its class on the same "
                 "obs_has_clock column. The noac hold is this lane's alone, so those rows "
                 "must stay held in any composition of the two candidates."),
    }


def cross_class(fit: pd.DataFrame, sco: pd.DataFrame, cols: list[str]) -> dict:
    """One head across both classes, priced on the two populations apart.

    The counterfactual, not the design.  `arm_lgb` serves the rows its scored frame
    marks `usable`, so widening that mask to every non-`noac` row of the harvested
    airport is exactly "a head that spans both classes"; the observation features are
    missing on the no-clock rows and LightGBM carries them as missing.
    """
    air = fit.airport.iloc[0] if len(fit) else None
    f2 = fit[fit.airport.eq(air)].copy()
    s2 = sco[sco.airport.eq(air) & ~sco.noac.astype(bool)].copy()
    if not len(f2) or not len(s2):
        return {"skipped": "no harvested airport in the smoke split"}
    s2["usable"] = True                                   # the widened serving mask
    d, info = A.arm_lgb(f2, s2, cols)
    d = A.apply_floor(s2.pbase.to_numpy("float64"), d)
    r = s2.rbase.to_numpy("float64")
    v = 2 * r * d - d * d
    have = A.served_class(s2)
    out = {"airport": air, "fit_rows": int(len(f2)), "scored_rows": int(len(s2)),
           "weights": info["weights"], "by_class": {}}
    for name, m in (("with_a_clock", have), ("no_clock_of_its_own", ~have)):
        out["by_class"][name] = {
            "rows": int(m.sum()),
            "rows_moved": int((m & (d != 0)).sum()),
            "mse_gain_s2_per_row": float(np.mean(v[m])) if m.any() else 0.0,
            "rms_update_s": (float(np.sqrt(np.mean(d[m & (d != 0)] ** 2)))
                             if (m & (d != 0)).any() else 0.0)}
    out["what_this_lane_serves_instead"] = (
        "0 s2 on the no-clock rows: they stay at the base and are exp-047-queue's, whose "
        "neighbourhood block reads +583 s2 on them (exp-047 RESULTS.md \u00a70)")
    out["exp_047_reference"] = {"with_a_clock_s2": 6884, "no_clock_s2": -588,
                                "source": "exp-047-queue/RESULTS.md step 1"}
    return out


def main() -> None:
    # the pilot week's own rule version, not the serving one: these seven days are
    # re-derived on this laptop from the pilot's stored points and carry a producer
    # identity the harvest's kernels cannot match, so the serving resolver rejects them.
    # A machinery check may say so out loud; the fold table and the serve may not.
    week_rule, _ = B.one_rule([(["EHAM"], FIT_DAYS + SCO_DAYS)])
    pan = A.panel_with_block(BA.load(), rule=week_rule)["july_loo"]
    fit = pan[pan.usable & pan.day.isin(FIT_DAYS)].reset_index(drop=True)
    sco = pan[pan.day.isin(SCO_DAYS)].reset_index(drop=True)
    if not len(fit) or not sco.usable.any():
        rep = {"skipped": True, "rule_required": week_rule,
               "fit_rows": int(len(fit)), "scored_usable": int(sco.usable.sum()),
               "why": ("no row of the pilot week carries an observation under the rule "
                       "version this check resolved, so there is no machinery to check")}
        (EXP / "analysis/smoke.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        print(json.dumps(rep, indent=2, sort_keys=True))
        return
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    ship = None
    try:
        ship, scols = A.shipped_cols({f: pan for f in A.FOLDS})
    except Exception as e:                                    # noqa: BLE001
        print(f"[warn] shipped block unavailable: {e}")
        scols = []

    rep = {"design": {"fit_days": FIT_DAYS, "score_days": SCO_DAYS,
                      "rule_required": week_rule,
                      "airport": "EHAM (the only airport harvested)",
                      "fit_rows": int(len(fit)),
                      "scored_rows": int(len(sco)),
                      "scored_usable": int(sco.usable.sum())},
           "warning": ("A machinery check on one airport-week. No number here is a "
                       "measurement and none enters the ledger. Arm c is additionally "
                       "flattered here: the shipped feature block is cached per month, "
                       "so a WITHIN-July split lets the fit and the scored rows share a "
                       "month, which the real rotation of code/arms.py forbids. The real "
                       "table fits on the other two fold months."),
           "arms": {}}

    d, info = A.arm_a(fit, sco, ["airport"])
    rep["arms"]["a_linear"] = report(sco, d) | {"info": info}

    d, info = A.arm_a(fit, sco, ["airport", "obs_wide"])
    rep["arms"]["a_linear_reach"] = report(sco, d) | {"info": info}

    d, info = A.arm_lgb(fit, sco, obs_cols)
    rep["arms"]["b_lgb"] = report(sco, d) | {"info": info}

    if ship is not None and scols:
        W = ship["july_loo"]
        f2 = fit.join(W.reindex(fit.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
        s2 = sco.join(W.reindex(sco.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
        d, info = A.arm_lgb(f2, s2, obs_cols + scols)
        rep["arms"]["c_lgb_shipped"] = report(sco, d) | {"info": info}

    # the fitting frame for the counterfactual is EVERY row of the fit days, not only
    # the usable ones, because a head that spans both classes fits across both too
    fit_all = pan[pan.day.isin(FIT_DAYS) & pan.airport.eq(fit.airport.iloc[0])
                  & ~pan.noac.astype(bool)].reset_index(drop=True)
    rep["cross_class"] = cross_class(fit_all, sco, obs_cols)
    rep["handover"] = handover(pan)

    for name, rec in rep["arms"].items():
        assert rec["rows_moved"] <= rep["design"]["scored_usable"], name
        print(f"{name}: moved {rec['rows_moved']} rows, "
              f"{rec['rmse_base_moved']:.1f} -> {rec['rmse_arm_moved']:.1f} s on them, "
              f"gain {rec['mse_gain_s2_per_moved_row']:+.0f} s2/moved row")

    (EXP / "analysis/smoke.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                        default=float))


if __name__ == "__main__":
    main()
