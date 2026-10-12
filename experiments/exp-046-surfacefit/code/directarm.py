"""Item (5): the label modelled DIRECTLY, instead of the base's residual.

Desk, 2026-09-09: *"the fitting slot is free: run (5), the direct label model on the
observation, as one arm on the three folds under the ordinary gate -- a number."*

Every arm this lane has priced -- `a_linear` through `c_lgb_shipped_nbr` -- corrects the
RESIDUAL of a price base it inherits from `exp-047-queue`.  This one predicts the label
from the observation and the row, and the update is what that prediction implies:

    residual form   d = s_air * lgb(features -> y - pbase)
    direct form     d = s_air * (lgb(features -> y) - pbase)

**What it gives up, and it is deliberate.** The residual form inherits the other lane's
head for free: whatever v14's neighbourhood update and v17's lit rule bought on this lane's
rows is already inside `pbase` and the residual only has to correct what is left.  The
direct form throws that away and re-earns it from the observation.  There is weak prior
evidence against it in the arm table already -- `a_linear` (+2.22) and `b_lgb` (+4.19) both
work WITHOUT the shipped base and both price far below `c_lgb_shipped` (+8.10) -- which
says the base carries more than half the gain.  This prices the question instead of
arguing it.

**Matched rotation.** Everything except the target is held: the same three folds, the same
fit-on-the-other-two rotation, the same seed, the same five contiguous day blocks for the
cross-fit, the same per-airport amplitude, the same features, the same floor, the same
day-clustered intervals, the same registered tail clauses.  `pbase` is NOT among the
features -- a direct model that reads the base is the residual model wearing another name,
and the second variant below says what that costs.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-046-surfacefit/code/directarm.py
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

import serve as S                                                     # noqa: E402,F401
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402

OUT = EXP / "analysis/direct-arm.json"
HEAD = "leaves_127x300_lr050"          # the capacity that ships, so the comparison is fair
PARAMS = {"num_leaves": 127, "n_estimators": 300, "learning_rate": 0.05}


def direct(fit: pd.DataFrame, sco: pd.DataFrame, cols: list[str],
           with_pbase: bool) -> tuple[np.ndarray, dict]:
    """`arm_lgb`, with the LABEL as the target and `pbase` subtracted at the end.

    The cross-fitted amplitude is fitted on out-of-fold predictions exactly as the
    residual arm's is, so the two differ in the target and in nothing else.
    """
    c = list(cols) + (["pbase"] if with_pbase else [])
    Xf = fit[c].to_numpy("float32")
    yf = fit.y_true.to_numpy("float64")
    pf = fit.pbase.to_numpy("float64")
    blk = A._dayblocks(fit.day)
    oof = np.zeros(len(fit))
    for b in np.unique(blk):
        tr = blk != b
        if tr.sum() < 100 or (~tr).sum() == 0:
            continue
        oof[~tr] = A._lgb_fit(Xf[tr], yf[tr], A.SEED, PARAMS).predict(Xf[~tr])
    fitc = fit.copy()
    fitc["d_raw"] = oof - pf                      # the update the direct model implies
    w, n = A._per_airport_amplitude(fitc, "d_raw")
    full = A._lgb_fit(Xf, yf, A.SEED, PARAMS)
    d = np.zeros(len(sco))
    m = sco.usable.to_numpy(bool)
    if m.any():
        pred = full.predict(sco.loc[m, c].to_numpy("float32"))
        raw = pred - sco.loc[m, "pbase"].to_numpy("float64")
        wv = sco.loc[m, "airport"].map(w).fillna(0.0).to_numpy("float64")
        d[m] = np.nan_to_num(raw * wv, nan=0.0)
    return d, {"weights": w, "fit_rows_by_airport": n, "n_features": len(c),
               "oof_rmse_vs_label": float(np.sqrt(np.mean((yf - oof) ** 2))),
               "base_rmse_on_fit_rows": float(np.sqrt(np.mean((yf - pf) ** 2)))}


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    rule, _ = A.serving_rule()
    out = {"what": ("the label modelled directly instead of the base's residual, on the "
                    "same rotation, seed, cross-fit, amplitude, floor and clauses"),
           "ordered_by": "the desk, 2026-09-09, next-step table item (5)",
           "head_for_comparison": HEAD, "lgb": PARAMS,
           "matcher_sha256": A.MATCHER_SHA,
           "rule": {"sha256": rule,
                    "block_cache_keys": {m: A.block_cache_key(m, rule)
                                         for m in A.FOLD_MONTH.values()}},
           "arms": {}}

    for name, with_pbase in (("d_direct", False), ("d_direct_with_pbase", True)):
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
            d, info = direct(fit, sco, list(obs_cols) + scols, with_pbase)
            d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{name} {f}: a noac row moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{name} {f}: a row outside the with-clock class moved"
            d_by_fold[f] = d
            detail[f] = info
        rec = BA.price(pan, d_by_fold, rng)
        rec["detail"] = {f: {k: v[k] for k in ("weights", "n_features",
                                               "oof_rmse_vs_label",
                                               "base_rmse_on_fit_rows")}
                         for f, v in detail.items()}
        rec["tail_clauses"] = {f: A.tail_clauses(pan[f], d_by_fold[f]) for f in A.FOLDS}
        rec["seconds"] = round(time.time() - t0, 1)
        out["arms"][name] = rec
        np.savez_compressed(A.WORK / f"d-{name}.npz", **{f: d_by_fold[f] for f in A.FOLDS})
        ci = rec["board_all_row"]["interval"]["ci95"]
        print(f"{name:22s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{ci[0]:+7.4f},{ci[1]:+7.4f}]  clean "
              f"{rec['board_clean']['gain_s']:+8.4f}  dec "
              f"{rec['december_rmse']['gain_s']:+8.4f}  every fold clear "
              f"{rec['every_fold_interval_clear']}  {rec['seconds']:.0f}s", flush=True)

    cap = json.loads((EXP / "analysis/capacity.json").read_text())["settings"][HEAD]
    out["the_residual_arm_at_the_same_capacity"] = {
        "setting": HEAD, "board_all_row_gain_s": cap["board_all_row"]["gain_s"],
        "ci95": cap["board_all_row"]["interval"]["ci95"]}
    for n, r in out["arms"].items():
        out["arms"][n]["against_the_residual_arm_s"] = (
            r["board_all_row"]["gain_s"] - cap["board_all_row"]["gain_s"])
    print(f"\nthe residual arm at {HEAD}: "
          f"{cap['board_all_row']['gain_s']:+.4f}")
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written", OUT)


if __name__ == "__main__":
    main()
