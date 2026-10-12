"""Arm (c) at other capacities, on the same rotation and the same clauses.

The desk, 2026-09-08 18:10 UTC: "capacity of the residual model (leaves, rounds, learning
rate) at matched rotation".  Nothing here is served; this is the fold side asking whether
the predeclared LightGBM is the right size for the correction, or whether it is leaving
gain on the table by being too small (or buying its gain with variance by being too large).

**Matched rotation** means everything except the three capacity numbers is held: the same
three folds, the same fit-on-the-other-two rotation, the same seed, the same five
contiguous day blocks for the cross-fit amplitude, the same per-airport amplitude, the same
103 features, the same floor, and the same day-clustered intervals.  The predeclared
setting (`analysis/predeclared.json`) is row 0 of the table and is the only one that may
ever be served without a new predeclaration.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/capacity.py
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

ARM = "c_lgb_shipped"
#: The sweep.  `None` is the predeclared setting, unchanged, and it must reproduce
#: `analysis/arms.json`'s `c_lgb_shipped` exactly -- that reproduction is the check that
#: this file's rotation really is the arm set's.
SETTINGS: dict[str, dict | None] = {
    "predeclared_31x300_lr050": None,
    "leaves_63x300_lr050": {"num_leaves": 63},
    "leaves_127x300_lr050": {"num_leaves": 127},
    "leaves_15x600_lr050": {"num_leaves": 15, "n_estimators": 600},
    "rounds_31x900_lr020": {"n_estimators": 900, "learning_rate": 0.02},
    "rounds_63x900_lr020": {"num_leaves": 63, "n_estimators": 900,
                            "learning_rate": 0.02},
    "rounds_63x1500_lr010": {"num_leaves": 63, "n_estimators": 1500,
                             "learning_rate": 0.01},
    "leafsize_63x900_lr020_mcs30": {"num_leaves": 63, "n_estimators": 900,
                                    "learning_rate": 0.02, "min_child_samples": 30},
    # The curve was still rising at 127 leaves and had not turned over, so the climb is
    # extended ONE STEP AT A TIME.  The desk, 2026-09-08 18:40 UTC: "stop the leaf climb
    # at the first setting that fails a clause or turns over; do not add settings past
    # that".  So 255 is added alone; 511 is added only if 255 both passes every clause and
    # beats 127, and the file records where the climb stopped and why.
    "leaves_255x300_lr050": {"num_leaves": 255},
}
#: where the leaf climb stopped, written by `main()` so the record says it and not a
#: reader's inference
CLIMB = ["leaves_15x600_lr050", "predeclared_31x300_lr050", "leaves_63x300_lr050",
         "leaves_127x300_lr050", "leaves_255x300_lr050", "leaves_511x300_lr050"]


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    rule, rule_report = A.serving_rule()
    # a setting already priced under the SAME rule and the same block caches is kept;
    # the sweep is extended, never silently re-run under a changed instrument
    prev = (json.loads((EXP / "analysis/capacity.json").read_text())
            if (EXP / "analysis/capacity.json").exists() else {})
    out = {
        "what": ("arm (c) at other capacities, on the same rotation, the same seed, the "
                 "same cross-fit and the same clauses; nothing here is served"),
        "arm": ARM, "n_features": len(obs_cols) + len(scols),
        "rule": {"sha256": rule,
                 "block_cache_keys": {m: A.block_cache_key(m, rule)
                                      for m in A.FOLD_MONTH.values()}},
        "predeclared_lgb": dict(A.LGB),
        "settings": {},
    }
    keep = (prev.get("rule", {}).get("sha256") == rule
            and prev.get("rule", {}).get("block_cache_keys") == out["rule"]["block_cache_keys"])
    if keep:
        out["settings"] = dict(prev.get("settings", {}))
    for name, params in SETTINGS.items():
        if name in out["settings"]:
            r = out["settings"][name]
            print(f"{name:32s} kept from the last sweep: all-row "
                  f"{r['board_all_row']['gain_s']:+8.4f}", flush=True)
            continue
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
            cols = list(obs_cols) + scols
            d, info = A.arm_lgb(fit, sco, cols, params=params)
            d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{name} {f}: a noac row moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{name} {f}: a row outside the with-clock class moved"
            d_by_fold[f] = d
            detail[f] = {k: info[k] for k in ("weights", "n_features",
                                              "oof_rmse_vs_r11", "r11_rmse_on_fit_rows")}
        rec = BA.price(pan, d_by_fold, rng)
        rec["detail"] = detail
        rec["lgb"] = params
        rec["tail_clauses"] = {f: A.tail_clauses(pan[f], d_by_fold[f]) for f in A.FOLDS}
        rec["seconds"] = round(time.time() - t0, 1)
        out["settings"][name] = rec
        np.savez_compressed(EXP / f"work/d-capacity-{name}.npz", **d_by_fold)
        tc = rec["tail_clauses"]
        print(f"{name:32s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{rec['board_all_row']['interval']['ci95'][0]:+7.3f},"
              f"{rec['board_all_row']['interval']['ci95'][1]:+7.3f}] "
              f"clean {rec['board_clean']['gain_s']:+8.4f} "
              f"dec {rec['december_rmse']['gain_s']:+8.4f} "
              f"clear {str(rec['every_fold_interval_clear']):5s} tail "
              + " ".join(f"{f[:3]} {tc[f]['registered_clause_y_over_3600s']['rmse_gain_s']:+7.3f}"
                         for f in A.FOLDS)
              + f"  {rec['seconds']:.0f}s", flush=True)
        (EXP / "analysis/capacity.json").write_text(
            json.dumps(out, indent=2, sort_keys=True))
    # the climb's own stopping rule, recorded rather than left to the reader
    climb = [(n, out["settings"][n]) for n in CLIMB if n in out["settings"]]
    stop = None
    for i, (n, r) in enumerate(climb):
        tails_pass = all(r["tail_clauses"][f]["registered_clause_y_over_3600s"]
                         ["passes_the_clause"] for f in A.FOLDS)
        clear = bool(r["every_fold_interval_clear"])
        turned = (i > 0 and r["board_all_row"]["gain_s"]
                  <= climb[i - 1][1]["board_all_row"]["gain_s"])
        if not tails_pass or not clear or turned:
            stop = {"at": n, "failed_a_clause": not (tails_pass and clear),
                    "turned_over": bool(turned)}
            break
    out["leaf_climb"] = {
        "order": [n for n, _ in climb],
        "all_row_gain_s": {n: r["board_all_row"]["gain_s"] for n, r in climb},
        "stopped_at": stop,
        "rule": ("desk, 2026-09-08 18:40 UTC: stop at the first setting that fails a "
                 "clause or turns over; do not add settings past that"),
    }
    (EXP / "analysis/capacity.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print("leaf climb:", out["leaf_climb"]["all_row_gain_s"], "stopped at", stop)
    print("written to analysis/capacity.json")


if __name__ == "__main__":
    main()
