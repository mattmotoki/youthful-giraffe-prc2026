"""Did the harvest's new july days move the gate?  Two arms, re-priced, nothing registered.

2026-09-09: the v16 dry run refused to write a candidate.  `serve.py`'s guard (Sol round 1,
blocking 3) compares the block cache keys `analysis/arms.json` was priced under against the
ones the serve resolves, and 2025-07's had moved from `4521a8fe…` to `6a27ed27…`.  The rule
version and `block.py` are both unchanged; what moved is the DATA -- the harvest delivered
2025-06-30 for nine airports, a boundary day the july block reads, taking the month from 278
to 287 present days and its observed segments from 118,586 to 122,687 (+3.5 %).

The decision -- re-price the whole pool, or pin the block to the priced population -- is the
desk's.  This file measures the number that decides it: **how much the gate actually moves**.
It prices only the two arms the v16 gate is built from,

  c_lgb_shipped@leaves_127x300_lr050            the head v15 ships
  c_lgb_shipped@leaves_255x300_lr050 + guard    the v16 candidate

on the block the serve now resolves, with the same rotation, seed, cross-fit, floor, hold and
clauses as `capacity.py` and `guardarm.py`, and compares every gate quantity with the one the
record carries.  It writes `analysis/block-move.json` and `work/d-blockmove-*.npz` ONLY: no
file the serve or either gate reads is touched, so the record stays exactly as reviewed until
the desk rules.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/blockmove.py
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

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402
import guard as G                                                     # noqa: E402

HOLD = ("LFPG",)
SETTINGS = {"leaves_127x300_lr050": {"num_leaves": 127},
            "leaves_255x300_lr050": {"num_leaves": 255}}
OUT = EXP / "analysis/block-move.json"


def priced_before() -> dict:
    """What the record carries, read from the files the gate was read on."""
    cap = json.loads((EXP / "analysis/capacity.json").read_text())
    gua = json.loads((EXP / "analysis/guard-arms.json").read_text())
    return {
        "capacity_block_cache_keys": cap.get("rule", {}).get("block_cache_keys"),
        "unheld_all_row_gain_s": {
            n: cap["settings"][n]["board_all_row"]["gain_s"]
            for n in SETTINGS if n in cap.get("settings", {})},
        "guarded_unheld_all_row_gain_s": {
            n: r["board_all_row"]["gain_s"] for n, r in gua.get("settings", {}).items()},
        "guard_cut_s_by_scored_fold": gua.get("guard", {}).get("cut_s_by_scored_fold"),
        "guard_cut_s_for_the_ranking_serve": gua.get("guard", {}).get(
            "cut_s_for_the_ranking_serve"),
    }


def held_price(pan: dict, d_by_fold: dict, rng) -> dict:
    """The fold gate's own quantity: the same vector with the held airport zeroed."""
    d = {}
    for f in A.FOLDS:
        v = d_by_fold[f].astype("float64").copy()
        v[np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))] = 0.0
        d[f] = v
    return BA.price(pan, d, rng)


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    rule, _ = A.serving_rule()
    keys_now = {m: A.block_cache_key(m, rule) for m in A.FOLD_MONTH.values()}
    before = priced_before()

    out = {
        "what": ("the two arms the v16 gate is built from, re-priced on the block the "
                 "serve now resolves, so the desk can see how far the gate moves; "
                 "nothing here is registered and no gate reads these files"),
        "why": ("the harvest delivered 2025-06-30 for nine airports into the july fold "
                "block after analysis/arms.json and analysis/capacity.json were priced"),
        "rule": {"sha256": rule, "block_cache_keys_now": keys_now,
                 "block_cache_keys_when_priced": before["capacity_block_cache_keys"]},
        "rows_by_fold": {f: int(len(pan[f])) for f in A.FOLDS},
        "usable_rows_by_fold": {f: int(pan[f].usable.sum()) for f in A.FOLDS},
        "before": before, "after": {},
    }

    vec = {}
    for name, params in SETTINGS.items():
        t0 = time.time()
        d_by_fold = {}
        for f in A.FOLDS:
            others = [g for g in A.FOLDS if g != f]
            fit = pd.concat([pan[g][pan[g].usable] for g in others], ignore_index=True)
            sco = pan[f]
            fit = fit.join(pd.concat([ship[g] for g in others])
                           .reindex(fit.MVT_ID_mvt.to_numpy("float64"))
                           .reset_index(drop=True))
            sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                           .reset_index(drop=True))
            d, _ = A.arm_lgb(fit, sco, list(obs_cols) + scols, params=params)
            d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{name} {f}: a noac row moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{name} {f}: a row outside the with-clock class moved"
            d_by_fold[f] = d
        vec[name] = d_by_fold
        np.savez_compressed(EXP / f"work/d-blockmove-{name}.npz", **d_by_fold)
        rec = BA.price(pan, d_by_fold, rng)
        out["after"][name] = {
            "unheld_all_row_gain_s": rec["board_all_row"]["gain_s"],
            "unheld_all_row_ci95": rec["board_all_row"]["interval"]["ci95"],
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name:28s} unheld {rec['board_all_row']['gain_s']:+8.4f} "
              f"(was {before['unheld_all_row_gain_s'].get(name, float('nan')):+8.4f})"
              f"  {out['after'][name]['seconds']:.0f}s", flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True))

    # the guard, frozen in code, but its CUT is derived from the fitting months -- so the
    # cut moves with the block and both readings are reported
    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    serve_cut = G.cut(pan, list(A.FOLDS))
    out["guard"] = {
        "quantile": G.Q,
        "cut_s_by_scored_fold_now": {f: round(c, 2) for f, c in cuts.items()},
        "cut_s_by_scored_fold_when_frozen": before["guard_cut_s_by_scored_fold"],
        "cut_s_for_the_ranking_serve_now": round(serve_cut, 2),
        "cut_s_for_the_ranking_serve_when_frozen":
            before["guard_cut_s_for_the_ranking_serve"],
        "note": ("guard.py freezes the QUANTILE, not the cut; the cut is a quantile of "
                 "pbase - obs_taxi on the fitting months, so more july rows move it. The "
                 "serve reads the frozen cut out of analysis/guard-arms.json and refuses "
                 "a mismatched quantile, so this is a number for the desk, not a drift "
                 "the serve would accept silently"),
    }
    guarded = {}
    for f in A.FOLDS:
        d = vec["leaves_255x300_lr050"][f].astype("float64").copy()
        d[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, cuts[f])] = 0.0
        guarded[f] = d
    np.savez_compressed(EXP / "work/d-blockmove-guard995.npz", **guarded)
    rec_g = BA.price(pan, guarded, rng)
    out["after"]["leaves_255x300_lr050+guard995"] = {
        "unheld_all_row_gain_s": rec_g["board_all_row"]["gain_s"],
        "unheld_all_row_ci95": rec_g["board_all_row"]["interval"]["ci95"],
    }

    # the two gate quantities, both with the hold applied exactly as the serve applies it
    held_g = held_price(pan, guarded, np.random.default_rng(A.SEED))
    out["fold_gate"] = {
        "held": list(HOLD),
        "all_row_gain_s_now": held_g["board_all_row"]["gain_s"],
        "all_row_ci95_now": held_g["board_all_row"]["interval"]["ci95"],
        "clean_gain_s_now": held_g["board_clean"]["gain_s"],
        "every_fold_interval_clear_now": bool(held_g["every_fold_interval_clear"]),
        "by_fold_now": {f: held_g["by_fold"][f]["all_row"] for f in A.FOLDS},
    }
    out["tail_clauses_now"] = {
        f: A.tail_clauses(pan[f], guarded[f])["registered_clause_y_over_3600s"]
        for f in A.FOLDS}

    # the increment over the head, built the way serve.increment_gate builds it
    hm = {f: np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD)) for f in A.FOLDS}
    inc, pan_head = {}, {}
    for f in A.FOLDS:
        a = guarded[f].copy()
        b = vec["leaves_127x300_lr050"][f].astype("float64").copy()
        a[hm[f]] = 0.0
        b[hm[f]] = 0.0
        inc[f] = a - b
        q = pan[f].copy()
        q["pbase"] = q["pbase"].to_numpy("float64") + b
        q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
        pan_head[f] = q
    rec_i = BA.price(pan_head, inc, np.random.default_rng(A.SEED))
    out["increment_over_the_head"] = {
        "head_arm": "c_lgb_shipped@leaves_127x300_lr050",
        "arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "all_row_gain_s_now": rec_i["board_all_row"]["gain_s"],
        "all_row_ci95_now": rec_i["board_all_row"]["interval"]["ci95"],
        "clean_gain_s_now": rec_i["board_clean"]["gain_s"],
        "december_gain_s_now": rec_i["december_rmse"]["gain_s"],
        "every_fold_interval_clear_now": bool(rec_i["every_fold_interval_clear"]),
        "by_fold_now": {f: rec_i["by_fold"][f]["all_row"] for f in A.FOLDS},
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))

    fg = out["fold_gate"]
    ig = out["increment_over_the_head"]
    print(f"\nfold gate (LFPG held)  {fg['all_row_gain_s_now']:+8.4f} "
          f"[{fg['all_row_ci95_now'][0]:+7.3f},{fg['all_row_ci95_now'][1]:+7.3f}]  "
          f"clear {fg['every_fold_interval_clear_now']}")
    print(f"increment over the head {ig['all_row_gain_s_now']:+8.4f} "
          f"[{ig['all_row_ci95_now'][0]:+7.3f},{ig['all_row_ci95_now'][1]:+7.3f}]  "
          f"clear {ig['every_fold_interval_clear_now']}  dec {ig['december_gain_s_now']:+7.3f}")
    print("guard cut for the serve  now %.2f, frozen %.2f"
          % (out["guard"]["cut_s_for_the_ranking_serve_now"],
             out["guard"]["cut_s_for_the_ranking_serve_when_frozen"]))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
