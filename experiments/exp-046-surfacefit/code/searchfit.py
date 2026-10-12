"""Order 6: the SEARCH-level cross-fit -- what did choosing the winner cost?

Desk order, 2026-09-09, once orders 4 and 5 had removed the instrument and the magnitude:

    the remaining candidate for the spread is LANE-LEVEL selection: the arm table is about
    30 arms and 37 readings on the same three folds, and a within-step cross-fit cannot see
    the premium of choosing the winner of that search. Order 6, arithmetic only from
    arms.json / crossfit.json: search-level cross-fit -- select the lane's best arm on two
    folds, read it on the third, each fold held out in turn; report the cross-fitted price
    of 'the arm this lane would have chosen' beside the in-sample winner's, for the v13,
    v15 and v16 steps. If the premium lines up with 1.75 / 2.01 / 0.36 it is the currency;
    if not, say what is left.

**The difference from `crossfit.py`.**  That file cross-fits WITHIN one step: five leaf
counts, one ladder.  This cross-fits the SEARCH: on each held-out fold the lane's whole
candidate pool is ranked by the other two folds and the winner of THAT ranking is read on
the fold it never saw.  A premium that lives in choosing which arm to serve at all -- not
in choosing how many leaves -- is invisible to the first and visible to this.

**Three arms are reported per step, not two.**  The arm the lane SERVED is not always the
pool's in-sample winner (at v13 it is not, and at v15 it is not), so the premium the desk
asked for -- in-sample winner minus cross-fitted winner -- is reported on its own terms,
and what the lane served sits beside it as a third number.

**Arithmetic only.**  Every number is read from `analysis/arms.json`, `capacity.json` and
`guard-arms.json`, which carry each arm's per-fold `rmse_base`/`rmse_arm` and per-fold
`mse_gain_s2`.  Nothing is refitted and no vector is loaded; the board figure is
`base.board()` over the same fold RMSEs the gate reads, so an in-sample number here
reproduces the one already in `logs/reprice.log`.

**score-feedback rule.**  The board readings appear as READINGS in `what_is_left` -- a diagnosis of the
transfer and a candidate projection currency.  No arm is ranked by them and nothing is
fitted to them.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/searchfit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

OUT = EXP / "analysis/searchfit.json"
#: excluded BY NAME, as the arm-choice rule already excludes them: a control or a placebo
#: is not something this lane would ever have served
EXCLUDE = ("placebo", "noobs", "_control")
#: readings, not fits -- the board's own RMSE after each accepted submission
BOARD_RMSE = {"v12": 262.7418, "v13": 248.4818, "v14": 248.2904,
              "v15": 246.8263, "v16": 246.7053}


def pool_from(path: Path, key: str, tag: str) -> dict:
    """`name -> {fold -> (rmse_base, rmse_arm, mse_gain_s2)}` for one priced file."""
    d = json.loads(path.read_text())[key]
    out = {}
    for name, rec in d.items():
        if any(x in name for x in EXCLUDE):
            continue
        bf = rec.get("by_fold") or {}
        if not all(f in bf for f in A.FOLDS):
            continue
        out[f"{tag}:{name}"] = {
            f: {"rmse_base": float(bf[f]["rmse_base"]),
                "rmse_arm": float(bf[f]["rmse_arm"]),
                "mse_gain_s2": float(bf[f]["all_row"]["mse_gain_s2"])}
            for f in A.FOLDS}
    return out


def scored_rmse(rec: dict) -> dict:
    return {f: rec[f]["rmse_arm"] for f in BA.SHARES}


def head_rmse(pool: dict, ref: dict | None) -> dict:
    """The RMSE the step starts from: the reference arm's, or v12's own base."""
    if ref is not None:
        return scored_rmse(ref)
    return {f: next(iter(pool.values()))[f]["rmse_base"] for f in BA.SHARES}


def step_score(rec: dict, ref: dict | None, folds: list[str]) -> float:
    """The lane's own arm-choice criterion, restricted to `folds`.

    It is the mean per-fold `mse_gain_s2` over the step, so a selection made on two folds
    ranks the STEP and not the level the step starts from.
    """
    return sum(rec[g]["mse_gain_s2"] - (ref[g]["mse_gain_s2"] if ref else 0.0)
               for g in folds) / len(folds)


def main() -> None:
    arms = pool_from(EXP / "analysis/arms.json", "arms", "arm")
    caps = pool_from(EXP / "analysis/capacity.json", "settings", "cap")
    guards = pool_from(EXP / "analysis/guard-arms.json", "settings", "guard")

    out = {"what": ("the search-level cross-fit: the lane's own arm choice made out of "
                    "fold, so the premium of choosing the winner of a search over three "
                    "folds becomes visible"),
           "ordered_by": "the desk, 2026-09-09, order 6",
           "arithmetic_only": ("read from analysis/arms.json, capacity.json and "
                               "guard-arms.json; nothing refitted, no vector loaded"),
           "excluded_by_name": list(EXCLUDE),
           "matcher_sha256": A.MATCHER_SHA,
           "d11": ("board readings appear only in what_is_left, as a diagnosis of the "
                   "transfer; no arm is ranked by them and nothing is fitted to them"),
           "steps": {}}

    STEPS = {
        "v13": {"pool": arms, "ref": None, "served": "arm:c_lgb_shipped",
                "board_from": "v12", "board_to": "v13",
                "what": "this lane's whole with-clock update, chosen from the arm table"},
        "v15_increment": {"pool": {**arms, **caps},
                          "ref": caps["cap:predeclared_31x300_lr050"],
                          "served": "cap:leaves_127x300_lr050",
                          "board_from": "v14", "board_to": "v15",
                          "what": "the step from the predeclared setting to 127 leaves"},
        "v16_increment": {"pool": {**caps, **guards},
                          "ref": caps["cap:leaves_127x300_lr050"],
                          "served": "guard:leaves_255x300_lr050+guard995",
                          "board_from": "v15", "board_to": "v16",
                          "what": "the step from 127 leaves to 255 leaves plus the guard"},
    }

    for step, cfg in STEPS.items():
        pool, ref = cfg["pool"], cfg["ref"]
        head = head_rmse(pool, ref)
        board_head = BA.board(head)

        def board_gain_of(rec: dict) -> float:
            return board_head - BA.board(scored_rmse(rec))

        # --- in sample: the pool's own winner, ranked on all three folds at once ---
        ins_name = max(pool, key=lambda n: step_score(pool[n], ref, list(A.FOLDS)))
        ins_gain = board_gain_of(pool[ins_name])

        # --- out of sample: select on two folds, read the third, each fold in turn ---
        picks, rmse_cross = {}, {}
        for f in A.FOLDS:
            sel = [g for g in A.FOLDS if g != f]
            best = max(pool, key=lambda n: step_score(pool[n], ref, sel))
            picks[f] = {"selected_on": sel, "chosen": best,
                        "score_on_the_selecting_folds_s2":
                            round(step_score(pool[best], ref, sel), 2),
                        "its_price_on_the_held_out_fold_s2":
                            round(step_score(pool[best], ref, [f]), 2)}
            rmse_cross[f] = pool[best][f]["rmse_arm"]
        xf_gain = board_head - BA.board({f: rmse_cross[f] for f in BA.SHARES})

        served_gain = board_gain_of(pool[cfg["served"]])
        obs = BOARD_RMSE[cfg["board_from"]] - BOARD_RMSE[cfg["board_to"]]

        rec = {
            "what": cfg["what"], "pool_size": len(pool),
            "board_head_rmse_s": board_head,
            "in_sample_winner": {"arm": ins_name, "board_gain_s": ins_gain},
            "cross_fitted": {"board_gain_s": xf_gain, "picks": picks,
                             "all_folds_chose_the_same_arm":
                                 len({p["chosen"] for p in picks.values()}) == 1},
            "search_premium_s": ins_gain - xf_gain,
            "served_by_this_lane": {"arm": cfg["served"], "board_gain_s": served_gain,
                                    "is_the_in_sample_winner": cfg["served"] == ins_name},
            "board": {"from": cfg["board_from"], "to": cfg["board_to"],
                      "observed_delta_s": obs,
                      "over_the_served_face": obs / served_gain,
                      "over_the_cross_fitted_face": obs / xf_gain},
        }
        out["steps"][step] = rec
        print(f"{step:15s} pool {len(pool):3d}  in-sample winner {ins_name}", flush=True)
        print(f"{'':15s} in-sample {ins_gain:+.4f}  cross-fitted {xf_gain:+.4f}  "
              f"PREMIUM {ins_gain - xf_gain:+.4f}", flush=True)
        print(f"{'':15s} served {cfg['served']} {served_gain:+.4f}  "
              f"board delta {obs:+.4f}  transfer x{obs / served_gain:.2f}", flush=True)
        for f, pk in picks.items():
            print(f"    scored {f:9s} on {'+'.join(pk['selected_on']):22s} "
                  f"-> {pk['chosen']}", flush=True)

    # ---- what is left, if the premium does not line up: the ERROR LEVEL ----------------
    # An MSE gain buys seconds at a rate set by the level it is taken at: for a gain g
    # measured in s^2 on a fold whose RMSE is L, the second gain is L - sqrt(L^2 - g),
    # which is about g/(2L).  The folds sit near 330 s and the board near 250 s, so the
    # SAME s^2 gain is worth more seconds on the board than on the folds -- by construction
    # and in the same direction for every step.  It is a reading of the transfer, not a fit.
    left = {"what": ("the error LEVEL the same s2 gain is converted at: the folds are a "
                     "harder population than the 2026 ranking rows, so the identical "
                     "mean-squared reduction buys more seconds on the board"),
            "formula": "seconds(g, L) = L - sqrt(L**2 - g), about g/(2L)",
            "per_step": {}}
    for step, cfg in STEPS.items():
        r = out["steps"][step]
        pool, ref = cfg["pool"], cfg["ref"]
        head = head_rmse(pool, ref)
        g = BA.board(head) ** 2 - BA.board(scored_rmse(pool[cfg["served"]])) ** 2
        L = BOARD_RMSE[cfg["board_from"]]
        transported = L - (L ** 2 - g) ** 0.5
        left["per_step"][step] = {
            "s2_gain_of_the_served_arm": g,
            "fold_level_s": BA.board(head), "board_level_s": L,
            "face_at_the_fold_level_s": r["served_by_this_lane"]["board_gain_s"],
            "transported_to_the_board_level_s": transported,
            "level_multiplier": transported / r["served_by_this_lane"]["board_gain_s"],
            "observed_transfer": r["board"]["over_the_served_face"],
            "still_unexplained_multiplier":
                r["board"]["over_the_served_face"]
                / (transported / r["served_by_this_lane"]["board_gain_s"])}
    # order 5's magnitude ratio is the only other measured currency; compose the two so
    # the record carries the combination, not two numbers a reader has to multiply
    mag = json.loads((EXP / "analysis/magnitude.json").read_text())["by_update"]
    for step, v in left["per_step"].items():
        m = float(mag[step]["mean_d2_ratio_on_served_class"])
        v["magnitude_ratio_order_5"] = m
        v["level_times_magnitude"] = v["level_multiplier"] * m
        v["still_unexplained_after_both"] = (v["observed_transfer"]
                                             / v["level_times_magnitude"])
    out["what_is_left"] = left
    print("\nwhat is left -- the error level the gain is converted at:", flush=True)
    for step, v in left["per_step"].items():
        print(f"  {step:15s} fold level {v['fold_level_s']:7.2f} s, board level "
              f"{v['board_level_s']:7.2f} s -> level x{v['level_multiplier']:.3f}; "
              f"observed x{v['observed_transfer']:.3f}; still "
              f"x{v['still_unexplained_multiplier']:.3f}; with order 5's magnitude "
              f"x{v['magnitude_ratio_order_5']:.3f} still "
              f"x{v['still_unexplained_after_both']:.3f}", flush=True)

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
