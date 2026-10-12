"""exp-061-sparseprior: the round-1 review answer to finding 3 --- the SEPARATE AUDIT, extended.

`code/audit61.py` recomputed exp-060's board point and one-day board band for each control and
then copied exp-060's own `pass` and `advance` booleans, so its record did not support the claim
that the WHOLE sixteen-condition gate was reproduced.  This job recomputes, from exp-060's frozen
row losses and draws and from nothing else of exp-060's:

  * every control's jul point, jan point and board point;
  * every control's jul, jan and board bands at block lengths 1, 2 and 3 days;
  * the sixteen conditions themselves, so `pass` and `advance` are DERIVED here, not copied;
  * the winner-adjusted point and its one-day band, with the winning sham re-derived;
  * every one of the eleven arms' board gains at exp-060's primary bound level, its rank, and the
    winner-adjusted three-day band, each compared with exp-060's per-arm recorded value.

The first pass of this job wrote `analysis/audit61-r1.json` and reported fifteen mismatches, both
of them defects of THIS comparison and not of exp-060: it compared the per-fold bands in RMSE
seconds against a record that holds them in s2/row, and it held the per-fold points to the board
tolerance of 1e-12 although the two summation orders differ by 1.2e-12 relative.  That record is
kept; this pass writes `analysis/audit61-r2.json`.

Like `audit61.py` it is a terminal artifact: no producing job of exp-061 imports it or reads
`analysis/audit61-r1.json`, and it runs only after every exp-061 record exists.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/audit61r1.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

E60A = E.E60EXP / "analysis"
TOL = 1e-12
#: exp-060 summed its per-fold quantities over date sums and this job over the row column, so the
#: two agree to summation order (1.2e-12 relative on jul) and not to the board tolerance
TOL_FOLD = 1e-9


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.out_path(E.ANALYSIS / "audit61-r2.json")
    NO.refuse_existing([out], a.overwrite)
    t0 = time.time()
    launch = E.hold_load("audit r1")
    if not (E.ANALYSIS / "audit61.json").exists():
        raise SystemExit("the extended audit runs after analysis/audit61.json exists")

    rec = {"what": __doc__.strip().splitlines()[0], "launch": launch, "code_sha256": E.code_shas(),
           "gate": {}, "bound": {}}
    bad = []

    oos60 = E.load_json(E60A / "oos.json")
    rows60 = pd.read_parquet(E.ROOT / oos60["artifacts"]["rows"]["path"])
    draws60 = np.load(E.ROOT / oos60["artifacts"]["draws"]["path"])
    for key in ("rows", "draws"):
        aa = oos60["artifacts"][key]
        if E.sha_file(E.ROOT / aa["path"]) != aa["sha256"]:
            raise SystemExit(f"exp-060's frozen receipt {aa['path']} has changed")

    di = {r: rows60[rows60["rotation"] == r]["day_index"].to_numpy("int64") for r in E.ORDER}
    k = {r: int(di[r].max()) + 1 for r in E.ORDER}
    loss = {r: {f"{aa}|B": rows60[rows60["rotation"] == r][f"loss:{aa}|B"].to_numpy("float64")
                for aa in E.E60.ARMS} for r in E.ORDER}
    dr = {r: {L: draws60[f"{r}_{L}"] for L in E.BLOCK_LENGTHS} for r in E.ORDER}
    for r in E.ORDER:
        for L in E.BLOCK_LENGTHS:
            if not np.array_equal(E.block_draws(k[r], L, E.draw_seed(r, L)), dr[r][L]):
                bad.append(f"exp-060 draws {r} block {L}")

    def fold_mse_draws(arm: str, r: str, L: int) -> np.ndarray:
        e = np.bincount(di[r], weights=loss[r][arm], minlength=k[r])
        cnt = np.bincount(di[r], minlength=k[r]).astype("float64")
        idx = dr[r][L]
        return e[idx].sum(1) / cnt[idx].sum(1)

    def board_rmse_draws(arm: str, L: int) -> np.ndarray:
        return np.sqrt(sum(s * fold_mse_draws(arm, r, L) for r, s in E.BOARD.items()))

    def fold_point(arm: str, r: str) -> float:
        return float(loss[r][arm].mean())

    def board_point(arm: str) -> float:
        return float(np.sqrt(sum(s * loss[r][arm].mean() for r, s in E.BOARD.items())))

    # ---- the sixteen conditions, control by control ---------------------------------------------
    derived_advance = True
    for c in E.E60.GATE_CONTROLS:
        A_, B_ = f"{c}|B", "true|B"
        folds = {}
        for r in E.ORDER:
            fp = fold_point(A_, r) - fold_point(B_, r)                # s2/row, true over control
            # exp-060 records the per-fold band in MSE units (s2/row), the board band in RMSE
            bands = {f"ci_block_{L}d": E.ci(fold_mse_draws(A_, r, L) - fold_mse_draws(B_, r, L))
                     for L in E.BLOCK_LENGTHS}
            folds[r] = {"mse_gain_s2_per_row": fp, **bands}
        bg = board_point(A_) - board_point(B_)
        bbands = {f"ci_block_{L}d": E.ci(board_rmse_draws(A_, L) - board_rmse_draws(B_, L))
                  for L in E.BLOCK_LENGTHS}
        below = any(bands[f"ci_block_{L}d"][1] < 0 for L in (2, 3)
                    for bands in [folds["jul"], folds["jan"], bbands])
        cond = {"jul_point_positive": folds["jul"]["mse_gain_s2_per_row"] > 0,
                "jan_point_positive": folds["jan"]["mse_gain_s2_per_row"] > 0,
                "board_point_positive": bg > 0,
                "board_1d_lower_endpoint_positive": bbands["ci_block_1d"][0] > 0,
                "no_2d_or_3d_band_wholly_below_zero": not below}
        derived_pass = all(cond.values())
        derived_advance = derived_advance and derived_pass
        w = oos60["gate"][c]
        cc = oos60["contrasts"][f"true_vs_{c}"]["B"]
        checks = {"board_gain_s": rel(bg, w["board_gain_s"]) < TOL,
                  "board_ci_1d": max(rel(bbands["ci_block_1d"][i], w["board_ci_1d"][i])
                                     for i in (0, 1)) < TOL,
                  "pass": derived_pass == w["pass"],
                  "band_wholly_below_zero_2d_3d": below == w["band_wholly_below_zero_2d_3d"],
                  "fold_points": all(rel(folds[r]["mse_gain_s2_per_row"],
                                         cc["folds"][r]["mse_gain_s2_per_row"]) < TOL_FOLD
                                     for r in E.ORDER),
                  "fold_bands": all(rel(folds[r][f"ci_block_{L}d"][i],
                                        cc["folds"][r][f"ci_block_{L}d"][i]) < TOL_FOLD
                                    for r in E.ORDER for L in E.BLOCK_LENGTHS for i in (0, 1)),
                  "board_bands": all(rel(bbands[f"ci_block_{L}d"][i],
                                         cc["board"][f"ci_block_{L}d"][i]) < TOL
                                     for L in E.BLOCK_LENGTHS for i in (0, 1)),
                  "fold_points_positive": ({"jul": cond["jul_point_positive"],
                                            "jan": cond["jan_point_positive"]}
                                           == w["fold_points_positive"])}
        rec["gate"][c] = {"recomputed_board_gain_s": bg,
                          "recorded_board_gain_s": w["board_gain_s"],
                          "recomputed_fold_points_s2_per_row":
                              {r: folds[r]["mse_gain_s2_per_row"] for r in E.ORDER},
                          "recomputed_board_bands": bbands,
                          "recomputed_fold_bands":
                              {r: {kk: vv for kk, vv in folds[r].items() if kk != "mse_gain_s2_per_row"}
                               for r in E.ORDER},
                          "conditions": cond, "derived_pass": derived_pass,
                          "recorded_pass": w["pass"], "checks": checks,
                          "equal": all(checks.values())}
        if not all(checks.values()):
            bad.append(f"exp-060 gate {c}")

    pt = {arm: board_point(f"{arm}|B") for arm in E.E60.ARMS}
    best = min(E.E60.SHAM_ARMS, key=lambda s: pt[s])
    w60 = oos60["winner_adjusted"]["B"]
    g1 = np.stack([board_rmse_draws(f"{s}|B", 1) for s in E.E60.SHAM_ARMS]).min(0) \
        - board_rmse_draws("true|B", 1)
    wpt = pt[best] - pt["true"]
    wok = (rel(wpt, w60["point_s"]) < TOL and best == w60["winning_sham_on_the_original_rows"]
           and max(rel(E.ci(g1)[i], w60["ci_block_1d"][i]) for i in (0, 1)) < TOL)
    derived_advance = derived_advance and wpt > 0 and E.ci(g1)[0] > 0
    rec["gate"]["winner_adjusted"] = {"recomputed_point_s": wpt, "recorded_point_s": w60["point_s"],
                                      "recomputed_ci_1d": E.ci(g1),
                                      "recorded_ci_1d": w60["ci_block_1d"],
                                      "recomputed_winner": best,
                                      "recorded_winner": w60["winning_sham_on_the_original_rows"],
                                      "equal": bool(wok)}
    if not wok:
        bad.append("exp-060 winner-adjusted gate")
    rec["gate"]["derived_advance"] = bool(derived_advance)
    rec["gate"]["recorded_advance"] = oos60["gate"]["advance"]
    if bool(derived_advance) != bool(oos60["gate"]["advance"]):
        bad.append("exp-060 advance")

    # ---- every arm's board gain at exp-060's primary bound level ---------------------------------
    b60 = E.load_json(E60A / "bound-r2.json")
    level = b60["winner_adjusted_three_day_band"]["level"]
    per = {r: E.load_json(E60A / f"bound-r2-{r}.json") for r in E.ORDER}
    ref = {r: per[r]["arms"]["zero"][level]["sse"] for r in E.ORDER}
    rowsn = {r: per[r]["rows_all"] for r in E.ORDER}
    arms = [aa for aa in b60["levels"][level]["board_gain_s"]]
    gains = {}
    for arm in arms:
        base = float(np.sqrt(sum(s * ref[r] / rowsn[r] for r, s in E.BOARD.items())))
        got = float(np.sqrt(sum(s * per[r]["arms"][arm][level]["sse"] / rowsn[r]
                                for r, s in E.BOARD.items())))
        gains[arm] = base - got
    ranked = ["true"] + list(E.E60.SHAM_ARMS)          # the registered eleven: true and its shams
    order = sorted(ranked, key=lambda x: -gains[x])
    per_arm = {arm: {"recomputed_board_gain_s": gains[arm],
                     "recorded_board_gain_s": b60["levels"][level]["board_gain_s"][arm],
                     "equal": rel(gains[arm],
                                  b60["levels"][level]["board_gain_s"][arm]) < 1e-9}
               for arm in arms}
    rank = order.index("true") + 1
    rec["bound"] = {"level": level, "per_arm": per_arm,
                    "derived_rank_of_true": rank,
                    "recorded_rank_of_true": b60["levels"][level]["true_rank_of_11"],
                    "rank_equal": rank == b60["levels"][level]["true_rank_of_11"],
                    "arms_compared": len(arms), "ranked_arms": len(ranked),
                    "all_arms_equal": all(v["equal"] for v in per_arm.values())}
    if not rec["bound"]["all_arms_equal"] or not rec["bound"]["rank_equal"]:
        bad.append("exp-060 per-arm bound gains")

    rec["mismatches"] = bad
    rec["all_reproduced"] = not bad
    rec["label_reads"] = E.LABEL_READS
    rec["opened_paths"] = list(E.OPENED)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["terminal_artifact"] = ("analysis/audit61-r2.json is read by no producing job of exp-061; "
                                "no exp-061 module imports audit61r1.py")
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"written {out.name}: all_reproduced={rec['all_reproduced']} "
          f"mismatches={bad} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
