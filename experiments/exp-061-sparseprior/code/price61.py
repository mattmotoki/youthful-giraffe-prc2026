"""exp-061-sparseprior: price the hard sparse-key correction on V and apply the fixed gate
(SPEC 4).

Runs only after all three rotations' corrections are frozen: it checks every fit record's freeze
time against the V label read.  V is read ONCE per rotation and every arm is priced on ALL V rows
(the correction moves the population rows only).  Reused from exp-060 unchanged: the UTC-date
grouping, 2,000 moving-block draws at block lengths 1, 2 and 3 days with the same seeds and the
same draws for every arm, the board recombination at the ranking shares, COMMON.md's full fold and
serving tables, and the sixteen-condition level-B gate.

The V labels helped select this family before registration (SPEC 0), so the gate is a fixed
HISTORICAL-REUSE comparison: a pass keeps the path open and cannot advance Q12.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-061-sparseprior/code/price61.py
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

LEVELS = ("B", "A")


def fold_stats(la, lb, di, k, draws):
    v = la - lb
    num = np.bincount(di, weights=v, minlength=k)
    cnt = np.bincount(di, minlength=k).astype("float64")
    out = {"mse_gain_s2_per_row": float(v.mean()),
           "rmse_gain_s": float(np.sqrt(la.mean()) - np.sqrt(lb.mean())),
           "date_sums_s2": [float(x) for x in num]}
    for L, idx in draws.items():
        out[f"ci_block_{L}d"] = E.ci(num[idx].sum(1) / cnt[idx].sum(1))
    return out


def board_stats(la: dict, lb: dict, di: dict, k: dict, draws: dict):
    ma = sum(E.BOARD[r] * la[r].mean() for r in E.BOARD)
    mb = sum(E.BOARD[r] * lb[r].mean() for r in E.BOARD)
    out = {"board_gain_s": float(np.sqrt(ma) - np.sqrt(mb))}
    for L in E.BLOCK_LENGTHS:
        m0 = np.zeros(E.BOOT)
        m1 = np.zeros(E.BOOT)
        for r, s in E.BOARD.items():
            e0 = np.bincount(di[r], weights=la[r], minlength=k[r])
            e1 = np.bincount(di[r], weights=lb[r], minlength=k[r])
            c = np.bincount(di[r], minlength=k[r]).astype("float64")
            idx = draws[r][L]
            m0 += s * e0[idx].sum(1) / c[idx].sum(1)
            m1 += s * e1[idx].sum(1) / c[idx].sum(1)
        out[f"ci_block_{L}d"] = E.ci(np.sqrt(m0) - np.sqrt(m1))
    return out


def board_rmse_draws(loss: dict, di: dict, k: dict, draws: dict, L: int) -> np.ndarray:
    m = np.zeros(E.BOOT)
    for r, s in E.BOARD.items():
        e = np.bincount(di[r], weights=loss[r], minlength=k[r])
        c = np.bincount(di[r], minlength=k[r]).astype("float64")
        idx = draws[r][L]
        m += s * e[idx].sum(1) / c[idx].sum(1)
    return np.sqrt(m)


def rmse_table(y, pred, nonm, ids, fold):
    e2 = (y - pred) ** 2
    t = {"rmse": float(np.sqrt(e2.mean())),
         "rmse_le3600": float(np.sqrt(e2[y <= 3600].mean())),
         "rmse_gt3600": float(np.sqrt(e2[y > 3600].mean())) if (y > 3600).any() else None,
         "rmse_nonm": float(np.sqrt(e2[nonm].mean())) if nonm.any() else None,
         "rows": int(len(y)), "rows_gt3600": int((y > 3600).sum()), "rows_nonm": int(nonm.sum())}
    t["rmse_clean"] = float(E.D.rmse_clean(y, pred, ids, fold))
    return t


def gate(res: dict, wadj: dict) -> dict:
    """SPEC 4.2, applied mechanically: sixteen conditions."""
    out = {}
    for c in E.GATE_CONTROLS:
        cc = res["contrasts"][f"true_vs_{c}"]["B"]
        f_pos = {r: cc["folds"][r]["mse_gain_s2_per_row"] > 0 for r in E.BOARD}
        b = cc["board"]
        below = (any(b[f"ci_block_{L}d"][1] < 0 for L in (2, 3))
                 or any(cc["folds"][r][f"ci_block_{L}d"][1] < 0 for r in E.BOARD for L in (2, 3)))
        ok = (all(f_pos.values()) and b["board_gain_s"] > 0 and b["ci_block_1d"][0] > 0 and not below)
        out[c] = {"fold_points_positive": f_pos, "board_gain_s": b["board_gain_s"],
                  "board_ci_1d": b["ci_block_1d"], "band_wholly_below_zero_2d_3d": below,
                  "pass": bool(ok)}
    w = {"point_s": wadj["point_s"], "ci_block_1d": wadj["ci_block_1d"],
         "pass": bool(wadj["point_s"] > 0 and wadj["ci_block_1d"][0] > 0)}
    out["winner_adjusted"] = w
    out["advance"] = bool(all(out[c]["pass"] for c in E.GATE_CONTROLS) and w["pass"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out_rec = E.out_path(E.ANALYSIS / "oos61.json")
    rows_p = E.out_path(E.ART / "oos61-rows.parquet")
    draws_p = E.out_path(E.ART / "oos61-draws.npz")
    oof_paths = [E.out_path(E.EXP / f"oof/{arm}-iso_{r}_v.parquet") for arm in ("base", "zero", "true")
                 for r in E.ORDER]
    NO.refuse_existing([out_rec, rows_p, draws_p] + oof_paths, a.overwrite)
    t0 = time.time()
    launch = E.hold_load("price")

    inputs, masks, fits, z, h = {}, {}, {}, {}, {}
    for r in E.ORDER:
        inputs[r] = E.load_json(E.ANALYSIS / f"inputs61-{r}.json")
        masks[r] = E.load_json(E.ANALYSIS / f"mask61-{r}.json")
        fits[r] = E.load_json(E.ANALYSIS / f"fit61-{r}.json")
        for rec, key in ((inputs[r], "npz"), (masks[r], "npz"), (fits[r], "pred_npz")):
            if E.sha_file(E.ROOT / rec[key]) != rec[f"{key}_sha256"]:
                raise SystemExit(f"{rec[key]} is not the frozen file")
        z[r] = np.load(E.ROOT / inputs[r]["npz"])
        h[r] = np.load(E.ROOT / fits[r]["pred_npz"])
        E.pin_months(inputs[r])
    last_freeze = max(fits[r]["freeze_unix"] for r in E.ORDER)

    # ---- the only label reads of this job: V, after every rotation is frozen ----------------
    y, p, pop, di, k, nonm, ids = {}, {}, {}, {}, {}, {}, {}
    for r in E.ORDER:
        V = inputs[r]["months"]["V"]
        ids[r] = z[r]["ids_V"]
        y[r] = E.dep_labels(V, ids[r], "scoring month V, once, after every rotation is frozen")
        if not np.all(np.isfinite(y[r])):
            raise SystemExit(f"{V}: a departure without a label")
        p[r], pop[r], nonm[r] = z[r]["p_V"], z[r]["pop_V"], z[r]["nonm_V"]
        di[r] = z[r]["day_V"]
        k[r] = int(di[r].max()) + 1
    for rd in E.LABEL_READS:
        if rd["unix"] <= last_freeze:
            raise SystemExit("a V label was read before every rotation's corrections were frozen")

    draws = {r: {L: E.block_draws(k[r], L, E.draw_seed(r, L)) for L in E.BLOCK_LENGTHS}
             for r in E.ORDER}

    loss = {r: {"base": (y[r] - p[r]) ** 2} for r in E.ORDER}
    pred_v = {r: {} for r in E.ORDER}
    for r in E.ORDER:
        for arm in E.ARMS:
            for lev in LEVELS:
                pr = E.final_pred(p[r], h[r][f"h_{arm}_{lev}"], pop[r])
                pred_v[r][f"{arm}|{lev}"] = pr
                loss[r][f"{arm}|{lev}"] = (y[r] - pr) ** 2
        for lev in LEVELS:
            loss[r][f"sham_mean|{lev}"] = np.mean([loss[r][f"{s}|{lev}"] for s in E.SHAM_ARMS],
                                                  axis=0)

    def contrast(A_: str, B_: str) -> dict:
        folds = {r: fold_stats(loss[r][A_], loss[r][B_], di[r], k[r], draws[r]) for r in E.ORDER}
        board = board_stats({r: loss[r][A_] for r in E.BOARD}, {r: loss[r][B_] for r in E.BOARD},
                            {r: di[r] for r in E.BOARD}, {r: k[r] for r in E.BOARD},
                            {r: draws[r] for r in E.BOARD})
        return {"A": A_, "B": B_, "folds": folds, "board": board}

    res = {"what": __doc__.strip().splitlines()[0], "launch": launch, "code_sha256": E.code_shas(),
           "boot": E.BOOT, "board_shares": E.BOARD,
           "historical_reuse": ("the family, the n_A < 50 hypothesis and the threshold grid came "
                                "from a post-hoc read of exp-060's V labels; this gate is a fixed "
                                "historical-reuse comparison and cannot advance Q12"),
           "draw_seeds": {r: {L: E.draw_seed(r, L) for L in E.BLOCK_LENGTHS} for r in E.ORDER},
           "label_reads": E.LABEL_READS, "opened_paths": list(E.OPENED),
           "inputs": {r: {"record_npz_sha256": inputs[r]["npz_sha256"],
                          "masks_npz_sha256": masks[r]["npz_sha256"],
                          "matched_null_resolved": masks[r]["matched_null"]["resolved"],
                          "matched_null_failing_cells": masks[r]["matched_null"]["failing_cells"],
                          "fit_pred_npz_sha256": fits[r]["pred_npz_sha256"],
                          "fit_freeze_unix": fits[r]["freeze_unix"]} for r in E.ORDER},
           "picks": {r: {arm: {lev: {kk: fits[r]["arms"][arm]["levels"][lev][kk]
                                     for kk in ("n_star", "lam", "c", "cell", "betas_by_airport",
                                                "beta_nonzero_cells", "prior_nonzero_columns")}
                               for lev in LEVELS} for arm in E.ARMS} for r in E.ORDER},
           "contrasts": {}}
    for c in E.GATE_CONTROLS:
        res["contrasts"][f"true_vs_{c}"] = {lev: contrast(f"{c}|{lev}", f"true|{lev}")
                                            for lev in LEVELS}
    res["contrasts"]["true_vs_sham_mean"] = {lev: contrast(f"sham_mean|{lev}", f"true|{lev}")
                                             for lev in LEVELS}
    for arm in E.ARMS + ("sham_mean",):
        res["contrasts"][f"{arm}_vs_base"] = {"B": contrast("base", f"{arm}|B")}
    res["contrasts"]["true_B_vs_true_A"] = {"B": contrast("true|A", "true|B")}

    # ---- the winner-adjusted sham contrast (SPEC 4.2) ---------------------------------------
    def board_point(arm_key: str) -> float:
        return float(np.sqrt(sum(E.BOARD[r] * loss[r][arm_key].mean() for r in E.BOARD)))
    wadj = {}
    for lev in LEVELS:
        sham_pt = {s: board_point(f"{s}|{lev}") for s in E.SHAM_ARMS}
        best_s = min(sham_pt, key=sham_pt.get)
        pt = sham_pt[best_s] - board_point(f"true|{lev}")
        w = {"definition": "min over the ten shams of the recombined board RMSE, minus true's",
             "point_s": pt, "winning_sham_on_the_original_rows": best_s,
             "sham_board_rmse_s": sham_pt, "true_board_rmse_s": board_point(f"true|{lev}")}
        for L in E.BLOCK_LENGTHS:
            rm = {s: board_rmse_draws({r: loss[r][f"{s}|{lev}"] for r in E.BOARD}, di, k, draws, L)
                  for s in E.SHAM_ARMS}
            tr = board_rmse_draws({r: loss[r][f"true|{lev}"] for r in E.BOARD}, di, k, draws, L)
            mat = np.stack([rm[s] for s in E.SHAM_ARMS])
            g = mat.min(0) - tr
            w[f"ci_block_{L}d"] = E.ci(g)
            w[f"winning_sham_counts_block_{L}d"] = {
                s: int(n) for s, n in zip(E.SHAM_ARMS,
                                          np.bincount(mat.argmin(0), minlength=len(E.SHAM_ARMS)))}
        wadj[lev] = w
    res["winner_adjusted"] = wadj
    res["pairwise_sham_contrasts"] = {lev: {s: res["contrasts"][f"true_vs_{s}"][lev]["board"]
                                            for s in E.SHAM_ARMS} for lev in LEVELS}
    res["gate"] = gate(res, wadj["B"])
    res["gate"]["note"] = ("dec is the calm control with zero board weight and cannot pass or fail "
                           "the gate; level A is descriptive and enters no gate condition")

    # ---- SPEC 4.3, the V half: how far each sham's served prediction is from true's -----------
    res["served_prediction_difference_V"] = {
        r: {s: {"max_abs_s": float(np.max(np.abs(pred_v[r][f"{s}|B"] - pred_v[r]["true|B"]))),
                "rows_differing": int((pred_v[r][f"{s}|B"] != pred_v[r]["true|B"]).sum()),
                "population_rows_differing": int(((pred_v[r][f"{s}|B"] != pred_v[r]["true|B"])
                                                  & pop[r]).sum())}
            for s in E.SHAM_ARMS} for r in E.ORDER}

    tables = {}
    for r in E.ORDER:
        fold = f"iso_{r}_v"
        tables[r] = {"base": rmse_table(y[r], p[r], nonm[r], ids[r], fold)}
        for arm in E.ARMS:
            for lev in LEVELS:
                tables[r][f"{arm}|{lev}"] = rmse_table(y[r], pred_v[r][f"{arm}|{lev}"], nonm[r],
                                                       ids[r], fold)
    res["tables"] = tables

    NO.refuse_existing([out_rec, rows_p, draws_p] + oof_paths, a.overwrite)
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for r in E.ORDER:
        df = pd.DataFrame({"rotation": r, "MVT_ID_mvt": ids[r], "day_index": di[r], "pop": pop[r]})
        for key, v in loss[r].items():
            df[f"loss:{key}"] = v
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_parquet(rows_p, index=False)
    np.savez(draws_p, **{f"{r}_{L}": draws[r][L] for r in E.ORDER for L in E.BLOCK_LENGTHS})
    (E.EXP / "oof").mkdir(parents=True, exist_ok=True)
    for r in E.ORDER:
        for arm in ("base", "zero", "true"):
            pred = p[r] if arm == "base" else pred_v[r][f"{arm}|B"]
            pd.DataFrame({"MVT_ID_mvt": ids[r], "fold": f"iso_{r}_v", "y_true": y[r],
                          "y_pred": pred}).to_parquet(E.EXP / f"oof/{arm}-iso_{r}_v.parquet",
                                                      index=False)
    res["artifacts"] = {"rows": {"path": str(rows_p.relative_to(E.ROOT)), "sha256": E.sha_file(rows_p)},
                        "draws": {"path": str(draws_p.relative_to(E.ROOT)), "sha256": E.sha_file(draws_p)},
                        "oof": {str(q.relative_to(E.ROOT)): E.sha_file(q) for q in oof_paths}}
    res["row_loss_sha256"] = {r: {key: E.sha_arr(v) for key, v in sorted(loss[r].items())}
                              for r in E.ORDER}
    res["seconds"] = round(time.time() - t0, 1)
    out_rec.write_text(json.dumps(res, indent=1, sort_keys=True))
    g = res["gate"]
    print(json.dumps({c: g[c]["pass"] for c in E.GATE_CONTROLS}
                     | {"winner_adjusted": g["winner_adjusted"]["pass"], "advance": g["advance"]}),
          flush=True)
    for c in E.GATE_CONTROLS:
        print(f"  true vs {c:12s} board {g[c]['board_gain_s']:+.4f} s  1d {g[c]['board_ci_1d']}",
              flush=True)
    w = wadj["B"]
    print(f"  winner-adjusted (level B): {w['point_s']:+.4f} s, 1d {w['ci_block_1d']}, "
          f"3d {w['ci_block_3d']}, winner {w['winning_sham_on_the_original_rows']}", flush=True)
    print(f"written {out_rec.relative_to(E.ROOT)} ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
