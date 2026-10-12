"""exp-054-arrivalrepr iteration 2 (SPEC section 2, recorded rule): the paired all-row endpoint of the keyless
arms beside iteration 1's full row-only and true arms.

Predictors per fold: v22; iteration 1's `rowonly` and `true` (their recorded `work/h-<fold>.parquet`,
sha-checked); the keyless arms `true_k, gmean_k, donor_k, shuffle_k, rowonly_k`
(`work/h2-<fold>.parquet`, sha-checked).  Every final predictor is `max(60, p_v22 + h)`, v22's bytes
outside P (asserted).  Every contrast and board figure is computed by iteration 1's own functions
(`price.contrast`, `price.board`, `price.predictor_table`, `price.segments`), so the statistics,
intervals, seeds and reconciliations are exactly iteration 1's.  Then the registered readings of
SPEC section 2.6.  Arithmetic on recorded vectors; no fit.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/price2.py
    # mechanics only, synthetic labels, scratch (reads the scratch h2 files):
    ... price2.py --dry-run --out-dir <scratch>
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
import e54 as E                                                       # noqa: E402
import price as P1                                                    # noqa: E402  (iteration 1's code, unchanged)

NO = E.NO
KL = ["true_k", "gmean_k", "donor_k", "shuffle_k", "rowonly_k"]
CTRL_K = ["gmean_k", "donor_k", "shuffle_k"]
PRIMARY2 = ("rowonly_k", "true_k")                 # (A, B): gain of B over A, the matched contrast
D36 = ("rowonly", "true_k")                        # the comparison with iteration 1's full row-only arm
CONTRASTS2 = ([PRIMARY2, D36, ("v22", "true_k")] + [(c, "true_k") for c in CTRL_K]
              + [("true", "true_k")]
              + [("v22", a) for a in KL if a != "true_k"]
              + [("rowonly_k", c) for c in CTRL_K]
              + [("rowonly", "rowonly_k")] + [("rowonly", c) for c in CTRL_K])


def key(a: str, b: str) -> str:
    return f"{b}_over_{a}"


def load(fold: str, out_dir: Path, dry: bool) -> tuple[pd.DataFrame, dict, dict]:
    j2 = json.loads((out_dir / f"analysis/fit2-{fold}.json").read_text())
    h2p = out_dir / f"work/h2-{fold}.parquet"
    if E.sha_file(h2p) != j2["h_parquet"]["sha256"]:
        raise SystemExit(f"{h2p} moved since its fit record was written")
    j1 = json.loads(E.fit_json(fold).read_text())
    h1p = E.h_path(fold)
    if E.sha_file(h1p) != j1["h_parquet"]["sha256"]:
        raise SystemExit(f"{h1p} moved since its fit record was written")
    H2, H1 = pd.read_parquet(h2p), pd.read_parquet(h1p)
    cols = E.PANEL_LABEL_FREE + ([] if dry else ["clean", "y_true"])
    pan = E.panel(fold, cols)
    ids = pan.MVT_ID_mvt.to_numpy("float64")
    for h in (H1, H2):
        if not np.array_equal(h.MVT_ID_mvt.to_numpy("float64"), ids):
            raise SystemExit(f"{fold}: h rows are not in panel order")
    pop = E.population(pan, fold)
    if not (np.array_equal(pop, H1["pop"].to_numpy(bool)) and np.array_equal(pop, H2["pop"].to_numpy(bool))):
        raise SystemExit(f"{fold}: P moved between the fits and the pricing")
    df = pan.copy()
    df["pop"] = pop
    if dry:
        df["clean"] = True
        df["y_true"] = df.p_v22.to_numpy("float64") + np.random.default_rng(E.SEED).normal(0.0, 170.0, len(df))
    frame = E.departure_frame(E.fold_month(fold))
    df["no_nm"] = E.no_nm_mask(frame, ids)
    df["r"] = df.y_true.to_numpy("float64") - df.p_v22.to_numpy("float64")
    pv = df.p_v22.to_numpy("float64")
    P = {"v22": pv.copy()}
    src = {"rowonly": H1["h_rowonly"], "true": H1["h_true"], **{a: H2[f"h_{a}"] for a in KL}}
    for a, hs in src.items():
        h = hs.to_numpy("float64")
        if np.any(h[~pop] != 0):
            raise SystemExit(f"{fold}: arm {a} moves a row outside P")
        p = np.maximum(E.MIN_PRED, pv + h)
        if not np.array_equal(p[~pop], pv[~pop]):
            raise SystemExit(f"{fold}: arm {a} changes v22's bytes outside P")
        P[a] = p
    return df, P, {"fit2": j2, "fit1": j1}


def readings2(C: dict, B: dict, dfs: dict, PP: dict) -> dict:
    """SPEC 2.6: the original advance rule (not relaxed) with the recorded rule comparator, and the three
    descriptive readings."""
    folds = E.FOLDS

    def fold_rule(a, b):
        k = key(a, b)
        pts = {f: C[f][k]["mse_gain_s2_per_row"] for f in folds}
        lo1 = {f: C[f][k]["ci_block_1d"][0] for f in folds}
        return pts, lo1, all(v >= 0 for v in pts.values()), sum(lo1[f] > 0 for f in folds) >= 2

    p_pts, p_lo, p_nn, p_ci = fold_rule(*PRIMARY2)
    d_pts, d_lo, d_nn, d_ci = fold_rule(*D36)
    cond = {"beats_v22_on_board": B[key("v22", "true_k")]["gain_s"] > 0,
            "beats_rowonly_k_on_board": B[key(*PRIMARY2)]["gain_s"] > 0,
            **{f"beats_{c}_on_board": B[key(c, "true_k")]["gain_s"] > 0 for c in CTRL_K},
            "beats_iteration1_rowonly_on_board": B[key(*D36)]["gain_s"] > 0,
            "primary_no_negative_fold_point": p_nn,
            "primary_1d_interval_above_zero_on_2_folds": p_ci,
            "d36_no_negative_fold_point": d_nn,
            "d36_1d_interval_above_zero_on_2_folds": d_ci}
    adv = all(cond.values())
    prim = B[key(*PRIMARY2)]
    relevance = bool(prim["ci_block_1d"][0] > 0 and p_nn)
    spec = {c: {"gain_s": B[key(c, "true_k")]["gain_s"], "ci_block_1d": B[key(c, "true_k")]["ci_block_1d"]}
            for c in CTRL_K}
    specificity = bool(all(v["ci_block_1d"][0] > 0 for v in spec.values()))
    # descriptive: the share of the raw keys' within-month gain that the factors recover, in board MSE
    bm = {x: sum(E.SHARES[f] * float(np.mean((dfs[f].y_true.to_numpy("float64") - PP[f][x]) ** 2))
                 for f in E.SHARES) for x in ("rowonly", "rowonly_k", "true_k")}
    den = bm["rowonly_k"] - bm["rowonly"]
    share = (bm["rowonly_k"] - bm["true_k"]) / den if den != 0 else None
    return {"conditions": cond, "advances": adv,
            "primary_fold_points_s2": p_pts, "primary_fold_1d_lower": p_lo,
            "d36_fold_points_s2": d_pts, "d36_fold_1d_lower": d_lo,
            "relevance_true_k_over_rowonly_k": {"gain_s": prim["gain_s"], "ci_block_1d": prim["ci_block_1d"],
                                                "holds": relevance},
            "specificity_true_k_over_each_control": {"by_control": spec, "holds": specificity},
            "substitution_true_k_over_iteration1_rowonly": {"gain_s": B[key(*D36)]["gain_s"],
                                                            "ci_block_1d": B[key(*D36)]["ci_block_1d"]},
            "raw_key_cost_rowonly_k_over_rowonly_s": B[key("rowonly", "rowonly_k")]["gain_s"],
            "recovered_share_board_mse": share, "board_mse": bm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else E.EXP
    if args.dry_run and out_dir == E.EXP:
        raise SystemExit("a dry run writes to --out-dir only")
    OUT = out_dir / "analysis/iter2.json"
    OOF = [out_dir / f"oof/iter2-{f}.parquet" for f in E.FOLDS]
    NO.refuse_existing([OUT] + OOF, args.overwrite)
    print(f"guard: no existing artifact at {OUT} or the three oof files", flush=True)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dfs, PP, C, T, FJ = {}, {}, {}, {}, {}
    for f in E.FOLDS:
        df, P, fj = load(f, out_dir, args.dry_run)
        dfs[f], PP[f], FJ[f] = df, P, fj
        seg = P1.segments(df)
        T[f] = P1.predictor_table(df, P, seg)
        C[f] = {key(a, b): P1.contrast(df, P[a], P[b], seg) for a, b in CONTRASTS2}
        print(f"  [{f}] true_k over rowonly_k {C[f][key(*PRIMARY2)]['mse_gain_s2_per_row']:+.3f} s^2/row "
              f"{[round(x, 3) for x in C[f][key(*PRIMARY2)]['ci_block_1d']]}; true_k over rowonly "
              f"{C[f][key(*D36)]['mse_gain_s2_per_row']:+.3f}; rowonly_k over rowonly "
              f"{C[f][key('rowonly', 'rowonly_k')]['mse_gain_s2_per_row']:+.3f}", flush=True)
    B = {key(a, b): P1.board(dfs, PP, a, b) for a, b in CONTRASTS2}
    R = readings2(C, B, dfs, PP)
    rec = {"what": "exp-054 iteration 2 (D36): keyless arms beside iteration 1's full row-only and true",
           "dry_run": args.dry_run,
           "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("e54.py", "price.py", "price2.py")},
           "shares": E.SHARES, "contrasts": [key(a, b) for a, b in CONTRASTS2],
           "primary": key(*PRIMARY2), "d36_comparator": key(*D36),
           "inputs": {f: {"fit2_json_sha256": E.sha_file(out_dir / f"analysis/fit2-{f}.json"),
                          "h2_parquet_sha256": FJ[f]["fit2"]["h_parquet"]["sha256"],
                          "fit1_json_sha256": E.sha_file(E.fit_json(f)),
                          "h1_parquet_sha256": FJ[f]["fit1"]["h_parquet"]["sha256"]} for f in E.FOLDS},
           "folds": {f: {"predictors": T[f], "contrasts": C[f]} for f in E.FOLDS},
           "board": B, "readings": R,
           "all_reconcile": bool(all(C[f][k]["reconciles"] for f in E.FOLDS for k in C[f])
                                 and all(b["reconciles"] for b in B.values())),
           "board_rmse": {x: float(np.sqrt(sum(E.SHARES[f] * np.mean(
               (dfs[f].y_true.to_numpy("float64") - PP[f][x]) ** 2) for f in E.SHARES)))
               for x in ["v22", "rowonly", "true"] + KL}}
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT] + OOF, args.overwrite)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    for f, p in zip(E.FOLDS, OOF):
        pd.DataFrame({"MVT_ID_mvt": dfs[f].MVT_ID_mvt.to_numpy("float64"), "fold": f,
                      "y_true": dfs[f].y_true.to_numpy("float64"), "y_pred": PP[f]["true_k"],
                      "y_pred_rowonly_k": PP[f]["rowonly_k"], "y_pred_rowonly": PP[f]["rowonly"],
                      "y_pred_v22": PP[f]["v22"]}).to_parquet(p, index=False)
    b = B[key(*PRIMARY2)]
    print(f"  PRIMARY board true_k over rowonly_k {b['gain_s']:+.5f} s {[round(x, 5) for x in b['ci_block_1d']]}; "
          f"true_k over rowonly {B[key(*D36)]['gain_s']:+.5f} s; advances {R['advances']}; "
          f"relevance {R['relevance_true_k_over_rowonly_k']['holds']}; "
          f"specificity {R['specificity_true_k_over_each_control']['holds']}; reconcile {rec['all_reconcile']}")
    print(f"written {OUT} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
