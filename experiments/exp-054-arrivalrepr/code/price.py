"""exp-054-arrivalrepr: the paired all-row endpoint of every arm (SPEC 1.9-1.11).

Per fold: every arm's final predictor is `p_arm = max(60, p_v22 + h_arm)` (h = 0 outside P, where
the bytes must stay v22's), scored on the panel's original all-row labels.  Reported per
predictor and fold: RMSE on all rows, y <= 3600, y > 3600, COMMON.md's no-NM rows, clean rows,
P, outside P, and every serving regime; rows changed; rows floored.  Per registered contrast:
the paired per-row gain v = (y - p_A)^2 - (y - p_B)^2, its fold mean with 1-, 2- and 3-day
moving-block intervals (exp-050 `block_ci`), the exact SSE reconciliation, segment SSE
partitions and airport SSE; on the board (july_loo and january at the ranking shares, december
beside it at zero weight) the point gain and the recombined 1-, 2- and 3-day intervals
(exp-050 `board_ci`).  Then the registered readings: the advance rule and the iteration-2 route.

Arithmetic on recorded vectors; no fit.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/price.py
    # mechanics only, synthetic labels, scratch:
    ... price.py --dry-run --out-dir <scratch>
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

D, NO, NR = E.D, E.NO, E.NR
CONTROLS = ["const", "gmean", "donor", "shuffle"]
PRIMARY = ("rowonly", "true")                  # (A, B): gain of B over A
CONTRASTS = ([PRIMARY, ("v22", "true")] + [(c, "true") for c in CONTROLS]
             + [("v22", a) for a in E.ARMS if a != "true"]
             + [("rowonly", c) for c in CONTROLS])
REGIMES = ["nm_bulk", "nm_late", "noac", "nonm"]
TOL = 1e-6


def rmse(e: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else None


def load(fold: str, out_dir: Path, dry: bool) -> tuple[pd.DataFrame, dict, dict]:
    fj = json.loads((out_dir / f"analysis/fit-{fold}.json").read_text())
    hp = out_dir / f"work/h-{fold}.parquet"
    if E.sha_file(hp) != fj["h_parquet"]["sha256"]:
        raise SystemExit(f"{hp} moved since its fit record was written")
    Hh = pd.read_parquet(hp)
    # a dry run reads neither the label nor the label-derived `clean` flag
    cols = E.PANEL_LABEL_FREE + ([] if dry else ["clean", "y_true"])
    pan = E.panel(fold, cols)
    if not np.array_equal(Hh.MVT_ID_mvt.to_numpy("float64"), pan.MVT_ID_mvt.to_numpy("float64")):
        raise SystemExit(f"{fold}: h rows are not in panel order")
    pop = E.population(pan, fold)
    if not np.array_equal(pop, Hh["pop"].to_numpy(bool)):
        raise SystemExit(f"{fold}: P moved between the fit and the pricing")
    df = pan.copy()
    df["pop"] = pop
    if dry:
        df["clean"] = True
        df["y_true"] = df.p_v22.to_numpy("float64") + np.random.default_rng(E.SEED).normal(0.0, 170.0, len(df))
    frame = E.departure_frame(E.fold_month(fold))
    df["no_nm"] = E.no_nm_mask(frame, df.MVT_ID_mvt.to_numpy("float64"))
    df["r"] = df.y_true.to_numpy("float64") - df.p_v22.to_numpy("float64")
    pv = df.p_v22.to_numpy("float64")
    P = {"v22": pv.copy()}
    for a in E.ARMS:
        h = Hh[f"h_{a}"].to_numpy("float64")
        if np.any(h[~pop] != 0):
            raise SystemExit(f"{fold}: arm {a} moves a row outside P")
        p = np.maximum(E.MIN_PRED, pv + h)
        if not np.array_equal(p[~pop], pv[~pop]):
            raise SystemExit(f"{fold}: arm {a} changes v22's bytes outside P")
        P[a] = p
    return df, P, fj


def segments(df: pd.DataFrame) -> dict:
    y = df.y_true.to_numpy("float64")
    reg = df.regime.astype(str).to_numpy()
    seg = {"all": np.ones(len(df), bool), "le3600": y <= 3600, "gt3600": y > 3600,
           "no_nm": df.no_nm.to_numpy(bool), "clean": df.clean.to_numpy(bool),
           "P": df["pop"].to_numpy(bool), "not_P": ~df["pop"].to_numpy(bool)}
    for g in REGIMES:
        seg[f"regime_{g}"] = reg == g
    return seg


def predictor_table(df: pd.DataFrame, P: dict, seg: dict) -> dict:
    y = df.y_true.to_numpy("float64")
    pv = P["v22"]
    out = {}
    for name, p in P.items():
        e = y - p
        out[name] = {"rows_changed_vs_v22": int((p != pv).sum()),
                     "rows_at_floor": int((p == E.MIN_PRED).sum()),
                     "segments": {s: {"rows": int(m.sum()), "rmse": rmse(e[m]),
                                      "sse": float(np.sum(e[m] ** 2))} for s, m in seg.items()}}
    return out


def contrast(df: pd.DataFrame, pa: np.ndarray, pb: np.ndarray, seg: dict) -> dict:
    y = df.y_true.to_numpy("float64")
    ea, eb = y - pa, y - pb
    v = ea ** 2 - eb ** 2
    days = df.day.to_numpy()
    rng = np.random.default_rng(E.SEED)            # the same day draws for every contrast
    sse_a, sse_b = float(np.sum(ea ** 2)), float(np.sum(eb ** 2))
    out = {"rows": int(len(v)), "rows_differing": int((pa != pb).sum()),
           "mse_gain_s2_per_row": float(v.mean()),
           "rmse_gain_s": float(np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2))),
           **{k: v_ for k, v_ in NR.block_ci(v, days, rng).items()},
           "sse_a": sse_a, "sse_b": sse_b, "sse_gain": sse_a - sse_b,
           "sse_gain_rowsum": float(v.sum())}
    rec_ok = abs(out["sse_gain"] - out["sse_gain_rowsum"]) <= TOL * max(1.0, sse_a)
    parts = {}
    for name, keys in (("le_gt_3600", ["le3600", "gt3600"]), ("P_notP", ["P", "not_P"]),
                       ("regimes", [f"regime_{g}" for g in REGIMES])):
        s = {k: float(v[seg[k]].sum()) for k in keys}
        parts[name] = {"by_segment_sse_gain": s,
                       "sums_to_total": abs(sum(s.values()) - out["sse_gain_rowsum"]) <= TOL * max(1.0, sse_a)}
        rec_ok &= parts[name]["sums_to_total"]
    out["partitions"] = parts
    out["segments_sse_gain"] = {k: float(v[m].sum()) for k, m in seg.items()}
    out["segments_mse_gain_s2_per_row"] = {k: (float(v[m].mean()) if m.any() else None)
                                           for k, m in seg.items()}
    out["by_airport_sse_gain"] = {a: float(v[m].sum()) for a, m in
                                  ((a, df.airport.astype(str).to_numpy() == a)
                                   for a in sorted(df.airport.astype(str).unique()))}
    out["reconciles"] = bool(rec_ok)
    return out


def board(dfs: dict, P: dict, a: str, b: str) -> dict:
    u0 = {f: P[f][a] - dfs[f].p_v22.to_numpy("float64") for f in E.SHARES}
    u1 = {f: P[f][b] - dfs[f].p_v22.to_numpy("float64") for f in E.SHARES}
    out = {"gain_s": NR.board_point(dfs, u0, u1),
           **NR.board_ci(dfs, u0, u1, np.random.default_rng(E.SEED), lengths=(1, 2, 3)),
           "gain_clean_s": NR.board_point(dfs, u0, u1, clean=True)}
    # recompute from fold MSEs as a reconciliation
    mse = {x: {f: float(np.mean((dfs[f].y_true.to_numpy("float64") - P[f][x]) ** 2)) for f in E.SHARES}
           for x in (a, b)}
    out["gain_s_from_fold_mse"] = float(np.sqrt(sum(E.SHARES[f] * mse[a][f] for f in E.SHARES))
                                        - np.sqrt(sum(E.SHARES[f] * mse[b][f] for f in E.SHARES)))
    out["reconciles"] = abs(out["gain_s"] - out["gain_s_from_fold_mse"]) <= 1e-9
    return out


def readings(C: dict, B: dict, folds: list[str]) -> dict:
    """SPEC 1.11: the advance rule and the iteration-2 route, applied mechanically."""
    key = lambda a, b: f"{b}_over_{a}"
    prim = key(*PRIMARY)
    pts = {f: C[f][prim]["mse_gain_s2_per_row"] for f in folds}
    lo1 = {f: C[f][prim]["ci_block_1d"][0] for f in folds}
    cond = {
        "beats_v22_on_board": B[key("v22", "true")]["gain_s"] > 0,
        "beats_rowonly_on_board": B[prim]["gain_s"] > 0,
        **{f"beats_{c}_on_board": B[key(c, "true")]["gain_s"] > 0 for c in CONTROLS},
        "primary_no_negative_fold_point": all(v >= 0 for v in pts.values()),
        "primary_1d_interval_above_zero_on_2_folds": sum(lo1[f] > 0 for f in folds) >= 2}
    adv = all(cond.values())
    strict = all(C[f][key(a, b)]["mse_gain_s2_per_row"] >= 0 for f in folds
                 for a, b in [PRIMARY, ("v22", "true")] + [(c, "true") for c in CONTROLS])
    g_true = B[prim]["gain_s"]
    g_ctrl = {c: B[key("rowonly", c)]["gain_s"] for c in ["gmean", "donor", "shuffle"]}
    matched = [c for c, g in g_ctrl.items() if g >= g_true]
    signs = {f: np.sign(v) for f, v in pts.items()}
    if adv:
        route = ("A", "the advance rule clears: small shared encoder against the same controls")
    elif g_true > 0 and cond["primary_no_negative_fold_point"] and matched:
        best = max(matched, key=lambda c: (g_ctrl[c], -["gmean", "shuffle", "donor"].index(c)))
        route = ("B", f"a simpler control ({best}) matches the gain: the departure-outcome "
                      "representation as the departure-only test at matched capacity")
    elif any(s > 0 for s in signs.values()) and any(s < 0 for s in signs.values()):
        route = ("C", "the primary contrast changes sign across folds: one narrower "
                      "pre-registered shared key (stand only) at matched capacity")
    else:
        route = ("D", "none of A, B, C: the registered runway-removal sensitivity (stand and "
                      "operator only) at matched capacity")
    return {"primary_fold_points_s2": pts, "primary_fold_1d_lower": lo1, "conditions": cond,
            "advances": adv, "stricter_reading_no_negative_fold_point_on_every_contrast": strict,
            "board_gain_true_over_rowonly_s": g_true, "board_gain_controls_over_rowonly_s": g_ctrl,
            "controls_matching_true": matched, "route": route[0], "route_text": route[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else E.EXP
    if args.dry_run and out_dir == E.EXP:
        raise SystemExit("a dry run writes to --out-dir only")
    OUT = out_dir / "analysis/iter1.json"
    OOF = [out_dir / f"oof/iter1-{f}.parquet" for f in E.FOLDS]
    NO.refuse_existing([OUT] + OOF, args.overwrite)
    print(f"guard: no existing artifact at {OUT} or the three oof files", flush=True)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dfs, PP, C, T, FJ = {}, {}, {}, {}, {}
    for f in E.FOLDS:
        df, P, fj = load(f, out_dir, args.dry_run)
        dfs[f], PP[f], FJ[f] = df, P, fj
        seg = segments(df)
        T[f] = predictor_table(df, P, seg)
        C[f] = {f"{b}_over_{a}": contrast(df, P[a], P[b], seg) for a, b in CONTRASTS}
        c = C[f][f"true_over_rowonly"]
        print(f"  [{f}] true over rowonly {c['mse_gain_s2_per_row']:+.3f} s^2/row "
              f"{[round(x, 3) for x in c['ci_block_1d']]}; true over v22 "
              f"{C[f]['true_over_v22']['mse_gain_s2_per_row']:+.3f}; rowonly over v22 "
              f"{C[f]['rowonly_over_v22']['mse_gain_s2_per_row']:+.3f}", flush=True)
    B = {f"{b}_over_{a}": board(dfs, PP, a, b) for a, b in CONTRASTS}
    R = readings(C, B, E.FOLDS)
    rec = {"what": "exp-054 iteration 1: paired all-row endpoint of every arm",
           "dry_run": args.dry_run, "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("e54.py", "price.py")},
           "shares": E.SHARES, "contrasts": [f"{b}_over_{a}" for a, b in CONTRASTS],
           "primary": f"{PRIMARY[1]}_over_{PRIMARY[0]}",
           "inputs": {f: {"fit_json": str(out_dir / f"analysis/fit-{f}.json"),
                          "h_parquet_sha256": FJ[f]["h_parquet"]["sha256"]} for f in E.FOLDS},
           "folds": {f: {"predictors": T[f], "contrasts": C[f]} for f in E.FOLDS},
           "board": B, "readings": R,
           "all_reconcile": bool(all(C[f][k]["reconciles"] for f in E.FOLDS for k in C[f])
                                 and all(b["reconciles"] for b in B.values())),
           "board_rmse": {x: float(np.sqrt(sum(E.SHARES[f] * np.mean(
               (dfs[f].y_true.to_numpy("float64") - PP[f][x]) ** 2) for f in E.SHARES)))
               for x in ["v22"] + E.ARMS}}
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT] + OOF, args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    for f, p in zip(E.FOLDS, OOF):
        pd.DataFrame({"MVT_ID_mvt": dfs[f].MVT_ID_mvt.to_numpy("float64"), "fold": f,
                      "y_true": dfs[f].y_true.to_numpy("float64"), "y_pred": PP[f]["true"],
                      "y_pred_rowonly": PP[f]["rowonly"], "y_pred_v22": PP[f]["v22"]}).to_parquet(p, index=False)
    b = B["true_over_rowonly"]
    print(f"  PRIMARY board true over rowonly {b['gain_s']:+.5f} s {[round(x, 5) for x in b['ci_block_1d']]}; "
          f"advances {R['advances']}; route {R['route']}: {R['route_text']}; reconcile {rec['all_reconcile']}")
    print(f"written {OUT} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
