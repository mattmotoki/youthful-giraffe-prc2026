"""exp-052-nmv22: iteration 2's matched capacity control (SPEC 2.2), no label read.

Every frozen 400-round expert of iteration 1 (real and NULL, BLIND / AWARE / SHAM, 15 units per
fold, sha-checked against its expert record) is evaluated at its first k trees, k in G.  With no
row or column sampling and a fixed learning rate, the first k trees ARE the k-round fit of the
same recipe, so each k is a matched capacity control at zero new fits.

Check (a), in this script: at k = 400 every recomputed outer and inner value must equal the value
iteration 1 stored in work/resid[-null]-<fold>.parquet (bitwise, else |diff| < 1e-9 s, else stop).

Check (b), `--exactness-fit`, after the truncated values exist: ONE refit of july_loo AWARE o0 at
100 rounds (the same fitting rows and target as iteration 1's o0; it reads those fitting labels
to fit, as SPEC 1.6 step 3 did) must predict block 0's P rows as the 400-round o0 truncated at
k = 100 does.  No score is computed here.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_trunc.py
    .venv/bin/python experiments/exp-052-nmv22/code/rx_trunc.py --exactness-fit
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
import rx_common as C                                                 # noqa: E402

NF, NE, NO = C.NF, C.NE, C.NO
G = [10, 25, 50, 100, 200, 400]                                       # SPEC 2.2
TOL = 1e-9


def trunc_path(fold: str, null: bool) -> Path:
    return C.EXP / "work" / f"{C.tagged('trunc', null)}-{fold}.parquet"


def units() -> list[tuple[str, int, tuple]]:
    return [("o", b, (b,)) for b in range(C.K)] + [("i", b, (b, j)) for b, j in C.PAIRS]


def unit_key(kind: str, out_blocks: tuple) -> str:
    return f"o{out_blocks[0]}" if kind == "o" else f"i{out_blocks[0]}{out_blocks[1]}"


def matrices(fold: str, voc: dict) -> dict:
    """P, its blocks and the three matrices, exactly as rx_fit builds them (no label)."""
    pan = C.panel(fold, C.LABEL_FREE)
    pop = C.population(pan, fold)
    P = pan[pop].reset_index(drop=True)
    ids = P.MVT_ID_mvt.to_numpy("float64")
    blk = C.blocks(P)
    frozen = C.features(fold, NF.AWARE_FEATURES + ["keep_reachable"])
    Fp = frozen.set_index("MVT_ID_mvt").reindex(ids)
    if Fp[NF.BLIND_FEATURES[0]].isna().any():
        raise SystemExit(f"{fold}: a P row has no frozen feature row")
    X = {"blind": NE.encode(Fp, NF.BLIND_FEATURES, voc),
         "aware": NE.encode(Fp, NF.AWARE_FEATURES, voc)}
    sham_perm = C.airday_perm((P.airport.astype(str) + "|" + P.day.astype(str)).to_numpy(),
                              np.random.default_rng(C.SEED_SHAM))
    Xs = X["aware"].copy()
    for c in NF.NM_FEATURES:
        Xs[c] = pd.Series(X["aware"][c].array.take(sham_perm), index=Xs.index)
    X["sham"] = Xs
    return {"P": P, "ids": ids, "blk": blk, "X": X, "Fp": Fp,
            "sham_perm_sha256": C.sha_arr(sham_perm)}


def matrix_sha(X: pd.DataFrame) -> str:
    """sha256 over every column's codes or values, in column order."""
    cols = []
    for c in X.columns:
        s = X[c]
        cols.append(s.cat.codes.to_numpy("float64") if hasattr(s, "cat") else s.to_numpy("float64"))
    return C.sha_arr(*cols)


def truncate(fold: str, null: bool, M: dict, lgb) -> tuple[pd.DataFrame, dict]:
    ej = C.experts_json(fold, None, null)
    rec = json.loads(ej.read_text())
    proof = json.loads((C.ROOT / rec["proof"]).read_text())
    if not (proof["passes"] and proof["prediction_proof"]["passes"]):
        raise SystemExit(f"{fold}: the blindness proof did not pass")
    if rec["sham"]["perm_sha256"] != M["sham_perm_sha256"]:
        raise SystemExit(f"{fold}: the rebuilt SHAM permutation is not iteration 1's")
    rp = C.resid_path(fold, None, null)
    if C.sha_file(rp) != rec["resid_sha256"]:
        raise SystemExit(f"{rp.name} moved since iteration 1 wrote it")
    R = pd.read_parquet(rp)
    ids, blk, X = M["ids"], M["blk"], M["X"]
    if not np.array_equal(R.MVT_ID_mvt.to_numpy("float64"), ids):
        raise SystemExit(f"{fold}: iteration 1's rows are not P in panel order")
    fits = {(f["type"], f["unit"]): f for f in rec["fits"]}
    U = {}
    for k in G:
        for T in C.TYPES:
            U[f"u_{T}_o_k{k}"] = np.full(len(ids), np.nan)
            for b in range(C.K):
                U[f"u_{T}_i{b}_k{k}"] = np.full(len(ids), np.nan)
    checked = 0
    for T in C.TYPES:
        for kind, b, out_blocks in units():
            key = unit_key(kind, out_blocks)
            f = fits[(T, key)]
            path = C.ROOT / f["model"]
            if C.sha_file(path) != f["model_sha256"]:
                raise SystemExit(f"{path} moved since iteration 1 wrote it")
            bst = lgb.Booster(model_file=str(path))
            if bst.num_trees() != C.ROUNDS:
                raise SystemExit(f"{path.name}: {bst.num_trees()} trees, not {C.ROUNDS}")
            checked += 1
            pr = np.isin(blk, out_blocks)
            Xp = X[T][pr]
            for k in G:
                pred = np.asarray(bst.predict(Xp, num_iteration=k, num_threads=C.THREADS))
                if kind == "o":
                    U[f"u_{T}_o_k{k}"][pr] = pred
                else:
                    bb, jj = out_blocks
                    U[f"u_{T}_i{bb}_k{k}"][pr & (blk == jj)] = pred[blk[pr] == jj]
                    U[f"u_{T}_i{jj}_k{k}"][pr & (blk == bb)] = pred[blk[pr] == bb]
    # completeness, and check (a): k = 400 reproduces iteration 1
    diffs, bitwise = {}, True
    for T in C.TYPES:
        for k in G:
            if np.isnan(U[f"u_{T}_o_k{k}"]).any():
                raise SystemExit(f"{fold}: an outer value of {T} at k={k} is missing")
            for b in range(C.K):
                v = U[f"u_{T}_i{b}_k{k}"]
                if np.isnan(v[blk != b]).any() or not np.isnan(v[blk == b]).all():
                    raise SystemExit(f"{fold}: inner values of {T} block {b} k={k} are malformed")
        for col in [f"u_{T}_o"] + [f"u_{T}_i{b}" for b in range(C.K)]:
            old = R[col].to_numpy("float64")
            new = U[f"{col}_k400"]
            fin = np.isfinite(old)
            if not np.array_equal(fin, np.isfinite(new)):
                raise SystemExit(f"{fold} {col}: the k = 400 values cover other rows")
            same = np.array_equal(old[fin].view("uint64"), new[fin].view("uint64"))
            bitwise &= same
            diffs[col] = 0.0 if same else float(np.max(np.abs(old[fin] - new[fin])))
    worst = max(diffs.values())
    if worst >= TOL:
        raise SystemExit(f"{fold}{' NULL' if null else ''}: k = 400 differs from iteration 1 "
                         f"by {worst!r} s; stopping (SPEC 2.2 (a))")
    out = pd.DataFrame({"MVT_ID_mvt": ids, "block": blk, **U})
    info = {"models_checked": checked, "experts_json": str(ej.relative_to(C.ROOT)),
            "experts_json_sha256": C.sha_file(ej), "resid_sha256": rec["resid_sha256"],
            "check_a_k400_bitwise": bool(bitwise), "check_a_max_abs_diff_s": worst,
            "check_a_columns": len(diffs)}
    return out, info


def exactness_fit(lgb, voc: dict) -> dict:
    """SPEC 2.2 (b): one refit of july_loo AWARE o0 at 100 rounds against the truncation."""
    fold, T, b = "july_loo", "aware", 0
    M = matrices(fold, voc)
    P, ids, blk, X, Fp = M["P"], M["ids"], M["blk"], M["X"], M["Fp"]
    keep = Fp.keep_reachable.to_numpy(bool)
    y = C.panel(fold, ["MVT_ID_mvt", "y_true"]).set_index("MVT_ID_mvt").y_true
    y = y.reindex(ids).to_numpy("float64")
    r = y - P.p_v22.to_numpy("float64")
    fitmask = keep & (y > 0)
    rows = fitmask & (blk != b)
    rec1 = json.loads(C.experts_json(fold).read_text())
    f1 = [f for f in rec1["fits"] if f["type"] == T and f["unit"] == f"o{b}"][0]
    if int(rows.sum()) != f1["fit_rows"]:
        raise SystemExit("the refit's rows are not iteration 1's o0 rows")
    gate = C.hold_load(f"{fold} {T} o{b} at 100 rounds")
    t1 = time.time()
    cats = [c for c in C.COLS[T] if c in NF.CATS]
    m = lgb.LGBMRegressor(n_estimators=100, n_jobs=C.THREADS, random_state=C.SEED, **C.PARAMS)
    m.fit(X[T][rows], r[rows], categorical_feature=cats)
    sec = time.time() - t1
    pr = blk == b
    p_refit = np.asarray(m.predict(X[T][pr]))
    tr = pd.read_parquet(trunc_path(fold, False), columns=["MVT_ID_mvt", "u_aware_o_k100"])
    if not np.array_equal(tr.MVT_ID_mvt.to_numpy("float64"), ids):
        raise SystemExit("the truncation file's rows are not P in panel order")
    p_tr = tr.u_aware_o_k100.to_numpy("float64")[pr]
    same = bool(np.array_equal(p_refit.view("uint64"), p_tr.view("uint64")))
    diff = 0.0 if same else float(np.max(np.abs(p_refit - p_tr)))
    return {"fold": fold, "type": T, "unit": f"o{b}", "rounds": 100, "fit_rows": int(rows.sum()),
            "pred_rows": int(pr.sum()), "seconds": round(sec, 2), **gate,
            "bitwise_equal_to_truncation": same, "max_abs_diff_s": diff,
            "passes": bool(diff < TOL)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exactness-fit", action="store_true",
                    help="SPEC 2.2 (b): the one refit, after the truncated values exist")
    args = ap.parse_args()
    import lightgbm as lgb
    voc = json.loads((C.E50 / "work/vocab.json").read_text())
    OUT = C.EXP / "analysis/trunc.json"
    if args.exactness_fit:
        OUTB = C.EXP / "analysis/trunc-exactness.json"
        NO.refuse_existing([OUTB], args.overwrite)
        rec = {"what": "SPEC 2.2 (b): one refit at 100 rounds against the truncated 400-round "
                       "model", "pinned_inputs": C.check_pinned(),
               "code_sha256": C.sha_file(Path(__file__)), "trunc_json_sha256": C.sha_file(OUT),
               **exactness_fit(lgb, voc)}
        NO.refuse_existing([OUTB], args.overwrite)
        OUTB.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        print(json.dumps(rec, indent=2, default=float))
        if not rec["passes"]:
            raise SystemExit("SPEC 2.2 (b) FAILS: the truncation is not the k-round fit")
        return
    outs = [OUT] + [trunc_path(f, nl) for f in C.FOLDS for nl in (False, True)]
    NO.refuse_existing(outs, args.overwrite)
    rec = {"what": "exp-052 iteration 2 (SPEC 2.2): frozen experts at their first k trees",
           "G": G, "pinned_inputs": C.check_pinned(),
           "code_sha256": C.sha_file(Path(__file__)), "lightgbm": lgb.__version__, "runs": {}}
    t0 = time.time()
    for f in C.FOLDS:
        M = matrices(f, voc)
        shas = {T: matrix_sha(X) for T, X in M["X"].items()}
        for nl in (False, True):
            t1 = time.time()
            out, info = truncate(f, nl, M, lgb)
            p = trunc_path(f, nl)
            NO.refuse_existing([p], args.overwrite)
            out.to_parquet(p, index=False)
            info.update({"matrix_sha256": shas, "sham_perm_sha256": M["sham_perm_sha256"],
                         "trunc": str(p.relative_to(C.ROOT)), "trunc_sha256": C.sha_file(p),
                         "seconds": round(time.time() - t1, 1), "finished_utc": C.utc()})
            rec["runs"].setdefault("null" if nl else "real", {})[f] = info
            print(f"{C.utc()} {f}{' NULL' if nl else ''}: {info['models_checked']} models, "
                  f"k=400 bitwise {info['check_a_k400_bitwise']} (max diff "
                  f"{info['check_a_max_abs_diff_s']:.3g}), written {p.name} "
                  f"({info['seconds']:.0f}s)", flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
