"""exp-052-nmv22: the residual experts for one or more folds, with the blindness proof first.

SPEC section 1, per fold, in this order (1.6, with step 3 added after Sol round 1, finding B1):

  1. freeze P from the panel's regime and guard columns (no label) and assert Q08's sha;
  2. FEATURE-LEVEL PROOF: rebuild the BLIND block from the masked month frame, require it
     byte-identical to the frozen feature parquet, then mutate every NM input on every row
     (null; within-airport-day shuffle) and require the BLIND block byte-identical again;
  3. LABEL-FREE PREDICTION-LEVEL PROOF: the 15 BLIND units fitted on p_v22 (v22's fold-side
     prediction: no label, and a function of the NM inputs, so a model that could read an NM
     column would use it) must predict every P row byte-identically from the baseline and from
     each mutated matrix;
  4. only now read the labels, and fit the 15 BLIND experts (5 outer, 10 inner pairs; 1.5);
  5. the same prediction-level check on the 15 fitted BLIND experts.
     `analysis/proof-<fold>.json` is written here;
  6. only then fit the 15 AWARE and the 15 SHAM experts;
  7. write every expert's predictions on the rows it predicts (`work/resid-<fold>.parquet`).

No validation label enters a computation before step 4, and no label vector is formed before it
(after Sol round 2, R2-1).  Step 2 builds the ranking-like month frame itself (`masked_month`):
it reads the raw month and blanks the departure block time and target in place, as the ranking
file has them, before any other operation.  exp-050's `month_frame` also formed and returned the
departure labels, which is why it is no longer called.  The frozen feature parquet is read
column by column, so its label column `y` is never read.  The proof records the time it finished
and the experts record the time the labels were first read.  A proof failure writes the proof
record and stops the job before any label is read (steps 2-3) or before any AWARE fit (step 5).
No score is computed here: this file never compares a prediction with a label.  Every fit launch
holds while the 1-minute load average is above 10.  Outputs are write-once (`--overwrite`).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-052-nmv22/code/rx_fit.py july_loo december january
    # re-fit into a fresh directory (every output goes there):
    ... rx_fit.py july_loo december january --out-dir experiments/exp-052-nmv22/work/refit
    # mechanics only, synthetic target, no label read, scratch outputs:
    ... rx_fit.py july_loo --dry-run --out-dir <scratch>
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

NF, NE, NO, D = C.NF, C.NE, C.NO, C.D


def mutate_airday(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Every NM input permuted independently inside airport x UTC day, on every row
    (departures keyed by ADEP_mvt, arrivals by ADES_mvt)."""
    out = frame.copy()
    dep = out["PHASE_mvt"].to_numpy() == "DEP"
    apt = np.where(dep, out["ADEP_mvt"].astype(str).to_numpy(),
                   out["ADES_mvt"].astype(str).to_numpy())
    day = out["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").fillna("NaT").to_numpy()
    key = np.char.add(np.char.add(apt.astype("U8"), "|"), day.astype("U10"))
    for c in NF.NM_INPUTS:
        perm = C.airday_perm(key, rng)
        out[c] = pd.Series(out[c].array.take(perm), index=out.index)
    return out


def frozen_features(fold: str, cols: list[str]) -> pd.DataFrame:
    """The frozen feature parquet of the fold month, sha-checked as exp-050's loader checks it,
    with only `MVT_ID_mvt` and `cols` read: its label column `y` is never read."""
    month = D.FOLDS[fold].valid_month
    rec = json.loads((C.E50 / "analysis/features.json").read_text())["months"][month]
    p = NF.feat_path(month)
    if C.sha_file(p) != rec["parquet_sha256"]:
        raise SystemExit(f"{p.name} moved since exp-050's analysis/features.json was written")
    return pd.read_parquet(p, columns=["MVT_ID_mvt"] + cols)


def masked_month(month: str) -> pd.DataFrame:
    """The month frame as the ranking file has it, with no label vector formed: right after the
    read, the departure block time and target are blanked in place (the same frame as
    `D.mask_like_ranking`, which also forms and returns the departure labels)."""
    frame = D.load_month(month)
    dep = frame["PHASE_mvt"] == "DEP"
    frame.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    frame.loc[dep, D.TARGET] = np.nan
    return frame


def units() -> list[tuple[str, str, tuple]]:
    """The 15 fitting units of one type: 5 outer (one block left out), 10 inner (a pair)."""
    return ([("o", f"o{b}", (b,)) for b in range(C.K)]
            + [("i", f"i{b}{j}", (b, j)) for b, j in C.PAIRS])


def same_block(A: pd.DataFrame, B: pd.DataFrame, cols: list[str]) -> dict:
    return {c: NE._same(A[c].reset_index(drop=True), B[c].reset_index(drop=True))
            for c in cols}


def nm_changed_rows(N0: pd.DataFrame, N1: pd.DataFrame, keep: np.ndarray) -> int:
    num0, num1 = N0[NF.NM_NUM].to_numpy("float64"), N1[NF.NM_NUM].to_numpy("float64")
    num_ch = ~np.all((num0 == num1) | (np.isnan(num0) & np.isnan(num1)), axis=1)
    cat_ch = np.zeros(len(N0), bool)
    for c in NF.NM_CATS:
        cat_ch |= N0[c].astype(str).to_numpy() != N1[c].astype(str).to_numpy()
    return int(((num_ch | cat_ch) & keep).sum())


def feature_proof(fold: str, frozen: pd.DataFrame, pop_ids: np.ndarray) -> tuple[dict, dict]:
    """Section 1.6 step 2: no label enters.  Returns the record and the mutated BLIND blocks."""
    month = D.FOLDS[fold].valid_month
    masked = masked_month(month)                 # no label vector is formed
    B0 = NF.blind_features(masked)
    N0 = NF.nm_features(masked)
    rec = {"month": month, "departure_rows": int(len(B0)),
           "eligible_rows": int(len(pop_ids)),
           "eligible_ids_sha256": C.sha_arr(np.sort(pop_ids))}
    order_ok = bool(np.array_equal(B0.MVT_ID_mvt.to_numpy("float64"),
                                   frozen.MVT_ID_mvt.to_numpy("float64")))
    fz = same_block(B0, frozen, NF.BLIND_FEATURES) if order_ok else {}
    rec["rebuilt_equals_frozen_parquet"] = {
        "row_order_identical": order_ok, "all_blind_columns_identical": bool(fz) and all(fz.values()),
        "columns_differing": [c for c, v in fz.items() if not v]}
    keep = np.isin(B0.MVT_ID_mvt.to_numpy("float64"), pop_ids)
    ok_all = rec["rebuilt_equals_frozen_parquet"]["all_blind_columns_identical"]
    mutated = {}
    for mode in ("null", "airday_shuffle"):
        if mode == "null":
            mut = NE.mutate_nm(masked, np.random.default_rng(C.SEED_SHUFFLE), "null")
        else:
            mut = mutate_airday(masked, np.random.default_rng(C.SEED_SHUFFLE))
        changed = {c: int((mut[c].astype(str).to_numpy() != masked[c].astype(str).to_numpy()).sum())
                   for c in NF.NM_INPUTS}
        B1 = NF.blind_features(mut)
        N1 = NF.nm_features(mut)
        same_order = bool(np.array_equal(B0.MVT_ID_mvt.to_numpy(), B1.MVT_ID_mvt.to_numpy()))
        fs = same_block(B0, B1, NF.BLIND_FEATURES) if same_order else {}
        nm_rows = nm_changed_rows(N0, N1, keep) if same_order else 0
        ok = same_order and bool(fs) and all(fs.values()) and nm_rows > 0
        ok_all &= ok
        mutated[mode] = B1
        rec[mode] = {"nm_input_values_changed_all_rows": changed,
                     "nm_inputs_changed_total": int(sum(changed.values())),
                     "row_order_identical": same_order,
                     "blind_features_byte_identical": bool(fs) and all(fs.values()),
                     "blind_feature_columns_differing": [c for c, v in fs.items() if not v],
                     "eligible_rows_whose_nm_features_changed": nm_rows,
                     "passes": bool(ok)}
    rec["passes"] = bool(ok_all)
    return rec, mutated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+", choices=C.FOLDS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="mechanics only: synthetic target, no label read, 20 rounds")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="all outputs go there: required with --dry-run; with a real run, a "
                         "re-fit into a fresh directory")
    ap.add_argument("--null", action="store_true",
                    help="SPEC 1.15 (a): fit on r22 permuted among the fitting rows inside "
                         "airport x day (seed 5801); every output is tagged -null")
    args = ap.parse_args()
    if args.dry_run and args.out_dir is None:
        raise SystemExit("--dry-run writes only to --out-dir")
    if args.out_dir is not None and not args.dry_run:
        args.out_dir = args.out_dir.resolve()
        if C.EXP.resolve() not in args.out_dir.parents:
            raise SystemExit("a real re-fit writes inside the experiment directory")
    rounds = 20 if args.dry_run else C.ROUNDS
    out_a = args.out_dir or C.EXP / "analysis"
    out_w = args.out_dir or C.EXP / "work"
    out_a.mkdir(parents=True, exist_ok=True)
    out_w.mkdir(parents=True, exist_ok=True)
    nl = args.null

    def outs(fold):
        return {"proof": out_a / f"{C.tagged('proof', nl)}-{fold}.json",
                "experts": C.experts_json(fold, out_a, nl),
                "resid": C.resid_path(fold, out_w, nl),
                "models": out_w / f"{C.tagged('models', nl)}-{fold}"}
    for f in args.folds:
        o = outs(f)
        NO.refuse_existing([o["proof"], o["experts"], o["resid"], o["models"]], args.overwrite)
    print(f"{C.utc()} no existing artifact at any output path; pinned inputs verified", flush=True)
    voc = json.loads((C.E50 / "work/vocab.json").read_text())
    import lightgbm as lgb

    for fold in args.folds:
        t0 = time.time()
        o = outs(fold)
        o["models"].mkdir(parents=True, exist_ok=True)
        # 1. P, from label-free panel columns
        pan = C.panel(fold, C.LABEL_FREE)
        pop = C.population(pan, fold)
        P = pan[pop].reset_index(drop=True)
        ids = P.MVT_ID_mvt.to_numpy("float64")
        blk = C.blocks(P)
        frozen = frozen_features(fold, NF.AWARE_FEATURES + ["keep_reachable"])
        # 2. the feature-level proof
        proof, mutated = feature_proof(fold, frozen, ids)
        proof["feature_proof_finished_utc"] = C.utc()
        print(f"{C.utc()} {fold}: feature-level proof "
              f"{'PASSES' if proof['passes'] else 'FAILS'}", flush=True)
        if not proof["passes"]:
            NO.refuse_existing([o["proof"]], args.overwrite)
            o["proof"].write_text(json.dumps(proof, indent=2, sort_keys=True, default=float))
            raise SystemExit(f"{fold}: THE FEATURE-LEVEL BLINDNESS PROOF FAILS; stopping "
                             f"before any fit and any score")
        # the matrices, P rows in panel order
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
        moved = int((sham_perm != np.arange(len(sham_perm))).sum())
        Xmut = {m: NE.encode(B.set_index("MVT_ID_mvt").reindex(ids), NF.BLIND_FEATURES, voc)
                for m, B in mutated.items()}
        keep = Fp.keep_reachable.to_numpy(bool)         # four fixed ids (D.keep_reachable)
        # 3. the label-free prediction-level proof, before any label is read
        cols_b = C.COLS["blind"]
        cats_b = [c for c in cols_b if c in NF.CATS]
        t_lf = P.p_v22.to_numpy("float64")
        lfp = {}
        for kind, key, out_blocks in units():
            rows = keep & ~np.isin(blk, out_blocks)
            gate = C.hold_load(f"{fold} label-free blind {key}")
            m = lgb.LGBMRegressor(n_estimators=rounds, n_jobs=C.THREADS,
                                  random_state=C.SEED, **C.PARAMS)
            m.fit(X["blind"][rows], t_lf[rows], categorical_feature=cats_b)
            p0 = np.asarray(m.predict(X["blind"]))
            same = {mode: bool(np.array_equal(p0.view("uint64"),
                                              np.asarray(m.predict(Xm)).view("uint64")))
                    for mode, Xm in Xmut.items()}
            lfp[key] = {"fit_rows": int(rows.sum()), "baseline_pred_sha256": C.sha_arr(p0),
                        "baseline_pred_sd_s": float(np.std(p0)),
                        "corr_pred_with_target": float(np.corrcoef(p0, t_lf)[0, 1]),
                        "byte_identical": same, "passes": all(same.values()), **gate}
        proof["label_free_prediction_proof"] = {
            "target": "p_v22, v22's fold-side prediction (no label; a function of the NM inputs)",
            "rounds": rounds, "models": lfp, "rows_predicted": int(len(P)),
            "passes": all(v["passes"] for v in lfp.values())}
        proof["label_free_prediction_proof_finished_utc"] = C.utc()
        proof["label_free_prediction_proof_finished_unix"] = time.time()
        proof["passes"] = bool(proof["passes"] and proof["label_free_prediction_proof"]["passes"])
        print(f"{C.utc()} {fold}: label-free prediction-level proof over {len(lfp)} BLIND units "
              f"{'PASSES' if proof['passes'] else 'FAILS'}", flush=True)
        if not proof["passes"]:
            NO.refuse_existing([o["proof"]], args.overwrite)
            o["proof"].write_text(json.dumps(proof, indent=2, sort_keys=True, default=float))
            raise SystemExit(f"{fold}: THE LABEL-FREE PREDICTION-LEVEL PROOF FAILS; stopping "
                             f"before any label is read")
        # 4. labels, only now, and only for fitting
        labels_read = None
        if args.dry_run:
            r = np.random.default_rng(0).normal(0.0, 170.0, len(P))
            fitmask = keep.copy()
        else:
            labels_read = {"utc": C.utc(), "unix": time.time()}
            y = C.panel(fold, ["MVT_ID_mvt", "y_true"])
            y = y.set_index("MVT_ID_mvt").y_true.reindex(ids).to_numpy("float64")
            r = y - P.p_v22.to_numpy("float64")
            fitmask = keep & (y > 0)
        null_rec = None
        if nl:
            key = (P.airport.astype(str) + "|" + P.day.astype(str)).to_numpy()
            r0 = r
            r = C.null_residual(r0, fitmask, key)
            null_rec = {"seed": C.SEED_NULL, "rows_permuted_among": int(fitmask.sum()),
                        "rows_whose_target_moved": int((r != r0).sum()),
                        "r_null_sha256": C.sha_arr(r)}
        rec = {"fold": fold, "dry_run": bool(args.dry_run), "null": null_rec,
               "labels_first_read": labels_read,
               "population_rows": int(len(P)),
               "population_ids_sha256": C.sha_arr(np.sort(ids)),
               "block_rows": {int(b): int((blk == b).sum()) for b in range(C.K)},
               "fitting_rows_all_blocks": int(fitmask.sum()),
               "fitting_rows_excluded": int((~fitmask).sum()),
               "params": C.PARAMS, "rounds": rounds, "seed": C.SEED, "threads": C.THREADS,
               "sham": {"seed": C.SEED_SHAM, "rows_whose_nm_block_moved": moved,
                        "perm_sha256": C.sha_arr(sham_perm)},
               "pinned_inputs": C.check_pinned(), "fits": []}
        U = {f"u_{T}_o": np.full(len(P), np.nan) for T in C.TYPES}
        U.update({f"u_{T}_i{b}": np.full(len(P), np.nan) for T in C.TYPES for b in range(C.K)})
        models = {T: {} for T in C.TYPES}

        def fit_type(T: str) -> None:
            cols = C.COLS[T]
            cats = [c for c in cols if c in NF.CATS]
            for kind, key, out_blocks in units():
                rows = fitmask & ~np.isin(blk, out_blocks)
                gate = C.hold_load(f"{fold} {T} {key}")
                t1 = time.time()
                m = lgb.LGBMRegressor(n_estimators=rounds, n_jobs=C.THREADS,
                                      random_state=C.SEED, **C.PARAMS)
                m.fit(X[T][rows], r[rows], categorical_feature=cats)
                sec = time.time() - t1
                path = o["models"] / f"{T}-{key}.txt"
                NO.refuse_existing([path], args.overwrite)
                m.booster_.save_model(str(path))
                models[T][key] = m
                pr = np.isin(blk, out_blocks)
                pred = m.predict(X[T][pr])
                if kind == "o":
                    U[f"u_{T}_o"][pr] = pred
                else:
                    bb, jj = out_blocks
                    on_j, on_b = pr & (blk == jj), pr & (blk == bb)
                    # E_{bj} on block j is scoring block b's development value, and vice versa
                    U[f"u_{T}_i{bb}"][on_j] = pred[blk[pr] == jj]
                    U[f"u_{T}_i{jj}"][on_b] = pred[blk[pr] == bb]
                rec["fits"].append({"type": T, "unit": key, "left_out_blocks": list(out_blocks),
                                    "fit_rows": int(rows.sum()), "pred_rows": int(pr.sum()),
                                    "seconds": round(sec, 2), "model": str(path.relative_to(C.ROOT))
                                    if not args.dry_run else str(path),
                                    "model_sha256": C.sha_file(path), **gate})
                print(f"  {C.utc()} {fold} {T:5s} {key:3s} fit {int(rows.sum()):>7,} rows "
                      f"{sec:5.1f}s load {gate['launch_load_1m']:.2f}", flush=True)

        fit_type("blind")
        # 5. the prediction-level check on the fitted experts, before any AWARE or SHAM fit
        pp = {}
        for key, m in models["blind"].items():
            p0 = np.asarray(m.predict(X["blind"]))
            same = {mode: bool(np.array_equal(p0.view("uint64"),
                                              np.asarray(m.predict(Xm)).view("uint64")))
                    for mode, Xm in Xmut.items()}
            pp[key] = {"baseline_pred_sha256": C.sha_arr(p0), "byte_identical": same,
                       "passes": all(same.values())}
        proof["prediction_proof"] = {"models": pp, "rows_predicted": int(len(P)),
                                     "passes": all(v["passes"] for v in pp.values())}
        proof["passes"] = bool(proof["passes"] and proof["prediction_proof"]["passes"])
        proof["prediction_proof_finished_utc"] = C.utc()
        NO.refuse_existing([o["proof"]], args.overwrite)
        o["proof"].write_text(json.dumps(proof, indent=2, sort_keys=True, default=float))
        print(f"{C.utc()} {fold}: prediction-level proof over {len(pp)} BLIND experts "
              f"{'PASSES' if proof['passes'] else 'FAILS'}", flush=True)
        if not proof["passes"]:
            raise SystemExit(f"{fold}: THE BLINDNESS PROOF FAILS; stopping before the AWARE fit "
                             f"and every score")
        rec["proof"] = str(o["proof"].relative_to(C.ROOT)) if not args.dry_run else str(o["proof"])
        rec["proof_sha256"] = C.sha_file(o["proof"])
        # 6. AWARE and SHAM
        fit_type("aware")
        fit_type("sham")
        # 7. the per-row values
        for T in C.TYPES:
            if np.isnan(U[f"u_{T}_o"]).any():
                raise SystemExit(f"{fold}: an outer value of {T} is missing")
            for b in range(C.K):
                miss = np.isnan(U[f"u_{T}_i{b}"]) & (blk != b)
                if miss.any() or not np.isnan(U[f"u_{T}_i{b}"][blk == b]).all():
                    raise SystemExit(f"{fold}: inner values of {T} for block {b} are malformed")
        out = pd.DataFrame({"MVT_ID_mvt": ids, "block": blk, **U})
        NO.refuse_existing([o["resid"]], args.overwrite)
        out.to_parquet(o["resid"], index=False)
        rec["resid"] = str(o["resid"].relative_to(C.ROOT)) if not args.dry_run else str(o["resid"])
        rec["resid_sha256"] = C.sha_file(o["resid"])
        rec["seconds"] = round(time.time() - t0, 1)
        rec["finished_utc"] = C.utc()
        NO.refuse_existing([o["experts"]], args.overwrite)
        o["experts"].write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        print(f"{C.utc()} {fold}: 45 fits, written {o['resid'].name} ({rec['seconds']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
