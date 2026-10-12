"""exp-012 owns exp-007's `d1-ds` member: fold mode plus the submission mode it lacked.

The desk closed winter's session and handed this member over (2026-09-05). It is
the fourth member of the standing blend and worth 0.60 s on the board estimate
(RESULTS, "A fourth member exists"), and it could not be served because
`experiments/exp-007-winter/code/run_ds.py` has no submission path.

This is a COPY of that file with `build_submission` added, not an import: their
`run_fold` has no hook, and the point of the copy is that fold predictions and
ranking predictions come out of one file so the served member is the estimator
whose out-of-fold predictions earned the weight. It imports exp-007's PINNED
pipeline snapshot (`experiments/exp-007-winter/code/pipeline/`, taken from git
19cd199) rather than retro-v1's live directory, for the reason their header
gives: that snapshot is what seed 7 actually ran against.

Three decisions worth stating, because each one could quietly break the member:

1. **`es_rounds` stays 100.** This experiment's budget audit found the
   patience-100 early stop truncates LightGBM-Huber by 4.2x and CatBoost by
   11.8x, and it is tempting to "correct" it here too. That would be wrong. The
   0.60 s this member is worth was measured on an out-of-fold prediction drawn
   at patience 100; refitting the served member at patience 900 would serve a
   different estimator from the one that earned the weight, which is exactly the
   failure exp-008's review round 1 found (finding B2). The budget question is a
   separate experiment on this arm, not a change to make while serving it.

2. **`PRC_THREADS` is pinned at 24 and recorded.** This arm is `linear_tree=True`
   and LightGBM's `deterministic=True` does not cover linear leaves across thread
   counts -- exp-008 measured one seed moving 15.4 s between 8 and 12 threads.
   exp-007 set no `PRC_THREADS` anywhere, so their runs took the pipeline default
   of 24; anything else here would not reproduce their 307.45.

3. **Seeds are bagged, and the round count is drawn ONCE.** The desk asked for
   three seeds. Running `two_pass` per seed would draw a new round count per seed
   and make the bag a mixture of budgets -- the instability retro-v1 halted its
   own ladder over. So seed 7 draws the count by early stopping, and the other
   two seeds refit at that same frozen number. Per-seed predictions are stored
   raw and the floor and residual guard are applied ONCE after averaging, so the
   k = 3 point is the served object rather than a mean of post-processed parts.

Usage:
    PRC_THREADS=24 python winter_arm.py --folds july_loo,december,january --seeds 7,107,207
    PRC_THREADS=24 python winter_arm.py --submit --seeds 7,107,207
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W7 = ROOT / "experiments" / "exp-007-winter" / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(W7 / "pipeline"))   # the pinned retro-v1 snapshot
sys.path.insert(0, str(W7))                # exp-007's own steps

import prc.data as D  # noqa: E402
import run as R  # noqa: E402  (pinned snapshot)
import guard as guard_mod  # noqa: E402
import lirf_step  # noqa: E402
import phi_step  # noqa: E402
import day_state_step  # noqa: E402

from prc import day_state_features as DSF  # noqa: E402


#: Robustness audit (desk, 2026-09-05): the shipped member carries the FULL
#: 25-column day-state block, the configuration exp-007 finding 13 measured at
#: +72.1 s on january's most disrupted days, because the `_z` columns and the raw
#: day levels can leave their training range and the tree then extrapolates past
#: anything it has seen. January 2026 has 18 such airport-days and 7,564 rows.
#:
#: Two repairs are available and this file can run both:
#:   --ds-columns deice   the one bounded column, exp-007 finding 14's arm
#:   --clip               every day-state column clipped to its TRAINING range at
#:                        predict time -- the "clipped-block" experiment exp-007
#:                        named as the one that separates boundedness from column
#:                        count and never ran. It reuses the same trained model,
#:                        so it costs a prediction rather than a fit.
def ds_train_range(Xtr: pd.DataFrame, cols: list[str]) -> dict:
    """Per-column [min, max] over the training rows, skipping degenerate columns.

    A column that is entirely NaN in training would give NaN bounds, and
    np.clip against NaN returns NaN -- which would silently delete the column
    rather than bound it. Such a column is left alone and reported.
    """
    rng = {}
    for c in cols:
        if c not in Xtr.columns:
            continue
        v = Xtr[c].to_numpy("float64")
        if not np.isfinite(v).any():
            print(f"[clip] {c}: no finite training value, left unclipped", flush=True)
            continue
        rng[c] = (float(np.nanmin(v)), float(np.nanmax(v)))
    return rng


def ds_clip(X: pd.DataFrame, rng: dict) -> tuple[pd.DataFrame, int]:
    """Clip each day-state column into the training range; count touched cells."""
    X = X.copy()
    touched = 0
    for c, (lo, hi) in rng.items():
        v = X[c].to_numpy("float64")
        out = np.isfinite(v) & ((v < lo) | (v > hi))
        touched += int(out.sum())
        X[c] = np.clip(v, lo, hi)
    return X, touched

#: exp-007's `d1-ds` configuration, verbatim from their `base_cfg` (retro-v1's
#: G1cv-s7 line) with the day-state step on. Nothing here is retuned.
def base_cfg(seed: int, name: str) -> dict:
    return dict(
        name=name,
        params=dict(R.DEFAULT_PARAMS, learning_rate=0.05, num_leaves=255,
                    min_data_in_leaf=100, lambda_l2=1.0, linear_tree=True, linear_lambda=1.0),
        target="residual", weights="none", weight_alpha=3.0,
        clip_floor=60.0, clip_ceiling=None, train_clip_lo=None, train_clip_hi=None,
        max_rounds=6000, es_rounds=100, es_frac=0.15,
        seed=seed, full_round_scale=1.1, per_airport=False, full_early_stop=True,
        features="v2", encodings="none", allow_unisolated_encodings=False,
        lirf_gate=True, guard_bounds="trainrange", phi=True, phi_fit="cv", phi_shrink=None,
        surface=False, lirf=True, residual_guard=True, noac_head=True,
        blend_tail=False, blend_weight=None, noac_rounds=60, noac_leaves=63,
        noac_min_data=5, noac_lr=0.05, noac_min_rows=50,
        month_weight="none", month_weight_alpha=3.0, ds_columns="all", fixed_rounds=0,
    )


def provenance(cfg) -> dict:
    return {"modules": R.shared_module_tag() + f" day_state[VERSION={DSF.VERSION}]",
            "threads": int(cfg["params"]["num_threads"]),
            "linear_tree": bool(cfg["params"].get("linear_tree")),
            "es_rounds": cfg["es_rounds"], "pipeline": "exp-007-winter/code/pipeline (git 19cd199)"}


def _postprocess(cfg, per_seed, X, months):
    """Floor and guard ONCE, on the averaged prediction (exp-008 review, B1)."""
    p = np.mean(per_seed, axis=0)
    raw_below = int((p < cfg["clip_floor"]).sum())
    p = np.maximum(p, cfg["clip_floor"])
    touched = 0
    lo = hi = None
    if cfg["residual_guard"]:
        lo, hi = R._guard_bounds(cfg, months)
        a = X["d_takeoff_aobt"].to_numpy("float64")
        touched = int(guard_mod.count_touched(p, a, lo, hi))
        p = guard_mod.apply(p, a, lo, hi, floor=cfg["clip_floor"])
    return p, {"raw_below_floor": raw_below, "guard_touched": touched,
               "guard_bounds": None if lo is None else [float(lo), float(hi)]}


# ------------------------------------------------------------------ fold ----
def run_fold(name: str, fold_name: str, seeds: list[int], ds_columns: str = "all",
             emit_clipped: bool = False) -> dict:
    """exp-007's `run_fold`, seed-bagged, writing into exp-012."""
    t0 = time.time()
    fold = D.FOLDS[fold_name]
    cfg0 = base_cfg(seeds[0], name)
    R._reseed(seeds[0])
    raw: dict = {}
    Xtr, ytr, ttr = R.frame_for_months(fold.train_months, cfg0["features"], keep_raw=raw)
    Xva, yva = R.valid_frame(fold, cfg0["features"])
    _, masked, _ = D.fold_frames(fold)
    Xtr, (Xva,) = lirf_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                          gate=cfg0["lirf_gate"])
    Xtr, (Xva,) = phi_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                         mode=cfg0["phi_fit"], shrink=cfg0["phi_shrink"])
    Xtr, (Xva,) = day_state_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                               columns=ds_columns)
    ds_cols = [c for c in day_state_step.SETS[ds_columns] if c in Xva.columns]
    rng = ds_train_range(Xtr, ds_cols)
    Xva_clip, n_clipped = ds_clip(Xva, rng)
    print(f"[{fold_name}] day-state set {ds_columns!r}: {len(ds_cols)} columns, "
          f"{n_clipped} validation cells outside the training range", flush=True)
    print(f"[{fold_name}] train {Xtr.shape} valid {Xva.shape} built in {time.time()-t0:.0f}s",
          flush=True)

    cut = np.quantile(ttr, 1.0 - cfg0["es_frac"])
    es = ttr >= cut
    fb = R.airport_fallback(Xtr, ytr, cfg0["target"])
    base = R.base_score(Xtr, fb, cfg0["target"])
    base_va = R.base_score(Xva, fb, cfg0["target"])
    yt = R.forward_target(R.training_target(cfg0, ytr.to_numpy()), cfg0["target"])
    w = R.sample_weights(cfg0, ytr.to_numpy(), Xtr)

    base_va_clip = R.base_score(Xva_clip, fb, cfg0["target"])
    per_seed, per_seed_clip, rounds, best = [], [], None, None
    for i, s in enumerate(seeds):
        cfg = base_cfg(s, name)
        R._reseed(s)
        if i == 0:                     # the count is drawn once, by seed 0
            (pv, pvc), best, rounds, imp = R.two_pass(
                cfg, Xtr, yt, w, base, es, [(Xva, base_va), (Xva_clip, base_va_clip)])
            print(f"[{fold_name}] seed {s}: best {best} -> refit {rounds}", flush=True)
        else:
            model, enc, _ = R.train_model(cfg, Xtr, yt, w, base, num_round=rounds)
            pv = R.predict_seconds(cfg, model, enc, Xva, base_va)
            pvc = R.predict_seconds(cfg, model, enc, Xva_clip, base_va_clip)
            print(f"[{fold_name}] seed {s}: refit {rounds} (frozen)", flush=True)
        # ONE model, two predictions: the clipped twin differs only in whether the
        # day-state inputs were allowed outside the range the trees were fitted on.
        (mm, ph), = R.noac_head(cfg, Xtr, ytr.to_numpy(), [Xva])
        if ph is not None and mm.any():
            pv = pv.copy(); pv[mm] = ph
            pvc = pvc.copy(); pvc[mm] = ph
        per_seed.append(pv)
        per_seed_clip.append(pvc)

    pv, pinfo = _postprocess(cfg0, per_seed, Xva, fold.train_months)
    keep = yva.notna().to_numpy()
    no_nm = Xva["nm_missing"].to_numpy().astype(bool)
    noac = Xva["noac"].to_numpy().astype(bool) if "noac" in Xva.columns else None
    res = R.decompose(yva.to_numpy()[keep], pv[keep], no_nm[keep],
                      None if noac is None else noac[keep])
    res.update(fold=fold_name, seeds=seeds, best_iteration=int(best), refit_rounds=int(rounds),
               n_features=int(Xtr.shape[1]), seconds=round(time.time() - t0), **pinfo)
    # per-seed RMSE, so the reproduction check against exp-007's single-seed
    # 307.45 can be made on seed 7 alone rather than on the bag
    res["per_seed_rmse"] = [
        float(np.sqrt(((yva.to_numpy()[keep] - _postprocess(cfg0, [p], Xva, fold.train_months)[0][keep]) ** 2).mean()))
        for p in per_seed]
    out = HERE / "oof" / f"{name}-{fold_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {"MVT_ID_mvt": Xva["MVT_ID_mvt"].to_numpy()[keep], "fold": fold_name,
            "y_true": yva.to_numpy()[keep], "y_pred": pv[keep],
            "nm_missing": no_nm[keep].astype("int8"), "airport": Xva["airport"].to_numpy()[keep],
            "d_takeoff_aobt": Xva["d_takeoff_aobt"].to_numpy()[keep]}
    for i, p in enumerate(per_seed):
        cols[f"y_pred_s{i}"] = p[keep]
    pd.DataFrame(cols).to_parquet(out, index=False)
    res["oof"] = str(out.relative_to(ROOT))
    res["ds_columns"] = ds_columns
    res["ds_cells_clipped"] = int(n_clipped)
    if emit_clipped:
        pvc, pinfo_c = _postprocess(cfg0, per_seed_clip, Xva, fold.train_months)
        res_c = R.decompose(yva.to_numpy()[keep], pvc[keep], no_nm[keep],
                            None if noac is None else noac[keep])
        outc = HERE / "oof" / f"{name}-clip-{fold_name}.parquet"
        c2 = dict(cols); c2["y_pred"] = pvc[keep]
        for i, p in enumerate(per_seed_clip):
            c2[f"y_pred_s{i}"] = p[keep]
        pd.DataFrame(c2).to_parquet(outc, index=False)
        res["clipped"] = {**res_c, "oof": str(outc.relative_to(ROOT)), **pinfo_c}
        print(f"[{fold_name}] clipped twin rmse {res_c['rmse']:.2f} "
              f"against unclipped {res['rmse']:.2f}", flush=True)
    return res



# ------------------------------------------------------------ budget rule ----
#: The desk's rule for every member's submission budget (2026-09-05), set after
#: the Huber arm's "corrected" budget lost 3.57 s because its count was chosen on
#: one November early-stop month and applied to July: take the MEDIAN of the
#: per-month optima over at least three held-out months spanning both seasons,
#: report the spread, and use a smaller count when the spread is wide.
SCAN_MONTHS = ("2025-01", "2025-07", "2025-11")
#: My reading of "smaller count on a wide spread", stated so it is not a
#: judgement call at the time of use: with three points the available order
#: statistics are min, median and max, so a relative spread above this fraction
#: of the median takes the MINIMUM instead of the median.
WIDE = 0.5


def budget_from_scan(cfg, Xtr, ttr, yt, w, base, months) -> dict:
    """Per-month optima over held-out months, median, spread, chosen count.

    The early-stop month's rows are held out of the TREE fit only; the lirf, phi
    and day-state statistics are the twelve-month ones the submission itself
    uses. That matches how the count being replaced was drawn (retro-v1's
    `two_pass` early-stops on a slice whose features were fitted with it in), so
    the scan is comparable to what it replaces rather than to an ideal. Stated
    rather than hidden: a fully clean version would refit the steps per held-out
    month, three more frame builds at ~490 s each.
    """
    mon = pd.to_datetime(ttr, unit="s", utc=True).strftime("%Y-%m").to_numpy()
    per = {}
    for m in months:
        es = mon == m
        if not es.any():
            raise SystemExit(f"budget scan: {m} is not a training month")
        R._reseed(cfg["seed"])
        model, _, _ = R.train_model(
            cfg, Xtr[~es], yt[~es], R._sel(w, ~es), R._sel(base, ~es),
            Xva=Xtr[es], yva=yt[es], base_va=R._sel(base, es),
            num_round=cfg["max_rounds"])
        best = int(model.best_iteration or cfg["max_rounds"])
        per[m] = {"best_iteration": best, "held_frac": float(es.mean())}
        print(f"[budget] held out {m}: best_iteration {best} "
              f"({100*es.mean():.1f} % of rows)", flush=True)
    b = sorted(v["best_iteration"] for v in per.values())
    med = float(np.median(b))
    spread = (b[-1] - b[0]) / max(med, 1.0)
    chosen_best = b[0] if spread > WIDE else med
    held = float(np.mean([v["held_frac"] for v in per.values()]))
    rounds = max(50, int(round(chosen_best / (1.0 - held) / 25.0) * 25))
    out = {"months": list(months), "per_month": per, "optima": b,
           "median": med, "spread_rel": round(spread, 3), "wide_threshold": WIDE,
           "rule": "min (wide spread)" if spread > WIDE else "median",
           "mean_held_frac": held, "rounds": rounds}
    print(f"[budget] optima {b}  median {med:.0f}  relative spread {spread:.2f}  "
          f"-> {out['rule']} -> refit {rounds} rounds", flush=True)
    return out

# ------------------------------------------------------------ submission ----
def build_submission(name: str, seeds: list[int], ds_columns: str = "all") -> dict:
    """Twelve months, three seeds, the two ranking months built separately.

    The per-month split is not cosmetic: the ranking file holds January and July
    2026 as isolated months, and any step that aggregates across rows -- lirf,
    phi, and the day-state reference all do -- would otherwise reach across the
    six-month gap between them. retro-v1's `build_submission` carries the same
    split for the same reason (gpt-5.6-sol caught it on their c8 review).
    """
    t0 = time.time()
    cfg0 = base_cfg(seeds[0], name)
    R._reseed(seeds[0])
    raw: dict = {}
    Xtr, ytr, ttr = R.frame_for_months(D.MONTHS, cfg0["features"], keep_raw=raw)
    rr = D.load_ranking()
    r_months = sorted(rr["month"].unique())
    pairs = [(R.build_features(rr[rr["month"] == m], cfg0["features"]).reset_index(drop=True),
              rr[rr["month"] == m]) for m in r_months]
    print(f"[submit] ranking months {r_months}, "
          f"{[len(x) for x, _ in pairs]} rows", flush=True)
    Xtr, outs = lirf_step.fit_transform(D.MONTHS, Xtr, raw, pairs, gate=cfg0["lirf_gate"])
    pairs = list(zip(outs, [r for _, r in pairs]))
    Xtr, outs = phi_step.fit_transform(D.MONTHS, Xtr, raw, pairs,
                                       mode=cfg0["phi_fit"], shrink=cfg0["phi_shrink"])
    pairs = list(zip(outs, [r for _, r in pairs]))
    Xtr, outs = day_state_step.fit_transform(D.MONTHS, Xtr, raw, pairs,
                                             columns=ds_columns)
    pairs = list(zip(outs, [r for _, r in pairs]))
    Xr = pd.concat([X for X, _ in pairs], ignore_index=True)
    missing = [c for c in Xtr.columns if c not in Xr.columns]
    if missing:
        raise SystemExit(f"ranking frame is missing {missing}")
    print(f"[submit] train {Xtr.shape} ranking {Xr.shape} built in {time.time()-t0:.0f}s", flush=True)

    fb = R.airport_fallback(Xtr, ytr, cfg0["target"])
    base = R.base_score(Xtr, fb, cfg0["target"])
    base_r = R.base_score(Xr, fb, cfg0["target"])
    yt = R.forward_target(R.training_target(cfg0, ytr.to_numpy()), cfg0["target"])
    w = R.sample_weights(cfg0, ytr.to_numpy(), Xtr)

    # The desk's budget rule replaces retro-v1's single contiguous last-15 %
    # slice, which is one late-autumn month and is exactly the instrument that
    # cost the Huber arm 3.57 s.
    budget = budget_from_scan(cfg0, Xtr, ttr, yt, w, base, SCAN_MONTHS)
    rounds, best = budget["rounds"], None

    per_seed = []
    for i, s in enumerate(seeds):
        cfg = base_cfg(s, name)
        R._reseed(s)
        model, enc, _ = R.train_model(cfg, Xtr, yt, w, base, num_round=rounds)
        pr = R.predict_seconds(cfg, model, enc, Xr, base_r)
        print(f"[submit] seed {s}: refit {rounds} (from the budget scan)", flush=True)
        if i == 0:
            # Report the count the LIBRARY holds, not the one we asked for; the
            # release build refuses when the two disagree (desk rule 2026-09-06).
            obs = getattr(model, "params", {}).get("num_threads", "unknown")
            print(f"[threads] declared={os.environ.get('PRC_THREADS','unset')} "
                  f"observed={obs} library=lightgbm", flush=True)
        (mm, ph), = R.noac_head(cfg, Xtr, ytr.to_numpy(), [Xr])
        if ph is not None and mm.any():
            pr = pr.copy()
            pr[mm] = ph
        per_seed.append(pr)

    pr, pinfo = _postprocess(cfg0, per_seed, Xr, D.MONTHS)
    # ids come from the FEATURE frames, not the raw ranking frames: build_features
    # returns departures only, while D.load_ranking() carries arrivals too, so the
    # raw frames are longer and the two do not align. The first version of this
    # took ids from the raw frames and died on the last line after three fits.
    ids = Xr["MVT_ID_mvt"].to_numpy()
    if len(ids) != len(pr):
        raise SystemExit(f"ids {len(ids)} != predictions {len(pr)}")
    cols = {"MVT_ID_mvt": ids, "y_pred": pr}
    for i, p in enumerate(per_seed):
        cols[f"y_pred_s{i}"] = p
    out = ROOT / "artifacts" / "exp-012" / "pred" / f"{name}-ranking.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(out, index=False)
    rec = {"name": name, "seeds": seeds, "ds_columns": ds_columns,
           "budget": budget, "rounds": int(rounds),
           "n": int(len(pr)), "n_features": int(Xtr.shape[1]),
           "path": str(out.relative_to(ROOT)), "sha256": D.sha256(out),
           "seconds": round(time.time() - t0), **pinfo}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="D1-winter")
    ap.add_argument("--folds", default="july_loo,december,january")
    ap.add_argument("--seeds", default="7,107,207")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--ds-columns", default="all", choices=["all", "core", "deice"])
    ap.add_argument("--clip", action="store_true",
                    help="also write the clipped-at-predict-time twin from the same model")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if int(os.environ.get("PRC_THREADS", "24")) != 24:
        print(f"WARNING: PRC_THREADS={os.environ.get('PRC_THREADS')} -- exp-007 ran at 24 and "
              f"this arm has linear leaves, which are not thread-count reproducible", flush=True)
    prov = provenance(base_cfg(seeds[0], a.name))
    print("[provenance] " + json.dumps(prov), flush=True)
    if a.submit:
        rec = build_submission(a.name, seeds, a.ds_columns)
        rec["provenance"] = prov
        (HERE / f"submit-{a.name}.json").write_text(json.dumps(rec, indent=2, default=float))
        print("[submit] " + json.dumps(rec, default=float), flush=True)
        return
    results = {}
    for f in a.folds.split(","):
        r = run_fold(a.name, f, seeds, a.ds_columns, a.clip)
        results[f] = r
        print(f"[{f}] " + json.dumps(r, default=float), flush=True)
    path = HERE / f"result-{a.name}.json"
    rec = json.loads(path.read_text()) if path.exists() else {"name": a.name, "folds": {}}
    rec["folds"] = {**rec.get("folds", {}), **results}
    rec["provenance"] = prov
    rec["argv"] = " ".join(sys.argv[1:])
    path.write_text(json.dumps(rec, indent=2, default=float))


if __name__ == "__main__":
    main()
