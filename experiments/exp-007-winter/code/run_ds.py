"""exp-007: the retro-v1 pipeline with and without prc.day_state_features.

Both arms run through this file, so the control is not a saved number from
another experiment's climb log but a matched run of the identical code path with
the day-state step switched off. That is what makes the paired bootstrap in
`bootstrap_ds.py` a measurement of the columns rather than of a code drift.

The model configuration is retro-v1's `G1cv-s7` line verbatim:
  --features v2 --target residual --linear-tree --clip-floor 60 --noac-head
  --residual-guard --phi --phi-fit cv --lirf

  python experiments/exp-007-winter/code/run_ds.py --name d1 --arms base,ds \
      --folds july,december,january --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
# PINNED pipeline. `code/pipeline/` is a snapshot of retro-v1's run.py and its step
# modules taken from git 19cd199, which is the state on disk when this
# experiment's first arm started (2026-09-05 05:40). retro-v1 committed 3922b2a
# twenty minutes later, adding `cfg["fixed_rounds"]` and `cfg["seed_bag"]` to
# `two_pass`; the running arm did not raise KeyError on those, which proves it
# had loaded the earlier file. Importing the snapshot instead of the live
# directory is what makes seed 7 and seed 17 -- and any later re-run -- the same
# code. It is a snapshot, not a fork: nothing in it is edited.
sys.path.insert(0, str(HERE / "pipeline"))
sys.path.insert(0, str(HERE))

import prc.data as D  # noqa: E402
import run as R  # noqa: E402  (the pinned snapshot of retro-v1's pipeline)
import guard as guard_mod  # noqa: E402
import lirf_step  # noqa: E402
import phi_step  # noqa: E402
import day_state_step  # noqa: E402


def base_cfg(seed: int, name: str) -> dict:
    """retro-v1 G1cv-s7, verbatim."""
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


#: months whose conditions resemble a ranking January (docs/shift-2026.md)
WINTER_MONTHS = ("2025-01", "2025-02", "2025-11", "2025-12")


def winter_mask(months, ttr) -> np.ndarray:
    """True on training rows whose take-off month is one of WINTER_MONTHS."""
    mon = pd.to_datetime(ttr, unit="s", utc=True).strftime("%Y-%m").to_numpy()
    return np.isin(mon, [m for m in WINTER_MONTHS if m in months])


def month_weights(cfg, months, ttr):
    """Sample weights that up-weight the winter training months.

    Question 4's second half asks whether the January half of the submission wants
    its own model. `docs/shift-2026.md` argues the opposite -- no 2025 month
    carries January 2026's conditions, so all twelve months are needed to see
    enough disrupted days -- so both arms are measured rather than assumed.

    `winter_only` is not a weight: it SUBSETS the training rows before anything
    else is computed, so the airport fallback and the early-stopping split are
    the winter months' own too. The shared statistics that are fitted on the
    fold's training months (phi, the lirf signals, the day-state reference) stay
    on all of them: this arm asks which months the TREE learns from, not how much
    data a label-free convention statistic should see.
    """
    mode = cfg.get("month_weight", "none")
    if mode in ("none", "winter_only"):
        return None
    return np.where(winter_mask(months, ttr), float(cfg["month_weight_alpha"]), 1.0)


def fixed_pass(cfg, Xtr, yt, w, base, targets, rounds: int):
    """The pinned `two_pass` with its round SELECTION removed.

    `two_pass` early-stops on the last 15 % of training rows and refits for
    `best / 0.85` rounds, so adding columns changes the round count as well as the
    features. On the july fold that count swings between 50 and 385 across
    near-identical retro-v1 lines, so a single july arm pair cannot separate "the
    columns helped" from "this arm drew a longer fit". This refits both arms for
    exactly the same declared number of rounds, which removes the round draw from
    the comparison entirely. It is a CONTROL, not a candidate: the number is
    declared on the command line and no variant of it is screened.
    """
    model, enc, feats = R.train_model(cfg, Xtr, yt, w, base, num_round=rounds)
    preds = [R.predict_seconds(cfg, model, enc, X, b) for X, b in targets]
    return preds, rounds, rounds, pd.Series(model.feature_importance("gain"), index=feats)


def run_fold(cfg, fold_name: str, use_ds: bool) -> dict:
    """retro-v1's run_fold with the day-state step inserted and outputs in exp-007.

    Copied rather than imported because retro-v1's `run_fold` has no hook. Both
    arms go through this same copy, so a difference between them is the columns.
    """
    fold = D.FOLDS[fold_name]
    R._reseed(cfg["seed"])
    t0 = time.time()
    raw: dict = {}
    Xtr, ytr, ttr = R.frame_for_months(fold.train_months, cfg["features"], keep_raw=raw)
    Xva, yva = R.valid_frame(fold, cfg["features"])
    _, masked, _ = D.fold_frames(fold)
    Xtr, (Xva,) = lirf_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)], gate=cfg["lirf_gate"])
    Xtr, (Xva,) = phi_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                         mode=cfg["phi_fit"], shrink=cfg["phi_shrink"])
    if use_ds:
        Xtr, (Xva,) = day_state_step.fit_transform(fold.train_months, Xtr, raw, [(Xva, masked)],
                                                   columns=cfg.get("ds_columns", "all"))
    if cfg.get("month_weight") == "winter_only":
        keep = winter_mask(fold.train_months, ttr)
        print(f"[{fold_name}] winter_only: {int(keep.sum()):,} of {len(keep):,} training rows kept "
              f"({', '.join(m for m in WINTER_MONTHS if m in fold.train_months)})", flush=True)
        Xtr, ytr, ttr = Xtr[keep].reset_index(drop=True), ytr[keep].reset_index(drop=True), ttr[keep]
    print(f"[{fold_name}/{'ds' if use_ds else 'base'}] train {Xtr.shape} valid {Xva.shape} "
          f"built in {time.time()-t0:.0f}s", flush=True)

    cut = np.quantile(ttr, 1.0 - cfg["es_frac"])
    es = ttr >= cut
    fb = R.airport_fallback(Xtr, ytr, cfg["target"])
    base = R.base_score(Xtr, fb, cfg["target"])
    yt = R.forward_target(R.training_target(cfg, ytr.to_numpy()), cfg["target"])
    w = R.sample_weights(cfg, ytr.to_numpy(), Xtr)
    mw = month_weights(cfg, fold.train_months, ttr)
    if mw is not None:
        w = mw if w is None else np.asarray(w, dtype="float64") * mw
        print(f"[{fold_name}] month weighting {cfg['month_weight']} x{cfg['month_weight_alpha']}: "
              f"{int((mw > 1).sum()):,} of {len(mw):,} training rows up-weighted", flush=True)
    base_va = R.base_score(Xva, fb, cfg["target"])
    if cfg.get("fixed_rounds"):
        (pv,), best, refit, imp = fixed_pass(cfg, Xtr, yt, w, base, [(Xva, base_va)],
                                             int(cfg["fixed_rounds"]))
    else:
        (pv,), best, refit, imp = R.two_pass(cfg, Xtr, yt, w, base, es, [(Xva, base_va)])

    (mm, ph), = R.noac_head(cfg, Xtr, ytr.to_numpy(), [Xva])
    if ph is not None and mm.any():
        pv[mm] = ph
    lo, hi = R._guard_bounds(cfg, fold.train_months)
    av = Xva["d_takeoff_aobt"].to_numpy("float64")
    n_touch = guard_mod.count_touched(pv, av, lo, hi)
    pv = guard_mod.apply(pv, av, lo, hi, floor=cfg["clip_floor"])

    keep = yva.notna().to_numpy()
    no_nm = Xva["nm_missing"].to_numpy().astype(bool)
    noac = Xva["noac"].to_numpy().astype(bool) if "noac" in Xva.columns else None
    res = R.decompose(yva.to_numpy()[keep], pv[keep], no_nm[keep], None if noac is None else noac[keep])
    res.update(fold=fold_name, arm="ds" if use_ds else "base", best_iteration=int(best),
               refit_rounds=refit, guard_touched=int(n_touch), seconds=round(time.time() - t0),
               n_features=int(Xtr.shape[1]))
    out = EXP / "oof" / f"{cfg['name']}-{fold_name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "MVT_ID_mvt": Xva["MVT_ID_mvt"].to_numpy()[keep], "fold": fold_name,
        "y_true": yva.to_numpy()[keep], "y_pred": pv[keep],
        "nm_missing": no_nm[keep].astype("int8"), "airport": Xva["airport"].to_numpy()[keep],
    }).to_parquet(out, index=False)
    res["oof"] = str(out.relative_to(ROOT))
    (EXP / "work" / f"importance-{cfg['name']}-{fold_name}.csv").write_text(
        imp.sort_values(ascending=False).to_csv(header=["gain"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="d1")
    ap.add_argument("--arms", default="base,ds")
    ap.add_argument("--folds", default="july,december,january")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--month-weight", default="none",
                    choices=["none", "winter", "winter_only"],
                    help="question 4: up-weight or restrict to the winter training months")
    ap.add_argument("--month-weight-alpha", type=float, default=3.0)
    ap.add_argument("--ds-columns", default="all", choices=["all", "core", "deice"],
                    help="the whole 25-column block, prc.day_state_features.CORE, or "
                         "DEICE (ds_deice_hr alone)")
    ap.add_argument("--fixed-rounds", type=int, default=0,
                    help="CONTROL: refit both arms for exactly this many rounds, removing the "
                         "early-stopping round draw from the arm comparison")
    a = ap.parse_args()

    import prc.day_state_features as DS
    results = {}
    for arm in a.arms.split(","):
        for f in a.folds.split(","):
            cfg = base_cfg(a.seed, f"{a.name}-{arm}-s{a.seed}")
            cfg["month_weight"] = a.month_weight
            cfg["month_weight_alpha"] = a.month_weight_alpha
            cfg["ds_columns"] = a.ds_columns
            cfg["fixed_rounds"] = a.fixed_rounds
            r = run_fold(cfg, f, use_ds=(arm == "ds"))
            results[f"{arm}/{f}"] = r
            print(f"[{arm}/{f}] " + json.dumps(r), flush=True)
            with open(EXP / "climb.log", "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{a.name}-{arm}-s{a.seed}\t{f}\t"
                         f"rmse={r['rmse']:.2f}\tbulk={r['rmse_bulk']:.2f}\ttail={r['rmse_tail']:.2f}\t"
                         f"no_nm={r['rmse_no_nm']:.2f}\tnoac={r.get('rmse_noac', float('nan')):.2f}\t"
                         f"rounds={r['refit_rounds']}\tnfeat={r['n_features']}\t"
                         f"arm={arm}\tds={a.ds_columns}\tfixed={a.fixed_rounds}\t"
                         f"seed={a.seed}\tmods={R.shared_module_tag()} "
                         f"day_state[VERSION={DS.VERSION}]\n")
    (EXP / "work" / f"result-{a.name}-s{a.seed}.json").write_text(
        json.dumps({"name": a.name, "seed": a.seed, "argv": " ".join(sys.argv[1:]),
                    "folds": results, "shared_modules": R.shared_module_provenance(),
                    "day_state_version": DS.VERSION}, indent=2, default=float))


if __name__ == "__main__":
    main()
