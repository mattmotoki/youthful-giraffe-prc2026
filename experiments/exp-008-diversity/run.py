"""exp-008 diversity: one structurally different learner, four folds, OOF out.

  python run.py --kind cat  --name C1 --rounds-m6 900 --rounds-m11 1200
  python run.py --kind lgb  --name L0 --variant lin --select          # round choice
  python run.py --kind xgb  --name X1 --seed-bag 5

Every arm shares the frozen v2 frame (frames.py), the residual target
(y - (take-off - AOBT), airport-median fallback where there is no NM join), a
per-learner noac head anchored on the schedule, the 60 s floor and the
parameter-free residual guard, so the only thing that changes between arms is
the learner.  Round counts are fixed per fold family and never chosen against a
validation month; `--select` runs the one-time choice and prints it.
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
RETRO = ROOT / "experiments" / "exp-001-retro-v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RETRO))

import prc.data as D  # noqa: E402
import frames  # noqa: E402
import learners as L  # noqa: E402
import guard as guard_mod  # noqa: E402

TAIL_CUT = 3600.0
GUARD_CACHE = HERE / "guard_bounds.json"


# ------------------------------------------------------------- pieces ----
def guard_bounds(fold: str) -> tuple[float, float]:
    """Observed range of r = BLOCK - AOBT over the fold's training months.

    Parameter-free (quantiles 0 and 1) and computed from training months alone,
    so the validation month never enters it.  Cached because it is a pure
    function of the fold.
    """
    try:
        cache = json.loads(GUARD_CACHE.read_text()) if GUARD_CACHE.exists() else {}
    except (json.JSONDecodeError, OSError):
        cache = {}   # two streams share this file; a torn read just recomputes
    if fold not in cache:
        months = D.MONTHS if fold == "_full12" else D.FOLDS[fold].train_months
        lo, hi = guard_mod.bounds_from_months(months, 0.0, 1.0)
        cache[fold] = [lo, hi]
        tmp = GUARD_CACHE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache, indent=2))
        tmp.replace(GUARD_CACHE)   # atomic, so no reader ever sees a partial file
    return tuple(cache[fold])


def anchors(tr: pd.DataFrame, va: pd.DataFrame):
    """Residual anchor: take-off minus NM off-block, airport-median taxi where
    there is no NM join.  The median comes from training rows only."""
    fb = tr["y"].groupby(tr["airport"].astype("object").to_numpy()).median()
    med = float(fb.median())

    def base(X):
        b = X["d_takeoff_aobt"].to_numpy(dtype="float64").copy()
        m = ~np.isfinite(b)
        if m.any():
            b[m] = X.loc[m, "airport"].astype("object").map(fb).fillna(med).to_numpy("float64")
        return b

    return base(tr), base(va)


def decompose(y, p, no_nm, noac) -> dict:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    se = (y - p) ** 2
    bulk = y <= TAIL_CUT
    return {
        "rmse": float(np.sqrt(se.mean())),
        "rmse_bulk": float(np.sqrt(se[bulk].mean())),
        "rmse_tail": float(np.sqrt(se[~bulk].mean())) if (~bulk).any() else float("nan"),
        "rmse_no_nm": float(np.sqrt(se[no_nm].mean())) if no_nm.any() else float("nan"),
        "rmse_noac": float(np.sqrt(se[noac].mean())) if noac.any() else float("nan"),
        "n": int(len(y)), "n_tail": int((~bulk).sum()), "n_no_nm": int(no_nm.sum()),
        "n_noac": int(noac.sum()),
        "sse_share_tail": float(se[~bulk].sum() / se.sum()),
        "sse_share_no_nm": float(se[no_nm].sum() / se.sum()),
        "sse_share_noac": float(se[noac].sum() / se.sum()) if noac.any() else float("nan"),
        "bias": float((p - y).mean()),
    }


#: Rounds for the noac head, per learner kind. The head is the same object for
#: every arm -- a small model on 63 to 538 rows anchored on take-off minus
#: schedule -- but a round count is not portable between learners: CatBoost's
#: oblivious trees take much smaller steps than a 63-leaf LightGBM tree at the
#: same learning rate, and the main model's own frozen counts show the size of
#: that difference (3,625 against 175 on the six-month family). Giving every
#: head LightGBM's 60 rounds would have underfitted the CatBoost head on the
#: rows that carry a fifth of the squared error, and the arm comparison would
#: then have read as a learner difference. `--noac-rounds` overrides, for the
#: sensitivity run recorded in RESULTS.md.
NOAC_ROUNDS = {"lgb": 60, "xgb": 60, "cat": 600, "mlp": 60}


def report_threads(learner, kind: str) -> None:
    """Print the thread count the LIBRARY holds, not the one we asked for.

    The release manifest records this value and `scripts/build_candidate.py`
    refuses to proceed when it disagrees with the declared one (desk rule,
    2026-09-06). A declared count that is not the count that ran makes the
    manifest lie in the one field the reproducibility contract rests on --
    which is exactly what happened to `prc.nmtail`, whose hardcoded 8 sat
    behind a config declaring 6.
    """
    obs, lib = "unknown", kind
    try:
        m = getattr(learner, "model_", None) or getattr(learner, "model", None)
        if kind == "lgb":
            obs, lib = m.params.get("num_threads", "unknown"), "lightgbm"
        elif kind == "cat":
            obs, lib = m.get_all_params().get("thread_count", "unknown"), "catboost"
        elif kind == "xgb":
            obs, lib = m.attributes().get("nthread") or m.save_config(), "xgboost"
            if not isinstance(obs, (int, str)) or (isinstance(obs, str) and len(obs) > 12):
                import json as _j
                obs = _j.loads(m.save_config())["learner"]["generic_param"]["nthread"]
        elif kind == "mlp":
            import torch
            obs, lib = torch.get_num_threads(), "torch"
    except Exception as e:  # never let reporting break a fit
        obs, lib = f"unreadable ({type(e).__name__})", kind
    print(f"[threads] declared={os.environ.get('PRC_THREADS','unset')} "
          f"observed={obs} library={lib}", flush=True)


def make_learner(cfg, seed, head=False):
    kind = cfg["kind"]
    if kind == "lgb":
        if head:
            return L.LgbLearner("head", variant="plain", params=dict(
                num_leaves=cfg["noac_leaves"], min_data_in_leaf=cfg["noac_min_data"],
                learning_rate=cfg["noac_lr"], bagging_fraction=1.0, bagging_freq=0,
                feature_fraction=0.8, max_cat_threshold=16, cat_smooth=5.0,
                min_data_per_group=5))
        return L.LgbLearner(cfg["name"], variant=cfg["variant"], monotone=cfg["monotone"],
                            huber_delta=cfg["huber_delta"],
                            params=dict(learning_rate=cfg["lr"], num_leaves=cfg["leaves"]))
    if kind == "cat":
        if head:
            return L.CatLearner("head", depth=6, gpu=False,
                                params=dict(learning_rate=cfg["noac_lr"], verbose=0))
        return L.CatLearner(cfg["name"], depth=cfg["depth"], gpu=not cfg["cpu"],
                            params=dict(learning_rate=cfg["lr"]))
    if kind == "xgb":
        if head:
            return L.XgbLearner("head", max_depth=6, gpu=False,
                                params=dict(learning_rate=cfg["noac_lr"], min_child_weight=1.0,
                                            subsample=1.0))
        return L.XgbLearner(cfg["name"], max_depth=cfg["depth"], gpu=not cfg["cpu"],
                            params=dict(learning_rate=cfg["lr"]))
    if kind == "mlp":
        if head:  # 500-odd rows is no place for a network; the head stays a small tree
            return L.LgbLearner("head", variant="plain", params=dict(
                num_leaves=cfg["noac_leaves"], min_data_in_leaf=cfg["noac_min_data"],
                learning_rate=cfg["noac_lr"], bagging_fraction=1.0, bagging_freq=0,
                feature_fraction=0.8, max_cat_threshold=16, cat_smooth=5.0, min_data_per_group=5))
        return L.MlpLearner(cfg["name"], epochs=cfg["epochs"], lr=cfg["lr"], gpu=not cfg["cpu"])
    raise ValueError(kind)


def noac_predictions(cfg, Xtr, ytr, Xva, cats, seed):
    """A model for the placeholder-block-time rows alone, anchored on the schedule.

    Same architecture as retro-v1's head (exp-001 SPEC, "The noac rows get their
    own model"), but built from *this arm's* learner, so the blend has diversity
    on the rows that carry the most squared error rather than one shared head.
    """
    m = Xtr["noac"].to_numpy().astype(bool)
    mv = Xva["noac"].to_numpy().astype(bool)
    if m.sum() < cfg["noac_min_rows"] or not mv.any():
        return mv, None
    b_tr = np.nan_to_num(Xtr.loc[m, "d_takeoff_sched"].to_numpy("float64"))
    b_va = np.nan_to_num(Xva.loc[mv, "d_takeoff_sched"].to_numpy("float64"))
    head = make_learner(cfg, seed, head=True)
    rounds = cfg["noac_rounds"] or NOAC_ROUNDS[cfg["kind"]]
    head.fit(Xtr[m], ytr[m] - b_tr, cats, rounds, seed)
    p = head.predict(Xva[mv]) + b_va
    print(f"[noac] head on {int(m.sum())} rows -> {int(mv.sum())} valid rows, "
          f"{rounds} rounds", flush=True)
    return mv, p


# ---------------------------------------------------------------- fold ----
def run_fold(cfg, fold: str) -> dict:
    t0 = time.time()
    tr, va = frames.fold_frames(fold)
    feats = frames.feature_columns(tr)
    cats = frames.cat_columns(tr)
    Xtr, Xva = tr[feats], va[feats]
    ytr, yva = tr["y"].to_numpy("float64"), va["y"].to_numpy("float64")
    keep = np.isfinite(yva)
    base_tr, base_va = anchors(tr, va)
    if cfg["target"] == "residual":
        rtr = ytr - base_tr
        off_va = base_va
    else:
        rtr, off_va = ytr, np.zeros(len(va))
    print(f"[{fold}] train {Xtr.shape} valid {Xva.shape} loaded in {time.time()-t0:.0f}s", flush=True)

    rounds = cfg["rounds"][frames.FAMILY[fold]]
    if cfg["select"]:
        # One-time round choice.  The early-stopping slice is the LAST TRAINING
        # MONTH -- never the validation month -- so the count is chosen on the
        # same kind of generalisation the fold asks for without touching it.
        mm = tr["month_str"].to_numpy()
        # The early-stop slice is one whole TRAINING month, the last by default.
        # `--es-month` names a different one: CatBoost's count differs eightfold
        # between the fold families, and the two candidate causes -- training
        # size and the season of the held-out month -- are separable only by
        # holding out a summer month from an eleven-month training set.
        es_month = cfg.get("es_month") or frames.ES_MONTH.get(fold) or mm.max()
        if es_month not in set(mm):
            raise SystemExit(f"--es-month {es_month} is not a training month of {fold}")
        es = mm == es_month
        lr = make_learner(cfg, cfg["seed"])
        lr.fit(Xtr[~es], rtr[~es], cats, cfg["max_rounds"], cfg["seed"],
               Xes=Xtr[es], yes=rtr[es], es_rounds=cfg["es_rounds"])
        best = lr.best_iteration or cfg["max_rounds"]
        # An MLP's unit is an epoch, not a tree: it already sees every row each
        # pass, so refitting on the held-out month's rows too does not call for
        # more epochs, and rounding to a multiple of 25 would be nonsense.
        refit = (int(best) if cfg["kind"] == "mlp"
                 else int(round(best / (1.0 - es.mean()) / 25.0) * 25))
        out = {"fold": fold, "family": frames.FAMILY[fold], "es_month": str(es_month),
               "held_frac": float(es.mean()), "best_iteration": int(best), "refit_rounds": refit,
               "seconds": round(time.time() - t0)}
        print(f"[select] {json.dumps(out)}", flush=True)
        return out

    seeds = [cfg["seed"] + 100 * i for i in range(max(1, cfg["seed_bag"]))]
    per_seed = []
    imp = None
    for s in seeds:
        lr = make_learner(cfg, s)
        lr.fit(Xtr, rtr, cats, rounds, s)
        p = lr.predict(Xva) + off_va
        mv, ph = noac_predictions(cfg, Xtr, ytr, Xva, cats, s)
        if ph is not None:
            p = p.copy()
            p[mv] = ph
        per_seed.append(p)
        try:
            g = lr.importance()
            imp = g if imp is None else imp + g
        except Exception:
            pass
        print(f"[{fold}] seed {s} done at {time.time()-t0:.0f}s", flush=True)

    pv = np.mean(per_seed, axis=0)
    pv = np.maximum(pv, cfg["clip_floor"])
    if cfg["residual_guard"]:
        lo, hi = guard_bounds(fold)
        av = Xva["d_takeoff_aobt"].to_numpy("float64")
        n = guard_mod.count_touched(pv, av, lo, hi)
        pv = guard_mod.apply(pv, av, lo, hi, floor=cfg["clip_floor"])
        print(f"[{fold}] residual guard [{lo:.0f}, {hi:.0f}] touched {n} rows", flush=True)

    no_nm = Xva["nm_missing"].to_numpy().astype(bool)
    noac = Xva["noac"].to_numpy().astype(bool)
    res = decompose(yva[keep], pv[keep], no_nm[keep], noac[keep])
    res.update(fold=fold, rounds=rounds, seeds=seeds, seconds=round(time.time() - t0))

    cols = {
        "MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy()[keep], "fold": fold,
        "y_true": yva[keep], "y_pred": pv[keep],
        "nm_missing": no_nm[keep].astype("int8"), "noac": noac[keep].astype("int8"),
        "airport": Xva["airport"].to_numpy()[keep],
        "d_takeoff_aobt": Xva["d_takeoff_aobt"].to_numpy("float64")[keep],
        "d_aobt_lobt": Xva["d_aobt_lobt"].to_numpy("float64")[keep],
    }
    # Per-seed predictions are stored RAW -- before the floor and the guard --
    # so that the seed-bag curve can average first and post-process once, which
    # is what `y_pred` is. Guarding each seed and then averaging is a different
    # estimator and its k = n point would not equal the arm's reported number.
    for i, p in enumerate(per_seed):
        cols[f"y_pred_s{i}"] = p[keep]
    # --outdir lets another experiment reuse this harness without writing into
    # exp-008's directory. exp-008 lost a finding to two runs sharing an output
    # path, so a run that belongs to another experiment writes there instead.
    # resolve(): a relative --outdir like "../exp-012-blend2/oof" writes the
    # parquet fine and then throws in relative_to(ROOT) below, AFTER the file is
    # on disk -- so the run looks like it wrote its OOF and then died. Cost one
    # january fold of the MLP before it was spotted.
    outdir = (Path(cfg["outdir"]).resolve() if cfg.get("outdir") else HERE / "oof")
    out = outdir / f"{cfg['name']}-{fold}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(out, index=False)
    res["oof"] = str(out.relative_to(ROOT))
    if imp is not None and len(imp):
        (HERE / f"importance-{cfg['name']}-{fold}.csv").write_text(
            (imp / len(seeds)).sort_values(ascending=False).to_csv(header=["gain"]))
    return res


# ---------------------------------------------------------- submission ----
def run_submission(cfg) -> dict:
    """Fit the arm on all twelve 2025 months and predict the ranking file.

    The round count is the eleven-month family's frozen count scaled by 12/11,
    because the only thing that changes between an eleven-month fold and the
    submission fit is one more month of rows; no early stop runs here, so no
    ranking row and no held-out month can influence the count.
    """
    t0 = time.time()
    tr = frames.all_training()
    va = frames.ranking_frame()
    feats = frames.feature_columns(tr)
    cats = frames.cat_columns(tr)
    missing = [c for c in feats if c not in va.columns]
    if missing:
        raise ValueError(f"ranking frame is missing {missing}")
    Xtr, Xva = tr[feats], va[feats]
    ytr = tr["y"].to_numpy("float64")
    base_tr, base_va = anchors(tr, va)
    rtr = ytr - base_tr if cfg["target"] == "residual" else ytr
    off_va = base_va if cfg["target"] == "residual" else np.zeros(len(va))
    rounds = int(round(cfg["rounds"]["m11"] * 12 / 11))
    print(f"[submit] train {Xtr.shape} ranking {Xva.shape} rounds {rounds} "
          f"loaded in {time.time()-t0:.0f}s", flush=True)

    seeds = [cfg["seed"] + 100 * i for i in range(max(1, cfg["seed_bag"]))]
    per_seed = []
    for s_ in seeds:
        lr = make_learner(cfg, s_)
        lr.fit(Xtr, rtr, cats, rounds, s_)
        if s_ == seeds[0]:
            report_threads(lr, cfg["kind"])
        p = lr.predict(Xva) + off_va
        mv, ph = noac_predictions(cfg, Xtr, ytr, Xva, cats, s_)
        if ph is not None:
            p = p.copy()
            p[mv] = ph
        per_seed.append(p)
        print(f"[submit] seed {s_} done at {time.time()-t0:.0f}s", flush=True)

    # The served member must be the SAME ESTIMATOR the blend weights were fitted
    # on. `run_fold` floors at 60 s and applies the residual guard before writing
    # OOF, so a ranking prediction written raw would be a different object: the
    # weights would have been fitted on floored, guarded members and applied to
    # unfloored ones. Measured on the first version of these files, 78 of
    # L0-lin's and 77 of L2-hub's ranking rows sat below the floor (minimum
    # -4,962 s) while T-tail, which floors inside prc.tail_model, had none — an
    # asymmetry across members, concentrated in the no-NM regime.
    # The guard bounds come from all twelve 2025 months, which is what the
    # submission fits; no ranking row enters them.
    lo, hi = guard_bounds("_full12")
    pv = np.mean(per_seed, axis=0)
    pv = np.maximum(pv, cfg["clip_floor"])
    av = Xva["d_takeoff_aobt"].to_numpy("float64")
    if cfg["residual_guard"]:
        n = guard_mod.count_touched(pv, av, lo, hi)
        pv = guard_mod.apply(pv, av, lo, hi, floor=cfg["clip_floor"])
        print(f"[submit] residual guard [{lo:.0f}, {hi:.0f}] touched {n} rows", flush=True)
    print(f"[submit] floor at {cfg['clip_floor']:.0f} s; "
          f"{int((np.mean(per_seed, axis=0) < cfg['clip_floor']).sum())} raw rows were below it", flush=True)
    cols = {"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "y_pred": pv}
    for i, p in enumerate(per_seed):
        cols[f"y_pred_s{i}"] = p
    out = ROOT / "artifacts" / "exp-008" / "pred" / f"{cfg['name']}-ranking.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_parquet(out, index=False)
    rec = {"name": cfg["name"], "rounds": rounds, "seeds": seeds, "n": int(len(va)),
           "path": str(out.relative_to(ROOT)), "sha256": D.sha256(out),
           "seconds": round(time.time() - t0)}
    print("[submit] " + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["lgb", "cat", "xgb", "mlp"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default=",".join(frames.FOLDS))
    ap.add_argument("--target", default="residual", choices=["residual", "raw"])
    ap.add_argument("--variant", default="lin", choices=["lin", "plain"])
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--huber-delta", type=float, default=None)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seed-bag", type=int, default=1)
    ap.add_argument("--rounds-m6", type=int, default=300)
    ap.add_argument("--rounds-m11", type=int, default=400)
    ap.add_argument("--rounds-m11s", type=int, default=0,
                    help="eleven months, summer validation (july_loo); 0 falls back to m11")
    ap.add_argument("--select", action="store_true", help="one-time round choice, no OOF written")
    ap.add_argument("--es-month", default=None,
                    help="training month to early-stop on (default: the last one)")
    ap.add_argument("--max-rounds", type=int, default=4000)
    ap.add_argument("--es-rounds", type=int, default=100)
    ap.add_argument("--clip-floor", type=float, default=60.0)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--noac-rounds", type=int, default=0,
                    help="0 uses NOAC_ROUNDS for the learner kind")
    ap.add_argument("--noac-leaves", type=int, default=63)
    ap.add_argument("--noac-min-data", type=int, default=5)
    ap.add_argument("--noac-lr", type=float, default=0.05)
    ap.add_argument("--noac-min-rows", type=int, default=50)
    ap.add_argument("--outdir", default=None,
                    help="directory for OOF output (default: this experiment's oof/)")
    ap.add_argument("--submit", action="store_true",
                    help="fit all twelve 2025 months and write ranking predictions")
    a = ap.parse_args()

    cfg = dict(vars(a), rounds={"m6": a.rounds_m6, "m11": a.rounds_m11,
                                "m11s": a.rounds_m11s or a.rounds_m11},
               residual_guard=not a.no_guard, huber_delta=a.huber_delta)
    if a.submit:
        rec = run_submission(cfg)
        (HERE / f"submit-{a.name}.json").write_text(json.dumps(rec, indent=2, default=float))
        return
    results = {f: run_fold(cfg, f) for f in a.folds.split(",")}
    for f, r in results.items():
        print(f"[{f}] " + json.dumps(r), flush=True)
    if a.select:
        (HERE / f"select-{a.name}.json").write_text(json.dumps(results, indent=2, default=float))
        return

    # Merge into any existing record for this arm: a run with --folds july_loo
    # is a repair of one fold, and it must not delete the other three. Each fold
    # entry carries the argv that produced it, so a merged record still says
    # exactly how every number in it was made.
    path = HERE / f"result-{a.name}.json"
    argv = " ".join(sys.argv[1:])
    for r in results.values():
        r["argv"] = argv
    record = json.loads(path.read_text()) if path.exists() else {"name": a.name, "folds": {}}
    record["folds"] = {**record.get("folds", {}), **results}
    record["argv"] = argv
    record.setdefault("argv_history", []).append(argv)
    record["frame"] = {"version": frames.VERSION, "schema": frames.SCHEMA_FINGERPRINT,
                       "numeric_2025_07": frames.NUMERIC_FINGERPRINT}
    record["threads"], record["gpu"] = L.THREADS, L.GPU
    path.write_text(json.dumps(record, indent=2, default=float))
    with open(HERE / "climb.log", "a") as fh:
        for f, r in results.items():
            fh.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{a.name}\t{f}\trmse={r['rmse']:.1f}\t"
                f"bulk={r['rmse_bulk']:.1f}\ttail={r['rmse_tail']:.1f}\tno_nm={r['rmse_no_nm']:.1f}\t"
                f"noac={r['rmse_noac']:.1f}\trounds={r['rounds']}\t{' '.join(sys.argv[1:])}\t"
                f"frame={frames.VERSION}/{frames.SCHEMA_FINGERPRINT}\n")
    print(json.dumps(record, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
