"""exp-147/148: the two exp-008 members of base6 (C2-cat, L2-hub8725) under one week-pair fit.

    PRC_SOURCE_ROOT=<tree with the exp-008 code and artifacts/exp-008/frames> \
    python -u experiments/exp-147-foldserve/code/members/exp008_wp.py --member C2-cat \
        --fit outer-3 --out DIR --threads N [--device cpu|gpu] [--smoke]

`--fit` is a name in `pilot.FITS` (`outer-k`, `inner-a-b` with a < b, `final`); `--split` is an
alias. A fit trains on the label rows of its first split (`Split.label_mask`, checked with
`assert_training_frame`, target `y`) and writes one artifact per `pilot.artifacts_of(fit)`:
an inner fit writes `inner-a-b` (owner a) and `inner-b-a` (owner b) from the same model.
`final` trains on every 2025 departure and scores the 344,841 ranking departures.

The member code is exp-008's own, imported from `$PRC_SOURCE_ROOT` (run.py, frames.py,
learners.py, exp-001 guard.py and features.py, prc/data.py): `make_learner`, `anchors`
(residual target and airport-median fallback from training rows), `noac_predictions` (the
noac head from training rows), `frames.all_training`, `frames.ranking_frame`, the guard's
`apply` and `count_touched`. The member configuration is what run.py's own argument parser
builds from the member's command (`member_cfg`). `run_fold` and `run_submission` are not called:
they hard-code the fold in three places, so `fit_predict` is their body (run.py:205-275 and
324-372) with these changes only:

* rows: `all_training()` for every fit; training rows are the split's label rows, scored rows
  are its held rows with `y` dropped (run.py:205 reads a month fold);
* rounds: the member's eleven-month count for every fit, final included (see ROUNDS_RULE);
* guard bounds: the observed range (quantiles 0 and 1) of r = BLOCK - AOBT over the training
  rows, computed as `d_takeoff_aobt - y` of the cache (equal to the raw difference on every
  row of the 12 months); run.py:44-62 reads whole training months from disk and caches them in
  the exp-008 lane, which this driver never writes;
* outputs: `pilot.write` under `--out`; no oof file, result JSON, importance CSV, climb.log
  or guard_bounds.json in exp-008; no `y_true` in the artifact.

The artifact holds, per scored departure, what the fold OOF holds (exp-012 pinned/published):
`y_pred` (seed mean after the noac head, the 60 s floor and the residual guard), `y_pred_s0`
(raw, before floor and guard), and the label-free gate columns `nm_missing`, `noac`, `airport`,
`d_takeoff_aobt`, `d_aobt_lobt`. `MVT_ID_mvt` is float64, as in the fold OOF.

`--device gpu` (C2-cat only) is the member's GPU path (learners.py:206-207): task_type GPU,
devices $PRC_GPU, which defaults here to "0" (the one device under CUDA_VISIBLE_DEVICES=1).
`--device cpu` passes the member's `--cpu`, as the month lane did (exp-047 member_folds.py:106).
`--smoke` runs the same path on a 2 % random sample of the training rows plus every training
noac row (so the head's 50-row minimum is met), 20 main and 20 head rounds, under `<out>/smoke/`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP147_ROOT = HERE.parents[3]

#: The member commands (run.py argv without --folds/--outdir/--submit).
MEMBERS = {
    "C2-cat": {
        "argv": ["--kind", "cat", "--name", "C2-d8p", "--depth", "8",
                 "--rounds-m6", "3625", "--rounds-m11", "5325"],
        "devices": ("cpu", "gpu"),
        "source": "exp-008-diversity/run_gpu5.sh:12-13 (folds, GPU); run_submit_gpu.sh:8 (ranking, "
                  "GPU); exp-047-queue/work/member_folds.py:104-106 (month lane, --cpu)",
        "served_rounds": 5809,
    },
    "L2-hub8725": {
        "argv": ["--kind", "lgb", "--name", "L2-hub8725", "--variant", "plain",
                 "--huber-delta", "1200", "--rounds-m6", "3225", "--rounds-m11", "8725"],
        "devices": ("cpu",),
        "source": "exp-012-blend2/run_hub8725_folds.sh:14-16 (folds); run_hub8725_ranking.sh:18-19 "
                  "(ranking); exp-047-queue/work/member_folds.py:95-97 (month lane)",
        "served_rounds": 9518,
    },
}
ROUNDS_RULE = ("the member's eleven-month count (cfg rounds m11) for every fit, final included. run.py "
               "gives a fold the fixed count of its family (run.py:219; frames.py:56-66, where ten-month "
               "folds keep m11 on purpose because the count is part of the member's identity) and "
               "scales it by 12/11 only in run_submission (run.py:336). Every fold file of this member "
               "(exp-012 july_loo/december/january, exp-047 month lane) used m11; the stacker is fitted "
               "on fold predictions, so final uses the same count, not the served 12/11 count.")
GUARD_RULE = ("observed range (np.quantile 0 and 1) of r = d_takeoff_aobt - y over this fit's training "
              "rows; guard.bounds_from_months (exp-001 guard.py:41-45) computes the same range from the "
              "raw months")
SMOKE = {"fraction": 0.02, "seed": 0, "rounds": 20, "noac_rounds": 20, "keep_all_noac": True}
#: sha256 of artifacts/exp-008/frames/v2-2025-MM.parquet on the Mac (built 2026-09-08).
CACHE_SHA = {
    "2025-01": "eec3b363bb3e21150efe6d44bdec258801c7f8b41876351ae53f5fe1f1e15ea9",
    "2025-02": "2e1a45372c7c042c443b0a5dd04684b8d932d08986ad5cfefa5d8e91749ca50f",
    "2025-03": "291822fd0a74aab1f2e38a84d4b65c7374c62b6b3df71c4c720349bc39910c46",
    "2025-04": "72c3938d4e5b058ee1e479cec62bc8b7ad1348b1ebe5469e9655f1803871d1da",
    "2025-05": "24528331a69ee03b1061fd8148aed1bdc26120afc11a97a8c1d9c7eea4c2602f",
    "2025-06": "01330a0bbb2e4399bc2604150ce60bbf70acf14564f861c1404a57fd31c5f138",
    "2025-07": "46add2ccf1e0c537f7d9f1eb1a0f14befd0a2654566e554a44d51abe45e48117",
    "2025-08": "20c107bd31d13dd8891c9d1f103dc66f7c7c604008fcf0c0dac0d52a790addd8",
    "2025-09": "c9aaf19cadbe460dc11d708b8e867788af18987a789a6707b9575944b08b4176",
    "2025-10": "de2704f91ff1484ace81f5831e5de4a1e8f433176b7f13d992d4c3704e305ed7",
    "2025-11": "5380f1ddb8facd277ae37bfade8dd9b8798ce159351f362e7ec026d3c80b09ef",
    "2025-12": "8c6ff33b80385dfbe2b733fe98965d2eac35e262987b1e9bb79977b62f8d321c",
}
#: sha256 of v2-ranking.parquet as frames.ranking_frame() built it on the Mac on 2026-09-30 from
#: the live ranking file (15,082,958 bytes; ids equal the published ranking files, in order).
RANKING_SHA = "3910a8544f71f0c8c1204b1b0b09188e12c9ab4809f298d7b7bda8ad56aec23a"
GATES = ["nm_missing", "noac", "airport", "d_takeoff_aobt", "d_aobt_lobt"]
ID = "MVT_ID_mvt"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--member", required=True, choices=sorted(MEMBERS))
    ap.add_argument("--fit", "--split", dest="fit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.device not in MEMBERS[a.member]["devices"]:
        ap.error(f"{a.member} runs on {'/'.join(MEMBERS[a.member]['devices'])} only")
    return a


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_member_code(src: Path):
    """exp-008 run.py from SRC, with prc.data bound to SRC's file (pilot binds `prc` to the
    exp-147 tree first, and `import prc.data` would otherwise load that tree's copy)."""
    if "prc.data" not in sys.modules:
        spec = importlib.util.spec_from_file_location("prc.data", src / "prc" / "data.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["prc.data"] = mod
        spec.loader.exec_module(mod)
        import prc
        prc.data = mod
    spec = importlib.util.spec_from_file_location(
        "exp008_run", src / "experiments" / "exp-008-diversity" / "run.py")
    R = importlib.util.module_from_spec(spec)
    sys.modules["exp008_run"] = R          # so the code record below finds run.py
    spec.loader.exec_module(R)
    for m in (R, R.frames, R.L, R.guard_mod, R.D, sys.modules["features"]):
        if not Path(m.__file__).resolve().is_relative_to(src.resolve()):
            raise SystemExit(f"member module {m.__name__} loaded from {m.__file__}, not {src}")
    return R


def member_cfg(R, argv: list[str]) -> dict:
    """The cfg run.py's own `main` builds from `argv`, taken before it runs a fold."""
    class _Cfg(Exception):
        pass

    def grab(cfg, fold):
        raise _Cfg(cfg)

    keep_fold, keep_argv = R.run_fold, sys.argv
    R.run_fold, sys.argv = grab, ["run.py", *argv]
    try:
        R.main()
    except _Cfg as e:
        return e.args[0]
    finally:
        R.run_fold, sys.argv = keep_fold, keep_argv
    raise RuntimeError("run.py main returned without building a cfg")


def learner_settings(lr, kind: str) -> dict:
    if kind == "lgb":
        return {"library": "lightgbm", "params": dict(lr.model.params),
                "threads_observed": lr.model.params.get("num_threads")}
    p = lr.model.get_all_params()
    return {"library": "catboost", "params": p, "threads_observed": p.get("thread_count")}


def fit_predict(R, cfg, tr, va, rounds, log):
    """run.py run_fold/run_submission body (run.py:206-275, 326-372) on given rows."""
    import numpy as np

    F = R.frames
    feats, cats = F.feature_columns(tr), F.cat_columns(tr)
    missing = [c for c in feats if c not in va.columns]
    if missing:
        raise ValueError(f"scored frame is missing {missing}")
    Xtr, Xva = tr[feats], va[feats]
    ytr = tr["y"].to_numpy("float64")
    base_tr, base_va = R.anchors(tr, va)
    if cfg["target"] == "residual":
        rtr, off_va = ytr - base_tr, base_va
    else:
        rtr, off_va = ytr, np.zeros(len(va))
    seeds = [cfg["seed"] + 100 * i for i in range(max(1, cfg["seed_bag"]))]
    per_seed, info = [], {"seeds": seeds, "features": len(feats), "categoricals": cats,
                          "noac_train_rows": int(Xtr["noac"].to_numpy().astype(bool).sum()),
                          "noac_scored_rows": int(Xva["noac"].to_numpy().astype(bool).sum()),
                          "noac_head_used": []}
    t0 = time.monotonic()
    for s_ in seeds:
        lr = R.make_learner(cfg, s_)
        lr.fit(Xtr, rtr, cats, rounds, s_)
        if s_ == seeds[0]:
            R.report_threads(lr, cfg["kind"])
            info["learner"] = learner_settings(lr, cfg["kind"])
        p = lr.predict(Xva) + off_va
        mv, ph = R.noac_predictions(cfg, Xtr, ytr, Xva, cats, s_)
        info["noac_head_used"].append(ph is not None)
        if ph is not None:
            p = p.copy()
            p[mv] = ph
        per_seed.append(p)
        log(f"seed {s_} done at {time.monotonic() - t0:.0f}s")
    info["fit_seconds"] = round(time.monotonic() - t0, 1)
    raw = np.mean(per_seed, axis=0)
    pv = np.maximum(raw, cfg["clip_floor"])
    r = tr["d_takeoff_aobt"].to_numpy("float64") - ytr
    r = r[np.isfinite(r)]
    lo, hi = float(np.quantile(r, 0.0)), float(np.quantile(r, 1.0))
    av = Xva["d_takeoff_aobt"].to_numpy("float64")
    info.update(guard_bounds=[lo, hi], raw_below_floor=int((raw < cfg["clip_floor"]).sum()))
    if cfg["residual_guard"]:
        info["guard_touched"] = R.guard_mod.count_touched(pv, av, lo, hi)
        pv = R.guard_mod.apply(pv, av, lo, hi, floor=cfg["clip_floor"])
        log(f"residual guard [{lo:.0f}, {hi:.0f}] touched {info['guard_touched']} rows")
    return pv, per_seed, info


def main() -> None:
    a = parse_args()
    src = Path(os.environ.get("PRC_SOURCE_ROOT") or sys.exit("set PRC_SOURCE_ROOT")).resolve()
    # learners.py reads PRC_THREADS and PRC_GPU at import; the flag wins over the environment.
    for v in ("PRC_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    if a.device == "gpu":
        os.environ.setdefault("PRC_GPU", "0")
    t_start = time.monotonic()

    def log(msg):
        print(f"[{a.member} {a.fit}{' smoke' if a.smoke else ''}] {msg} "
              f"({time.monotonic() - t_start:.0f}s)", flush=True)

    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(HERE.parent))
    import pilot as P
    WP = P.WP

    if a.fit not in P.FITS:
        raise SystemExit(f"--fit {a.fit} is not one of {', '.join(P.FITS)}")
    arts = P.artifacts_of(a.fit)
    out = a.out / "smoke" if a.smoke else a.out
    stage = a.member
    for s, _ in arts:
        for ext in (".parquet", ".json"):
            if (out / f"{stage}-{s.name}{ext}").exists():
                raise SystemExit(f"{out / f'{stage}-{s.name}{ext}'} exists; write-once")

    R = load_member_code(src)
    F, D = R.frames, R.D
    spec = MEMBERS[a.member]
    # run.py's --cpu applies only to learners with a GPU path (cat here); lgb is always CPU.
    argv = list(spec["argv"]) + (["--cpu"] if a.device == "cpu" and "gpu" in spec["devices"] else [])
    if a.smoke:
        argv += ["--rounds-m11", str(SMOKE["rounds"]), "--noac-rounds", str(SMOKE["noac_rounds"])]
    cfg = member_cfg(R, argv)
    rounds = int(cfg["rounds"]["m11"])
    log(f"member cfg from run.py argv {' '.join(argv)}; rounds {rounds}; device {a.device}; "
        f"threads {R.L.THREADS}; PRC_GPU {os.environ.get('PRC_GPU')}")

    # ---- inputs: the twelve month caches (read-only; frames.load_cached builds a missing one,
    # which this driver refuses to trigger)
    inputs = {}
    for m in D.MONTHS:
        p = F.cache_path(m)
        if not p.exists():
            raise SystemExit(f"missing month cache {p}")
        h = sha(p)
        inputs[str(p.relative_to(src))] = {"sha256": h, "bytes": p.stat().st_size,
                                           "matches_mac_2026_09_08": h == CACHE_SHA[m]}
    A = F.all_training()
    side = pd.DataFrame({WP.TIME_COLUMN: pd.to_datetime(A["t"].to_numpy(), unit="s", utc=True)})
    first = arts[0][0]
    lab = first.label_mask(side)
    log(f"{len(A):,} cached 2025 departures; {int(lab.sum()):,} label rows under {first.name}")

    # ---- scored rows
    ranking_rec = None
    if a.fit == "final":
        # frames.ranking_frame() unchanged; its cache moves out of the exp-008 lane unless the
        # lane already holds one (the served fit read artifacts/exp-008/frames/v2-ranking.parquet).
        lane = F.CACHE / f"{F.VERSION}-ranking.parquet"
        cache_dir = F.CACHE if lane.exists() else a.out / "cache" / "exp-008-frames"
        built = not (cache_dir / f"{F.VERSION}-ranking.parquet").exists()
        keep, F.CACHE = F.CACHE, cache_dir
        try:
            va = F.ranking_frame()
        finally:
            F.CACHE = keep
        rp = cache_dir / f"{F.VERSION}-ranking.parquet"
        ranking_rec = {"path": str(rp), "sha256": sha(rp), "bytes": rp.stat().st_size,
                       "built_this_run": built}
        ranking_rec["matches_mac_2026_09_30"] = ranking_rec["sha256"] == RANKING_SHA
        if built:
            raw = D.data_dir() / "ranking.parquet"
            ranking_rec["built_from"] = {"path": str(raw), "sha256": sha(raw),
                                         "bytes": raw.stat().st_size}
        inputs["ranking_frame"] = ranking_rec
        va = va.reset_index(drop=True)
        score_masks = {"final": np.ones(len(va), dtype=bool)}
    else:
        sc = {s.name: s.score_mask(side) for s, _ in arts}
        any_sc = np.logical_or.reduce(list(sc.values()))
        if (any_sc & lab).any():
            raise WP.LabelLeak(f"{a.fit}: {int((any_sc & lab).sum())} rows are both label and scored")
        va = A.loc[any_sc].drop(columns=["y"]).reset_index(drop=True)
        score_masks = {k: v[any_sc] for k, v in sc.items()}
    tr = A.loc[lab].reset_index(drop=True)
    side_tr = side.loc[lab].reset_index(drop=True)
    del A, side

    smoke_rec = False
    if a.smoke:
        rng = np.random.default_rng(SMOKE["seed"])
        keep_rows = (rng.random(len(tr)) < SMOKE["fraction"]) | tr["noac"].to_numpy().astype(bool)
        tr, side_tr = tr.loc[keep_rows].reset_index(drop=True), side_tr.loc[keep_rows].reset_index(drop=True)
        smoke_rec = {**SMOKE, "train_rows_kept": int(len(tr))}
    for s, _ in arts:
        s.assert_training_frame(side_tr, target=tr["y"].to_numpy(), what=f"{a.member} {a.fit}")
    if np.isin(va[ID].to_numpy(), tr[ID].to_numpy()).any():
        raise WP.LabelLeak(f"{a.fit}: a scored id is also a training id")
    log(f"train {len(tr):,} rows, score {len(va):,} rows")

    pv, per_seed, info = fit_predict(R, cfg, tr, va, rounds, log)
    if not np.isfinite(pv).all():
        raise RuntimeError(f"{int((~np.isfinite(pv)).sum())} non-finite predictions")

    pred_all = pd.DataFrame({ID: va[ID].to_numpy("float64"), "y_pred": pv})
    for i, p in enumerate(per_seed):
        pred_all[f"y_pred_s{i}"] = p
    for c in GATES:
        pred_all[c] = va[c].to_numpy()
    pred_all["nm_missing"] = pred_all["nm_missing"].astype("int8")
    pred_all["noac"] = pred_all["noac"].astype("int8")
    pred_all["d_takeoff_aobt"] = pred_all["d_takeoff_aobt"].astype("float64")
    pred_all["d_aobt_lobt"] = pred_all["d_aobt_lobt"].astype("float64")

    code = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        f = Path(f).resolve()
        if f.is_relative_to(src) and not f.is_relative_to(src / ".venv") and f.suffix == ".py":
            code[str(f.relative_to(src))] = sha(f)
    code_exp147 = {str(p.relative_to(EXP147_ROOT)): sha(p) for p in
                   (Path(__file__).resolve(), HERE.parent / "pilot.py",
                    EXP147_ROOT / "prc" / "weekpair.py")}
    import lightgbm
    libs = {"numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__,
            "python": platform.python_version()}
    if cfg["kind"] == "cat":
        import catboost
        libs["catboost"] = catboost.__version__
    record = {
        "member": a.member, "fit": a.fit, "artifacts": [s.name for s, _ in arts],
        "member_argv": argv, "member_command_source": spec["source"],
        "settings": {"rounds": rounds, "rounds_rule": ROUNDS_RULE,
                     "served_ranking_rounds": spec["served_rounds"],
                     "seeds": info["seeds"], "kind": cfg["kind"],
                     "depth": cfg["depth"] if cfg["kind"] == "cat" else None,
                     "loss": ("RMSE" if cfg["kind"] == "cat" else f"huber alpha={cfg['huber_delta']}"),
                     "learning_rate": cfg["lr"], "target": cfg["target"],
                     "device": a.device, "threads": a.threads,
                     "threads_observed": info["learner"]["threads_observed"],
                     "prc_gpu": os.environ.get("PRC_GPU"),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "noac_rounds": cfg["noac_rounds"] or R.NOAC_ROUNDS[cfg["kind"]],
                     "noac_min_rows": cfg["noac_min_rows"], "clip_floor": cfg["clip_floor"],
                     "residual_guard": cfg["residual_guard"], "guard_bounds": info["guard_bounds"],
                     "guard_rule": GUARD_RULE, "learner": info["learner"],
                     "cfg": {k: v for k, v in cfg.items()}},
        "smoke": smoke_rec,
        "inputs": inputs, "code": code, "code_exp147": code_exp147, "libraries": libs,
        "fit_seconds": info["fit_seconds"], "train_rows": int(len(tr)),
        "scored_rows_fit": int(len(va)), "features": info["features"],
        "noac_train_rows": info["noac_train_rows"], "noac_scored_rows_fit": info["noac_scored_rows"],
        "noac_head_used": info["noac_head_used"], "guard_touched_fit": info.get("guard_touched"),
        "raw_below_floor_fit": info["raw_below_floor"],
    }
    for s, owner in arts:
        pred = pred_all.loc[score_masks[s.name]].reset_index(drop=True)
        rec = {**record, "scored_rows": int(len(pred)),
               "seconds_total": round(time.monotonic() - t_start, 1)}
        path = P.write(out, stage, s, owner, pred, rec)
        log(f"wrote {path} ({len(pred):,} rows, owner {owner})")


if __name__ == "__main__":
    main()
