"""The transient-window arm on stack-5: the rotation, the two controls, the screen.

One LightGBM per scored fold, fitted on the other two folds' stack-5 residual and
subtracted from stack-5.  Everything is measured against **stack-5 on the same
instrument** — same rows, same member out-of-fold files, same rotation — because
that is an instrument-relative delta: the same rotation stands behind both sides
(`SPEC.md` §3).  The channel is not closed here, nothing below is nested, and the
delta is not claimed to be free of the channel -- only measured against a base that
carries the same one.

Three arms, always run together:

  `window`   the aggregates plus the row columns — the candidate
  `row`      the identical fit with every window aggregate deleted
  `shuffle`  the identical fit with the window block permuted within airport-day

`row` prices re-fitting the head on its own columns; `shuffle` prices the
airport-day level and whatever the rotation hands out for free.  The arm's claim
is the distance between `window` and the larger of the two.

    python arm.py candidate       # the pre-registered configuration and its controls
    python arm.py screen          # width / rounds / leaves / fit-population sensitivity
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

EXP = P.EXP
THREADS = int(os.environ.get("PRC_THREADS", "2"))   # assignment share
#: Fixed before any stack-5 number was computed: the configuration K1 chose on
#: blend-5 (`experiments/exp-027-step/` §8d).  `SPEC.md` §1, last bullet.
CANDIDATE = dict(win=600.0, rounds=400, leaves=31, fit_pop="all", apply_pop="all",
                 use_base=True)
CLIP = 1200.0
LGB = dict(objective="regression", learning_rate=0.03, min_data_in_leaf=200,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           verbose=-1, seed=11, deterministic=True, force_row_wise=True)


def rebase(data: dict, which: str) -> dict:
    """Swap the vector the arm corrects.  `blend5` reproduces K1's base on stack-5's
    rows, which is how the two experiments' numbers are made comparable."""
    if which == "stack5":
        return data
    out = {}
    for f, d in data.items():
        e = d.copy()
        e["base"] = d["anchor"].to_numpy("float64")
        out[f] = e
    return out


def design(data: dict, win: float) -> dict:
    return {f: Fe.window_features(data[f], win) for f in P.FOLDS}


def _cols(X: pd.DataFrame, arm: str, use_base: bool) -> list[str]:
    """`win_only` deletes every row-level column, so the arm can add a window
    state and cannot re-calibrate the head on its own inputs."""
    if arm == "row":
        keep = list(Fe.ROW_ONLY)
    elif arm == "win_only":
        keep = Fe.window_cols(X)
    else:
        keep = list(X.columns)
    return [c for c in keep if use_base or c != "base"]


def run(data: dict, X: dict, arm: str = "window", rounds: int = 400, leaves: int = 31,
        fit_pop: str = "all", apply_pop: str = "all", use_base: bool = True,
        win: float = 600.0, seed: int = 11, boot: bool = False) -> tuple[pd.DataFrame, dict]:
    """One rotation. Returns (per-fold rows, {fold: adjusted vector})."""
    if arm == "shuffle":
        X = {f: Fe.shuffle_within_day(X[f], data[f], seed=1000 + i)
             for i, f in enumerate(P.FOLDS)}
    cols = _cols(X[P.FOLDS[0]], arm, use_base)
    rows, preds = [], {}
    for f in P.FOLDS:
        src = [g for g in P.FOLDS if g != f]
        # the fitting rows: the other two folds, minus the unreached-rollover ids
        # (whose labels are date artefacts, not taxi times) and minus the noac
        # rows when the arm is fitted outside them
        keep = []
        for g in src:
            k = ~data[g]["MVT_ID_mvt"].isin(D.excluded_ids(g)).to_numpy()
            if fit_pop == "no_noac":
                k &= (data[g]["regime"].to_numpy() != "noac")
            keep.append(k)
        Xtr = pd.concat([X[g].loc[k, cols] for g, k in zip(src, keep)], ignore_index=True)
        # winsorised residual: the fit must not be driven by the 0.3 % of rows the
        # tail lane owns, which have nothing to do with a window state
        ytr = np.clip(np.concatenate([(data[g]["base"].to_numpy("float64")
                                       - data[g]["y_true"].to_numpy("float64"))[k]
                                      for g, k in zip(src, keep)]), -CLIP, CLIP)
        p = dict(LGB, num_leaves=leaves, num_threads=THREADS, seed=seed,
                 bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
                 data_random_seed=seed + 3)
        mdl = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=rounds)

        ap = data[f]
        adj = mdl.predict(X[f][cols])
        if apply_pop == "no_noac":
            adj = np.where(ap["regime"].to_numpy() != "noac", adj, 0.0)
        base = ap["base"].to_numpy("float64")
        y = ap["y_true"].to_numpy("float64")
        ids = ap["MVT_ID_mvt"].to_numpy("float64")
        # `stack5.py`'s fold convention is the floor alone; the residual guard is a
        # serve-time step there.  `n_guard` prices that asymmetry rather than
        # assuming it away.
        new = np.maximum(base - adj, B.FLOOR)
        guarded = B.apply_residual_guard(new, ap["d_takeoff_aobt"].to_numpy("float64"),
                                         *(-20657.0, 9069.0), B.FLOOR)
        preds[f] = new
        r0 = D.rmse_both(y, base, ids, f)
        r1 = D.rmse_both(y, new, ids, f)
        nn = ap["regime"].to_numpy() != "noac"
        rec = dict(arm=arm, win=win, rounds=rounds, leaves=leaves, fit_pop=fit_pop,
                   apply_pop=apply_pop, use_base=use_base, fold=f, n=len(ap),
                   n_fit=len(Xtr), base_clean=r0["rmse_clean"], clean=r1["rmse_clean"],
                   gain_clean=r0["rmse_clean"] - r1["rmse_clean"],
                   base_all=r0["rmse"], all=r1["rmse"],
                   base_nonoac=float(np.sqrt(np.mean((base[nn] - y[nn]) ** 2))),
                   nonoac=float(np.sqrt(np.mean((new[nn] - y[nn]) ** 2))),
                   adj_sd=float(adj.std()), adj_mean=float(adj.mean()),
                   n_floor=int((base - adj < B.FLOOR).sum()),
                   n_guard=int((guarded != new).sum()),
                   guard_clean=D.rmse_clean(y, guarded, ids, f),
                   **{f"d_{k}": v for k, v in P.decomp(y, new, ap).items()})
        rec["gain_nonoac"] = rec["base_nonoac"] - rec["nonoac"]
        if boot:
            cl = P.clusters(ap)
            rec["delta"], rec["lo"], rec["hi"] = P.boot_delta(y, new, base, cl)
        rows.append(rec)
    return pd.DataFrame(rows), preds


def summarise(T: pd.DataFrame) -> pd.Series:
    """Board estimates and the three-fold gate for one arm's rotation."""
    g = T.set_index("fold")
    return pd.Series({
        "clean_board": P.board({f: g.loc[f, "clean"] for f in P.SHARES}),
        "base_board": P.board({f: g.loc[f, "base_clean"] for f in P.SHARES}),
        "nonoac_board": P.board({f: g.loc[f, "nonoac"] for f in P.SHARES}),
        "base_nonoac_board": P.board({f: g.loc[f, "base_nonoac"] for f in P.SHARES}),
        "worst_fold_gain": float(g["gain_clean"].min()),
        "three_folds": bool((g["gain_clean"] > 0).all()),
    })


def board_row(T: pd.DataFrame) -> dict:
    s = summarise(T)
    k = T.iloc[0]
    return dict(arm=k["arm"], win=k["win"], rounds=int(k["rounds"]), leaves=int(k["leaves"]),
                fit_pop=k["fit_pop"], apply_pop=k["apply_pop"], use_base=bool(k["use_base"]),
                july_loo=float(T.set_index("fold").loc["july_loo", "gain_clean"]),
                december=float(T.set_index("fold").loc["december", "gain_clean"]),
                january=float(T.set_index("fold").loc["january", "gain_clean"]),
                clean_board=float(s["clean_board"]),
                gain_board=float(s["base_board"] - s["clean_board"]),
                gain_nonoac=float(s["base_nonoac_board"] - s["nonoac_board"]),
                worst_fold=float(s["worst_fold_gain"]), three_folds=bool(s["three_folds"]))


def climb(line: str) -> None:
    with open(EXP / "climb.log", "a") as fh:
        fh.write(line.rstrip() + "\n")


def do_candidate() -> None:
    t0 = time.time()
    data = P.load()
    X = design(data, CANDIDATE["win"])
    print(f"[arm] base gate passed, clean board "
          f"{P.board({f: D.rmse_clean(data[f]['y_true'], data[f]['base'], data[f]['MVT_ID_mvt'], f) for f in P.FOLDS}):.4f}",
          flush=True)
    allrows, rows = [], []
    for arm in ("window", "row", "shuffle"):
        T, preds = run(data, X, arm=arm, boot=(arm == "window"), **CANDIDATE)
        allrows.append(T)
        rows.append(board_row(T))
        r = rows[-1]
        print(f"[{arm:8s}] july_loo {r['july_loo']:+7.3f}  december {r['december']:+7.3f}  "
              f"january {r['january']:+7.3f}  |  clean board {r['gain_board']:+7.3f}  "
              f"no-noac {r['gain_nonoac']:+7.3f}  three-folds {r['three_folds']}", flush=True)
        climb(f"{time.strftime('%Y-%m-%dT%H:%M')}  arm={arm} win={CANDIDATE['win']:.0f} "
              f"rounds={CANDIDATE['rounds']} leaves={CANDIDATE['leaves']} "
              f"fit={CANDIDATE['fit_pop']}/{CANDIDATE['apply_pop']}  "
              f"july_loo {r['july_loo']:+.3f}  december {r['december']:+.3f}  "
              f"january {r['january']:+.3f}  board {r['gain_board']:+.3f}  "
              f"no-noac {r['gain_nonoac']:+.3f}  gap-closed "
              f"{r['gain_board'] / 18.77:.3f} of STRUCTURAL-STEP's 18.8 s blend-5 "
              f"window oracle (this experiment's own oracle is in RESULTS SS 3)")
        if arm == "window":
            for f in P.FOLDS:
                ap = data[f]
                pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy("float64"),
                              "fold": f, "y_true": ap["y_true"].to_numpy("float64"),
                              "y_pred": preds[f],
                              "y_pred_base": ap["base"].to_numpy("float64"),
                              "adj": ap["base"].to_numpy("float64") - preds[f]}
                             ).to_parquet(EXP / "oof" / f"exp-028-window-{f}.parquet", index=False)
    T = pd.concat(allrows, ignore_index=True)
    T.to_csv(EXP / "analysis" / "candidate_rotation.csv", index=False)
    pd.DataFrame(rows).to_csv(EXP / "analysis" / "candidate_board.csv", index=False)
    (EXP / "analysis" / "candidate.json").write_text(json.dumps(
        {"config": CANDIDATE, "threads": THREADS, "lightgbm": lgb.__version__,
         "clip": CLIP, "arms": rows, "wall_s": round(time.time() - t0, 1)}, indent=2))
    print(f"\n[arm] {time.time() - t0:.0f}s wall", flush=True)


SCREEN = (
    [dict(CANDIDATE, win=w) for w in (300.0, 900.0)]
    + [dict(CANDIDATE, rounds=r) for r in (200, 800)]
    + [dict(CANDIDATE, leaves=l) for l in (15, 63)]
    + [dict(CANDIDATE, fit_pop="no_noac"),
       dict(CANDIDATE, fit_pop="no_noac", apply_pop="no_noac"),
       dict(CANDIDATE, apply_pop="no_noac"),
       dict(CANDIDATE, use_base=False)]
)


def do_screen() -> None:
    t0 = time.time()
    data = P.load()
    cache, rows, allrows = {}, [], []
    for cfg in SCREEN:
        w = cfg["win"]
        if w not in cache:
            cache[w] = design(data, w)
        T, _ = run(data, cache[w], arm="window", **cfg)
        allrows.append(T)
        rows.append(board_row(T))
        r = rows[-1]
        print(f"[screen] win={w:4.0f} rounds={cfg['rounds']:3d} leaves={cfg['leaves']:2d} "
              f"fit={cfg['fit_pop']:7s} apply={cfg['apply_pop']:7s} base={int(cfg['use_base'])}  "
              f"folds {r['july_loo']:+6.3f} {r['december']:+6.3f} {r['january']:+6.3f}  "
              f"board {r['gain_board']:+6.3f}  no-noac {r['gain_nonoac']:+6.3f}  "
              f"3f {r['three_folds']}", flush=True)
        climb(f"{time.strftime('%Y-%m-%dT%H:%M')}  screen win={w:.0f} rounds={cfg['rounds']} "
              f"leaves={cfg['leaves']} fit={cfg['fit_pop']}/{cfg['apply_pop']} "
              f"base={int(cfg['use_base'])}  july_loo {r['july_loo']:+.3f}  "
              f"december {r['december']:+.3f}  january {r['january']:+.3f}  "
              f"board {r['gain_board']:+.3f}  no-noac {r['gain_nonoac']:+.3f}")
    pd.concat(allrows, ignore_index=True).to_csv(EXP / "analysis" / "screen_rotation.csv",
                                                 index=False)
    pd.DataFrame(rows).to_csv(EXP / "analysis" / "screen_board.csv", index=False)
    print(f"\n[screen] {time.time() - t0:.0f}s wall", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "candidate"
    {"candidate": do_candidate, "screen": do_screen}[what]()
