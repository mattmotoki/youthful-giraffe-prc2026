"""The desk's construction (a): the window aggregates INSIDE stack-5's stacker.

The second-layer arm (`arm.py`) puts a residual model on top of a residual
stacker and pays its own refit cost — 2.2 s on the `row` control — which the
window columns then only buy back (`diagnose.py`).  This is the one-layer
version: the identical stacker, the identical rotated blend-5 anchor, the
identical five seeds and 300 rounds, with the window aggregates appended to the
member columns.  Nothing else changes, so the difference between the two
variants below is the window columns and nothing else.

Both variants are fitted in the same process on the same frames, and the
no-window variant must reproduce `exp-026`'s published rotation before either is
reported.  The comparison inherits the rotation's indirect channel — a scored
fold's labels reach the stacker through the other folds' member out-of-fold
columns (`exp-026` RESULTS §4a) — which is why the arm is compared against
stack-5 on the same instrument and never called nested.

    PRC_THREADS=2 python stack5w.py
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
import arrivals as Ar  # noqa: E402
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
R26 = P.ROOT / "experiments/exp-026-rebuild/code"
sys.path.insert(0, str(R26))
import frames as F  # noqa: E402
import rule as R  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

THREADS = int(os.environ.get("PRC_THREADS", "2"))
WIN = float(os.environ.get("PRC_WIN", "600"))
PUBLISHED_STACK5 = {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303}
#: Every artifact this script writes is named by the window width, because two
#: widths are two different candidates and a shared filename would let the second
#: run silently replace the first one's out-of-fold vectors.
TAG = f"w{int(WIN)}"
#: Which feature blocks to append to the member columns, one variant each.  `dep`
#: is the departure-side window block (`features.py`); `arr` is the arrivals'
#: unmasked taxi-in and lateness over the same window (`arrivals.py`), which no
#: committed out-of-fold file carries and which needs `data/raw`.
VARIANTS = os.environ.get("PRC_VARIANTS", "dep,arr,dep+arr").split(",")


def panel_of(m: pd.DataFrame) -> pd.DataFrame:
    """`features.window_features`'s canonical panel from exp-026's fold frame."""
    q = m[["MVT_ID_mvt", "airport", "t", "d_takeoff_aobt", "d_aobt_lobt",
           "regime", "nm_missing", "noac", "y_true"]].copy()
    q["base"] = 0.0          # dropped below; the stacker has the member columns
    return q


def block_for(fold: str, dep: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The feature block for one fold, from a spec `{blocks: [...], win: float}`.

    The one function `serve.py` and the rotation both call, so a candidate can
    never be served on a block the rotation did not measure.
    """
    pan = panel_of(dep) if "base" not in dep.columns else dep
    parts = []
    dep = [b for b in spec["blocks"] if b.startswith("dep")]
    if dep:
        X = Fe.window_features(pan, spec["win"])
        for b in dep:
            parts.append(Fe.select(X, b.removesuffix("d")))
            if b.endswith("d"):     # the deviation form of the same block
                parts.append(Fe.deviate(Fe.select(X, b.removesuffix("d")), pan, spec["win"]))
    if "arr" in spec["blocks"]:
        parts.append(Ar.for_fold(fold, pan, spec["win"]))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)


def ranking_blocks(rk: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """The feature block over the ranking months, in `rk`'s row order.

    `rk` is `serve_stack5.ranking_frame`'s output — the members joined to the
    gates table — so the panel is built from the raw file and then reindexed onto
    it.  The reindex is checked, not assumed.
    """
    raw = D.load_ranking(columns=sorted(set(Fe.RAW_COLS) | set(Ar.RAW_COLS)))
    pan = Fe.panel_from_movements(raw)
    pan["base"] = 0.0
    order = pd.Index(pan["MVT_ID_mvt"].to_numpy("float64")).get_indexer(
        rk["MVT_ID_mvt"].to_numpy("float64"))
    if (order < 0).any():
        raise SystemExit(f"{int((order < 0).sum())} member rows are not in the ranking file")
    parts = []
    dev = [b for b in spec["blocks"] if b.startswith("dep") and b.endswith("d")]
    if dev:
        # Sol round 1, Standards 9.  `block_for` builds the deviation form by
        # subtracting an airport baseline; here the frame is January and July
        # 2026 together, so one baseline per airport would mix two months and be
        # a different feature from the one the rotation fitted.  The deviation
        # form measured +0.061 s of board (RESULTS SS 4) and was not carried, so
        # this raises instead of serving a feature the rotation never scored.
        raise SystemExit(f"{'+'.join(dev)}: the deviation block is fit-only; it needs a "
                         f"per-month airport baseline before it can be served")
    if any(b.startswith("dep") for b in spec["blocks"]):
        X = Fe.window_features(pan, spec["win"])
        parts += [Fe.select(X, b) for b in spec["blocks"] if b.startswith("dep")]
    if "arr" in spec["blocks"]:
        parts.append(Ar.arrival_features(pan, Ar.arrival_table(raw), spec["win"]))
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1).iloc[order]
    return out.reset_index(drop=True)


def arm(fit, ap, seed, a_fit, a_ap, names, Xf=None, Xa=None):
    """`stack5.arm` with optional extra feature blocks concatenated."""
    Xfit, Xap = S.member_features(fit, names), S.member_features(ap, names)
    if Xf is not None:
        Xfit = pd.concat([Xfit.reset_index(drop=True), Xf.reset_index(drop=True)], axis=1)
        Xap = pd.concat([Xap.reset_index(drop=True), Xa.reset_index(drop=True)], axis=1)
    p = dict(S.LGB_BASE, num_threads=THREADS, seed=seed, bagging_seed=seed + 1,
             feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
    mdl = lgb.train(p, lgb.Dataset(Xfit, label=fit["y_true"].to_numpy("float64") - a_fit),
                    num_boost_round=S.ROUNDS)
    return mdl.predict(Xap) + a_ap


def main() -> None:
    t0 = time.time()
    names = S.B5
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}

    # the baseline gate exp-026 runs before any stacker number
    anchor = {}
    for g in F.FOLDS:
        src = pd.concat([data[h] for h in F.FOLDS if h != g], ignore_index=True)
        anchor[g] = R.apply_hybrid(data[g], names, R.fit_hybrid(src, names))
    for f in F.FOLDS:
        got = float(np.sqrt(np.mean((data[f]["y_true"].to_numpy("float64") - anchor[f]) ** 2)))
        if abs(got - S.PUBLISHED[f]) > 1e-3:
            raise SystemExit(f"blend-5 does not reproduce on {f}: {got:.4f}")
    print(f"[stack5w] blend-5 reproduces on all three folds, frames in "
          f"{time.time()-t0:.0f}s, win={WIN:.0f}s", flush=True)

    # the feature blocks, on the same rows, through the same functions serve.py uses
    need = {b for v in VARIANTS for b in v.split("+")}
    blocks = {}
    for f in F.FOLDS:
        pan = panel_of(data[f])
        got = {}
        for b in need:
            got[b] = block_for(f, pan, {"blocks": [b], "win": WIN})
        blocks[f] = got
    print("[stack5w] blocks " + "  ".join(
        f"{b}={blocks[F.FOLDS[0]][b].shape[1]}col" for b in sorted(need)), flush=True)

    def block(f: str, variant: str) -> pd.DataFrame:
        return pd.concat([blocks[f][b].reset_index(drop=True) for b in variant.split("+")],
                         axis=1)

    recs, saved = [], {}
    for f in F.FOLDS:
        ap, y = data[f], data[f]["y_true"].to_numpy("float64")
        ids = ap["MVT_ID_mvt"].to_numpy("float64")
        src = [g for g in F.FOLDS if g != f]
        fit = pd.concat([data[g] for g in src], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in src])
        ref = np.maximum(anchor[f], B.FLOOR)
        reg = ap["regime"].to_numpy()
        cl = S.clusters(ap)
        got = {"blend-5 (anchor)": ref}
        todo = [("stack-5", None, None)] + [
            (f"stack-5+{v}", pd.concat([block(g, v) for g in src], ignore_index=True),
             block(f, v)) for v in VARIANTS]
        for lab, xf, xa in todo:
            bag = np.maximum(np.mean([arm(fit, ap, s, a_fit, anchor[f], names, xf, xa)
                                      for s in S.SEEDS], axis=0), B.FLOOR)
            got[f"{lab} keep-noac"] = np.where(reg == "noac", ref, bag)
            got[f"{lab} full"] = bag
        saved[f] = got
        # COMMON.md's out-of-fold format for every variant this run measured, plus
        # the two vectors a serve-time run needs to reproduce: the anchor and the
        # stack-5 the variant is compared against.
        # `y_pred_anchor` (blend-5) is deliberately NOT written: it is already
        # committed three times over, in exp-026's and exp-027's out-of-fold files,
        # and this experiment reads it from there.  The `full` variants are in the
        # rotation table and nothing downstream needs their vectors.
        cols = {"MVT_ID_mvt": ids, "fold": f, "y_true": y,
                "y_pred_stack5": got["stack-5 keep-noac"]}
        for v in VARIANTS:
            cols[f"y_pred_{v.replace('+', '_')}"] = got[f"stack-5+{v} keep-noac"]
        cols["y_pred"] = got[f"stack-5+{VARIANTS[0]} keep-noac"]
        pd.DataFrame(cols).to_parquet(
            P.EXP / "oof" / f"exp-028-stack5w-{TAG}-{f}.parquet", index=False)
        # Sol round 1, Standards 2: the point estimate is clean, so the interval
        # is too.  `no_noac` drops the rows the served block never touches.
        ok = ~ap["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (reg != "noac")
        for lab, p in got.items():
            r = D.rmse_both(y, p, ids, f)
            ref5 = got["stack-5 keep-noac"]
            d5, lo5, hi5 = S.boot_delta(y[ok], p[ok], ref5[ok], pd.factorize(cl[ok])[0])
            dn, lon, hin = S.boot_delta(y[nn], p[nn], ref5[nn], pd.factorize(cl[nn])[0])
            recs.append(dict(fold=f, variant=lab, rmse_clean=r["rmse_clean"],
                             vs_stack5=d5, lo=lo5, hi=hi5,
                             vs_stack5_no_noac=dn, lo_no_noac=lon, hi_no_noac=hin,
                             **R.decomp(y, p, ap)))
        print(f"[{f}] " + "  ".join(
            f"{k}={D.rmse_clean(y, v, ids, f):.3f}" for k, v in got.items()), flush=True)

    T = pd.DataFrame(recs)
    T.round(4).to_csv(P.EXP / "analysis" / f"stack5w_rotation-{TAG}.csv", index=False)
    piv = T.pivot(index="variant", columns="fold", values="rmse_clean")[list(F.FOLDS)]
    piv["clean board"] = [P.board({f: piv.loc[v, f] for f in P.SHARES}) for v in piv.index]
    piv["vs stack-5"] = piv["clean board"] - piv.loc["stack-5 keep-noac", "clean board"]
    print("\n=== rotated, clean ===")
    print(piv.round(3).to_string())
    # Sol round 1, Standards 5: this used to be recorded and never enforced.
    off = {f: float(piv.loc["stack-5 keep-noac", f] - PUBLISHED_STACK5[f]) for f in F.FOLDS}
    if max(abs(v) for v in off.values()) > 1e-3:
        raise SystemExit(f"the no-window rotation does not reproduce exp-026's stack-5: {off}")
    print("\n=== day-clustered 95 % against stack-5 keep-noac (clean; then clean, no noac) ===")
    for r in T[T.variant.str.startswith("stack-5+")].sort_values(["variant", "fold"]).itertuples():
        print(f"  {r.variant:26s} {r.fold:9s} {r.vs_stack5:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]"
              f"   no-noac {r.vs_stack5_no_noac:+7.3f} "
              f"[{r.lo_no_noac:+7.3f}, {r.hi_no_noac:+7.3f}]")

    # the gate the assignment sets: every fold must move the right way
    gains = {v: {f: float(piv.loc["stack-5 keep-noac", f]
                          - piv.loc[f"stack-5+{v} keep-noac", f]) for f in F.FOLDS}
             for v in VARIANTS}
    for v in VARIANTS:
        print(f"\n  {v:9s} gains  " + "  ".join(f"{f} {gains[v][f]:+.3f}" for f in F.FOLDS)
              + f"   board {piv.loc['stack-5 keep-noac', 'clean board'] - piv.loc[f'stack-5+{v} keep-noac', 'clean board']:+.3f}"
              + f"   three-folds {all(x > 0 for x in gains[v].values())}")
    rec = {"win": WIN, "threads": THREADS, "lightgbm": lgb.__version__,
           "variants": VARIANTS,
           "block_cols": {b: int(blocks[F.FOLDS[0]][b].shape[1]) for b in sorted(need)},
           "reproduces_stack5": {f: round(float(piv.loc["stack-5 keep-noac", f]
                                                - PUBLISHED_STACK5[f]), 4) for f in F.FOLDS},
           "reproduces_tol": 1e-3,
           "clean": {v: {f: round(float(piv.loc[v, f]), 4) for f in F.FOLDS} for v in piv.index},
           "clean_board": {v: round(float(piv.loc[v, "clean board"]), 4) for v in piv.index},
           "gain_per_fold": {v: {f: round(x, 4) for f, x in gains[v].items()}
                             for v in VARIANTS},
           "three_folds": {v: bool(all(x > 0 for x in gains[v].values())) for v in VARIANTS},
           "wall_s": round(time.time() - t0, 1)}
    (P.EXP / "analysis" / f"stack5w-{TAG}.json").write_text(json.dumps(rec, indent=2))
    with open(P.EXP / "climb.log", "a") as fh:
        for v in VARIANTS:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M')}  stack5w win={WIN:.0f} blocks={v} "
                     f"(inside the stacker)  "
                     + "  ".join(f"{f} {gains[v][f]:+.3f}" for f in F.FOLDS)
                     + f"  board {rec['clean_board']['stack-5 keep-noac'] - rec['clean_board'][f'stack-5+{v} keep-noac']:+.3f}"
                     f"  three-folds {rec['three_folds'][v]}\n")
    print(f"\n[stack5w] {time.time() - t0:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
