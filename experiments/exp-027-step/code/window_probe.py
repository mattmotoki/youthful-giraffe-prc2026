"""Kill test K1, runnable with no competition data: is the head's shared
airport-window error component reachable from window aggregates of columns the
ranking file gives us?

The budget (`docs/STRUCTURAL-STEP.md`) prices an oracle that knows the shared
airport x 10-minute component of the head's error at 18.8 s of clean board, and
measures that every one of blend-5's members carries it at the same size --
including `V3-allin`, the only member built on the surface + stand + weather
frame. So the component is real, large, and untouched.

This probe asks the cheapest version of "is it reachable": build window
aggregates from the ONLY observables the committed OOF carries
(`d_takeoff_aobt`, `d_aobt_lobt`, the regime mix, the movement count, the hour),
fit a small LightGBM on the head's residual over two folds, and score the third.
It is a LOWER bound on the real candidate -- the real one has the arrivals'
unmasked block and taxi-in times, the runway, the stand and the queue -- so a
positive here is decisive and a zero here is not a close.

SUPERSEDED, 2026-09-07 (round-1 review, blocking finding 2). This docstring used
to claim "everything is out-of-fold by construction ... the probe's own fit never
sees the fold it scores". The probe's own fit does not, but the head's residual
does: the blend weights behind december's prediction were fitted on july and
january, so training on december to score july carries july's labels. The nested
repair and the re-measured numbers are `code/k1_nested.py`; the figures this file
prints are the leaky ones and are kept only to reproduce the record.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
WIN = 600.0


#: Which head's residual to work on. `blend5` is this experiment's own rebuild;
#: `stack5` is `exp-026-rebuild`'s keep-noac stacker, clean board 258.0372, which
#: is the head the desk actually ships. The covariates (`regime`, `airport`,
#: `d_takeoff_aobt`, `d_aobt_lobt`, `t`) come from the blend-5 frame either way --
#: they are properties of the row, not of the head.
HEADS = {
    "blend5": ("experiments/exp-027-step/oof/blend5-{fold}.parquet", "y_pred"),
    "stack5": ("experiments/exp-026-rebuild/oof/stack5-{fold}.parquet", "y_pred"),
}


def load(fold: str, head: str = "blend5") -> pd.DataFrame:
    d = pd.read_parquet(ROOT / f"experiments/exp-027-step/oof/blend5-{fold}.parquet")
    t = pd.read_parquet(ROOT / f"experiments/exp-016-stack/oof/exp-016-oof-{fold}.parquet",
                        columns=["MVT_ID_mvt", "t"])
    d = d.merge(t, on="MVT_ID_mvt", how="left")
    if head != "blend5":
        path, col = HEADS[head]
        h = pd.read_parquet(ROOT / path.format(fold=fold))[["MVT_ID_mvt", col]]
        h = h.rename(columns={col: "_head"})
        d = d.merge(h, on="MVT_ID_mvt", how="inner")
        assert d["_head"].notna().all(), f"{head}: head prediction missing on some rows"
        d["y_pred"] = d.pop("_head")
    d = d[~d["MVT_ID_mvt"].isin(D.excluded_ids(fold))].reset_index(drop=True)
    d["err"] = d["y_pred"] - d["y_true"]
    return d


def window_features(d: pd.DataFrame, win: float = WIN) -> pd.DataFrame:
    """Aggregates of INPUT-READABLE columns over the departure's airport-window.

    Nothing here reads a label or a prediction: `d_takeoff_aobt`, `d_aobt_lobt`,
    the regime flags and the movement count are all computable from the ranking
    file alone. The window is the departure's own 10-minute bucket plus the two
    either side, so it is a state description, not a leak.
    """
    d = d.copy()
    d["bucket"] = (d["t"] // win).astype("int64")
    src = d.groupby(["airport", "bucket"]).agg(
        w_n=("t", "size"),
        w_ta_mean=("d_takeoff_aobt", "mean"), w_ta_med=("d_takeoff_aobt", "median"),
        w_ta_sd=("d_takeoff_aobt", "std"), w_ta_p90=("d_takeoff_aobt", lambda v: v.quantile(0.9)),
        w_al_mean=("d_aobt_lobt", "mean"), w_al_p90=("d_aobt_lobt", lambda v: v.quantile(0.9)),
        w_late=("regime", lambda v: (v == "nm_late").mean()),
        w_nonm=("nm_missing", "mean"),
    ).reset_index()
    out = d[["airport", "bucket"]].merge(src, on=["airport", "bucket"], how="left")
    # the same aggregates one and two buckets back and forward: a state that is
    # rising is not the same state as one that is falling
    for lag in (-2, -1, 1, 2):
        s = src.copy(); s["bucket"] = s["bucket"] - lag
        s = s.rename(columns={c: f"{c}_l{lag}" for c in s.columns if c.startswith("w_")})
        out = out.merge(s, on=["airport", "bucket"], how="left")
    out = out.drop(columns=["airport", "bucket"])
    ts = pd.to_datetime(d["t"], unit="s", utc=True)
    out["hour"] = ts.dt.hour.to_numpy()
    out["dow"] = ts.dt.dayofweek.to_numpy()
    out["apt"] = pd.Categorical(d["airport"], categories=sorted(d["airport"].unique())).codes
    out["own_ta"] = d["d_takeoff_aobt"].to_numpy()
    out["own_al"] = d["d_aobt_lobt"].to_numpy()
    out["y_pred"] = d["y_pred"].to_numpy()
    return out


ROW_ONLY = ["hour", "dow", "apt", "own_ta", "own_al", "y_pred"]


def main(rounds: int = 400, leaves: int = 31, threads: int = 1, win: float = WIN,
         arm: str = "window", offload_noac: int = 0):
    """`arm="row"` is the CONTROL: the same fit with every window aggregate
    removed, so a gain there is re-fitting on the row's own columns and not the
    window state. The window arm has to beat it or the finding is not about the
    window."""
    data = {f: load(f) for f in FOLDS}
    X = {f: window_features(data[f], win) for f in FOLDS}
    if arm == "row":
        X = {f: X[f][ROW_ONLY] for f in FOLDS}
    b2 = sum(SHARES[f] * (data[f]["err"] ** 2).mean() for f in SHARES)
    board = b2 ** 0.5
    print(f"head blend-5, clean board {board:.4f}   arm={arm}  window={win:.0f}s  "
          f"rounds={rounds}  cols={X[FOLDS[0]].shape[1]}")
    tot_after = 0.0
    nn_b2 = sum(SHARES[f] * ((data[f].loc[data[f]["regime"] != "noac", "y_pred"]
                              - data[f].loc[data[f]["regime"] != "noac", "y_true"]) ** 2).mean()
                for f in SHARES)
    nn_after = 0.0
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        Xtr = pd.concat([X[g] for g in src], ignore_index=True)
        # winsorised residual target: the fit must not be driven by the 0.3 % of
        # rows the tail lane owns, which have nothing to do with a window state
        ytr = np.clip(pd.concat([data[g]["err"] for g in src], ignore_index=True), -1200, 1200)
        if offload_noac:
            keep = pd.concat([data[g]["regime"] != "noac" for g in src],
                             ignore_index=True).to_numpy()
            Xtr, ytr = Xtr[keep], ytr[keep]
        m = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": leaves,
                       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "num_threads": threads, "seed": 11},
                      lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
        adj = m.predict(X[f])
        d = data[f]
        if offload_noac:
            adj = np.where((d["regime"] != "noac").to_numpy(), adj, 0.0)
        # `load` has already dropped this fold's unreached-rollover ids, so the
        # frame IS the clean population and `rmse_clean` would raise on it.
        r0 = D.rmse(d["y_true"], d["y_pred"])
        r1 = D.rmse(d["y_true"], d["y_pred"] - adj)
        nn = d["regime"] != "noac"
        r0n = float(np.sqrt(((d.loc[nn, "y_pred"] - d.loc[nn, "y_true"]) ** 2).mean()))
        r1n = float(np.sqrt(((d.loc[nn, "y_pred"] - adj[nn.to_numpy()] - d.loc[nn, "y_true"]) ** 2).mean()))
        print(f"  {f:9s} rmse {r0:8.3f} -> {r1:8.3f}  ({r0-r1:+.3f})   "
              f"no-noac {r0n:8.3f} -> {r1n:8.3f}  ({r0n-r1n:+.3f})   adj sd {adj.std():.1f}")
        if f in SHARES:
            tot_after += SHARES[f] * ((d["y_pred"] - adj - d["y_true"]) ** 2).mean()
            nn_after += SHARES[f] * r1n ** 2
    print(f"  board {board:.4f} -> {tot_after**0.5:.4f}   gain {board - tot_after**0.5:+.3f} s"
          f"   |  no-noac board {nn_b2**0.5:.4f} -> {nn_after**0.5:.4f}   gain {nn_b2**0.5 - nn_after**0.5:+.3f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]), float(a[3]), a[4] if len(a) > 4 else "window",
         int(a[5]) if len(a) > 5 else 0)
