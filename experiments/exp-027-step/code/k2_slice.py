"""K2, the part of it that needs no competition data.

recorded rule (2026-09-07) put S1 -- the transient airport window -- on the board as a lane.
`window` (exp-028) takes the K1 arm onto stack-5; this file is the other half:
how much of the 19.4 s ceiling do OBSERVABLE window aggregates reach, and which
shape of aggregate does the reaching. K1 used six aggregates of two columns and
took +1.073 s of no-noac clean board. K2's real family adds the arrivals' unmasked
`TAXITIME_SEC_mvt` and `BLOCK_TIME_UTC_mvt`, the runway split and the
configuration change, none of which is in any committed out-of-fold file.

What IS reachable from committed OOF, and is measured here:

* **shape**, not source. Dispersion (p10, p90, sd), the row's **excess** over its
  own window, its **rank** inside the window, the window's **trend** against the
  previous one, and a second **scale** (the surrounding hour) are all forms K1 did
  not have and all of them are available on the two columns it did have. If the
  gain is in the shape, K2's arrival-side and runway blocks inherit it; if the
  gain stalls at K1's level, K2's expected value is the arrival side alone.
* the window distribution of the **head's own predictions**, which is legal at
  serve time (they are our output on the ranking months, not a label) and is the
  cheapest description of "what kind of window is this".
* where the gain lands, by regime and by airport, so K2 knows whether to gate.

Everything stays out-of-fold: the head's OOF is already rotated, and the probe
fits on two folds and scores the third.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_probe import load, FOLDS, SHARES  # noqa: E402

AGG = {"mean": "mean", "med": "median", "sd": "std"}


def _cell(d, win):
    return (d["airport"] + "|" + (d["t"] // win).astype("int64").astype(str)).to_numpy()


def _block(d, win, tag):
    """Window aggregates at one scale, plus the row's position inside its window."""
    g = pd.DataFrame({
        "c": _cell(d, win),
        "ta": d["d_takeoff_aobt"].to_numpy(), "al": d["d_aobt_lobt"].to_numpy(),
        "hp": d["y_pred"].to_numpy(),
        "late": (d["regime"] == "nm_late").to_numpy().astype(float),
        "nonm": d["nm_missing"].to_numpy().astype(float),
        "t": d["t"].to_numpy(),
    })
    a = g.groupby("c").agg(
        n=("t", "size"),
        ta_mean=("ta", "mean"), ta_med=("ta", "median"), ta_sd=("ta", "std"),
        ta_p10=("ta", lambda v: v.quantile(0.10)), ta_p90=("ta", lambda v: v.quantile(0.90)),
        al_mean=("al", "mean"), al_med=("al", "median"), al_p90=("al", lambda v: v.quantile(0.90)),
        hp_mean=("hp", "mean"), hp_sd=("hp", "std"), hp_p90=("hp", lambda v: v.quantile(0.90)),
        late=("late", "mean"), nonm=("nonm", "mean"), t0=("t", "min"),
    ).add_prefix(f"{tag}_").reset_index()
    out = pd.DataFrame({"c": g["c"]}).merge(a, on="c", how="left")
    # LEAD AND LAG. Measured to be the strong half of the family: K1's arm, which
    # had lead/lag and no shape, takes +1.162 s of no-noac clean board against
    # +0.760 s for a single-scale shape family without it. A window whose state is
    # rising is not the same state as one that is falling.
    apt = a["c"].str.rsplit("|", n=1).str[0]
    idx = a["c"].str.rsplit("|", n=1).str[1].astype("int64")
    for lag in (-2, -1, 1, 2):
        nb = a.copy()
        nb["c"] = apt + "|" + (idx - lag).astype(str)
        nb = nb.rename(columns={c: f"{c}_l{lag}" for c in nb.columns if c != "c"})
        out = out.merge(nb, on="c", how="left")
    for k in ("ta_med", "ta_mean", "hp_mean", "n"):
        out[f"{tag}_d_{k}"] = out[f"{tag}_{k}"] - out[f"{tag}_{k}_l-1"]
    # the row's own position inside its window: excess and rank
    out[f"{tag}_ta_excess"] = g["ta"].to_numpy() - out[f"{tag}_ta_med"].to_numpy()
    out[f"{tag}_al_excess"] = g["al"].to_numpy() - out[f"{tag}_al_med"].to_numpy()
    out[f"{tag}_hp_excess"] = g["hp"].to_numpy() - out[f"{tag}_hp_mean"].to_numpy()
    r = pd.Series(g["ta"].to_numpy()).groupby(g["c"].to_numpy()).rank(pct=True)
    out[f"{tag}_ta_rank"] = r.to_numpy()
    out[f"{tag}_since"] = g["t"].to_numpy() - out[f"{tag}_t0"].to_numpy()
    return out.drop(columns=["c"])


def features(d: pd.DataFrame, scales=(600.0, 3600.0)) -> pd.DataFrame:
    parts = [_block(d, w, f"w{int(w)}") for w in scales]
    X = pd.concat(parts, axis=1)
    ts = pd.to_datetime(d["t"], unit="s", utc=True)
    X["hour"] = ts.dt.hour.to_numpy()
    X["dow"] = ts.dt.dayofweek.to_numpy()
    X["apt"] = pd.Categorical(d["airport"], categories=sorted(d["airport"].unique())).codes
    X["own_ta"] = d["d_takeoff_aobt"].to_numpy()
    X["own_al"] = d["d_aobt_lobt"].to_numpy()
    X["y_pred"] = d["y_pred"].to_numpy()
    return X


def main(rounds: int = 400, leaves: int = 31, threads: int = 2, scales: str = "600,3600",
         offload_noac: int = 1, head: str = "blend5"):
    """`offload_noac` drops the `noac` rows from the fit and leaves their
    prediction untouched, the way `exp-016`'s stacker offloads them to the anchor.
    It is not a tidiness choice: 402 rows carry 20.45 % of the board budget, and a
    residual arm with enough capacity to notice them spends itself there. The
    single-scale run WITHOUT it puts 361 % of its gain on `noac` and loses 0.215 s
    of clean board while gaining 0.629 s outside them."""
    sc = tuple(float(s) for s in scales.split(","))
    data = {f: load(f, head) for f in FOLDS}
    X = {f: features(data[f], sc) for f in FOLDS}
    b2 = sum(SHARES[f] * (data[f]["err"] ** 2).mean() for f in SHARES)
    nnb2 = sum(SHARES[f] * (data[f].loc[data[f]["regime"] != "noac", "err"] ** 2).mean()
               for f in SHARES)
    print(f"K2 slice: scales={sc}  cols={X[FOLDS[0]].shape[1]}  rounds={rounds} leaves={leaves}"
          f"  offload_noac={bool(offload_noac)}  head={head}")
    print(f"  clean board {b2**0.5:.4f}   no-noac {nnb2**0.5:.4f}")
    after, nn_after = 0.0, 0.0
    per = {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        Xtr = pd.concat([X[g] for g in src], ignore_index=True)
        ytr = np.clip(pd.concat([data[g]["err"] for g in src], ignore_index=True), -1200, 1200)
        if offload_noac:
            keep = pd.concat([data[g]["regime"] != "noac" for g in src], ignore_index=True).to_numpy()
            Xtr, ytr = Xtr[keep], ytr[keep]
        m = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": leaves,
                       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "num_threads": threads, "seed": 11},
                      lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
        d = data[f]
        adj = m.predict(X[f])
        nn = (d["regime"] != "noac").to_numpy()
        if offload_noac:
            adj = np.where(nn, adj, 0.0)
        e0, e1 = d["err"].to_numpy(), d["err"].to_numpy() - adj
        r0, r1 = float(np.sqrt((e0 ** 2).mean())), float(np.sqrt((e1 ** 2).mean()))
        n0 = float(np.sqrt((e0[nn] ** 2).mean())); n1 = float(np.sqrt((e1[nn] ** 2).mean()))
        print(f"  {f:9s} {r0:8.3f} -> {r1:8.3f} ({r0-r1:+.3f})   no-noac {n0:8.3f} -> {n1:8.3f} ({n0-n1:+.3f})")
        per[f] = (d, e0, e1)
        if f in SHARES:
            after += SHARES[f] * (e1 ** 2).mean()
            nn_after += SHARES[f] * (e1[nn] ** 2).mean()
    print(f"  board {b2**0.5:.4f} -> {after**0.5:.4f}  gain {b2**0.5-after**0.5:+.3f} s"
          f"   |  no-noac {nnb2**0.5:.4f} -> {nn_after**0.5:.4f}  gain {nnb2**0.5-nn_after**0.5:+.3f} s")
    print("\n  where the gain lands (share of the board-weighted MSE removed):")
    for key in ("regime", "airport"):
        tot = sum(SHARES[f] * ((per[f][1] ** 2).mean() - (per[f][2] ** 2).mean()) for f in SHARES)
        rows = {}
        for f in SHARES:
            d, e0, e1 = per[f]
            n = len(d)
            for k, idx in d.groupby(key).indices.items():
                rows[k] = rows.get(k, 0.0) + SHARES[f] * ((e0[idx] ** 2).sum() - (e1[idx] ** 2).sum()) / n
        s = pd.Series(rows).sort_values(ascending=False)
        print(f"   by {key}: " + "  ".join(f"{k} {100*v/tot:+.0f}%" for k, v in s.items()))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]), a[3] if len(a) > 3 else "600,3600",
         int(a[4]) if len(a) > 4 else 1, a[5] if len(a) > 5 else "blend5")
