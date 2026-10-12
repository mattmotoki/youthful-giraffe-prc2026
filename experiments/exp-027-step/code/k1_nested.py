"""K1 re-measured under a NESTED rotation, plus a leave-one-out window control.

Round-1 review (`docs/reviews/step-round1.md`, blocking finding 2) is correct and
this file is the repair. `code/window_probe.py` claimed the probe "never sees the
fold it scores". The probe's own fit does not, but the head's residual does: the
blend weights that produced december's prediction were fitted on july and
january, so a december residual carries july's labels. When K1 trains on
december and january and scores july, july's labels reach the fit through that
door.

The repair is a nested rotation. To score fold `f`, the training folds' head
predictions are rebuilt from weights fitted on the ONE remaining fold, which is
neither `f` nor the fold being predicted. The scored fold `f` keeps its ordinary
rotated prediction, whose weights already exclude `f`. Nothing in the chain that
produces a number for `f` has seen `f`.

Second control, from the same review (non-blocking): the departure's own
600 s bucket contains the departure itself, so a window aggregate is not proof
that the signal comes from OTHER departures. `--loo` removes the row's own
contribution from its own bucket (count n-1, mean (sum-x)/(n-1)); the lead and
lag buckets cannot contain the row and are untouched. If the gain survives the
leave-one-out, the signal is the neighbours.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.blend as B      # noqa: E402
import prc.data as D       # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_oof as H       # noqa: E402

FOLDS = H.FOLDS
SHARES = H.SHARES


def nested_preds(target: str = "nm_late", mode: str = "nnls", outer: str = "nnls"):
    """`out[f][g]` = head prediction on fold `g` from weights fitted on the one
    fold that is neither `f` nor `g`. With three folds that source is unique."""
    data = {f: H.fold_data(f) for f in FOLDS}
    k = len(data[FOLDS[0]]["names"])
    out = {f: {} for f in FOLDS}
    for f in FOLDS:
        for g in FOLDS:
            if g == f:
                continue
            h = [x for x in FOLDS if x not in (f, g)][0]
            selh = (data[h]["regime"] == target) & data[h]["ok"]
            w5 = B.fit_weights(data[h]["P"][selh], data[h]["y"][selh], mode)
            w4 = B.fit_weights(data[h]["P"][~selh][:, :k - 1], data[h]["y"][~selh], outer)
            d = data[g]
            sel = (d["regime"] == target) & d["ok"]
            p = np.empty(len(d["y"]))
            p[sel] = d["P"][sel] @ w5
            p[~sel] = d["P"][~sel][:, :k - 1] @ w4
            out[f][g] = np.maximum(p, B.FLOOR)
    return out, data


def frames() -> dict:
    fr = {}
    for f in FOLDS:
        d = pd.read_parquet(ROOT / f"experiments/exp-027-step/oof/blend5-{f}.parquet")
        t = pd.read_parquet(ROOT / f"experiments/exp-016-stack/oof/exp-016-oof-{f}.parquet",
                            columns=["MVT_ID_mvt", "t"])
        fr[f] = d.merge(t, on="MVT_ID_mvt", how="left")
    return fr


def feats(d: pd.DataFrame, win: float, loo: bool) -> pd.DataFrame:
    d = d.copy()
    d["bucket"] = (d["t"] // win).astype("int64")
    g = d.groupby(["airport", "bucket"])
    # Round-2 review, blocking finding 1: `sum` skips nulls but `size` counts
    # them, so sum/size understated every mean in a window holding a null
    # `d_takeoff_aobt` (3,938 / 1,649 / 1,435 rows per fold). Carry the non-null
    # COUNT per column and divide by that, here and in the leave-one-out.
    src = g.agg(w_n=("t", "size"),
                w_ta_sum=("d_takeoff_aobt", "sum"), w_ta_cnt=("d_takeoff_aobt", "count"),
                w_al_sum=("d_aobt_lobt", "sum"), w_al_cnt=("d_aobt_lobt", "count"),
                w_late_sum=("regime", lambda v: (v == "nm_late").sum()),
                w_nonm_sum=("nm_missing", "sum"),
                w_ta_med=("d_takeoff_aobt", "median"), w_ta_sd=("d_takeoff_aobt", "std"),
                w_ta_p90=("d_takeoff_aobt", lambda v: v.quantile(0.9)),
                w_al_p90=("d_aobt_lobt", lambda v: v.quantile(0.9))).reset_index()
    own = d[["airport", "bucket"]].merge(src, on=["airport", "bucket"], how="left")
    def _div(num, den):
        den = np.asarray(den, "float64")
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    if loo:
        # remove the departure's own contribution from its own bucket; the
        # order-statistic columns cannot be corrected this way and are dropped.
        # A row whose own value is null contributes nothing to the sum and
        # nothing to the count, so both subtractions are gated on notna.
        n = own["w_n"].to_numpy("float64") - 1.0
        ta, al = d["d_takeoff_aobt"].to_numpy("float64"), d["d_aobt_lobt"].to_numpy("float64")
        ta_ok, al_ok = np.isfinite(ta), np.isfinite(al)
        out = pd.DataFrame({
            "w_n": n,
            "w_ta_mean": _div(own["w_ta_sum"].to_numpy() - np.where(ta_ok, ta, 0.0),
                              own["w_ta_cnt"].to_numpy() - ta_ok.astype(float)),
            "w_al_mean": _div(own["w_al_sum"].to_numpy() - np.where(al_ok, al, 0.0),
                              own["w_al_cnt"].to_numpy() - al_ok.astype(float)),
            "w_late": _div(own["w_late_sum"].to_numpy()
                           - (d["regime"] == "nm_late").to_numpy().astype(float), n),
            "w_nonm": _div(own["w_nonm_sum"].to_numpy() - d["nm_missing"].to_numpy(), n),
        })
    else:
        out = pd.DataFrame({
            "w_n": own["w_n"].to_numpy("float64"),
            "w_ta_mean": _div(own["w_ta_sum"].to_numpy(), own["w_ta_cnt"].to_numpy()),
            "w_al_mean": _div(own["w_al_sum"].to_numpy(), own["w_al_cnt"].to_numpy()),
            "w_late": _div(own["w_late_sum"].to_numpy(), own["w_n"].to_numpy()),
            "w_nonm": _div(own["w_nonm_sum"].to_numpy(), own["w_n"].to_numpy()),
            "w_ta_med": own["w_ta_med"].to_numpy(), "w_ta_sd": own["w_ta_sd"].to_numpy(),
            "w_ta_p90": own["w_ta_p90"].to_numpy(), "w_al_p90": own["w_al_p90"].to_numpy(),
        })
    # lead and lag buckets never contain the row, so they need no correction
    lagsrc = src.copy()
    lagsrc["w_ta_mean"] = _div(lagsrc["w_ta_sum"], lagsrc["w_ta_cnt"])
    lagsrc["w_al_mean"] = _div(lagsrc["w_al_sum"], lagsrc["w_al_cnt"])
    lagsrc["w_late"] = _div(lagsrc["w_late_sum"], lagsrc["w_n"])
    lagsrc["w_nonm"] = _div(lagsrc["w_nonm_sum"], lagsrc["w_n"])
    lagsrc = lagsrc.drop(columns=["w_ta_sum", "w_ta_cnt", "w_al_sum", "w_al_cnt",
                                  "w_late_sum", "w_nonm_sum"])
    key = d[["airport", "bucket"]]
    for lag in (-2, -1, 1, 2):
        s = lagsrc.copy(); s["bucket"] = s["bucket"] - lag
        s = s.rename(columns={c: f"{c}_l{lag}" for c in s.columns if c.startswith("w_")})
        j = key.merge(s, on=["airport", "bucket"], how="left").drop(columns=["airport", "bucket"])
        out = pd.concat([out.reset_index(drop=True), j.reset_index(drop=True)], axis=1)
    ts = pd.to_datetime(d["t"], unit="s", utc=True)
    out["hour"] = ts.dt.hour.to_numpy()
    out["dow"] = ts.dt.dayofweek.to_numpy()
    out["apt"] = pd.Categorical(d["airport"], categories=sorted(d["airport"].unique())).codes
    out["own_ta"] = d["d_takeoff_aobt"].to_numpy()
    out["own_al"] = d["d_aobt_lobt"].to_numpy()
    return out


ROW_ONLY = ["hour", "dow", "apt", "own_ta", "own_al", "y_pred"]


def main(rounds=400, leaves=31, threads=1, win=600.0, arm="window", offload=1,
         nested=1, loo=0):
    fr = frames()
    ids = {f: fr[f]["MVT_ID_mvt"].to_numpy() for f in FOLDS}
    npred, data = (nested_preds() if nested else (None, None))
    if nested:  # align the nested vectors to the parquet row order
        for f in FOLDS:
            for g in npred[f]:
                order = pd.Series(np.arange(len(data[g]["frame"])),
                                  index=data[g]["frame"]["MVT_ID_mvt"].to_numpy())
                npred[f][g] = npred[f][g][order.reindex(ids[g]).to_numpy()]
    keep = {f: ~pd.Series(ids[f]).isin(D.excluded_ids(f)).to_numpy() for f in FOLDS}
    X = {f: feats(fr[f], win, loo) for f in FOLDS}
    err0 = {f: (fr[f]["y_pred"] - fr[f]["y_true"]).to_numpy() for f in FOLDS}
    b2 = sum(SHARES[f] * (err0[f][keep[f]] ** 2).mean() for f in SHARES)
    nn = {f: ((fr[f]["regime"] != "noac").to_numpy() & keep[f]) for f in FOLDS}
    nn_b2 = sum(SHARES[f] * (err0[f][nn[f]] ** 2).mean() for f in SHARES)
    print(f"head blend-5  nested={nested}  loo={loo}  arm={arm}  win={win:.0f}s  "
          f"cols={X[FOLDS[0]].shape[1] + 1}  clean board {b2**0.5:.4f}  no-noac {nn_b2**0.5:.4f}")
    tot, nna = 0.0, 0.0
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        Xg, yg, kg = [], [], []
        for g in src:
            x = X[g].copy()
            p = npred[f][g] if nested else fr[g]["y_pred"].to_numpy()
            x["y_pred"] = p
            Xg.append(x)
            yg.append(np.clip(p - fr[g]["y_true"].to_numpy(), -1200, 1200))
            k = keep[g].copy()
            if offload:
                k &= (fr[g]["regime"] != "noac").to_numpy()
            kg.append(k)
        Xtr = pd.concat(Xg, ignore_index=True)
        if arm == "row":
            Xtr = Xtr[ROW_ONLY]
        ytr = np.concatenate(yg)
        km = np.concatenate(kg)
        m = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": leaves,
                       "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "num_threads": threads, "seed": 11},
                      lgb.Dataset(Xtr[km], ytr[km]), num_boost_round=rounds)
        Xf = X[f].copy(); Xf["y_pred"] = fr[f]["y_pred"].to_numpy()
        adj = m.predict(Xf[ROW_ONLY] if arm == "row" else Xf)
        if offload:
            adj = np.where((fr[f]["regime"] != "noac").to_numpy(), adj, 0.0)
        e0, e1 = err0[f], err0[f] - adj
        r0 = float(np.sqrt((e0[keep[f]] ** 2).mean())); r1 = float(np.sqrt((e1[keep[f]] ** 2).mean()))
        r0n = float(np.sqrt((e0[nn[f]] ** 2).mean())); r1n = float(np.sqrt((e1[nn[f]] ** 2).mean()))
        print(f"  {f:9s} rmse {r0:8.3f} -> {r1:8.3f} ({r0-r1:+.3f})   "
              f"no-noac {r0n:8.3f} -> {r1n:8.3f} ({r0n-r1n:+.3f})")
        # the four slices COMMON.md requires of every RESULTS table
        k = keep[f]
        lo = k & (fr[f]["y_true"] <= 3600).to_numpy()
        hi = k & (fr[f]["y_true"] > 3600).to_numpy()
        nmz = k & (fr[f]["nm_missing"] == 1).to_numpy()
        rr = lambda e, m: float(np.sqrt((e[m] ** 2).mean()))  # noqa: E731
        print(f"            slices  y<=3600 {rr(e0,lo):8.3f} -> {rr(e1,lo):8.3f}   "
              f"y>3600 {rr(e0,hi):9.3f} -> {rr(e1,hi):9.3f}   "
              f"no-NM {rr(e0,nmz):8.3f} -> {rr(e1,nmz):8.3f}")
        if f in SHARES:
            tot += SHARES[f] * (e1[keep[f]] ** 2).mean()
            nna += SHARES[f] * r1n ** 2
    print(f"  board {b2**0.5:.4f} -> {tot**0.5:.4f}  gain {b2**0.5-tot**0.5:+.3f} s"
          f"   |  no-noac {nn_b2**0.5:.4f} -> {nna**0.5:.4f}  gain {nn_b2**0.5-nna**0.5:+.3f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]), float(a[3]), a[4], int(a[5]), int(a[6]), int(a[7]))
