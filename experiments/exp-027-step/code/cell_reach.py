"""K2's real question, asked without a residual layer: can OBSERVABLE window
aggregates -- including the arrivals' unmasked taxi-in and block times, the runway
split and the configuration change -- predict the shared component of the head's
error?

Why not just fit a layer. `window` (exp-028) measured that the K1 arm on top of
stack-5 gives +0.002 s, and this experiment measures +0.809 s no-noac from a much
richer family but with january moving the wrong way. A second residual layer over
a stacker pays a refit cost that the window columns then have to buy back, so a
layer result confounds "is the signal there" with "is a layer the way to deliver
it". It is not, and the desk has ruled that K2 goes inside a member frame.

So this file asks the question at the CELL level, where there is no refit cost:
the target is the airport-window MEAN residual, the features are that window's
observable aggregates, the fit is on two folds and the score on the third, and the
answer is the fraction of the shared mass a predictor of the cell mean recovers.
Multiplied by the ceiling in `shared_component.py`, that is what K2 is worth.

Legality (`docs/input-rule.md`): departure `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` are blanked by `prc.data.mask_like_ranking` before anything is
computed, and are re-blanked defensively here. Arrival block and taxi times are
kept, which the rule states explicitly and `prc.surface_features` already relies
on. One month at a time, always.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
from window_probe import load, FOLDS, SHARES  # noqa: E402

WIN = 600.0


def _ep(s):
    v = s.to_numpy(dtype="datetime64[ns]")
    out = v.astype("int64").astype("float64")
    out[np.isnat(v)] = np.nan
    return out / 1e9


def month_cells(fold: str, win: float = WIN) -> pd.DataFrame:
    """Observable airport-window aggregates for one validation month.

    Built from the month frame masked exactly as the ranking file is, so every
    column here exists for January and July 2026 too.
    """
    month = D.FOLDS[fold].valid_month
    raw = D.load_month(month)
    masked, _ = D.mask_like_ranking(raw)
    ph = masked["PHASE_mvt"].to_numpy()
    dep, arr = ph == "DEP", ph == "ARR"
    apt = np.where(dep, masked["ADEP_mvt"].astype(str), masked["ADES_mvt"].astype(str))
    t = _ep(masked["MVT_TIME_UTC_mvt"])
    blk = _ep(masked["BLOCK_TIME_UTC_mvt"])
    taxi = np.array(masked["TAXITIME_SEC_mvt"], dtype="float64")
    blk[dep] = np.nan            # defensive: never read a departure's own label
    taxi[dep] = np.nan
    aobt = _ep(masked["AOBT_3_flt"])
    lobt = _ep(masked["LOBT_flt"])
    rwy = masked["RUNWAY_mvt"].astype(str).to_numpy()

    b = (t // win).astype("int64")
    dd = pd.DataFrame({"apt": apt[dep], "b": b[dep], "rwy": rwy[dep],
                       "ta": t[dep] - aobt[dep], "al": aobt[dep] - lobt[dep]})
    aa = pd.DataFrame({"apt": apt[arr], "b": b[arr], "rwy": rwy[arr],
                       "taxiin": taxi[arr], "onblk": blk[arr], "land": t[arr]})
    q = lambda p: (lambda v: v.quantile(p))  # noqa: E731
    dg = dd.groupby(["apt", "b"]).agg(
        d_n=("ta", "size"), d_ta_mean=("ta", "mean"), d_ta_med=("ta", "median"),
        d_ta_sd=("ta", "std"), d_ta_p10=("ta", q(0.1)), d_ta_p90=("ta", q(0.9)),
        d_al_med=("al", "median"), d_al_p90=("al", q(0.9)),
        d_rwy_n=("rwy", "nunique"),
    )
    ag = aa.groupby(["apt", "b"]).agg(
        a_n=("taxiin", "size"), a_ti_mean=("taxiin", "mean"), a_ti_med=("taxiin", "median"),
        a_ti_sd=("taxiin", "std"), a_ti_p90=("taxiin", q(0.9)),
        a_ti_cnt=("taxiin", "count"), a_rwy_n=("rwy", "nunique"),
    )
    ag["a_ti_iqr"] = aa.groupby(["apt", "b"])["taxiin"].apply(
        lambda v: v.quantile(0.75) - v.quantile(0.25))

    # Round-2 review, blocking finding 4: `onblk` and `land` were built and never
    # aggregated, so the arrivals' BLOCK TIME was not in the feature set at all
    # and the claim that it was is false. It is now, on its own grid: an arrival
    # that lands in one window goes on-block in another, and "how many aircraft
    # reached a stand in this window" is a different state from "how many landed".
    ab = aa.assign(bb=np.floor(aa["onblk"].to_numpy() / win))
    ab = ab[np.isfinite(ab["bb"])]
    ab["bb"] = ab["bb"].astype("int64")
    bg = (ab.groupby(["apt", "bb"])
            .agg(a_blk_n=("onblk", "size"), a_blk_ti_mean=("taxiin", "mean"),
                 a_blk_ti_p90=("taxiin", q(0.9)),
                 a_blk_land_mean=("land", "mean"), a_blk_onblk_mean=("onblk", "mean"))
            .reset_index().rename(columns={"bb": "b"}))
    # the window's own clock, so a mean epoch becomes an offset inside the window
    bg["a_blk_land_off"] = bg["a_blk_land_mean"] - bg["b"] * win
    bg["a_blk_onblk_off"] = bg["a_blk_onblk_mean"] - bg["b"] * win
    bg = bg.drop(columns=["a_blk_land_mean", "a_blk_onblk_mean"])
    ag = ag.reset_index().merge(bg, on=["apt", "b"], how="outer").set_index(["apt", "b"])
    # the busiest departure runway and the busiest arrival runway, and whether
    # either changed against the previous window: a configuration turn-over
    dmode = dd.groupby(["apt", "b"])["rwy"].agg(lambda v: v.mode().iat[0] if len(v) else "")
    amode = aa.groupby(["apt", "b"])["rwy"].agg(lambda v: v.mode().iat[0] if len(v) else "")
    C = dg.join(ag, how="outer").join(dmode.rename("d_rwy_mode")).join(amode.rename("a_rwy_mode"))
    C = C.reset_index()
    C["hour"] = ((C["b"] * win) % 86400) // 3600
    for src, tag in ((dmode.rename("d_rwy_mode"), "d"), (amode.rename("a_rwy_mode"), "a")):
        prev = src.reset_index(); prev["b"] = prev["b"] + 1
        prev = prev.rename(columns={prev.columns[-1]: f"{tag}_rwy_prev"})
        C = C.merge(prev, on=["apt", "b"], how="left")
        C[f"{tag}_cfg_changed"] = (C[f"{tag}_rwy_mode"] != C[f"{tag}_rwy_prev"]).astype(float)
    # keep the busiest-runway modes as features in their own right (round-2
    # finding 4: they were dropped after the change flag was derived, so the
    # "busiest-runway mode" the documents claimed was not in the set)
    for cc in ("d_rwy_mode", "a_rwy_mode"):
        C[f"{cc}_code"] = pd.factorize(C[cc].astype(str))[0].astype("float64")
    C = C.drop(columns=["d_rwy_mode", "a_rwy_mode", "d_rwy_prev", "a_rwy_prev"])
    # lead and lag on the numeric blocks
    num = [c for c in C.columns if c not in ("apt", "b", "hour")]
    for lag in (-2, -1, 1, 2):
        s = C[["apt", "b"] + num].copy(); s["b"] = s["b"] - lag
        s = s.rename(columns={c: f"{c}_l{lag}" for c in num})
        C = C.merge(s, on=["apt", "b"], how="left")
    return C


def main(minn: int = 4, rounds: int = 300, threads: int = 2, head: str = "stack5",
         arm: str = "full", win: float = WIN):
    """`arm="dep"` drops every arrival-side and runway column: the control that
    says whether the arrivals carry state that `MVT − AOBT` does not."""
    data = {f: load(f, head) for f in FOLDS}
    cells = {f: month_cells(f, win) for f in FOLDS}
    T = {}
    for f in FOLDS:
        d = data[f]
        d = d[d["regime"] != "noac"]
        g = pd.DataFrame({"apt": d["airport"].to_numpy(),
                          "b": (d["t"] // win).astype("int64").to_numpy(),
                          "e": d["err"].to_numpy()}).groupby(["apt", "b"])["e"]
        t = pd.DataFrame({"m": g.mean(), "n": g.size(), "v": g.var(ddof=1)}).reset_index()
        t = t[(t["n"] >= minn) & t["v"].notna()]
        T[f] = t.merge(cells[f], on=["apt", "b"], how="left")
    drop = ["apt", "b", "m", "n", "v"]
    cols = [c for c in T[FOLDS[0]].columns if c not in drop]
    if arm == "dep":
        cols = [c for c in cols if not (c.startswith("a_") or "rwy" in c or "cfg" in c)]
    print(f"head={head}  arm={arm}  win={win:.0f}s  minn={minn}  cells/fold="
          f"{ {f: len(T[f]) for f in FOLDS} }  cols={len(cols)}")
    # Pass 1: the cross-fitted prediction for every fold. Each fold's predictor
    # is trained on the other two only.
    P = {}
    for f in FOLDS:
        src = [g for g in FOLDS if g != f]
        tr = pd.concat([T[g] for g in src], ignore_index=True)
        m = lgb.train({"objective": "regression", "learning_rate": 0.04, "num_leaves": 31,
                       "min_data_in_leaf": 60, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "num_threads": threads, "seed": 11},
                      lgb.Dataset(tr[cols], tr["m"], weight=tr["n"]), num_boost_round=rounds)
        P[f] = m.predict(T[f][cols])

    # Pass 2: score. Round-2 review, blocking finding 5: the shrinkage alpha was
    # chosen with the SAME fold's target, which makes "at the best scale" an
    # in-sample number. alpha for fold f is now fitted on the OTHER folds'
    # cross-fitted predictions, so the shrunk arm is honestly out of sample. The
    # in-sample figure is printed beside it, because the gap between them is the
    # size of the optimism the review found.
    tot_sh = tot_ex = 0.0
    bests, bests_is = [], []
    for f in FOLDS:
        t, p = T[f], P[f]
        n, mm, v = t["n"].to_numpy(), t["m"].to_numpy(), t["v"].to_numpy()
        shared = (n * mm ** 2 - v).sum()                 # unbiased shared mass in these cells
        # mass a predictor of the cell mean removes: n*(m^2 - (m-p)^2) = n*(2mp - p^2).
        # `m` carries noise of variance v/n, but that noise is independent of `p`
        # (fitted on the other two folds), so this is unbiased for the mass a
        # predictor of the TRUE cell mean removes.
        removed = (n * (2 * mm * p - p ** 2)).sum()
        num = den = 0.0
        for g in FOLDS:
            if g == f:
                continue
            ng, mg, pg = T[g]["n"].to_numpy(), T[g]["m"].to_numpy(), P[g]
            num += (ng * mg * pg).sum(); den += (ng * pg ** 2).sum()
        alpha = num / max(den, 1e-9)                     # out of fold
        best = (n * (2 * mm * (alpha * p) - (alpha * p) ** 2)).sum()
        a_is = (n * mm * p).sum() / max((n * p ** 2).sum(), 1e-9)   # in sample, for comparison
        best_is = a_is * (n * mm * p).sum()
        print(f"  {f:9s} cells {len(t):>6,}  shared {shared:12,.0f}  removed {removed:12,.0f}"
              f" = {100*removed/shared:5.1f} %   at best scale {100*best/shared:5.1f} %"
              f" (alpha {alpha:.2f} oof, {a_is:.2f} in-sample -> {100*best_is/shared:.1f} %)"
              f"   corr(p,m) {np.corrcoef(p, mm)[0,1]:+.3f}")
        if f in SHARES:
            tot_sh += SHARES[f] * shared / len(data[f])
            tot_ex += SHARES[f] * removed / len(data[f])
            bests.append(SHARES[f] * best / len(data[f]))
            bests_is.append(SHARES[f] * best_is / len(data[f]))
    tot_bs = sum(bests)
    tot_bs_is = sum(bests_is)
    b2 = sum(SHARES[f] * (data[f]["err"] ** 2).mean() for f in SHARES)
    board = b2 ** 0.5
    ceil = board - np.sqrt(max(b2 - tot_sh, 0))
    got = board - np.sqrt(max(b2 - tot_ex, 0))
    bs = board - np.sqrt(max(b2 - tot_bs, 0))
    bs_is = board - np.sqrt(max(b2 - tot_bs_is, 0))
    print(f"  board {board:.4f}  ceiling on these cells {ceil:+.2f} s  "
          f"REACHED {got:+.2f} s ({100*tot_ex/tot_sh:.1f} %)   at the best scale "
          f"{bs:+.2f} s ({100*tot_bs/tot_sh:.1f} %, out of fold; "
          f"{bs_is:+.2f} s / {100*tot_bs_is/tot_sh:.1f} % if alpha is chosen in sample)")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 4, int(a[1]) if len(a) > 1 else 300,
         int(a[2]) if len(a) > 2 else 2, a[3] if len(a) > 3 else "stack5",
         a[4] if len(a) > 4 else "full", float(a[5]) if len(a) > 5 else WIN)
