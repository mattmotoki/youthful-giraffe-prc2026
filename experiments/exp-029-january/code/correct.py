"""The additive January correction: three shapes, fitted out of sample, measured.

The served object is a function of a **served vector** and a label-free
exceedance, so it composes on top of stack-5 today and on top of the window arm
later:

    corrected = max(pred + c(exceedance), FLOOR)          c(x) = 0 for x <= 0

`c` is fitted on the 2025 airport-days whose daily mean of
`MVT_TIME_UTC_mvt - AOBT_3_flt` clears that airport's training maximum.  Two
honesty settings are reported for every shape:

  loo    leave-one-DAY-out, the assignment's instruction.  A january day is
         corrected by a rule that saw the other january day, so a january
         number carries a january-label channel of its own.
  lofo   leave-one-FOLD-out: the rule applied to a fold is fitted only on the
         other two folds' days.  This is the instrument the gate is read on.

`gain` is positive when the correction REDUCES RMSE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import envelope as E  # noqa: E402
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
import frames as F26  # noqa: E402

WINSOR = (0.025, 0.975)
#: the arm whose corrected vector is written to `oof/`; see RESULTS.md §4.
SERVED = "step|lofo|winsor"


# ------------------------------------------------------------ the shapes ----
REGIMES = ("nm_bulk", "nm_late", "nonm", "noac")


def regime_targets(g: pd.DataFrame, winsor: bool) -> pd.DataFrame:
    """Per (flagged day, blend regime): rows and the SSE-optimal constant.

    RESULTS.md section 5: 92 % of the rows on the five envelope days are
    `nm_bulk`/`nm_late` and stack-5 is nearly unbiased on them, while 19 `noac`
    rows on one day carry 48 % of those days' squared error.  These two arms
    ask whether the correction belongs on the regime rather than on the day.
    """
    r = g["y_true"].to_numpy("float64") - g["y_pred"].to_numpy("float64")
    if winsor:
        lo, hi = np.quantile(r, WINSOR)
        r = np.clip(r, lo, hi)
    d = pd.DataFrame({"fold": g["fold"].to_numpy(), "apt": g["apt"].to_numpy(),
                      "day": g["day"].to_numpy(), "breg": g["breg"].to_numpy(), "r": r})
    return (d.groupby(["fold", "apt", "day", "breg"], observed=True)
            .agg(n=("r", "size"), c_opt=("r", "mean")).reset_index())


def fit_regime(t: pd.DataFrame) -> dict:
    """Shape D: one constant per blend regime over the flagged days."""
    return {g: float((s["c_opt"] * s["n"]).sum() / s["n"].sum())
            for g, s in t.groupby("breg", observed=True)}


def apply_regime(p: dict, x: np.ndarray, breg: np.ndarray) -> np.ndarray:
    c = np.zeros(len(x))
    for g, v in p.items():
        c[breg == g] = v
    return np.where(x <= 0.0, 0.0, c)


def day_targets(g: pd.DataFrame, winsor: bool) -> pd.DataFrame:
    """Per flagged day: rows, exceedance, and the SSE-optimal additive constant.

    The optimal constant under squared loss is the mean of `y_true - pred`.
    Winsorising the row residual first trades that optimality for stability;
    one 2025 day (LIRF 2025-07-28) has an RMSE of 1,152 s and would otherwise
    set the fit on its own.
    """
    r = g["y_true"].to_numpy("float64") - g["y_pred"].to_numpy("float64")
    if winsor:
        lo, hi = np.quantile(r, WINSOR)
        r = np.clip(r, lo, hi)
    d = pd.DataFrame({"fold": g["fold"].to_numpy(), "apt": g["apt"].to_numpy(),
                      "day": g["day"].to_numpy(),
                      "exceedance": g["exceedance"].to_numpy("float64"), "r": r})
    t = d.groupby(["fold", "apt", "day"], observed=True).agg(
        n=("r", "size"), exceedance=("exceedance", "first"), c_opt=("r", "mean")).reset_index()
    return t


def fit_step(t: pd.DataFrame, clear: float = E.CLEAR, shrink: float = 1.0) -> dict:
    """Shape A: one constant per exceedance band, 0-`clear` and past it.

    `shrink` multiplies both constants.  Five days is a sample size at which the
    unshrunk mean is the wrong estimator, and shrinking toward zero is the
    cheapest defence; `shrink` is a declared setting, not a fitted one, so the
    reported number stays out of sample.
    """
    out = {}
    for lab, m in (("lo", t["exceedance"] <= clear), ("hi", t["exceedance"] > clear)):
        s = t[m]
        out[lab] = (shrink * float((s["c_opt"] * s["n"]).sum() / s["n"].sum())
                    if len(s) else 0.0)
    return out


def apply_step(p: dict, x: np.ndarray, clear: float = E.CLEAR) -> np.ndarray:
    return np.where(x <= 0.0, 0.0, np.where(x <= clear, p["lo"], p["hi"]))


def fit_linear(t: pd.DataFrame, cap: bool) -> dict:
    """Shape B: c(x) = a + b*x, row-weighted least squares on the flagged days."""
    X = np.stack([np.ones(len(t)), t["exceedance"].to_numpy("float64")], axis=1)
    w = t["n"].to_numpy("float64")
    A = X * w[:, None]
    coef, *_ = np.linalg.lstsq(A.T @ X, A.T @ t["c_opt"].to_numpy("float64"), rcond=None)
    lo, hi = (float(t["c_opt"].min()), float(t["c_opt"].max())) if cap else (-np.inf, np.inf)
    return {"a": float(coef[0]), "b": float(coef[1]), "lo": lo, "hi": hi}


def apply_linear(p: dict, x: np.ndarray) -> np.ndarray:
    c = np.clip(p["a"] + p["b"] * x, p["lo"], p["hi"])
    return np.where(x <= 0.0, 0.0, c)


def fit_stress(t: pd.DataFrame, restrict: bool) -> dict:
    """Shape C (mine): a line in the scale-free day stress, fitted on ALL days.

    Shapes A and B see five airport-days.  This one sees every airport-day in
    the fitting folds and asks the same question of all of them, so its slope
    has a standard error rather than a shrug.  Stress is

        s = (day_span_mean - p95_train[apt]) / (p95_train[apt] - p50_train[apt])

    which is 0 at the airport's 95th training percentile and 1 one
    inter-percentile step above it, so airports with different traffic mixes are
    on one axis.  `restrict=True` serves it only where the max envelope is
    already cleared, which is the assignment's "must not harm in-envelope rows"
    by construction; `restrict=False` serves it everywhere and is reported as
    the variant that tests whether the restriction is costing anything.
    """
    X = np.stack([np.ones(len(t)), t["stress"].to_numpy("float64")], axis=1)
    w = t["n"].to_numpy("float64")
    A = X * w[:, None]
    coef, *_ = np.linalg.lstsq(A.T @ X, A.T @ t["c_opt"].to_numpy("float64"), rcond=None)
    return {"a": float(coef[0]), "b": float(coef[1]), "restrict": bool(restrict)}


def apply_stress(p: dict, stress: np.ndarray, exceedance: np.ndarray) -> np.ndarray:
    c = p["a"] + p["b"] * stress
    return np.where(exceedance <= 0.0, 0.0, c) if p["restrict"] else c


# ------------------------------------------------------- clustered bootstrap -
def clusters(m: pd.DataFrame) -> np.ndarray:
    """exp-026 `stack5.clusters`, from the calendar day this experiment already has."""
    return pd.factorize(m["apt"].astype(str) + "|"
                        + m["day"].dt.strftime("%Y-%m-%d"))[0]


def boot_gain(y, p_new, p_ref, cl, ids, fold, draws=2000, seed=7):
    """Day-clustered 95 % interval on (ref RMSE - new RMSE), the clean rows only.

    exp-026 `stack5.boot_delta` with the sign flipped so a positive number is a
    gain, and with the unreached-rollover rows removed first so the interval is
    around the same quantity `rmse_clean` reports.
    """
    ex = D.excluded_ids(fold)
    keep = ~np.isin(np.asarray(ids, "float64"), list(ex)) if ex else np.ones(len(y), bool)
    y, cl = np.asarray(y, "float64")[keep], np.asarray(cl)[keep]
    e_new = (np.asarray(p_new, "float64")[keep] - y) ** 2
    e_ref = (np.asarray(p_ref, "float64")[keep] - y) ** 2
    cl = pd.factorize(cl)[0]
    k = cl.max() + 1
    order = np.argsort(cl, kind="stable")
    starts = np.searchsorted(cl[order], np.arange(k))
    ends = np.searchsorted(cl[order], np.arange(k), side="right")
    sn = np.add.reduceat(e_new[order], starts)
    sr = np.add.reduceat(e_ref[order], starts)
    cnt = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = W @ cnt
    out = np.sqrt((W @ sr) / n) - np.sqrt((W @ sn) / n)
    return (float(np.sqrt(e_ref.mean()) - np.sqrt(e_new.mean())),
            float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


# ------------------------------------------------------------------- stress --
def add_stress(R: pd.DataFrame) -> pd.DataFrame:
    """The scale-free day stress shape C reads, from the training months alone."""
    t25 = E.days_2025()
    out = []
    for f in E.FOLDS:
        tr = t25[t25["month"].isin(list(D.FOLDS[f].train_months))]
        q = tr.groupby("apt")["day_span_mean"].agg(
            p95=lambda s: s.quantile(0.95), p50="median")
        g = R[R["fold"] == f].merge(q, left_on="apt", right_index=True, how="left")
        g["stress"] = (g["day_span_mean"] - g["p95"]) / (g["p95"] - g["p50"])
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------- run --
def main() -> None:
    R = add_stress(pd.read_parquet(EXP / "analysis" / "fold_rows.parquet"))
    R["day"] = pd.to_datetime(R["day"])
    base = "y_pred"                                  # stack-5 keep-noac
    flag = R["flagged"].to_numpy()

    # Every target -- including the winsorisation bounds -- is computed from
    # the FIT rows of the unit being held out, never from all five days. The
    # bounds move 130-180 s depending on which fold is dropped, so computing
    # them once over everything would have let the scored fold set the clip.
    arms = [(shape, hold, winsor)
            for winsor in (False, True)
            for shape in ("step", "step-half", "step-winter", "step-bulk",
                          "step-regime", "linear", "linear-capped", "stress",
                          "stress-all")
            for hold in ("loo", "lofo")]
    all_days = R[flag][["fold", "apt", "day"]].drop_duplicates()

    def with_stress(t: pd.DataFrame) -> pd.DataFrame:
        return t.merge(R.groupby(["fold", "apt", "day"], observed=True)["stress"]
                       .first().rename("stress"), on=["fold", "apt", "day"])

    recs, preds, params = [], {}, {}
    x = R["exceedance"].to_numpy("float64")
    sarr = R["stress"].to_numpy("float64")
    breg = R["breg"].to_numpy()
    bulk = breg == "nm_bulk"
    for shape, hold, winsor in arms:
        name = f"{shape}|{hold}|{'winsor' if winsor else 'raw'}"
        c = np.zeros(len(R))
        pars = {}
        units = (list(all_days.itertuples(index=False, name=None)) if hold == "loo"
                 else [(f, None, None) for f in E.FOLDS])
        for u in units:
            uf, ua, ud = u
            if hold == "loo":
                held = ((R["fold"] == uf) & (R["apt"] == ua)
                        & (R["day"] == ud)).to_numpy()
            else:
                held = (R["fold"] == uf).to_numpy()
            fit = ~held
            T5 = day_targets(R[flag & fit], winsor)
            tgt = held
            if shape == "step":
                p = fit_step(T5); v = apply_step(p, x)
            elif shape == "step-half":
                p = fit_step(T5, shrink=0.5); v = apply_step(p, x)
            elif shape == "step-winter":
                # repair attempt: the two days that disagree most are a summer
                # day and a winter day, and January is winter, so fit on the
                # winter analogue days alone.
                w = flag & fit & R["fold"].isin(["january", "december"]).to_numpy()
                p = fit_step(day_targets(R[w], winsor)); v = apply_step(p, x)
            elif shape == "step-bulk":
                # the band step fitted on, and served to, `nm_bulk` rows only:
                # 77 % of the flagged rows and the only slice with enough of
                # them for a mean to mean anything.
                p = fit_step(day_targets(R[flag & fit & bulk], winsor))
                v = np.where(bulk, apply_step(p, x), 0.0)
            elif shape == "step-regime":
                p = fit_regime(regime_targets(R[flag & fit], winsor))
                v = apply_regime(p, x, breg)
            elif shape == "linear":
                p = fit_linear(T5, cap=False); v = apply_linear(p, x)
            elif shape == "linear-capped":
                p = fit_linear(T5, cap=True); v = apply_linear(p, x)
            elif shape == "stress":
                p = fit_stress(with_stress(day_targets(R[fit], winsor)), restrict=True)
                v = apply_stress(p, sarr, x)
            else:
                p = fit_stress(with_stress(day_targets(R[fit], winsor)), restrict=False)
                v = apply_stress(p, sarr, x)
            c[tgt] = v[tgt]
            pars[str(u)] = p
        params[name] = pars
        pred = np.maximum(R[base].to_numpy("float64") + c, B.FLOOR)
        preds[name] = pred
        R[f"c::{name}"] = c
        for f in E.FOLDS:
            m = (R["fold"] == f).to_numpy()
            y = R.loc[m, "y_true"].to_numpy("float64")
            ids = R.loc[m, "MVT_ID_mvt"].to_numpy("float64")
            ref = R.loc[m, base].to_numpy("float64")
            cl = clusters(R[m])
            g, lo, hi = boot_gain(y, pred[m], ref, cl, ids, f)
            inenv = ~R.loc[m, "flagged"].to_numpy()
            e0 = ref[inenv] - y[inenv]
            e1 = pred[m][inenv] - y[inenv]
            recs.append(dict(arm=name, shape=shape, hold=hold, winsor=winsor, fold=f,
                             base_clean=D.rmse_clean(y, ref, ids, f),
                             new_clean=D.rmse_clean(y, pred[m], ids, f),
                             gain=g, lo=lo, hi=hi,
                             n_moved=int((c[m] != 0).sum()),
                             mean_c=float(c[m][c[m] != 0].mean()) if (c[m] != 0).any() else 0.0,
                             inenv_rmse_base=float(np.sqrt((e0 ** 2).mean())),
                             inenv_rmse_new=float(np.sqrt((e1 ** 2).mean()))))
    T = pd.DataFrame(recs)
    T["board_gain"] = np.nan
    for arm, g in T.groupby("arm"):
        b0 = F26.board_estimate({f: g.loc[g.fold == f, "base_clean"].iloc[0] for f in F26.SHARES})
        b1 = F26.board_estimate({f: g.loc[g.fold == f, "new_clean"].iloc[0] for f in F26.SHARES})
        T.loc[T.arm == arm, "board_gain"] = b0 - b1
    T.to_csv(EXP / "analysis" / "arms.csv", index=False)
    (EXP / "analysis" / "arm_params.json").write_text(json.dumps(params, indent=1, default=str))

    # `stress-all` under leave-one-DAY-out can only move the five held-out days,
    # so it collapses onto `stress`; it is reported only under `lofo`.
    T = T[~((T["shape"] == "stress-all") & (T["hold"] == "loo"))].copy()
    piv = T.pivot_table(index=["shape", "hold", "winsor"], columns="fold",
                        values="gain")[list(E.FOLDS)]
    piv["board"] = T.groupby(["shape", "hold", "winsor"])["board_gain"].first()
    print("=== gain in s (positive = RMSE down), clean, stack-5 keep-noac as the base ===")
    print(piv.round(3).to_string())
    print("\n=== january, with the day-clustered 95 % interval ===")
    for r in T[T.fold == "january"].itertuples():
        print(f"  {r.arm:34s} {r.gain:+7.3f} [{r.lo:+7.3f}, {r.hi:+7.3f}]  "
              f"moved {r.n_moved:5d} rows, mean c {r.mean_c:+8.1f} s, "
              f"in-envelope RMSE {r.inenv_rmse_base:.3f} -> {r.inenv_rmse_new:.3f}")

    # ---- the ceiling of the whole lever on this instrument -------------------
    print("\n=== oracle ceilings on the fold instrument (in-sample, not servable) ===")
    ce = []
    for f in E.FOLDS:
        m = (R["fold"] == f).to_numpy()
        g = R[m]
        y = g["y_true"].to_numpy("float64")
        ids = g["MVT_ID_mvt"].to_numpy("float64")
        ref = g[base].to_numpy("float64")
        b = D.rmse_clean(y, ref, ids, f)
        for lab, mask in (("flagged rows", g["flagged"].to_numpy()),
                          ("flagged rows, >30 s", (g["exceedance"] > E.CLEAR).to_numpy())):
            # best additive constant per flagged DAY, fitted on the day itself
            c = np.zeros(len(g))
            for (a, d), s in g[mask].groupby(["apt", "day"], observed=True):
                idx = ((g["apt"] == a) & (g["day"] == d) & mask).to_numpy()
                c[idx] = float((s["y_true"] - s[base]).mean())
            p1 = np.maximum(ref + c, B.FLOOR)
            # and the absolute bound: those rows predicted perfectly
            p2 = ref.copy(); p2[mask] = y[mask]
            ce.append(dict(fold=f, slice=lab, n=int(mask.sum()), base_clean=b,
                           oracle_day_constant=b - D.rmse_clean(y, p1, ids, f),
                           oracle_perfect=b - D.rmse_clean(y, p2, ids, f)))
    C = pd.DataFrame(ce)
    C.to_csv(EXP / "analysis" / "ceilings.csv", index=False)
    print(C.round(3).to_string(index=False))

    R[["MVT_ID_mvt", "fold", "apt", "day", "y_true", "y_pred", "exceedance",
       "flagged", "stress"] + [c for c in R.columns if c.startswith("c::")]
      ].to_parquet(EXP / "analysis" / "corrections.parquet", index=False)

    # ---- the corrected out-of-fold files, COMMON.md's columns ---------------
    # `y_pred` is the corrected vector; `y_pred_base` is stack-5 keep-noac, so
    # the file is self-checking and a later arm can be re-scored against it.
    for f in E.FOLDS:
        m = (R["fold"] == f).to_numpy()
        pd.DataFrame({
            "MVT_ID_mvt": R.loc[m, "MVT_ID_mvt"].to_numpy("float64"),
            "fold": f, "y_true": R.loc[m, "y_true"].to_numpy("float64"),
            "y_pred": preds[SERVED][m],
            "y_pred_base": R.loc[m, base].to_numpy("float64"),
            "correction": R.loc[m, f"c::{SERVED}"].to_numpy("float64"),
            "exceedance": R.loc[m, "exceedance"].to_numpy("float64"),
        }).to_parquet(EXP / "oof" / f"exp-029-{f}.parquet", index=False)
    print(f"\n[oof] wrote exp-029-{{fold}}.parquet for the {SERVED} arm")


if __name__ == "__main__":
    main()
