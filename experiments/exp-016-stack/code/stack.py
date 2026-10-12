"""exp-016: a learned stacker that decides the mixing per ROW.

The blend owner measured that flat weights beat a fitted global vector and that
per-regime weights add little. Both of those are *constant* weights over a
population. This asks whether the weights should be a function of the row.

Three variants, one control, two baselines, all under the same rotation: fit on
two folds' out-of-fold predictions, score the third, never on its own.

  a1  LightGBM on [member predictions + gating features] -> y, monotone +1 on
      every member prediction, heavy regularisation.
  a2  the same learner on the RESIDUAL from the flat average, no monotone
      constraint (it does not transfer to a residual target).
  b   ridge on member predictions with feature-gated weights:
      y = sum_j (w_j0 + sum_k beta_jk g_k(x)) * p_j, no intercept.
  c   per-regime NNLS -- the control, and what the standing blend already does.

  flat4     equal weights on the four general members.
  blend2    the shipped rule: flat4 outside `nm_late`, fitted five inside.

Every comparison is a day-clustered bootstrap of the RMSE difference, because
the errors are correlated within an airport-day and a row bootstrap would
understate the interval by a lot.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = M.EXP / "analysis"
THREADS = int(os.environ.get("PRC_THREADS", "4"))  # desk compute cap, docs/COMPUTE.md compute allocation
GENERAL = ("T-tailv2", "C2-cat", "L2-hub", "D1-winter")   # total members
SEED = 7


# --------------------------------------------------------------- helpers ----
def clusters(m: pd.DataFrame) -> np.ndarray:
    day = np.floor(m["t"].to_numpy("float64") / 86400.0).astype("int64")
    return pd.factorize(pd.Series(m["airport"].astype(str)) + "|" + pd.Series(day).astype(str))[0]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, "float64") - np.asarray(p, "float64")) ** 2)))


def design(m: pd.DataFrame, names) -> tuple[np.ndarray, np.ndarray]:
    """(P, defined) -- member predictions and where each is defined."""
    P = np.stack([m[n].to_numpy("float64") for n in names], axis=1)
    return P, np.isfinite(P)


def flat4(m: pd.DataFrame) -> np.ndarray:
    P, _ = design(m, GENERAL)
    return np.maximum(P.mean(axis=1), B.FLOOR)


def fit_blend2(fit: pd.DataFrame, ap: pd.DataFrame) -> np.ndarray:
    """The shipped rule: flat over the four general members outside `nm_late`,
    NNLS over all five inside it."""
    names5 = list(M.NAMES)
    Pf, okf = design(fit, names5)
    sel = (fit["regime"].to_numpy() == "nm_late") & okf[:, names5.index("N1-nmtail")]
    w5 = B.fit_weights(Pf[sel], fit["y_true"].to_numpy("float64")[sel], "nnls")
    Pa, oka = design(ap, names5)
    selA = (ap["regime"].to_numpy() == "nm_late") & oka[:, names5.index("N1-nmtail")]
    out = np.empty(len(ap))
    out[selA] = Pa[selA] @ w5
    idx4 = [names5.index(n) for n in GENERAL]
    out[~selA] = Pa[~selA][:, idx4] @ np.full(len(idx4), 1.0 / len(idx4))
    return np.maximum(out, B.FLOOR)


def fit_regime_nnls(fit: pd.DataFrame, ap: pd.DataFrame) -> np.ndarray:
    """Variant c: per-regime NNLS over the four total members (the control)."""
    Pf, _ = design(fit, GENERAL)
    Pa, _ = design(ap, GENERAL)
    W = B.fit_per_regime(Pf, fit["y_true"].to_numpy("float64"),
                         fit["regime"].to_numpy(), "nnls", min_rows=50)
    return np.maximum(B.apply_weights(Pa, ap["regime"].to_numpy(), W), B.FLOOR)


# ------------------------------------------------------------ variant a ----
LGB_BASE = dict(objective="regression", metric="rmse", learning_rate=0.03,
                num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.7,
                bagging_fraction=0.7, bagging_freq=1, lambda_l2=100.0,
                num_threads=THREADS, verbosity=-1, force_row_wise=True,
                deterministic=True, seed=SEED, bagging_seed=SEED + 1,
                feature_fraction_seed=SEED + 2, data_random_seed=SEED + 3)
LGB_ROUNDS = 300


def stack_features(m: pd.DataFrame, names) -> pd.DataFrame:
    X = pd.DataFrame({n: m[n].to_numpy("float64") for n in names})
    P, ok = design(m, names)
    with np.errstate(invalid="ignore"):
        X["mem_mean"] = np.nanmean(P, axis=1)
        X["mem_std"] = np.nanstd(P, axis=1)
        X["mem_range"] = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
    X["n_defined"] = ok.sum(axis=1).astype("float64")
    for c in M.GATE_NUM:
        X[c] = m[c].to_numpy("float64")
    X["ds_deice_hr"] = m["ds_deice_hr"].to_numpy("float64")
    X["noac"] = m["noac"].to_numpy("float64")
    X["nm_missing"] = m["nm_missing"].to_numpy("float64")
    X["airport"] = pd.Categorical(m["airport"].astype(str)).codes.astype("int32")
    X["regime"] = pd.Categorical(m["regime"].astype(str),
                                 categories=list(B.REGIMES)).codes.astype("int32")
    return X


def fit_lgb(fit: pd.DataFrame, ap: pd.DataFrame, residual: bool, monotone: bool,
            names=M.NAMES, rounds=LGB_ROUNDS) -> np.ndarray:
    import lightgbm as lgb

    Xf, Xa = stack_features(fit, names), stack_features(ap, names)
    yf = fit["y_true"].to_numpy("float64")
    base_f, base_a = (flat4(fit), flat4(ap)) if residual else (0.0, 0.0)
    p = dict(LGB_BASE)
    if monotone:
        p["monotone_constraints"] = [1 if c in names else 0 for c in Xf.columns]
        p["monotone_constraints_method"] = "advanced"
    ds = lgb.Dataset(Xf, label=yf - base_f, categorical_feature=["airport", "regime"])
    mdl = lgb.train(p, ds, num_boost_round=rounds)
    return np.maximum(mdl.predict(Xa) + base_a, B.FLOOR), mdl, list(Xf.columns)


# ------------------------------------------------------------ variant b ----
#: Gates the member weights are allowed to vary with. Kept small on purpose:
#: two folds of OOF is the whole fitting set, and every gate multiplies the
#: parameter count by the member count.
def gate_matrix(m: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    reg = m["regime"].to_numpy()
    d = m["d_aobt_lobt"].to_numpy("float64")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    cols, names = [np.ones(len(m))], ["1"]
    for r in B.REGIMES[1:]:                      # nm_bulk is the reference level
        cols.append((reg == r).astype("float64"))
        names.append(f"regime={r}")
    for lo, hi, lab in ((-np.inf, 0, "aobt_lobt<0"), (1800, np.inf, "aobt_lobt>1800")):
        cols.append((np.isfinite(d) & (d >= lo) & (d < hi)).astype("float64"))
        names.append(lab)
    for lo, hi, lab in ((-np.inf, 600, "mvt_aobt<600"), (1800, np.inf, "mvt_aobt>1800")):
        cols.append((np.isfinite(a) & (a >= lo) & (a < hi)).astype("float64"))
        names.append(lab)
    cols.append((m["ds_deice_hr"].to_numpy("float64") > 0.2).astype("float64"))
    names.append("deice_hr>0.2")
    return np.stack(cols, axis=1), names


def fit_ridge_gated(fit: pd.DataFrame, ap: pd.DataFrame, names=M.NAMES,
                    ridge: float = 1e-3) -> np.ndarray:
    def build(m):
        P, ok = design(m, names)
        P = np.where(ok, P, 0.0)
        G, gn = gate_matrix(m)
        cols = [P[:, j:j + 1] * G for j in range(P.shape[1])]
        return np.concatenate(cols, axis=1), [f"{n}:{g}" for n in names for g in gn]

    A, cn = build(fit)
    y = fit["y_true"].to_numpy("float64")
    Gm = A.T @ A
    Gm = Gm + ridge * (np.trace(Gm) / max(Gm.shape[0], 1) + 1.0) * np.eye(Gm.shape[0])
    w = np.linalg.solve(Gm, A.T @ y)
    Aa, _ = build(ap)
    return np.maximum(Aa @ w, B.FLOOR), dict(zip(cn, map(float, w)))


# -------------------------------------------------------------- bootstrap ----
def boot_delta(y, p_new, p_ref, cl, draws=2000, seed=SEED) -> tuple[float, float, float]:
    """RMSE(new) - RMSE(ref), day-clustered. Negative means `new` is better."""
    y = np.asarray(y, "float64")
    e_new = (np.asarray(p_new, "float64") - y) ** 2
    e_ref = (np.asarray(p_ref, "float64") - y) ** 2
    k = cl.max() + 1
    order = np.argsort(cl, kind="stable")
    starts = np.searchsorted(cl[order], np.arange(k))
    ends = np.searchsorted(cl[order], np.arange(k), side="right")
    sn = np.add.reduceat(e_new[order], starts) if k else np.array([])
    sr = np.add.reduceat(e_ref[order], starts)
    cnt = (ends - starts).astype("float64")
    rng = np.random.default_rng(seed)
    # Vectorised over draws: a per-draw multinomial count vector times the three
    # cluster sums, rather than 2,000 fancy-indexed reductions. Same estimator,
    # and it matters at a 4-thread cap.
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = W @ cnt
    out = np.sqrt((W @ sn) / n) - np.sqrt((W @ sr) / n)
    point = np.sqrt(e_new.mean()) - np.sqrt(e_ref.mean())
    return point, float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


# ------------------------------------------------------------------ main ----
def rotate(variants=("flat4", "blend2", "c-regime-nnls", "a1-lgb-mono",
                     "a2-lgb-resid", "b-ridge-gated")) -> pd.DataFrame:
    moved = M.verify()
    if moved:
        raise SystemExit(f"member bytes moved since the pin: {moved}")
    data = {f: M.fold_frame(f) for f in M.FOLDS}
    rows, preds = [], {}
    for f in M.FOLDS:
        t0 = time.time()
        ap = data[f]
        fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
        y = ap["y_true"].to_numpy("float64")
        cl = clusters(ap)
        got = {}
        got["flat4"] = flat4(ap)
        got["blend2"] = fit_blend2(fit, ap)
        got["c-regime-nnls"] = fit_regime_nnls(fit, ap)
        if "a1-lgb-mono" in variants:
            got["a1-lgb-mono"] = fit_lgb(fit, ap, residual=False, monotone=True)[0]
        if "a2-lgb-resid" in variants:
            got["a2-lgb-resid"] = fit_lgb(fit, ap, residual=True, monotone=False)[0]
        if "b-ridge-gated" in variants:
            got["b-ridge-gated"] = fit_ridge_gated(fit, ap)[0]
        preds[f] = got
        for v, p in got.items():
            d2, lo2, hi2 = boot_delta(y, p, got["blend2"], cl)
            d4, lo4, hi4 = boot_delta(y, p, got["flat4"], cl)
            rows.append(dict(fold=f, variant=v, n=len(y), rmse=rmse(y, p),
                             vs_blend2=d2, b2_lo=lo2, b2_hi=hi2,
                             vs_flat4=d4, f4_lo=lo4, f4_hi=hi4))
        print(f"[{f}] {time.time()-t0:.0f}s  " +
              "  ".join(f"{v}={rmse(y, p):.2f}" for v, p in got.items()), flush=True)
    T = pd.DataFrame(rows)
    be = {v: M.board_estimate({f: T[(T.fold == f) & (T.variant == v)].rmse.iloc[0]
                               for f in M.SHARES}) for v in T.variant.unique()}
    T["board"] = T["variant"].map(be)
    return T, preds


if __name__ == "__main__":
    M.config_from_argv()
    T, preds = rotate()
    T.to_csv(OUT / "rotation.csv", index=False)
    piv = T.pivot(index="variant", columns="fold", values="rmse")[list(M.FOLDS)]
    piv["board"] = T.groupby("variant")["board"].first()
    print("\n=== rotated RMSE ===")
    print(piv.sort_values("board").round(3).to_string())
    print("\n=== against the shipped blend-2 rule (negative = better), day-clustered ===")
    print(T.pivot(index="variant", columns="fold", values="vs_blend2").round(3).to_string())
    print("\n=== 95 % intervals, vs blend2 ===")
    for r in T.itertuples():
        print(f"  {r.fold:9s} {r.variant:15s} {r.vs_blend2:+7.3f} [{r.b2_lo:+7.3f}, {r.b2_hi:+7.3f}]"
              f"   vs flat4 {r.vs_flat4:+7.3f} [{r.f4_lo:+7.3f}, {r.f4_hi:+7.3f}]")
