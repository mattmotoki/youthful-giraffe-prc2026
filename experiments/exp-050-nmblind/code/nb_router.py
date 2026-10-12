"""exp-050-nmblind: the registered router, its controls, the oracle and the repair.

Everything here is RESULTS §1's registration, executed.  Per fold:

    r   = y - p_v22                     p_v22 = the frozen fold side of v22 (nb_panel)
    d   = p_blind - p_aware              on the population P, 0 elsewhere
    p   = max(60, p_v22 + c*d)          c in [0, 1]; a row with c*d = 0 is p_v22's bytes
    u   = p - p_v22                      the update that is priced

Coefficients for the rows of date block b are fitted on P's rows in the OTHER four
blocks of the same month (development blocks), never on b.  Every variant uses that one
cross-fit:

    zero        c = 0
    global      c_g = clip(sum(r d) / sum(d d), 0, 1) over the development rows
    router      four clock-disagreement cells; raw LS coefficient per cell, shrunk to c_g
                by w = n_dev / (n_dev + N0), then clipped to [0, 1]
    difficulty  the same, cells = quartiles of p_blind (cut points from development rows)
    placebo     the same as router, cell labels permuted within airport x day (seed 50)
    repair      §1.9's support topology: c_g outside the disagreement support, one
                bounded coefficient on the support only if its benefit sign is stable
                across the four inner development blocks, c_g there otherwise

The oracle is per row over the closed interval: argmin over c in [0, 1] of
(y - max(60, p_v22 + c d))^2, checked against a 101-point grid.

    .venv/bin/python experiments/exp-050-nmblind/code/nb_router.py july_loo [december january]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import nb_experts as NE                                               # noqa: E402
import nb_panel as NP                                                 # noqa: E402
import prc.data as D                                                  # noqa: E402

FOLDS = ["july_loo", "december", "january"]
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
K_BLOCKS = 5
N0 = 1000.0
BOOT = 2000
SEED = 50
DIS_SCHED_S = 900.0
DIS_LOBT_S = 300.0
MIN_PRED = 60.0
VARIANTS = ["global", "router", "difficulty", "placebo", "repair"]


def sha_arr(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes())
    return h.hexdigest()


def dayblocks(days: np.ndarray, k: int = K_BLOCKS) -> np.ndarray:
    """`k` contiguous calendar-day blocks of the month, shared by every airport."""
    u = np.sort(np.unique(days))
    code = {d: min(int(i * k / len(u)), k - 1) for i, d in enumerate(u)}
    return np.array([code[x] for x in days], dtype=int)


def load_fold(fold: str) -> pd.DataFrame:
    pan = NP.load(fold)
    ex = pd.read_parquet(EXP / f"oof/experts-{fold}.parquet")
    rec = json.loads((EXP / f"analysis/experts-{fold}.json").read_text())
    if NE.sha_file(EXP / f"oof/experts-{fold}.parquet") != rec["expert_oof_sha256"]:
        raise SystemExit(f"oof/experts-{fold}.parquet moved since it was written")
    if not rec["blindness_proof"]["passes"]:
        raise SystemExit(f"{fold}: the blindness proof did not pass; no score may be read")
    fe = NE.load_month(D.FOLDS[fold].valid_month)[["MVT_ID_mvt", "aobt_sched", "aobt_lobt"]]
    df = pan.merge(ex.drop(columns="fold"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    df = df.merge(fe, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if not np.array_equal(df.MVT_ID_mvt.to_numpy(), pan.MVT_ID_mvt.to_numpy()):
        raise SystemExit(f"{fold}: a join reordered the panel")
    df["pop"] = (df.regime.astype(str).to_numpy() == "nm_bulk") & ~df.guard_held.to_numpy(bool)
    if df.loc[df["pop"], ["p_blind", "p_aware"]].isna().any().any():
        raise SystemExit(f"{fold}: a population row has no expert prediction")
    df["d"] = np.where(df["pop"], df.p_blind - df.p_aware, 0.0)
    a_s = np.abs(df.aobt_sched.to_numpy("float64"))
    a_l = np.abs(df.aobt_lobt.to_numpy("float64"))
    df["dis_sched"] = np.where(np.isfinite(a_s), a_s > DIS_SCHED_S, False)
    df["dis_lobt"] = np.where(np.isfinite(a_l), a_l > DIS_LOBT_S, False)
    df["cell"] = 2 * df.dis_sched.astype(int) + df.dis_lobt.astype(int)
    df["block"] = dayblocks(df.day.to_numpy())
    df["r"] = df.y_true.to_numpy("float64") - df.p_v22.to_numpy("float64")
    return df


def _ls(r: np.ndarray, d: np.ndarray) -> float:
    b = float(np.dot(d, d))
    return float(np.dot(r, d)) / b if b > 0 else float("nan")


def _global(r, d, dev) -> float:
    c = _ls(r[dev], d[dev])
    return float(np.clip(c, 0.0, 1.0)) if np.isfinite(c) else 0.0


def _cells(r, d, cell, dev, cg, ncell=4) -> tuple[np.ndarray, list]:
    c = np.empty(ncell)
    info = []
    for k in range(ncell):
        m = dev & (cell == k)
        n = int(m.sum())
        raw = _ls(r[m], d[m])
        raw_used = raw if np.isfinite(raw) else cg
        w = n / (n + N0)
        c[k] = float(np.clip(w * raw_used + (1.0 - w) * cg, 0.0, 1.0))
        info.append({"cell": k, "dev_rows": n, "raw": raw, "weight": w, "coef": c[k]})
    return c, info


def _repair(r, d, cell, block, dev, cg) -> tuple[float | None, dict]:
    """§1.9: one coefficient on the disagreement support S = cells {1,2,3}, used only if
    its benefit over c_g is positive on EACH of the four inner development blocks, each
    inner block read with coefficients fitted on the other three (an inner cross-fit)."""
    S = cell > 0
    m = dev & S
    cs = _ls(r[m], d[m])
    if not np.isfinite(cs):
        return None, {"support_dev_rows": int(m.sum()), "coef": None, "stable": False}
    cs = float(np.clip(cs, 0.0, 1.0))
    inner = sorted(set(block[dev].tolist()))
    gains = []
    for j in inner:
        fit_j = dev & (block != j)
        cg_j = _global(r, d, fit_j)
        cs_j = _ls(r[fit_j & S], d[fit_j & S])
        cs_j = float(np.clip(cs_j, 0.0, 1.0)) if np.isfinite(cs_j) else cg_j
        mj = m & (block == j)
        gains.append(float(np.sum((r[mj] - cg_j * d[mj]) ** 2 - (r[mj] - cs_j * d[mj]) ** 2)))
    stable = bool(len(gains) == len(inner) and all(g > 0 for g in gains))
    return (cs if stable else None), {"support_dev_rows": int(m.sum()), "coef": cs,
                                      "inner_block_gain_s2": gains, "stable": stable}


def permuted_d(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """The noise-matched reference: d permuted among population rows inside airport x day,
    so its distribution is kept and its row-level alignment with r is destroyed."""
    pop = df["pop"].to_numpy(bool)
    d = df.d.to_numpy("float64")
    out = d.copy()
    key = (df.airport.astype(str) + "|" + df.day.astype(str)).to_numpy()
    for idx in pd.Series(np.arange(len(df)))[pop].groupby(key[pop]).indices.values():
        rows = np.flatnonzero(pop)[idx]
        out[rows] = d[rows][rng.permutation(len(rows))]
    return out


def crossfit(df: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, dict]:
    """Per-row coefficient for every variant, fitted away from each scoring block."""
    pop = df["pop"].to_numpy(bool)
    r, d = df.r.to_numpy("float64"), df.d.to_numpy("float64")
    cell = df.cell.to_numpy(int)
    block = df.block.to_numpy(int)
    pb = df.p_blind.to_numpy("float64")
    # placebo cells: the router's labels permuted inside airport x day, population rows only
    pl = cell.copy()
    key = (df.airport.astype(str) + "|" + df.day.astype(str)).to_numpy()
    for idx in pd.Series(np.arange(len(df)))[pop].groupby(key[pop]).indices.values():
        rows = np.flatnonzero(pop)[idx]
        pl[rows] = cell[rows][rng.permutation(len(rows))]
    coef = {v: np.zeros(len(df)) for v in VARIANTS}
    fit = {v: {} for v in VARIANTS}
    for b in range(K_BLOCKS):
        dev = pop & (block != b)
        sco = pop & (block == b)
        cg = _global(r, d, dev)
        coef["global"][sco] = cg
        fit["global"][b] = {"coef": cg, "dev_rows": int(dev.sum())}
        c, info = _cells(r, d, cell, dev, cg)
        coef["router"][sco] = c[cell[sco]]
        fit["router"][b] = info
        cuts = np.quantile(pb[dev], [0.25, 0.5, 0.75])
        qcell = np.searchsorted(cuts, pb, "right")
        c, info = _cells(r, d, qcell, dev, cg)
        coef["difficulty"][sco] = c[qcell[sco]]
        fit["difficulty"][b] = {"cuts": cuts.tolist(), "cells": info}
        c, info = _cells(r, d, pl, dev, cg)
        coef["placebo"][sco] = c[pl[sco]]
        fit["placebo"][b] = info
        cs, rinfo = _repair(r, d, cell, block, dev, cg)
        S = cell > 0
        coef["repair"][sco & ~S] = cg
        coef["repair"][sco & S] = cg if cs is None else cs
        fit["repair"][b] = dict(rinfo, global_coef=cg)
    return coef, fit


def update(df: pd.DataFrame, c: np.ndarray, d: np.ndarray | None = None) -> np.ndarray:
    """The floored update; exactly 0.0 where c*d is 0 (the row keeps v22's bytes)."""
    p0 = df.p_v22.to_numpy("float64")
    cd = c * (df.d.to_numpy("float64") if d is None else d)
    u = np.zeros(len(df))
    m = cd != 0
    u[m] = np.maximum(p0[m] + cd[m], MIN_PRED) - p0[m]
    return u


def oracle(df: pd.DataFrame, d: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Per-row best c in [0, 1], interior values included, on the floored prediction."""
    pop = df["pop"].to_numpy(bool)
    y, p0 = df.y_true.to_numpy("float64"), df.p_v22.to_numpy("float64")
    r = df.r.to_numpy("float64")
    d = df.d.to_numpy("float64") if d is None else d
    with np.errstate(divide="ignore", invalid="ignore"):
        cu = np.where(d != 0, np.clip(r / d, 0.0, 1.0), 0.0)

    def loss(c):
        cd = c * d
        p = np.where(cd != 0, np.maximum(p0 + cd, MIN_PRED), p0)
        return (y - p) ** 2
    cands = np.vstack([np.zeros(len(df)), np.ones(len(df)), cu])
    L = np.vstack([loss(c) for c in cands])
    best = np.argmin(L, axis=0)
    c_star = np.where(pop, cands[best, np.arange(len(df))], 0.0)
    grid = np.min(np.vstack([loss(np.full(len(df), g)) for g in np.linspace(0, 1, 101)]), 0)
    lo = loss(c_star)
    info = {"grid_check_max_excess_s2": float(np.max((lo - grid)[pop])) if pop.any() else 0.0,
            "rows_interior": int((pop & (c_star > 0) & (c_star < 1)).sum()),
            "rows_at_1": int((pop & (c_star == 1)).sum()),
            "rows_at_0": int((pop & (c_star == 0)).sum())}
    return c_star, info


# ------------------------------------------------------------------ statistics ----

def _day_codes(days: np.ndarray):
    codes, uniq = pd.factorize(pd.Series(days))
    order = np.argsort(np.asarray(uniq).astype(str))
    rank = np.empty(len(uniq), int)
    rank[order] = np.arange(len(uniq))
    return rank[codes], len(uniq)


def day_ci(v: np.ndarray, days: np.ndarray, rng) -> list[float]:
    di, k = _day_codes(days)
    num = np.bincount(di, weights=v, minlength=k)
    cnt = np.bincount(di, minlength=k).astype("float64")
    dr = rng.integers(0, k, size=(BOOT, k))
    b = num[dr].sum(1) / cnt[dr].sum(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def block_ci(v: np.ndarray, days: np.ndarray, rng, lengths=(1, 2, 3)) -> dict:
    """Moving-block bootstrap of mean(v) at 1, 2 and 3 contiguous days (exp-039 packet)."""
    di, k = _day_codes(days)
    num = np.bincount(di, weights=v, minlength=k)
    cnt = np.bincount(di, minlength=k).astype("float64")
    out = {}
    for L in lengths:
        nb = int(np.ceil(k / L))
        st = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
        idx = np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1),
                         k - 1)[:, :k]
        b = num[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
        out[f"ci_block_{L}d"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
    return out


def fold_table(df: pd.DataFrame, u: np.ndarray, rng) -> dict:
    """The paired all-row reading of one update on one fold, with the COMMON.md splits."""
    y = df.y_true.to_numpy("float64")
    r = df.r.to_numpy("float64")
    days = df.day.to_numpy()
    cl = df.clean.to_numpy(bool)
    v = 2 * r * u - u * u
    out = {"rows": int(len(df)), "rows_changed": int((u != 0).sum()),
           "mse_gain_s2_per_row": float(v.mean()), "ci95_day": day_ci(v, days, rng),
           "multi_day": block_ci(v, days, rng),
           "rmse_v22": float(np.sqrt(np.mean(r ** 2))),
           "rmse_new": float(np.sqrt(np.mean((r - u) ** 2))),
           "clean_mse_gain_s2_per_row": float(v[cl].mean()),
           "rmse_v22_clean": float(np.sqrt(np.mean(r[cl] ** 2))),
           "rmse_new_clean": float(np.sqrt(np.mean((r[cl] - u[cl]) ** 2)))}
    out["rmse_gain_s"] = out["rmse_v22"] - out["rmse_new"]
    out["rmse_gain_clean_s"] = out["rmse_v22_clean"] - out["rmse_new_clean"]
    for name, m in (("le3600", y <= 3600), ("gt3600", y > 3600),
                    ("nonm", df.regime.astype(str).to_numpy() == "nonm"),
                    ("population", df["pop"].to_numpy(bool))):
        out[f"rmse_v22_{name}"] = float(np.sqrt(np.mean(r[m] ** 2))) if m.any() else None
        out[f"rmse_new_{name}"] = (float(np.sqrt(np.mean((r[m] - u[m]) ** 2)))
                                   if m.any() else None)
        out[f"rows_{name}"] = int(m.sum())
    out["by_cell"] = {}
    for k in range(4):
        m = df["pop"].to_numpy(bool) & (df.cell.to_numpy(int) == k)
        out["by_cell"][k] = {"rows": int(m.sum()), "rows_changed": int((u[m] != 0).sum()),
                             "sse_gain_Ms2": float(v[m].sum() / 1e6)}
    out["by_airport"] = {a: {"rows": int(m.sum()), "sse_gain_Ms2": float(v[m].sum() / 1e6)}
                         for a, m in ((a, df.airport.astype(str).to_numpy() == a)
                                      for a in sorted(df.airport.astype(str).unique()))}
    return out


def board_seconds(mse0: dict, mse1: dict) -> float:
    b0 = np.sqrt(sum(SHARES[f] * mse0[f] for f in SHARES))
    b1 = np.sqrt(sum(SHARES[f] * mse1[f] for f in SHARES))
    return float(b0 - b1)


def board_ci(dfs: dict, u0: dict, u1: dict, rng, clean: bool = False,
             lengths=(1,)) -> dict:
    """Board-second gain of u1 over u0 with a recombined day-block bootstrap.

    Each ranking-share fold is resampled on its own days (blocks of L contiguous days)
    and the board is rebuilt from the resampled MSEs; a fold without a reading carries
    u0 == u1 and contributes only its level."""
    out = {}
    for L in lengths:
        m0 = np.zeros(BOOT)
        m1 = np.zeros(BOOT)
        for f, s in SHARES.items():
            df = dfs[f]
            m = df.clean.to_numpy(bool) if clean else np.ones(len(df), bool)
            r = df.r.to_numpy("float64")[m]
            di, k = _day_codes(df.day.to_numpy()[m])
            e0 = np.bincount(di, weights=(r - u0[f][m]) ** 2, minlength=k)
            e1 = np.bincount(di, weights=(r - u1[f][m]) ** 2, minlength=k)
            c = np.bincount(di, minlength=k).astype("float64")
            nb = int(np.ceil(k / L))
            st = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
            idx = np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1),
                             k - 1)[:, :k]
            m0 += s * e0[idx].sum(1) / c[idx].sum(1)
            m1 += s * e1[idx].sum(1) / c[idx].sum(1)
        g = np.sqrt(m0) - np.sqrt(m1)
        out[f"ci_block_{L}d"] = [float(x) for x in np.percentile(g, [2.5, 97.5])]
    return out


def board_point(dfs: dict, u0: dict, u1: dict, clean: bool = False) -> float:
    mse0, mse1 = {}, {}
    for f in SHARES:
        df = dfs[f]
        m = df.clean.to_numpy(bool) if clean else np.ones(len(df), bool)
        r = df.r.to_numpy("float64")[m]
        mse0[f] = float(np.mean((r - u0[f][m]) ** 2))
        mse1[f] = float(np.mean((r - u1[f][m]) ** 2))
    return board_seconds(mse0, mse1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+", choices=FOLDS)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    # every fold's panel is loaded so the board level is exact; a fold not read keeps u = 0
    dfs, have = {}, []
    for f in FOLDS:
        if f in args.folds:
            dfs[f] = load_fold(f)
            have.append(f)
        else:
            p = NP.load(f)
            p["pop"] = False
            p["r"] = p.y_true.to_numpy("float64") - p.p_v22.to_numpy("float64")
            dfs[f] = p
    rec = {"folds_read": have, "shares": SHARES, "K_blocks": K_BLOCKS, "N0": N0,
           "boot": BOOT, "seed": SEED, "cells": {"dis_sched_s": DIS_SCHED_S,
                                                 "dis_lobt_s": DIS_LOBT_S},
           "basis": ("the whole board: july_loo and january read" if
                     {"july_loo", "january"} <= set(have) else
                     "partial: only " + ", ".join(h for h in have if h in SHARES)
                     + " read; the other ranking-share fold carries u = 0"),
           "folds": {}, "board": {}}
    zero = {f: np.zeros(len(dfs[f])) for f in FOLDS}
    U = {v: dict(zero) for v in VARIANTS + ["oracle", "oracle_placebo"]}
    for f in have:
        df = dfs[f]
        rng = np.random.default_rng(SEED)
        coef, fit = crossfit(df, rng)
        cst, oinfo = oracle(df)
        d_perm = permuted_d(df, np.random.default_rng(SEED + 1))
        cst_p, oinfo_p = oracle(df, d_perm)
        fr = {"population_rows": int(df["pop"].sum()),
              "population_ids_sha256": sha_arr(np.sort(df.MVT_ID_mvt[df["pop"]].to_numpy())),
              "v22_vector_sha256": sha_arr(df.MVT_ID_mvt, df.p_v22),
              "d_rms_population": float(np.sqrt(np.mean(df.d[df["pop"]] ** 2))),
              "cell_rows": {int(k): int(v) for k, v in
                            df.loc[df["pop"], "cell"].value_counts().sort_index().items()},
              "block_rows": {int(k): int(v) for k, v in
                             df.loc[df["pop"], "block"].value_counts().sort_index().items()},
              "expert_rmse_population": {
                  n: float(np.sqrt(np.mean((df.y_true - df[f"p_{n}"])[df["pop"]] ** 2)))
                  for n in ("blind", "aware")},
              "oracle": oinfo, "oracle_placebo": oinfo_p, "fits": fit, "variants": {}}
        outside = ~df["pop"].to_numpy(bool)
        for v in VARIANTS + ["oracle", "oracle_placebo"]:
            c = {"oracle": cst, "oracle_placebo": cst_p}.get(v, coef.get(v))
            u = update(df, c, d_perm if v == "oracle_placebo" else None)
            if np.any(u[outside] != 0):
                raise SystemExit(f"{f}/{v}: a row outside the population moved")
            U[v][f] = u
            fr["variants"][v] = fold_table(df, u, np.random.default_rng(SEED))
        # the paired router-minus-control readings, per row of the whole fold
        r = df.r.to_numpy("float64")
        days = df.day.to_numpy()
        for a, b in (("router", "global"), ("router", "difficulty"), ("router", "placebo"),
                     ("repair", "global"), ("difficulty", "global")):
            w = (r - U[b][f]) ** 2 - (r - U[a][f]) ** 2
            fr[f"{a}_minus_{b}"] = {"mse_s2_per_row": float(w.mean()),
                                    "ci95_day": day_ci(w, days, np.random.default_rng(SEED)),
                                    "multi_day": block_ci(w, days, np.random.default_rng(SEED)),
                                    "rmse_s": float(np.sqrt(np.mean((r - U[b][f]) ** 2))
                                                    - np.sqrt(np.mean((r - U[a][f]) ** 2)))}
        rec["folds"][f] = fr
        print(f"{f}: population {fr['population_rows']:,}  d rms {fr['d_rms_population']:.2f}  "
              f"experts blind {fr['expert_rmse_population']['blind']:.2f} aware "
              f"{fr['expert_rmse_population']['aware']:.2f}", flush=True)
        for v in VARIANTS + ["oracle", "oracle_placebo"]:
            t = fr["variants"][v]
            print(f"   {v:10s} changed {t['rows_changed']:>7,}  s2/row {t['mse_gain_s2_per_row']:+9.3f} "
                  f"[{t['ci95_day'][0]:+8.3f},{t['ci95_day'][1]:+8.3f}]  rmse {t['rmse_gain_s']:+.4f} s "
                  f"clean {t['rmse_gain_clean_s']:+.4f} s", flush=True)
        for key in [k for k in fr if "_minus_" in k]:
            x = fr[key]
            print(f"   {key:24s} s2/row {x['mse_s2_per_row']:+8.3f} "
                  f"[{x['ci95_day'][0]:+8.3f},{x['ci95_day'][1]:+8.3f}]", flush=True)
    for v in VARIANTS + ["oracle", "oracle_placebo"]:
        rng = np.random.default_rng(SEED)
        rec["board"][v] = {"all_row_gain_s": board_point(dfs, zero, U[v]),
                           "all_row_ci": board_ci(dfs, zero, U[v], rng, lengths=(1, 2, 3)),
                           "clean_gain_s": board_point(dfs, zero, U[v], clean=True),
                           "december_rmse_gain_s": (
                               rec["folds"]["december"]["variants"][v]["rmse_gain_s"]
                               if "december" in rec["folds"] else None)}
    for a, b in (("router", "global"), ("router", "difficulty"), ("router", "placebo"),
                 ("repair", "global"), ("difficulty", "global")):
        rng = np.random.default_rng(SEED)
        rec["board"][f"{a}_minus_{b}"] = {
            "all_row_gain_s": board_point(dfs, U[b], U[a]),
            "all_row_ci": board_ci(dfs, U[b], U[a], rng, lengths=(1, 2, 3)),
            "clean_gain_s": board_point(dfs, U[b], U[a], clean=True)}
    rec["board_v22_all_row_rmse"] = float(np.sqrt(sum(
        SHARES[f] * np.mean(dfs[f].r.to_numpy("float64") ** 2) for f in SHARES)))
    print("\nboard (" + rec["basis"] + ")")
    for k, x in rec["board"].items():
        print(f"   {k:24s} all-row {x['all_row_gain_s']:+.5f} s  ci1d "
              f"[{x['all_row_ci']['ci_block_1d'][0]:+.5f},{x['all_row_ci']['ci_block_1d'][1]:+.5f}]"
              f"  clean {x['clean_gain_s']:+.5f}", flush=True)
    # per-row record: the registered router's final prediction, plus every input to it
    for f in have:
        df = dfs[f]
        o = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt.to_numpy("float64"), "fold": f,
                          "y_true": df.y_true.to_numpy("float64"),
                          "y_pred": df.p_v22.to_numpy("float64") + U["router"][f],
                          "p_v22": df.p_v22.to_numpy("float64"),
                          "d": df.d.to_numpy("float64"), "cell": df.cell.to_numpy(int),
                          "block": df.block.to_numpy(int), "pop": df["pop"].to_numpy(bool)})
        for v in VARIANTS + ["oracle", "oracle_placebo"]:
            o[f"u_{v}"] = U[v][f]
        # the full per-row record is large (17 MB a fold), so it lives in git-ignored work/
        # with its hash; the committed OOF keeps the COMMON.md columns plus the global form
        full = EXP / f"work/router-rows-{f}.parquet"
        o.to_parquet(full, index=False)
        rec["folds"][f]["router_rows"] = str(full.relative_to(ROOT))
        rec["folds"][f]["router_rows_sha256"] = NE.sha_file(full)
        slim = o[["MVT_ID_mvt", "fold", "y_true", "y_pred"]].copy()
        slim["y_pred_global"] = df.p_v22.to_numpy("float64") + U["global"][f]
        path = EXP / f"oof/router-{f}.parquet"
        slim.to_parquet(path, index=False)
        rec["folds"][f]["router_oof"] = str(path.relative_to(ROOT))
        rec["folds"][f]["router_oof_sha256"] = NE.sha_file(path)
        # per-date contributions of the router and the global coefficient
        r = df.r.to_numpy("float64")
        dt = pd.DataFrame({"day": df.day.to_numpy(), "rows": 1,
                           "sse_v22": r ** 2,
                           "sse_global": (r - U["global"][f]) ** 2,
                           "sse_router": (r - U["router"][f]) ** 2}).groupby("day").sum()
        dt.to_csv(EXP / f"analysis/dates-{f}.csv")
    rec["seconds"] = round(time.time() - t0, 1)
    tag = args.tag or "-".join(have)
    out = EXP / f"analysis/router-{tag}.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("written", out.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
