"""exp-052-nmv22: iteration 1's read (SPEC 1.7-1.11), arithmetic on the frozen expert values.

Per fold, on the population P (outside P every predictor keeps v22's bytes, asserted):

    r      = y - p_v22
    d22    = u_blind - u_aware          (outer values: the frozen scoring direction)
    p(c)   = p_v22 + (u_aware + c * d22),  c in [0, 1]

Scoring block b's coefficients are fitted on P's rows of the other four blocks with the
INNER values of the experts that saw neither b nor the row's own block (SPEC 1.5).  The
same machinery prices the sham contrast d_sham = u_blind - u_sham around q_sham.  The old
Q08 contrast is reproduced through exp-050's own code and must hit its recorded board
router - global to 1e-12.

The script refuses a fold whose blindness proof did not pass or whose expert file moved.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_price.py
    ... rx_price.py --dry-run --out-dir <scratch>      # synthetic labels, no label read
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402

NR, NO = C.NR, C.NO
Q08_ROUTER_MINUS_GLOBAL = 0.00230499969450193
SEGS = ["P", "nm_bulk guard-held", "nm_late", "nonm", "noac", "guard-held (every regime)"]


# ------------------------------------------------------------------------- loading ----

def load(fold: str, dry: bool, src: Path | None, null: bool = False) -> dict:
    cols = C.LABEL_FREE + ([] if dry else ["y_true"])
    pan = C.panel(fold, cols)
    pop = C.population(pan, fold)
    ej = C.experts_json(fold, src, null)
    rec = json.loads(ej.read_text())
    proof = json.loads(Path(rec["proof"] if Path(rec["proof"]).is_absolute()
                            else C.ROOT / rec["proof"]).read_text())
    if not (proof["passes"] and proof["prediction_proof"]["passes"]):
        raise SystemExit(f"{fold}: the blindness proof did not pass; no score may be read")
    rp = C.resid_path(fold, src if dry else None, null)
    if C.sha_file(rp) != rec["resid_sha256"]:
        raise SystemExit(f"{rp.name} moved since it was written")
    R = pd.read_parquet(rp)
    ids = pan.MVT_ID_mvt.to_numpy("float64")
    if not np.array_equal(R.MVT_ID_mvt.to_numpy("float64"), ids[pop]):
        raise SystemExit(f"{fold}: the expert rows are not P in panel order")
    p22 = pan.p_v22.to_numpy("float64")
    if dry:
        y = p22 + np.random.default_rng(1).normal(0.0, 170.0, len(pan))
    else:
        y = pan.y_true.to_numpy("float64")
    if null:
        # SPEC 1.15 (a): the NULL's label on P is p_v22 + r_null, rebuilt exactly as rx_fit
        kp = C.features(fold, ["keep_reachable"]).set_index("MVT_ID_mvt").reindex(ids[pop])
        yP = y[pop]
        fitmask = kp.keep_reachable.to_numpy(bool) & (yP > 0)
        key = (pan.airport.astype(str) + "|" + pan.day.astype(str)).to_numpy()[pop]
        rn = C.null_residual(yP - p22[pop], fitmask, key)
        if not dry and C.sha_arr(rn) != rec["null"]["r_null_sha256"]:
            raise SystemExit(f"{fold}: the rebuilt NULL target is not the one the experts fitted")
        y = y.copy()
        y[pop] = p22[pop] + rn
    F = C.features(fold, ["aobt_sched", "aobt_lobt"]).set_index("MVT_ID_mvt").reindex(ids)
    a_s = np.abs(F.aobt_sched.to_numpy("float64"))
    a_l = np.abs(F.aobt_lobt.to_numpy("float64"))
    cell = (2 * np.where(np.isfinite(a_s), a_s > NR.DIS_SCHED_S, False).astype(int)
            + np.where(np.isfinite(a_l), a_l > NR.DIS_LOBT_S, False).astype(int))
    blk_all = C.blocks(pan)
    if not np.array_equal(blk_all[pop], R.block.to_numpy()):
        raise SystemExit(f"{fold}: the expert file's blocks are not Q08's")
    df = pd.DataFrame({"MVT_ID_mvt": ids, "airport": pan.airport.astype(str).to_numpy(),
                       "day": pan.day.to_numpy(), "regime": pan.regime.astype(str).to_numpy(),
                       "guard_held": pan.guard_held.to_numpy(bool),
                       "clean": pan.clean.to_numpy(bool), "y": y, "p_v22": p22,
                       "r": y - p22, "pop": pop, "block": blk_all, "cell": cell})
    O = {T: R[f"u_{T}_o"].to_numpy("float64") for T in C.TYPES}
    I = {T: {b: R[f"u_{T}_i{b}"].to_numpy("float64") for b in range(C.K)} for T in C.TYPES}
    return {"df": df, "O": O, "I": I, "experts_json": str(ej), "experts_sha": C.sha_file(ej),
            "proof": proof, "rec": rec}


def full(df: pd.DataFrame, uP: np.ndarray) -> np.ndarray:
    u = np.zeros(len(df))
    u[df["pop"].to_numpy(bool)] = uP
    return u


# -------------------------------------------------------------------- coefficients ----

def crossfit(F: dict, base: str, alt: str, cells_list: list[np.ndarray]) -> tuple:
    """Global and cell coefficients per scoring block, from inner development values.

    Returns (per-P-row global coefficient, [per-P-row cell coefficient per cells array],
    fit records).  `cells_list[0]` is the registered router; the rest are placebos."""
    df = F["df"]
    P = df["pop"].to_numpy(bool)
    rP = df.r.to_numpy("float64")[P]
    blk = df.block.to_numpy(int)[P]
    cg = np.full(P.sum(), np.nan)
    cc = [np.full(P.sum(), np.nan) for _ in cells_list]
    fits = []
    for b in range(C.K):
        dev, sco = blk != b, blk == b
        bI, aI = F["I"][base][b], F["I"][alt][b]
        d_dev = aI[dev] - bI[dev]
        r_dev = rP[dev] - bI[dev]
        if not (np.isfinite(d_dev).all() and np.isfinite(r_dev).all()):
            raise SystemExit("a development value is missing")
        one = np.ones(dev.sum(), bool)
        g = NR._global(r_dev, d_dev, one)
        cg[sco] = g
        rec = {"block": b, "dev_rows": int(dev.sum()), "score_rows": int(sco.sum()),
               "global": g, "d_rms_dev": float(np.sqrt(np.mean(d_dev ** 2)))}
        for i, cells in enumerate(cells_list):
            cP = cells[P]
            c, info = NR._cells(r_dev, d_dev, cP[dev], one, g)
            cc[i][sco] = c[cP[sco]]
            if i == 0:
                rec["router"] = info
        fits.append(rec)
    return cg, cc, fits


def oracle(raO: np.ndarray, dO: np.ndarray) -> tuple[np.ndarray, dict]:
    with np.errstate(divide="ignore", invalid="ignore"):
        cst = np.where(dO != 0, np.clip(raO / dO, 0.0, 1.0), 0.0)
    lo = (raO - cst * dO) ** 2
    grid = np.min(np.vstack([(raO - g * dO) ** 2 for g in np.linspace(0, 1, 101)]), 0)
    return cst, {"grid_check_max_excess_s2": float(np.max(lo - grid)),
                 "rows_interior": int(((cst > 0) & (cst < 1)).sum()),
                 "rows_at_1": int((cst == 1).sum()), "rows_at_0": int((cst == 0).sum())}


# -------------------------------------------------------------------- statistics ----

def segments(df: pd.DataFrame) -> dict:
    reg, held, P = df.regime.to_numpy(), df.guard_held.to_numpy(bool), df["pop"].to_numpy(bool)
    seg = {"P": P, "nm_bulk guard-held": (reg == "nm_bulk") & held}
    for g in ("nm_late", "nonm", "noac"):
        seg[g] = reg == g
    seg["guard-held (every regime)"] = held
    part = sum(seg[k].astype(int) for k in SEGS[:5])
    if not (part == 1).all():
        raise SystemExit("the five regime segments do not partition the fold")
    return seg


def rmse(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else None


def table(df: pd.DataFrame, u: np.ndarray, rng_seed: int = NR.SEED) -> dict:
    """Predictor p_v22 + u against v22 on one fold (u is 0 outside P, asserted)."""
    P = df["pop"].to_numpy(bool)
    if np.any(u[~P] != 0):
        raise SystemExit("a row outside P moved")
    p22 = df.p_v22.to_numpy("float64")
    pred = p22.copy()
    pred[P] = p22[P] + u[P]
    if not np.array_equal(pred[~P].view("uint64"), p22[~P].view("uint64")):
        raise SystemExit("a row outside P lost v22's bytes")
    r, y = df.r.to_numpy("float64"), df.y.to_numpy("float64")
    e = y - pred
    v = r ** 2 - e ** 2
    days = df.day.to_numpy()
    cl = df.clean.to_numpy(bool)
    out = {"rows": int(len(df)), "rows_changed": int((u != 0).sum()),
           "rows_below_60s": int((pred[P] < 60).sum()),
           "mse_gain_s2_per_row": float(v.mean()),
           "ci95_day": NR.day_ci(v, days, np.random.default_rng(rng_seed)),
           "multi_day": NR.block_ci(v, days, np.random.default_rng(rng_seed)),
           "rmse_v22": rmse(r), "rmse_new": rmse(e),
           "rmse_v22_clean": rmse(r[cl]), "rmse_new_clean": rmse(e[cl])}
    out["rmse_gain_s"] = out["rmse_v22"] - out["rmse_new"]
    out["rmse_gain_clean_s"] = out["rmse_v22_clean"] - out["rmse_new_clean"]
    for name, m in (("le3600", y <= 3600), ("gt3600", y > 3600)):
        out[f"rows_{name}"] = int(m.sum())
        out[f"rmse_v22_{name}"], out[f"rmse_new_{name}"] = rmse(r[m]), rmse(e[m])
    out["segments"] = {k: {"rows": int(m.sum()), "rows_changed": int((u[m] != 0).sum()),
                           "rmse_v22": rmse(r[m]), "rmse_new": rmse(e[m])}
                       for k, m in segments(df).items()}
    cell = df.cell.to_numpy(int)
    out["by_cell"] = {int(k): {"rows": int((P & (cell == k)).sum()),
                               "sse_gain_Ms2": float(v[P & (cell == k)].sum() / 1e6)}
                      for k in range(4)}
    ap = df.airport.to_numpy()
    out["by_airport"] = {a: float(v[ap == a].sum() / 1e6) for a in sorted(set(ap))}
    return out


def paired(df: pd.DataFrame, uA: np.ndarray, uB: np.ndarray) -> dict:
    """B against A on one fold: s2/row with 1/2/3-day intervals, and the RMSE gain."""
    r = df.r.to_numpy("float64")
    w = (r - uA) ** 2 - (r - uB) ** 2
    days = df.day.to_numpy()
    cl = df.clean.to_numpy(bool)
    return {"mse_s2_per_row": float(w.mean()),
            "ci95_day": NR.day_ci(w, days, np.random.default_rng(NR.SEED)),
            "multi_day": NR.block_ci(w, days, np.random.default_rng(NR.SEED)),
            "rmse_s": rmse(r - uA) - rmse(r - uB),
            "rmse_clean_s": rmse((r - uA)[cl]) - rmse((r - uB)[cl])}


def board(dfs: dict, uA: dict, uB: dict) -> dict:
    return {"all_row_gain_s": NR.board_point(dfs, uA, uB),
            "all_row_ci": NR.board_ci(dfs, uA, uB, np.random.default_rng(NR.SEED),
                                      lengths=(1, 2, 3)),
            "clean_gain_s": NR.board_point(dfs, uA, uB, clean=True)}


def perm_summary(real: float, refs: list[float]) -> dict:
    a = np.asarray(refs)
    return {"real": real, "n": int(len(a)), "mean": float(a.mean()),
            "p2_5": float(np.percentile(a, 2.5)), "p97_5": float(np.percentile(a, 97.5)),
            "max": float(a.max()), "min": float(a.min()),
            "rank_of_real_among_n_plus_1": int(1 + (a < real).sum()),
            "real_exceeds_all": bool((a < real).all())}


# ---------------------------------------------------------------------------- main ----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--null", action="store_true",
                    help="SPEC 1.15 (a): price the NULL run against its permuted label")
    args = ap.parse_args()
    if args.dry_run != (args.out_dir is not None):
        raise SystemExit("--dry-run and --out-dir go together")
    nl = args.null
    out_a = args.out_dir or C.EXP / "analysis"
    out_o = args.out_dir or C.EXP / "oof"
    out_w = args.out_dir or C.EXP / "work"
    OUT = out_a / f"{C.tagged('iter1', nl)}.json"
    rows_p = {f: out_w / f"{C.tagged('iter1', nl)}-rows-{f}.parquet" for f in C.FOLDS}
    oof_p = {} if nl else {f: out_o / f"iter1-{f}.parquet" for f in C.FOLDS}
    NO.refuse_existing([OUT, *rows_p.values(), *oof_p.values()], args.overwrite)
    t0 = time.time()
    FD = {f: load(f, args.dry_run, args.out_dir, nl) for f in C.FOLDS}
    dfs = {f: FD[f]["df"] for f in C.FOLDS}
    zero = {f: np.zeros(len(dfs[f])) for f in C.FOLDS}
    rec = {"what": "exp-052 iteration 1: d22 = u_blind - u_aware on v22's residual"
                   + (" -- NULL run (SPEC 1.15): experts and label on the permuted residual"
                      if nl else ""),
           "dry_run": bool(args.dry_run), "null": bool(nl), "pinned_inputs": C.check_pinned(),
           "code_sha256": {p.name: C.sha_file(p) for p in sorted(C.HERE.glob("rx_*.py"))},
           "shares": NR.SHARES, "boot": NR.BOOT, "boot_seed": NR.SEED, "n_perm": C.N_PERM,
           "seeds": {"placebo": [C.SEEDS_PLACEBO[0], C.SEEDS_PLACEBO[-1]],
                     "oracle_d22": [C.SEEDS_ORACLE_D22[0], C.SEEDS_ORACLE_D22[-1]],
                     "oracle_sham": [C.SEEDS_ORACLE_SHAM[0], C.SEEDS_ORACLE_SHAM[-1]]},
           "folds": {}, "board": {}, "board_paired": {}, "perm": {}}
    U = {}          # predictor -> fold -> full-length u
    PMSE = {}       # "<pre>placebo" / "<pre>oracle_perm" -> fold -> [all-row MSE per draw]

    def put(name, f, u):
        U.setdefault(name, {})[f] = u

    def mse(df, u):
        return float(np.mean((df.r.to_numpy("float64") - u) ** 2))

    def rel(p: Path) -> str:
        return str(p) if args.dry_run else str(p.relative_to(C.ROOT))

    for f in C.FOLDS:
        F = FD[f]
        df = F["df"]
        P = df["pop"].to_numpy(bool)
        rP = df.r.to_numpy("float64")[P]
        key = (df.airport.astype(str) + "|" + df.day.astype(str)).to_numpy()[P]
        cellP = df.cell.to_numpy(int)[P]
        ua, ub, us = F["O"]["aware"], F["O"]["blind"], F["O"]["sham"]
        d22, dsh = ub - ua, ub - us
        # the router cells and the 50 placebo relabellings (both contrasts share them)
        cells_full = [df.cell.to_numpy(int)]
        for s in C.SEEDS_PLACEBO:
            perm = C.airday_perm(key, np.random.default_rng(s))
            c = df.cell.to_numpy(int).copy()
            c[P] = cellP[perm]
            cells_full.append(c)
        fr = {"population_rows": int(P.sum()),
              "population_ids_sha256": C.sha_arr(np.sort(df.MVT_ID_mvt.to_numpy()[P])),
              "v22_vector_sha256": C.sha_arr(df.MVT_ID_mvt, df.p_v22),
              "experts_json_sha256": F["experts_sha"],
              "cell_rows": {int(k): int((cellP == k).sum()) for k in range(4)},
              "block_rows": {int(k): int((df.block.to_numpy()[P] == k).sum()) for k in range(C.K)},
              "u_rms": {T: rmse(F["O"][T]) for T in C.TYPES},
              "u_mean": {T: float(F["O"][T].mean()) for T in C.TYPES},
              "d22": {"rms": rmse(d22), "mean": float(d22.mean()),
                      "share_abs_gt_10s": float(np.mean(np.abs(d22) > 10)),
                      "rms_by_cell": {int(k): rmse(d22[cellP == k]) for k in range(4)},
                      "corr_with_r_minus_u_aware": float(np.corrcoef(d22, rP - ua)[0, 1])},
              "d_sham": {"rms": rmse(dsh), "mean": float(dsh.mean()),
                         "corr_with_r_minus_u_sham": float(np.corrcoef(dsh, rP - us)[0, 1])},
              "experts_rmse_on_P": {"v22": rmse(rP), **{f"q_{T}": rmse(rP - F["O"][T])
                                                        for T in C.TYPES}}}
        for tag, base, alt in (("d22", "aware", "blind"), ("sham", "sham", "blind")):
            bO, aO = F["O"][base], F["O"][alt]
            dO, raO = aO - bO, rP - bO
            cg, cc, fits = crossfit(F, base, alt, cells_full)
            pre = "" if tag == "d22" else "sham_"
            put(f"{pre}global", f, full(df, bO + cg * dO))
            put(f"{pre}router", f, full(df, bO + cc[0] * dO))
            cst, oinfo = oracle(raO, dO)
            put(f"{pre}oracle", f, full(df, bO + cst * dO))
            fr[f"{pre}fits"] = fits
            fr[f"{pre}oracle_info"] = oinfo
            # placebo routers (cell labels permuted inside airport x day), all-row MSE only
            PMSE.setdefault(f"{pre}placebo", {})[f] = [
                mse(df, full(df, bO + cc[i] * dO)) for i in range(1, len(cells_full))]
            # oracle permutation references (d permuted inside airport x day), MSE only
            seeds = C.SEEDS_ORACLE_D22 if tag == "d22" else C.SEEDS_ORACLE_SHAM
            op = []
            for s in seeds:
                dp = dO[C.airday_perm(key, np.random.default_rng(s))]
                cp, _ = oracle(raO, dp)
                op.append(mse(df, full(df, bO + cp * dp)))
            PMSE.setdefault(f"{pre}oracle_perm", {})[f] = op
        put("q_aware", f, full(df, ua))
        put("q_blind", f, full(df, ua + 1.0 * d22))
        put("q_sham", f, full(df, us))
        rec["folds"][f] = fr
        print(f"{C.utc()} {f}: P {P.sum():,}  d22 rms {fr['d22']['rms']:.2f}  "
              f"u rms {fr['u_rms']}", flush=True)

    # ---- the old Q08 contrast, reproduced through exp-050's own code (not in a dry run)
    if not args.dry_run and not nl:
        d50 = {f: NR.load_fold(f) for f in C.FOLDS}
        old = {"global": {}, "router": {}}
        for f in C.FOLDS:
            if not np.array_equal(d50[f].MVT_ID_mvt.to_numpy(), dfs[f].MVT_ID_mvt.to_numpy()) \
                    or not np.array_equal(d50[f]["pop"].to_numpy(), dfs[f]["pop"].to_numpy()):
                raise SystemExit(f"{f}: exp-050's rows are not this experiment's rows")
            coef, _ = NR.crossfit(d50[f], np.random.default_rng(NR.SEED))
            for v in old:
                old[v][f] = NR.update(d50[f], coef[v])
        got = NR.board_point(d50, old["global"], old["router"])
        if abs(got - Q08_ROUTER_MINUS_GLOBAL) > 1e-12:
            raise SystemExit(f"the old d reproduces router - global {got!r}, not Q08's "
                             f"{Q08_ROUTER_MINUS_GLOBAL!r}")
        rec["old_d"] = {"reproduced_board_router_minus_global": got,
                        "recorded": Q08_ROUTER_MINUS_GLOBAL, "tolerance": 1e-12,
                        "board_router_minus_global": board(d50, old["global"], old["router"]),
                        "board_global": board(d50, zero, old["global"]),
                        "board_router": board(d50, zero, old["router"]),
                        "folds": {f: {"router_minus_global": paired(d50[f], old["global"][f],
                                                                    old["router"][f])}
                                  for f in C.FOLDS}}
        print(f"old d reproduces Q08's board router - global {got:.14f}", flush=True)

    # ---- per-fold tables against v22, and the paired differences
    NAMES = ["q_aware", "q_blind", "q_sham", "global", "router", "oracle",
             "sham_global", "sham_router", "sham_oracle"]
    PAIRS = [("global", "router"), ("q_aware", "global"), ("q_aware", "router"),
             ("q_aware", "q_blind"), ("q_sham", "q_aware"), ("q_sham", "q_blind"),
             ("q_aware", "oracle"), ("sham_global", "sham_router"), ("q_sham", "sham_global"),
             ("q_sham", "sham_oracle")]
    for f in C.FOLDS:
        df = dfs[f]
        fr = rec["folds"][f]
        fr["predictors"] = {n: table(df, U[n][f]) for n in NAMES}
        fr["paired"] = {f"{b}_minus_{a}": paired(df, U[a][f], U[b][f]) for a, b in PAIRS}
        fr["decomposition_s2_per_row"] = {
            "v22_to_q_aware": fr["predictors"]["q_aware"]["mse_gain_s2_per_row"],
            "q_aware_to_global": fr["paired"]["global_minus_q_aware"]["mse_s2_per_row"],
            "global_to_router": fr["paired"]["router_minus_global"]["mse_s2_per_row"],
            "v22_to_router": fr["predictors"]["router"]["mse_gain_s2_per_row"]}
        # permutation references, points only (s2/row over the reference's own base)
        for pre, base in (("", "q_aware"), ("sham_", "q_sham")):
            m0 = mse(df, U[base][f])
            fr[f"{pre}oracle_perm_s2_per_row_over_base"] = perm_summary(
                m0 - mse(df, U[f"{pre}oracle"][f]), [m0 - x for x in PMSE[f"{pre}oracle_perm"][f]])
            m0 = mse(df, U[f"{pre}global"][f])
            fr[f"{pre}placebo_minus_global_s2_per_row"] = perm_summary(
                m0 - mse(df, U[f"{pre}router"][f]), [m0 - x for x in PMSE[f"{pre}placebo"][f]])
    for n in NAMES:
        rec["board"][n] = board(dfs, zero, U[n])
        rec["board"][n]["december_rmse_gain_s"] = rec["folds"]["december"]["predictors"][n]["rmse_gain_s"]
    for a, b in PAIRS:
        rec["board_paired"][f"{b}_minus_{a}"] = board(dfs, U[a], U[b])
    dec = {"v22_to_q_aware": rec["board"]["q_aware"]["all_row_gain_s"],
           "q_aware_to_global": rec["board_paired"]["global_minus_q_aware"]["all_row_gain_s"],
           "global_to_router": rec["board_paired"]["router_minus_global"]["all_row_gain_s"],
           "v22_to_router": rec["board"]["router"]["all_row_gain_s"]}
    if abs(dec["v22_to_q_aware"] + dec["q_aware_to_global"] + dec["global_to_router"]
           - dec["v22_to_router"]) > 1e-9:
        raise SystemExit("the board decomposition does not telescope")
    rec["board"]["decomposition_s"] = dec

    def board_from_mse(m0: dict, m1: dict) -> float:
        return float(np.sqrt(sum(s * m0[f] for f, s in NR.SHARES.items()))
                     - np.sqrt(sum(s * m1[f] for f, s in NR.SHARES.items())))
    for pre, base in (("", "q_aware"), ("sham_", "q_sham")):
        m0 = {f: mse(dfs[f], U[base][f]) for f in C.FOLDS}
        real = NR.board_point(dfs, U[base], U[f"{pre}oracle"])
        refs = [board_from_mse(m0, {f: PMSE[f"{pre}oracle_perm"][f][k] for f in C.FOLDS})
                for k in range(C.N_PERM)]
        rec["perm"][f"{pre}oracle_over_base_board_s"] = perm_summary(real, refs)
        m0 = {f: mse(dfs[f], U[f"{pre}global"][f]) for f in C.FOLDS}
        real = NR.board_point(dfs, U[f"{pre}global"], U[f"{pre}router"])
        refs = [board_from_mse(m0, {f: PMSE[f"{pre}placebo"][f][k] for f in C.FOLDS})
                for k in range(C.N_PERM)]
        rec["perm"][f"{pre}placebo_minus_global_board_s"] = perm_summary(real, refs)

    # ---- the registered readings (SPEC 1.11)
    S = rec["board_paired"]["router_minus_global"]
    folds_neg = [f for f in C.FOLDS
                 if rec["folds"][f]["paired"]["router_minus_global"]["mse_s2_per_row"] < 0]
    orc = rec["board_paired"]["oracle_minus_q_aware"]["all_row_gain_s"]
    above = {f: rec["folds"][f]["oracle_perm_s2_per_row_over_base"]["real_exceeds_all"]
             for f in C.FOLDS}
    rec["readings"] = {
        "primary_S_board_router_minus_global_s": S["all_row_gain_s"],
        "S_ci_1d": S["all_row_ci"]["ci_block_1d"],
        "folds_with_negative_router_minus_global": folds_neg,
        "reversed": bool(S["all_row_ci"]["ci_block_1d"][0] > 0 and not folds_neg),
        "ceiling_board_oracle_over_q_aware_s": orc,
        "oracle_above_noise_reference_by_fold": above,
        "useful_raw_ceiling": bool(orc >= 0.10 and all(above.values())),
    }
    rec["readings"]["route"] = (
        "reversed: witness router beside the four cells" if rec["readings"]["reversed"] else
        "useful raw ceiling, four cells fail: rare physical reliability witness"
        if rec["readings"]["useful_raw_ceiling"] else
        "lacks alignment: smallest matched residual-target or capacity control")

    # ---- per-row files
    for f in C.FOLDS:
        df = dfs[f]
        rows = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt, "fold": f, "y_true": df.y,
                             "p_v22": df.p_v22, "pop": df["pop"], "block": df.block,
                             "cell": df.cell})
        for n in NAMES:
            rows[f"u_{n}"] = U[n][f]
        NO.refuse_existing([rows_p[f]], args.overwrite)
        rows.to_parquet(rows_p[f], index=False)
        rec["folds"][f]["rows_file"] = rel(rows_p[f])
        rec["folds"][f]["rows_sha256"] = C.sha_file(rows_p[f])
        if nl:
            continue
        slim = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt, "fold": f, "y_true": df.y,
                             "y_pred": df.p_v22 + U["router"][f],
                             "y_pred_global": df.p_v22 + U["global"][f],
                             "y_pred_q_aware": df.p_v22 + U["q_aware"][f]})
        NO.refuse_existing([oof_p[f]], args.overwrite)
        slim.to_parquet(oof_p[f], index=False)
        rec["folds"][f]["oof"] = rel(oof_p[f])
        rec["folds"][f]["oof_sha256"] = C.sha_file(oof_p[f])
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))

    # ---- the console summary
    print("\nboard (july_loo + january at the shares), gain over v22, s:")
    for n in NAMES:
        x = rec["board"][n]
        print(f"  {n:12s} {x['all_row_gain_s']:+.5f} [{x['all_row_ci']['ci_block_1d'][0]:+.5f},"
              f"{x['all_row_ci']['ci_block_1d'][1]:+.5f}] clean {x['clean_gain_s']:+.5f}  "
              f"dec {x['december_rmse_gain_s']:+.5f}")
    print("paired:")
    for k, x in rec["board_paired"].items():
        print(f"  {k:28s} {x['all_row_gain_s']:+.5f} [{x['all_row_ci']['ci_block_1d'][0]:+.5f},"
              f"{x['all_row_ci']['ci_block_1d'][1]:+.5f}] clean {x['clean_gain_s']:+.5f}")
    for k, x in rec["perm"].items():
        print(f"  perm {k}: real {x['real']:+.5f}  refs mean {x['mean']:+.5f} "
              f"[{x['p2_5']:+.5f},{x['p97_5']:+.5f}] max {x['max']:+.5f} rank "
              f"{x['rank_of_real_among_n_plus_1']}")
    print("readings:", json.dumps(rec["readings"], default=float))
    print(f"written {OUT} ({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
