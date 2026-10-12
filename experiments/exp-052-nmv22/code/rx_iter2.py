"""exp-052-nmv22: iteration 2's read (SPEC 2.3-2.5), the matched capacity control, priced.

For each run (real, then NULL) and fold, the expert values come from `rx_trunc.py`'s truncated
models.  The SELECTED predictor takes, for each scoring block b, the budget

    k*_b = argmin_{k in G} 1/2 [ mean (r - u_aware_inner(k))^2 + mean (r - u_blind_inner(k))^2 ]

over P's rows of the four other blocks (the rows and inner values that fit b's coefficients;
ties to the smaller k), one k*_b for all three types.  Block b is then scored with the outer
values at k*_b and coefficients fitted on the inner values at k*_b, by iteration 1's own rules
and functions (`rx_price.crossfit`, `oracle`, `table`, `paired`, `board`, `perm_summary`,
unchanged).  The fixed grid (k on every block, k in G) is priced the same way and reported only.

At k = 400 the grid must reproduce iteration 1's board figures to 1e-12 (asserted).

    .venv/bin/python experiments/exp-052-nmv22/code/rx_iter2.py
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
import rx_price as RP                                                 # noqa: E402
import rx_trunc as RT                                                 # noqa: E402

NR, NO = C.NR, C.NO
G = RT.G
NAMES = ["q_aware", "q_blind", "q_sham", "global", "router", "oracle",
         "sham_global", "sham_router", "sham_oracle"]
PAIRS = [("global", "router"), ("q_aware", "global"), ("q_aware", "router"),
         ("q_aware", "q_blind"), ("q_sham", "q_aware"), ("q_sham", "q_blind"),
         ("q_aware", "oracle"), ("sham_global", "sham_router"), ("q_sham", "sham_global"),
         ("q_sham", "sham_oracle")]


def rmse(x):
    return float(np.sqrt(np.mean(x ** 2)))


def load_trunc(fold: str, null: bool, trunc_rec: dict) -> pd.DataFrame:
    info = trunc_rec["runs"]["null" if null else "real"][fold]
    p = C.ROOT / info["trunc"]
    if C.sha_file(p) != info["trunc_sha256"]:
        raise SystemExit(f"{p.name} moved since rx_trunc wrote it")
    return pd.read_parquet(p)


def select_k(F: dict, TR: pd.DataFrame) -> tuple[list[int], dict]:
    """SPEC 2.3: k*_b from the development loss of AWARE and BLIND inner values."""
    df = F["df"]
    P = df["pop"].to_numpy(bool)
    rP = df.r.to_numpy("float64")[P]
    blk = df.block.to_numpy(int)[P]
    if not np.array_equal(TR.MVT_ID_mvt.to_numpy("float64"),
                          df.MVT_ID_mvt.to_numpy("float64")[P]):
        raise SystemExit("the truncation file's rows are not P in panel order")
    if not np.array_equal(TR.block.to_numpy(int), blk):
        raise SystemExit("the truncation file's blocks are not the panel's")
    ks, losses = [], {}
    for b in range(C.K):
        dev = blk != b
        L = {}
        for k in G:
            la = np.mean((rP[dev] - TR[f"u_aware_i{b}_k{k}"].to_numpy("float64")[dev]) ** 2)
            lb = np.mean((rP[dev] - TR[f"u_blind_i{b}_k{k}"].to_numpy("float64")[dev]) ** 2)
            L[k] = float(0.5 * (la + lb))
        best = min(G, key=lambda k: (L[k], k))
        ks.append(int(best))
        losses[b] = {int(k): v for k, v in L.items()}
    return ks, losses


def with_values(F: dict, TR: pd.DataFrame, ks: list[int]) -> dict:
    """A copy of iteration 1's fold dict whose O / I are the truncated values at k_b."""
    blk = TR.block.to_numpy(int)
    O = {}
    for T in C.TYPES:
        o = np.full(len(TR), np.nan)
        for b in range(C.K):
            m = blk == b
            o[m] = TR[f"u_{T}_o_k{ks[b]}"].to_numpy("float64")[m]
        O[T] = o
    I = {T: {b: TR[f"u_{T}_i{b}_k{ks[b]}"].to_numpy("float64") for b in range(C.K)}
         for T in C.TYPES}
    return {**F, "O": O, "I": I}


def price(FD: dict, full_tables: bool) -> tuple[dict, dict]:
    """Iteration 1's pricing (rx_price.main's body, unchanged rules) on the given values."""
    dfs = {f: FD[f]["df"] for f in C.FOLDS}
    zero = {f: np.zeros(len(dfs[f])) for f in C.FOLDS}
    rec = {"folds": {}, "board": {}, "board_paired": {}, "perm": {}}
    U, PMSE = {}, {}

    def put(name, f, u):
        U.setdefault(name, {})[f] = u

    def mse(df, u):
        return float(np.mean((df.r.to_numpy("float64") - u) ** 2))

    for f in C.FOLDS:
        F = FD[f]
        df = F["df"]
        P = df["pop"].to_numpy(bool)
        rP = df.r.to_numpy("float64")[P]
        key = (df.airport.astype(str) + "|" + df.day.astype(str)).to_numpy()[P]
        cellP = df.cell.to_numpy(int)[P]
        ua, ub, us = F["O"]["aware"], F["O"]["blind"], F["O"]["sham"]
        d22, dsh = ub - ua, ub - us
        cells_full = [df.cell.to_numpy(int)]
        for s in C.SEEDS_PLACEBO:
            perm = C.airday_perm(key, np.random.default_rng(s))
            c = df.cell.to_numpy(int).copy()
            c[P] = cellP[perm]
            cells_full.append(c)
        fr = {"k_by_block": F.get("k_by_block"), "population_rows": int(P.sum()),
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
            cg, cc, fits = RP.crossfit(F, base, alt, cells_full)
            pre = "" if tag == "d22" else "sham_"
            put(f"{pre}global", f, RP.full(df, bO + cg * dO))
            put(f"{pre}router", f, RP.full(df, bO + cc[0] * dO))
            cst, oinfo = RP.oracle(raO, dO)
            put(f"{pre}oracle", f, RP.full(df, bO + cst * dO))
            fr[f"{pre}fits"] = fits if full_tables else [
                {"block": x["block"], "global": x["global"],
                 "router": [c["coef"] for c in x["router"]]} for x in fits]
            fr[f"{pre}oracle_info"] = oinfo
            PMSE.setdefault(f"{pre}placebo", {})[f] = [
                mse(df, RP.full(df, bO + cc[i] * dO)) for i in range(1, len(cells_full))]
            seeds = C.SEEDS_ORACLE_D22 if tag == "d22" else C.SEEDS_ORACLE_SHAM
            op = []
            for s in seeds:
                dp = dO[C.airday_perm(key, np.random.default_rng(s))]
                cp, _ = RP.oracle(raO, dp)
                op.append(mse(df, RP.full(df, bO + cp * dp)))
            PMSE.setdefault(f"{pre}oracle_perm", {})[f] = op
        put("q_aware", f, RP.full(df, ua))
        put("q_blind", f, RP.full(df, ua + 1.0 * d22))
        put("q_sham", f, RP.full(df, us))
        rec["folds"][f] = fr
    for f in C.FOLDS:
        df = dfs[f]
        fr = rec["folds"][f]
        tabs = {n: RP.table(df, U[n][f]) for n in NAMES}       # asserts v22 bytes outside P
        fr["predictors"] = tabs if full_tables else {
            n: {k: t[k] for k in ("mse_gain_s2_per_row", "ci95_day", "rmse_gain_s",
                                  "rmse_gain_clean_s", "rmse_new", "rows_changed",
                                  "rows_below_60s")} for n, t in tabs.items()}
        fr["paired"] = {f"{b}_minus_{a}": RP.paired(df, U[a][f], U[b][f]) for a, b in PAIRS}
        fr["decomposition_s2_per_row"] = {
            "v22_to_q_aware": tabs["q_aware"]["mse_gain_s2_per_row"],
            "q_aware_to_global": fr["paired"]["global_minus_q_aware"]["mse_s2_per_row"],
            "global_to_router": fr["paired"]["router_minus_global"]["mse_s2_per_row"],
            "v22_to_router": tabs["router"]["mse_gain_s2_per_row"]}
        for pre, base in (("", "q_aware"), ("sham_", "q_sham")):
            m0 = mse(df, U[base][f])
            fr[f"{pre}oracle_perm_s2_per_row_over_base"] = RP.perm_summary(
                m0 - mse(df, U[f"{pre}oracle"][f]), [m0 - x for x in PMSE[f"{pre}oracle_perm"][f]])
            m0 = mse(df, U[f"{pre}global"][f])
            fr[f"{pre}placebo_minus_global_s2_per_row"] = RP.perm_summary(
                m0 - mse(df, U[f"{pre}router"][f]), [m0 - x for x in PMSE[f"{pre}placebo"][f]])
    for n in NAMES:
        rec["board"][n] = RP.board(dfs, zero, U[n])
        rec["board"][n]["december_rmse_gain_s"] = rec["folds"]["december"]["predictors"][n]["rmse_gain_s"]
    for a, b in PAIRS:
        rec["board_paired"][f"{b}_minus_{a}"] = RP.board(dfs, U[a], U[b])
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
        rec["perm"][f"{pre}oracle_over_base_board_s"] = RP.perm_summary(real, refs)
        m0 = {f: mse(dfs[f], U[f"{pre}global"][f]) for f in C.FOLDS}
        real = NR.board_point(dfs, U[f"{pre}global"], U[f"{pre}router"])
        refs = [board_from_mse(m0, {f: PMSE[f"{pre}placebo"][f][k] for f in C.FOLDS})
                for k in range(C.N_PERM)]
        rec["perm"][f"{pre}placebo_minus_global_board_s"] = RP.perm_summary(real, refs)
    return rec, U


def reversed_(rec: dict, name: str) -> dict:
    """Q08's confirmation rule on a board paired difference."""
    S = rec["board_paired"][name]
    neg = [f for f in C.FOLDS if rec["folds"][f]["paired"][name]["mse_s2_per_row"] < 0]
    return {"board_s": S["all_row_gain_s"], "ci": S["all_row_ci"], "clean_s": S["clean_gain_s"],
            "folds_with_negative_point": neg,
            "reversed": bool(S["all_row_ci"]["ci_block_1d"][0] > 0 and not neg)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = {nl: C.EXP / f"analysis/{C.tagged('iter2', nl)}.json" for nl in (False, True)}
    OUTR = C.EXP / "analysis/iter2-readings.json"
    rows_p = {(f, nl): C.EXP / f"work/{C.tagged('iter2', nl)}-rows-{f}.parquet"
              for f in C.FOLDS for nl in (False, True)}
    oof_p = {f: C.EXP / f"oof/iter2-{f}.parquet" for f in C.FOLDS}
    NO.refuse_existing([*OUT.values(), OUTR, *rows_p.values(), *oof_p.values()], args.overwrite)
    tj = C.EXP / "analysis/trunc.json"
    trunc_rec = json.loads(tj.read_text())
    ex = json.loads((C.EXP / "analysis/trunc-exactness.json").read_text())
    if not ex["passes"]:
        raise SystemExit("SPEC 2.2 (b) did not pass; no iteration-2 score may be read")
    if trunc_rec["G"] != G:
        raise SystemExit("the truncation grid is not the registered one")
    iter1 = {nl: json.loads((C.EXP / f"analysis/{C.tagged('iter1', nl)}.json").read_text())
             for nl in (False, True)}
    t0 = time.time()
    RES = {}
    for nl in (False, True):
        tag = "NULL" if nl else "real"
        FD = {f: RP.load(f, False, None, nl) for f in C.FOLDS}
        TR = {f: load_trunc(f, nl, trunc_rec) for f in C.FOLDS}
        rec = {"what": f"exp-052 iteration 2 (SPEC 2.3-2.5), {tag} run: matched capacity "
                       f"control by truncation", "null": nl, "G": G,
               "pinned_inputs": C.check_pinned(),
               "code_sha256": {p.name: C.sha_file(p) for p in sorted(C.HERE.glob("rx_*.py"))},
               "trunc_json_sha256": C.sha_file(tj),
               "trunc_exactness_sha256": C.sha_file(C.EXP / "analysis/trunc-exactness.json"),
               "shares": NR.SHARES, "boot": NR.BOOT, "boot_seed": NR.SEED, "n_perm": C.N_PERM,
               "selection": {}, "grid": {}}
        # ---- the selected capacity (the one predictor per fold)
        FS = {}
        for f in C.FOLDS:
            ks, losses = select_k(FD[f], TR[f])
            FS[f] = with_values(FD[f], TR[f], ks)
            FS[f]["k_by_block"] = ks
            rec["selection"][f] = {"k_by_block": ks, "dev_loss_by_block": losses}
            print(f"{C.utc()} {tag} {f}: k* by block {ks}", flush=True)
        sel, U = price(FS, full_tables=True)
        rec["selected"] = sel
        # ---- the fixed grid, report only
        for k in G:
            FG = {}
            for f in C.FOLDS:
                FG[f] = with_values(FD[f], TR[f], [k] * C.K)
                FG[f]["k_by_block"] = [k] * C.K
            g, _ = price(FG, full_tables=False)
            rec["grid"][str(k)] = g
            print(f"{C.utc()} {tag} k={k:3d}: q_aware {g['board']['q_aware']['all_row_gain_s']:+.5f}"
                  f"  router {g['board']['router']['all_row_gain_s']:+.5f}"
                  f"  S {g['board_paired']['router_minus_global']['all_row_gain_s']:+.5f}"
                  f"  oracle/q_aware {g['board_paired']['oracle_minus_q_aware']['all_row_gain_s']:+.5f}",
                  flush=True)
        # k = 400 must be iteration 1, exactly
        g4, i1 = rec["grid"]["400"], iter1[nl]
        for n in NAMES:
            a, b = g4["board"][n]["all_row_gain_s"], i1["board"][n]["all_row_gain_s"]
            if abs(a - b) > 1e-12:
                raise SystemExit(f"{tag}: grid k = 400 {n} {a!r} is not iteration 1's {b!r}")
        for p, v in i1["board_paired"].items():
            if abs(g4["board_paired"][p]["all_row_gain_s"] - v["all_row_gain_s"]) > 1e-12:
                raise SystemExit(f"{tag}: grid k = 400 {p} is not iteration 1's")
        rec["grid_k400_reproduces_iteration_1"] = True
        # ---- per-row files
        for f in C.FOLDS:
            df = FS[f]["df"]
            rows = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt, "fold": f, "y_true": df.y,
                                 "p_v22": df.p_v22, "pop": df["pop"], "block": df.block,
                                 "cell": df.cell})
            for n in NAMES:
                rows[f"u_{n}"] = U[n][f]
            NO.refuse_existing([rows_p[(f, nl)]], args.overwrite)
            rows.to_parquet(rows_p[(f, nl)], index=False)
            sel["folds"][f]["rows_file"] = str(rows_p[(f, nl)].relative_to(C.ROOT))
            sel["folds"][f]["rows_sha256"] = C.sha_file(rows_p[(f, nl)])
            if nl:
                continue
            slim = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt, "fold": f, "y_true": df.y,
                                 "y_pred": df.p_v22 + U["router"][f],
                                 "y_pred_global": df.p_v22 + U["global"][f],
                                 "y_pred_q_aware": df.p_v22 + U["q_aware"][f]})
            NO.refuse_existing([oof_p[f]], args.overwrite)
            slim.to_parquet(oof_p[f], index=False)
            sel["folds"][f]["oof"] = str(oof_p[f].relative_to(C.ROOT))
            sel["folds"][f]["oof_sha256"] = C.sha_file(oof_p[f])
        rec["seconds"] = round(time.time() - t0, 1)
        NO.refuse_existing([OUT[nl]], args.overwrite)
        OUT[nl].write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
        RES[nl] = rec
        print(f"written {OUT[nl]} ({rec['seconds']:.0f}s)", flush=True)

    # ---- the registered readings (SPEC 2.5)
    real, null = RES[False]["selected"], RES[True]["selected"]
    S, Sn = reversed_(real, "router_minus_global"), reversed_(null, "router_minus_global")
    Ss = reversed_(real, "sham_router_minus_sham_global")
    if S["reversed"] and not Sn["reversed"] and not Ss["reversed"]:
        primary = "bad contrast: the 400-round contrast hid a routable NM-reliability signal"
    elif not S["reversed"]:
        primary = "missing destination for this router and these inputs"
    else:
        primary = "not separable from noise"
    qa = real["board"]["q_aware"]
    qa_neg = [f for f in C.FOLDS if real["folds"][f]["predictors"]["q_aware"]["mse_gain_s2_per_row"] < 0]
    rank = {f: {"real_beats_all_50": real["folds"][f]["oracle_perm_s2_per_row_over_base"]["real_exceeds_all"],
                "null_beats_all_50": null["folds"][f]["oracle_perm_s2_per_row_over_base"]["real_exceeds_all"]}
            for f in C.FOLDS}
    rd = {"inputs": {("iter2-null" if nl else "iter2"): {"path": str(OUT[nl].relative_to(C.ROOT)),
                                                        "sha256": C.sha_file(OUT[nl])}
                     for nl in (False, True)},
          "k_star": {"real": {f: RES[False]["selection"][f]["k_by_block"] for f in C.FOLDS},
                     "null": {f: RES[True]["selection"][f]["k_by_block"] for f in C.FOLDS}},
          "primary": {"S_star_real": S, "S_star_null": Sn, "S_star_sham_real_run": Ss,
                      "reading": primary},
          "Q3_star": {"q_aware_star_board": qa, "q_blind_star_board": real["board"]["q_blind"],
                      "global_star_board": real["board"]["global"],
                      "router_star_board": real["board"]["router"],
                      "null_q_aware_star_board": null["board"]["q_aware"],
                      "q_aware_folds_with_negative_point": qa_neg,
                      "capacity_repairs_the_endpoints": bool(qa["all_row_ci"]["ci_block_1d"][1] >= 0),
                      "residual_refit_gains_on_v22": bool(qa["all_row_ci"]["ci_block_1d"][0] > 0
                                                          and not qa_neg)},
          "Q1_star": {"oracle_over_q_aware_real_s": real["board_paired"]["oracle_minus_q_aware"]["all_row_gain_s"],
                      "oracle_over_q_aware_null_s": null["board_paired"]["oracle_minus_q_aware"]["all_row_gain_s"],
                      "rank_tests": rank,
                      "above_noise_by_fold_amended": {f: bool(v["real_beats_all_50"] and not v["null_beats_all_50"])
                                                      for f, v in rank.items()},
                      "corr_d22_r_minus_u_aware": {
                          f: {"real": real["folds"][f]["d22"]["corr_with_r_minus_u_aware"],
                              "null": null["folds"][f]["d22"]["corr_with_r_minus_u_aware"]}
                          for f in C.FOLDS}},
          "nm_value_star": {"q_aware_minus_q_sham_real_s": real["board_paired"]["q_aware_minus_q_sham"]["all_row_gain_s"],
                            "q_aware_minus_q_sham_real_ci": real["board_paired"]["q_aware_minus_q_sham"]["all_row_ci"],
                            "q_aware_minus_q_sham_null_s": null["board_paired"]["q_aware_minus_q_sham"]["all_row_gain_s"]}}
    NO.refuse_existing([OUTR], args.overwrite)
    OUTR.write_text(json.dumps(rd, indent=2, sort_keys=True, default=float))
    print(json.dumps(rd, indent=2, default=float))


if __name__ == "__main__":
    main()
