"""Q01 -- rebase every staged vector since stack-5 on the submitted predictor.

    PRC_THREADS=4 python q01.py

Writes the aligned packet (`packet/`), the one-row-per-candidate table
(`analysis/q01-table.csv`), the per-date paired SSE (`analysis/q01-dates.csv`),
the composition-risk estimate (`analysis/q01-composition.csv`) and the
provenance/hash record (`analysis/q01-provenance.json`).

The join is one-to-one on `MVT_ID_mvt` in the reference panel's own order, and
`code/vectors.py` refuses any vector that is not in that order, so no candidate
is compared on a different row set.  The identity row -- v10 against itself --
is kept in the table on purpose: it must read `n_changed = 0`, `A = B = 0` and a
gain of exactly zero, and if it does not, the arithmetic below is wrong.
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
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402
import vectors as V  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
BASE = "v10 (exp-036 refitbulk-mask)"
#: The fold whose composition each ranking month is priced against.  December
#: has no ranking counterpart and is excluded from the risk estimate by name.
RANKING_OF = {"july_loo": "2026-07", "january": "2026-01"}
GATES = "experiments/exp-026-rebuild/candidates/ranking-gates.parquet"
DRAWS = 2000


def panel() -> dict:
    ref, vec, prov, files = V.load()
    out = {}
    for f in FOLDS:
        d = ref[f]
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        day = A.day_of(d["t"].to_numpy("float64"))
        out[f] = {
            "ids": ids, "y": d["y_true"].to_numpy("float64"),
            "airport": d["airport"].astype(str).to_numpy(),
            "regime": d["regime"].astype(str).to_numpy(),
            "day": day,
            "clean": ~d["MVT_ID_mvt"].isin(V.D.excluded_ids(f)).to_numpy(),
            "blk1": A.blocks(day, 1), "blk3": A.blocks(day, 3), "blk7": A.blocks(day, 7),
        }
    return ref, vec, prov, files, out


def composition(P: dict) -> dict:
    """Per-fold row weights that carry the fold to its ranking month's airport x regime mix.

    Vectors fixed, weights fixed: this is a risk estimate under a different
    composition, not a fit.  `unsupported` is the share of the ranking month's
    rows whose (airport, regime) cell has NO fold row behind it -- the part of
    the target the folds cannot price at all.
    """
    g = pd.read_parquet(ROOT / GATES)
    out = {}
    for f, month in RANKING_OF.items():
        r = g[g["month"] == month]
        tgt = (r.groupby([r["airport"].astype(str), r["regime"].astype(str)])
                .size().rename("n_rank"))
        cell = pd.Series(list(zip(P[f]["airport"], P[f]["regime"])))
        src = cell.value_counts().rename("n_fold")
        tgt.index = tgt.index.map(tuple)
        j = pd.concat([tgt, src], axis=1)
        supported = j["n_fold"].notna() & (j["n_fold"] > 0)
        unsupported_mass = float(j.loc[~supported, "n_rank"].fillna(0).sum() / tgt.sum())
        share_t = j["n_rank"].fillna(0.0) / float(tgt.sum())
        share_s = j["n_fold"].fillna(0.0) / float(len(cell))
        w_cell = (share_t / share_s.replace(0.0, np.nan)).fillna(0.0)
        w = cell.map(w_cell).fillna(0.0).to_numpy("float64")
        # rows in cells the ranking month never visits get weight 0; renormalise
        # so the weighted count equals the fold's own row count.
        s = w.sum()
        out[f] = {"w": w * (len(w) / s) if s > 0 else w,
                  "unsupported_target_mass": unsupported_mass,
                  "zero_weight_rows": int((w == 0).sum()),
                  "n_target_cells": int(len(tgt)),
                  "n_unsupported_cells": int((~supported).sum())}
    return out


def main() -> None:
    t0 = time.time()
    ref, vec, prov, files, P = panel()
    comp = composition(P)
    p10 = vec[BASE]
    names = list(vec)
    print(f"[q01] {len(names)} vectors, baseline {BASE}", flush=True)

    rows, dates, cols = [], [], {}
    dcols = {f: {} for f in FOLDS}
    for i, name in enumerate(names):
        cand = vec[name]
        st_all, st_cl, parts_all, parts_cl, parts_w = {}, {}, {}, {}, {}
        dec = {}
        for f in FOLDS:
            p = P[f]
            pr = A.per_row(p["y"], p10[f], cand[f])
            r, d = pr["r"], pr["d"]
            dcols[f][f"d{i:02d}"] = d
            P[f]["_r"], P[f]["_d"] = r, d
            allrows = np.ones(len(d), bool)
            st_all[f] = A.fold_stats(r, d, allrows)
            st_cl[f] = A.fold_stats(r, d, p["clean"])
            e10 = r * r
            ec = (r - d) ** 2
            parts_all[f] = (p["blk1"], e10, ec)
            parts_cl[f] = (p["blk1"][p["clean"]], e10[p["clean"]], ec[p["clean"]])
            # composition-reweighted risk, vectors fixed
            if f in comp:
                w = comp[f]["w"]
                parts_w[f] = {"mse_v10": float(np.average(e10, weights=w)),
                              "mse_cand": float(np.average(ec, weights=w))}
            # per-date paired SSE
            if float(np.abs(d).max()) > 0.0:
                t = pd.DataFrame({"day": p["day"], "e10": e10, "ec": ec, "d": d})
                gg = t.groupby("day").agg(n=("d", "size"), n_changed=("d", lambda s: int((s != 0).sum())),
                                          sse_v10=("e10", "sum"), sse_cand=("ec", "sum"))
                gg = gg.reset_index()
                gg.insert(0, "fold", f)
                gg.insert(0, "candidate", name)
                gg["dsse"] = gg["sse_cand"] - gg["sse_v10"]
                dates.append(gg)
        # -- bounded amplitudes fitted away from the rows they are scored on --
        loo = A.scale_loo({f: st_all[f]["A"] for f in FOLDS},
                          {f: st_all[f]["B"] for f in FOLDS},
                          {f: st_all[f]["n"] for f in FOLDS})
        loo_c = A.scale_loo({f: st_cl[f]["A"] for f in FOLDS},
                            {f: st_cl[f]["B"] for f in FOLDS},
                            {f: st_cl[f]["n"] for f in FOLDS})
        cv = {f: A.scale_cv(P[f]["_r"], P[f]["_d"], P[f]["blk3"],
                            np.ones(len(P[f]["day"]), bool)) for f in FOLDS}
        cv_c = {f: A.scale_cv(P[f]["_r"], P[f]["_d"], P[f]["blk3"], P[f]["clean"])
                for f in FOLDS}
        # the amplitude is fixed by the fit above; the interval resamples the
        # DAYS it is scored on, so it is an interval on the scored gain
        parts_loo = {}
        for f in A.BOARD_FOLDS:
            r_, d_, sf = P[f]["_r"], P[f]["_d"], loo[f]["s"]
            parts_loo[f] = (P[f]["blk1"], r_ * r_, (r_ - sf * d_) ** 2)
        ci_loo = A.boot_board(parts_loo, DRAWS)
        b_all = A.board_gain(st_all)
        b_cl = A.board_gain(st_cl)
        ci1 = A.boot_board(parts_all, DRAWS)
        ci3 = A.boot_board({f: (P[f]["blk3"], parts_all[f][1], parts_all[f][2])
                            for f in A.BOARD_FOLDS}, DRAWS)
        ci7 = A.boot_board({f: (P[f]["blk7"], parts_all[f][1], parts_all[f][2])
                            for f in A.BOARD_FOLDS}, DRAWS)
        ci_cl = A.boot_board(parts_cl, DRAWS)
        # composition-reweighted board gain
        if parts_w:
            bw10 = np.sqrt(sum(A.SHARES[f] * parts_w[f]["mse_v10"] for f in A.BOARD_FOLDS))
            bwc = np.sqrt(sum(A.SHARES[f] * parts_w[f]["mse_cand"] for f in A.BOARD_FOLDS))
        else:
            bw10 = bwc = float("nan")
        rec = {
            "candidate": name,
            "family": prov[name]["source"].split(" ")[0],
            "gated": prov[name]["gated"],
            "clean_board_own": prov[name]["clean_board"],
            "n_changed_july": st_all["july_loo"]["n_changed"],
            "n_changed_dec": st_all["december"]["n_changed"],
            "n_changed_jan": st_all["january"]["n_changed"],
            "max_abs_d": max(st_all[f]["max_abs_d"] for f in FOLDS),
            # -- the primary endpoint: ALL-ROW Jan/July at the ranking weights --
            "gain_board_allrow": b_all["gain_board"],
            "ci_lo_allrow": ci1[0], "ci_hi_allrow": ci1[1],
            "ci_lo_3day": ci3[0], "ci_hi_3day": ci3[1],
            "ci_lo_7day": ci7[0], "ci_hi_7day": ci7[1],
            "gain_mse_allrow": b_all["gain_mse"],
            "gain_board_allrow_fitted": b_all["gain_board_fitted"],
            "gain_board_loo_amp": float(
                np.sqrt(sum(A.SHARES[f] * st_all[f]["mse_v10"] for f in A.BOARD_FOLDS))
                - np.sqrt(sum(A.SHARES[f] * (st_all[f]["mse_v10"] - loo[f]["gain_mse"])
                              for f in A.BOARD_FOLDS))),
            "gain_board_cv_amp": float(
                np.sqrt(sum(A.SHARES[f] * st_all[f]["mse_v10"] for f in A.BOARD_FOLDS))
                - np.sqrt(sum(A.SHARES[f] * (st_all[f]["mse_v10"] - cv[f]["gain_mse"])
                              for f in A.BOARD_FOLDS))),
            "ci_lo_loo_amp": ci_loo[0], "ci_hi_loo_amp": ci_loo[1],
            "gain_board_loo_amp_clean": float(
                np.sqrt(sum(A.SHARES[f] * st_cl[f]["mse_v10"] for f in A.BOARD_FOLDS))
                - np.sqrt(sum(A.SHARES[f] * (st_cl[f]["mse_v10"] - loo_c[f]["gain_mse"])
                              for f in A.BOARD_FOLDS))),
            "gain_board_cv_amp_clean": float(
                np.sqrt(sum(A.SHARES[f] * st_cl[f]["mse_v10"] for f in A.BOARD_FOLDS))
                - np.sqrt(sum(A.SHARES[f] * (st_cl[f]["mse_v10"] - cv_c[f]["gain_mse"])
                              for f in A.BOARD_FOLDS))),
            "gain_rmse_december_loo_amp": float(
                np.sqrt(st_all["december"]["mse_v10"])
                - np.sqrt(max(st_all["december"]["mse_v10"] - loo["december"]["gain_mse"], 0.0))),
            "d_max_on_eham": max(float(np.abs(P[f]["_d"][P[f]["airport"] == "EHAM"]).max())
                                 for f in FOLDS),
            "d_max_on_noac": max(float(np.abs(P[f]["_d"][P[f]["regime"] == "noac"]).max())
                                 for f in FOLDS),
            **{f"amp_loo_{f}": loo[f]["s"] for f in FOLDS},
            **{f"amp_cv_{f}": f"{cv[f]['s'][0]:.4f}/{cv[f]['s'][1]:.4f}" for f in FOLDS},
            # -- clean, the convention the LEDGER quotes --
            "gain_board_clean": b_cl["gain_board"],
            "ci_lo_clean": ci_cl[0], "ci_hi_clean": ci_cl[1],
            # -- december, alone --
            "gain_rmse_december": st_all["december"]["gain_rmse"],
            "gain_rmse_december_clean": st_cl["december"]["gain_rmse"],
            # -- composition risk, vectors fixed --
            "gain_board_rankmix": float(bw10 - bwc),
            # -- alignment algebra, per fold, all rows --
            **{f"A_{f}": st_all[f]["A"] for f in FOLDS},
            **{f"B_{f}": st_all[f]["B"] for f in FOLDS},
            **{f"gain2AB_{f}": st_all[f]["gain_mse_full"] for f in FOLDS},
            **{f"scale_{f}": st_all[f]["scale"] for f in FOLDS},
            **{f"gainrmse_{f}": st_all[f]["gain_rmse"] for f in FOLDS},
        }
        rows.append(rec)
        cols[f"d{i:02d}"] = name
        print(f"  {name:34s} all-row {rec['gain_board_allrow']:+8.4f} "
              f"[{ci1[0]:+7.4f},{ci1[1]:+7.4f}]  clean {rec['gain_board_clean']:+8.4f}  "
              f"dec {rec['gain_rmse_december']:+8.4f}  scale(jul) "
              + (f"{rec['scale_july_loo']:+6.3f}" if rec["scale_july_loo"] is not None else "  n/a")
              + f"  changed {rec['n_changed_july']:6d}", flush=True)

    T = pd.DataFrame(rows).sort_values("gain_board_allrow", ascending=False)
    (EXP / "analysis").mkdir(exist_ok=True)
    T.to_csv(EXP / "analysis" / "q01-table.csv", index=False)
    pd.concat(dates, ignore_index=True).to_csv(EXP / "analysis" / "q01-dates.csv", index=False)
    pd.DataFrame([{"fold": f, "ranking_month": RANKING_OF[f], **{k: v for k, v in c.items()
                   if k != "w"}} for f, c in comp.items()]).to_csv(
        EXP / "analysis" / "q01-composition.csv", index=False)

    (EXP / "packet").mkdir(exist_ok=True)
    man = {}
    for f in FOLDS:
        p = P[f]
        out = pd.DataFrame({"MVT_ID_mvt": p["ids"], "airport": p["airport"],
                            "regime": p["regime"], "day": p["day"],
                            "y_true": p["y"], "p_v10": p10[f],
                            "r": p["y"] - p10[f], "clean": p["clean"]})
        for k, v in dcols[f].items():
            out[k] = v
            # the contract asks for the per-row PAIRED SSE by name, not only for
            # the r and d it is derivable from (Sol round 1, blocking 12)
            out["s" + k[1:]] = v * v - 2.0 * out["r"].to_numpy("float64") * v
        path = EXP / "packet" / f"rows-{f}.parquet"
        out.to_parquet(path, index=False, compression="zstd")
        man[f] = {"path": str(path.relative_to(ROOT)), "rows": len(out),
                  "sha256": V.D.sha256(path)}
    (EXP / "packet" / "columns.json").write_text(json.dumps(
        {"baseline": BASE, "d_columns": cols,
         "sse_columns": {"s" + k[1:]: v for k, v in cols.items()},
         "sse_note": "s<NN> = d*d - 2*r*d = (y - p_cand)^2 - (y - p_v10)^2, the per-row "
                     "paired SSE contribution of candidate d<NN> on this fold's rows",
         "note": "d<NN> = p_candidate - p_v10 on this fold's rows, float64, "
                 "in the reference panel's own MVT_ID_mvt order"}, indent=2))

    prov_out = {"baseline": BASE, "files": files, "vectors": prov, "manifest": man,
                "draws": DRAWS, "wall_s": round(time.time() - t0, 1),
                "endpoint": "paired all-row Jan/July MSE at the ranking month shares "
                            f"{A.SHARES}; clean and december reported separately"}
    (EXP / "analysis" / "q01-provenance.json").write_text(json.dumps(prov_out, indent=2, default=str))
    print(f"\n[q01] {len(T)} rows written in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
