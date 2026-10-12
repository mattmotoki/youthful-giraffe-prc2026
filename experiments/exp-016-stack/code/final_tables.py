"""The two tables review round 2 found missing for the SHIPPED configurations.

N5: the assignment asks for a day-clustered bootstrap "against blend-2 AND
against the flat four-member average". Both were produced for the round-1 and
round-2 arms and for neither shipped configuration.
N6: COMMON.md requires the <=3600 / >3600 / no-NM decomposition on every RESULTS
table, and only one table carried it.
N7: the OOF deliverable had no producing code; this file writes it, so it is
reproducible, and reads back the columns the other tables are built from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import verify_r1 as V  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402

OUT = M.EXP / "analysis"
COLS = {"stack-1": "y_pred", "stack-2": "y_pred_a2_keep_noac",
        "a5/full": "y_pred_a5_full", "a2/full": "y_pred_a2_full",
        "blend2": "y_pred_blend2", "flat4": "y_pred_flat4"}


def write_oof() -> None:
    """Regenerate oof/exp-016-oof-{fold}.parquet, the COMMON.md deliverable."""
    for f in M.FOLDS:
        d = M.fold_frame(f)
        fit = pd.concat([M.fold_frame(g) for g in M.FOLDS if g != f], ignore_index=True)
        y = d["y_true"].to_numpy("float64")
        ref = S.fit_blend2(fit, d)
        flat = S.flat4(d)
        reg = d["regime"].to_numpy()
        bags = {}
        for lab, cols in (("a5", V.NOGATE), ("a2", V.ALLC)):
            bags[lab] = np.maximum(np.mean([V.arm(fit, d, s, cols)[0] for s in V.SEEDS], axis=0),
                                   B.FLOOR)
        out = pd.DataFrame({
            "MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(), "fold": f, "y_true": y,
            "y_pred": np.where(reg == "noac", ref, bags["a5"]),
            "y_pred_a2_keep_noac": np.where(reg == "noac", ref, bags["a2"]),
            "y_pred_a5_full": bags["a5"], "y_pred_a2_full": bags["a2"],
            "y_pred_blend2": ref, "y_pred_flat4": flat,
            "regime": reg, "airport": d["airport"].astype(str).to_numpy(),
            "noac": d["noac"].to_numpy(), "nm_missing": d["nm_missing"].to_numpy(),
            "t": d["t"].to_numpy("float64")})
        out.to_parquet(M.EXP / "oof" / f"exp-016-oof{M.TAG}-{f}.parquet", index=False)
        print(f"[oof] {f} {out.shape} y_pred rmse {S.rmse(y, out.y_pred):.3f}", flush=True)


def main() -> None:
    if "--rebuild-oof" in sys.argv:
        write_oof()
    recs = []
    for f in M.FOLDS:
        d = pd.read_parquet(M.EXP / "oof" / f"exp-016-oof{M.TAG}-{f}.parquet")
        y = d["y_true"].to_numpy("float64")
        cl = pd.factorize(d["airport"].astype(str) + "|" +
                          np.floor(d["t"].to_numpy("float64") / 86400).astype("int64").astype(str))[0]
        for lab, col in COLS.items():
            p = d[col].to_numpy("float64")
            r = dict(fold=f, variant=lab, **V.decomp(y, p, d))
            for ref_lab in ("blend2", "flat4"):
                q = d[COLS[ref_lab]].to_numpy("float64")
                dd, lo, hi = S.boot_delta(y, p, q, cl)
                r[f"vs_{ref_lab}"], r[f"{ref_lab}_lo"], r[f"{ref_lab}_hi"] = dd, lo, hi
            recs.append(r)
    T = pd.DataFrame(recs)
    T.to_csv(OUT / f"final_tables{M.TAG}.csv", index=False)
    print("\n=== decomposition, every configuration, every fold ===")
    print(T[["fold", "variant", "rmse", "le3600", "gt3600", "no_nm", "noac"]]
          .round(2).to_string(index=False))
    print("\n=== day-clustered 95 %, against BOTH baselines (the assignment asked for both) ===")
    for r in T[~T.variant.isin(("blend2", "flat4"))].itertuples():
        print(f"  {r.fold:9s} {r.variant:9s} vs blend2 {r.vs_blend2:+7.3f} "
              f"[{r.blend2_lo:+7.3f}, {r.blend2_hi:+7.3f}]   "
              f"vs flat4 {r.vs_flat4:+7.3f} [{r.flat4_lo:+7.3f}, {r.flat4_hi:+7.3f}]")


if __name__ == "__main__":
    M.config_from_argv()
    main()
