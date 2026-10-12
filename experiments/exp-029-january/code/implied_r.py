"""Is stack-5 mis-extrapolating on January 2026's flagged days? The label-free test.

The five 2025 analogue days top out at 73 s of exceedance and January 2026
reaches 1,350 s, so the fitted correction is being asked to extrapolate an
order of magnitude past its evidence. This file asks the extrapolation question
directly, from the served vector, without a label.

Decomposition (`exp-007/code/q1_decompose.py`):

    span  = MVT_TIME_UTC_mvt - AOBT_3_flt          label-free, in the ranking file
    r     = BLOCK_TIME_UTC_mvt - AOBT_3_flt        the block-versus-NM offset
    y     = span - r                               the target

so a prediction implies a residual `r_hat = span - pred`. `r` is an
administrative offset between the airport's block time and the Network
Manager's off-block time; it is not a congestion quantity, so a day whose taxi
time explodes should NOT move it. If stack-5's implied `r` on January 2026's
flagged days runs far outside that airport's 2025 range of daily mean `r`, the
model is handing the day's excess span to the offset instead of to taxi, which
is exp-014's finding restated as a servable test.

`BLOCK_TIME_UTC_mvt` is read from the 2025 TRAINING months only, where it is
present and legal (`docs/input-rule.md`). Nothing here reads a 2026 label:
the 2026 side is `span - pred` and the threshold comes from 2025.
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
import envelope as E  # noqa: E402
import prc.data as D  # noqa: E402

SERVED = ROOT / "submissions/candidates/exp-026-stack-5.parquet"
COLS = ["PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]


def day_r_2025() -> pd.DataFrame:
    """Per airport-day of 2025: mean true `r`, mean span. All twelve months."""
    parts = []
    for m in D.MONTHS:
        f = D.load_month(m, columns=COLS)
        f = f[f["PHASE_mvt"] == "DEP"]
        parts.append(pd.DataFrame({
            "apt": f["ADEP_mvt"].astype(str).to_numpy(),
            "day": f["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "span": (f["MVT_TIME_UTC_mvt"] - f["AOBT_3_flt"]).dt.total_seconds().to_numpy(),
            "r": (f["BLOCK_TIME_UTC_mvt"] - f["AOBT_3_flt"]).dt.total_seconds().to_numpy()}))
    t = pd.concat(parts, ignore_index=True)
    return (t.groupby(["apt", "day"], observed=True)
            .agg(n=("span", "size"), span_mean=("span", "mean"), r_mean=("r", "mean"))
            .reset_index())


def main() -> None:
    if not SERVED.exists():
        raise SystemExit(f"{SERVED} is missing; exp-026 code/serve_stack5.py writes it")
    T = day_r_2025()
    T.to_csv(EXP / "analysis" / "day_r_2025.csv", index=False)
    env = T.groupby("apt").agg(
        r_min=("r_mean", "min"), r_p05=("r_mean", lambda s: s.quantile(0.05)),
        r_med=("r_mean", "median"), r_p95=("r_mean", lambda s: s.quantile(0.95)),
        r_max=("r_mean", "max"), span_max=("span_mean", "max"))
    print("=== 2025 daily mean block-minus-AOBT offset `r`, per airport (twelve months) ===")
    print(env.round(1).to_string(), flush=True)

    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                 "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
    d = rk[rk["PHASE_mvt"] == "DEP"].copy()
    d["apt"] = d["ADEP_mvt"].astype(str)
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.floor("D")
    d["month"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    d["span"] = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    p = pd.read_parquet(SERVED)
    tgt = "TAXITIME_SEC_mvt" if "TAXITIME_SEC_mvt" in p.columns else "y_pred"
    p = p.rename(columns={tgt: "pred"})[["MVT_ID_mvt", "pred"]]
    d = d.merge(p, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if d["pred"].isna().any():
        raise SystemExit("the served vector does not cover every ranking departure")
    d["r_hat"] = d["span"] - d["pred"]

    fl = pd.read_parquet(EXP / "oof" / "ranking_flag.parquet")
    d = d.merge(fl[["MVT_ID_mvt", "exceedance", "flagged"]], on="MVT_ID_mvt",
                how="left", validate="one_to_one")
    g = (d.groupby(["month", "apt", "day"], observed=True)
         .agg(n=("pred", "size"), r_n=("r_hat", "count"),
              exceedance=("exceedance", "first"),
              span_mean=("span", "mean"), pred_mean=("pred", "mean"),
              r_hat_mean=("r_hat", "mean")).reset_index())
    g = g.merge(env, left_on="apt", right_index=True, how="left")
    g["r_over_2025_max"] = g["r_hat_mean"] - g["r_max"]
    g["r_over_2025_p95"] = g["r_hat_mean"] - g["r_p95"]
    jan = g[g["month"] == "2026-01"]
    print("\n=== January 2026: implied daily `r` against that airport's 2025 range ===")
    show = ["apt", "day", "n", "exceedance", "span_mean", "pred_mean", "r_hat_mean",
            "r_med", "r_max", "r_over_2025_max"]
    print(jan[jan["exceedance"] > 0].sort_values("exceedance", ascending=False)[show]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"), flush=True)

    # TWO-SIDED, because a one-sided range check is the exact error exp-014's
    # round-1 review caught on the day-state exposure count (its finding M8).
    g["r_under_2025_min"] = g["r_min"] - g["r_hat_mean"]
    g["r_under_2025_p05"] = g["r_p05"] - g["r_hat_mean"]
    jan = g[g["month"] == "2026-01"]
    g.to_csv(EXP / "analysis" / "implied_r_2026.csv", index=False)
    for lab, m in (("flagged", jan["exceedance"] > 0),
                   ("in envelope", jan["exceedance"] <= 0)):
        s = jan[m]
        w = s["r_n"].to_numpy("float64")
        w_all = s["n"].to_numpy("float64")
        def cnt(col):
            k = s[col] > 0
            return f"{int(k.sum())} days / {int(s.loc[k, 'n'].sum()):,} rows"
        print(f"\n{lab}: {len(s)} airport-days, {int(w_all.sum()):,} rows of which "
              f"{int(w.sum()):,} have an observed `r`, row-weighted implied r "
              f"{np.average(s['r_hat_mean'], weights=w):+.1f} s on the observed rows "
              f"({np.average(s['r_hat_mean'], weights=w_all):+.1f} s counting all rows, "
              f"which mixes two denominators -- Sol round 1, M9)")
        print(f"    ABOVE the airport's 2025 daily-mean r: max {cnt('r_over_2025_max')}, "
              f"p95 {cnt('r_over_2025_p95')}")
        print(f"    BELOW it:                              min {cnt('r_under_2025_min')}, "
              f"p05 {cnt('r_under_2025_p05')}")

    # the same question on the five 2025 analogue days, where the truth exists
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")
    R = R[R["flagged"].to_numpy()]
    rows = []
    for (f, a, dd), s in R.groupby(["fold", "apt", "day"], observed=True):
        tr = T[(T["apt"] == a) & (T["day"] == pd.Timestamp(dd))]
        rows.append(dict(fold=f, apt=a, day=str(pd.Timestamp(dd).date()), n=len(s),
                         r_true=float(tr["span_mean"].iloc[0] - tr["r_mean"].iloc[0] * 0 + 0)
                         if False else float(tr["r_mean"].iloc[0]),
                         r_hat=float((s["span"] - s["y_pred"]).mean()),
                         r_max_2025=float(env.loc[a, "r_max"]),
                         r_med_2025=float(env.loc[a, "r_med"])))
    A = pd.DataFrame(rows)
    A["r_hat_minus_true"] = A["r_hat"] - A["r_true"]
    A["r_hat_over_2025_max"] = A["r_hat"] - A["r_max_2025"]
    A.to_csv(EXP / "analysis" / "implied_r_analogue.csv", index=False)
    print("\n=== the five 2025 analogue days: implied `r` against the true `r` ===")
    print(A.round(1).to_string(index=False), flush=True)
    print("\n  (r_hat - r_true is exactly minus the bias: pred = span - r_hat, "
          "y = span - r_true)")

    (EXP / "analysis" / "implied_r.json").write_text(json.dumps({
        "served": str(SERVED.relative_to(ROOT)), "served_sha256": D.sha256(SERVED),
        "january_flagged_days_past_2025_r_max":
            int((jan.loc[jan["exceedance"] > 0, "r_over_2025_max"] > 0).sum()),
        "january_flagged_days": int((jan["exceedance"] > 0).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
