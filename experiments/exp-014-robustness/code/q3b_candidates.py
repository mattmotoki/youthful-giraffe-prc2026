"""Q3b: the same file checks on BOTH shipped candidates, and the January re-mix.

Candidate 2 landed while this audit was running. Everything question 3 asks is
re-run on it beside candidate 1, and the two are differenced row by row so the
desk can see where the 0.79 s of board estimate comes from and what it costs.

The re-mix: the january FOLD carries 0.73 % of its rows on days past their
airport's training envelope; January 2026 carries 4.95 %. Holding the blend's
measured per-slice RMSE fixed and changing only that share is the cheapest
honest statement of how much worse the January half should be than the fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = F.EXP / "analysis"
CANDS = {"candidate 1": "submissions/candidates/exp-012-blend-1.parquet",
         "candidate 2": "submissions/candidates/exp-012-blend-2.parquet"}
BIG = 20000.0
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}


def main() -> None:
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AOBT_3_flt", "LOBT_flt", "AIRCRAFT_TYPE_mvt",
                                 "FLIGHT_ID_mvt", "STAND_mvt"])
    dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    dep["regime"] = B.regimes_from_raw(dep, B.LATE_CUT)
    dep["obs"] = (dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]).dt.total_seconds()
    dep["apt"] = dep["ADEP_mvt"].astype(str)
    dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.floor("D")
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    ids = dep["MVT_ID_mvt"].to_numpy("float64")

    rows = []
    for name, path in CANDS.items():
        p = F.candidate(path).reindex(ids).to_numpy("float64")
        dep[name] = p
        rows.append(dict(candidate=name, sha256=D.sha256(F.ROOT / path)[:12],
                         n=len(p), n_nan=int((~np.isfinite(p)).sum()),
                         n_negative=int((p < 0).sum()), n_below_floor=int((p < 60.0 - 1e-9).sum()),
                         min=float(p.min()), max=float(p.max()),
                         median=float(np.median(p)), p99=float(np.quantile(p, 0.99)),
                         n_above_20000=int((p > BIG).sum()),
                         mean_jan=float(p[(dep["month"] == "2026-01").to_numpy()].mean()),
                         mean_jul=float(p[(dep["month"] == "2026-07").to_numpy()].mean())))
    Chk = pd.DataFrame(rows)
    Chk.to_csv(OUT / "q3b_candidate_checks.csv", index=False)
    print("=== file checks, both candidates ===")
    print(Chk.round(2).to_string(index=False), flush=True)

    dep["d21"] = dep["candidate 2"] - dep["candidate 1"]
    print(f"\ncandidate 2 - candidate 1: rms {float(np.sqrt((dep['d21']**2).mean())):.2f} s, "
          f"max {float(dep['d21'].max()):.1f}, min {float(dep['d21'].min()):.1f}, "
          f"{int((dep['d21'] != 0).sum())} of {len(dep)} rows differ", flush=True)
    by = dep.groupby(["month", "regime"], observed=True).agg(
        n=("d21", "size"), c1=("candidate 1", "mean"), c2=("candidate 2", "mean"),
        d_mean=("d21", "mean"), d_rms=("d21", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index()
    by.to_csv(OUT / "q3b_candidate_delta_by_regime.csv", index=False)
    print("\n=== where the two candidates differ ===")
    print(by.round(1).to_string(index=False), flush=True)

    # on the 18 exposed January days and on EHAM's month
    E = pd.read_csv(OUT / "q2b_exposed_days_2026.csv", parse_dates=["day"])
    key = set(zip(E["airport"], E["day"]))
    dep["exposed"] = [(a, d) in key for a, d in zip(dep["apt"], dep["day"])]
    dep["eham_jan"] = (dep["apt"] == "EHAM") & (dep["month"] == "2026-01")
    sl = []
    for label, m in (("january 2026, all", dep["month"] == "2026-01"),
                     ("january 2026, 18 exposed days", dep["exposed"]),
                     ("EHAM january 2026", dep["eham_jan"]),
                     ("july 2026, all", dep["month"] == "2026-07")):
        g = dep[m]
        sl.append(dict(slice=label, n=len(g), c1=float(g["candidate 1"].mean()),
                       c2=float(g["candidate 2"].mean()), d_mean=float(g["d21"].mean()),
                       obs=float(g["obs"].mean()),
                       r_hat_c1=float((g["obs"] - g["candidate 1"]).mean()),
                       r_hat_c2=float((g["obs"] - g["candidate 2"]).mean())))
    S = pd.DataFrame(sl)
    S.to_csv(OUT / "q3b_candidates_on_exposed.csv", index=False)
    print("\n=== the two candidates on the exposed slices ===")
    print(S.round(1).to_string(index=False), flush=True)

    # ------------------------------------------------ the January re-mix ----
    FD = pd.read_csv(OUT / "q2b_fold_days.csv", parse_dates=["day"])
    jan = FD[FD["fold"] == "january"]
    ex = jan[jan["over_tr_max"] > 0]
    n_ex, n_all = int(ex["n"].sum()), int(jan["n"].sum())
    rmse_ex = float(np.sqrt(np.average(ex["rmse"] ** 2, weights=ex["n"])))
    T26 = pd.read_csv(OUT / "q2_days_2026.csv", parse_dates=["day"])
    j26 = T26[T26["month"] == "2026-01"]
    n26_ex = int(j26[j26["over_max"] > 0]["n"].sum())
    n26 = int(j26["n"].sum())
    print(f"\n=== the January re-mix ===")
    print(f"january fold: {n_ex} of {n_all} rows ({100*n_ex/n_all:.2f} %) on days past the "
          f"airport's training envelope, RMSE {rmse_ex:.0f} s there")
    print(f"January 2026: {n26_ex} of {n26} rows ({100*n26_ex/n26:.2f} %) on such days")
    out = []
    for cand, jan_rmse in (("candidate 1", 343.2647), ("candidate 2", 343.7263)):
        rest = float(np.sqrt((n_all * jan_rmse ** 2 - n_ex * rmse_ex ** 2) / (n_all - n_ex)))
        f = n26_ex / n26
        remix = float(np.sqrt((1 - f) * rest ** 2 + f * rmse_ex ** 2))
        jl = {"candidate 1": 297.9491, "candidate 2": 295.9916}[cand]
        board0 = float(np.sqrt(SHARES["july_loo"] * jl ** 2 + SHARES["january"] * jan_rmse ** 2))
        board1 = float(np.sqrt(SHARES["july_loo"] * jl ** 2 + SHARES["january"] * remix ** 2))
        out.append(dict(candidate=cand, january_fold=jan_rmse, rmse_off_exposed=rest,
                        rmse_on_exposed=rmse_ex, january_remixed=remix,
                        board_reported=board0, board_remixed=board1, move=board1 - board0))
    Rm = pd.DataFrame(out)
    Rm.to_csv(OUT / "q3b_january_remix.csv", index=False)
    print(Rm.round(2).to_string(index=False), flush=True)
    print("\nThis holds the per-slice RMSE fixed and changes only the share of rows on\n"
          "exposed days. It is a LOWER BOUND: January 2026's exposed days clear their\n"
          "airport's maximum by up to 1,350 s, against 73 s for the worst 2025 day the\n"
          "folds contain, so the slice RMSE itself should be worse than 2025's.", flush=True)
    json.dump({"n_ex": n_ex, "n_all": n_all, "rmse_ex": rmse_ex,
               "n26_ex": n26_ex, "n26": n26}, open(OUT / "q3b_remix_inputs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
