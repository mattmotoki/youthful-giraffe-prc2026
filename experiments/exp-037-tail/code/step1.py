"""Step 1: does the chain exist, how wide is the gap, and how pure is it?

`u = g - y = B_off - B_in` is the stand's occupancy time for the departing
aircraft IF the chain is the same aircraft.  A pure chain cannot make `u < 0`:
two aircraft cannot hold one stand at once.  So `P(u < 0)` is the chain's
impurity, measured on the labelled 2025 departures, and the low quantiles of
`u` are what a bound has to be fitted on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import LAB  # noqa: E402

QS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
VARIANTS = ("g_any", "g_strict", "g_prev2")


def rows(d: pd.DataFrame, scope: str, var: str, band: str) -> dict:
    g = d[var].to_numpy(float)
    y = d.y_true.to_numpy(float)
    u = g - y
    m = np.isfinite(u)
    r = {"scope": scope, "variant": var, "band": band, "n": len(d),
         "chained": float(np.isfinite(g).mean()) if len(d) else np.nan,
         "n_chained": int(m.sum())}
    if m.sum() >= 5:
        r["viol_rate"] = float((u[m] < 0).mean())
        for q, v in zip(QS, np.quantile(u[m], QS)):
            r[f"u_q{q:g}"] = float(v)
        r["g_med_h"] = float(np.median(g[np.isfinite(g)]) / 3600.0)
    return r


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet")
    out = []
    for scope, d0 in (("all", a), ("noac", a[a.noac == 1])):
        for var in VARIANTS:
            out.append(rows(d0, scope, var, "ALL"))
            for lab in LAB:
                out.append(rows(d0[d0.band == lab], scope, var, lab))
    t = pd.DataFrame(out)
    t.to_csv(EXP / "analysis" / "step1_purity.csv", index=False)
    cols = ["scope", "variant", "band", "n", "chained", "n_chained", "g_med_h",
            "viol_rate", "u_q0.001", "u_q0.01", "u_q0.05", "u_q0.5"]
    with pd.option_context("display.width", 200):
        print(t[cols].round(4).to_string(index=False))

    # per-airport purity of the primary variant, all rows
    ap = []
    for apt, d in a.groupby("apt"):
        ap.append(rows(d, apt, "g_any", "ALL"))
    pd.DataFrame(ap).to_csv(EXP / "analysis" / "step1_by_airport.csv", index=False)
    print()
    print(pd.DataFrame(ap)[["scope", "n", "chained", "viol_rate", "g_med_h",
                            "u_q0.001", "u_q0.01"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
