"""Step 2: the 388 read through the chain, label-free.

The bound is `g - q`, with `q` the low quantile of `u = g - y` measured on the
2025 departures of the SAME AIRPORT.  Two variants: `g_any` (the last arrival
before take-off) and `g_prev2` (the one before that, which survives a stand
re-occupied while the aircraft taxied).  `q` is quoted at three levels so the
reading does not depend on one choice.

Nothing here reads a 2026 label; there is none.  The 2025 quantiles are the
only labelled input and they are a calibration of the instrument, not a fit to
the rows being read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import LAB  # noqa: E402

V9 = ROOT / "submissions" / "candidates" / "exp-027-k2.parquet"
LEVELS = (0.001, 0.01, 0.05)
VARIANTS = ("g_any", "g_prev2")


def margins(a: pd.DataFrame) -> pd.DataFrame:
    """Per airport and variant, the low quantiles of `u = g - y` on all 2025 departures."""
    rows = []
    for apt, d in a.groupby("apt"):
        for var in VARIANTS:
            u = (d[var] - d.y_true).to_numpy(float)
            u = u[np.isfinite(u)]
            r = {"apt": apt, "variant": var, "n": len(u)}
            for lv in LEVELS:
                r[f"q{lv:g}"] = float(np.quantile(u, lv))
            r["viol_at_q0"] = float((u < 0).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet")
    b = pd.read_parquet(EXP / "work" / "dep_2026.parquet")
    m = margins(a)
    m.to_csv(EXP / "analysis" / "step2_margins.csv", index=False)
    print(m.round(1).to_string(index=False))

    v9 = pd.read_parquet(V9)
    n = b[b.noac == 1].copy()
    n["pred"] = v9.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(n.MVT_ID_mvt.to_numpy()).to_numpy()
    assert n.pred.notna().all() and len(n) == 388

    key = m.set_index(["apt", "variant"])
    for var in VARIANTS:
        for lv in LEVELS:
            q = np.array([key.loc[(ap, var), f"q{lv:g}"] for ap in n.apt])
            n[f"bound_{var}_{lv:g}"] = n[var] - q
            n[f"over_{var}_{lv:g}"] = n.pred > n[f"bound_{var}_{lv:g}"]

    rows = []
    for lab in LAB + ["ALL"]:
        d = n if lab == "ALL" else n[n.band == lab]
        if not len(d):
            continue
        r = {"band": lab, "n": len(d), "pred_med": d.pred.median(),
             "ms_med": d.mvt_sched.median(),
             "g_any_med": d.g_any.median(), "g_prev2_med": d.g_prev2.median(),
             "chained_any": float(d.g_any.notna().mean()),
             "chained_prev2": float(d.g_prev2.notna().mean())}
        for var in VARIANTS:
            for lv in LEVELS:
                c = d[f"over_{var}_{lv:g}"] & d[var].notna()
                r[f"over_{var}_{lv:g}"] = int(c.sum())
                ex = (d.pred - d[f"bound_{var}_{lv:g}"])[c]
                r[f"excess_med_{var}_{lv:g}"] = float(ex.median()) if len(ex) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "step2_the_388.csv", index=False)
    n.to_parquet(EXP / "work" / "the_388_chained.parquet")
    with pd.option_context("display.width", 250):
        print()
        print(t[["band", "n", "pred_med", "ms_med", "g_any_med", "g_prev2_med",
                 "chained_any", "chained_prev2",
                 "over_g_any_0.001", "over_g_any_0.01", "over_g_prev2_0.001",
                 "over_g_prev2_0.01", "excess_med_g_prev2_0.001"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
