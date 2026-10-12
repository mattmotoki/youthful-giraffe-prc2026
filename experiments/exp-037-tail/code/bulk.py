"""§7: does the chain bound ORDINARY rows usefully?

Two readings, both against the same per-airport margins §2 fitted.

2025 (labelled): how often a TRUE ordinary label breaks the bound.  This is the
bound's error rate on the population it was built for.

2026 (label-free): how many of the 344,453 non-noac ranking rows the shipped v9
prediction puts above the bound, and how far.  This is what a clip would move,
not what it would gain; no fold scores it here.
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
from step2 import LEVELS, VARIANTS, V9  # noqa: E402

COLS_2025 = ["apt", "noac", "y_true", "g_any", "g_prev2"]
COLS_2026 = ["MVT_ID_mvt", "apt", "noac", "g_any", "g_prev2"]


def main() -> None:
    m = pd.read_csv(EXP / "analysis" / "step2_margins.csv")

    def margin(apt: pd.Series, var: str, lv: float) -> np.ndarray:
        """Per-row margin, by a vectorised map: a per-row `.loc` on a MultiIndex
        is 2 million lookups a column and takes minutes."""
        s_ = m[m.variant == var].set_index("apt")[f"q{lv:g}"]
        return apt.astype(str).map(s_).to_numpy("float64")

    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet", columns=COLS_2025)
    rows = []
    for var in VARIANTS:
        for lv in LEVELS:
            q = margin(a.apt, var, lv)
            bd = a[var].to_numpy(float) - q
            bulk = (a.noac.to_numpy() == 0) & np.isfinite(bd)
            viol = (a.y_true.to_numpy(float) > bd) & bulk
            ex = np.where(viol, a.y_true.to_numpy(float) - bd, 0.0)
            rows.append({"variant": var, "level": lv, "bulk_n": int(bulk.sum()),
                         "viol": int(viol.sum()), "viol_share": float(viol.sum() / bulk.sum()),
                         "excess_med": float(np.median(ex[viol])) if viol.any() else np.nan,
                         "excess_rms_all": float(np.sqrt(np.mean(ex ** 2)))})
    t25 = pd.DataFrame(rows)
    t25.to_csv(EXP / "analysis" / "bulk_bound_2025.csv", index=False)
    del a

    b = pd.read_parquet(EXP / "work" / "dep_2026.parquet", columns=COLS_2026)
    v9 = pd.read_parquet(V9).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b["pred"] = v9.reindex(b.MVT_ID_mvt.to_numpy()).to_numpy()
    rows = []
    for var in VARIANTS:
        for lv in LEVELS:
            q = margin(b.apt, var, lv)
            bd = b[var].to_numpy(float) - q
            bulk = (b.noac.to_numpy() == 0) & np.isfinite(bd)
            over = bulk & (b.pred.to_numpy(float) > bd)
            ex = np.where(over, b.pred.to_numpy(float) - bd, 0.0)
            rows.append({"variant": var, "level": lv, "bulk_n": int(bulk.sum()),
                         "bulk_over": int(over.sum()),
                         "bulk_share": float(over.sum() / bulk.sum()),
                         "excess_med": float(np.median(ex[over])) if over.any() else np.nan,
                         "excess_p99": float(np.quantile(ex[over], 0.99)) if over.any() else np.nan,
                         "excess_max": float(ex.max()),
                         "clip_rms_file": float(np.sqrt(np.mean(ex ** 2)))})
    t26 = pd.DataFrame(rows)
    t26.to_csv(EXP / "analysis" / "bulk_bound.csv", index=False)
    with pd.option_context("display.width", 200):
        print("2025 ordinary departures whose TRUE label breaks the bound")
        print(t25.round(5).to_string(index=False))
        print()
        print("2026 ordinary ranking rows the shipped v9 puts above the bound")
        print(t26.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
