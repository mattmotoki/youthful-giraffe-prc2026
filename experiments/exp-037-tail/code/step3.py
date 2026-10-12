"""Step 3: the placeholder question.

Two label-free indicators per 2025 noac row, both computed exactly as they are
on the 388:

    excluded   `mvt_sched > bound`: the placeholder anchor (y = mvt_sched) is
               above what the chain allows, so a row that took that value would
               have been on its stand before the previous arrival left it.
    over_base  not used here; §4 owns the prediction.

Against them, the record's placeholder class `ph = |mvt_sched - y| < 300 s`.
Intervals are calendar-day-clustered bootstraps: the noac rows are one
airport's feed and are day-concentrated (exp-033 §5).

Also here: `P(y > bound)`, the rate at which a TRUE 2025 noac label breaks the
bound.  That is the false-clip rate a clipping head pays, and it is the number
that decides whether §4 may clip at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import LAB  # noqa: E402
from step2 import LEVELS, VARIANTS, margins  # noqa: E402

NBOOT = 4000
RNG = np.random.default_rng(20260908)


def day_boot(d: pd.DataFrame, stat, nboot: int = NBOOT) -> tuple:
    """Day-clustered bootstrap of `stat(frame)`; returns (point, lo, hi)."""
    days = d.day.to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    point = stat(d)
    out = np.empty(nboot)
    for b in range(nboot):
        pick = RNG.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_by_day[i] for i in pick])
        out[b] = stat(d.iloc[rows])
    ok = np.isfinite(out)
    lo, hi = (np.quantile(out[ok], [0.025, 0.975]) if ok.sum() > 50 else (np.nan, np.nan))
    return point, float(lo), float(hi)


def _diff(d: pd.DataFrame) -> float:
    a = d.ph[d.excluded].mean() if d.excluded.any() else np.nan
    b = d.ph[~d.excluded].mean() if (~d.excluded).any() else np.nan
    return float(a - b)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet")
    n = pd.read_parquet(EXP / "work" / "noac_2025.parquet")
    m = margins(a).set_index(["apt", "variant"])

    rows, safe = [], []
    for var in VARIANTS:
        for lv in LEVELS:
            q = np.array([m.loc[(ap, var), f"q{lv:g}"] for ap in n.apt.astype(str)])
            bound = n[var].to_numpy(float) - q
            d0 = n.assign(bound=bound, excluded=(n.mvt_sched.to_numpy(float) > bound),
                          y_over=(n.y_true.to_numpy(float) > bound))
            d0 = d0[np.isfinite(bound)]
            for lab in LAB + ["ALL"]:
                d = d0 if lab == "ALL" else d0[d0.band == lab]
                if len(d) < 10 or d.excluded.nunique() < 2:
                    continue
                pt, lo, hi = day_boot(d, _diff)
                rows.append({"variant": var, "level": lv, "band": lab, "n": len(d),
                             "n_excluded": int(d.excluded.sum()),
                             "P_ph_excluded": float(d.ph[d.excluded].mean()),
                             "P_ph_rest": float(d.ph[~d.excluded].mean()),
                             "P_ph_all": float(d.ph.mean()),
                             "diff": pt, "lo": lo, "hi": hi})
            for lab in LAB + ["ALL"]:
                d = d0 if lab == "ALL" else d0[d0.band == lab]
                if not len(d):
                    continue
                safe.append({"variant": var, "level": lv, "band": lab, "n": len(d),
                             "P_y_over_bound": float(d.y_over.mean()),
                             "P_y_over_ph": float(d.y_over[d.ph == 1].mean()) if (d.ph == 1).any() else np.nan,
                             "P_y_over_np": float(d.y_over[d.ph == 0].mean()) if (d.ph == 0).any() else np.nan})
    t = pd.DataFrame(rows)
    s = pd.DataFrame(safe)
    t.to_csv(EXP / "analysis" / "step3_conditional.csv", index=False)
    s.to_csv(EXP / "analysis" / "step3_bound_safety.csv", index=False)
    with pd.option_context("display.width", 220):
        print("P(ph | the chain excludes the placeholder anchor) minus P(ph | it does not)")
        print(t.round(3).to_string(index=False))
        print()
        print("P(a TRUE 2025 noac label breaks the bound) -- the false-clip rate")
        print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
