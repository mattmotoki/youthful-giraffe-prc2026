"""Step 1d: the placement, with the same-day free neighbour removed.

`describe.py`'s reference let a 2025 row take its nearest neighbour from its OWN
calendar day, where the day-level block (`dep_day_n`, `arr_day_ph_rate`,
`noac_day_*`) is constant and the operational state is shared.  A 2026 row can
never have such a neighbour, so that reference is biased toward calling 2026
rows far out.  Here the 2025 leave-one-out reference excludes same-day
neighbours, which is the comparison the 2026 rows actually face.

Two spaces, because the day block is the half of the feature set that a
classifier can memorise a day from (`twosample.py`): the full separator space,
and the row-level space with the six day-level columns dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import describe as S  # noqa: E402

DAY_BLOCK = ("dep_day_n", "noac_day_n", "noac_day_share", "noac_day_count",
             "nonm_day_count", "arr_day_ph_rate")


def place(a, b, sc, tag, store=None):
    d_ba = S.gower(b, a, sc)
    d_aa = S.gower(a, a, sc)
    same_day = a.day.to_numpy()[:, None] == a.day.to_numpy()[None, :]
    d_aa = np.where(same_day, np.inf, d_aa)          # no free same-day neighbour
    loo = d_aa.min(1)
    ref50, ref95 = np.percentile(loo, [50, 95])
    nn1 = d_ba.min(1)
    place = np.where(nn1 <= ref50, "inside", np.where(nn1 <= ref95, "edge", "outside"))
    print(f"\n{tag}: 2025 LOO nn1 (cross-day) median {ref50:.4f} p95 {ref95:.4f}; "
          f"2026 nn1 median {np.median(nn1):.4f}")
    print("  placement:", dict(pd.Series(place).value_counts()))
    if store is not None:
        store[f"nn1_{tag}"] = nn1
        store[f"place_{tag}"] = place
        store[f"loo_{tag}"] = loo
        store[f"d_ba_{tag}"] = d_ba
        store[f"d_aa_{tag}"] = d_aa
    out = []
    for lab in S.LAB:
        m = (b.band == lab).to_numpy()
        if not m.sum():
            continue
        p = (a.band == lab).to_numpy()
        out.append({"space": tag, "band": lab, "n2026": int(m.sum()), "n2025": int(p.sum()),
                    "nn1_med_2026": float(np.median(nn1[m])),
                    "loo_med_2025": float(np.median(loo[p])),
                    "inside": int((place[m] == "inside").sum()),
                    "edge": int((place[m] == "edge").sum()),
                    "outside": int((place[m] == "outside").sum())})
    return pd.DataFrame(out)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_d.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["day"] = pd.to_datetime(a.mvt_time, utc=True).dt.floor("D")
    store = {}
    sc = S.scales(a)
    t1 = place(a, b, sc, "full", store)
    sc2 = {k: v for k, v in sc.items() if k not in DAY_BLOCK}
    t2 = place(a, b, sc2, "rowlevel", store)
    t = pd.concat([t1, t2], ignore_index=True)
    t.to_csv(EXP / "analysis" / "placement.csv", index=False)
    print()
    print(t.to_string(index=False))
    for k in ("full", "rowlevel"):
        b[f"nn1_{k}"] = store[f"nn1_{k}"]
        b[f"place_{k}"] = store[f"place_{k}"]
        np.save(EXP / "work" / f"d_ba_{k}.npy", store[f"d_ba_{k}"])
        np.save(EXP / "work" / f"d_aa_{k}.npy", store[f"d_aa_{k}"])
    b.to_parquet(EXP / "work" / "noac_2026_d.parquet")
    a.to_parquet(EXP / "work" / "noac_2025_d.parquet")


if __name__ == "__main__":
    main()
