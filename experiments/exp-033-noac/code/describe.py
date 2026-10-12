"""Step 1: the 388 as a population, label-free, against the 1,507 of 2025.

Nothing here reads a 2025 label; the 2026 side has none.  The feature space is
the one `prc.tail_model`'s noac placeholder separator uses (`NOAC_FEATURES_V2`).

Distance is Gower with a robust scale: per numeric feature |x-y| divided by the
2025 noac population's IQR (std where the IQR is zero, feature dropped where
both are), clipped at 3 so no single feature owns a pair; per categorical
feature 0 or 1; a feature missing on either side is not counted for that pair;
the total is the mean over the counted features.  The reference distribution is
the LEAVE-ONE-OUT nearest-neighbour distance inside the 2025 cloud itself, so a
2026 row is placed inside / at the edge / outside by the same instrument that
measures 2025's own density.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
from prc.tail_model import NOAC_FEATURES_V2, NOAC_CATS_V2  # noqa: E402

BAND = [-1e18, 0, 1800, 3600, 7200, 14400, 28800, 1e18]
LAB = ["neg", "0-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8h+"]
CLIP = 3.0
NUM = tuple(c for c in NOAC_FEATURES_V2 if c not in NOAC_CATS_V2)


def add_band(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["band"] = pd.cut(d.mvt_sched, BAND, labels=LAB, right=False).astype(str)
    return d


def scales(a: pd.DataFrame) -> dict:
    out = {}
    for c in NUM:
        v = a[c].to_numpy("float64")
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        s = np.subtract(*np.percentile(v, [75, 25]))
        if s <= 0:
            s = float(np.std(v))
        if s > 0:
            out[c] = float(s)
    return out


def gower(q: pd.DataFrame, r: pd.DataFrame, sc: dict) -> np.ndarray:
    """(len(q), len(r)) Gower distance, query rows against reference rows."""
    tot = np.zeros((len(q), len(r)))
    cnt = np.zeros((len(q), len(r)))
    for c, s in sc.items():
        x = q[c].to_numpy("float64")[:, None]
        y = r[c].to_numpy("float64")[None, :]
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.minimum(np.abs(x - y) / s, CLIP)
        tot += np.where(ok, d, 0.0)
        cnt += ok
    for c in NOAC_CATS_V2:
        x = q[c].astype(str).to_numpy()[:, None]
        y = r[c].astype(str).to_numpy()[None, :]
        miss = (x == "nan") | (y == "nan")
        d = (x != y).astype(float)
        tot += np.where(miss, 0.0, d)
        cnt += ~miss
    return tot / np.maximum(cnt, 1)


def main() -> None:
    a = add_band(pd.read_parquet(EXP / "work" / "noac_2025.parquet"))
    b = add_band(pd.read_parquet(EXP / "work" / "noac_2026.parquet"))
    sc = scales(a)
    print(f"{len(sc)} numeric features scaled, {len(NOAC_CATS_V2)} categorical")

    d_ba = gower(b, a, sc)                      # 388 x 1507
    d_aa = gower(a, a, sc)                      # 1507 x 1507, LOO reference
    np.fill_diagonal(d_aa, np.inf)

    order = np.argsort(d_ba, axis=1)
    b["nn1"] = d_ba[np.arange(len(b)), order[:, 0]]
    b["nn5"] = np.sort(d_ba, axis=1)[:, :5].mean(1)
    b["nn1_idx"] = order[:, 0]
    a["loo_nn1"] = d_aa.min(1)
    a["loo_nn5"] = np.sort(d_aa, axis=1)[:, :5].mean(1)

    # NO placement here.  This module's leave-one-out reference lets a 2025 row
    # take a neighbour from its OWN calendar day, which a 2026 row can never
    # have, so it is biased.  `place.py` is the only source of a placement
    # (Sol round 1, S4); the distances below are kept for the description.
    print(f"2025 LOO nn1 (same-day neighbours allowed, NOT the placement "
          f"reference): median {np.median(a.loo_nn1):.4f}")

    np.save(EXP / "work" / "d_ba.npy", d_ba)
    np.save(EXP / "work" / "d_aa.npy", d_aa)
    a.to_parquet(EXP / "work" / "noac_2025_d.parquet")
    b.to_parquet(EXP / "work" / "noac_2026_d.parquet")

    rows = []
    for lab in LAB:
        q = b[b.band == lab]
        p = a[a.band == lab]
        if not len(q):
            continue
        apt = q.adep.value_counts()
        rows.append({
            "band": lab, "n2026": len(q), "n2025": len(p),
            "jan2026": int((q.month == "2026-01").sum()),
            "jul2026": int((q.month == "2026-07").sum()),
            "apt_2026": "; ".join(f"{k} {v}" for k, v in apt.items()),
            "apt_2025_top": "; ".join(f"{k} {v}" for k, v in p.adep.value_counts().head(3).items()),
            "hour_med_2026": float(q.hour.median()), "hour_med_2025": float(p.hour.median()),
            "mvt_sched_med_2026": float(q.mvt_sched.median()),
            "mvt_sched_med_2025": float(p.mvt_sched.median()),
            "date_cross_2026": float(q.sched_date_cross.mean()),
            "date_cross_2025": float(p.sched_date_cross.mean()),
            "noac_day_share_2026": float(q.noac_day_share.mean()),
            "noac_day_share_2025": float(p.noac_day_share.mean()),
            "arr_day_ph_rate_2026": float(q.arr_day_ph_rate.mean()),
            "arr_day_ph_rate_2025": float(p.arr_day_ph_rate.mean()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "population.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(t[["band", "n2026", "n2025", "jan2026", "jul2026", "hour_med_2026",
                 "hour_med_2025", "mvt_sched_med_2026",
                 "mvt_sched_med_2025"]].to_string(index=False))
    print()
    print(t[["band", "apt_2026", "apt_2025_top", "date_cross_2026", "date_cross_2025",
             "noac_day_share_2026", "noac_day_share_2025",
             "arr_day_ph_rate_2026", "arr_day_ph_rate_2025"]].to_string(index=False))
    print()
    print("2026 airports overall:", dict(b.adep.value_counts()))
    print("2025 airports overall:", dict(a.adep.value_counts()))
    print("placement: see place.py / analysis/placement.csv")


if __name__ == "__main__":
    main()
