"""Step 1b: where the nearest neighbours live, and which features push a 2026 row out."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402
from prc.tail_model import NOAC_FEATURES_V2, NOAC_CATS_V2  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from describe import LAB, CLIP, NUM, scales  # noqa: E402


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_d.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    d_ba = np.load(EXP / "work" / "d_ba.npy")
    sc = scales(a)

    nn = np.argsort(d_ba, axis=1)[:, 0]
    b["nn_month"] = a.month.to_numpy()[nn]
    b["nn_band"] = a.band.to_numpy()[nn]
    b["nn_apt"] = a.adep.to_numpy()[nn]
    print("NN band == query band:", float((b.nn_band.to_numpy() == b.band.to_numpy()).mean()))
    print("NN month distribution:")
    print(b.nn_month.value_counts().sort_index().to_string())
    print("\n2025 noac rows per month (the reference supply):")
    print(a.month.value_counts().sort_index().to_string())

    # per-feature share of the nn1 distance, averaged over the 388
    rows = []
    for c, s in sc.items():
        x = b[c].to_numpy("float64")
        y = a[c].to_numpy("float64")[nn]
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.where(ok, np.minimum(np.abs(x - y) / s, CLIP), np.nan)
        rows.append({"feature": c, "kind": "num", "mean_d": float(np.nanmean(d)),
                     "n_at_clip": int(np.nansum(d >= CLIP - 1e-9)),
                     "coverage_2026": float(np.isfinite(x).mean())})
    for c in NOAC_CATS_V2:
        x = b[c].astype(str).to_numpy()
        y = a[c].astype(str).to_numpy()[nn]
        miss = (x == "nan") | (y == "nan")
        d = np.where(miss, np.nan, (x != y).astype(float))
        rows.append({"feature": c, "kind": "cat", "mean_d": float(np.nanmean(d)),
                     "n_at_clip": int(np.nansum(d >= 1.0)),
                     "coverage_2026": float((~(x == "nan")).mean())})
    t = pd.DataFrame(rows).sort_values("mean_d", ascending=False)
    t.to_csv(EXP / "analysis" / "nn_feature_share.csv", index=False)
    print("\nper-feature mean distance to the nearest 2025 neighbour (top 14):")
    print(t.head(14).to_string(index=False))
    print("\nbottom 6:")
    print(t.tail(6).to_string(index=False))

    # the regime's own rate, label-free
    print("\nnoac share of departures, by month:")
    out = []
    for m in D.MONTHS:
        raw = D.load_month(m, columns=["PHASE_mvt", "AIRCRAFT_TYPE_mvt", "ADEP_mvt"])
        dep = raw[raw.PHASE_mvt == "DEP"]
        lirf = dep[dep.ADEP_mvt == "LIRF"]
        out.append({"month": m, "dep": len(dep), "noac": int(dep.AIRCRAFT_TYPE_mvt.isna().sum()),
                    "share_pct": 100 * dep.AIRCRAFT_TYPE_mvt.isna().mean(),
                    "lirf_dep": len(lirf), "lirf_noac": int(lirf.AIRCRAFT_TYPE_mvt.isna().sum()),
                    "lirf_share_pct": 100 * lirf.AIRCRAFT_TYPE_mvt.isna().mean()})
    rank = D.load_ranking(columns=["MVT_TIME_UTC_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "ADEP_mvt"])
    for m, g in rank.groupby("month", sort=True):
        dep = g[g.PHASE_mvt == "DEP"]
        lirf = dep[dep.ADEP_mvt == "LIRF"]
        out.append({"month": m, "dep": len(dep), "noac": int(dep.AIRCRAFT_TYPE_mvt.isna().sum()),
                    "share_pct": 100 * dep.AIRCRAFT_TYPE_mvt.isna().mean(),
                    "lirf_dep": len(lirf), "lirf_noac": int(lirf.AIRCRAFT_TYPE_mvt.isna().sum()),
                    "lirf_share_pct": 100 * lirf.AIRCRAFT_TYPE_mvt.isna().mean()})
    r = pd.DataFrame(out)
    r.to_csv(EXP / "analysis" / "noac_rate.csv", index=False)
    print(r.round(4).to_string(index=False))
    b.to_parquet(EXP / "work" / "noac_2026_d.parquet")


if __name__ == "__main__":
    main()
