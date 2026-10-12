"""Q1c: how hard does each member push on the day-state-exposed rows?

The blend dilutes: `D1-winter` carries 0.25 of candidate 2 outside `nm_late`, so
a member-level push of P seconds reaches the shipped bytes as about P/4 minus
whatever the other three do. exp-007's harm was measured on a STANDALONE arm, so
the member's own push is the quantity comparable to it, and the blend's is the
quantity that matters for the upload. Both are reported.
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
CFG2 = F.ROOT / "release" / "config" / "exp-012-blend2" / "blend-config-2.json"


def main() -> None:
    cfg = json.loads(CFG2.read_text())
    frame, names = B.stack_members(cfg["ranking"], F.ROOT)
    ds = pd.read_parquet(F.ROOT / "artifacts" / "exp-014" / "frames" / "ds-ranking.parquet",
                         columns=["MVT_ID_mvt", "apt", "day", "month_str"])
    zx = pd.read_csv(OUT / "q1b_daystate_by_airport.csv")   # for context only
    dsr = pd.read_parquet(F.ROOT / "artifacts" / "exp-014" / "frames" / "ds-ranking.parquet")
    tr = pd.read_parquet(F.ROOT / "artifacts" / "exp-014" / "frames" / "ds-train.parquet")
    zcols = [c for c in dsr.columns if c.endswith("_z")]
    lim = tr.groupby("apt", observed=True)[zcols].max()
    over = np.zeros(len(dsr), dtype=int)
    for c in zcols:
        hi = dsr["apt"].map(lim[c]).to_numpy("float64")
        v = dsr[c].to_numpy("float64")
        over += (np.isfinite(v) & (v > hi)).astype(int)
    ds = ds.assign(n_z_over=over)

    m = frame.merge(ds, on="MVT_ID_mvt", how="inner")
    r = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
                                "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt"])
    r = r[r["PHASE_mvt"] == "DEP"].copy()
    r["regime"] = B.regimes_from_raw(r, float(cfg["late_cut"]))
    r["d_takeoff_aobt"] = (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
    m = m.merge(r[["MVT_ID_mvt", "regime", "d_takeoff_aobt"]], on="MVT_ID_mvt", how="inner")

    # prepare members exactly as build_candidate does, so the numbers are the
    # ones the blend actually averages
    P = m[names].to_numpy("float64").copy()
    a = m["d_takeoff_aobt"].to_numpy("float64")
    lo, hi = cfg["guard_bounds"]
    floor = float(cfg["floor"])
    for j in range(P.shape[1]):
        bad = ~np.isfinite(P[:, j])
        P[bad, j] = floor
        P[:, j] = np.maximum(P[:, j], floor)
        P[:, j] = B.apply_residual_guard(P[:, j], a, float(lo), float(hi), floor)
    for j, n in enumerate(names):
        m[n] = P[:, j]

    gen = [n for n in names if n != "N1-nmtail"]          # the four general members
    others = [n for n in gen if n != "D1-winter"]
    m["others_mean"] = m[others].mean(axis=1)
    m["push"] = m["D1-winter"] - m["others_mean"]

    rows = []
    for month in ("2026-01", "2026-07"):
        s = m["month_str"] == month
        for label, sub in ((f"{month} all", m[s]),
                           (f"{month} no `_z` over", m[s & (m["n_z_over"] == 0)]),
                           (f"{month} 1-2 `_z` over", m[s & m["n_z_over"].between(1, 2)]),
                           (f"{month} 3+ `_z` over", m[s & (m["n_z_over"] >= 3)]),
                           (f"{month} 6+ `_z` over", m[s & (m["n_z_over"] >= 6)])):
            if not len(sub):
                continue
            d = dict(slice=label, n=len(sub), push_mean=float(sub["push"].mean()),
                     push_p50=float(sub["push"].median()),
                     push_p90=float(sub["push"].quantile(0.9)),
                     blend_effect=float(sub["push"].mean() / len(gen)))
            for n in names:
                d[n] = float(sub[n].mean())
            rows.append(d)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "q1c_member_push.csv", index=False)
    print("=== `D1-winter` minus the other three general members, by day-state exposure ===")
    print("(`blend_effect` = push / 4, what a flat quarter-weight carries into the file)")
    print(T.round(1).to_string(index=False), flush=True)

    # EHAM January, the sharpest slice
    e = m[(m["apt"] == "EHAM") & (m["month_str"] == "2026-01")]
    print(f"\nEHAM January 2026 ({len(e)} rows): push mean {e['push'].mean():.1f} s, "
          f"blend effect {e['push'].mean()/len(gen):.1f} s", flush=True)
    e6 = e[e["n_z_over"] >= 6]
    print(f"EHAM January, 6+ `_z` over ({len(e6)} rows): push mean {e6['push'].mean():.1f} s, "
          f"blend effect {e6['push'].mean()/len(gen):.1f} s", flush=True)


if __name__ == "__main__":
    main()
