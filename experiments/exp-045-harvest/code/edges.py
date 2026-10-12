"""exp-045-harvest: is any box clipping its own aprons?

A box that is too small at one airport looks exactly like a dark airport in the receipt
table, and re-fetching a month costs hundreds of kernel hours, so the boxes are checked
before the bulk runs rather than after.  The test uses the one position the summaries
carry: `stand_lat`/`stand_lon`, the place a departure pushed back from.  Stands piled
against a box edge mean the box cut the apron; stands well inside every edge mean it did
not.  Reports, per airport, how far the nearest stand is from each of the four edges and
how many stands sit within `NEAR_M` of one.

    .venv/bin/python experiments/exp-045-harvest/code/edges.py [YYYY-MM-DD ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ART = ROOT / "artifacts/exp-045"
NEAR_M = 500.0


def main(days: list[str]) -> None:
    boxes = json.loads((EXP / "analysis/boxes.json").read_text())["boxes"]
    out = {}
    for a, b in boxes.items():
        fs = [p for p in sorted((ART / f"summaries/{a}").glob("*.parquet"))
              if not days or p.stem in days]
        if not fs:
            continue
        s = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
        s = s[(s.kind == "departure") & s.stand_lat.notna()]
        if not len(s):
            continue
        mlat = (b["lat"][0] + b["lat"][1]) / 2
        my = 110540.0
        mx = 111320.0 * np.cos(np.radians(mlat))
        d = np.c_[(s.stand_lat - b["lat"][0]) * my, (b["lat"][1] - s.stand_lat) * my,
                  (s.stand_lon - b["lon"][0]) * mx, (b["lon"][1] - s.stand_lon) * mx]
        near = d.min(axis=1)
        out[a] = {
            "departure_stands": int(len(s)),
            "min_m_to_edge": {k: round(float(v), 0) for k, v in
                              zip(("south", "north", "west", "east"), d.min(axis=0))},
            "stands_within_500m_of_an_edge": int((near < NEAR_M).sum()),
            "share_within_500m": round(float((near < NEAR_M).mean()), 4),
            "p01_m_to_edge": round(float(np.quantile(near, 0.01)), 0),
            "median_m_to_edge": round(float(np.median(near)), 0),
        }
    (EXP / "analysis/edges.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"{'airport':8} {'stands':>7} {'p01 m':>7} {'median m':>9} {'<500 m':>8}  "
          f"nearest edge (S/N/W/E, m)")
    for a, r in out.items():
        e = r["min_m_to_edge"]
        print(f"{a:8} {r['departure_stands']:7d} {r['p01_m_to_edge']:7.0f} "
              f"{r['median_m_to_edge']:9.0f} {r['share_within_500m']:8.2%}  "
              f"{e['south']:.0f}/{e['north']:.0f}/{e['west']:.0f}/{e['east']:.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
