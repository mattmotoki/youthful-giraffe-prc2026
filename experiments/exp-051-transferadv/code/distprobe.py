"""exp-051, after iteration 1 was read and before iteration 2 was registered: the distance probe.
ZERO FIT.  POST HOC -- not registered in SPEC I1; it changes no I1 verdict.

Question it answers for the iteration-2 choice: is the half-day carrier strong because its
days are ADJACENT to the scored days (SPEC proxy limit 3), or because of what the month itself
carries?  Each scored row has a distance in days to the nearest day of the carrier half.  With
four contiguous blocks the distances run 1-8.  If adjacency drives it, the carrier's own value
and each step's discount must weaken as the distance grows.

    .venv/bin/python experiments/exp-051-transferadv/code/distprobe.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common as C
from common import A, NO

OUT = C.ANALYSIS / "distprobe.json"
BINS = [(1, 2), (3, 4), (5, 8)]


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    out = {"section": "exp-051 post hoc distance probe on the I1 vectors (zero fit)",
           "bins_days": BINS, "folds": {}}
    for f in A.FOLDS:
        p = pan_ship[f]
        hv = C.half_of_rows(pan_ship, f)
        day = pd.to_datetime(p.day).dt.normalize()
        ud = np.sort(day.unique())
        hmap = dict(zip(day, hv))
        hday = np.array([hmap[d] for d in ud])
        dist = {}
        for d, h in zip(ud, hday):
            other = ud[hday != h]
            dist[d] = int(np.min(np.abs((other - d).astype("timedelta64[D]").astype(int))))
        rd = day.map(dist).to_numpy()
        y = p.y_true.to_numpy("float64") - p.pbase.to_numpy("float64")
        BL = {L: np.where(hv == 1, C.load_vec(C.fit_path(f, 0, L)[0])[f],
                          C.load_vec(C.fit_path(f, 1, L)[0])[f]) for L in range(4)}
        P = {L: C.load_vec(C.PLAIN[L])[f] for L in range(4)}
        rec = {"days_per_bin": {}, "rows_per_bin": {}, "L0_half_days_over_member1_s2": {},
               "steps": {}}
        for b in BINS:
            k = f"{b[0]}-{b[1]}"
            m = (rd >= b[0]) & (rd <= b[1])
            rec["days_per_bin"][k] = int(sum(1 for v in dist.values() if b[0] <= v <= b[1]))
            rec["rows_per_bin"][k] = int(m.sum())
            r0 = y - P[0]
            i0 = BL[0] - P[0]
            rec["L0_half_days_over_member1_s2"][k] = float(np.mean((2 * r0 * i0 - i0 * i0)[m]))
        for L in (1, 2, 3):
            e = {}
            for b in BINS:
                k = f"{b[0]}-{b[1]}"
                m = (rd >= b[0]) & (rd <= b[1])
                g = {}
                for nm, X in (("BL", BL), ("P", P)):
                    r = y - X[L - 1]
                    inc = X[L] - X[L - 1]
                    g[nm] = float(np.mean((2 * r * inc - inc * inc)[m]))
                e[k] = {"BL_s2": g["BL"], "P_s2": g["P"], "ratio": g["BL"] / g["P"]}
            rec["steps"][C.STEP[L]] = e
        out["folds"][f] = rec
        print(f, "carrier own value by bin", {k: round(v, 1) for k, v in
                                              rec["L0_half_days_over_member1_s2"].items()},
              "| v20 ratio", {k: round(v["ratio"], 3) for k, v in rec["steps"]["v20"].items()},
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"written to {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
