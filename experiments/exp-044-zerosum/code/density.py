"""Step 1 -- is Q03's small-cell concentration density, or mechanism?

Q03's T6 read the pair reduction by cell size and found it concentrated in 4-9
row cells, with the 10-19 stratum changing sign.  Two readings fit that:

* DENSITY -- the gain lives in quiet traffic, and "small cell" is a proxy for
  "few departures per minute".  Change the grid and the gain follows the traffic,
  not the row count.
* MECHANISM -- the gain is a function of how many rows the cell holds.  Change
  the grid and the gain follows the row count.

The two are separated by a fixed alternate grid each way (300 s and 1,200 s
against Q03's 600 s) and a traffic rate that does not depend on the grid at all:
the number of eligible departures at the same airport within +/- 300 s of the
row's take-off clock.  A 1,200 s grid puts quiet periods into the large-row
band; a 300 s grid puts busy periods into the small-row band.  Whichever ordering
survives conditioning on the other is the driver.

The matched-subset check is the second half: every cell contributes exactly its
first four rows under the label-free (cell, movement id) order, so every cell
supplies exactly two pairs and no cell is weighted by its size.  The ridge is
unchanged throughout -- only the evaluation subset moves.

    OMP_NUM_THREADS=3 python density.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import rowfeat as R                         # noqa: E402

EXP = zs.EXP
FOLDS = zs.FOLDS
PRE = zs.PRE["step1_density_check"]
CELL_BANDS = [tuple(b) for b in PRE["stratum_cell_rows"]]
RATE_BANDS = [tuple(b) for b in PRE["stratum_traffic_rate"]["bands"]]


def band(v, bands):
    out = np.full(len(v), -1, "int64")
    for i, (lo, hi) in enumerate(bands):
        out[(v >= lo) & (v <= hi)] = i
    return out


def label(b):
    lo, hi = b
    return f"{lo}-{hi}" if hi < 10 ** 8 else f"{lo}+"


def prepare(fold: str, panel: pd.DataFrame) -> dict:
    d = zs.dep(R.month_of(fold))
    lv = zs.levels_of([d])
    X, _ = zs.design(d, lv)
    idx = pd.Series(np.arange(len(d)), index=d["MVT_ID_mvt"].astype("float64").to_numpy())
    take = idx.reindex(panel["MVT_ID_mvt"].to_numpy("float64"))
    if take.isna().any():
        raise SystemExit(f"{fold}: a panel row has no feature row")
    X = X[take.to_numpy("int64")]
    air = panel["airport"].astype(str).to_numpy()
    reg = panel["regime"].astype(str).to_numpy()
    t = panel["t"].to_numpy("float64")
    elig = (reg == zs.ELIGIBLE_REGIME) & ~np.isin(air, zs.HOLD_AIRPORTS)
    return {"fold": fold, "X": X, "air": air, "reg": reg, "t": t, "elig": elig,
            "r": panel["r"].to_numpy("float64"),
            "y": panel["y_true"].to_numpy("float64"),
            "p10": panel["p10"].to_numpy("float64"),
            "ids": panel["MVT_ID_mvt"].to_numpy("float64"),
            "days": zs.V.day(t),
            "rate": zs.traffic_rate(air, t, elig)}


def grid_arm(S: dict, bin_s: int) -> dict:
    """The unchanged ridge on one grid: h, the scored cohort and the update."""
    code, ncell = zs.cellcode(S["air"], S["reg"], S["t"], bin_s)
    ei = np.flatnonzero(S["elig"])
    blk = zs.blocks_of(S["days"])
    X, r = S["X"], S["r"]
    h = np.zeros(len(r))
    for b in range(zs.BLOCKS):
        fit, sc = ei[blk[ei] != b], ei[blk[ei] == b]
        fc, _ = pd.factorize(code[fit])
        nfc = fc.max() + 1
        Xf = X[fit].astype("float64")
        rc = zs.project(r[fit], fc, nfc)
        h[sc] = zs.ridge_h(Xf, rc, fc, nfc, X[sc].astype("float64"))
    cnt = np.bincount(code[ei], minlength=ncell)
    m0 = S["elig"] & (cnt[code] >= zs.MIN_CELL)
    cm, _ = pd.factorize(code[m0])
    d0 = np.zeros(len(r))
    d0[m0] = zs.project(h[m0], cm, cm.max() + 1)
    return {"bin_s": bin_s, "code": code, "ncell": ncell, "cnt": cnt,
            "m0": m0, "h": h, "d0": d0}


def pair_table(S: dict, G: dict, amp: float, cap: int | None) -> dict:
    ei = np.flatnonzero(S["elig"])
    ce, nce = pd.factorize(G["code"][ei])
    ia, ib = zs.pairs_in_cells(ce, S["ids"][ei], len(nce), zs.MIN_CELL, cap)
    ia, ib = ei[ia], ei[ib]
    dr = S["r"][ia] - S["r"][ib]
    dh = amp * (G["h"][ia] - G["h"][ib])
    red = dr ** 2 - (dr - dh) ** 2
    cb = band(G["cnt"][G["code"][ia]], CELL_BANDS)
    rb = band(np.rint((S["rate"][ia] + S["rate"][ib]) / 2).astype("int64"), RATE_BANDS)
    return {"n_pairs": int(len(ia)), "reduction_s2": float(red.mean()),
            "red": red, "cell_band": cb, "rate_band": rb,
            "day": S["fold"] + "|" + S["days"][ia]}


def twoway(parts: list[dict]) -> dict:
    red = np.concatenate([p["red"] for p in parts])
    cb = np.concatenate([p["cell_band"] for p in parts])
    rb = np.concatenate([p["rate_band"] for p in parts])
    out = {"by_cell_rows": {}, "by_traffic_rate": {}, "table": [], "table_all": []}
    for i, b in enumerate(CELL_BANDS):
        m = cb == i
        if m.sum():
            out["by_cell_rows"][label(b)] = {"n": int(m.sum()),
                                             "reduction_s2": round(float(red[m].mean()), 2)}
    for j, b in enumerate(RATE_BANDS):
        m = rb == j
        if m.sum():
            out["by_traffic_rate"][label(b)] = {"n": int(m.sum()),
                                                "reduction_s2": round(float(red[m].mean()), 2)}
    # Sol round 2, blocking 3: the 200-pair minimum below was an implementation
    # choice, not a predeclared one.  Every non-empty two-way cell is now written
    # with its pair count, and the minimum is carried as a FLAG so the verdict can
    # be read with it and without it.
    out["two_way_min_pairs_not_predeclared"] = 200
    for i, bi in enumerate(CELL_BANDS):
        for j, bj in enumerate(RATE_BANDS):
            m = (cb == i) & (rb == j)
            if m.sum():
                out["table_all"].append({"cell_rows": label(bi), "traffic_rate": label(bj),
                                         "n": int(m.sum()),
                                         "reduction_s2": round(float(red[m].mean()), 2)})
                if m.sum() >= 200:
                    out["table"].append({"cell_rows": label(bi), "traffic_rate": label(bj),
                                         "n": int(m.sum()),
                                         "reduction_s2": round(float(red[m].mean()), 2)})
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(zs.SEED)
    pan = zs.V.load()
    S = {f: prepare(f, pan[f]) for f in FOLDS}
    print(f"[prepare] {time.time() - t0:.0f}s", flush=True)

    out = {"step": "1 density check", "grids": list(zs.GRIDS), "folds": {},
           "pooled": {}, "board": {}}
    for bin_s in zs.GRIDS:
        G = {f: grid_arm(S[f], bin_s) for f in FOLDS}
        amp = {}
        for f in FOLDS:
            num = den = 0.0
            for g in FOLDS:
                if g == f:
                    continue
                m = G[g]["m0"]
                num += float((S[g]["r"][m] * G[g]["d0"][m]).sum())
                den += float((G[g]["d0"][m] ** 2).sum())
            amp[f] = num / den if den > 0 else 0.0
        full, matched, ex = [], [], {}
        for f in FOLDS:
            s, g = S[f], G[f]
            pf = pair_table(s, g, amp[f], None)
            pm = pair_table(s, g, amp[f], zs.MIN_CELL)
            full.append(pf)
            matched.append(pm)
            upd = amp[f] * g["d0"]
            p_new = s["p10"] + upd
            ex[f] = {"all_v10": float(zs.D.rmse(s["y"], s["p10"])),
                     "all_new": float(zs.D.rmse(s["y"], p_new)),
                     "clean_v10": float(zs.D.rmse_clean(s["y"], s["p10"], s["ids"], f)),
                     "clean_new": float(zs.D.rmse_clean(s["y"], p_new, s["ids"], f))}
            v = 2 * s["r"][g["m0"]] * upd[g["m0"]] - upd[g["m0"]] ** 2
            pt, lo, hi = zs.boot_days(v, np.ones(int(g["m0"].sum())),
                                      s["days"][g["m0"]], rng)
            key = f"{bin_s}|{f}"
            out["folds"][key] = {
                "fold": f, "bin_s": bin_s, "amplitude_cross_month": round(amp[f], 4),
                "n_cells": int(g["ncell"]), "n_scored_cohort": int(g["m0"].sum()),
                "median_cell_rows": float(np.median(g["cnt"][g["code"][g["m0"]]])),
                "cell_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                "gain_all": round(ex[f]["all_v10"] - ex[f]["all_new"], 4),
                "gain_clean": round(ex[f]["clean_v10"] - ex[f]["clean_new"], 4),
                "pairs_full": pf["n_pairs"], "reduction_full_s2": round(pf["reduction_s2"], 2),
                "pairs_matched": pm["n_pairs"],
                "reduction_matched_s2": round(pm["reduction_s2"], 2)}
            print(f"  [{bin_s:>4}s {f:>8}] cohort {int(g['m0'].sum()):>7,} cells "
                  f"{g['ncell']:>6,} amp {amp[f]:.4f}  all-row {out['folds'][key]['gain_all']:+.4f} "
                  f"pair full {pf['reduction_s2']:+8.1f} matched {pm['reduction_s2']:+8.1f}",
                  flush=True)
        out["pooled"][str(bin_s)] = {"full": twoway(full), "matched": twoway(matched)}
        for tag, parts in (("full", full), ("matched", matched)):
            v = np.concatenate([p["red"] for p in parts])
            day = np.concatenate([p["day"] for p in parts])
            pt, lo, hi = zs.boot_days(v, np.ones(len(v)), day, rng)
            out["pooled"][str(bin_s)][tag]["pooled_s2"] = round(pt, 2)
            out["pooled"][str(bin_s)][tag]["ci"] = [round(lo, 2), round(hi, 2)]
        v10 = zs.V.board({f: ex[f]["all_v10"] for f in FOLDS})
        arm = zs.V.board({f: ex[f]["all_new"] for f in FOLDS})
        cv, ca = (zs.V.board({f: ex[f][k] for f in FOLDS}) for k in ("clean_v10", "clean_new"))
        out["board"][str(bin_s)] = {"all_row_v10": round(v10, 4), "all_row_arm": round(arm, 4),
                                    "gain_all_row": round(v10 - arm, 4),
                                    "clean_v10": round(cv, 4), "clean_arm": round(ca, 4),
                                    "gain_clean": round(cv - ca, 4)}
        print(f"  [{bin_s:>4}s] all-row board {v10:.4f} -> {arm:.4f} "
              f"({v10 - arm:+.4f} s)\n", flush=True)
        del G
    out["wall_s"] = round(time.time() - t0, 1)
    p = EXP / "analysis" / "density.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"written {p} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
