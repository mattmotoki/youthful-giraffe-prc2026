"""The hypothesis table for the 388 noac rows, and what the board says of them.

Two candidates that differ from the uploaded blend-5 (v2) on exactly those rows
now have board readings:

    v4  exp-002's noac head        shift RMS 1857 s   275.2197   (+9.2566)
    v5  the tail-v4 head undone    shift RMS  371 s   266.2918   (+0.3287)

The truth on those rows is unknown, but each reading is one exact equation in
it.  With s the shift on a row (new head minus blend-5) and e = blend5 - y the
error blend-5 makes there, the change in the board's total squared error is

    dSSE = sum s^2  +  2 * sum s * e                                      (1)

The first term is the PRICE OF MOVING: it is paid whatever the direction, and
it is known exactly from the two vectors.  The second is the only part that
carries information about the truth, and (1) solves for it.  Writing

    rho = sum s*e / sum s^2                                              (2)

a shift pays only when rho < -0.5; rho = 0 is a change uncorrelated with
blend-5's error, which costs exactly its own size; rho > 0 is a change that
moves AWAY from the truth.  The oracle has rho = -1.

**rho is a property of a change AND the base it is made from**, and the two
directions are tied exactly: reversing a shift sends rho to -(rho + 1), because
the reversed shift is measured against an error that has itself moved by s.  A
change must therefore be read from the base it was actually made from.  Read
from v2, undoing tail v4's noac head has rho = +0.066; read the way the change
was really made -- ADDING that head to the earlier one -- the same numbers give
rho = -1.066, direction-correct and close to the oracle's -1.  That is why it
transferred at all, and it is the only direction-correct noac change on record
(desk, BOARD READING 5 CORRECTION).

score-feedback rule (desk, 2026-09-07) forbids any further upload designed to read a single
cell: steering noac predictions by board readings is learning from the ranking
process (`docs/input-rule.md`).  The cell table below is therefore a READING of
v4 and v5, not a plan.  Because dSSE is additive over disjoint rows, it says
what each cell contributed to the price of moving; it cannot say each cell's
rho, and no upload here may be built to find out.

Because squared error is per-row, dSSE is exactly additive over disjoint sets
of rows.  A probe that carries the new head on one cell only therefore measures
that cell's contribution exactly, and cells partition.

Cells are exp-002's own bands on `mvt_sched` (take-off minus schedule), the
variable its model orders the placeholder probability by.  Measured on the
1,507 noac rows of 2025, P(placeholder) runs 0.05 on the negative band up to
0.81 on 4-8h and 0.78 on 8h+.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402

N_BOARD = 344841
V2 = 265.9631
READINGS = {"v4 exp-002 head": ("exp002", 275.2197),
            "v5 tail-v4 undone": ("tailv4", 266.2918)}
BAND = [-1e18, 0, 1800, 3600, 7200, 14400, 28800, 1e18]
LAB = ["neg", "0-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8h+"]
#: P(placeholder) per band, measured on the 1,507 noac rows of 2025.
PH_2025 = {"neg": 0.053, "0-30m": 0.508, "30-60m": 0.410, "1-2h": 0.373,
           "2-4h": 0.590, "4-8h": 0.805, "8h+": 0.780}


def load() -> pd.DataFrame:
    sub = F.ROOT / "submissions" / "candidates"
    b5 = pd.read_parquet(F.ROOT / "submissions" / "uploaded" / "youthful-giraffe_v2.parquet")
    idc = [c for c in b5.columns if "ID" in c.upper()][0]
    vc = [c for c in b5.columns if c != idc][0]
    b5 = b5.rename(columns={idc: "MVT_ID_mvt", vc: "blend5"})[["MVT_ID_mvt", "blend5"]]
    b5["MVT_ID_mvt"] = b5["MVT_ID_mvt"].astype("float64")
    e2 = pd.read_parquet(F.ROOT / "experiments" / "exp-002-tail"
                         / "noac_ranking_predictions.parquet")
    e2["MVT_ID_mvt"] = e2["MVT_ID_mvt"].astype("float64")
    d = e2.rename(columns={"y_pred": "exp002"}).merge(b5, on="MVT_ID_mvt", how="left")
    if d["blend5"].isna().any():
        raise SystemExit("noac ids do not all appear in v2")
    t4 = pd.read_parquet(sub / "exp-026-noac-drop-tailv4.parquet")
    t4 = t4.rename(columns={t4.columns[0]: "MVT_ID_mvt", t4.columns[1]: "tailv4"})
    t4["MVT_ID_mvt"] = t4["MVT_ID_mvt"].astype("float64")
    d = d.merge(t4, on="MVT_ID_mvt", how="left")
    d["band"] = pd.cut(d.mvt_sched, BAND, labels=LAB, right=False)
    d["P_ph_2025"] = d.band.map(PH_2025).astype(float)
    return d


def solve(d: pd.DataFrame, col: str, score: float) -> dict:
    """Equation (1) over all moved rows: the price of moving, and rho."""
    s = (d[col] - d.blend5).to_numpy("float64")
    dsse = N_BOARD * (score ** 2 - V2 ** 2)
    price = float(np.sum(s ** 2))
    sse_term = (dsse - price) / 2.0          # = sum s*e
    noise_score = float(np.sqrt(V2 ** 2 + price / N_BOARD))
    return {"n_moved": int((s != 0).sum()),
            "shift_rms": float(np.sqrt(np.mean(s ** 2))),
            "dSSE": dsse, "dScore": score - V2,
            "price_of_moving_sse": price,
            "price_of_moving_score": noise_score - V2,
            "score_if_rho_zero": noise_score,
            "sum_s_e": sse_term,
            "rho_from_v2": sse_term / price,
            # the same change read from the other base; see the module docstring
            "rho_from_own_base": -(sse_term / price + 1.0),
            "pays_if_rho_below": -0.5}


def main() -> None:
    d = load()
    print(f"{len(d)} noac ranking rows\n")

    print("=== the hypothesis table: exp-002's mvt_sched bands ===")
    rows = []
    for lab, g in d.groupby("band", observed=False):
        s4 = (g.exp002 - g.blend5).to_numpy("float64")
        s5 = (g.tailv4 - g.blend5).to_numpy("float64")
        rows.append(dict(band=lab, n=len(g), P_ph_2025=PH_2025[lab],
                         mvt_sched_med=float(g.mvt_sched.median()) if len(g) else np.nan,
                         blend5_mean=float(g.blend5.mean()) if len(g) else np.nan,
                         exp002_mean=float(g.exp002.mean()) if len(g) else np.nan,
                         v4_shift_mean=float(s4.mean()) if len(g) else np.nan,
                         v4_shift_rms=float(np.sqrt(np.mean(s4 ** 2))) if len(g) else np.nan,
                         v4_price_sse=float(np.sum(s4 ** 2)),
                         v5_shift_rms=float(np.sqrt(np.mean(s5 ** 2))) if len(g) else np.nan,
                         v5_price_sse=float(np.sum(s5 ** 2))))
    T = pd.DataFrame(rows)
    T["v4_price_share"] = T.v4_price_sse / T.v4_price_sse.sum()
    T["v4_price_score"] = np.sqrt(V2 ** 2 + T.v4_price_sse / N_BOARD) - V2
    print(T.round(3).to_string(index=False))

    print("\n=== equation (1): the price of moving, and the direction ===")
    out = {}
    for lab, (col, score) in READINGS.items():
        r = solve(d, col, score)
        out[lab] = r
        print(f"\n{lab}: scored {score:.4f}, {r['dScore']:+.4f} on v2")
        print(f"  shift RMS                       {r['shift_rms']:9.1f} s over {r['n_moved']} rows")
        print(f"  price of moving (rho = 0)       {r['price_of_moving_score']:+9.4f} s "
              f"-> board would read {r['score_if_rho_zero']:.4f}")
        print(f"  observed                        {r['dScore']:+9.4f} s")
        print(f"  therefore rho, read from v2     {r['rho_from_v2']:+9.4f}   (pays only below -0.5)")
        print(f"  the same change from its own base{r['rho_from_own_base']:+9.4f}")
    # ---- the three-way regime, label-free and serve-time computable --------
    # rollover: mvt_sched >= 8 h AND the schedule crosses the local calendar day
    # (`prc.tail_features._sched_date_cross`; measured on 2025, all 11 rollovers
    # above 8 h cross and none of the 12 non-crossing rows is one).
    cf = F.EXP / "analysis" / "noac_ranking_cells.parquet"
    if cf.exists():
        c = pd.read_parquet(cf)
        c["MVT_ID_mvt"] = c["MVT_ID_mvt"].astype("float64")
        m = d.merge(c[["MVT_ID_mvt", "cell"]], on="MVT_ID_mvt", how="left")
        print("\n=== v4's price of moving, by exp-002's own three-way regime ===")
        rr = []
        for cell, g in m.groupby("cell"):
            s = (g.exp002 - g.blend5).to_numpy("float64")
            pr = float(np.sum(s ** 2))
            rr.append(dict(cell=cell, n=len(g), blend5_mean=float(g.blend5.mean()),
                           exp002_mean=float(g.exp002.mean()),
                           shift_mean=float(s.mean()),
                           shift_rms=float(np.sqrt(np.mean(s ** 2))),
                           price_sse=pr,
                           price_score=float(np.sqrt(V2 ** 2 + pr / N_BOARD) - V2)))
        R = pd.DataFrame(rr).sort_values("price_sse", ascending=False)
        R["share"] = R.price_sse / R.price_sse.sum()
        print(R.round(3).to_string(index=False))
        R.to_csv(F.EXP / "analysis" / "noac_regime_cells.csv", index=False)

    (F.EXP / "analysis" / "noac_cells.json").write_text(json.dumps(
        {"cells": json.loads(T.to_json(orient="records")), "board": out}, indent=2))
    T.to_csv(F.EXP / "analysis" / "noac_cells.csv", index=False)
    print("\nwritten: analysis/noac_cells.csv, analysis/noac_cells.json")


if __name__ == "__main__":
    main()
