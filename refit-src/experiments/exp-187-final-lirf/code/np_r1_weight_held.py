"""Repair read for the no-plan r1 rows with the R4mks base (desk, 2026-10-09 07:30 UTC): refit the base weight a in
max(60, a * base + (1 - a) * mix) on the held r1 rows of one month and score it on the other month, both ways.

The served form has a = 1 - W = 0.75 (exp-167 noplan_form.W = 0.25). Rows: the r1 rows of the 397 held LIRF no-plan rows
(`exp-167 results/noplan-form-v3-fold-001.parquet`), base = max(60, R4mks held F), mix = max(60, p ms + (1 - p) m_real).
The fit is a least-squares grid over a in [0, 1] (step 0.001), with the floor. The out-of-month read scores the January
rows with the July fit and the July rows with the January fit, against a = 0.75 on the same base (prc/gscore.py G, and
the MSE change per fired row and per held row). References: a = 0, 0.75 and 1 on all r1 rows, and the released-F base
at a = 0.75. Mac only.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/np_r1_weight_held.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import np_design_held as H  # noqa: E402

GS, C, NF = H.GS, H.C, H.NF
ID = H.ID
OUT = H.ROOT / "experiments/exp-187-final-lirf/out/v/np-r1-weight-held.json"
GRID = np.round(np.arange(0, 1.0005, 0.001), 3)


def r1_value(a: float, b: np.ndarray, mix: np.ndarray) -> np.ndarray:
    return np.maximum(C.FLOOR, a * b + (1 - a) * mix)


def fit(b: np.ndarray, mix: np.ndarray, y: np.ndarray) -> dict:
    sse = np.array([((y - r1_value(a, b, mix)) ** 2).sum() for a in GRID])
    k = int(np.argmin(sse))
    den = ((b - mix) ** 2).sum()
    return {"a": float(GRID[k]), "rows": int(len(y)), "sse_per_row": float(sse[k] / len(y)),
            "a_closed_form_no_floor": float(((b - mix) * (y - mix)).sum() / den) if den > 0 else None}


def main():
    assert H.sha(H.HELD) == H.HELD_SHA
    h = pd.read_parquet(H.HELD)
    f = pd.read_parquet(H.INP / "stack9-024/frame.parquet", columns=[ID, "y", "F"])
    ids = f[ID].to_numpy("int64")
    x = pd.concat([pd.read_parquet(H.INP / f"stack10-025/R4mksFjjrefit-month-2025-{m}.parquet", columns=[ID, "pred_Sb"]) for m in ("01", "07")])
    X = pd.Series(x.pred_Sb.to_numpy("float64"), index=x[ID].astype("int64")).reindex(ids).to_numpy()
    F = f.F.to_numpy("float64")
    y = GS.labels(ids)
    hid = h[ID].to_numpy("int64")
    pos = pd.Index(ids).get_indexer(hid)
    b_rel, b_r4 = np.maximum(C.FLOOR, F[pos]), np.maximum(C.FLOOR, X[pos])
    assert np.array_equal(b_rel, h.base.to_numpy(float))
    ms, p, mr = h.ms.to_numpy(float), h.p.to_numpy(float), h.m_real.to_numpy(float)
    mix = np.maximum(C.FLOOR, p * ms + (1 - p) * mr)
    r1 = (h.part == "r1").to_numpy()
    month = h.month.to_numpy()
    assert np.array_equal(r1_value(1 - NF.W, b_rel, mix)[r1], h.cal.to_numpy(float)[r1]), "a = 0.75 on the released base is not the saved value"
    yh = y[pos]
    fits = {m: fit(b_r4[r1 & (month == m)], mix[r1 & (month == m)], yh[r1 & (month == m)]) for m in (1, 7)}
    fits["both"] = fit(b_r4[r1], mix[r1], yh[r1])
    v_ref = h.cal.to_numpy(float).copy()                       # the R4mks design at a = 0.75: base swapped on r1 and c20
    tb_v = H.value(h, b_r4, {"r_lo": 87168.0, "r_hi": 88392.0})
    v_ref[:] = tb_v
    assert np.array_equal(v_ref[r1], r1_value(0.75, b_r4, mix)[r1])

    def vec(v: np.ndarray) -> np.ndarray:
        out = F.copy()
        out[pos] = v
        return out

    def read(v_new: np.ndarray, v_old: np.ndarray) -> dict:
        d = (yh - v_new) ** 2 - (yh - v_old) ** 2
        sc = GS.score(ids, vec(v_old), vec(v_new), y=y)
        by = {str(m): {"r1_rows": int((r1 & (month == m)).sum()), "mse_change_per_fired_row": float(d[r1 & (month == m)].mean()),
                       "mse_change_per_held_row": float(d[month == m].sum() / len(ids))} for m in (1, 7)}
        return {"mse_change_per_fired_row": float(d[r1].mean()), "mse_change_per_held_row": float(d.sum() / len(ids)),
                "G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                "G_without_top_days": sc["G_without_top_days"], "G_by_month": sc["views"]["month"], "by_month": by}

    oom = v_ref.copy()
    oom[r1 & (month == 1)] = r1_value(fits[7]["a"], b_r4, mix)[r1 & (month == 1)]
    oom[r1 & (month == 7)] = r1_value(fits[1]["a"], b_r4, mix)[r1 & (month == 7)]
    refs = {}
    for a in (0.0, 0.75, 1.0, fits["both"]["a"]):
        v = v_ref.copy()
        v[r1] = r1_value(a, b_r4, mix)[r1]
        refs[f"a={a:g}"] = read(v, v_ref)
    v_rel = v_ref.copy()
    v_rel[r1] = r1_value(0.75, b_rel, mix)[r1]
    rec = {"what": "refit of the r1 base weight a with the R4mks held base; out-of-month reads against a = 0.75",
           "sign": "mse change = new minus old (negative: better); G = old minus new (positive: better)",
           "fits": fits,
           "out_of_month_vs_a075": read(oom, v_ref),
           "in_sample_refs_vs_a075": refs,
           "released_base_a075_vs_r4mks_a075_on_r1": read(v_rel, v_ref),
           "note": "The out-of-month read scores January with the July fit and July with the January fit; G and the "
                   "per-held-row change are for both months together; by_month gives each direction."}
    H.OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps(rec, indent=1, default=str))


if __name__ == "__main__":
    main()
