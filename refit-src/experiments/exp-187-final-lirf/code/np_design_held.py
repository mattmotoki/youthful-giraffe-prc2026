"""The np_base design change through the no-plan rule on the held January/July 2025 rows (desk, 2026-10-09 07:23 UTC).

np_v3cal with the released held F (fold anchor, `stack9-024/frame.parquet` F) against np_v3cal with finalroute's R4mks
held F (`stack10-025/R4mksFjjrefit-month-2025-{01,07}.parquet` pred_Sb, the f-cost / v32-cost input). The 397 held
no-plan rows and their parts come from the saved fold rows (`exp-167 results/noplan-form-v3-fold-001.parquet`). The part
of a row does not depend on the base; the base enters only r1 (max(60, 0.75 base + 0.25 mix)), c20 (the base clipped to
[min(ms, r_lo), max(ms, r_hi)], ten-month tables) and base (none on the fold). The old values are rebuilt from the saved
inputs and must equal the saved v3 and cal; then the base is swapped. Score: prc/gscore.py (G, airport-day interval).

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/np_design_held.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-167-target/code"))
from prc import gscore as GS  # noqa: E402
import common as C  # noqa: E402
import noplan_form as NF  # noqa: E402
import noplan_form_v2 as V2  # noqa: E402

ID = "MVT_ID_mvt"
TEN = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
INP = ROOT / "experiments/exp-169-stack-plan/inputs"
HELD = ROOT / "experiments/exp-167-target/results/noplan-form-v3-fold-001.parquet"
HELD_SHA = "5b6855d93e35b5ba03c2d33adfa1df88c6119c0b726b79814e2535b906726bff"
NB26 = ROOT / "experiments/exp-187-final-lirf/out/final-npee84b9f3/noplan-rows.parquet"
OUT = ROOT / "experiments/exp-187-final-lirf/out/v/np-design-held.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def value(h: pd.DataFrame, b: np.ndarray, tb: dict) -> np.ndarray:
    ms, p, mr = h.ms.to_numpy(float), h.p.to_numpy(float), h.m_real.to_numpy(float)
    mix = np.maximum(C.FLOOR, p * ms + (1 - p) * mr)
    c20 = np.clip(b, np.minimum(ms, tb["r_lo"]), np.maximum(ms, tb["r_hi"]))
    part = h.part.to_numpy()
    v = h.cal.to_numpy(float).copy()
    v = np.where(part == "r1", np.maximum(C.FLOOR, (1 - NF.W) * b + NF.W * mix), v)
    v = np.where(part == "c20", np.maximum(C.FLOOR, c20), v)
    v = np.where(part == "base", np.maximum(C.FLOOR, b), v)
    return v


def main():
    assert sha(HELD) == HELD_SHA
    h = pd.read_parquet(HELD)
    hid = h[ID].to_numpy("int64")
    f = pd.read_parquet(INP / "stack9-024/frame.parquet", columns=[ID, "y", "F"])
    ids = f[ID].to_numpy("int64")
    assert len(ids) == 344_419 and set(hid) <= set(ids)
    x = pd.concat([pd.read_parquet(INP / f"stack10-025/R4mksFjjrefit-month-2025-{m}.parquet", columns=[ID, "pred_Sb"]) for m in ("01", "07")])
    X = pd.Series(x.pred_Sb.to_numpy("float64"), index=x[ID].astype("int64")).reindex(ids).to_numpy()
    F = f.F.to_numpy("float64")
    assert np.isfinite(X).all() and np.isfinite(F).all()
    pos = pd.Index(ids).get_indexer(hid)
    b_old, b_new = np.maximum(C.FLOOR, F[pos]), np.maximum(C.FLOOR, X[pos])
    assert np.array_equal(b_old, h.base.to_numpy(float)), "saved base is not max(60, released held F)"
    t25 = C.raw_2025()
    d25 = V2.add_v2(NF.classes(NF.noplan_rows(t25)), V2.arrival_share(t25))
    tb = NF.tables(d25[d25.month.isin(TEN)])
    v_old = value(h, b_old, tb)
    assert np.array_equal(v_old, h.cal.to_numpy(float)), "the rebuilt values differ from the saved cal"
    rb = h.part.isin(["r1", "c20", "base"]).to_numpy()
    assert np.array_equal(v_old[rb], h.v3.to_numpy(float)[rb])
    v_new = value(h, b_new, tb)
    y = GS.labels(ids)
    assert np.allclose(y, f.y.to_numpy(float))
    yh = y[pos]
    d = (yh - v_new) ** 2 - (yh - v_old) ** 2                   # MSE change per row: negative = R4mks better
    base_vec = F.copy()
    base_vec[pos] = v_old
    cand = base_vec.copy()
    cand[pos] = v_new
    sc = GS.score(ids, base_vec, cand, y=y)
    month = h.month.to_numpy()
    days = pd.read_parquet(ROOT / "data/raw/training_2025-01-01_2025-02-01.parquet", columns=[ID, "MVT_TIME_UTC_mvt"])
    days = pd.concat([days, pd.read_parquet(ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet", columns=[ID, "MVT_TIME_UTC_mvt"])])
    day = pd.Series(days.MVT_TIME_UTC_mvt.dt.date.astype(str).to_numpy(), index=days[ID].astype("int64")).reindex(hid).to_numpy()
    by = lambda m: {"rows": int(m.sum()), "moved": int((m & (v_new != v_old)).sum()), "mse_change_per_row": float(d[m].mean()) if m.any() else None,  # noqa: E731
                    "mse_change_per_all_held_row": float(d[m].sum() / len(ids))}
    rec = {"what": "np_v3cal with R4mks held F against np_v3cal with the released held F, 397 held LIRF no-plan rows (January/July 2025)",
           "sign": "mse change = new minus old (negative: R4mks is better); gscore G = old minus new (positive: R4mks is better)",
           "inputs_sha256": {"held_rows": HELD_SHA, "frame": sha(INP / "stack9-024/frame.parquet"),
                             **{f"R4mks-{m}": sha(INP / f"stack10-025/R4mksFjjrefit-month-2025-{m}.parquet") for m in ("01", "07")}},
           "tables": {"r_lo": tb["r_lo"], "r_hi": tb["r_hi"]}, "rebuild_old_equals_saved_cal": True,
           "base_move_on_397": {"rms": float(np.sqrt(((b_new - b_old) ** 2).mean())), "bias": float((b_new - b_old).mean())},
           "value_move_on_397": {"rows": int((v_new != v_old).sum()), "rms": float(np.sqrt(((v_new - v_old) ** 2).mean())),
                                 "bias": float((v_new - v_old).mean())},
           "all": by(np.ones(len(h), bool)), "by_month": {str(m): by(month == m) for m in (1, 7)},
           "by_part": {p: by(h.part.to_numpy() == p) for p in sorted(h.part.unique())},
           "by_airport": {"LIRF": by(np.ones(len(h), bool))},
           "gscore": {"G": sc["G"], "lo95": sc["interval"]["lo95"], "hi95": sc["interval"]["hi95"],
                      "G_without_top_days": sc["G_without_top_days"], "months": sc["views"]["month"]},
           "by_day": {str(k): float(v) for k, v in pd.Series(d).groupby(day).sum().sort_values().items()}}
    top = np.argsort(-np.abs(d))[:5]
    rec["top_rows"] = [{"MVT_ID_mvt": int(hid[i]), "part": h.part.iloc[i], "y": float(yh[i]), "old": float(v_old[i]),
                        "new": float(v_new[i]), "mse_change": float(d[i])} for i in top]
    n26 = pd.read_parquet(NB26)
    rec["rows_2026"] = {str(m): {"noplan": int((n26.month == m).sum()), "base_parts_r1_c20": int(((n26.month == m) & n26.part.isin(["r1", "c20", "base"])).sum())}
                        for m in (1, 2, 6, 7)}
    OUT.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("base_move_on_397", "value_move_on_397", "all", "by_month", "by_part", "gscore", "rows_2026")}, indent=1, default=str))


if __name__ == "__main__":
    main()
