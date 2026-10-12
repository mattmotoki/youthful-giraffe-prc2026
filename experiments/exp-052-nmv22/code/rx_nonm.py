"""exp-052-nmv22: the no-NM rows of COMMON.md, priced for every predictor (Sol round 1, B2).

COMMON.md defines the no-NM rows as the departures with no NM join: every `_flt` column null.
The tables rendered before the review showed the `nonm` regime alone under "no-NM".  The
regimes (`prc.blend.regimes_from_raw`) key on `FLIGHT_ID_mvt` and `AIRCRAFT_TYPE_mvt`, not on
the `_flt` columns: every `nonm` and `noac` row has no NM join, and a few `nm_bulk` rows (a
flight id, but every `_flt` value null) have none either.  Per fold this script:

1. builds the COMMON.md mask directly from the raw month, in the panel's row order, and
   requires nonm | noac to lie inside it; the rest of the mask is counted by regime and by P;
2. for iteration 1 and 2, real and NULL, prices every predictor (p_v22 + u, u = 0 outside P)
   on the mask: rows, rows changed, RMSE, rows above 3600 s.  Each predictor's all-row RMSE
   must equal its analysis JSON's to 1e-9, and on nonm | noac every predictor must keep v22's
   bytes and match the RMSE pooled from the JSON's `nonm` and `noac` segments.

Arithmetic on recorded vectors; no fit.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_nonm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402

D, NO = C.D, C.NO
RUNS = {"iter1": ("iter1.json", None), "iter1-null": ("iter1-null.json", None),
        "iter2": ("iter2.json", "selected"), "iter2-null": ("iter2-null.json", "selected")}


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def pooled(seg: dict, key: str) -> float:
    a, b = seg["nonm"], seg["noac"]
    return float(np.sqrt((a["rows"] * a[key] ** 2 + b["rows"] * b[key] ** 2)
                         / (a["rows"] + b["rows"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = C.EXP / "analysis/nonm.json"
    NO.refuse_existing([OUT], args.overwrite)
    rec = {"what": "no-NM rows (COMMON.md: no NM join, every _flt column null), every predictor",
           "code_sha256": C.sha_file(Path(__file__)), "folds": {}}
    J = {}
    for run, (name, sub) in RUNS.items():
        p = C.EXP / "analysis" / name
        j = json.loads(p.read_text())
        J[run] = (j[sub] if sub else j, C.sha_file(p))
    for f in C.FOLDS:
        month = D.FOLDS[f].valid_month
        flt = [c for c in pq.read_schema(D.month_file(month)).names if c.endswith("_flt")]
        raw = D.load_month(month, ["MVT_ID_mvt", "PHASE_mvt"] + flt)
        dep = raw[raw.PHASE_mvt == "DEP"]
        direct = pd.Series(dep[flt].isna().all(axis=1).to_numpy(),
                           index=dep.MVT_ID_mvt.to_numpy("float64"))
        if not direct.index.is_unique:
            raise SystemExit(f"{f}: departure ids are not unique")
        pan = C.panel(f, ["MVT_ID_mvt", "regime", "guard_held"])
        ids = pan.MVT_ID_mvt.to_numpy("float64")
        if not np.isin(ids, direct.index.to_numpy()).all():
            raise SystemExit(f"{f}: a panel row is not a departure of {month}")
        M = direct.reindex(ids).to_numpy(bool)
        reg = pan.regime.astype(str).to_numpy()
        U = np.isin(reg, ["nonm", "noac"])
        if np.any(U & ~M):
            raise SystemExit(f"{f}: a nonm or noac row has an NM join")
        Pm = C.population(pan, f)
        fr = {"month": month, "flt_columns": flt, "panel_rows": int(len(ids)),
              "no_nm_rows": int(M.sum()), "nonm_or_noac_rows": int(U.sum()),
              "rows_by_regime": {g: int((M & (reg == g)).sum()) for g in sorted(set(reg))},
              "rows_in_P": int((M & Pm).sum()), "runs": {}}
        for run, (j, jsha) in J.items():
            info = j["folds"][f]
            rp = C.ROOT / info["rows_file"]
            if C.sha_file(rp) != info["rows_sha256"]:
                raise SystemExit(f"{rp.name} moved since its analysis JSON was written")
            rows = pd.read_parquet(rp)
            if not np.array_equal(rows.MVT_ID_mvt.to_numpy("float64"), ids):
                raise SystemExit(f"{f} {run}: the rows file is not the panel's order")
            if not np.array_equal(rows["pop"].to_numpy(bool), Pm):
                raise SystemExit(f"{f} {run}: the rows file's P is not the panel's")
            y, p22 = rows.y_true.to_numpy("float64"), rows.p_v22.to_numpy("float64")
            out = {"analysis_json_sha256": jsha, "rows_file": info["rows_file"],
                   "rows_gt3600": int((y[M] > 3600).sum()),
                   "v22": {"rmse_s": rmse((y - p22)[M]),
                           "rmse_nonm_or_noac_s": rmse((y - p22)[U])}}
            for c in (c for c in rows.columns if c.startswith("u_")):
                n = c[2:]
                u = rows[c].to_numpy("float64")
                if np.any(u[~Pm] != 0):
                    raise SystemExit(f"{f} {run} {n}: a row outside P moved")
                pred = p22 + u
                ref = info["predictors"][n]["rmse_new"]
                if abs(rmse(y - pred) - ref) > 1e-9:
                    raise SystemExit(f"{f} {run} {n}: all-row RMSE misses the JSON's {ref}")
                pu = pooled(info["predictors"][n]["segments"], "rmse_new")
                if abs(rmse((y - pred)[U]) - pu) > 1e-9 or np.any(u[U] != 0):
                    raise SystemExit(f"{f} {run} {n}: nonm | noac disagrees with the JSON")
                out[n] = {"rmse_s": rmse((y - pred)[M]), "rows_changed": int((u[M] != 0).sum()),
                          "rmse_nonm_or_noac_s": pu}
            fr["runs"][run] = out
        rec["folds"][f] = fr
        o = fr["runs"]["iter1"]
        print(f"{f}: {fr['no_nm_rows']:,} no-NM rows {fr['rows_by_regime']}, {fr['rows_in_P']} "
              f"in P; v22 {o['v22']['rmse_s']:.3f} s (nonm | noac {fr['nonm_or_noac_rows']:,} "
              f"rows, {o['v22']['rmse_nonm_or_noac_s']:.3f} s); iter1 router "
              f"{o['router']['rmse_s']:.3f}, iter2 router* "
              f"{fr['runs']['iter2']['router']['rmse_s']:.3f}", flush=True)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
