"""exp-050-nmblind: the final predictors by regime (q08 step 5, "report ... each regime").

Arithmetic on frozen artifacts only, no fit: the pinned v22 panel (`nb_panel.load`, sha
checked) gives each row's regime and guard flag; `work/router-rows-<fold>.parquet` (sha
checked against the router JSON) gives every variant's update `u`, so a variant's
prediction is `p_v22 + u`.  For each fold and each regime segment it records rows, v22's
RMSE, and every variant's changed-row count and RMSE.  Two assertions: every variant
leaves every row outside the population at v22's bytes, and the segments' squared errors
add back to the router JSON's all-row RMSE.

    .venv/bin/python experiments/exp-050-nmblind/code/nb_regimes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
import nb_experts as NE                                               # noqa: E402
import nb_panel as NP                                                 # noqa: E402
import nooverwrite as NO                                              # noqa: E402

FOLDS = ["july_loo", "december", "january"]
ROUTER = EXP / "analysis/router-july_loo-december-january.json"
OUT = EXP / "analysis/regimes-july_loo-december-january.json"
VARIANTS = ["global", "router", "difficulty", "placebo", "repair", "oracle", "oracle_placebo"]


def segments(pan: pd.DataFrame) -> dict[str, np.ndarray]:
    reg = pan.regime.astype(str).to_numpy()
    held = pan.guard_held.to_numpy(bool)
    seg = {"nm_bulk, population P": (reg == "nm_bulk") & ~held,
           "nm_bulk, guard-held": (reg == "nm_bulk") & held}
    for r in ("nm_late", "nonm", "noac"):
        seg[r] = reg == r
    seg["guard-held, every regime"] = held
    part = sum(seg[k].astype(int) for k in ("nm_bulk, population P", "nm_bulk, guard-held",
                                             "nm_late", "nonm", "noac"))
    assert (part == 1).all(), "the five regime segments must partition the fold"
    return seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="replace the existing JSON")
    args = ap.parse_args()
    NO.refuse_existing([OUT], args.overwrite)
    rj = json.loads(ROUTER.read_text())
    rec = {"router_json": str(ROUTER.relative_to(ROOT)), "router_json_sha256": NE.sha_file(ROUTER),
           "folds": {}}
    for f in FOLDS:
        fr = rj["folds"][f]
        rows_path = ROOT / fr["router_rows"]
        if NE.sha_file(rows_path) != fr["router_rows_sha256"]:
            raise SystemExit(f"{rows_path.name} moved since the router JSON was written")
        pan = NP.load(f)
        rr = pd.read_parquet(rows_path)
        ids = pan.MVT_ID_mvt.to_numpy("float64")
        if not np.array_equal(ids, rr.MVT_ID_mvt.to_numpy("float64")):
            raise SystemExit(f"{f}: panel and router rows are not in the same row order")
        y, p0 = pan.y_true.to_numpy("float64"), pan.p_v22.to_numpy("float64")
        assert np.array_equal(y, rr.y_true.to_numpy("float64"))
        assert np.array_equal(p0.view("uint64"), rr.p_v22.to_numpy("float64").view("uint64"))
        pop = rr["pop"].to_numpy(bool)
        seg = segments(pan)
        assert np.array_equal(pop, seg["nm_bulk, population P"])
        r = y - p0
        out = {"rows": int(len(pan)), "segments": {}}
        for v in VARIANTS:
            u = rr[f"u_{v}"].to_numpy("float64")
            pred = p0 + u
            outside = ~pop
            assert np.array_equal(pred[outside].view("uint64"), p0[outside].view("uint64")), \
                f"{f} {v}: a row outside P moved"
            sse = sum(float(np.sum((r[m] - u[m]) ** 2)) for k, m in seg.items()
                      if k != "guard-held, every regime")
            want = fr["variants"][v]["rmse_new"]
            assert abs(np.sqrt(sse / len(r)) - want) <= 1e-9 * want, (f, v)
        for k, m in seg.items():
            s = {"rows": int(m.sum()),
                 "rmse_v22": float(np.sqrt(np.mean(r[m] ** 2))) if m.any() else None,
                 "variants": {}}
            for v in VARIANTS:
                u = rr[f"u_{v}"].to_numpy("float64")
                s["variants"][v] = {
                    "rows_changed": int((u[m] != 0).sum()),
                    "rmse_new": float(np.sqrt(np.mean((r[m] - u[m]) ** 2))) if m.any() else None}
            out["segments"][k] = s
            print(f"{f:9s} {k:26s} {s['rows']:>8,}  v22 {s['rmse_v22'] or float('nan'):9.3f}  "
                  + "  ".join(f"{v} {s['variants'][v]['rmse_new'] or float('nan'):.3f}"
                              f" ({s['variants'][v]['rows_changed']:,})" for v in VARIANTS[:5]),
                  flush=True)
        rec["folds"][f] = out
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
