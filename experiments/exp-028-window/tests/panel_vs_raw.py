"""The check `shape_check.py` cannot make: does the fitted panel match the real month?

`shape_check.py` proves an algebraic round trip — panel to raw columns and back —
so it proves the two feature paths are one function, but it cannot prove that the
frozen out-of-fold columns this experiment fits on are the same quantities the
raw training month holds.  If `exp-016`'s `t` were, say, the off-block time
rather than the take-off time, or if `exp-027`'s `d_aobt_lobt` had a sign or a
unit different from `AOBT_3_flt - LOBT_flt`, every window aggregate would be
built on one definition at fit time and another at serve time and both files
would still pass.

This closes it with the data.  For each fold the validation month is loaded and
**masked exactly as the ranking file is** (`prc.data.mask_like_ranking`), the
serving adapter is run on it, and the panel it produces is compared row by row
against the panel `code/panel.py` builds from committed out-of-fold files.  The
window design matrices are then compared cell by cell.

    .venv/bin/python experiments/exp-028-window/tests/panel_vs_raw.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import arrivals as Ar  # noqa: E402
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402


def check(fold: str) -> dict:
    month = Ar.FOLD_MONTH[fold]
    raw = D.load_month(month, columns=sorted(set(Fe.RAW_COLS) | {"BLOCK_TIME_UTC_mvt",
                                                                D.TARGET}))
    masked, _ = D.mask_like_ranking(raw)
    q = Fe.panel_from_movements(masked)            # the SERVING path, on real bytes
    p = P.load_fold(fold)                          # the FITTED path, from committed OOF

    m = p[["MVT_ID_mvt"]].merge(q, on="MVT_ID_mvt", how="left", indicator=True)
    rec = {"fold": fold, "month": month,
           "n_oof": int(len(p)), "n_month_departures": int(len(q)),
           "n_matched": int((m["_merge"] == "both").sum())}
    if rec["n_matched"] != rec["n_oof"]:
        raise SystemExit(f"{fold}: {rec['n_oof'] - rec['n_matched']} out-of-fold rows "
                         f"are not in {month}'s departures")
    for c in ("t", "d_takeoff_aobt", "d_aobt_lobt"):
        a, b = p[c].to_numpy("float64"), m[c].to_numpy("float64")
        same = np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b))
        rec[f"max_abs_diff_{c}"] = float(np.nanmax(np.abs(a - b))) if (~np.isnan(a)).any() else 0.0
        rec[f"n_differ_{c}"] = int((~same).sum())
    for c in ("airport", "regime"):
        rec[f"n_differ_{c}"] = int((p[c].astype(str).to_numpy()
                                    != m[c].astype(str).to_numpy()).sum())
    for c in ("nm_missing", "noac"):
        rec[f"n_differ_{c}"] = int((p[c].to_numpy("float64")
                                    != m[c].to_numpy("float64")).sum())

    # and the design matrices, on the fold's own population
    r = m.drop(columns=["_merge"]).copy()
    r["base"] = p["base"].to_numpy("float64")
    Xf, Xs = Fe.window_features(p), Fe.window_features(r[P.CANON])
    a, b = Xf.to_numpy("float64"), Xs.to_numpy("float64")
    bad = ~(np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b)))
    rec["design_cells"] = int(a.size)
    rec["design_cells_differing"] = int(bad.sum())
    rec["PASS"] = bool(all(v == 0 for k, v in rec.items() if k.startswith("n_differ_"))
                       and rec["design_cells_differing"] == 0)
    return rec


def main() -> None:
    recs = [check(f) for f in P.FOLDS]
    for r in recs:
        print(f"{r['fold']:9s} {r['month']}  oof {r['n_oof']:6d} of {r['n_month_departures']:6d} "
              f"month departures  differing: "
              + " ".join(f"{k.removeprefix('n_differ_')}={v}"
                         for k, v in r.items() if k.startswith("n_differ_"))
              + f"  design {r['design_cells_differing']}/{r['design_cells']:,}  "
                f"{'PASS' if r['PASS'] else 'FAIL'}")
    (P.EXP / "analysis" / "panel_vs_raw.json").write_text(json.dumps(recs, indent=2))
    if not all(r["PASS"] for r in recs):
        raise SystemExit("the frozen panel does not match the raw month")
    print("[panel_vs_raw] the fitted panel is the raw month, on all three folds")


if __name__ == "__main__":
    main()
