"""exp-046-surfacefit: a label-free check on the January observation, before serving.

The dry run of `code/serve.py` moved the 2026-01-01/02 EHAM rows at an update RMS of
275 s against 81 s on the July fold rows -- 3.4x, which is outside the assignment's
"within a third" serving gate.  This asks whether that is January or whether it is a
defect, using nothing but inputs:

    obs_taxi - (MVT_TIME - AOBT_3)

is the observation's disagreement with the Network Manager off-block on the SAME row.
It needs no label, so it can be computed on a ranking month and on a training month and
compared directly.  If January's disagreement is simply wider, the observation is seeing
a real winter effect (de-icing, remote pads, low-visibility pushbacks -- the pilot's own
"January is unmeasured" caveat) and the fix is a January-fitted offset and weight.  If
the disagreement is the same width but the update is not, the defect is in the serving
path.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/janwarn.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import arms as A                                             # noqa: E402
import base as BA                                            # noqa: E402
from imports import load as _load                            # noqa: E402

S = _load("exp-046-surfacefit", "serve")


def describe(x: pd.Series) -> dict:
    x = x.dropna()
    return {"rows": int(len(x)), "median": float(x.median()),
            "iqr": float(x.quantile(.75) - x.quantile(.25)),
            "rmse_about_median": float(np.sqrt(((x - x.median()) ** 2).mean())),
            "p01": float(x.quantile(.01)), "p99": float(x.quantile(.99))}


def main() -> None:
    pan = A.panel_with_block(BA.load())["july_loo"]
    jul = pan[pan.usable & (pan.airport == "EHAM")].copy()
    jul["aobt_taxi"] = jul.d_takeoff_aobt
    dep, _ = S.ranking_block()
    dep["aobt_taxi"] = dep.mvt_s - (dep.AOBT_3_flt
                                    - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    rk = dep[dep.usable & (dep.airport == "EHAM")].copy()
    rk["month"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")

    rep = {"question": ("is the 3.4x wider served update a January effect or a defect? "
                        "obs_taxi - (MVT_TIME - AOBT_3) uses no label."),
           "july_2025_fold_EHAM": {
               "obs_taxi": describe(jul.obs_taxi),
               "aobt_taxi": describe(jul.aobt_taxi),
               "disagreement": describe(jul.obs_taxi - jul.aobt_taxi)},
           "ranking_EHAM_by_month": {}}
    for m, g in rk.groupby("month"):
        rep["ranking_EHAM_by_month"][m] = {
            "days": sorted(g.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique().tolist()),
            "obs_taxi": describe(g.obs_taxi),
            "aobt_taxi": describe(g.aobt_taxi),
            "disagreement": describe(g.obs_taxi - g.aobt_taxi)}
    a = rep["july_2025_fold_EHAM"]["disagreement"]["rmse_about_median"]
    for m, r in rep["ranking_EHAM_by_month"].items():
        b = r["disagreement"]["rmse_about_median"]
        r["width_ratio_to_july"] = b / a if a else None
        r["median_shift_vs_july_s"] = (r["disagreement"]["median"]
                                       - rep["july_2025_fold_EHAM"]["disagreement"]["median"])
    (EXP / "analysis/janwarn.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                          default=float))
    print(json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
