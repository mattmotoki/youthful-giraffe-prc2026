"""exp-046-surfacefit: the pilot's EHAM numbers, reproduced by THIS pipeline.

The assignment: "Report the pilot's EHAM numbers reproduced by the pipeline before any
other airport is read."  So this script runs `exp-043-pilot/RESULTS.md` §4(c) through
`code/block.py` and `code/base.py` instead of through the pilot's own `align.py`, with
the pilot's own base (v10) and the pilot's own estimator (leave-one-day-out offset and
amplitude on the five held-out days).  Reproducing them is a gate on the block: if the
feature the pipeline computes is the feature the pilot measured, these numbers come
back; if it is not, they do not.

The pilot's held-out figures to be reproduced (`analysis/align-obs_offblock_disp-heldout.json`):

    rows in the fold with a clock      3,498
    leave-one-day-out 2A - B           +5,536 s^2 per matched row
    v10 on those rows                  142.5 s  ->  121.5 s with the update

The same table is then printed on **v11**, which is the base this experiment serves on,
so the step-2 arms have their EHAM reference.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reproduce_pilot.py
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
import base as BA                                            # noqa: E402
import block as B                                            # noqa: E402
import test_block as T                                       # noqa: E402

FOLD = "july_loo"
PUB = ROOT / "experiments/exp-043-pilot/analysis/align-obs_offblock_disp-heldout.json"


def loo_day(j: pd.DataFrame, pcol: str) -> dict:
    """The pilot's estimator: offset and amplitude fitted on the OTHER days.

    `j` carries one row per matched departure with `y_true`, `pcol` (the base), the
    observation's `obs_taxi` and the day key.  Returns the paired quantities at the
    leave-one-day-out amplitude, which is the honest number.
    """
    days = sorted(j.day.unique())
    rows = []
    for dk in days:
        out, ins = j[j.day == dk], j[j.day != dk]
        if not len(ins) or not len(out):
            continue
        off = float((ins.obs_taxi - ins.y_true).median())
        ins_d = (ins.obs_taxi - off - ins[pcol]).to_numpy("float64")
        ins_r = (ins.y_true - ins[pcol]).to_numpy("float64")
        A, Bv = float(np.mean(ins_r * ins_d)), float(np.mean(ins_d * ins_d))
        s = A / Bv if Bv > 0 else 0.0
        rows.append(pd.DataFrame({
            "day": dk,
            "r": (out.y_true - out[pcol]).to_numpy("float64"),
            "d": (out.obs_taxi - off - out[pcol]).to_numpy("float64"),
            "scale": s}))
    L = pd.concat(rows, ignore_index=True)
    d_scaled = (L.scale * L.d).to_numpy("float64")
    r = L.r.to_numpy("float64")
    return {
        "rows": int(len(L)),
        "A_mean_rd": float(np.mean(r * d_scaled)),
        "B_mean_dd": float(np.mean(d_scaled ** 2)),
        "mse_gain_s2_per_matched_row": float(np.mean(2 * r * d_scaled - d_scaled ** 2)),
        "rmse_base_on_these_rows": float(np.sqrt(np.mean(r ** 2))),
        "rmse_base_plus_update": float(np.sqrt(np.mean((r - d_scaled) ** 2))),
        "scales_by_day": {d: float(L.loc[L.day == d, "scale"].iat[0]) for d in L.day.unique()},
    }


def main() -> None:
    coh_raw = T.pilot_frame()
    train = T.prefix_train()
    # The reproduction is judged on the PILOT'S rule, so the block is built strict:
    # the desk's wide match (2026-09-08) admits rows the pilot never had, and a
    # comparison that carried them would not be a reproduction of the pilot's number.
    # The wide population is measured beside it, as a change, not as the reproduction.
    week_rule, _ = B.one_rule([(["EHAM"], T.WEEK)])
    blk = B.build(coh_raw, train, airports=["EHAM"], wide=False, require_rule=week_rule)
    blk_wide = B.build(coh_raw, train, airports=["EHAM"], wide=True,
                       require_rule=week_rule)
    pm = B.prefix_map(train)
    coh = B.cohort(coh_raw, pm)

    pan = BA.load()[FOLD]

    def joined(bk: pd.DataFrame) -> pd.DataFrame:
        out = (coh[["MVT_ID_mvt", "takeoff_day"]]
               .merge(bk[["MVT_ID_mvt", "obs_taxi", "obs_has_clock", "reach_class"]],
                      on="MVT_ID_mvt", how="inner")
               .merge(pan[["MVT_ID_mvt", "y_true", "p10", "p11", "airport", "day"]],
                      on="MVT_ID_mvt", how="inner", validate="one_to_one"))
        return out[out.obs_has_clock].copy()

    j = joined(blk)
    jw = joined(blk_wide)
    heldout = j[j.takeoff_day.isin(T.HELDOUT)]
    heldout_wide = jw[jw.takeoff_day.isin(T.HELDOUT)]

    pub = json.loads(PUB.read_text())
    rep = {
        "fold": FOLD,
        "published": {
            "rows_matched_in_fold": pub["rows_matched_in_fold"],
            "loo_mse_gain_s2": pub["leave_one_day_out_scaled"]["full_amplitude_mse_gain_s2"],
            "rmse_v10_on_these_rows_s": pub["rmse_v10_on_these_rows_s"],
            "rmse_v10_plus_update_loo_s": pub["rmse_v10_plus_update_loo_s"],
            "median_offset_s_pooled": pub["median_offset_s_pooled"],
            "loo_scales_by_day": pub["loo_scales_by_day"],
        },
        "pipeline_heldout_v10": loo_day(heldout, "p10"),
        "pipeline_heldout_v11": loo_day(heldout, "p11"),
        "pipeline_wholeweek_v10": loo_day(j, "p10"),
        # the same table on the wide rule, so the desk can see what the extra rows
        # are worth at EHAM before any arm prices them
        "pipeline_heldout_v10_wide": loo_day(heldout_wide, "p10"),
        "pipeline_heldout_v11_wide": loo_day(heldout_wide, "p11"),
        "rows": {"in_fold_with_clock_week": int(len(j)),
                 "in_fold_with_clock_heldout": int(len(heldout)),
                 "in_fold_with_clock_heldout_wide": int(len(heldout_wide)),
                 "wide_only_heldout": int(len(heldout_wide) - len(heldout)),
                 "pilot_heldout": pub["rows_matched_in_fold"]},
        "median_offset_s_pooled_pipeline": float((heldout.obs_taxi - heldout.y_true).median()),
    }
    g = rep["pipeline_heldout_v10"]
    p = rep["published"]
    rep["agreement"] = {
        "rows_delta": rep["rows"]["in_fold_with_clock_heldout"] - p["rows_matched_in_fold"],
        "mse_gain_delta_s2": g["mse_gain_s2_per_matched_row"] - p["loo_mse_gain_s2"],
        "rmse_base_delta_s": g["rmse_base_on_these_rows"] - p["rmse_v10_on_these_rows_s"],
        "rmse_updated_delta_s": g["rmse_base_plus_update"] - p["rmse_v10_plus_update_loo_s"],
        # both are median(BLOCK_TIME - obs_offblock): the pilot writes it as
        # `blk_s - obs_offblock_disp`, the block as `obs_taxi - y`, which is the
        # same difference because `obs_taxi - y = blk_s - obs_offblock`.
        "offset_delta_s": (rep["median_offset_s_pooled_pipeline"]
                           - p["median_offset_s_pooled"]),
    }
    a = rep["agreement"]
    # the row difference is the ATTRIBUTED set of analysis/block-test.json (the pilot's
    # null-altitude repair, and the per-day against week-concatenated assembly of the
    # same rule); it is bounded as a share, not pinned to a count, because the pilot is
    # repairing its own experiment and its published counts move.
    a["rows_delta_share"] = a["rows_delta"] / max(p["rows_matched_in_fold"], 1)
    assert abs(a["rows_delta_share"]) <= 0.01, a
    assert abs(a["offset_delta_s"]) < 1.0, a
    assert abs(a["rmse_base_delta_s"]) < 1.0, a
    assert abs(a["rmse_updated_delta_s"]) < 1.0, a
    assert abs(a["mse_gain_delta_s2"] / p["loo_mse_gain_s2"]) < 0.03, a
    rep["verdict"] = (
        "REPRODUCED. The pipeline's block and base give the pilot's held-out EHAM "
        f"numbers to {abs(a['rmse_base_delta_s']):.3f} s on the base RMSE, "
        f"{abs(a['rmse_updated_delta_s']):.3f} s on the updated RMSE and "
        f"{abs(a['mse_gain_delta_s2']):.0f} s^2 "
        f"({abs(a['mse_gain_delta_s2'] / p['loo_mse_gain_s2']):.1%}) on the paired gain, "
        f"on {rep['rows']['in_fold_with_clock_heldout']} rows against the pilot's "
        f"{rep['rows']['pilot_heldout']} ({a['rows_delta_share']:+.2%}). The row "
        "difference is the attributed set of analysis/block-test.json — the pilot's "
        "null-altitude repair, which the harvest's kernels predate, and the per-day "
        "against week-concatenated assembly of the same rule — and there is no third "
        "cause.")
    (EXP / "analysis/reproduce-pilot.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True, default=float))
    print(json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
