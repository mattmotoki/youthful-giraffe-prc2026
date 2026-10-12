"""exp-043-pilot step 3(c): the observation's residual alignment on the contract.

The queue's shared measurement contract (`docs/pro-reasoning-2026-09-08/QUEUE.md`)
defines the paired quantities on the SERVED vector, not on a member:

    r = y - p_v10                         the current predictor's residual
    d = p_obs - p_v10                     what this observation would move
    A = mean(r*d),  B = mean(d^2),  full-amplitude MSE gain = 2A - B,
    unrestricted fitted scale = A / B     (never fitted and scored on the same rows)

with `p_obs = MVT_TIME - obs_offblock - median_offset`, the observation's own
reconstruction of the taxi time.  The fold-side v10 comes from exp-038
`code/vectors.py`, which rebuilds it from committed rotations and gates it on
exp-036's published clean triple; nothing is refitted here.

Both the median offset and the scale are fitted LEAVE-ONE-DAY-OUT: the day being
scored contributes to neither.  With one airport-week this is discovery evidence
with a wide band, and the day-clustered interval says how wide.

  .venv/bin/python experiments/exp-043-pilot/code/align.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(EXP / "code"))
import match as M                                          # noqa: E402
FOLD = "july_loo"
V10 = "v10 (exp-036 refitbulk-mask)"
RNG = np.random.default_rng(20260908)


def v10_july() -> pd.DataFrame:
    import vectors as V                                   # noqa: PLC0415
    ref, vec, prov, _ = V.load(verbose=False)
    if V10 not in vec:
        raise SystemExit(f"{V10} did not gate; exp-038 vectors.py says why")
    r = ref[FOLD]
    out = pd.DataFrame({"MVT_ID_mvt": r["MVT_ID_mvt"].to_numpy("float64"),
                        "y_true": r["y_true"].to_numpy("float64"),
                        "airport": r["airport"].astype(str).to_numpy(),
                        "p_v10": vec[V10][FOLD]})
    out.attrs["clean"] = prov[V10]["clean"]
    out.attrs["sha"] = prov[V10]["sha256_by_fold"][FOLD]
    return out


def paired(r: np.ndarray, d: np.ndarray) -> dict:
    A, B = float(np.mean(r * d)), float(np.mean(d * d))
    return {"rows": int(len(r)), "A_mean_rd": A, "B_mean_dd": B,
            "full_amplitude_mse_gain_s2": 2 * A - B,
            "fitted_scale_A_over_B": (A / B if B > 0 else None),
            "scaled_mse_gain_s2": (A * A / B if B > 0 else 0.0)}


DEV_DAYS = ("2025-07-07", "2025-07-08")     # the development side, SPEC.md


def _regime_move(a: dict, b: dict) -> dict:
    """How much the pooled-offset and leave-one-day-out-offset regimes differ.

    Reported per quantity, because `2A - B` is a near-cancellation and moves by far
    more in relative terms than either of the two numbers it is made of.
    """
    out = {"what": ("A, B and 2A-B under the offset pooled over the scored rows, "
                    "against the same three under a leave-one-day-out offset")}
    for k, name in (("A_mean_rd", "A"), ("B_mean_dd", "B"),
                    ("full_amplitude_mse_gain_s2", "2A_minus_B")):
        x, y = a[k], b[k]
        out[name] = {"pooled": round(x, 2), "leave_one_day_out": round(y, 2),
                     "relative_move_pct": round(100 * abs(x - y) / abs(x), 3)}
    out["largest_relative_move_pct"] = max(
        out[n]["relative_move_pct"] for n in ("A", "B", "2A_minus_B"))
    return out


def main(kind: str = "obs_offblock_strict", tag: str = "") -> None:
    suf = f"-{tag}" if tag else ""
    m = pd.read_parquet(EXP / f"work/matched{suf}.parquet")
    v = v10_july()
    j = m.merge(v, on="MVT_ID_mvt", how="inner", validate="one_to_one")
    j = j[j[kind].notna()].copy()
    j["dayk"] = j.day.dt.strftime("%Y-%m-%d")
    if not len(j):
        raise SystemExit(f"no matched row with {kind} joins the july_loo fold")

    j["r"] = j.y_true - j.p_v10
    days = sorted(j.dayk.unique())
    rows = []
    for d in days:
        out, ins = j[j.dayk == d], j[j.dayk != d]
        if not len(ins):
            continue
        off = float((ins.blk_s - ins[kind]).median())              # fitted off the day
        p_obs = out.mvt_s - out[kind] - off
        dd = (p_obs - out.p_v10).to_numpy("float64")
        rr = out.r.to_numpy("float64")
        ins_off = float((ins.blk_s - ins[kind]).median())
        ins_d = (ins.mvt_s - ins[kind] - ins_off - ins.p_v10).to_numpy("float64")
        ins_r = ins.r.to_numpy("float64")
        A, B = float(np.mean(ins_r * ins_d)), float(np.mean(ins_d * ins_d))
        s = A / B if B > 0 else 0.0
        rows.append(pd.DataFrame({"MVT_ID_mvt": out.MVT_ID_mvt.to_numpy(), "dayk": d,
                                  "r": rr, "d": dd, "scale_loo": s}))
    L = pd.concat(rows, ignore_index=True)

    off_all = float((j.blk_s - j[kind]).median())
    d_all = (j.mvt_s - j[kind] - off_all - j.p_v10).to_numpy("float64")
    r_all = j.r.to_numpy("float64")

    # The clean transport test the reviewer asked for: fit the median offset AND
    # the amplitude on the DEVELOPMENT days only, then score the held-out days
    # with both frozen.  Leave-one-day-out above cross-fits inside the held-out
    # set, which is valid but is not a days-1/2-to-days-3/7 statement.
    dev = None
    week_path = EXP / "work/matched-week.parquet"
    week = pd.read_parquet(week_path) if week_path.exists() else None
    if week is not None:
        week_fold = week.merge(v, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        week_fold = week_fold[week_fold[kind].notna()].copy()
        week_fold["dayk"] = week_fold.day.dt.strftime("%Y-%m-%d")
        dev_rows = week_fold[week_fold.dayk.isin(DEV_DAYS)]
        if len(dev_rows) and not j.dayk.isin(DEV_DAYS).any():
            dev_offset = float((dev_rows.blk_s - dev_rows[kind]).median())
            dev_d = (dev_rows.mvt_s - dev_rows[kind] - dev_offset
                     - dev_rows.p_v10).to_numpy("float64")
            dev_r = (dev_rows.y_true - dev_rows.p_v10).to_numpy("float64")
            dev_A, dev_B = float(np.mean(dev_r * dev_d)), float(np.mean(dev_d * dev_d))
            dev_scale = dev_A / dev_B if dev_B > 0 else 0.0
            scored_d = (j.mvt_s - j[kind] - dev_offset - j.p_v10).to_numpy("float64")
            dev = {"fitted_on": list(DEV_DAYS), "fit_rows": int(len(dev_rows)),
                   "median_offset_s": dev_offset, "scale": dev_scale,
                   "scored_rows": int(len(j))} | paired(r_all, dev_scale * scored_d)
            dev["rmse_v10_plus_update_s"] = float(
                np.sqrt(((r_all - dev_scale * scored_d) ** 2).mean()))
            # the assignment asks for a day-clustered interval on every number, and
            # the frozen arm is the headline, so it gets the same exact scheme the
            # accuracy table uses: all 5^5 resamples of the five scored days
            dev["day_interval_gain_s2"] = M.day_interval(
                pd.DataFrame({"r": r_all, "d": dev_scale * scored_d}),
                j.dayk.reset_index(drop=True),
                lambda f: 2 * float((f.r * f.d).mean()) - float((f.d * f.d).mean()))
            dev["day_interval_rmse_after_s"] = M.day_interval(
                pd.DataFrame({"r": r_all, "d": dev_scale * scored_d}),
                j.dayk.reset_index(drop=True),
                lambda f: float(np.sqrt(((f.r - f.d) ** 2).mean())))

    res = {
        "kind": kind, "fold": FOLD, "v10_sha256": v.attrs["sha"],
        "v10_clean_rmse": v.attrs["clean"],
        "rows_matched_in_fold": int(len(j)),
        "median_offset_s_pooled": off_all,
        # round 5: this used to assert "less than 0.02 %", which held for A and B
        # and not for 2A-B, where the two nearly cancel and a small absolute move
        # is a large relative one.  The moves are computed now, not asserted.
        "in_sample_offset_regime": _regime_move(
            paired(r_all, d_all), paired(L.r.to_numpy(), L.d.to_numpy())),
        "in_sample": paired(r_all, d_all),
        # d is already multiplied by the day's out-of-day scale, so this block's
        # 2A-B IS the honest MSE gain per matched row and its A/B is a check that
        # the scale transported (near 1 means it did)
        "frozen_on_development_days": dev,
        "leave_one_day_out_scaled": paired(L.r.to_numpy(), (L.scale_loo * L.d).to_numpy()),
        "leave_one_day_out_unscaled": paired(L.r.to_numpy(), L.d.to_numpy()),
        "loo_scales_by_day": {d: float(L.loc[L.dayk == d, "scale_loo"].iat[0])
                              for d in L.dayk.unique()},
        "rmse_v10_on_these_rows_s": float(np.sqrt((r_all ** 2).mean())),
        "rmse_obs_on_these_rows_s": float(np.sqrt(((r_all - d_all) ** 2).mean())),
        # the same result read as an RMSE: v10 on the matched rows, and v10 plus
        # the leave-one-day-out scaled update on the same rows
        "rmse_v10_plus_update_loo_s": float(np.sqrt(
            ((L.r - L.scale_loo * L.d) ** 2).mean())),
        # the assignment asks for a day-clustered interval on every number: for
        # the gain, the honest form with five clusters is the leave-one-day-out
        # range of the scored days
        "loo_gain_by_scored_day_s2": {
            d: paired(L.loc[L.dayk == d, "r"].to_numpy(),
                      (L.loc[L.dayk == d, "scale_loo"] * L.loc[L.dayk == d, "d"]).to_numpy()
                      )["full_amplitude_mse_gain_s2"] for d in sorted(L.dayk.unique())},
        "v10_serves_at_eham": ("stack-5, through v7: EHAM is not in exp-036's kept set "
                               "(EDDM EGLL LEBL LEMD LFPG LIRF LSZH LTFM), so the vector "
                               "this update would move at EHAM is v7's, which is what v10 "
                               "serves there"),
        "note": ("A one-airport week of the july_loo fold. `in_sample` fits the offset and "
                 "reads the scale on the same rows and is therefore optimistic; "
                 "`leave_one_day_out_scaled` is the honest one, because both the median "
                 "offset and the amplitude come from the other days only. Its "
                 "full_amplitude_mse_gain_s2 is the per-matched-row MSE the update buys."),
    }
    p = EXP / f"analysis/align-{kind}{suf}.json"
    p.write_text(json.dumps(res, indent=2, sort_keys=True, default=float))
    print(json.dumps(res, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "obs_offblock_strict",
         sys.argv[2] if len(sys.argv) > 2 else "")
