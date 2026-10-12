"""What is different about LIRF's July 2025 rows that every stronger arm pays on.

The desk, 2026-09-08: "the LIRF july tail -- what is different about LIRF's July 2025 fold
rows (receipt was 93 percent there) that every arm with more capacity or more columns pays
on".

`analysis/tail-lirf-held.json` established the fact: **every** july tail failure in the
fold-side push is one airport.  Zero LIRF's update and `c_nbr` goes -5.796 -> +2.389,
`c_geom_rwy` -4.515 -> +1.415, 255 leaves -3.043 -> +1.088, with every fold interval still
clear, and 164 of the 173 damaged july rows are LIRF's.  It did not say WHY.

This is a fold-side diagnosis on a month that carries labels, so `y` is used: the tail
clause is defined on `y > 3600 s` and the question is about that population.  Nothing here
becomes a serving rule -- the desk's 2026-09-08 19:05 decision stands, LIRF stays in -- so
the objection that closed the hold (a rule fitted off the scored rows) does not arise.

Four questions:

1. **Where is the damage?**  Per day and per row, the added MSE on the july tail, so a
   backlog day is separated from a month-long property.
2. **What does the observation say on those rows?**  If the ground run is SHORT while the
   label is enormous, the observation is telling the model the opposite of the truth, and a
   stronger model believes it harder -- which is exactly the observed capacity gradient.
3. **Is LIRF's july population different from LIRF's december one**, which the fit also
   sees, and from other airports' july rows?
4. **Does the damage grow with capacity?**  The five leaf settings are already priced and
   their vectors saved, so the gradient is free.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/lirfjuly.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

FOLD = "july_loo"
AP = "LIRF"
TAIL = 3600.0
CLIMB = ["c_lgb_shipped@leaves_15x600_lr050", "c_lgb_shipped",
         "c_lgb_shipped@leaves_63x300_lr050", "c_lgb_shipped@leaves_127x300_lr050",
         "c_lgb_shipped@leaves_255x300_lr050"]
COLS = ["obs_taxi", "obs_taxi_motion", "obs_seg_taxi", "dt_takeoff", "clock_ok",
        "seg_points", "seg_span_s", "seg_gap_max_s", "stand_dist_km", "stand_stop_s",
        "seen_before_move_s", "time_to_250m", "arr_present", "arr_stand_occupancy_s",
        "matched_on_modal", "obs_ambiguous"]
OUT = EXP / "analysis/lirf-july.json"


def med(v) -> float | None:
    v = np.asarray(v, dtype="float64")
    v = v[np.isfinite(v)]
    return round(float(np.median(v)), 2) if len(v) else None


def main() -> None:
    pan = A.panel_with_block(BA.load())
    p = pan[FOLD]
    y = p.y_true.to_numpy("float64")
    pb = p.pbase.to_numpy("float64")
    r = y - pb
    ap = p.airport.to_numpy().astype(str)
    day = p.day.astype(str).to_numpy()
    usable = p.usable.to_numpy(bool)
    tail = y > TAIL
    out = {"what": __doc__.strip().splitlines()[0], "fold": FOLD, "airport": AP,
           "tail": f"y > {TAIL:.0f} s"}

    d = np.load(S.npz_path("c_lgb_shipped@leaves_127x300_lr050"))[FOLD].astype("float64")
    added = (r - d) ** 2 - r ** 2          # >0 means the arm made this row worse

    # --- 1. where the damage is ---------------------------------------------
    out["tail_by_airport"] = {
        a: {"tail_rows": int(((ap == a) & tail).sum()),
            "tail_rows_moved": int(((ap == a) & tail & (d != 0)).sum()),
            "added_mse": round(float(added[(ap == a) & tail].sum()), 1)}
        for a in sorted(set(ap))}
    m = (ap == AP) & tail
    per = pd.DataFrame({"day": day[m], "added": added[m], "y": y[m], "r": r[m],
                        "d": d[m]}).groupby("day").agg(
        rows=("added", "size"), added=("added", "sum"),
        median_y=("y", "median"), median_r=("r", "median"), median_d=("d", "median"))
    per = per.sort_values("added", ascending=False)
    tot = float(per.added.sum())
    out["lirf_july_tail_by_day"] = {
        "days_with_a_tail_row": int(len(per)),
        "tail_rows": int(m.sum()),
        "added_mse_total": round(tot, 1),
        "share_in_the_worst_1_3_5_days": [round(float(per.added.head(k).sum() / tot), 4)
                                          for k in (1, 3, 5)] if tot else None,
        "worst_days": [{"day": str(i), "rows": int(x.rows),
                        "added_mse": round(float(x.added), 1),
                        "median_y": round(float(x.median_y), 1),
                        "median_r": round(float(x.median_r), 1),
                        "median_d": round(float(x.median_d), 1)}
                       for i, x in per.head(8).iterrows()]}

    # --- 2. what the observation says ---------------------------------------
    def profile(sel) -> dict:
        q = p[sel]
        rec = {"rows": int(sel.sum()),
               "with_clock": int(q.usable.sum()),
               "median_y": med(y[sel]), "median_pbase": med(pb[sel]),
               "median_r": med(r[sel]), "median_d": med(d[sel])}
        s2 = sel & usable
        for c in COLS:
            if c in p:
                rec[c] = med(pd.to_numeric(p[c], errors="coerce").to_numpy("float64")[s2])
        v = pd.to_numeric(p.obs_taxi, errors="coerce").to_numpy("float64")
        ok = s2 & np.isfinite(v)
        if ok.sum() > 20:
            rec["corr_obs_taxi_with_y"] = round(float(np.corrcoef(v[ok], y[ok])[0, 1]), 3)
            rec["corr_obs_taxi_with_r"] = round(float(np.corrcoef(v[ok], r[ok])[0, 1]), 3)
            rec["median_y_minus_obs_taxi"] = med(y[ok] - v[ok])
        return rec

    out["profiles"] = {
        "LIRF july tail": profile((ap == AP) & tail),
        "LIRF july bulk": profile((ap == AP) & ~tail),
        "other airports july tail": profile((ap != AP) & tail),
        "other airports july bulk": profile((ap != AP) & ~tail),
    }
    for f in ("december", "january"):
        q = pan[f]
        yy = q.y_true.to_numpy("float64")
        aa = q.airport.to_numpy().astype(str)
        s = (aa == AP) & (yy > TAIL)
        vv = pd.to_numeric(q.obs_taxi, errors="coerce").to_numpy("float64")
        uu = q.usable.to_numpy(bool)
        ok = s & uu & np.isfinite(vv)
        out["profiles"][f"LIRF {f} tail"] = {
            "rows": int(s.sum()), "with_clock": int((s & uu).sum()),
            "median_y": med(yy[s]),
            "median_y_minus_obs_taxi": med(yy[ok] - vv[ok]) if ok.sum() else None,
            "corr_obs_taxi_with_y": (round(float(np.corrcoef(vv[ok], yy[ok])[0, 1]), 3)
                                     if ok.sum() > 20 else None)}

    # --- 4. the capacity gradient, from the saved vectors --------------------
    grad = {}
    for arm in CLIMB:
        z = S.npz_path(arm)
        if not z.exists():
            continue
        dv = np.load(z)[FOLD].astype("float64")
        ad = (r - dv) ** 2 - r ** 2
        grad[arm] = {
            "added_mse_on_the_lirf_july_tail": round(float(ad[(ap == AP) & tail].sum()), 1),
            "added_mse_on_every_other_july_tail": round(
                float(ad[(ap != AP) & tail].sum()), 1),
            "median_update_on_the_lirf_july_tail": med(dv[(ap == AP) & tail]),
            "rows_moved_on_the_lirf_july_tail": int(((ap == AP) & tail & (dv != 0)).sum())}
    out["capacity_gradient"] = grad

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("tail rows and added MSE by airport:")
    for a, v in out["tail_by_airport"].items():
        print(f"  {a} tail {v['tail_rows']:4d} moved {v['tail_rows_moved']:4d} "
              f"added {v['added_mse']:+14.1f}")
    print("\nLIRF july tail by day:", json.dumps(
        {k: v for k, v in out["lirf_july_tail_by_day"].items() if k != "worst_days"}))
    for w in out["lirf_july_tail_by_day"]["worst_days"]:
        print("   ", w)
    print("\nprofiles:")
    for k, v in out["profiles"].items():
        print(f"  {k:28s} rows {v['rows']:5d} y {v['median_y']} "
              f"obs_taxi {v.get('obs_taxi')} y-obs {v.get('median_y_minus_obs_taxi')} "
              f"corr(obs,y) {v.get('corr_obs_taxi_with_y')} r {v.get('median_r')} "
              f"d {v.get('median_d')}")
    print("\ncapacity gradient:")
    for k, v in grad.items():
        print(f"  {k:42s} lirf {v['added_mse_on_the_lirf_july_tail']:+14.1f} "
              f"others {v['added_mse_on_every_other_july_tail']:+12.1f} "
              f"median d {v['median_update_on_the_lirf_july_tail']}")
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
