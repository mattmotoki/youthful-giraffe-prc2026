"""Why the served update at LFPG is 1.49x the fold side's, measured without a label.

The RMS band gate compares the served update's per-airport RMS with the fold side's; at
LFPG alone the ratio is 1.487 (116.38 s served against 78.24 fold) and the assignment's
clause is "explain before the desk uploads".  The explanation has to be LABEL-FREE, since
the ranking months carry no `TAXITIME_SEC_mvt`, so it is built from three quantities that
exist on both sides: the spread of the observation itself, its disagreement with the base
prediction (which is what the arm corrects), and its disagreement with the Network
Manager's own off-block, which is `exp-046`'s `janwarn` instrument.
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

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402


def spread(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"n": int(len(v)), "median": float(q[2]),
            "iqr": float(q[3] - q[1]), "p5_p95": float(q[4] - q[0]),
            "rms_about_the_median": float(np.sqrt(np.mean((v - q[2]) ** 2)))}


def main(airports: tuple[str, ...] = ("LFPG", "EHAM", "EDDF")) -> None:
    pan = A.panel_with_block(BA.load())
    dep, _ = S.ranking_block()
    base = pd.read_parquet(S.BASE_FILE).set_index("MVT_ID_mvt")[D.TARGET]
    dep["pbase"] = dep.MVT_ID_mvt.map(base).to_numpy("float64")
    out: dict = {"why": __doc__.strip().splitlines()[0], "by_airport": {}}
    for ap in airports:
        rec: dict = {"fold": {}, "ranking": {}}
        for f in A.FOLDS:
            p = pan[f]
            m = (p.airport.astype(str) == ap).to_numpy() & p.usable.to_numpy(bool)
            if not m.any():
                rec["fold"][f] = {"n": 0}
                continue
            g = p[m]
            rec["fold"][f] = {
                "obs_taxi": spread(g.obs_taxi.to_numpy("float64")),
                "obs_minus_base": spread((g.obs_taxi - g.pbase).to_numpy("float64")),
                "obs_minus_nm": spread(
                    (g.obs_taxi - (g.MVT_TIME_UTC_mvt - g.AOBT_3_flt).dt.total_seconds()
                     ).to_numpy("float64")) if "AOBT_3_flt" in g else {"n": 0},
            }
        m = (dep.airport.astype(str) == ap).to_numpy() & dep.usable.to_numpy(bool)
        for mon in ("2026-01", "2026-07"):
            mm = m & (dep.month.astype(str) == mon).to_numpy()
            if not mm.any():
                rec["ranking"][mon] = {"n": 0}
                continue
            g = dep[mm]
            rec["ranking"][mon] = {
                "obs_taxi": spread(g.obs_taxi.to_numpy("float64")),
                "obs_minus_base": spread((g.obs_taxi - g.pbase).to_numpy("float64")),
                "obs_minus_nm": spread(
                    (g.obs_taxi - (g.MVT_TIME_UTC_mvt - g.AOBT_3_flt).dt.total_seconds()
                     ).to_numpy("float64")) if "AOBT_3_flt" in g else {"n": 0},
            }
        out["by_airport"][ap] = rec
    # THE DECIDING TEST (desk, 2026-09-08 18:40): is the excess the INSTRUMENT or the
    # POPULATION?  LFPG's 2026-01 receipt falls from 24 % to 5 % across the month, and a
    # row heard on a 5 % day is heard by chance -- its ground run is more likely to be a
    # fragment, and a fragment's `obs_offblock` is noisier.  So the observation's own
    # QUALITY features are compared between the fold rows and the ranking rows, and the
    # ranking rows are split by their day's receipt.  If the spread rises as receipt falls
    # and the quality features degrade with it, the excess is the instrument and the
    # airport is held; if the rows look like the fold's rows, it is the population and the
    # breach is accepted.
    qcols = [c for c in ("seg_points", "n_ground_points", "seg_gap_max_s",
                         "seg_dt_median_s", "seg_span_s", "obs_clock_dt")
             if c in dep.columns]
    qual: dict = {"columns": qcols, "by_airport": {}}
    for ap in airports:
        rec = {"fold": {}, "ranking": {}}
        for f in A.FOLDS:
            g = pan[f]
            m = (g.airport.astype(str) == ap).to_numpy() & g.usable.to_numpy(bool)
            rec["fold"][f] = {c: float(np.nanmedian(g.loc[m, c])) if m.any() and c in g
                              else None for c in qcols}
            rec["fold"][f]["n"] = int(m.sum())
        m = (dep.airport.astype(str) == ap).to_numpy() & dep.usable.to_numpy(bool)
        for mon in ("2026-01", "2026-07"):
            mm = m & (dep.month.astype(str) == mon).to_numpy()
            rec["ranking"][mon] = {c: float(np.nanmedian(dep.loc[mm, c])) if mm.any()
                                   else None for c in qcols}
            rec["ranking"][mon]["n"] = int(mm.sum())
        # and the spread by the DAY's own match share, which stands in for receipt
        g = dep[m].copy()
        if len(g):
            g["day"] = g.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
            allr = dep[(dep.airport.astype(str) == ap).to_numpy()].copy()
            allr["day"] = allr.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
            share = allr.groupby("day").usable.mean()
            g["day_share"] = g.day.map(share)
            g["ob"] = (g.obs_taxi - g.pbase).to_numpy("float64")
            bins = pd.cut(g.day_share, [0, .05, .10, .20, .50, 1.01],
                          labels=["<5%", "5-10%", "10-20%", "20-50%", ">50%"])
            rec["by_day_receipt"] = {
                str(k): {"rows": int(len(v)),
                         "obs_minus_base_rms_about_median": float(
                             np.sqrt(np.mean((v.ob - v.ob.median()) ** 2))),
                         "obs_minus_base_iqr": float(v.ob.quantile(.75) - v.ob.quantile(.25))}
                for k, v in g.groupby(bins, observed=True)}
        qual["by_airport"][ap] = rec
    out["observation_quality_and_receipt"] = qual
    for ap, rec in qual["by_airport"].items():
        print(f"-- {ap} quality medians")
        for side, d in (("fold", rec["fold"]), ("ranking", rec["ranking"])):
            for k, v in d.items():
                print(f"   {side:8s} {k:9s} " + " ".join(
                    f"{c}={v.get(c)}" for c in qcols + ["n"]))
        for k, v in rec.get("by_day_receipt", {}).items():
            print(f"   receipt {k:8s} rows {v['rows']:5d} obs-base rms "
                  f"{v['obs_minus_base_rms_about_median']:7.0f} iqr {v['obs_minus_base_iqr']:6.0f}")

    (EXP / "analysis/lfpgwhy.json").write_text(json.dumps(out, indent=2, sort_keys=True,
                                                          default=float))
    for ap, rec in out["by_airport"].items():
        print(f"== {ap}")
        for side, d in (("fold", rec["fold"]), ("ranking", rec["ranking"])):
            for k, v in d.items():
                o = v.get("obs_minus_base", {})
                t = v.get("obs_taxi", {})
                if not o:
                    print(f"   {side:8s} {k:9s} n=0")
                    continue
                print(f"   {side:8s} {k:9s} n={t.get('n',0):6d} | obs_taxi med "
                      f"{t.get('median',float('nan')):7.0f} iqr {t.get('iqr',float('nan')):6.0f}"
                      f" | obs-base med {o.get('median',float('nan')):7.0f} "
                      f"iqr {o.get('iqr',float('nan')):6.0f} rms {o.get('rms_about_the_median',float('nan')):7.0f}")


if __name__ == "__main__":
    main()
