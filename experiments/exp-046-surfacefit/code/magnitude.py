"""Order 5: how big is the update WHERE IT LANDS, on 2026 against the folds?

Desk order, 2026-09-09, after order 4 refuted the instrument candidate:

    the update's MAGNITUDE where it lands -- |d| distribution (RMS, quantiles) of v13's
    update, v15's increment and v16's increment on the served 2026 rows against the same
    update's |d| on the fold rows, per airport and month. ... if a well-calibrated
    unselected update moves the 2026 rows more than it moves the fold rows, its gain
    scales with that and the fold price is a lower bound by construction -- that is the
    candidate to test, and it is a property of the update.

**Why `mean(d^2)` is the quantity, not `RMS` alone.** The paired gain is
`2*mean(r*d) - mean(d*d)`.  A well-calibrated update has `d` equal to the part of the
residual it can explain, so `mean(r*d) = mean(d*d)` and the per-row gain is `mean(d*d)`
on EITHER side.  The projection takes the FOLD per-row gain and applies it to the 2026
rows, so under calibration the multiplier it should have used is exactly

    mean(d^2) on the served 2026 rows / mean(d^2) on the served fold rows

and that is the number reported here, per update, beside the board's own readings of
x1.75, x2.01 and x0.36.  RMS and the |d| quantiles are reported with it because a ratio
of means can be carried by a tail, and the desk asked to see the distribution.

**Label-free and no refit.**  The ranking side is arithmetic on files already uploaded --
each pair differenced -- and the fold side is the saved vector of the same arm, re-read.
No label of either year is touched and nothing is fitted.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/magnitude.py
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
ROOT = EXP.parents[1]

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import guard as G                                                     # noqa: E402

UP = ROOT / "submissions/uploaded"
HOLD = ("LFPG",)
OUT = EXP / "analysis/magnitude.json"

#: what the board read for each of these, from RESULTS 8.4
BOARD = {"v13": {"fold_gain_s": 8.128, "board_delta_s": -14.26, "transfer": 1.75},
         "v15_increment": {"fold_gain_s": 0.7296, "board_delta_s": -1.4641,
                           "transfer": 2.01},
         "v16_increment": {"fold_gain_s": 0.3350, "board_delta_s": -0.1210,
                           "transfer": 0.36}}


def upload(v: str) -> pd.DataFrame:
    return pd.read_parquet(UP / f"youthful-giraffe_v{v}.parquet")


def dist(x: np.ndarray, tag: str) -> dict:
    """`mean(d^2)`, RMS and the |d| quantiles of one side of one update."""
    a = np.abs(x)
    return {"rows": int(len(x)),
            "mean_d2": float(np.mean(x * x)) if len(x) else None,
            "rms": float(np.sqrt(np.mean(x * x))) if len(x) else None,
            "median_abs_d": float(np.median(a)) if len(x) else None,
            "p90_abs_d": float(np.percentile(a, 90)) if len(x) else None,
            "p99_abs_d": float(np.percentile(a, 99)) if len(x) else None,
            "max_abs_d": float(a.max()) if len(x) else None,
            "side": tag}


def main() -> None:
    dep, _ = S.ranking_block()
    dep["usable"] = A.served_class(dep)
    rk_served = (dep.usable.to_numpy(bool)
                 & ~np.isin(dep.airport.to_numpy().astype(str), list(HOLD)))
    rk_apt = dep.airport.to_numpy().astype(str)
    rk_month = dep.rank_month.to_numpy().astype(str)
    key = dep.MVT_ID_mvt.to_numpy("float64")

    files = {v: upload(v) for v in ("12", "13", "14", "15", "16")}
    for v, f in files.items():
        if not np.array_equal(f[S.D.ID].to_numpy("float64"), key):
            raise SystemExit(f"v{v} is not in the ranking template's row order")

    pan = A.panel_with_block(BA.load())
    npz = {"arm": dict(np.load(S.npz_path("c_lgb_shipped")))}
    for c in ("predeclared_31x300_lr050", "leaves_127x300_lr050", "leaves_255x300_lr050"):
        npz[c] = dict(np.load(EXP / f"work/d-capacity-{c}.npz"))
    gj = json.loads((EXP / "analysis/guard-arms.json").read_text())
    cut = float(gj["guard"]["cut_s_for_the_ranking_serve"])

    def fold_vec(kind: str, f: str) -> np.ndarray:
        p = pan[f]
        if kind == "v13":
            d = npz["arm"][f].astype("float64")
        elif kind == "v15_increment":
            d = (npz["leaves_127x300_lr050"][f].astype("float64")
                 - npz["predeclared_31x300_lr050"][f].astype("float64"))
        else:
            g = npz["leaves_255x300_lr050"][f].astype("float64").copy()
            g[G.mask(p.pbase.to_numpy("float64"), p.obs_taxi, cut)] = 0.0
            d = g - npz["leaves_127x300_lr050"][f].astype("float64")
        d = d.copy()
        d[np.isin(p.airport.to_numpy().astype(str), list(HOLD))] = 0.0
        return d

    PAIRS = {"v13": ("12", "13"), "v15_increment": ("14", "15"),
             "v16_increment": ("15", "16")}
    out = {"what": ("the magnitude of each served update where it lands, 2026 against the "
                    "folds, as the test of whether the fold price is a lower bound by "
                    "construction"),
           "ordered_by": "the desk, 2026-09-09, order 5",
           "label_free": "arithmetic on uploaded files and saved vectors; nothing fitted",
           "held": list(HOLD), "guard_cut_s": cut,
           "matcher_sha256": A.MATCHER_SHA,
           "note_on_bases": ("the ranking side is the bytes actually uploaded, built by "
                             "the PRE-rebuild matcher; the fold side is the re-priced "
                             "vector of the same arm. The two differ on 602 rows of "
                             "854,937 (RESULTS 14.1), which cannot move a ratio of means"),
           "by_update": {}}

    for kind, (a, b) in PAIRS.items():
        drk = (files[b][S.D.TARGET].to_numpy("float64")
               - files[a][S.D.TARGET].to_numpy("float64"))
        moved_rk = drk != 0
        dfold = {f: fold_vec(kind, f) for f in A.FOLDS}
        fold_all = np.concatenate([dfold[f] for f in A.FOLDS])
        fold_served = np.concatenate([
            dfold[f][(pan[f].usable.to_numpy(bool)
                      & ~np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD)))]
            for f in A.FOLDS])
        fold_moved = fold_all[fold_all != 0]

        rec = {
            "ranking_moved": dist(drk[moved_rk], "2026 rows this update moved"),
            "ranking_served_class": dist(drk[rk_served], "2026 served rows, zeros in"),
            "fold_moved": dist(fold_moved, "fold rows this update moved"),
            "fold_served_class": dist(fold_served, "fold served rows, zeros in"),
            "board": BOARD[kind],
        }
        for tag in ("moved", "served_class"):
            r, f = rec[f"ranking_{tag}"]["mean_d2"], rec[f"fold_{tag}"]["mean_d2"]
            rec[f"mean_d2_ratio_on_{tag}"] = round(r / f, 4) if f else None
        rec["by_airport"] = {}
        for apt in sorted(set(rk_apt)):
            sel = moved_rk & (rk_apt == apt)
            fsel = np.concatenate([
                dfold[f][(pan[f].airport.to_numpy().astype(str) == apt)] for f in A.FOLDS])
            fsel = fsel[fsel != 0]
            if sel.sum() < 50 or len(fsel) < 50:
                continue
            r2 = float(np.mean(drk[sel] ** 2))
            f2 = float(np.mean(fsel ** 2))
            rec["by_airport"][apt] = {"ranking_rows": int(sel.sum()),
                                      "fold_rows": int(len(fsel)),
                                      "ranking_mean_d2": r2, "fold_mean_d2": f2,
                                      "ratio": round(r2 / f2, 4) if f2 else None}
        rec["by_month"] = {}
        for m in sorted(set(rk_month)):
            sel = moved_rk & (rk_month == m)
            if sel.sum() < 50:
                continue
            r2 = float(np.mean(drk[sel] ** 2))
            rec["by_month"][m] = {"rows": int(sel.sum()), "ranking_mean_d2": r2,
                                  "ratio": round(r2 / rec["fold_moved"]["mean_d2"], 4)
                                  if rec["fold_moved"]["mean_d2"] else None}
        out["by_update"][kind] = rec

        print(f"{kind:16s} moved {rec['ranking_moved']['rows']:6d} rk / "
              f"{rec['fold_moved']['rows']:6d} fold   "
              f"rms {rec['ranking_moved']['rms']:7.2f} / {rec['fold_moved']['rms']:7.2f}   "
              f"mean_d2 ratio {rec['mean_d2_ratio_on_moved']:6.3f} (moved) "
              f"{rec['mean_d2_ratio_on_served_class']:6.3f} (class)   "
              f"board x{BOARD[kind]['transfer']}", flush=True)

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written", OUT)


if __name__ == "__main__":
    main()
