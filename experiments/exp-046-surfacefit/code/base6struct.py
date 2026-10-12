"""RESULTS section 47.61.4: is the base6 error a STRUCTURE the tree could carry between months?

The zero-fit precursor to 47.61.2.  Section 47.17 computed the RMS of `pbase_6 - pbase_true`
and registered a threshold on it; nobody computed its STRUCTURE.  Two questions, both pure
arithmetic on files already on disk, no fit and no board read:

  (A) is the per-airport base6 error profile STABLE across the three fold months?  An
      unstable profile is month-specific noise the tree cannot carry from an extension month
      to a fold month, and H5's transferable channel is shut at the airport level.

  (B) is that profile ALIGNED with the fold's own residual over the head?  This is the only
      path by which the mismatch could INFLATE a fold read rather than depress it: the
      extension rows carry target `r_true - e`, so they pull the arm DOWN by the per-airport
      mean of `e`.  That helps the fold's scored residual only where `e` runs OPPOSITE to the
      residual the arm is trying to fit.  A positive correlation between the two profiles
      means the pull is the wrong way and the mismatch DEPRESSES the face.

    .venv/bin/python experiments/exp-046-surfacefit/code/base6struct.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BA                                                       # noqa: E402

OOF = ROOT / "experiments/exp-047-queue/oof"
HEAD = EXP / "work/d-bundle-ungated1299-guard995.npz"
PRECOND = EXP / "analysis/precond-base6.json"
OUT = EXP / "analysis/base6struct.json"


def _rank(v: np.ndarray) -> np.ndarray:
    o = np.argsort(np.argsort(v))
    return o.astype("float64")


def _corr(a: np.ndarray, b: np.ndarray) -> dict:
    if len(a) < 3:
        return {"pearson": None, "spearman": None, "n": len(a)}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(np.corrcoef(_rank(a), _rank(b))[0, 1]),
            "n": int(len(a))}


def main() -> None:
    pan = BA.load()
    head = np.load(HEAD)
    pre = json.loads(PRECOND.read_text())

    prof: dict[str, dict] = {}
    per_fold: dict[str, dict] = {}
    for f in BA.FOLDS:
        p = pan[f]
        b6 = pd.read_parquet(OOF / f"base6--{f}.parquet").set_index("MVT_ID_mvt")
        idx = p.MVT_ID_mvt.to_numpy("float64")
        p6 = b6.y_pred.reindex(idx).to_numpy("float64")
        y6 = b6.y_true.reindex(idx).to_numpy("float64")
        y = p.y_true.to_numpy("float64")
        pb = p.pbase.to_numpy("float64")
        if not np.isfinite(p6).all():
            raise SystemExit(f"{f}: {int((~np.isfinite(p6)).sum())} rows did not join base6")
        if int((np.abs(y6 - y) > 1e-6).sum()):
            raise SystemExit(f"{f}: base6's y_true differs from the panel's -- bad join")

        e = p6 - pb                       # the base6 error, s
        d_head = head[f].astype("float64")
        r_head = (y - pb) - d_head        # the residual the arm is actually fitting
        air = p.airport.astype(str).to_numpy()

        rms_all = float(np.sqrt(np.mean(e ** 2)))
        stored = pre["by_fold"][f]["rms_s_all_matched"]
        if abs(rms_all - stored) > 1e-9:
            raise SystemExit(f"{f}: recomputed rms {rms_all!r} != precond-base6.json's "
                             f"{stored!r} -- the join or the base moved")

        by_air = {}
        for a in sorted(set(air)):
            m = air == a
            by_air[a] = {"rows": int(m.sum()),
                         "base6_error_mean_s": float(e[m].mean()),
                         "base6_error_rms_s": float(np.sqrt(np.mean(e[m] ** 2))),
                         "residual_over_head_mean_s": float(r_head[m].mean())}
        prof[f] = by_air

        # How much of the base6 error is the AIRPORT channel at all?  Anything the tree can
        # carry via an airport-level split lives in the between-airport component.
        n = np.array([by_air[a]["rows"] for a in by_air], "float64")
        mu = np.array([by_air[a]["base6_error_mean_s"] for a in by_air], "float64")
        ms = np.array([by_air[a]["base6_error_rms_s"] for a in by_air], "float64")
        grand = float(np.average(mu, weights=n))
        between = float(np.average((mu - grand) ** 2, weights=n))
        total = float(np.average(ms ** 2, weights=n)) - grand ** 2
        per_fold[f] = {
            "rows": int(len(p)),
            "rms_base6_error_s": rms_all,
            "rms_s_usable_the_precondition_headline": pre["by_fold"][f]["rms_s_usable"],
            "grand_mean_s": grand,
            "between_airport_variance_s2": between,
            "total_variance_s2": total,
            "between_airport_share": between / total,
            "corr_error_profile_vs_residual_profile": _corr(mu, np.array(
                [by_air[a]["residual_over_head_mean_s"] for a in by_air], "float64")),
        }

    # (A) Is the airport profile the SAME across fold months?  Computed on all airports and
    # again on the NINE that 47.17's precondition used: the tenth is LTFM, which has 0
    # occupied rows (47.20), so the arm never writes it and it cannot carry this mechanism.
    airports = sorted(set.intersection(*[set(prof[f]) for f in BA.FOLDS]))
    nine = sorted(set(json.loads(PRECOND.read_text())["by_fold"]["december"]["by_airport"]))
    if set(nine) - set(airports):
        raise SystemExit("the precondition names an airport the panel does not carry")
    cross, cross9 = {}, {}
    for f, g in combinations(BA.FOLDS, 2):
        for dst, cols in ((cross, airports), (cross9, nine)):
            a = np.array([prof[f][x]["base6_error_mean_s"] for x in cols], "float64")
            b = np.array([prof[g][x]["base6_error_mean_s"] for x in cols], "float64")
            dst[f"{f}_vs_{g}"] = dict(_corr(a, b),
                                      max_abs_difference_s=float(np.abs(a - b).max()))

    pear = [v["pearson"] for v in cross.values()]
    shares = [per_fold[f]["between_airport_share"] for f in BA.FOLDS]
    align = [per_fold[f]["corr_error_profile_vs_residual_profile"]["pearson"]
             for f in BA.FOLDS]
    out = {
        "what": __doc__.strip().splitlines()[0],
        "airports": airports,
        "per_airport_profile": prof,
        "per_fold": per_fold,
        "A_cross_month_correlation_of_the_error_profile": cross,
        "A_mean_pearson": float(np.mean(pear)),
        "A_cross_month_correlation_nine_airports_no_LTFM": cross9,
        "B_alignment_nine_airports_no_LTFM": {
            f: _corr(np.array([prof[f][x]["base6_error_mean_s"] for x in nine], "float64"),
                     np.array([prof[f][x]["residual_over_head_mean_s"] for x in nine],
                              "float64")) for f in BA.FOLDS},
        "B_alignment_with_the_residual_the_arm_fits": {
            f: per_fold[f]["corr_error_profile_vs_residual_profile"] for f in BA.FOLDS},
        "reading": {
            "channel_is_stable_across_months": bool(min(pear) > 0.5),
            "airport_channel_share_of_the_error": shares,
            "inflation_branch_open": bool(max(a for a in align if a is not None) < 0.0),
            "note": ("the extension rows carry target r_true - e, so they pull the arm DOWN "
                     "by the per-airport mean of e. That can only INFLATE a fold read where "
                     "e runs OPPOSITE to the residual the arm fits, i.e. where B is "
                     "NEGATIVE. B positive means the pull is the wrong way and the mismatch "
                     "DEPRESSES the face"),
            "limit": ("this tests the AIRPORT channel, which is the one 47.17 named. A null "
                      "here bounds the transferable structure at airport level and does not "
                      "eliminate a finer feature-conditional channel"),
        },
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    print(f"written {OUT.relative_to(ROOT)}\n")
    print("(A) cross-month correlation of the per-airport base6 error profile:")
    for k, v in cross.items():
        print(f"    {k:26} pearson {v['pearson']:+.4f}  spearman {v['spearman']:+.4f}"
              f"  max|diff| {v['max_abs_difference_s']:6.2f} s")
    print(f"    mean pearson {out['A_mean_pearson']:+.4f}   stable {out['reading']['channel_is_stable_across_months']}")
    print("\n     the airport channel's share of the base6 error variance:")
    for f in BA.FOLDS:
        print(f"    {f:10} between-airport {per_fold[f]['between_airport_share']*100:5.2f} %"
              f"   of total {per_fold[f]['total_variance_s2']:8.1f} s2"
              f"   (rms {per_fold[f]['rms_base6_error_s']:.2f} s)")
    print("\n(B) alignment of the error profile with the residual the arm fits:")
    for f in BA.FOLDS:
        c = per_fold[f]["corr_error_profile_vs_residual_profile"]
        print(f"    {f:10} pearson {c['pearson']:+.4f}  spearman {c['spearman']:+.4f}  n={c['n']}")
    print("\n    the same two, on the NINE the precondition used (LTFM has 0 occupied rows):")
    for k, v in cross9.items():
        print(f"      A {k:26} pearson {v['pearson']:+.4f}")
    for f in BA.FOLDS:
        v = out["B_alignment_nine_airports_no_LTFM"][f]
        print(f"      B {f:26} pearson {v['pearson']:+.4f}")
    print(f"\n    inflation branch open: {out['reading']['inflation_branch_open']}")


if __name__ == "__main__":
    main()
