"""RESULTS section 47.75 (recorded rule): the redundancy reconciliation -- the contrast, ZERO FIT.

Section 47.74.4 left a candidate: section 47.69's far-in-time depressant (0.724 / 0.788 per
writable row) is REDUNDANCY WITH MEMBER 1, because each far month is calendar-adjacent to a fold
month member 1 already holds.  The four fits are `onemonth.py --matched`: section 47.69's
estimand (the one-month extension over cached member 1, one fold fitted, priced by R.price)
with section 47.74's row-matched draw as the extension.  This reads the four saved vectors and
forms, per edge fold,

    R_f = G_f(F2 | member 1) / G_f(N2 | member 1)

where G_f is R.price's per-fold `mse_gain_s2` -- mean(2 r inc - inc^2) over EVERY row of the
fold, r = y - pbase - d_m1, inc = d - d_m1.  The two arms carry equal rows at equal airports, so
raw and per-row are one ratio.  The interval is a paired day bootstrap inside each fold; the
pooled R is the geometric mean of the two folds, from draws combined draw by draw.

    .venv/bin/python experiments/exp-046-surfacefit/code/redundfit.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402
import onemonth as OM                                                 # noqa: E402
import withinsplit as WS                                              # noqa: E402
import nooverwrite as NO                                              # noqa: E402

SEED = 4775
N_BOOT = 2000
OUT = EXP / "analysis/redundfit.json"
CODE = {"december": "dec", "january": "jan"}
# RESULTS 47.75.2, registered before any fit.  Comparators from the record, never fitted to.
PER_ROW_4769 = {"december": 0.724, "january": 0.788}
RAW_4769 = {"december": 0.477, "january": 0.903}
NO_M1_4774 = {"december": 1.1068, "january": 1.0291}
CUT = float(np.exp(0.5 * (np.log(np.sqrt(0.724 * 0.788)) + np.log(np.sqrt(1.1068 * 1.0291)))))


def tag(fold: str, arm: str) -> str:
    return f"only{WS.DESIGN[fold][arm][0].replace('-', '')}-folds-{CODE[fold]}-matched{arm}"


def band(lo: float, hi: float) -> str:
    return ("RETURNS" if hi < CUT else "STRADDLES" if lo < CUT else
            "PARTIAL" if hi < 1.0 else "ABSENT")


def main() -> None:
    t0 = time.time()
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true",
                    help="47.71.6: allow replacing an existing artifact")
    args = ps.parse_args()
    NO.refuse_existing([OUT], overwrite=args.overwrite)
    print(f"artifact {OUT.name}: no existing artifact; cut C = {CUT:.4f}", flush=True)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    d_m1 = {f: v.astype("float64") for f, v in np.load(OM.CACHED_M1).items()}

    rng = np.random.default_rng(SEED)
    res, logs = {}, {}
    for fold in CODE:
        p = pan[fold]
        r = p.rbase.to_numpy("float64") - d_m1[fold]
        days = p.day.to_numpy()
        codes, uniq = pd.factorize(days)
        k = len(uniq)
        draws = rng.integers(0, k, size=(N_BOOT, k))
        six = json.loads((EXP / f"analysis/withinsplit-{fold}.json").read_text())["plan"]
        sub = (p.usable.to_numpy(bool)
               & p.airport.astype(str).isin(six["airports"]).to_numpy())
        ent, G, Gb, Gs, Gsb = {}, {}, {}, {}, {}
        for arm in ("N2", "F2"):
            t = tag(fold, arm)
            d = np.load(EXP / f"work/d-onemonth-{t}-255-free.npz")
            for g in A.FOLDS:              # the unfitted folds must be carried EXACTLY
                if g != fold and not np.array_equal(d[g].astype("float64"), d_m1[g]):
                    raise SystemExit(f"{t}: fold {g} was not carried from member 1")
            rec = json.loads((EXP / f"analysis/onemonth-{t}.json").read_text())
            mp = rec["matched_draw"]
            if mp is None or mp["arm"] != arm or mp["rows"] != six["fit_rows_per_set"]:
                raise SystemExit(f"{t}: the artifact does not carry 47.74's matched draw")
            inc = d[fold].astype("float64") - d_m1[fold]
            v = 2 * r * inc - inc * inc
            G[arm] = float(v.mean())
            want = rec["prices"]["i_over_cached_member_1"]["by_fold"][fold]["mse_gain_s2"]
            if abs(G[arm] - want) > 1e-9 * max(1.0, abs(want)):
                raise SystemExit(f"{t}: G {G[arm]!r} is not onemonth's {want!r}")
            n = np.bincount(codes, weights=v, minlength=k)
            c = np.bincount(codes, minlength=k).astype(float)
            Gb[arm] = n[draws].sum(1) / c[draws].sum(1)
            cs = codes[sub]
            ns = np.bincount(cs, weights=v[sub], minlength=k)
            nc = np.bincount(cs, minlength=k).astype(float)
            Gs[arm] = float(v[sub].mean())
            Gsb[arm] = ns[draws].sum(1) / nc[draws].sum(1)
            ent[arm] = {"source": WS.DESIGN[fold][arm][0], "artifact": f"analysis/onemonth-{t}.json",
                        "rows_fitted_extra": mp["rows"], "mvt_id_sha16": mp["mvt_id_sha16"],
                        "rows_moved": int((inc != 0).sum()),
                        "G_whole_fold_s2": G[arm],
                        "G_ci95": [float(np.percentile(Gb[arm], 2.5)),
                                   float(np.percentile(Gb[arm], 97.5))],
                        "G_six_airport_class_rows_s2_record_only": Gs[arm]}
        rb = Gb["F2"] / Gb["N2"]
        rs = Gsb["F2"] / Gsb["N2"]
        ok = rb > 0
        logs[fold] = np.where(ok, np.log(np.where(ok, rb, 1.0)), np.nan)
        ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
        res[fold] = {"arms": ent, "R_point": G["F2"] / G["N2"], "R_ci95": ci,
                     "R_band_by_fold": band(*ci),
                     "share_of_draws_with_nonpositive_R": float((~ok).mean()),
                     "increments_clear_zero": all(ent[a]["G_ci95"][0] > 0 for a in ent),
                     "R_six_airport_class_rows_record_only": {
                         "point": Gs["F2"] / Gs["N2"],
                         "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
                         "rows": int(sub.sum())},
                     "comparators": {"per_row_4769": PER_ROW_4769[fold], "raw_4769": RAW_4769[fold],
                                     "no_member1_4774": NO_M1_4774[fold]},
                     "rows_scored": int(len(p)), "days": int(k)}
        print(f"{fold:9} N2 {G['N2']:+9.2f}  F2 {G['F2']:+9.2f}  R {G['F2']/G['N2']:.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]  six-airport R {Gs['F2']/Gs['N2']:.4f}", flush=True)

    kill = [f for f in CODE if not res[f]["increments_clear_zero"]]
    both = logs["december"] + logs["january"]
    okp = np.isfinite(both)
    pooled_draws = np.exp(0.5 * both[okp])
    Rp = float(np.sqrt(res["december"]["R_point"] * res["january"]["R_point"])) \
        if res["december"]["R_point"] > 0 and res["january"]["R_point"] > 0 else None
    pci = [float(np.percentile(pooled_draws, 2.5)), float(np.percentile(pooled_draws, 97.5))]
    split = (res["december"]["R_point"] < CUT) != (res["january"]["R_point"] < CUT)
    verdict = ("NOT POOLED (kill line: " + ", ".join(kill) + ")" if kill else
               "SPLIT" if split else band(*pci))
    out = {"section": "RESULTS 47.75", "seed": SEED, "n_boot": N_BOOT, "cut_C": CUT,
           "statistic": "R_f = G_f(F2|m1)/G_f(N2|m1); G = R.price mse_gain_s2 over the whole fold",
           "by_fold": res, "R_pooled": Rp, "R_pooled_ci95": pci,
           "pooled_draws_usable": int(okp.sum()), "verdict": verdict,
           "comparators_pooled": {"per_row_4769": float(np.sqrt(0.724 * 0.788)),
                                  "raw_4769": float(np.sqrt(0.477 * 0.903)),
                                  "no_member1_4774": float(np.sqrt(1.1068 * 1.0291))},
           "seconds": round(time.time() - t0, 1)}
    NO.refuse_existing([OUT], overwrite=args.overwrite)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"pooled R {Rp:.4f} [{pci[0]:.4f}, {pci[1]:.4f}]  verdict {verdict}")
    print(f"written to {OUT.relative_to(ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
