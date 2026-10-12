"""Why the three-seed member loses on the tail, and whether the loss is resolved.

The gate's tail clause is "`y > 3600` not worse by more than 1 s". Arm 1 reads
-5.60 s there on july_loo while gaining +0.179 s overall, so this file asks four
questions before anything is called a FAIL:

  1. is -5.60 s resolved, or is it 656 rows of noise? (paired bootstrap,
     calendar-day x airport clusters, the repo's own convention)
  2. is it a property of averaging, or of the individual seeds? (each seed's own
     tail, and every 1- and 2-seed subset)
  3. how many rows carry it?
  4. does a 2-seed average keep the bulk gain with less tail damage?

    python tail_diag.py --fold july_loo
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
import prc.data as D  # noqa: E402
import score as Sc  # noqa: E402

SEEDS = {"s0 (None)": "experiments/exp-027-step/oof/k2-window-{fold}.parquet",
         "s1 (20)": "experiments/exp-035-k2climb/oof/k2s1-{fold}.parquet",
         "s2 (33)": "experiments/exp-035-k2climb/oof/k2s2-{fold}.parquet"}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="july_loo")
    a = ap.parse_args()
    f = a.fold

    cols, y, ids = {}, None, None
    for k, tpl in SEEDS.items():
        d = pd.read_parquet(ROOT / tpl.format(fold=f)).sort_values("MVT_ID_mvt")
        if ids is None:
            ids = d["MVT_ID_mvt"].to_numpy("float64")
            y = d["y_true"].to_numpy("float64")
        elif not np.array_equal(ids, d["MVT_ID_mvt"].to_numpy("float64")):
            raise SystemExit(f"{k}: different ids")
        cols[k] = d["y_pred"].to_numpy("float64")

    hi = y > 3600
    print(f"{f}: {len(y)} rows, {int(hi.sum())} above 3600 s\n")
    print(f"{'subset':22s} {'all':>10s} {'y<=3600':>10s} {'y>3600':>10s}   "
          f"{'vs s0 all':>10s} {'vs s0 tail':>11s}")
    base = cols["s0 (None)"]
    r0, r0l, r0h = rmse(y, base), rmse(y[~hi], base[~hi]), rmse(y[hi], base[hi])
    out = {}
    names = list(SEEDS)
    subsets = ([(n,) for n in names]
               + [t for t in itertools.combinations(names, 2)]
               + [tuple(names)])
    for sub in subsets:
        p = np.mean([cols[n] for n in sub], axis=0)
        r, rl, rh = rmse(y, p), rmse(y[~hi], p[~hi]), rmse(y[hi], p[hi])
        lab = "+".join(s.split()[0] for s in sub)
        out[lab] = dict(all=r, lo=rl, hi=rh, gain_all=r0 - r, gain_lo=r0l - rl, gain_hi=r0h - rh)
        print(f"{lab:22s} {r:10.4f} {rl:10.4f} {rh:10.4f}   {r0 - r:+10.4f} {r0h - rh:+11.2f}")

    # ---- is the tail move resolved? paired day-clustered bootstrap ----------
    ref = Sc.reference()[f].set_index("MVT_ID_mvt")
    cl = Sc.P.clusters(ref.loc[ids].reset_index())
    p3 = np.mean([cols[n] for n in names], axis=0)
    print("\npaired calendar-day x airport bootstrap, three seeds against K2 "
          "(negative = the arm is better):")
    boot = {}
    for lab, m in (("all rows", np.ones(len(y), bool)), ("y<=3600", ~hi), ("y>3600", hi)):
        d, lo, hiq = Sc.P.boot_delta(y[m], p3[m], base[m], pd.factorize(cl[m])[0])
        boot[lab] = [round(x, 4) for x in (d, lo, hiq)]
        z = "contains zero" if lo <= 0 <= hiq else "clear of zero"
        print(f"  {lab:9s} delta {d:+9.4f}  [{lo:+9.4f}, {hiq:+9.4f}]  {z}")

    # ---- how many rows carry the tail move? --------------------------------
    e0 = (y[hi] - base[hi]) ** 2
    e3 = (y[hi] - p3[hi]) ** 2
    dd = e3 - e0
    order = np.argsort(-np.abs(dd))
    tot = dd.sum()
    print(f"\nthe tail move, by row (sum of squared-error change = {tot:+.4g}):")
    for k in (1, 3, 5, 10, 25):
        print(f"  the {k:3d} largest |change| carry {100 * dd[order[:k]].sum() / tot:6.1f} % of it")
    tid = ids[hi][order[:5]]
    print("\n  the five rows that move most:")
    print(f"  {'MVT_ID':>12s} {'y':>9s} {'K2':>9s} {'3-seed':>9s} {'s0':>9s} {'s1':>9s} {'s2':>9s}")
    for t in tid:
        i = int(np.where(ids == t)[0][0])
        print(f"  {t:12.0f} {y[i]:9.0f} {base[i]:9.0f} {p3[i]:9.0f} "
              + " ".join(f"{cols[n][i]:9.0f}" for n in names))

    rec = {"fold": f, "n": int(len(y)), "n_tail": int(hi.sum()), "subsets": out,
           "bootstrap_3seed_vs_k2": boot,
           "tail_concentration": {str(k): float(dd[order[:k]].sum() / tot)
                                  for k in (1, 3, 5, 10, 25)}}
    (EXP / "analysis" / f"tail-diag-{f}.json").write_text(json.dumps(rec, indent=1))
    print(f"\nwrote analysis/tail-diag-{f}.json")


if __name__ == "__main__":
    main()
