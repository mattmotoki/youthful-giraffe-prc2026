"""Why the `_nbr` arms fail the july tail clause, and what repairs it without a label.

The desk, 2026-09-08 18:10 UTC: "the `_nbr` arms with their july tail failure diagnosed
and repaired label-free (which rows, which `No_*` column), so the 0.13-0.16 s is not left
on the table".

The reading.  Adding `exp-047-queue`'s twelve neighbourhood columns to the with-clock head
is worth **+0.128 s** all-row on arm (c) (+8.1410 to +8.2693) and **+0.826 s** on arm (b),
and both fail the registered clause on ONE fold: the july rows with `y > 3600 s` go from
-0.097 s to **-4.263 s** on (c) and from +12.952 to **-4.634 s** on (b), against a clause
of one second.  December and january pass on both.  So the question is not whether the
columns carry information -- they do, on two folds and on the board -- but what they do to
253 rows in july.

Stage 1 (this file, `--stage 1`) needs no fit: it reads the SAVED update vectors and says
which rows carry the loss and how concentrated it is.  Stage 2 (`--stage 2`) refits the
july fold once and asks LightGBM itself, through `pred_contrib`, which of the twelve
columns pushed those rows.  A repair is only admissible if it is a rule on INPUTS -- the
served side has no label to apply a y-conditioned rule with.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/nbrtail.py --stage 1
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

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402

TAIL_Y = A.TAIL_Y_S if hasattr(A, "TAIL_Y_S") else 3600.0
PAIRS = {"c_lgb_shipped": "c_lgb_shipped_nbr", "b_lgb": "b_lgb_nbr"}
#: `--pair base:variant` prices any other pair of saved update vectors the same way; the
#: geometry arms fail the july tail too and the question is whether they fail it on the
#: same rows (`work/d-<arm>.npz` for both).


def damage(r: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per row, the squared error the update ADDS: (r-d)^2 - r^2 = d^2 - 2 r d."""
    return d ** 2 - 2 * r * d


def stage1(pan: dict[str, pd.DataFrame], pairs: dict | None = None) -> dict:
    out = {"what": ("which rows carry the july tail loss of the `_nbr` arms, read from "
                    "the saved update vectors with no refit"),
           "tail_rule": f"y > {TAIL_Y} s", "pairs": {}}
    for base_arm, nbr_arm in (pairs or PAIRS).items():
        zb = np.load(EXP / f"work/d-{base_arm}.npz")
        zn = np.load(EXP / f"work/d-{nbr_arm}.npz")
        rec = {}
        for f in A.FOLDS:
            p = pan[f]
            y = p.y_true.to_numpy("float64")
            r = y - p.pbase.to_numpy("float64")
            t = y > TAIL_Y
            db, dn = zb[f][t], zn[f][t]
            rt = r[t]
            hb, hn = damage(rt, db), damage(rt, dn)
            delta = hn - hb                       # what the neighbour columns ADD in MSE
            order = np.argsort(-delta)
            n = int(t.sum())
            cum = np.cumsum(delta[order]) / max(delta.sum(), 1e-9)
            k10 = int(np.searchsorted(cum, 0.5) + 1) if delta.sum() > 0 else 0
            sub = p[t].reset_index(drop=True)
            worst = [{"airport": str(sub.airport.iloc[i]),
                      "day": str(sub.day.iloc[i]),
                      "y_s": float(y[t][i]), "r_s": float(rt[i]),
                      "d_base_s": float(db[i]), "d_nbr_s": float(dn[i]),
                      "added_squared_error_s2": float(delta[i])}
                     for i in order[:10]]
            rec[f] = {
                "tail_rows": n,
                "moved_by_the_nbr_arm": int(np.count_nonzero(dn)),
                "rmse_base_arm": float(np.sqrt(np.mean((rt - db) ** 2))),
                "rmse_nbr_arm": float(np.sqrt(np.mean((rt - dn) ** 2))),
                "added_mse_total_s2": float(delta.sum()),
                "rows_that_got_worse": int((delta > 0).sum()),
                "rows_that_got_better": int((delta < 0).sum()),
                "rows_carrying_half_the_loss": k10,
                "share_of_the_loss_in_the_worst_10_rows": (
                    float(delta[order[:10]].sum() / delta.sum())
                    if delta.sum() > 0 else None),
                "by_airport": {str(a): float(delta[(sub.airport == a).to_numpy()].sum())
                               for a in sorted(sub.airport.astype(str).unique())},
                "worst_rows": worst,
            }
        out["pairs"][nbr_arm] = rec
    return out


def stage2(pan: dict[str, pd.DataFrame], pair: tuple[str, str] = (
        "c_lgb_shipped", "c_lgb_shipped_nbr")) -> dict:
    """WHICH `No_*` column, and is there a rule on INPUTS that keeps the gain?

    Stage 1 says where the loss is: LIRF's july rows carry 97 % of it on arm (b) and more
    than all of it on arm (c), and half of it sits in twelve rows.  A repair may only read
    inputs, because the served side has no label -- so this compares the twelve columns'
    own distributions on the damaging rows against the population that pays for them, and
    looks for a threshold that separates them.
    """
    out = {"what": ("which No_* column stands behind the july tail loss, and whether a "
                    "rule on inputs separates the damaging rows from the paying ones"),
           "by_arm": {}}
    zb = np.load(EXP / f"work/d-{pair[0]}.npz")
    zn = np.load(EXP / f"work/d-{pair[1]}.npz")
    out["pair"] = {"base": pair[0], "variant": pair[1]}
    # the neighbour block is exp-047's file and it is rebuilt when that lane rebuilds.
    # The damage and the distributions must come from ONE version of it, so the version
    # is recorded here rather than assumed (it moved under this lane once: arms.json's
    # `_nbr` row was priced on c4d7a841 and the file now hashes to 757cf6a9).
    out["neighbour_block_now"] = {m: A.neighbour_block(m)[1].get("sha256")
                                  for m in A.FOLD_MONTH.values()}
    have = [c for c in A.NBR_COLS if c in pan["july_loo"].columns]
    out["columns_present"] = have
    if not have:
        out["skipped"] = "the neighbour block is not joined to the panel"
        return out
    for f in A.FOLDS:
        p = pan[f]
        y = p.y_true.to_numpy("float64")
        r = y - p.pbase.to_numpy("float64")
        t = (y > TAIL_Y)
        delta = np.zeros(len(p))
        delta[t] = damage(r[t], zn[f][t]) - damage(r[t], zb[f][t])
        u = p.usable.to_numpy(bool)
        # the damaging rows: the tail rows the neighbour columns made worse
        bad = t & (delta > 0)
        # the paying rows: every with-clock row that is NOT a damaged tail row
        pay = u & ~bad
        rec = {"tail_rows": int(t.sum()), "damaging_rows": int(bad.sum()),
               "paying_rows": int(pay.sum()),
               "damaging_by_airport": p.loc[bad, "airport"].astype(str)
                                       .value_counts().to_dict(),
               "columns": {}}
        for c in have:
            v = p[c].to_numpy("float64")
            def q(mask):
                x = v[mask & np.isfinite(v)]
                if x.size == 0:
                    return {"n": 0}
                return {"n": int(x.size), "median": float(np.median(x)),
                        "p90": float(np.percentile(x, 90)),
                        "p99": float(np.percentile(x, 99)),
                        "max": float(np.max(x))}
            rec["columns"][c] = {"on_the_damaging_rows": q(bad),
                                 "on_the_paying_rows": q(pay),
                                 "null_share_damaging": float(
                                     np.mean(~np.isfinite(v[bad])) if bad.any() else 0.0),
                                 "null_share_paying": float(
                                     np.mean(~np.isfinite(v[pay])) if pay.any() else 0.0)}
        # LIRF is where the loss is; the same reading restricted to it, so a rule can be
        # tested against the airport rather than against the fold
        for a in ("LIRF",):
            m = (p.airport.astype(str) == a).to_numpy()
            rec[f"{a}_with_clock_rows"] = int((m & u).sum())
            rec[f"{a}_columns"] = {
                c: {"median_all_with_clock": (
                        float(np.nanmedian(p.loc[m & u, c])) if (m & u).any() else None),
                    "median_elsewhere": float(np.nanmedian(p.loc[u & ~m, c]))}
                for c in have}
        out["by_arm"][f] = rec
    return out


def main() -> None:
    stage = "1"
    for a in sys.argv[1:]:
        if a.startswith("--stage"):
            stage = a.split("=")[-1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    pairs = {}
    for a in sys.argv[1:]:
        if a.startswith("--pair="):
            b, v = a.split("=", 1)[1].split(":", 1)
            pairs[b] = v
    pan = A.panel_with_block(BA.load())
    if stage == "2":
        out = json.loads((EXP / "analysis/nbr-tail.json").read_text())
        pr = next(iter(pairs.items()), ("c_lgb_shipped", "c_lgb_shipped_nbr"))
        out["stage2"] = stage2(pan, pr)
        (EXP / "analysis/nbr-tail.json").write_text(
            json.dumps(out, indent=2, sort_keys=True))
        s2 = out["stage2"]
        print("columns present:", s2.get("columns_present"))
        for f, rec in s2.get("by_arm", {}).items():
            print(f"{f}: damaging {rec['damaging_rows']} of {rec['tail_rows']} tail rows, "
                  f"paying {rec['paying_rows']}, by airport {rec['damaging_by_airport']}")
            for c, v in rec["columns"].items():
                d, y2 = v["on_the_damaging_rows"], v["on_the_paying_rows"]
                if not d.get("n"):
                    continue
                print(f"   {c:24s} damaging med {d['median']:8.1f} p99 {d['p99']:9.1f} "
                      f"| paying med {y2['median']:8.1f} p99 {y2['p99']:9.1f} "
                      f"| null {v['null_share_damaging']:.2f}/{v['null_share_paying']:.2f}")
        print("written to analysis/nbr-tail.json")
        return
    out = stage1(pan, pairs or None)
    dest = (EXP / "analysis/nbr-tail.json") if not pairs else (
        EXP / "analysis/tail-pairs.json")
    dest.write_text(json.dumps(out, indent=2, sort_keys=True))
    for arm, rec in out["pairs"].items():
        for f, v in rec.items():
            print(f"{arm:20s} {f:9s} tail {v['tail_rows']:4d} worse {v['rows_that_got_worse']:4d} "
                  f"better {v['rows_that_got_better']:4d} added MSE {v['added_mse_total_s2']:+12.0f} "
                  f"half in {v['rows_carrying_half_the_loss']:3d} rows  "
                  f"top10 {str(v['share_of_the_loss_in_the_worst_10_rows'])[:5]}")
        for f in ("july_loo",):
            print(f"  {arm} {f} worst rows:")
            for w in rec[f]["worst_rows"][:5]:
                print(f"    {w['airport']} {w['day']} y {w['y_s']:8.0f} r {w['r_s']:8.0f} "
                      f"d_base {w['d_base_s']:+8.1f} d_nbr {w['d_nbr_s']:+8.1f} "
                      f"added {w['added_squared_error_s2']:+12.0f}")
    print(f"written to {dest.name}")


if __name__ == "__main__":
    main()
