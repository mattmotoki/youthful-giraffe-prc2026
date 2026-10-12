"""The §47.6 hypothesis, priced: does composition explain the fold-to-board transfer?

Section 47.6 measured that the population this lane WRITES is 25.06 % total-variation
distant, by airport, between the fold months and the ranking months, and offered a
hypothesis: the fold-to-board multiplier (1.68x, 1.41x, 1.29x, 0.26x, 0.16x, 1.75x, 1.03x,
2.01x, 0.36x ...) is not a constant of the lane but a property of each update's COMPOSITION,
because the folds and the board weight the airports differently.

This file prices that.  For one update it computes

  * the per-airport per-moved-row fold gain of the EXACT served increment, from the two fold
    vectors the base set pins by sha256;
  * the RAW board prediction -- the same gains at the FOLD's airport composition, which is
    what the record's at-face number is;
  * the REWEIGHTED board prediction -- the same gains at the RANKING file's composition, the
    airports the update actually moves on the board, counted from the byte diff.

Both are converted to board seconds at the previous upload's level, so they are comparable to
the observed board move.  **score-feedback rule: board readings are used here to TEST a pricing hypothesis and
nothing sets an amplitude or a parameter from them.**  Nothing here is a gate.

    .venv/bin/python experiments/exp-046-surfacefit/code/reweight.py            # every update
    .venv/bin/python experiments/exp-046-surfacefit/code/reweight.py --only=v8  # one
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
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402
import bagarm as BAG                                                  # noqa: E402
import serve as S                                                     # noqa: E402

OUT = EXP / "analysis/reweight.json"
RANK_SHARE = {"2026-07": 0.5571321275602379, "2026-01": 0.4428678724397621}

#: One row per update.  `arm`/`head` are the FOLD vectors; where the base set pins them by
#: sha256 the pin is checked, and where it does not the pair must REPRODUCE the fold gain the
#: record published for that upload -- section 25.2 is why a name is not accepted as
#: provenance here.  `bytes`/`prev` are the ranking-side byte pair the moved rows come from.
UPDATES = {
    "v13": {"arm": "work/d-c_lgb_shipped.npz", "head": None,
            "bytes": "submissions/uploaded/youthful-giraffe_v13.parquet",
            "prev": "submissions/uploaded/youthful-giraffe_v12.parquet",
            "level": 262.7418, "observed_move_s": -14.2600,
            "published_fold_gain_s": 8.1280, "arm_sha": None, "head_sha": None},
    "v15": {"arm": "work/d-blockmove-leaves_127x300_lr050.npz",
            "head": "work/d-c_lgb_shipped.npz",
            "bytes": "submissions/uploaded/youthful-giraffe_v15.parquet",
            "prev": "submissions/uploaded/youthful-giraffe_v14.parquet",
            "level": 248.2904, "observed_move_s": -1.4641,
            "published_fold_gain_s": 0.7296, "arm_sha": None, "head_sha": None},
    "v16": {"arm": "work/d-blockmove-guard995.npz",
            "head": "work/d-blockmove-leaves_127x300_lr050.npz",
            "bytes": "submissions/uploaded/youthful-giraffe_v16.parquet",
            "prev": "submissions/uploaded/youthful-giraffe_v15.parquet",
            "level": 246.8263, "observed_move_s": -0.1210,
            "published_fold_gain_s": 0.3350, "arm_sha": None, "head_sha": None},
    "v18": {"arm": "work/d-bundle-ungated1299-guard995.npz",
            "head": "work/d-passonly-6fdf3951-guard995.npz",
            "bytes": "submissions/uploaded/youthful-giraffe_v18.parquet",
            "prev": "submissions/uploaded/youthful-giraffe_v17.parquet",
            "level": 246.6571, "observed_move_s": -0.2966,
            "published_fold_gain_s": None,
            "arm_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                        "2a2eff9b7ff9b7469b6625ce00238684"),
            "head_sha": ("e137939b74070180192544134b28a2ed"
                         "49011071aa29449598fb9de87953029a")},
    # PROSPECTIVE -- section 47.10, both numbers registered before the upload
    "v8": {"arm": "work/d-bag10-255-free.npz",
           "head": "work/d-bundle-ungated1299-guard995.npz",
           "bytes": "submissions/candidates/exp-046-surface-v8.parquet",
           "prev": "submissions/uploaded/youthful-giraffe_v18.parquet",
           "level": 246.3605, "observed_move_s": None,
           "published_fold_gain_s": 0.24275,
           "arm_sha": ("d32a993b48d18fa428cb6c0127de34a5"
                       "ccb0ce5ad09d2dd0a3e89db7a35cd26e"),
           "head_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                        "2a2eff9b7ff9b7469b6625ce00238684")},
}


def sha(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def board_seconds(dmse: float, level: float) -> float:
    """A board-MSE reduction per ranking row, in board seconds at that upload's level."""
    inner = level ** 2 - dmse
    return float(level - np.sqrt(inner)) if inner > 0 else float("nan")


def ranking_moved(cand: Path, prev: Path, ap, rm) -> dict:
    a = pd.read_parquet(cand)
    b = pd.read_parquet(prev)
    av = pd.Series(a.TAXITIME_SEC_mvt.to_numpy("float64"),
                   index=a.MVT_ID_mvt.to_numpy("float64"))
    bv = pd.Series(b.TAXITIME_SEC_mvt.to_numpy("float64"),
                   index=b.MVT_ID_mvt.to_numpy("float64")).reindex(av.index)
    moved = av.to_numpy() != bv.to_numpy()
    air = ap.reindex(av.index).to_numpy()
    mon = rm.reindex(av.index).to_numpy()
    out = {}
    for m in RANK_SHARE:
        sel = mon == m
        c = pd.Series(air[sel & moved]).value_counts().to_dict()
        out[m] = {"rows": int(sel.sum()), "moved": int((sel & moved).sum()),
                  "by_airport": {k: int(v) for k, v in c.items()}}
    return out


def main() -> None:
    t0 = time.time()
    ps = argparse.ArgumentParser()
    ps.add_argument("--only", default=None)
    args = ps.parse_args()

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    S._UNGATED = True
    dep, _ = S.ranking_block()
    ap = pd.Series(dep.airport.astype(str).to_numpy(),
                   index=dep.MVT_ID_mvt.to_numpy("float64"))
    rm = pd.Series(dep.rank_month.astype(str).to_numpy(),
                   index=dep.MVT_ID_mvt.to_numpy("float64"))

    rec = {"what": __doc__.strip().splitlines()[0],
           "d11": ("board readings TEST the hypothesis here; no amplitude or parameter of "
                   "any candidate is set from them, and none of this is a gate"),
           "updates": {}}

    for name, u in UPDATES.items():
        if args.only and name != args.only:
            continue
        arm_p = EXP / u["arm"]
        head_p = (EXP / u["head"]) if u["head"] else None
        row = {"arm": u["arm"], "head": u["head"], "level_s": u["level"],
               "observed_board_move_s": u["observed_move_s"]}
        if not arm_p.exists() or (head_p is not None and not head_p.exists()):
            row["skipped"] = "a fold vector is not on disk"
            rec["updates"][name] = row
            print(f"{name}: SKIPPED -- fold vector missing")
            continue
        # provenance: a sha pin where the base set carries one, otherwise the published
        # fold gain must be reproduced -- a NAME is never accepted (section 25.2)
        prov = {}
        if u["arm_sha"]:
            prov["arm_sha_matches"] = bool(sha(arm_p) == u["arm_sha"])
        if u["head_sha"] and head_p is not None:
            prov["head_sha_matches"] = bool(sha(head_p) == u["head_sha"])
        if any(v is False for v in prov.values()):
            row["skipped"] = f"pinned sha does not match: {prov}"
            rec["updates"][name] = row
            print(f"{name}: SKIPPED -- {row['skipped']}")
            continue

        d_arm = {f: v.astype("float64") for f, v in np.load(arm_p).items()}
        d_head = ({f: v.astype("float64") for f, v in np.load(head_p).items()}
                  if head_p is not None else {f: np.zeros(len(pan[f])) for f in A.FOLDS})

        # the fold gain of the exact served increment, and its per-airport split
        per_air, dmse_raw, fold_rmse = {}, 0.0, {}
        num0 = num1 = 0.0
        for f in A.FOLDS:
            p = pan[f]
            y = p.y_true.to_numpy("float64")
            pb = p.pbase.to_numpy("float64") + d_head[f]
            inc = d_arm[f] - d_head[f]
            e0, e1 = y - pb, y - pb - inc
            se_gain = e0 ** 2 - e1 ** 2
            air = p.airport.astype(str).to_numpy()
            mv = inc != 0
            n = len(p)
            fold_rmse[f] = {"rmse_base_s": float(np.sqrt((e0 ** 2).mean())),
                            "rmse_arm_s": float(np.sqrt((e1 ** 2).mean())),
                            "rows_moved": int(mv.sum())}
            if f in BA.SHARES:
                num0 += BA.SHARES[f] * float((e0 ** 2).mean())
                num1 += BA.SHARES[f] * float((e1 ** 2).mean())
            day = p.day.to_numpy()
            for a in np.unique(air[mv]):
                s = mv & (air == a)
                d = per_air.setdefault(a, {"num": 0.0, "den": 0.0, "by_fold": {}})
                # a DAY-CLUSTERED interval, the convention this lane gates on: rows at one
                # airport on one day share a weather and a runway state, so an interval that
                # treats them as independent is too tight by a lot.  It is here because the
                # composition correction is driven by the airports with the FEWEST fold rows,
                # and a per-airport gain without an interval invites reading noise as signal.
                codes, uniq = pd.factorize(day[s])
                k = len(uniq)
                sums = np.bincount(codes, weights=se_gain[s], minlength=k)
                cnts = np.bincount(codes, minlength=k).astype("float64")
                rng2 = np.random.default_rng(BA.SEED)
                dr = rng2.integers(0, k, size=(2000, k)) if k > 1 else None
                ci = ([float(x) for x in np.percentile(
                          sums[dr].sum(1) / cnts[dr].sum(1), [2.5, 97.5])]
                      if dr is not None else None)
                d["by_fold"][f] = {"rows_moved": int(s.sum()), "days": k,
                                   "gain_s2_per_moved_row": float(se_gain[s].mean()),
                                   "ci95_day_clustered": ci}
                if f in BA.SHARES:
                    d["num"] += BA.SHARES[f] * float(se_gain[s].sum()) / n
                    d["den"] += BA.SHARES[f] * float(s.sum()) / n
        for a, d in per_air.items():
            d["pooled_gain_s2_per_moved_row"] = (d["num"] / d["den"]) if d["den"] else 0.0
        dmse_raw = sum(d["num"] for d in per_air.values())
        fold_gain_s = float(np.sqrt(num0) - np.sqrt(num1))
        row["fold_gain_s_recomputed"] = fold_gain_s
        row["published_fold_gain_s"] = u["published_fold_gain_s"]
        if u["published_fold_gain_s"] is not None:
            row["reproduces_the_published_fold_gain"] = bool(
                abs(fold_gain_s - u["published_fold_gain_s"]) < 0.02)
        row["provenance"] = prov or {"by": "reproduction of the published fold gain"}
        row["fold_side"] = fold_rmse
        row["per_airport"] = {a: {"pooled_gain_s2_per_moved_row":
                                  d["pooled_gain_s2_per_moved_row"],
                                  "by_fold": d["by_fold"]} for a, d in per_air.items()}

        mv_rank = ranking_moved(ROOT / u["bytes"], ROOT / u["prev"], ap, rm)
        row["ranking_moved"] = mv_rank
        dmse_rw = 0.0
        for m, sh in RANK_SHARE.items():
            blkm = mv_rank[m]
            tot = sum(per_air.get(a, {}).get("pooled_gain_s2_per_moved_row", 0.0) * c
                      for a, c in blkm["by_airport"].items())
            dmse_rw += sh * tot / blkm["rows"]
        row["board_mse_reduction_raw_s2"] = dmse_raw
        row["board_mse_reduction_reweighted_s2"] = dmse_rw
        row["prediction_raw_s"] = board_seconds(dmse_raw, u["level"])
        row["prediction_reweighted_s"] = board_seconds(dmse_rw, u["level"])
        row["reweight_factor"] = (dmse_rw / dmse_raw) if dmse_raw else None
        if u["observed_move_s"] is not None:
            obs = -u["observed_move_s"]
            row["observed_gain_s"] = obs
            row["transfer_raw"] = obs / row["prediction_raw_s"] if row["prediction_raw_s"] else None
            row["transfer_reweighted"] = (obs / row["prediction_reweighted_s"]
                                          if row["prediction_reweighted_s"] else None)
        rec["updates"][name] = row
        print(f"{name}: fold {fold_gain_s:+.5f} s (published "
              f"{u['published_fold_gain_s']}) | raw {row['prediction_raw_s']:+.4f} s | "
              f"reweighted {row['prediction_reweighted_s']:+.4f} s | factor "
              f"{row['reweight_factor']:.3f}"
              + (f" | observed {obs:+.4f} -> raw {row['transfer_raw']:.2f}x "
                 f"reweighted {row['transfer_reweighted']:.2f}x"
                 if u["observed_move_s"] is not None else "  [PROSPECTIVE]"), flush=True)

    rec["seconds"] = round(time.time() - t0, 1)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.only and prev.get("updates"):
        prev["updates"].update(rec["updates"])
        prev["seconds"] = rec["seconds"]
        rec = prev
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
