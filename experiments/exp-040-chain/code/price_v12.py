"""Revival (a): the held candidate's update recomposed on v12, the head the board holds.

    PRC_THREADS=4 .venv/bin/python experiments/exp-040-chain/code/price_v12.py

§9g named two revivals for `exp-040-chain.parquet`. The first was "the increment
clear on every fold against the head the board actually holds… recomposed on
k2climb's head if that ships". **It shipped**: v12 = `exp-035-k2climb-v11member`,
sha `48dd572d…`, board 262.7418 (BOARD READING 11, tail rule). So this file asks the
same question of the same update against the new base, with no fit and no new
member: `v12 + (H' - v10)`, where `H' - v10` is exactly the vector the candidate
already adds to v11.

v12's fold analogue is built the way exp-035 §26a builds the served file --

    head   = the climbed member through exp-027's refit-bulk arm
             (`exp-035-k2climb/oof/cand-k2c-mod-refit-bulk-{fold}.parquet`)
    masked = head at exp-036's eight kept airports, v7 elsewhere
    served = max(masked + 0.217 * (stack-5 - masked), 60)

-- and is **gated on exp-035 §26c's own published numbers** before any exp-040
number is read: v12 against v11 must return all-row board +0.3762, july_loo
+0.4809, december +0.5379 and january clean +0.3986. A v12 analogue that does not
reproduce those is not v12 and nothing may be priced on it.

Writes `analysis/price-v12.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
sys.path.insert(0, str(ROOT))
import align as A                                           # noqa: E402
import score as Sc                                          # noqa: E402
import price_v11 as P11                                     # noqa: E402
import prc.data as D                                        # noqa: E402

FOLDS = A.FOLDS
HELD = P11.HELD
S_V11 = P11.S_V11                       # 0.217, the amplitude v11 and v12 share
FLOOR = P11.FLOOR
K2C = ROOT / "experiments/exp-035-k2climb/oof/cand-k2c-mod-refit-bulk-{fold}.parquet"
KEEP = json.loads((ROOT / "experiments/exp-036-mask/analysis/arm-refitbulk-mask.json")
                  .read_text())["keep"]
#: exp-035 §26c, the v12 candidate's own published fold reading against v11.
PUB_V12 = {"board": 0.3762, "july_loo": 0.4809, "december": 0.5379,
           "january_clean": 0.3986}
GATE = 5e-3


def vectors() -> dict:
    """P11's y/v10/v11/candidate per fold, plus v12 and the candidate on v12."""
    V = P11.vectors()
    ref = Sc.reference()
    for f in FOLDS:
        d = pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        apt = d["airport"].astype(str).to_numpy()
        s5 = pd.Series(ref[f]["y_pred_stack5"].to_numpy("float64"),
                       index=ref[f]["MVT_ID_mvt"].to_numpy("float64")) \
            .reindex(ids).to_numpy()
        v6 = pd.Series(ref[f]["y_pred_v6"].to_numpy("float64"),
                       index=ref[f]["MVT_ID_mvt"].to_numpy("float64")) \
            .reindex(ids).to_numpy()
        v7 = np.where(apt == "EHAM", s5, v6)
        k = pd.read_parquet(K2C.as_posix().format(fold=f))
        head = pd.Series(k["y_pred"].to_numpy("float64"),
                         index=k["MVT_ID_mvt"].to_numpy("float64")).reindex(ids).to_numpy()
        if not np.isfinite(head).all():
            raise SystemExit(f"{f}: the climbed head is not defined on every packet row")
        on8 = np.isin(apt, KEEP)
        masked = np.where(on8, head, v7)
        V[f]["mask_moved"] = int((masked != head).sum())
        v12 = np.maximum(masked + S_V11 * (s5 - masked), FLOOR)
        u = V[f]["cand"] - V[f]["v11"]          # exactly (H' - v10), floor aside
        V[f]["v12"] = v12
        V[f]["cand12"] = np.maximum(v12 + u, FLOOR)
        V[f]["v12_floor_bound"] = int((V[f]["cand12"] != v12 + u).sum())
    return V


def main() -> None:
    V = vectors()

    # ---- the gate: v12 must reproduce exp-035 §26c against v11 --------------
    g = P11.price(V, "v11", "v12")
    clean = {}
    for f in FOLDS:
        d = pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        clean[f] = (D.rmse_clean(pd.Series(V[f]["y"]), pd.Series(V[f]["v11"]),
                                 pd.Series(ids), f)
                    - D.rmse_clean(pd.Series(V[f]["y"]), pd.Series(V[f]["v12"]),
                                   pd.Series(ids), f))
    off = {"board": g["gain_board"] - PUB_V12["board"],
           "july_loo": g["folds"]["july_loo"]["gain_rmse"] - PUB_V12["july_loo"],
           "december": g["folds"]["december"]["gain_rmse"] - PUB_V12["december"],
           "january_clean": clean["january"] - PUB_V12["january_clean"]}
    print("[gate] v12 against v11: " + "  ".join(f"{k} {v:+.4f}" for k, v in off.items()))
    if max(abs(v) for v in off.values()) > GATE:
        raise SystemExit(f"the v12 fold analogue does not reproduce exp-035 §26c: {off}")
    print("[gate] PASS — this is v12", flush=True)

    rec = {"base": "v12 (exp-035-k2climb-v11member, sha 48dd572d…, board 262.7418)",
           "gate_vs_exp035_26c": {"published": PUB_V12, "offset": off,
                                  "mask_moved_rows": {f: V[f]["mask_moved"] for f in FOLDS}},
           "floor_bound_rows": {f: V[f]["v12_floor_bound"] for f in FOLDS},
           "v12_vs_v11": g,
           "increment_vs_v12": P11.price(V, "v12", "cand12"),
           "composite_vs_v10": P11.price(V, "v10", "cand12"),
           "for_comparison_increment_vs_v11": P11.price(V, "v11", "cand")}
    (EXP / "analysis" / "price-v12.json").write_text(json.dumps(rec, indent=2))
    for k in ("v12_vs_v11", "increment_vs_v12", "composite_vs_v10",
              "for_comparison_increment_vs_v11"):
        r = rec[k]
        print(f"\n[{k}]  board {r['gain_board']:+.4f} "
              f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
        for f in FOLDS:
            x = r["folds"][f]
            print(f"  {f:9s} {x['gain_rmse']:+.4f} "
                  f"1d [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] "
                  f"3d [{x['ci_3day'][0]:+.4f}, {x['ci_3day'][1]:+.4f}] "
                  f"scale {x['scale']:.3f}  {'CLEAR' if x['clear'] else 'crosses zero'}")
        print(f"  every fold clear: 1-day {r['every_fold_clear']}  "
              f"3-day {r['every_fold_clear_3day']}")


if __name__ == "__main__":
    main()
