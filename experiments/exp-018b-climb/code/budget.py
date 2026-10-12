"""exp-018b step 1: re-select the round budget by the multi-month rule.

The desk's rule: the median of the per-month optima over three held-out months
spanning both seasons. The three are February (winter), May (shoulder) and
August (summer), each as a leave-one-month-out fold over the other eleven.
None of them is a fold exp-018 or exp-018b reports, so the budget is not
chosen on a scored fold -- which was review round 1's finding 2 about the
first lineage, and this is the repair.

The curve is the LightGBM main head only (`main_model="lgb"`): the budget being
chosen is its round count, and running CatBoost at every checkpoint would cost
four hours to move a number the checkpoint grid cannot resolve.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/budget.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"))
import dataset as D            # noqa: E402
import model as M              # noqa: E402
from prc import data           # noqa: E402

EXP = Path(__file__).resolve().parents[1]
MONTHS = ("2025-02", "2025-05", "2025-08")
STEP, MAX_ROUNDS = 200, 2800


def main() -> None:
    out = {}
    for m in MONTHS:
        fold = data.Fold(f"{m}_loo", tuple(x for x in data.MONTHS if x != m), m)
        t0 = time.time()
        train = D.load(fold.train_months)
        valid = D.load([fold.valid_month])
        y = valid["y"].to_numpy(dtype="float64")
        train, (valid,) = D.add_encodings(train, [valid])
        print(f"{m}: prepared in {time.time()-t0:.0f}s  train={train.shape}", flush=True)
        curve: list[float] = []

        # the curve, printed by model.fit_predict, is captured by re-implementing
        # the loop here so the per-checkpoint value is returned rather than logged
        pred, info = M.fit_predict(train, valid, prepared=True, rounds=MAX_ROUNDS,
                                   clip_z=6000, exact_rounds=300, ghost_mode="mixbase",
                                   main_model="lgb", curve=STEP, y_valid=y)
        # re-read the printed curve from the booster: score every checkpoint
        booster = info["booster"]
        base_v = M._base(valid)
        gv = valid["ghost"].to_numpy() > 0.5
        keep = np.isfinite(y)
        for n in range(STEP, MAX_ROUNDS + 1, STEP):
            raw = booster.predict(valid[info["feats"]], num_iteration=n)
            p = np.clip(base_v + raw, 30.0, None)
            curve.append(float(np.sqrt(np.mean((y[keep] - p[keep]) ** 2))))
        best = int((np.argmin(curve) + 1) * STEP)
        out[m] = {"curve": curve, "step": STEP, "argmin_rounds": best,
                  "best": min(curve), "at_1200": curve[1200 // STEP - 1],
                  "n_valid": int(keep.sum()), "n_ghost": int(gv.sum()),
                  "secs": round(time.time() - t0, 1)}
        print(f"{m}: argmin {best} rounds, {min(curve):.2f}; at 1200 {out[m]['at_1200']:.2f}", flush=True)
        print("   curve:", [round(c, 2) for c in curve], flush=True)
        del train, valid

    optima = [out[m]["argmin_rounds"] for m in MONTHS]
    out["_rule"] = {"months": list(MONTHS), "optima": optima,
                    "median": int(np.median(optima))}
    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / "budget.json").write_text(json.dumps(out, indent=1))
    print(f"\nper-month optima {optima} -> BUDGET = {int(np.median(optima))} rounds")


if __name__ == "__main__":
    main()
