"""Build `exp-016-stack-1.parquet`: the stacker applied to the members' ranking predictions.

The shipped object is exactly the rotated `keep-noac` configuration:

  * a five-seed bag of the `a5` stacker -- LightGBM on the FIVE member
    predictions and four summaries of them (`mem_mean`, `mem_std`, `mem_range`,
    `n_defined`), no row features, fitting the residual from the flat four-member
    average and adding it back;
  * fitted on all three folds' out-of-fold predictions (510,096 rows), which is
    the only labelled data that exists for it;
  * applied to the members' ranking predictions, prepared exactly as
    `prc.blend.build_candidate` prepares them (floor, then the residual guard on
    the twelve-month bounds);
  * **`noac` rows fall back to the flat four-member average**, which is what
    candidate 2 serves there. The stacker is fitted on 152 noac rows across two
    folds in the rotation and 191 across three here, and a non-linear function of
    five predictions fitted on that many heavy-tailed rows does not transfer
    (exp-008 finding 7 measured the linear version of the same failure).

Two mismatches are disclosed rather than assumed away, both in RESULTS.md: the
served members are trained on twelve months and the stacker learned to correct
eleven-month members, and the shipped stacker is fitted on three folds where the
rotation's stackers saw two.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import stack2 as S2  # noqa: E402
import why as W  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
sys.path.insert(0, str(M.ROOT / "experiments" / "exp-001-retro-v1"))
import guard as guard_mod  # noqa: E402

#: Two candidates, both `keep-noac`, differing only in whether the stacker sees
#: the row features. Round-1 review B2 showed the arm was never actually chosen:
#: at the seed mean the two are 0.019 s apart on the board and the four
#: (arm x restriction) configurations span 0.25 s while all beat blend2 by
#: 0.49-0.74 s. `analysis/arm_choice.csv` has the table.
ARMS = {
    "1": dict(cols="members", board=317.280,
              rotated={"july_loo": 295.527, "december": 213.934, "january": 342.689}),
    "2": dict(cols="all", board=317.529,
              rotated={"july_loo": 295.950, "december": 213.166, "january": 342.752}),
    # stack-3: the same members-only arm on exp-012's blend-3 member set (the
    # deice winter member). Numbers are what `analysis/swap_deice.csv` measured
    # with that member's OOF; the build refuses until its ranking predictions
    # exist, and the rotation should be re-run to confirm these before shipping.
    "3": dict(cols="members", board=317.215,
              rotated={"july_loo": 295.882, "december": 214.374, "january": 342.168}),
}
OFFLOAD = ("noac",)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "1"
    spec = ARMS[which]
    out_path = M.ROOT / "submissions" / "candidates" / f"exp-016-stack-{which}.parquet"
    # A candidate build needs every member SERVABLE, so ranking absences are
    # blocking here where the rotation tolerates them.
    bad = M.verify(require_ranking=True)
    if bad:
        raise SystemExit("member set is not servable:\n  " + "\n  ".join(bad))
    fit = pd.concat([M.fold_frame(f) for f in M.FOLDS], ignore_index=True)
    rk = M.ranking_frame()
    print(f"[submit] fit {fit.shape} ranking {rk.shape}", flush=True)

    import lightgbm as lgb
    Xf, Xa = S.stack_features(fit, M.NAMES), S.stack_features(rk, M.NAMES)
    keep = [c for c in Xf.columns
            if (c in S2.MEMBER_COLS if spec["cols"] == "members" else True)]
    yf = fit["y_true"].to_numpy("float64")
    base_f, base_a = S.flat4(fit), S.flat4(rk)
    # The rotation declared `airport` and `regime` as LightGBM categoricals
    # (stack.py, stack2.py, verify_r1.py, arm_choice.py). This build omitted them,
    # so for the arm that HAS those columns the served model was not the model
    # whose RMSEs were recorded -- review round 2, B2, measured at up to 149.7 s
    # per row. Declaring them here is what makes served == scored. For the
    # members-only arm `keep` contains neither column, so `cat` is empty and
    # candidate 1's bytes are unchanged.
    cat = [c for c in ("airport", "regime") if c in keep]
    print(f"[submit] {len(keep)} stack features, categoricals {cat or 'none'}", flush=True)
    preds = []
    for s in W.SEEDS:
        p = dict(S.LGB_BASE, seed=s, bagging_seed=s + 1,
                 feature_fraction_seed=s + 2, data_random_seed=s + 3)
        mdl = lgb.train(p, lgb.Dataset(Xf[keep], label=yf - base_f,
                                       categorical_feature=cat or None),
                        num_boost_round=S.LGB_ROUNDS)
        preds.append(mdl.predict(Xa[keep]))
        print(f"[submit] seed {s} done", flush=True)
    stacked = np.maximum(np.mean(preds, axis=0) + base_a, B.FLOOR)

    reg = rk["regime"].to_numpy()
    off = np.isin(reg, OFFLOAD)
    pred = np.where(off, np.maximum(base_a, B.FLOOR), stacked)
    lo, hi = -20657.0, 9069.0
    n_touch = guard_mod.count_touched(pred, rk["d_takeoff_aobt"].to_numpy("float64"), lo, hi)
    pred = guard_mod.apply(pred, rk["d_takeoff_aobt"].to_numpy("float64"), lo, hi, floor=B.FLOOR)
    print(f"[submit] {int(off.sum())} noac rows on the flat average; "
          f"residual guard touched {n_touch} rows", flush=True)

    sha = D.write_submission(pd.Series(pred, index=rk["MVT_ID_mvt"].to_numpy("float64")), out_path)
    tpl = D.load_submitting()
    got = pd.read_parquet(out_path)
    rec = {
        "path": str(out_path.relative_to(M.ROOT)), "sha256": sha, "n": int(len(got)),
        "members": list(M.NAMES),
        "stacker": f"{'a5 members-only' if spec['cols']=='members' else 'a2 members+gates'}, "
                   f"5-seed bag, 300 rounds",
        "offload": list(OFFLOAD), "fit_rows": int(len(fit)), "fit_folds": list(M.FOLDS),
        "n_noac_in_fit": int(fit["noac"].sum()),
        "board_estimate_rotated": spec["board"], "rotated": spec["rotated"],
        "guard_bounds": [lo, hi], "floor": B.FLOOR, "guard_touched": int(n_touch),
        "ids_match_template": bool(np.array_equal(got[D.ID].to_numpy("float64"),
                                                  tpl[D.ID].to_numpy("float64"))),
        "min": float(pred.min()), "median": float(np.median(pred)),
        "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
        "n_below_floor": int((pred < B.FLOOR - 1e-9).sum()),
        "n_nonfinite": int((~np.isfinite(pred)).sum()),
        "pins": M.PINS.name,
    }
    for name, path in (("candidate 2", "submissions/candidates/exp-012-blend-2.parquet"),):
        other = pd.read_parquet(M.ROOT / path)
        d = got[D.TARGET].to_numpy("float64") - other[D.TARGET].to_numpy("float64")
        rec[f"vs {name}"] = {"rms": float(np.sqrt((d ** 2).mean())),
                             "max": float(d.max()), "min": float(d.min()),
                             "n_differ": int((d != 0).sum())}
    (M.EXP / f"candidate-{which}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    M.config_from_argv()
    main()
