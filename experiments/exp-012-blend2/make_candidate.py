"""Write a numbered standing-blend candidate and refresh the repo-wide table.

  python make_candidate.py members-cand1.json ranking-cand1.json 1 [mode]

`mode` is `nnls` (default) or `flat`, and it is written into the candidate
config as `mode`, because the desk's blend rule makes it a per-candidate
decision: with four members, ship the flat average unless a fitted vector beats
flat in EVERY rotation. `flat` ignores the fitted vector entirely and gives each
member 1/k, so there is nothing to be unstable across months.

Shipped weights are fitted on ALL three validation months (the rotation is how
the candidate's score is estimated, not how its weights are chosen). Members are
prepared with the floor and the residual guard inside `prc.blend.build_candidate`
before blending, so a served member is the same estimator as the out-of-fold
member its weight was fitted on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import blend as BL  # noqa: E402  (exp-012's own; HERE is first on sys.path)
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "exp-001-retro-v1"))
import guard as guard_mod  # noqa: E402


def main() -> None:
    members_cfg, ranking_cfg, number = sys.argv[1], sys.argv[2], sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "nnls"
    if mode not in ("nnls", "ols", "flat"):
        raise SystemExit(f"mode must be nnls, ols or flat, not {mode!r}")
    mem = json.loads((HERE / members_cfg).read_text())["members"]
    rank = json.loads((HERE / ranking_cfg).read_text())
    missing = [k for k in mem if k not in rank]
    if missing:
        raise SystemExit(f"members without ranking predictions: {missing}")

    data, names = BL.load(mem)
    w, nrows = BL.fit_all(data, names, mode)
    T, W = BL.rotate(data, names, mode)
    per_fold = {r.fold: r.rmse for r in T.itertuples()}
    be = BL.board_estimate(per_fold)
    # The rule is decided on the rotation, so record both sides of it, not just
    # the winner: a later reader must be able to see that the rule was applied
    # rather than that the better number was picked.
    rule = {}
    for m in ("nnls", "flat"):
        Tm, _ = BL.rotate(data, names, m)
        rule[m] = {r.fold: round(r.rmse, 4) for r in Tm.itertuples()}
    rule["fitted_beats_flat_everywhere"] = all(
        rule["nnls"][f] < rule["flat"][f] for f in BL.FOLDS)
    print(f"blend rule: fitted beats flat in every rotation? "
          f"{rule['fitted_beats_flat_everywhere']}  -> shipping {mode}")
    print("shipped weights (mode %s, fitted on all three months, %d rows):" % (mode, nrows))
    print("  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, w)))
    print("rotated: " + "  ".join(f"{f} {per_fold[f]:.2f}" for f in BL.FOLDS)
          + f"   board estimate {be:.2f}")

    lo, hi = guard_mod.bounds_from_months(D.MONTHS, 0.0, 1.0)
    cfg = {"ranking": rank, "mode": mode, "blend_rule": rule,
           "weights": {"_all": dict(zip(names, map(float, w)))},
           "floor": B.FLOOR, "guard_bounds": [lo, hi], "late_cut": B.LATE_CUT,
           "prepare_members": True, "blend_version": B.VERSION,
           "fit_rows": int(nrows), "fit_folds": list(BL.FOLDS),
           "rotated": {f: round(per_fold[f], 4) for f in BL.FOLDS},
           "board_estimate": round(be, 4)}
    out = ROOT / "submissions" / "candidates" / f"exp-012-blend-{number}.parquet"
    rec = B.build_candidate(cfg, out, ROOT)
    cfg["candidate"] = rec
    (HERE / f"blend-config-{number}.json").write_text(json.dumps(cfg, indent=2))
    print("\ncandidate: " + json.dumps(rec, indent=2))
    print(f"\nwrote blend-config-{number}.json")


if __name__ == "__main__":
    main()
