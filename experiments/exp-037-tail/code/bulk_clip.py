"""§7, measured instead of argued: what the physical clip does to ORDINARY rows.

The first two versions of §7 said the second-previous arrival's bound is a
near-hard ceiling on ordinary departures, that a clip would rarely bite, and
priced the movement at "+18.6 s of board if it is uncorrelated with the shipped
error".  Round 2 of the review pointed out that this is the wrong null and that
the thing can simply be scored, out of fold, on the shipped vector.  So it is.

For each of the three shipped folds, the margin `q` is the low quantile of
`u = g_prev2 - y` over that fold's TRAINING months only, per airport, and the
arm is

    arm = min(shipped_oof, g_prev2 - q)      on the non-noac rows of the fold

against the shipped out-of-fold vector itself.  The noac rows are excluded
because §5 already scored the clip on them.  Intervals are calendar-day
clustered, paired, 4,000 replicates -- the same instrument as everywhere else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from evaluate import day_boot_gain, rmse, load_shipped, FOLDS  # noqa: E402

LEVELS = (0.001, 0.01)
VAR = "g_prev2"


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "dep_2025.parquet",
                        columns=["MVT_ID_mvt", "apt", "month", "day", VAR, "y_true", "noac"])
    ship = load_shipped().rename(columns={"y_pred": "shipped", "y_true": "y_ship"})
    m = a.merge(ship, on="MVT_ID_mvt", how="inner", validate="one_to_one")
    m = m[m.noac == 0].copy()
    assert np.allclose(m.y_true, m.y_ship), "label mismatch against the shipped out-of-fold file"
    rows = []
    for fold in FOLDS:
        tr_months = set(D.FOLDS[fold].train_months)
        d = m[m.fold == fold].copy()
        tr = a[a.month.isin(tr_months) & (a.noac == 0)]
        u = (tr[VAR] - tr.y_true)
        for lv in LEVELS:
            q = u.groupby(tr.apt).quantile(lv)
            bound = d[VAR].to_numpy(float) - d.apt.map(q).to_numpy(float)
            bound = np.where(np.isfinite(bound), bound, np.inf)
            arm = np.minimum(d.shipped.to_numpy(float), bound)
            d["arm"] = arm
            moved = arm != d.shipped.to_numpy(float)
            r = {"fold": fold, "level": lv, "n": len(d), "n_moved": int(moved.sum()),
                 "share_moved": float(moved.mean()),
                 "shift_rms": float(np.sqrt(np.mean((arm - d.shipped.to_numpy(float)) ** 2))),
                 "rmse_shipped": rmse(d.y_true, d.shipped), "rmse_arm": rmse(d.y_true, d.arm),
                 "rmse_shipped_on_moved": rmse(d.y_true[moved], d.shipped[moved]) if moved.any() else np.nan,
                 "rmse_arm_on_moved": rmse(d.y_true[moved], d.arm[moved]) if moved.any() else np.nan,
                 "true_label_over_bound": float((d.y_true.to_numpy(float) > bound).mean())}
            r.update(day_boot_gain(d, "shipped", "arm"))
            rows.append(r)
            print(f"  {fold} {lv:g}: moved {r['n_moved']}, gain {r['gain']:+.2f} s "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "bulk_clip.csv", index=False)
    with pd.option_context("display.width", 240):
        print()
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
