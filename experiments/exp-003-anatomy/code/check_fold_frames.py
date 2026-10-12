"""Show that the cached feature path is equivalent to `prc.data.fold_frames`.

COMMON.md asks for `fold_frames(fold)` behind every measured number. The
ablation harness reads per-month feature frames built by `features.py` instead,
for speed: rebuilding 1.9 M rows of neighbour features for every one of two
dozen ablations would cost more than the fits. This script is the argument that
the two paths are the same object:

1. the fold month lists in `ablate.py` are taken from `prc.data.FOLDS`;
2. `features.build_month` is bit-identical on the raw month and on the
   ranking-style masked month (`features.py check`), so nothing it produces
   could have come from the blanked columns;
3. the labels, ids and row counts the harness scores against are identical to
   the ones `fold_frames` hands back, which is what this script checks.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import prc.data as d
from features import ART


def check(fold_name):
    fold = d.FOLDS[fold_name]
    train, valid_masked, y_valid = d.fold_frames(fold)
    va = pd.read_parquet(ART / f"{fold.valid_month}.parquet",
                         columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
    dep = valid_masked["PHASE_mvt"] == "DEP"
    ref = pd.DataFrame({"MVT_ID_mvt": valid_masked.loc[dep, d.ID].to_numpy(),
                        "y": y_valid[dep].to_numpy()}).sort_values("MVT_ID_mvt")
    got = va[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].rename(
        columns={"TAXITIME_SEC_mvt": "y"}).sort_values("MVT_ID_mvt")
    ok_ids = np.array_equal(ref["MVT_ID_mvt"].to_numpy(), got["MVT_ID_mvt"].to_numpy())
    ok_y = np.allclose(ref["y"].to_numpy(), got["y"].to_numpy(), equal_nan=True)
    # the harness must never see a departure block time for the validation month
    masked_ok = bool(valid_masked.loc[dep, "BLOCK_TIME_UTC_mvt"].isna().all())
    tr_months = list(d.FOLDS[fold_name].train_months)
    tr_rows = sum(len(pd.read_parquet(ART / f"{m}.parquet", columns=["MVT_ID_mvt"]))
                  for m in tr_months)
    ref_tr = int((train["PHASE_mvt"] == "DEP").sum())
    print(f"{fold_name:9s} valid ids identical={ok_ids} labels identical={ok_y} "
          f"fold_frames masks dep block time={masked_ok} "
          f"train departures cache={tr_rows} fold_frames={ref_tr} "
          f"match={tr_rows == ref_tr}")
    return ok_ids and ok_y and masked_ok and tr_rows == ref_tr


if __name__ == "__main__":
    assert all(check(f) for f in ("july", "december", "january")), "cache path differs from fold_frames"
    print("OK: the cached feature path scores the same rows and labels as prc.data.fold_frames")
