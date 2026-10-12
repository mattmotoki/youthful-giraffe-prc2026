"""COMMON.md's required decomposition for the base and the served arm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import envelope as E  # noqa: E402
import frames as F26  # noqa: E402
import prc.data as D  # noqa: E402


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "corrections.parquet")
    # COMMON.md's "no-NM" slice is `nm_missing = 1`, NOT `prc.blend`'s `nonm`
    # regime: `regime_labels` assigns `noac` first, so a row that is both no-NM
    # and no-aircraft-type lands in `noac` and drops out of `nonm`. On july_loo
    # that is 339 rows and it moves the slice RMSE from 531.8 to 1,122.0. The
    # `exp-027 budget CORRECTION` ledger row is the same mistake caught the same
    # way, and this is the convention it settled on.
    B = pd.concat([F26.gates(f, check=False)[["MVT_ID_mvt", "nm_missing"]]
                   for f in E.FOLDS], ignore_index=True)
    R = R.merge(B, on="MVT_ID_mvt", how="left", validate="one_to_one")
    if R["nm_missing"].isna().any():
        raise SystemExit("some rows have no nm_missing flag")
    rec = []
    for f in E.FOLDS:
        g = R[R["fold"] == f]
        y = g["y_true"].to_numpy("float64")
        ids = g["MVT_ID_mvt"].to_numpy("float64")
        cols = {"stack-5 keep-noac (base)": g["y_pred"].to_numpy("float64")}
        for arm in ("step|lofo|winsor", "step-half|lofo|winsor"):
            cols[arm] = np.maximum(g["y_pred"].to_numpy("float64")
                                   + g[f"c::{arm}"].to_numpy("float64"), 60.0)
        for lab, p in cols.items():
            m = {"rmse": np.ones(len(g), bool), "le3600": y <= 3600,
                 "gt3600": y > 3600,
                 "no_nm": (g["nm_missing"].to_numpy() == 1)}
            row = dict(fold=f, variant=lab,
                       rmse_clean=D.rmse_clean(y, p, ids, f))
            for k, mm in m.items():
                row[k] = float(np.sqrt(np.mean((p[mm] - y[mm]) ** 2)))
            row["n_no_nm"] = int(m["no_nm"].sum())
            rec.append(row)
    T = pd.DataFrame(rec)
    T.to_csv(EXP / "analysis" / "decomp.csv", index=False)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
