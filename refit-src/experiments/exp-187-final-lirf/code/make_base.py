"""np_base input of the dry run: the saved value (the exp159 entry of exp-167's cal-002 run) on the 383 January/July
LIRF no-plan rows. February/June rows are left out; the runner fills them (--fill mix).

    .venv/bin/python experiments/exp-187-final-lirf/code/make_base.py
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
ID = "MVT_ID_mvt"
n = pd.read_parquet(ROOT / "experiments/exp-187-final-lirf/out/ranking/noplan-rows.parquet")
e = pd.read_parquet(ROOT / "submissions/candidates/exp159-fold-F-HB1-w50-x.parquet")
v = pd.Series(e.TAXITIME_SEC_mvt.to_numpy(float), index=e[ID].to_numpy("float64").astype("int64")).reindex(n[ID].to_numpy())
assert len(n) == 383 and np.isfinite(v).all() and np.array_equal(v.to_numpy(), n.base.to_numpy())
out = ROOT / "experiments/exp-187-final-lirf/out/np-base-janjul.parquet"
pd.DataFrame({ID: n[ID].to_numpy("int64"), "np_base": v.to_numpy()}).to_parquet(out, index=False)
print("wrote", out, len(n))
