"""Training and prediction rows of the s7 fits from a work/seq7 folder (the masks of w2/seq/fit.py, no GPU).
Released: held 1,702,170 and full 2,062,575 training rows (check_members7c.log).

    python train_rows.py <run>
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]) / "work/seq7"
meta = json.loads((D / "meta.json").read_text())
c_cat = np.load(D / "c_cat.npy"); y = np.load(D / "y.npy").astype(float); nm = np.load(D / "nm_raw.npy").astype(float)
month = np.load(D / "month.npy"); dom = np.load(D / "dom.npy"); is26 = np.load(D / "is2026.npy")
BUFFER = {(12, 31), (2, 1), (6, 30), (8, 1)}
has = np.isfinite(nm) & (c_cat[:, meta["c_cat"].index("actype")] > 0)
lab = has & ~is26 & np.isfinite(y) & (y < 20000)
buf = np.array([(int(m), int(x)) in BUFFER for m, x in zip(month, dom)])
held = lab & ~np.isin(month, [1, 7]) & ~buf
out = {"departures": int(len(y)), "rows_2026": int(is26.sum()), "has": int(has.sum()), "full_train": int(lab.sum()),
       "held_train": int(held.sum()), "full_equal_released": int(lab.sum()) == 2062575,
       "held_equal_released": int(held.sum()) == 1702170, "finite_y_on_2026": int(np.isfinite(y[is26]).sum()),
       "has_2026_by_month": {int(m): [int((has & is26 & (month == m)).sum()), int((is26 & (month == m)).sum())]
                             for m in np.unique(month[is26])}}
print(json.dumps(out))
