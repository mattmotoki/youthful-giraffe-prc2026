"""exp-185: the exp-046 class columns that the exp-130 package and the OPDI match read, for 2026 months,
from the exp-046 ranking block in its ungated (v25) form.

    PRC_DATA_DIR=... [PRC_RANKING_MODE=final] .venv/bin/python class_final.py <snapshot> <out_dir> <month> [...]

classvec.py cannot write February or June (it reads a served candidate that has no such rows), so this
run-only wrapper calls `serve.ranking_block()` of the snapshot with `_UNGATED = True` (as exp-148
features.py does) and writes `<out_dir>/<month>-v25.parquet` with
  obs_has_clock         the block's clock flag,
  served_by_exp_046     `usable` (arms.served_class), the column exp-148 features.py checks,
  noac                  no aircraft type,
  occupied_by_exp_046   obs_has_clock & ~noac (the relation exp-130 prepare.py asserts).
The months are set on serve.RANK_MONTHS and arms.RANK_MONTHS of the snapshot. January in the ranking
mode is the check against the saved 2026-01-v25.parquet.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

snap, out, months = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3:]
code = snap / "experiments/exp-046-surfacefit/code"
sys.path.insert(0, str(snap))
sys.path.insert(0, str(code))
import prc.data as D  # noqa: E402
import serve as S  # noqa: E402
import arms as A  # noqa: E402

assert all(m in getattr(D, "rank_months", lambda: ("2026-01", "2026-07"))() for m in months), months
S.RANK_MONTHS = list(months)
A.RANK_MONTHS = list(months)
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "analysis").mkdir()
    S.EXP = Path(tmp)
    S._UNGATED = True
    dep, blk = S.ranking_block()
    reach = json.loads((Path(tmp) / "analysis/ranking-reach.json").read_text())
dep["MVT_ID_mvt"] = dep["MVT_ID_mvt"].astype("int64")
assert dep["MVT_ID_mvt"].is_unique
out.mkdir(parents=True, exist_ok=True)
rec = {"snapshot_commit": (snap / "COMMIT").read_text().strip(), "mode": getattr(D, "ranking_mode", lambda: "ranking")(),
       "months": months, "rule_cover": blk.attrs.get("rule_cover"), "reach": reach, "files": {}}
for m in months:
    d = dep[dep.rank_month.eq(m)]
    cls = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].to_numpy(),
                        "obs_has_clock": d["obs_has_clock"].astype(bool).to_numpy(),
                        "served_by_exp_046": d["usable"].astype(bool).to_numpy(),
                        "noac": d["noac"].astype(bool).to_numpy()})
    cls["occupied_by_exp_046"] = cls.obs_has_clock & ~cls.noac
    p = out / f"{m}-v25.parquet"
    cls.to_parquet(p, index=False)
    rec["files"][m] = {"rows": int(len(cls)), "obs_has_clock": int(cls.obs_has_clock.sum()),
                       "served_by_exp_046": int(cls.served_by_exp_046.sum()), "noac": int(cls.noac.sum()),
                       "occupied_by_exp_046": int(cls.occupied_by_exp_046.sum()),
                       "by_airport": {a: {"rows": int(len(g)), "served": int(g.usable.sum()), "clock": int(g.obs_has_clock.sum())}
                                      for a, g in d.groupby("airport")}}
    print(m, rec["files"][m]["rows"], "clock", rec["files"][m]["obs_has_clock"], "served", rec["files"][m]["served_by_exp_046"], flush=True)
(out / "class-final.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
