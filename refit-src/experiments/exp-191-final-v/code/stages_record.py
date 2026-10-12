"""Stages record of the V steps in a run tree, in the --stages form of check_final_candidate.py:
{"stages": [{"name", "changed_by_month": {"2026-01": n, ...}, "board_only": reason (only where by design)}]}.

Replays iso/candidate.build step by step on the 2026 (ranking) split and counts the rows each stage changes:
EXT, LONG, LIRF (rows of the primary blend each step changes), CHAIN (rows of the rerun rule chain and T11b-L that
differ from v64), NOAC, ROLL (rows each post step changes), then L2C (work/w2/l2c_ranking.npy against c5) and ORD-S
(work/w2/ords_ranking.npy against L2C). Asserts that the replay gives the c5 values of work/w2/base_ranking.parquet.
For EXT, LONG, LIRF, NOAC, ROLL and L2C it also gives eligible_by_month (the rows the stage acts on, the masks of
iso/candidate.py and w2/l2c.py) and asserts that each changed row is eligible. board_only is set only where each new month
without a change has no eligible row; any other stage that changes January/July and not both new months gets a note.

    python stages_record.py <run> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2])
sys.path.insert(0, str(run / "iso")); sys.path.insert(0, str(run / "w2"))
import candidate as C  # noqa: E402
import chain  # noqa: E402

ID = "MVT_ID_mvt"
MONTHS = {1: "2026-01", 2: "2026-02", 6: "2026-06", 7: "2026-07"}
st = chain.load(); base = chain.run()
month = st["contexts"]["ranking"]["b"].month.to_numpy()


def by_month(mask):
    return {MONTHS[m]: int((mask & (month == m)).sum()) for m in MONTHS}


stages = []
elig = {}   # stage name -> the ranking rows the stage acts on (the masks of iso/candidate.py)
b26 = st["contexts"]["ranking"]["b"]
ms26 = pd.read_parquet(run / "work/feat/ranking.parquet", columns=["mvt_sched"]).mvt_sched.to_numpy(float)
elig["EXT"] = b26.extension.to_numpy(bool)
elig["LONG"] = C.long_mask(st, "ranking")
elig["LIRF"] = C.lirf_cell(st)
prim = {s: base[s]["primary"].copy() for s in ("fold", "ranking")}
for name in ("ext", "long", "lirf"):
    before = prim["ranking"].copy()
    C.STEPS[name](st, base, prim)
    changed = prim["ranking"] != before
    assert not (changed & ~elig[name.upper()]).any(), f"{name} changes a row outside its mask"
    stages.append({"name": name.upper(), "changed_by_month": by_month(changed)})
res = chain.run(primary=prim)
stages.append({"name": "CHAIN", "changed_by_month": by_month(res["ranking"]["v64"] != base["ranking"]["v64"])})
import roll_step as RS  # noqa: E402
for name in ("noac", "roll"):
    before = {s: res[s]["v64"].copy() for s in ("fold", "ranking")}
    if name == "noac":
        elig["NOAC"] = C._noac_mix("ranking", before["ranking"])
    else:
        elig["ROLL"] = ((b26.cls.eq("noac") & b26.airport.eq("LIRF")).to_numpy() & RS.cross_rows("ranking", b26[ID].to_numpy())
                        & (ms26 >= RS.ROLL_LO) & (ms26 < RS.DAY))
    for s in ("fold", "ranking"):
        C.POST[name](s, res[s], base)
    changed = res["ranking"]["v64"] != before["ranking"]
    assert not (changed & ~elig[name.upper()]).any(), f"{name} changes a row outside its mask"
    stages.append({"name": name.upper(), "changed_by_month": by_month(changed)})
b = pd.read_parquet(run / "work/w2/base_ranking.parquet", columns=["MVT_ID_mvt", "c5"])
c5 = b.c5.to_numpy(float)
assert np.array_equal(res["ranking"]["v64"], c5), "the stage replay does not give the c5 values"
l2c = np.load(run / "work/w2/l2c_ranking.npy")
import base as B  # noqa: E402
import l2c as L  # noqa: E402
rk = B.load("ranking")
assert np.array_equal(rk[ID].to_numpy(), b[ID].to_numpy())
elig["L2C"] = L.masks(rk, "ranking")
assert not ((l2c != c5) & ~elig["L2C"]).any(), "L2C changes a row outside its mask"
stages.append({"name": "L2C", "changed_by_month": by_month(l2c != c5)})
ords = run / "work/w2/ords_ranking.npy"
if ords.exists():
    stages.append({"name": "ORD-S", "changed_by_month": by_month(np.load(ords) != l2c)})
# board_only only where each new month without a change has no eligible row (finalplan's rule for its Z stages);
# a new month with eligible rows and no change stays flagged for the checker
for s in stages:
    ch = s["changed_by_month"]
    if s["name"] in elig:
        s["eligible_by_month"] = by_month(elig[s["name"]])
    missing = [m for m in ("2026-02", "2026-06") if ch[m] == 0]
    if (ch["2026-01"] + ch["2026-07"]) > 0 and missing:
        el = s.get("eligible_by_month")
        if el is not None and all(el[m] == 0 for m in missing):
            s["board_only"] = f"no eligible row in {', '.join(missing)} (eligible_by_month: the rows the stage acts on)"
        else:
            s["note"] = "changes January/July and not both new months: read before the checker"
rec = {"run": str(run), "rows": int(len(month)), "stages": stages}
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
