"""L2C's eligible rows against the clock rows of finalclock's head part (a count built another way, desk 10-09).

L2C acts on the clock rows with an NM join, minus the extension rows and the LIRF January-scope rows (w2/l2c.py
masks; the class is the pack's cls). finalclock's head part marks its clock rows in the column `clock`. Per month:
the head's clock rows, my chain's clock rows and the rows where the two differ; then the head's clock rows reduced
by the same three conditions (NM join, no extension, not LIRF January scope), next to L2C's eligible rows and the
rows L2C changes. Any differing row is named (count by month and airport, first ids). No pass line.

    python l2c_clock_check.py <run> <head part> <head SHA-256> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
run, head, want, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, str(run / "iso")); sys.path.insert(0, str(run / "w2"))
import base as B  # noqa: E402
import finalmode as FM  # noqa: E402
import l2c as L  # noqa: E402

got = hashlib.sha256(head.read_bytes()).hexdigest()
assert got == want, f"{head}: SHA-256 {got} is not {want}"
r = B.load("ranking")
h = pd.read_parquet(head, columns=[ID, "clock"])
assert np.array_equal(h[ID].to_numpy("float64").astype("int64"), r[ID].to_numpy("int64")), "head part is not in chain order"
assert h["clock"].notna().all()
hc = h["clock"].to_numpy().astype(bool)
mc = (r.cls == "clock").to_numpy()
joined = r.joined.to_numpy() == 1
ext = r.extension.to_numpy(bool)
lj = (r.airport.eq("LIRF") & r.month.isin(FM.JAN_SCOPE)).to_numpy()
elig = L.masks(r, "ranking")
assert np.array_equal(elig, mc & joined & ~ext & ~lj)
other = hc & joined & ~ext & ~lj
changed = np.load(run / "work/w2/l2c_ranking.npy") != r.c5.to_numpy(float)
assert not (changed & ~elig).any()
month = r.month.to_numpy(); ap = r.airport.to_numpy()


def by_month(m):
    return {int(k): int((m & (month == k)).sum()) for k in (1, 2, 6, 7)}


def named(m):
    return {"rows": int(m.sum()), "by_month": by_month(m),
            "by_month_airport": {f"{k[0]}-{k[1]}": int(v) for k, v in
                                 pd.Series(1, index=pd.MultiIndex.from_arrays([month[m], ap[m]])).groupby(level=[0, 1]).sum().items()},
            "first_ids": r[ID].to_numpy("int64")[m][:10].tolist()}


rec = {"run": str(run), "head_part": str(head), "head_sha256": got, "head_clock_dtype": str(h["clock"].dtype),
       "head_clock": by_month(hc), "chain_clock": by_month(mc),
       "clock_head_only": named(hc & ~mc), "clock_chain_only": named(mc & ~hc),
       "head_clock_joined": by_month(hc & joined), "head_clock_joined_no_extension": by_month(hc & joined & ~ext),
       "head_reduced_to_l2c_scope": by_month(other), "l2c_eligible": by_month(elig), "l2c_changed": by_month(changed),
       "eligible_head_only": named(other & ~elig), "eligible_chain_only": named(elig & ~other)}
# the month-loop trap: a new month where the head has L2C-scope clock rows and L2C has no eligible row
rec["warning_months"] = [k for k in (2, 6) if rec["head_reduced_to_l2c_scope"][k] > 0 and rec["l2c_eligible"][k] == 0]
out.write_text(json.dumps(rec, indent=1))
print(f"{'month':>5} {'head clock':>10} {'chain clock':>11} {'head scope':>10} {'L2C elig':>9} {'L2C changed':>11}")
for k in (1, 2, 6, 7):
    print(f"{k:>5} {rec['head_clock'][k]:>10} {rec['chain_clock'][k]:>11} {rec['head_reduced_to_l2c_scope'][k]:>10} "
          f"{rec['l2c_eligible'][k]:>9} {rec['l2c_changed'][k]:>11}")
print("clock rows head only", rec["clock_head_only"]["by_month"], "chain only", rec["clock_chain_only"]["by_month"])
print("L2C scope rows head only", rec["eligible_head_only"]["by_month"], "chain only", rec["eligible_chain_only"]["by_month"])
for k in rec["warning_months"]:
    print(f"WARNING month {k}: the head part has {rec['head_reduced_to_l2c_scope'][k]} L2C-scope clock rows, L2C has no eligible row")
