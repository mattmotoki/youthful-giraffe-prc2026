"""Why EDDM's false-match share is the worst of the ten, measured rather than assumed.

Round 3 finding 7 measured a worst-shift false-match share of 0.00120 at EDDM against a
lane-wide 0.00029, and the desk routed *"a look at EDDM's callsign pool"* into this
rebuild.  A pool that is unusually large is the obvious hypothesis: more broadcast forms
per ground run means more join keys, and every extra key is another chance for a wrong
departure to land on the segment.

This script tests that hypothesis against the alternative -- that what raises the share is
simply how MANY departure segments the airport offers a mistaken row inside the window --
by measuring both quantities per airport on the same month finding 7 was read on, and
correlating each with finding 7's per-airport shares.

    OMP_NUM_THREADS=3 .venv/bin/python \
        experiments/exp-046-surfacefit/analysis/blockpy-rebuild/density.py

Writes `analysis/eddm-density.json`.  Reads summaries only; fits nothing, serves nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))

import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

MONTH = "2026-01"                       # the month finding 7 was read on
rule, _ = A.serving_rule()

rk = D.load_ranking()
dep = rk[rk.PHASE_mvt == "DEP"].copy()
dep["_m"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
dep = dep[dep._m == MONTH]
n_dep = dep.groupby(dep.ADEP_mvt.astype(str)).size()

days = sorted(dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").unique())
allseg = B.load_summaries(B.AIRPORTS, days, require_rule=rule)
n_all = allseg.groupby(allseg.airport.astype(str)).size()
seg = allseg[allseg.kind.eq("departure")].copy()
if "seg_key" not in seg.columns:
    seg["seg_key"] = (seg.airport.astype(str) + "|" + seg.summary_day.astype(str) + "|"
                      + seg.index.astype(str))
forms = B.broadcast_forms(seg)

# THE DIRECT MECHANISM.  A shifted row matches falsely when its own join key is
# broadcast by ANOTHER departure segment of the same airport on the same day, near
# enough for the window.  That is key REUSE, and it is neither the pool size nor the
# segment count: an airport can broadcast many forms and still reuse none of them.
forms["_day"] = forms.summary_day.astype(str)
reuse = forms.groupby(["airport", "_day", "join_key"]).size()

rows = {}
for a in sorted(seg.airport.unique()):
    sa = seg[seg.airport == a]
    f = forms[forms.airport == a]
    r = reuse.xs(a, level="airport") if a in reuse.index.get_level_values(0) else None
    rows[a] = {
        "departures": int(n_dep.get(a, 0)),
        "departure_segments": int(len(sa)),
        "segments_per_departure": round(float(len(sa) / max(n_dep.get(a, 0), 1)), 4),
        "segments_all_kinds": int(n_all.get(a, 0)),
        "segments_all_kinds_per_departure": round(
            float(n_all.get(a, 0) / max(n_dep.get(a, 0), 1)), 4),
        "join_keys": int(len(f)),
        "mean_forms_per_segment": round(float(len(f) / max(len(sa), 1)), 4),
        "distinct_keys_per_day": int(r.size) if r is not None else 0,
        "mean_segments_per_key_per_day": round(float(r.mean()), 4) if r is not None else None,
        "share_of_keys_reused_that_day": round(float((r > 1).mean()), 5) if r is not None else None,
    }

FALSE = json.loads((EXP / "analysis/finding7.json").read_text())


def _shares(d: dict) -> dict:
    """finding 7's worst-shift false-match share per airport, wherever it stores it."""
    out = {}
    def walk(node, apt=None):
        if isinstance(node, dict):
            if "false_share" in node and apt:
                out[apt] = max(out.get(apt, 0.0), float(node["false_share"]))
            for k, v in node.items():
                walk(v, k if k in B.AIRPORTS else apt)
        elif isinstance(node, list):
            for v in node:
                walk(v, apt)
    walk(d)
    return out


share = _shares(FALSE)
for a, r in rows.items():
    r["false_share_worst_shift"] = share.get(a)

have = [a for a in rows if rows[a]["false_share_worst_shift"] is not None]


def _corr(x, y):
    x, y = np.asarray(x, "float64"), np.asarray(y, "float64")
    rp = float(np.corrcoef(x, y)[0, 1])
    rs = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
    return {"pearson": round(rp, 3), "spearman": round(rs, 3)}


out = {
    "month": MONTH,
    "rule_sha256": rule,
    "question": ("is EDDM's false-match share (finding 7: 0.00120, worst of ten) a "
                 "CALLSIGN POOL problem, as the desk's rebuild item assumed, or a "
                 "SEGMENT DENSITY problem?"),
    "by_airport": rows,
    "airports_with_a_share": sorted(have),
    "correlation_false_vs_mean_forms": _corr(
        [rows[a]["mean_forms_per_segment"] for a in have],
        [rows[a]["false_share_worst_shift"] for a in have]),
    "correlation_false_vs_segments_per_departure": _corr(
        [rows[a]["segments_per_departure"] for a in have],
        [rows[a]["false_share_worst_shift"] for a in have]),
    "correlation_false_vs_segments_all_kinds_per_departure": _corr(
        [rows[a]["segments_all_kinds_per_departure"] for a in have],
        [rows[a]["false_share_worst_shift"] for a in have]),
    "correlation_false_vs_share_of_keys_reused": _corr(
        [rows[a]["share_of_keys_reused_that_day"] for a in have],
        [rows[a]["false_share_worst_shift"] for a in have]),
    "correlation_false_vs_mean_segments_per_key": _corr(
        [rows[a]["mean_segments_per_key_per_day"] for a in have],
        [rows[a]["false_share_worst_shift"] for a in have]),
}
p = EXP / "analysis/eddm-density.json"
p.write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps({a: rows[a] for a in sorted(rows)}, indent=1))
for k, v in out.items():
    if k.startswith("correlation_"):
        print(f"{k:56s}", json.dumps(v))
print("written", p)
