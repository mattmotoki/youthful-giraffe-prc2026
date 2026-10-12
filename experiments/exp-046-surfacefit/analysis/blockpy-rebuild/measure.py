"""The rebuild's EXPECTED boundary diff, measured on a branch copy. block.py untouched.

Builds every month twice -- once with the shipped `block.py`, once with
`analysis/blockpy-rebuild/block_v2.py` -- and diffs the class column both ways, per airport.
Fold months build under their pin; ranking months build unpinned, as the serve does.

Writes nothing into `work/block-*.parquet`: the caches the serve reads are never touched.

    OMP_NUM_THREADS=3 .venv/bin/python \
        experiments/exp-046-surfacefit/analysis/blockpy-rebuild/measure.py <variant>

variant = "shipped" or "v2"
"""
import importlib.util, json, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
#: the CODE and the MEASUREMENTS live here, in `analysis/`, where they are committed; the
#: per-month class frames are 21 MB of rebuildable intermediate and go to the ignored tree
OUT = ROOT / "experiments/exp-046-surfacefit/work/blockpy-rebuild"
OUT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "experiments/exp-046-surfacefit/code"
sys.path.insert(0, str(CODE))

VARIANT = sys.argv[1]
if VARIANT == "v2":
    spec = importlib.util.spec_from_file_location("block", HERE / "block_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["block"] = mod
    spec.loader.exec_module(mod)

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

assert (VARIANT == "v2") == ("blockpy-rebuild" in str(Path(B.__file__))), "wrong module"
print("variant", VARIANT, "block from", B.__file__, flush=True)
print("held days:", sorted(B.HELD_AIRPORT_DAYS), flush=True)

R = A.R
#: EVERY column the block emits, not a chosen subset: the claim this measurement makes is
#: "no existing value moves", and a claim about a subset is not that claim.
COLS = None
rule, _ = A.serving_rule()
out = {}

for month in ["2025-01", "2025-07", "2025-12", "2026-01", "2026-07"]:
    ranking = month.startswith("2026")
    if ranking:
        rk = D.load_ranking()
        dep = rk[rk.PHASE_mvt == "DEP"].copy()
        dep["_m"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
        dep = dep[dep._m == month].drop(columns="_m").copy()
        pref = A.prefix_months_for("")
        pin = None
    else:
        dep = R.departures(month)
        pref = A.prefix_months_for(month)
        pin = B.pin_for_month(month)
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in pref], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    with B.pinned(pin):
        blk = B.build(dep, tr, require_rule=rule)
    g = blk.copy() if COLS is None else blk[[c for c in COLS if c in blk.columns]].copy()
    g["airport"] = dep.set_index("MVT_ID_mvt").ADEP_mvt.reindex(
        g.MVT_ID_mvt).astype(str).to_numpy()
    p = OUT / f"class-{month}-{VARIANT}.parquet"
    g.to_parquet(p, index=False)
    out[month] = {"rows": int(len(g)),
                  "obs_matched": int(g.obs_matched.sum()),
                  "obs_has_clock": int(g.obs_has_clock.sum()),
                  "path": p.name}
    print(month, json.dumps(out[month]), flush=True)

(HERE / f"summary-{VARIANT}.json").write_text(json.dumps(out, indent=2, sort_keys=True))
print("done", VARIANT)
