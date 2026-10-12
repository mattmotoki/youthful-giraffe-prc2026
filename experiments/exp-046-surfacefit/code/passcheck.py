"""The no-op guard: the turnaround pass must place its reference counts, per month.

The desk, 2026-09-09, with the MATCHER_SHA line: *assert per month that mode-3 rows equal
your reference counts so a no-op cannot pass*.  `block.build` runs the pass only when it is
handed the ARR rows, and a caller that forgets them gets a block with no mode-3 row and no
error, so this is the check that the wiring of `arms.month_block` and `code/serve.py` is
still in place after any edit.

The counts are read through the SAME paths the fold table and the serve use -- the fold
months through `arms.month_block` (pinned), the ranking months through `serve`'s own build
-- so the guard tests the wiring rather than re-deriving it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/passcheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import serve as S                                                     # noqa: E402,F401
import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

OUT = EXP / "analysis/pass-check.json"
#: measured on the branch copy and reproduced by the shipped file before the commit
#: (`analysis/chain-spec.md` §9.2, `logs/wirecheck.log`)
REFERENCE = {"2026-01": 1382, "2026-07": 1797,
             "2025-07": 1170, "2025-12": 861, "2025-01": 731}


def main() -> None:
    out = {"what": "mode-3 rows per month, through the paths the table and the serve use",
           "matcher_sha256": A.MATCHER_SHA, "reference": REFERENCE, "months": {}}
    ok = True

    for m in A.FOLD_MONTH.values():
        blk = A.month_block(m, A.prefix_months_for(m))
        n = int((blk.match_mode == 3).sum())
        rec = {"mode_3": n, "reference": REFERENCE[m], "rows": int(len(blk)),
               "obs_has_clock": int(blk.obs_has_clock.sum()),
               "pinned_airport_days": len(B.pin_for_month(m) or ()),
               "block_cache_key": A.block_cache_key(m, None)}
        rec["matches_reference"] = n == REFERENCE[m]
        ok &= rec["matches_reference"]
        out["months"][m] = rec
        print(f"{m}  mode-3 {n:5d}  reference {REFERENCE[m]:5d}  "
              f"{'OK' if rec['matches_reference'] else 'DIFFERS'}   "
              f"({rec['pinned_airport_days']} pinned airport-days, "
              f"key {rec['block_cache_key']})", flush=True)

    rk = D.load_ranking()
    rk["_m"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    tr = pd.concat([pd.read_parquet(D.month_file(x),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for x in A.prefix_months_for("")], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    rule, _ = A.serving_rule()
    for m in A.RANK_MONTHS:
        dep = rk[rk.PHASE_mvt.eq("DEP") & rk._m.eq(m)].drop(columns="_m").copy()
        arr = rk[rk.PHASE_mvt.eq("ARR") & rk._m.eq(m)]
        blk = B.build(dep, tr, require_rule=rule, arrivals=arr)
        n = int((blk.match_mode == 3).sum())
        rec = {"mode_3": n, "reference": REFERENCE[m], "rows": int(len(blk)),
               "obs_has_clock": int(blk.obs_has_clock.sum()),
               "placed_by_the_pass": blk.attrs["turnaround"]["placed"]}
        rec["matches_reference"] = n == REFERENCE[m]
        ok &= rec["matches_reference"]
        out["months"][m] = rec
        print(f"{m}  mode-3 {n:5d}  reference {REFERENCE[m]:5d}  "
              f"{'OK' if rec['matches_reference'] else 'DIFFERS'}", flush=True)

    out["every_month_matches_its_reference"] = bool(ok)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\nevery month matches its reference: {ok}\nwritten {OUT}")
    if not ok:
        raise SystemExit("a month does not place its reference count; the wiring or the "
                         "population moved, and nothing is priced until it is explained")


if __name__ == "__main__":
    main()
