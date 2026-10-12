"""The kill step of section 31.5: does the plane-metric correction move the fold class?

Rebuilds a fold month with `analysis/chain-check/block_ungated_km.py` -- the same probe with
the section 31.2 correction and ONE statement otherwise different -- and counts, row by row,
the `match_mode` 4 rows whose `chain_runway_agrees` flips against the block the candidate was
gated on. Nothing the review is holding is touched: the corrected blocks are written under
their own `-ungated-km` name and the shipped `-ungated` caches are only read.

The decision this feeds: at fold lower bounds of +98.90 / +77.22 / +47.99 s2/row over
87,868 / 100,939 / 69,839 moved rows, a change of tens of rows in the added class cannot move
an interval. If the count is that small, the refinement is recorded as MEASURED NEGLIGIBLE
rather than refitted.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/planekill.py [month ...]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import prc.data as D                                                  # noqa: E402
import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "block_ungated_km", EXP / "analysis/chain-check/block_ungated_km.py")
KM = importlib.util.module_from_spec(_spec)
sys.modules["block_ungated_km"] = KM
_spec.loader.exec_module(KM)

#: 2025-07 first: the largest board share (SHARES 0.557) and the cheapest of the three
ORDER = ["2025-07", "2025-12", "2025-01"]
OUT = EXP / "analysis/plane-kill.json"


def build_km(month: str, rule: str) -> pd.DataFrame:
    p = EXP / f"work/block-{month}-{A.block_cache_key(month, rule)}-ungated-km.parquet"
    if p.exists():
        return pd.read_parquet(p)
    dep, _ = D.mask_like_ranking(D.load_month(month))
    arr = dep[dep.PHASE_mvt == "ARR"].copy()
    dp = dep[dep.PHASE_mvt == "DEP"].copy()
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for(month)], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    KM.PROBE.clear()
    with KM.pinned(KM.pin_for_month(month)):
        blk = KM.build(dp, tr, require_rule=rule, arrivals=arr)
    blk.to_parquet(p, index=False)
    return blk


def gate(b: pd.DataFrame) -> np.ndarray:
    m4 = b.match_mode.fillna(-1).to_numpy("float64") == 4.0
    return (m4
            & (b.chain_gap_s.to_numpy("float64") > CF.GAP_CUT_S)
            & (b.chain_abs_dt_offblock.to_numpy("float64") <= 1299.0)
            & (b.chain_runway_agrees.fillna(0).to_numpy("float64") == 1.0))


def main() -> None:
    t0 = time.time()
    months = [m for m in sys.argv[1:] if m in ORDER] or ORDER[:1]
    rule = A.serving_rule()[0]
    rec = {"what": __doc__.strip().splitlines()[0],
           "probe": "analysis/chain-check/block_ungated_km.py",
           "correction": "section 31.2: equirectangular scaling in check (ii)",
           "months": {}}
    for m in months:
        old = CF.build_month(m, A.prefix_months_for(m), rule)
        new = build_km(m, rule)
        if not old.MVT_ID_mvt.equals(new.MVT_ID_mvt):
            old = old.sort_values("MVT_ID_mvt").reset_index(drop=True)
            new = new.sort_values("MVT_ID_mvt").reset_index(drop=True)
            if not old.MVT_ID_mvt.equals(new.MVT_ID_mvt):
                raise SystemExit(f"{m}: the two builds do not carry the same rows")
        n3_old = int((old.match_mode.fillna(-1) == 3).sum())
        n3_new = int((new.match_mode.fillna(-1) == 3).sum())
        if n3_old != n3_new:
            raise SystemExit(f"{m}: mode-3 rows moved, {n3_old} -> {n3_new}; check (ii) "
                             "is reached only by mode 4 and cannot do that")
        ao = old.chain_runway_agrees.fillna(-1).to_numpy("float64")
        an = new.chain_runway_agrees.fillna(-1).to_numpy("float64")
        m4 = old.match_mode.fillna(-1).to_numpy("float64") == 4.0
        go, gn = gate(old), gate(new)
        rec["months"][m] = {
            "mode3_unchanged": n3_old,
            "mode4_admitted": int(m4.sum()),
            "runway_answer_flips_on_mode4": int((m4 & (ao != an)).sum()),
            "flips_agree_to_disagree": int((m4 & (ao == 1.0) & (an != 1.0)).sum()),
            "flips_disagree_to_agree": int((m4 & (ao != 1.0) & (an == 1.0)).sum()),
            "kept_by_the_gate_shipped": int(go.sum()),
            "kept_by_the_gate_corrected": int(gn.sum()),
            "leaves_the_class": int((go & ~gn).sum()),
            "joins_the_class": int((gn & ~go).sum()),
            "net_change_in_the_class": int(gn.sum() - go.sum()),
        }
        r = rec["months"][m]
        print(f"{m}: mode3 {n3_old} unchanged | mode4 {r['mode4_admitted']} | "
              f"runway flips {r['runway_answer_flips_on_mode4']} "
              f"({r['flips_agree_to_disagree']} out, {r['flips_disagree_to_agree']} in) | "
              f"class {r['kept_by_the_gate_shipped']} -> {r['kept_by_the_gate_corrected']} "
              f"(net {r['net_change_in_the_class']:+d})  ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
