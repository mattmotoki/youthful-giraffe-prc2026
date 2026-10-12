"""The corrected ranking-side pool, for section 33.4's acceptance test.  No serve.

Section 32.1 registered the acceptance test for the plane-metric correction: `poolrecon.py`
against the CORRECTED ranking block must read `kept_only_by_the_pool` = 0 **and**
`kept_only_by_the_block` = 0 -- the two instruments reconciled in both directions, because
the whole 2,456 / 2,493 disagreement was check (ii)'s distance metric and nothing else.

This builds the two ranking months through `analysis/chain-check/block_ungated_km.py` and
dumps the same seven columns `serve.py` dumps under `--ungated`, so `poolrecon.py` can read
it unchanged.  **Nothing is written that a serve writes**: no candidate bytes, no manifest,
no vector -- one parquet under `work/` and nothing else.  The shipped `-ungated` pool and the
probe reviewed in round 7 are read and never touched.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/kmpool.py
"""
from __future__ import annotations

import importlib.util
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

RANK_MONTHS = ["2026-01", "2026-07"]
GATE = {"step_0_gap_cut_s": 62.405, "check_i_abs_dt_offblock_max_s": 1299.0,
        "check_ii_runway_agreement": True}
COLS = ["MVT_ID_mvt", "rank_month", "airport", "match_mode", "chain_gap_s",
        "chain_abs_dt_offblock", "chain_runway_agrees"]
OUT = EXP / "work/ranking-ungated-pool-km.parquet"


def main() -> None:
    t0 = time.time()
    rk = D.load_ranking()
    dep = rk[rk.PHASE_mvt == "DEP"].copy()
    if dep.BLOCK_TIME_UTC_mvt.notna().any() or dep[D.TARGET].notna().any():
        raise SystemExit("the ranking file carries a departure block time or target")
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for("")], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    rule, _ = A.serving_rule()
    dep["_m"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    parts = []
    for m in RANK_MONTHS:
        sel = dep._m == m
        if not sel.any():
            raise SystemExit(f"the ranking file carries no departure in {m}")
        arr = rk[rk.PHASE_mvt.eq("ARR")
                 & rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").eq(m)]
        KM.PROBE.clear()
        b = KM.build(dep[sel].drop(columns="_m").copy(), tr, require_rule=rule,
                     arrivals=arr)
        n3 = int((b.match_mode.fillna(-1) == 3).sum())
        b = CF.restrict(b, GATE["check_i_abs_dt_offblock_max_s"],
                        GATE["step_0_gap_cut_s"], GATE["check_ii_runway_agreement"])
        parts.append(b)
        print(f"{m}: built, mode-3 {n3}, rows {len(b)}  ({time.time()-t0:.0f}s)",
              flush=True)
    dep = dep.drop(columns="_m")
    blk = pd.concat(parts, ignore_index=True)
    dep = dep.merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["rank_month"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    pool = dep[np.isfinite(dep.chain_gap_s.to_numpy("float64"))
               | np.isfinite(dep.chain_abs_dt_offblock.to_numpy("float64"))]
    pool[COLS].reset_index(drop=True).to_parquet(OUT, index=False)
    keep = ((pool.chain_gap_s.to_numpy("float64") > GATE["step_0_gap_cut_s"])
            & (pool.chain_abs_dt_offblock.to_numpy("float64")
               <= GATE["check_i_abs_dt_offblock_max_s"])
            & (pool.chain_runway_agrees.fillna(0).to_numpy("float64") == 1.0))
    print(f"pool {len(pool)} rows, kept by the gate {int(keep.sum())}", flush=True)
    print("written to", OUT.relative_to(ROOT), f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
