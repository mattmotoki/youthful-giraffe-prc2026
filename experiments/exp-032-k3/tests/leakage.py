"""Direct leakage test on the block as exp-032-k3's member builds it.

Two perturbation tests on one real month, each rebuilding the whole 68-column
block through `wblock.build` and diffing it against the unperturbed block.

  A  DEPARTURE outcomes.  Push 300 departures' own BLOCK_TIME_UTC_mvt three
     hours later and their TAXITIME_SEC_mvt to 9999 s.  `wblock._one_month`
     blanks those two columns on every DEP row before the block is built, so
     the requirement is the strong one: NO cell of the block may move, not
     merely the perturbed rows' own cells.  A departure's outcome must be
     invisible to itself AND to its neighbours.

  B  ARRIVAL taxi-in, the positive control.  Arrivals are NOT blanked -- an
     arrival's taxi-in is a legal input (`docs/input-rule.md`) and the block
     reads it.  Perturbing 300 arrivals MUST move some departure rows, or
     test A proves nothing: a block of constants would pass it.
"""
import sys
import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp-032-k3/code"))
sys.path.insert(0, str(ROOT))
import wblock as W
import prc.data as D

MONTH = "2025-07"
f0 = D.load_month(MONTH)
b0 = W.build(f0)
ph = f0["PHASE_mvt"].to_numpy()
print(f"frame {f0.shape}   block {b0.shape}   DEP {(ph=='DEP').sum()}  ARR {(ph=='ARR').sum()}",
      flush=True)

a0 = b0.to_numpy("float64")
rng = np.random.default_rng(0)


def rebuild_after(sel, label):
    f1 = f0.copy()
    f1.loc[f1.index[sel], "BLOCK_TIME_UTC_mvt"] = (
        f1.loc[f1.index[sel], "BLOCK_TIME_UTC_mvt"] + pd.Timedelta(hours=3))
    f1.loc[f1.index[sel], "TAXITIME_SEC_mvt"] = 9999.0
    b1 = W.build(f1)
    assert list(b1.columns) == list(b0.columns) and len(b1) == len(b0)
    a1 = b1.to_numpy("float64")
    both_nan = np.isnan(a0) & np.isnan(a1)
    d = np.where(both_nan, 0.0, np.abs(a0 - a1))
    d = np.nan_to_num(d, nan=np.inf)          # NaN on one side only = a change
    moved = int((d.max(axis=1) > 0).sum())
    print(f"\n[{label}] perturbed {len(sel)} rows -> "
          f"{moved} of {len(a0)} block rows moved, max change {d.max():.6g}", flush=True)
    return moved, d


ok = {}
have = f0["BLOCK_TIME_UTC_mvt"].notna() & f0["TAXITIME_SEC_mvt"].notna()

dep = np.flatnonzero((ph == "DEP") & have.to_numpy())
mv, d = rebuild_after(rng.choice(dep, 300, replace=False), "A  departures")
ok["A"] = (mv == 0)
if not ok["A"]:
    print("   columns that moved:", list(np.array(b0.columns)[(d > 0).any(axis=0)])[:20])

arr = np.flatnonzero((ph == "ARR") & have.to_numpy())
mv, _ = rebuild_after(rng.choice(arr, 300, replace=False), "B  arrivals (control)")
ok["B"] = (mv > 0)

print(f"\nA  a departure's own outcome reaches the block: "
      f"{'NO  -> contract holds' if ok['A'] else 'YES -> LEAK'}")
print(f"B  an arrival's taxi-in reaches the block:      "
      f"{'YES -> the test can fail' if ok['B'] else 'NO  -> test is inert'}")
raise SystemExit(0 if all(ok.values()) else 1)
