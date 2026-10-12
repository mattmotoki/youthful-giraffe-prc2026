"""RESULTS section 47.70: the WRITABLE-ROW COUNT of 2025-06 and 2025-02, harvested.

ZERO FIT.  This runs `onemonth.nov_rows`, which is the same harvest every fit prints
before it fits, and counts what comes back.  It reads no target of the five cells 47.60.3
predicts -- a writable-row count is a COVARIATE known at harvest time, and 47.69 showed
that registering a prediction WITHOUT it is what made the 47.59.6 band wrong.  No
LightGBM is imported and nothing is fitted.
"""
import json, sys, pathlib
import numpy as np, pandas as pd
EXP = pathlib.Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(ROOT))
import arms as A, chainfold as CF
import rung511 as R
import onemonth as OM

rule = A.serving_rule()[0]
blk = {}
for f in A.FOLDS:
    mth = A.FOLD_MONTH[f]
    b = CF.build_month(mth, A.prefix_months_for(mth), rule)
    R.assert_mode3(b, mth, "shipped")
    blk[mth] = b
pan_ship = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
_, scols = A.shipped_cols(pan_ship)
print(f"panel built, {len(scols)} shipped columns", flush=True)

out = {}
for m in ("2025-06", "2025-02"):
    df = OM.nov_rows(scols, month=m)
    # SECTION 47.70.  `nov_rows` returns the WHOLE month; the WRITABLE rows are the
    # `usable` subset, which is what every fit actually carries (onemonth.py:331).
    # Counting len(df) instead gave 143,731 for 2025-02 against the registered 68,122 --
    # the 2025-02 control in this script is what caught it, and it is here for that.
    fit = df[df.usable.to_numpy(bool)]
    ap = sorted(fit["airport"].astype(str).unique())
    out[m] = {"writable_rows": int(len(fit)), "month_rows": int(len(df)),
              "airports": len(ap), "which": ap}
    print(f"{m}: {len(fit):,} writable of {len(df):,} month rows, "
          f"{len(ap)} airports {ap}", flush=True)

if out["2025-02"]["writable_rows"] != 68122:
    raise SystemExit(f"CONTROL FAILED: 2025-02 gives {out['2025-02']['writable_rows']}, "
                     f"not the registered 68,122 -- the counter is not the registered one")
print("control OK: 2025-02 reproduces its registered 68,122", flush=True)

p = EXP / "analysis/junerows.json"
p.write_text(json.dumps(out, indent=2, sort_keys=True))
print("written to", p, flush=True)
