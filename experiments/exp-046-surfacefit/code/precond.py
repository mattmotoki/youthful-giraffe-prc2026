"""RESULTS section 47.17's registered PRECONDITION: how far is base6 from the true price base?

Section 47.17 named the design's main risk before any data existed.  On 2025-11 the arm's
target is the residual of a SIX-member base; on the fold months it is the residual of the true
price base.  Where the two bases differ systematically, the 2025-11 rows carry
`pbase_6 - pbase_true` as signal the tree will fit and then misapply to fold rows -- and the
loss would look exactly like "rows do not transfer", the wrong conclusion about the wrong thing.

Both bases exist on the three fold months, so this is PURE ARITHMETIC, no fit.  Queue's §7bk
supplies base6 there ROTATED the same way `pbase_true` is (fold f fitted on the other two), so
the difference prices the missing member plus the refit, not the rotation.

**The threshold was registered before the numbers**: rms above 25 s on any fold and the screen
is run but its result is NOT evidence about rows, and this file says so before the read.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/precond.py
"""
from __future__ import annotations

import hashlib
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

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import rung511 as R                                                   # noqa: E402

OOF = ROOT / "experiments/exp-047-queue/oof"
OUT = EXP / "analysis/precond-base6.json"
THRESHOLD_S = 25.0                       # registered in section 47.17 before the data
SHAS = {"july_loo": "fd4a57932dc099d5", "december": "b465c317fd391bad",
        "january": "535bbce18b0d90b7", "2025-11": "339ca16ed421a718"}


def main() -> None:
    t0 = time.time()
    for name, want in SHAS.items():
        got = hashlib.sha256((OOF / f"base6--{name}.parquet").read_bytes()).hexdigest()
        if not got.startswith(want):
            raise SystemExit(f"base6--{name}: sha {got[:16]} != the registered {want}")
    print("all four base6 shas match the desk's line", flush=True)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})

    rec = {"what": __doc__.strip().splitlines()[0], "threshold_s": THRESHOLD_S,
           "registered_in": "RESULTS.md section 47.17, commit a89252a, before any of this ran",
           "shas": SHAS, "by_fold": {}}
    worst = 0.0
    for f in A.FOLDS:
        b6 = pd.read_parquet(OOF / f"base6--{f}.parquet")
        p = pan[f]
        m = b6.set_index("MVT_ID_mvt")
        idx = p.MVT_ID_mvt.to_numpy("float64")
        pred6 = m.y_pred.reindex(idx).to_numpy("float64")
        ytrue6 = m.y_true.reindex(idx).to_numpy("float64")
        pbase = p.pbase.to_numpy("float64")
        y = p.y_true.to_numpy("float64")
        matched = np.isfinite(pred6)
        # the join must be the same rows, or the difference is a row mismatch, not a base one
        bad = int((np.abs(ytrue6[matched] - y[matched]) > 1e-6).sum())
        if bad:
            raise SystemExit(f"{f}: {bad} rows where base6's y_true differs from the panel's "
                             f"-- the join is wrong and the difference below would be noise")
        d = pred6 - pbase
        usable = p.usable.to_numpy(bool) & matched
        ap = p.airport.to_numpy().astype(str)
        by_ap = {}
        for a in pd.unique(ap[usable]):
            s = usable & (ap == a)
            by_ap[a] = {"rows": int(s.sum()),
                        "rms_s": float(np.sqrt(np.mean(d[s] ** 2))),
                        "mean_s": float(d[s].mean())}
        rms_u = float(np.sqrt(np.mean(d[usable] ** 2)))
        worst = max(worst, rms_u)
        rec["by_fold"][f] = {
            "rows_in_panel": int(len(p)), "rows_matched": int(matched.sum()),
            "rows_usable_and_matched": int(usable.sum()),
            "rms_s_usable": rms_u,
            "rms_s_all_matched": float(np.sqrt(np.mean(d[matched] ** 2))),
            "mean_s_usable": float(d[usable].mean()),
            "p99_abs_s": float(np.percentile(np.abs(d[usable]), 99)),
            "rmse_base6_vs_y": float(np.sqrt(np.mean((y[usable] - pred6[usable]) ** 2))),
            "rmse_true_vs_y": float(np.sqrt(np.mean((y[usable] - pbase[usable]) ** 2))),
            "by_airport": dict(sorted(by_ap.items(), key=lambda kv: -kv[1]["rms_s"])),
            "passes_25s": bool(rms_u <= THRESHOLD_S)}
        print(f"  {f:9s} rms {rms_u:8.3f} s  mean {d[usable].mean():+8.3f}  "
              f"p99|d| {np.percentile(np.abs(d[usable]), 99):8.2f}  "
              f"rows {int(usable.sum()):,}  passes {rms_u <= THRESHOLD_S}", flush=True)

    rec["worst_fold_rms_s"] = worst
    rec["passes"] = bool(worst <= THRESHOLD_S)
    rec["verdict"] = (
        "PASS -- the target mismatch is a stated caveat and the screen means what it claims"
        if rec["passes"] else
        "FAIL -- the screen is run but its result is NOT evidence about rows, as registered")
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"\n  worst fold rms {worst:.3f} s against the registered {THRESHOLD_S:.0f} s "
          f"-> {rec['verdict']}")
    for f in A.FOLDS:
        top = list(rec["by_fold"][f]["by_airport"].items())[:3]
        print(f"    {f:9s} worst airports: "
              + ", ".join(f"{a} {v['rms_s']:.1f} s ({v['rows']:,})" for a, v in top))
    print("written to", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
