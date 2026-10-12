"""The bundle, gated WHOLE against the served head (desk 25.5's revival condition).

`code/chainfold.py` prices the ungated rows as an increment over **v18's own arm**, which
answers "what do they add to v18".  The desk's revival condition is a different question:

    the increment over the SERVED head clear on every fold -- the next with-clock gain
    bundles into this candidate and is gated whole.

So this prices `d_bundle - d_served_head`, where the bundle is the arm refitted on the
class that carries BOTH the turnaround pass and the gated ungated rows, and the served head
is the arm on the pre-pass matcher at this serve's own pins
(`work/d-passonly-6fdf3951-guard995.npz`, section 25.3 -- july and january pins identical,
december's cache one population behind).  That is the whole of what an upload would change,
priced in one number, with the ~10 % wrong placements inside it.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/chainbundle.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402

THR = 1299.0
OUT = EXP / "analysis/chain-bundle.json"
VEC = EXP / "work/d-bundle-ungated1299-guard995.npz"
HEADS = {"v18_own_arm": EXP / "work/d-guard-leaves_255x300_lr050-guard995.npz",
         "the_served_head_6fdf3951": EXP / "work/d-passonly-6fdf3951-guard995.npz"}


def main() -> None:
    t0 = time.time()
    rule = A.serving_rule()[0]
    blocks = {A.FOLD_MONTH[f]: CF.build_month(A.FOLD_MONTH[f],
                                              A.prefix_months_for(A.FOLD_MONTH[f]), rule)
              for f in A.FOLDS}
    gaps = {m: CF.GAP_CUT_S for m in blocks}
    pan = CF.panel_for(blocks, THR, gaps)
    added = {f: int(((pan[f].match_mode.fillna(-1) == 4)
                     & pan[f].usable.to_numpy(bool)).sum()) for f in A.FOLDS}
    print("rows added to the class:", added, flush=True)
    d_new = CF.fit_arm(pan)
    np.savez_compressed(VEC, **d_new)
    print(f"fitted and saved ({time.time()-t0:.0f}s)", flush=True)

    out = {"what": __doc__.strip().splitlines()[0], "threshold_s": THR,
           "rows_added_to_the_class": added, "vector": str(VEC.relative_to(ROOT)),
           "against": {}}
    for name, p in HEADS.items():
        z = np.load(p)
        d_head = {f: z[f].astype("float64") for f in A.FOLDS}
        rec = CF.price_increment(pan, d_new, d_head, np.random.default_rng(A.SEED))
        out["against"][name] = rec
        ci = rec["board_all_row"]["interval"]["ci95"]
        print(f"\nvs {name}:  {rec['board_all_row']['gain_s']:+.4f} "
              f"[{ci[0]:+.4f},{ci[1]:+.4f}]  clean {rec['board_clean']['gain_s']:+.4f}  "
              f"every-fold-clear {rec['every_fold_interval_clear']}", flush=True)
        for f in A.FOLDS:
            r = rec["by_fold"][f]; c = rec["by_fold_clean"][f]
            print(f"   {f:10s} s2/row {r['mse_gain_s2']:>+9.2f} ci "
                  f"[{r['ci95'][0]:>+8.2f},{r['ci95'][1]:>+8.2f}]   clean "
                  f"{c['mse_gain_s2']:>+9.2f} ci [{c['ci95'][0]:>+8.2f},{c['ci95'][1]:>+8.2f}]",
                  flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
