"""exp-051 iteration 2, POST HOC, zero fit: what the sham carrier does to the base by itself.

R2 read SHAM RAISES (every rho^S interval wholly above 1).  One number says why: the sham's own
value on july_loo, S0 over P0 at member 1 (June's rows with rbase permuted within airport,
over member 1 alone), beside the real June carrier's C0 over P0.  If the sham makes the base
WORSE, the steps priced over it have more to repair.  Not registered; it changes no reading.

    .venv/bin/python experiments/exp-051-transferadv/code/shamown.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common as C
from common import A, NO
import price2 as P2

OUT = C.ANALYSIS / "shamown.json"


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    p = pan_ship[P2.F]
    r = p.y_true.to_numpy("float64") - p.pbase.to_numpy("float64")
    P0 = C.load_vec(C.PLAIN[0])[P2.F]
    arms = {"S0_sham": P2.arm_vec("S", 1, 0)[P2.F], "C0_june": P2.arm_vec("C", 1, 0)[P2.F]}
    cd, uq = pd.factorize(p.day.to_numpy())
    rng = np.random.default_rng(P2.SEED)
    # the same july draws as price2.py: july_loo is the first fold drawn there
    dr = rng.integers(0, len(uq), size=(P2.N_BOOT, len(uq)))
    cnt = np.bincount(cd, minlength=len(uq)).astype("float64")
    out = {"what": __doc__.strip().splitlines()[0], "fold": P2.F, "member": 1, "arms": {}}
    for k, v in arms.items():
        inc = v - P0
        g = 2 * (r - P0) * inc - inc * inc          # the residual after the predecessor, P0
        sm = np.bincount(cd, weights=g, minlength=len(uq))
        bs = sm[dr].sum(1) / cnt[dr].sum(1)
        out["arms"][k] = {"july_s2_over_P0": float(g.mean()),
                          "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          "rows_moved": int((inc != 0).sum())}
        print(f"{k}: over member 1 on july {g.mean():+.2f} s2/row "
              f"[{out['arms'][k]['ci95'][0]:+.2f}, {out['arms'][k]['ci95'][1]:+.2f}]")
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"written to {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
