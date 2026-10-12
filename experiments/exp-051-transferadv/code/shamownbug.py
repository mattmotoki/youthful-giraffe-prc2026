"""exp-051, Sol round 1 answer B3: the receipt of shamown.py's first, wrong run.  ZERO FIT.

shamown.py's first run priced each arm over member 1 with the wrong residual, r = y - pbase in
place of r = y - pbase - P0.  Its output was written to analysis/shamown.json, and the fixed run
then replaced that file with --overwrite, against the write-once rule (SPEC section I1,
"operating limits").  Only logs/shamown-bug-r-without-P0.log kept the wrong numbers.  This script
rebuilds that first run's arithmetic exactly as it ran (shamown.py with the one wrong line) and
writes it under its own name.  It must reproduce the kept log's printed values to their printed
digits, or it writes nothing.  It is a receipt of an error, not a result: no reading uses it.

    .venv/bin/python experiments/exp-051-transferadv/code/shamownbug.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import common as C
from common import NO
import price2 as P2

OUT = C.ANALYSIS / "shamown-bug-r-without-P0.json"
LOG = C.EXP / "logs/shamown-bug-r-without-P0.log"
#: the first run's printed values (LOG), s2/row: point, 2.5 %, 97.5 %
PRINTED = {"S0_sham": (-14325.47, -15674.44, -13286.08), "C0_june": (-245.12, -405.97, -72.45)}


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
    dr = rng.integers(0, len(uq), size=(P2.N_BOOT, len(uq)))
    cnt = np.bincount(cd, minlength=len(uq)).astype("float64")
    out = {"what": __doc__.strip().splitlines()[0], "fold": P2.F, "member": 1,
           "wrong_line": "g = 2 * r * inc - inc * inc, with r = y - pbase (the fix uses r - P0)",
           "log": str(LOG.relative_to(C.ROOT)), "log_sha256": C.sha256(LOG), "arms": {}}
    for k, v in arms.items():
        inc = v - P0
        g = 2 * r * inc - inc * inc                  # THE FIRST RUN'S WRONG LINE, kept on purpose
        sm = np.bincount(cd, weights=g, minlength=len(uq))
        bs = sm[dr].sum(1) / cnt[dr].sum(1)
        got = (float(g.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        if any(abs(x - y) > 0.005 for x, y in zip(got, PRINTED[k])):
            raise SystemExit(f"{k}: {got} does not reproduce the kept log's {PRINTED[k]}")
        out["arms"][k] = {"july_s2_over_P0_WRONG": got[0], "ci95_WRONG": [got[1], got[2]],
                          "printed_in_log": list(PRINTED[k]), "rows_moved": int((inc != 0).sum())}
        print(f"{k}: the first run's wrong value {got[0]:+.2f} s2/row [{got[1]:+.2f}, {got[2]:+.2f}] "
              f"reproduces the log's {PRINTED[k]}")
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out, overwrite=a.overwrite)
    print(f"written to {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
