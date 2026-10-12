"""exp-056-q12advocate: how much each prior correlates with the base prediction on V (label-free).

RESULTS 5.1 reads the null dry run's free-beta value as the true prior proxying a residual that
depends on the base p (the synthetic floor max(1, .)), which permuted priors cannot proxy. This
job measures the premise: the correlation, on V's population rows, between the base p and the
unit prior q = f_stand + f_rwy + f_op of `true` and of each permuted sham. No label is read.

    .venv/bin/python experiments/exp-056-q12advocate/code/prior_p_corr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

OUT = E.ANALYSIS / "prior-p-corr.json"


def main() -> None:
    NO.refuse_existing([OUT], False)
    res = {"what": __doc__.strip().splitlines()[0], "code_sha256": E.code_shas(), "rotations": {}}
    for r in E.ORDER:
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        if E.sha_file(E.ROOT / inp["npz"]) != inp["npz_sha256"]:
            raise SystemExit("the inputs are not the frozen file")
        z = np.load(E.ROOT / inp["npz"])
        P = z["pop_V"]
        corr = {}
        for prior in ("true",) + tuple(f"sham{s:02d}" for s in E.SHAM_SEEDS):
            q = np.zeros(len(P))
            for f in E.PRIOR_FAMS:
                c = z[f"code_V_{f}"]
                q[c >= 0] += z[f"prior_{prior}_{f}"][c[c >= 0]]
            corr[prior] = float(np.corrcoef(q[P], z["p_V"][P])[0, 1])
        sh = np.array([v for k, v in corr.items() if k != "true"])
        res["rotations"][r] = {"corr": corr, "sham_mean": float(sh.mean()), "sham_max_abs": float(np.abs(sh).max())}
        print(f"{r}: corr(q, p) on P, true {corr['true']:+.3f}; shams mean {sh.mean():+.3f}, max |corr| "
              f"{np.abs(sh).max():.3f}")
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"written {OUT.name}")


if __name__ == "__main__":
    main()
