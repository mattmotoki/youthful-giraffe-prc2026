"""exp-060-coldprior: what the cold prior aligns with on V (post-hoc, descriptive, chooses nothing).

NOT registered in SPEC.  It runs after the verdict, the gate and the bound are frozen on disk, it
selects nothing, and no number here enters a reading, a gate, a pick or a bound.  It answers the
`why` that the assignment asks for, on the cold tables, with exp-056's own iteration-1 diagnostic
(its SPEC section 4.1 items 5 and 6) so the two are comparable:

  * alignment  a = sum(e*q) / sum(q*q)  of the unit cold prior q against the `zero` arm's V
    residual e, raw and after removing each airport's mean from both (the correction is a
    within-airport additive one, so the demeaned number is the one the correction can use);
  * the same per family, and per family per departure-support stratum n_A(k);
  * the correlation, for scale.

Inputs: `work/inputs60-<r>.npz` (frozen, label-free) and the committed `oof/zero-iso_<r>_v.parquet`
of this experiment, whose `y_true` comes from the single recorded V read of `price60.py`.  No new
month is opened.

    .venv/bin/python experiments/exp-060-coldprior/code/align60.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

STRATA = (("lt50", 0, 50), ("50_499", 50, 500), ("ge500", 500, 1 << 62))
PRIOR_ARMS = ("true", "gmean", "donor", "shuffle") + E.SHAM_ARMS


def align(e: np.ndarray, q: np.ndarray) -> dict:
    qq = float(q @ q)
    out = {"n": int(len(e)), "sum_q2": qq, "sum_eq": float(e @ q)}
    out["a"] = out["sum_eq"] / qq if qq > 0 else None
    if qq > 0 and e.std() > 0 and q.std() > 0:
        out["corr"] = float(np.corrcoef(e, q)[0, 1])
    else:
        out["corr"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / "align60.json"
    NO.refuse_existing([p], a.overwrite)
    t0 = time.time()
    rec = {"what": ("post-hoc descriptive alignment of the cold prior with the zero arm's V "
                    "residual; registered nowhere, chooses nothing"),
           "utc": E.utc(), "strata": [list(s) for s in STRATA], "rotations": {}}
    for r in E.ORDER:
        z = np.load(E.WORK / f"inputs60-{r}.npz")
        oof = pd.read_parquet(E.EXP / f"oof/zero-iso_{r}_v.parquet")
        ids = z["ids_V"]
        if not np.array_equal(oof["MVT_ID_mvt"].to_numpy("int64"), ids):
            raise SystemExit(f"{r}: the zero OOF is not in the frozen V id order")
        y = oof["y_true"].to_numpy("float64")
        pred = oof["y_pred"].to_numpy("float64")
        rows = z["pop_V"] & np.isfinite(y)
        e = (y - pred)[rows]
        apt = z["code_V_apt"][rows]
        ed = E.demean_by_airport(e, apt)
        codes = {f: z[f"code_V_{f}"][rows] for f in E.PRIOR_FAMS}
        nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
        out = {"V": E.ROT[r][0], "rows": int(rows.sum()),
               "zero_oof_sha256": E.sha_file(E.EXP / f"oof/zero-iso_{r}_v.parquet"),
               "inputs_npz_sha256": E.sha_file(E.WORK / f"inputs60-{r}.npz"), "arms": {}}
        for arm in PRIOR_ARMS:
            pv = {f: z[f"prior_{arm}_{f}"] for f in E.PRIOR_FAMS}
            per = {f: np.where(codes[f] >= 0, pv[f][np.maximum(codes[f], 0)], 0.0)
                   for f in E.PRIOR_FAMS}
            q = sum(per.values())
            rec_arm = {"all_families": {"raw": align(e, q),
                                        "airport_demeaned": align(ed, E.demean_by_airport(q, apt))},
                       "by_family": {}}
            for f in E.PRIOR_FAMS:
                qf = per[f]
                d = {"raw": align(e, qf),
                     "airport_demeaned": align(ed, E.demean_by_airport(qf, apt)), "by_support": {}}
                n_row = np.where(codes[f] >= 0, nA[f][np.maximum(codes[f], 0)], -1)
                for name, lo, hi in STRATA:
                    m = (n_row >= lo) & (n_row < hi)
                    d["by_support"][name] = (align(ed[m], E.demean_by_airport(qf, apt)[m])
                                             | {"share_of_rows": float(m.mean())})
                rec_arm["by_family"][f] = d
            out["arms"][arm] = rec_arm
        sh = [out["arms"][s]["all_families"]["airport_demeaned"]["a"] for s in E.SHAM_ARMS]
        out["sham_a_airport_demeaned"] = {"mean": float(np.mean(sh)), "min": float(np.min(sh)),
                                          "max": float(np.max(sh))}
        rec["rotations"][r] = out
    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    for r in E.ORDER:
        o = rec["rotations"][r]
        print(r, "true a(demeaned) =", round(o["arms"]["true"]["all_families"]["airport_demeaned"]["a"], 5),
              "| shams", {k: round(v, 5) for k, v in o["sham_a_airport_demeaned"].items()},
              "| corr", round(o["arms"]["true"]["all_families"]["airport_demeaned"]["corr"], 4), flush=True)
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
