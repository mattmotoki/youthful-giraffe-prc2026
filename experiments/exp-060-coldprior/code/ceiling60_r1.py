"""exp-060-coldprior: how large the per-stratum cold signal is on V (post-hoc, descriptive).

This REPLACES `code/ceiling60.py`, which review round 1 finding 1 showed is wrong in two ways:
it removed each airport's mean BEFORE masking a support stratum (so the masked column was not the
one a hard restriction would use), and it added the nine family x stratum values as if the columns
were orthogonal.  `analysis/ceiling60.json` and every number taken from it are withdrawn; this
job's `analysis/ceiling60-r1.json` replaces them.

What it computes, for `true` and the ten permuted shams, from the frozen inputs and the committed
`zero` OOF of this experiment:

  * `e` = the `zero` arm's V residual on the population rows, each airport's mean removed;
  * for each family f and support stratum s, `q[f,s]` = the unit cold prior of family f, set to
    zero outside stratum s, and THEN each airport's mean removed (mask first, demean second);
  * `one_scale`     - the best single scale of one column:  (e.q)^2 / (q.q);
  * `per_airport`   - the best scale per airport of one column: sum_a (e_a.q_a)^2 / (q_a.q_a);
  * `joint_nine`    - the best joint fit of all nine columns at once (least squares, pinv);
  * `joint_nine_per_airport` - the same fit run inside each airport.

Each is converted to board seconds against the `zero` arm's own fold RMSE.

NONE of these is an upper bound on the route section 12 of RESULTS names.  That route also
carries a threshold n*, a ridge penalty lambda, a support weight c, the served floor and a beta
per airport and family taken from a seven-value grid, and its regressor is the ridge-solved slope
vector, not the raw prior.  These numbers say how much of `zero`'s V residual the masked cold
prior can explain in sample; they do not say what the route would score.

Post-hoc, descriptive, registered nowhere, chooses nothing.

    .venv/bin/python experiments/exp-060-coldprior/code/ceiling60_r1.py
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
ARMS = ("true",) + E.SHAM_ARMS


def ls_gain(e: np.ndarray, Q: np.ndarray) -> float:
    """The squared-error reduction of the best least-squares fit of the columns of Q to e."""
    if Q.shape[1] == 0 or not np.any(Q):
        return 0.0
    beta, *_ = np.linalg.lstsq(Q, e, rcond=None)
    return float(e @ e - ((e - Q @ beta) @ (e - Q @ beta)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / "ceiling60-r1.json"
    NO.refuse_existing([p], a.overwrite)
    t0 = time.time()
    oos = E.load_json(E.ANALYSIS / "oos.json")
    z_rmse = {r: oos["tables"][r]["zero|B"]["rmse"] for r in E.ORDER}
    n_all = {r: oos["tables"][r]["zero|B"]["rows"] for r in E.ORDER}
    rec = {"what": ("how large the per-stratum cold signal is on V in sample; post-hoc, "
                    "descriptive, chooses nothing; NOT an upper bound on the route of RESULTS 12"),
           "replaces": "analysis/ceiling60.json (withdrawn; review round 1 finding 1)",
           "utc": E.utc(), "strata": [list(s) for s in STRATA], "arms": {},
           "zero_board_rmse_s": float(np.sqrt(sum(E.BOARD[r] * z_rmse[r] ** 2 for r in E.BOARD)))}
    red = {arm: {} for arm in ARMS}
    for r in E.ORDER:
        z = np.load(E.WORK / f"inputs60-{r}.npz")
        oof = pd.read_parquet(E.EXP / f"oof/zero-iso_{r}_v.parquet")
        if not np.array_equal(oof["MVT_ID_mvt"].to_numpy("int64"), z["ids_V"]):
            raise SystemExit(f"{r}: the zero OOF is not in the frozen V id order")
        y = oof["y_true"].to_numpy("float64")
        rows = z["pop_V"] & np.isfinite(y)
        e = (y - oof["y_pred"].to_numpy("float64"))[rows]
        apt = z["code_V_apt"][rows]
        ed = E.demean_by_airport(e, apt)
        codes = {f: z[f"code_V_{f}"][rows] for f in E.PRIOR_FAMS}
        nA = {f: z[f"nA_{f}"] for f in E.PRIOR_FAMS}
        order = np.argsort(apt, kind="stable")
        bounds = np.searchsorted(apt[order], np.arange(apt.max() + 2))
        for arm in ARMS:
            cols, names = [], []
            for f in E.PRIOR_FAMS:
                pv = z[f"prior_{arm}_{f}"]
                q_full = np.where(codes[f] >= 0, pv[np.maximum(codes[f], 0)], 0.0)
                n_row = np.where(codes[f] >= 0, nA[f][np.maximum(codes[f], 0)], -1)
                for name, lo, hi in STRATA:
                    q = np.where((n_row >= lo) & (n_row < hi), q_full, 0.0)   # mask FIRST
                    cols.append(E.demean_by_airport(q, apt))                  # then demean
                    names.append(f"{f}|{name}")
            Q = np.column_stack(cols)
            d = red[arm].setdefault(r, {})
            for j, nm in enumerate(names):
                q = Q[:, j]
                d[nm] = {"one_scale": ls_gain(ed, q[:, None]),
                         "per_airport": sum(ls_gain(ed[order][bounds[k]:bounds[k + 1]],
                                                    q[order][bounds[k]:bounds[k + 1], None])
                                            for k in range(len(bounds) - 1))}
            d["joint_nine"] = {"one_scale": ls_gain(ed, Q),
                               "per_airport": sum(ls_gain(ed[order][bounds[k]:bounds[k + 1]],
                                                          Q[order][bounds[k]:bounds[k + 1]])
                                                  for k in range(len(bounds) - 1))}
    for arm in ARMS:
        out = {}
        for nm in red[arm]["jul"]:
            out[nm] = {}
            for kind in ("one_scale", "per_airport"):
                new = {r: float(np.sqrt(max(z_rmse[r] ** 2 - red[arm][r][nm][kind] / n_all[r], 0.0)))
                       for r in E.ORDER}
                out[nm][kind] = {
                    "sse_reduction_s2": {r: red[arm][r][nm][kind] for r in E.ORDER},
                    "board_gain_s": rec["zero_board_rmse_s"] - float(
                        np.sqrt(sum(E.BOARD[r] * new[r] ** 2 for r in E.BOARD))),
                    "dec_gain_s": z_rmse["dec"] - new["dec"]}
        rec["arms"][arm] = out
    for nm in ("stand|lt50", "joint_nine"):
        for kind in ("one_scale", "per_airport"):
            sh = [rec["arms"][s][nm][kind]["board_gain_s"] for s in E.SHAM_ARMS]
            rec.setdefault("sham_board_gain_s", {})[f"{nm}|{kind}"] = {
                "mean": float(np.mean(sh)), "max": float(np.max(sh))}
    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    for nm in ("stand|lt50", "stand|ge500", "op|ge500", "joint_nine"):
        for kind in ("one_scale", "per_airport"):
            t = rec["arms"]["true"][nm][kind]["board_gain_s"]
            sh = [rec["arms"][s][nm][kind]["board_gain_s"] for s in E.SHAM_ARMS]
            print(f"true {nm:14s} {kind:12s} board {t:+.4f} s   sham mean {np.mean(sh):+.4f} "
                  f"max {np.max(sh):+.4f}", flush=True)
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
