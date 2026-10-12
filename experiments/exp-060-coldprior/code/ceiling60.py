"""exp-060-coldprior: the exact in-sample ceiling of a per-stratum cold prior (post-hoc, descriptive).

NOT registered in SPEC.  It runs after the verdict, the gate and the bound are frozen, it selects
nothing, and no number here enters a reading, a gate, a pick or a bound.  It exists to put an exact
number on the one route section 12 of RESULTS still names: restrict the cold prior to the sparsest
keys instead of weighting it smoothly.

For a stratum of rows, the largest squared-error reduction any single scale of the prior can give,
with the scale chosen on V itself, is `(sum e q)^2 / sum q^2`, where `e` is the `zero` arm's V
residual and `q` the unit cold prior, both with each airport's mean removed.  That is an upper
bound on what a hard restriction to that stratum can win, before any out-of-sample loss.  The job
converts it to board seconds through the `zero` arm's own fold RMSE.

    .venv/bin/python experiments/exp-060-coldprior/code/ceiling60.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def board(rmse: dict) -> float:
    return float(np.sqrt(sum(E.BOARD[r] * rmse[r] ** 2 for r in E.BOARD)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / "ceiling60.json"
    NO.refuse_existing([p], a.overwrite)
    t0 = time.time()
    al = E.load_json(E.ANALYSIS / "align60.json")
    oos = E.load_json(E.ANALYSIS / "oos.json")
    rows = {r: oos["tables"][r]["zero|B"]["rows"] for r in E.ORDER}
    z = {r: oos["tables"][r]["zero|B"]["rmse"] for r in E.ORDER}
    rec = {"what": ("exact in-sample ceiling of a hard per-stratum cold prior; post-hoc, "
                    "descriptive, registered nowhere, chooses nothing"),
           "utc": E.utc(), "zero_board_rmse_s": board(z), "arms": {},
           "align60_sha256": E.sha_file(E.ANALYSIS / "align60.json"),
           "definition": "(sum e q)^2 / sum q^2 on the stratum, e and q demeaned by airport"}
    for arm in ("true",) + E.SHAM_ARMS:
        out = {}
        for f in E.PRIOR_FAMS:
            for name, _, _ in ((s[0], s[1], s[2]) for s in
                               [("lt50", 0, 50), ("50_499", 50, 500), ("ge500", 500, 0)]):
                red, mse = {}, {}
                for r in E.ORDER:
                    st = al["rotations"][r]["arms"][arm]["by_family"][f]["by_support"][name]
                    red[r] = st["sum_eq"] ** 2 / st["sum_q2"] if st["sum_q2"] > 0 else 0.0
                    mse[r] = red[r] / rows[r]
                new = {r: float(np.sqrt(max(z[r] ** 2 - mse[r], 0.0))) for r in E.ORDER}
                out[f"{f}|{name}"] = {
                    "sse_reduction_s2": red, "mse_per_row_s2": mse,
                    "board_gain_s": rec["zero_board_rmse_s"] - board(new),
                    "dec_gain_s": z["dec"] - new["dec"]}
        # every family and stratum together, the ceiling of a free scale per (family, stratum)
        tot = {r: sum(v["sse_reduction_s2"][r] for v in out.values()) for r in E.ORDER}
        new = {r: float(np.sqrt(max(z[r] ** 2 - tot[r] / rows[r], 0.0))) for r in E.ORDER}
        out["all_families_and_strata"] = {
            "sse_reduction_s2": tot, "mse_per_row_s2": {r: tot[r] / rows[r] for r in E.ORDER},
            "board_gain_s": rec["zero_board_rmse_s"] - board(new), "dec_gain_s": z["dec"] - new["dec"]}
        rec["arms"][arm] = out
    sh = [rec["arms"][s]["stand|lt50"]["board_gain_s"] for s in E.SHAM_ARMS]
    rec["sham_stand_lt50_board_gain_s"] = {"mean": float(np.mean(sh)), "max": float(np.max(sh))}
    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    t = rec["arms"]["true"]
    for k in ("stand|lt50", "stand|50_499", "stand|ge500", "rwy|ge500", "op|ge500",
              "all_families_and_strata"):
        print(f"true {k:26s} board ceiling {t[k]['board_gain_s']:+.4f} s   dec "
              f"{t[k]['dec_gain_s']:+.4f} s", flush=True)
    print("shams, stand|lt50:", {k: round(v, 4) for k, v in rec["sham_stand_lt50_board_gain_s"].items()})
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
