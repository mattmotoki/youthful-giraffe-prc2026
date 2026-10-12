"""exp-052-nmv22: two UNREGISTERED diagnostics of iteration 2's read (arithmetic, no fit).

Written after the SPEC 2.5 readings were computed; they change no reading.  They answer two
"why" questions of RESULTS §7:

1. At k* the real oracle beats all 50 d22 permutations on every fold, the NULL's does not, yet
   corr(d22, r - u_aware) is about zero and the router loses.  Does the oracle's rank come from
   the SIZE of d22 tracking the size of the residual (which a bounded coefficient cannot use)
   rather than from its SIGN (which a router could use)?  Per fold, real and NULL, on P:
   corr(|d22|, |r_a|), the share of rows where sign(d22) = sign(r_a), and corr(d22, r_a) by cell.
2. How much of q_aware*'s gain over v22 is an airport-level constant of the fold month?  For
   scoring block b, each airport's mean residual over P's rows of the four other blocks (no
   shrinkage), added to v22 on P: priced on the board like every predictor, and paired
   against q_aware*.

    .venv/bin/python experiments/exp-052-nmv22/code/rx_diag.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_common as C                                                 # noqa: E402
import rx_price as RP                                                 # noqa: E402

NR, NO = C.NR, C.NO


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = C.EXP / "analysis/iter2-diag.json"
    NO.refuse_existing([OUT], args.overwrite)
    rec = {"what": "UNREGISTERED diagnostics of iteration 2 (RESULTS §7); no reading changes",
           "code_sha256": C.sha_file(Path(__file__)), "runs": {}}
    for nl in (False, True):
        tag = "null" if nl else "real"
        j = json.loads((C.EXP / f"analysis/{C.tagged('iter2', nl)}.json").read_text())
        FD = {f: RP.load(f, False, None, nl) for f in C.FOLDS}
        dfs = {f: FD[f]["df"] for f in C.FOLDS}
        zero = {f: np.zeros(len(dfs[f])) for f in C.FOLDS}
        Ua, Uc = {}, {}
        out = {"folds": {}}
        for f in C.FOLDS:
            df = dfs[f]
            info = j["selected"]["folds"][f]
            rp = C.ROOT / info["rows_file"]
            if C.sha_file(rp) != info["rows_sha256"]:
                raise SystemExit(f"{rp.name} moved since rx_iter2 wrote it")
            rows = pd.read_parquet(rp)
            if not np.array_equal(rows.MVT_ID_mvt.to_numpy("float64"),
                                  df.MVT_ID_mvt.to_numpy("float64")):
                raise SystemExit(f"{f}: the rows file is not the panel's order")
            P = df["pop"].to_numpy(bool)
            r = df.r.to_numpy("float64")
            ua = rows.u_q_aware.to_numpy("float64")
            d = (rows.u_q_blind.to_numpy("float64") - ua)[P]
            ra = (r - ua)[P]
            cell = df.cell.to_numpy(int)[P]
            nz = d != 0
            out["folds"][f] = {
                "corr_d22_r_a": corr(d, ra),
                "corr_abs_d22_abs_r_a": corr(np.abs(d), np.abs(ra)),
                "share_sign_agree": float(np.mean(np.sign(d[nz]) == np.sign(ra[nz]))),
                "corr_d22_r_a_by_cell": {int(k): corr(d[cell == k], ra[cell == k])
                                         for k in range(4)}}
            # airport constant, cross-fitted by date block
            blk = df.block.to_numpy(int)
            apt = df.airport.to_numpy()
            uc = np.zeros(len(df))
            for b in range(C.K):
                dev, sco = P & (blk != b), P & (blk == b)
                means = pd.Series(r[dev]).groupby(apt[dev]).mean()
                uc[sco] = pd.Series(apt[sco]).map(means).fillna(0.0).to_numpy("float64")
            Ua[f], Uc[f] = ua, uc
        out["airport_constant_over_v22"] = RP.board(dfs, zero, Uc)
        out["airport_constant_over_v22"]["december_rmse_gain_s"] = (
            RP.rmse(dfs["december"].r.to_numpy("float64"))
            - RP.rmse(dfs["december"].r.to_numpy("float64") - Uc["december"]))
        out["q_aware_star_over_v22"] = RP.board(dfs, zero, Ua)
        out["q_aware_star_minus_airport_constant"] = RP.board(dfs, Uc, Ua)
        rec["runs"][tag] = out
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(json.dumps(rec, indent=2, default=float))


if __name__ == "__main__":
    main()
