"""exp-051, Sol round 1 answer B4 for I1.1: COMMON.md's four populations for the rescore.  ZERO FIT.

I1.1 made no new vector.  It rescored exp-046's four saved increments (recorded rule's N2 and F2 on
december and january) over member 1, first against y - pbase - d_m1 and then against
y - base6 - d_m1.  This script gives, per fold and per population (all rows, target at or below
3600 s, above 3600 s, no-NM rows as populations.py builds them):

  * the RMSE of each predictor the rescore compares: pbase + d and base6 + d for d = d_m1, N2, F2;
  * each increment's G (s2 per fold row) split by population, under both scorings;
  * R = G(F2) / G(N2) under both scorings on the rows at or below 3600 s beside all rows (points,
    record only; they gate nothing).

Every all-row G must equal rescore-base6.json's to 1e-9 before anything is written.

    .venv/bin/python experiments/exp-051-transferadv/code/populations11.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as C
from common import NO, OM
import populations as PO
import redundfit as RF                                               # noqa: E402

OUT = C.ANALYSIS / "populations-i1-1.json"
OOF6 = C.ROOT / "experiments/exp-047-queue/oof"


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT], overwrite=a.overwrite)
    pan, _, _, _ = C.build_panels()
    d_m1 = C.load_vec(OM.CACHED_M1)
    js = json.loads((C.ANALYSIS / "rescore-base6.json").read_text())["by_fold"]
    rec = {"what": __doc__.strip().splitlines()[0], "cut_s": PO.CUT, "folds": {}}
    worst = 0.0
    for fold in RF.CODE:
        p = pan[fold]
        ids = p.MVT_ID_mvt.to_numpy("float64")
        y, pb = p.y_true.to_numpy("float64"), p.pbase.to_numpy("float64")
        b6 = pd.read_parquet(OOF6 / f"base6--{fold}.parquet").set_index("MVT_ID_mvt")
        p6 = b6.y_pred.reindex(ids).to_numpy("float64")
        if not np.isfinite(p6).all():
            raise SystemExit(f"{fold}: a row lacks base6")
        nonm, info = PO.no_nm_mask(ids, fold)
        M = {"all": np.ones(len(y), bool), "le3600": y <= PO.CUT, "gt3600": y > PO.CUT, "no_nm": nonm}
        d = {"m1": d_m1[fold]}
        for arm in ("N2", "F2"):
            d[arm] = np.load(C.E46 / f"work/d-onemonth-{RF.tag(fold, arm)}-255-free.npz")[fold].astype("float64")
        base = {"pbase": pb, "base6": p6}
        fr = {**info, "panel_rows": int(len(y)), "rows": {k: int(m.sum()) for k, m in M.items()},
              "rmse_s": {}, "G_s2_per_fold_row": {}, "R_point_record_only": {}}
        for bn, bv in base.items():
            for dn, dv in d.items():
                e = y - bv - dv
                fr["rmse_s"][f"{bn}+{dn}"] = {k: float(np.sqrt(np.mean(e[m] ** 2))) for k, m in M.items()}
        for bn, bv in base.items():
            # rescore.py's own residuals: rbase - d_m1 (recorded rule) and y - base6 - d_m1
            r = (p.rbase.to_numpy("float64") if bn == "pbase" else y - bv) - d["m1"]
            G = {}
            for arm in ("N2", "F2"):
                inc = d[arm] - d["m1"]
                v = 2 * r * inc - inc * inc
                G[arm] = {k: float(v[m].sum() / len(v)) for k, m in M.items()}
                ref = js[fold]["arms"][arm]["G_pbase_s2" if bn == "pbase" else "G_base6_s2"]
                worst = max(worst, abs(float(v.mean()) - ref))
                if abs(float(v.mean()) - ref) > 1e-9:
                    raise SystemExit(f"{fold} {arm} {bn}: G {float(v.mean())!r} is not the JSON's {ref!r}")
            fr["G_s2_per_fold_row"][bn] = G
            fr["R_point_record_only"][bn] = {k: G["F2"][k] / G["N2"][k] for k in ("all", "le3600")}
        rec["folds"][fold] = fr
        print(f"{fold}: rows {fr['rows']}")
        for k, v in fr["rmse_s"].items():
            print(f"   {k:10} " + "  ".join(f"{q} {v[q]:9.3f}" for q in PO.POPS))
        for bn, G in fr["G_s2_per_fold_row"].items():
            for arm, g in G.items():
                print(f"   G {bn:5} {arm}: " + "  ".join(f"{q} {g[q]:+8.2f}" for q in PO.POPS))
            print(f"   R {bn:5}: all {fr['R_point_record_only'][bn]['all']:.4f}  "
                  f"le3600 {fr['R_point_record_only'][bn]['le3600']:.4f}")
    rec["max_abs_diff_against_json"] = worst
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec, overwrite=a.overwrite)
    print(f"max |diff| against rescore-base6.json {worst:.2e}; written to {OUT.relative_to(C.ROOT)} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
