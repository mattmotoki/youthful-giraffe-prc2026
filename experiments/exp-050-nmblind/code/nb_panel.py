"""exp-050-nmblind: the fold side of the current v22 head, frozen for this experiment.

The v22 upload (`docs/SUBMISSIONS.md` row 22) is exp-046's v11, base set v25.  Its fold
side is exp-046's own construction, reproduced here by calling exp-046's code and not by
re-deriving it (`code/onemonthbag.py`, stack `seasonal`):

    blk      = chainfold.build_month(month, prefix, serving rule)   mode-3 pins asserted
    pan_ship = chainfold.panel_for(blk, rung511.THR, GAP_CUT_S)      pbase = v12's fold side
    p_v22    = pan_ship.pbase + d_seasonal                           d from the arm npz

`d_seasonal` is `work/d-seasonal-bag10-255-free.npz` of exp-046, sha-checked against the
value base set v25 declares.  The reproduction gate is exp-046's own recorded price of that
vector over v10's head (`analysis/seasonal-bag10.json`, board all-row gain
0.3980055221207408), recomputed here deterministically.  A miss is fatal.

Output, per fold: `work/panel-v22-<fold>.parquet` (git-ignored) and
`analysis/panel.json` with row counts, the populations, and the vector hashes.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-050-nmblind/code/nb_panel.py
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
EXP = HERE.parent
ROOT = EXP.parents[1]
E46 = ROOT / "experiments/exp-046-surfacefit"
sys.path.insert(0, str(E46 / "code"))
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import guard as G                                                     # noqa: E402
import rung511 as R                                                   # noqa: E402

ARM_NPZ = E46 / "work/d-seasonal-bag10-255-free.npz"
ARM_SHA = "3edfa35fee55e1d03ec2df72192eeb596732e62f8164b162e9ef42597ecea6f5"
HEAD_V10_NPZ = E46 / "work/d-threemonth-bag10-255-free.npz"
RECORDED_GAIN_OVER_V10 = 0.3980055221207408     # exp-046 analysis/seasonal-bag10.json
GATE = 1e-9
KEEP = ["MVT_ID_mvt", "airport", "t", "day", "regime", "nm_missing", "noac", "clean",
        "y_true", "pbase", "usable", "obs_has_clock", "reach_class"]
OUT = EXP / "analysis/panel.json"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_vec(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes())
    return h.hexdigest()


def panel_path(fold: str) -> Path:
    return EXP / f"work/panel-v22-{fold}.parquet"


def load(fold: str) -> pd.DataFrame:
    """The frozen fold-side v22 panel for one fold, as written by `main()`."""
    rec = json.loads(OUT.read_text())
    p = panel_path(fold)
    got = sha_file(p)
    if got != rec["folds"][fold]["parquet_sha256"]:
        raise SystemExit(f"{p.name} is {got[:12]}, not the frozen {rec['folds'][fold]['parquet_sha256'][:12]}")
    return pd.read_parquet(p)


def main() -> None:
    t0 = time.time()
    got = sha_file(ARM_NPZ)
    if got != ARM_SHA:
        raise SystemExit(f"the v22 arm npz is {got[:16]}, not base set v25's {ARM_SHA[:16]}")
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    print(f"panel built ({time.time() - t0:.0f}s)", flush=True)

    z = np.load(ARM_NPZ)
    d22 = {f: z[f].astype("float64") for f in A.FOLDS}
    zh = np.load(HEAD_V10_NPZ)
    d10 = {f: zh[f].astype("float64") for f in A.FOLDS}

    # the reproduction gate: exp-046's recorded board all-row price over v10's head
    rm_base, rm_arm = {}, {}
    for f in A.FOLDS:
        y = pan[f].y_true.to_numpy("float64")
        pb = pan[f].pbase.to_numpy("float64")
        rm_base[f] = float(np.sqrt(np.mean((y - pb - d10[f]) ** 2)))
        rm_arm[f] = float(np.sqrt(np.mean((y - pb - d22[f]) ** 2)))
    gain = BA.board(rm_base) - BA.board(rm_arm)
    if abs(gain - RECORDED_GAIN_OVER_V10) > GATE:
        raise SystemExit(f"the rebuilt v22 fold side prices {gain!r} over v10's head, not the "
                         f"recorded {RECORDED_GAIN_OVER_V10!r}")
    print(f"v22 fold side reproduces exp-046's price over v10's head: {gain:.12f}", flush=True)

    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    rec = {"what": __doc__.strip().splitlines()[0],
           "arm_npz": str(ARM_NPZ.relative_to(ROOT)), "arm_npz_sha256": ARM_SHA,
           "head_v10_npz": str(HEAD_V10_NPZ.relative_to(ROOT)),
           "head_v10_npz_sha256": sha_file(HEAD_V10_NPZ),
           "serving_rule": rule, "thr": R.THR, "gap_cut_s": CF.GAP_CUT_S,
           "reproduction": {"board_all_row_gain_over_v10_head": gain,
                            "recorded": RECORDED_GAIN_OVER_V10, "tolerance": GATE},
           "board_rmse_v22_all_row": BA.board(rm_arm),
           "folds": {}}
    for f in A.FOLDS:
        p = pan[f]
        out = p[KEEP].copy()
        out["d_v22"] = d22[f]
        out["p_v22"] = out.pbase.to_numpy("float64") + d22[f]
        held = (p.usable.to_numpy(bool)
                & G.mask(p.pbase.to_numpy("float64"), p.obs_taxi, cuts[f]))
        out["guard_held"] = held
        if np.any(d22[f][~p.usable.to_numpy(bool)] != 0):
            raise SystemExit(f"{f}: the v22 arm moves a row outside its class")
        if np.any(d22[f][held] != 0):
            raise SystemExit(f"{f}: the v22 arm moves a guard-held row")
        out["fold"] = f
        path = panel_path(f)
        out.to_parquet(path, index=False)
        y = out.y_true.to_numpy("float64")
        pv = out.p_v22.to_numpy("float64")
        reg = out.regime.astype(str)
        rec["folds"][f] = {
            "month": A.FOLD_MONTH[f], "rows": int(len(out)),
            "regime_rows": {k: int(v) for k, v in reg.value_counts().items()},
            "served_class_rows": int(p.usable.sum()), "guard_held_rows": int(held.sum()),
            "guard_cut_s": cuts[f], "rows_moved_by_v22_arm": int((d22[f] != 0).sum()),
            "rmse_v22_all": float(np.sqrt(np.mean((y - pv) ** 2))),
            "rmse_v22_le3600": float(np.sqrt(np.mean(((y - pv) ** 2)[y <= 3600]))),
            "rmse_v22_gt3600": float(np.sqrt(np.mean(((y - pv) ** 2)[y > 3600]))),
            "rmse_v22_nonm": float(np.sqrt(np.mean(((y - pv) ** 2)[reg.eq("nonm").to_numpy()]))),
            "vector_sha256_ids_then_p_v22": sha_vec(out.MVT_ID_mvt, pv),
            "parquet": str(path.relative_to(ROOT)), "parquet_sha256": sha_file(path)}
        print(f, json.dumps(rec["folds"][f], default=float), flush=True)
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print("written", OUT.relative_to(ROOT), f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
