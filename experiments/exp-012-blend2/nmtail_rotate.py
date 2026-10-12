"""Honest rotation for a member that is undefined on some rows.

exp-010's NM-tail head predicts only NM-joined departures: its OOF is null on
3,938 of july_loo's 190,713 rows (97.9 % coverage; 99.0 % and 99.1 % on the
winter folds). A member that does not cover a row cannot be given a weight
there, so the blend carries TWO weight vectors -- one fitted on the rows where
every member is defined, one on the rows where only the core is -- and each row
is scored under the vector that applies to it. That is `prc.blend`'s per-regime
machinery with coverage as the regime, which is the honest reading: the
alternative, filling the member's gap with a constant or with another member's
prediction, would invent a prediction the head never made.

The rotation is unchanged: a fold's weights come only from the other two months.

Usage: python nmtail_rotate.py [mode]      mode = nnls (default) | flat
"""
from __future__ import annotations

import json
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import blend as BL  # noqa: E402
import prc.blend as B  # noqa: E402

E8 = ROOT / "experiments/exp-008-diversity"
#: The member set. `NMTAIL_CORE` points at a members JSON to override it, so a
#: new candidate's member set is a config rather than an edit to this file: the
#: blend-4 path has to stay exactly reproducible while blend-5 uses different
#: members, and editing the default in place would have quietly rebuilt blend-4
#: as something else.
CORE = {
    "T-tailv2":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":    "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet",
    "L2-hub":    "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
    "D1-winter": "experiments/exp-012-blend2/published/D1-winter-deice-v1-{fold}.parquet",
    "F1-fresh":  "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet",
}
#: V4 OOF, republished here with the non-NM rows nulled to match both the
#: shipped convention and this member's ranking artifact. `mode` is retained
#: in the signature for the p/phi comparison, which reads exp-010 directly.
NMTAIL = "experiments/exp-012-blend2/published/N1-nmtail-v4-{fold}.parquet"

if os.environ.get("NMTAIL_CORE"):
    CORE = json.loads((Path(__file__).resolve().parent
                       / os.environ["NMTAIL_CORE"]).read_text())["members"]
    print(f"[core] {os.environ['NMTAIL_CORE']}: " + ", ".join(CORE), flush=True)


def load_one(path, fold, name):
    d = pd.read_parquet(ROOT / str(path).format(fold=fold))
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    keep = ["MVT_ID_mvt", "y_true", "y_pred"] + (["d_takeoff_aobt"] if "d_takeoff_aobt" in d else [])
    return d[keep].rename(columns={"y_pred": name})


def fold_data(fold, head_mode="phi"):
    lo, hi = json.loads((E8 / "guard_bounds.json").read_text())[fold]
    m = load_one(CORE["L2-hub"], fold, "L2-hub")
    for n, p in CORE.items():
        if n != "L2-hub":
            m = m.merge(load_one(p, fold, n)[["MVT_ID_mvt", n]], on="MVT_ID_mvt", how="inner")
    h = pd.read_parquet(ROOT / NMTAIL.format(fold=fold))
    h = h[h["fold"] == fold][["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": "N1-nmtail"})
    m = m.merge(h, on="MVT_ID_mvt", how="left")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    prep = lambda v: B.apply_residual_guard(
        np.maximum(np.asarray(v, "float64"), B.FLOOR), a, float(lo), float(hi), B.FLOOR)
    names = list(CORE) + ["N1-nmtail"]
    hv = m["N1-nmtail"].to_numpy("float64")
    ok = ~np.isnan(hv)
    P = np.stack([prep(m[n].to_numpy("float64")) for n in CORE]
                 + [prep(np.where(ok, hv, 0.0))], axis=1)
    return {"P": P, "y": m["y_true"].to_numpy("float64"), "ok": ok, "names": names}


def weights(P, y, mode, k):
    if mode == "flat":
        return np.full(k, 1.0 / k)
    return B.fit_weights(P[:, :k], y, mode)


def main(mode="nnls", head_mode="phi"):
    data = {f: fold_data(f, head_mode) for f in BL.FOLDS}
    k = len(data[BL.FOLDS[0]]["names"])
    print(f"members: {data[BL.FOLDS[0]]['names']}   head mode: {head_mode}   blend mode: {mode}")
    for f in BL.FOLDS:
        d = data[f]
        print(f"  {f:9s} coverage {d['ok'].sum():,}/{len(d['ok']):,} "
              f"({100*d['ok'].mean():.1f} %)")
    per_fold, per_fold_core, per_fold_plain = {}, {}, {}
    for f in BL.FOLDS:
        src = [g for g in BL.FOLDS if g != f]
        # covered rows: all five members; uncovered rows: the four core members
        Pc = np.concatenate([data[g]["P"][data[g]["ok"]] for g in src])
        yc = np.concatenate([data[g]["y"][data[g]["ok"]] for g in src])
        Pu = np.concatenate([data[g]["P"][~data[g]["ok"]][:, :k-1] for g in src])
        yu = np.concatenate([data[g]["y"][~data[g]["ok"]] for g in src])
        w5 = weights(Pc, yc, mode, k)
        w4 = weights(Pu, yu, mode, k - 1)
        d = data[f]
        p = np.empty(len(d["y"]))
        p[d["ok"]] = d["P"][d["ok"]] @ w5
        p[~d["ok"]] = d["P"][~d["ok"]][:, :k-1] @ w4
        p = np.maximum(p, B.FLOOR)
        per_fold[f] = float(np.sqrt(((d["y"] - p) ** 2).mean()))
        # CONTROL: the core alone under the SAME coverage split, so the
        # comparison isolates the member and not the two-vector machinery.
        # Fitting a separate vector on 1-2 % of rows is itself a cost, and
        # charging it to the member would be a rigged test.
        wc5 = weights(Pc[:, :k-1], yc, mode, k - 1)
        wc4 = weights(Pu, yu, mode, k - 1)
        pc = np.empty(len(d["y"]))
        pc[d["ok"]] = d["P"][d["ok"]][:, :k-1] @ wc5
        pc[~d["ok"]] = d["P"][~d["ok"]][:, :k-1] @ wc4
        pc = np.maximum(pc, B.FLOOR)
        per_fold_core[f] = float(np.sqrt(((d["y"] - pc) ** 2).mean()))
        # and the plain single-vector core, for the level
        wg = weights(np.concatenate([data[g]["P"][:, :k-1] for g in src]),
                     np.concatenate([data[g]["y"] for g in src]), mode, k - 1)
        pg = np.maximum(d["P"][:, :k-1] @ wg, B.FLOOR)
        per_fold_plain[f] = float(np.sqrt(((d["y"] - pg) ** 2).mean()))
        print(f"  {f:9s} core4/1vec {per_fold_plain[f]:8.3f}  core4/split {per_fold_core[f]:8.3f}  "
              f"+N1 {per_fold[f]:8.3f}  gain vs split {per_fold_core[f]-per_fold[f]:6.3f}  "
              f"w_cov={np.round(w5,3)}")
    bp = BL.board_estimate(per_fold_plain)
    b0, b1 = BL.board_estimate(per_fold_core), BL.board_estimate(per_fold)
    print(f"\n  board  core4 one vector {bp:.3f}   core4 coverage-split {b0:.3f}   "
          f"+N1 {b1:.3f}")
    print(f"  cost of the split alone {bp-b0:+.3f}   value of the member {b0-b1:+.3f}")
    return bp, b0, b1


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["nnls"]))
