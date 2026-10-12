"""exp-054-arrivalrepr: an UNREGISTERED zero-fit diagnostic, written after iteration 2 was measured.

Question (RESULTS section 6): why does removing the raw key categoricals help the within-month
row-only correction on july_loo and january but hurt it on december?  If a key's mean residual in
the four fitting blocks predicts its mean in the held block, a per-key level learned by a
categorical split transfers; if not, the split fits block noise.

Per fold and key (`apt|stand`, `apt|runway`, `operator`, `apt`), for each day block b: the mean of
r22 = y - p_v22 on the P rows of block b (n_b rows) and on the P rows of the other four blocks
(n_o rows), over keys with n_b >= 30 and n_o >= 100.  Reported per fold: the rows-weighted
correlation and the rows-weighted least-squares slope of the held-block mean on the other-block
mean, pooled over the five blocks; the keys and rows covered; and the between-key sd of the
other-block means.  Arithmetic on the panel's labels and the departure keys; no fit, no choice.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/diag2.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

NO = E.NO
OUT_DEFAULT = "analysis/iter2-diag.json"
MIN_B, MIN_O = 30, 100


def transfer(key: np.ndarray, r: np.ndarray, blk: np.ndarray) -> dict:
    xs, ys, ws = [], [], []
    for b in range(E.K):
        d = pd.DataFrame({"k": key, "r": r, "in_b": blk == b})
        g = d.groupby(["k", "in_b"])["r"].agg(["sum", "count"]).unstack("in_b", fill_value=0)
        nb, no_ = g[("count", True)], g[("count", False)]
        ok = (nb >= MIN_B) & (no_ >= MIN_O)
        xs.append((g[("sum", False)] / no_)[ok].to_numpy())
        ys.append((g[("sum", True)] / nb)[ok].to_numpy())
        ws.append(nb[ok].to_numpy("float64"))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return {"key_blocks": int(len(x)), "rows": int(w.sum()),
            "weighted_corr": float(cxy / np.sqrt(vx * vy)), "weighted_slope": float(cxy / vx),
            "sd_other_block_means_s": float(np.sqrt(vx)), "sd_held_block_means_s": float(np.sqrt(vy))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    rec = {"what": "UNREGISTERED zero-fit diagnostic after iteration 2: does a key's mean r22 in four "
                   "day blocks predict its mean in the fifth?", "min_rows_held_block": MIN_B,
           "min_rows_other_blocks": MIN_O, "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("e54.py", "diag2.py")},
           "folds": {}}
    for f in E.FOLDS:
        pan = E.panel(f, E.PANEL_LABEL_FREE + ["y_true"])
        pop = E.population(pan, f)
        blk = E.blocks(pan.day.to_numpy())
        frame = E.departure_frame(E.fold_month(f))
        dep = frame[frame["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
        ids = pan.MVT_ID_mvt.to_numpy("float64")
        dk = E.departure_keys(dep).set_index("MVT_ID_mvt").reindex(ids)
        Fe = E.row_features(frame).set_index("MVT_ID_mvt").reindex(ids)
        r = pan.y_true.to_numpy("float64") - pan.p_v22.to_numpy("float64")
        m = pop & np.isfinite(r) & (pan.y_true.to_numpy("float64") > 0)
        keys = {"apt_stand": Fe["apt_stand"], "apt_rwy": Fe["apt_rwy"], "op": Fe["op"], "apt": dk["apt"]}
        rec["folds"][f] = {}
        for name, s in keys.items():
            v = s.to_numpy(object)
            ok = m & pd.notna(s).to_numpy()
            rec["folds"][f][name] = transfer(v[ok].astype(str), r[ok], blk[ok])
            t = rec["folds"][f][name]
            print(f"  [{f}] {name:9s} corr {t['weighted_corr']:+.3f} slope {t['weighted_slope']:+.3f} "
                  f"sd(other) {t['sd_other_block_means_s']:.1f} s over {t['key_blocks']:,} key-blocks, "
                  f"{t['rows']:,} rows", flush=True)
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
