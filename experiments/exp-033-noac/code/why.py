"""Step 3c: why the correction measures nothing, and the honest repairs.

1. The shipped model's IMPLIED MIXTURE COORDINATE on the 388.  The per-band
   mixture is invertible: with `off = p*o_ph + (1-p)*o_np`, a prediction names
   the coordinate it sits at, `p = (o_np - off) / (o_np - o_ph)`.  It is affine
   within a band, so a median summarises it, but it is NOT a probability and NOT
   odds: the shipped model is not a band mixture and leaves [0, 1] on 46 of the
   388 rows (Sol round 1, S10; round 2, R2-6).  Where v7 already sits near the
   matched value rather than the whole-band value, the correction has little to
   add -- which is evidence against the lane's premise, not a refutation of it:
   these are medians of a long-tailed coordinate with no interval attached.
2. The correction restricted to the bands whose label difference is clear of
   zero -- 1-2h and 2-4h on the declared distinct-row estimand (`conditional2.py`).
3. k and space sensitivity of the fold gain, and a shrinkage curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
from describe import LAB  # noqa: E402
from direction import apply_head, fit_head, matched_mask, paired_rmse_diff, rmse  # noqa: E402

V7 = ROOT / "submissions" / "candidates" / "exp-030-eham-out.parquet"
OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof" / "stack5-{fold}.parquet"


def implied_p(pred, ms, h, bands):
    out = np.empty(len(pred))
    for i, lab in enumerate(bands):
        p, oph, onp = h[lab]
        off = ms[i] - pred[i]
        out[i] = (onp - off) / (onp - oph) if abs(onp - oph) > 1e-9 else np.nan
    return out


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)
    h_all, h_mat = fit_head(a), fit_head(a[a._m])

    v7 = pd.read_parquet(V7)
    v7 = v7.rename(columns={v7.columns[0]: "MVT_ID_mvt", v7.columns[1]: "v7"})
    v7["MVT_ID_mvt"] = v7.MVT_ID_mvt.astype("float64")
    b = b.merge(v7, on="MVT_ID_mvt", how="left")
    assert b.v7.notna().all(), "v7 does not cover the 388"
    b["p_v7"] = implied_p(b.v7.to_numpy(float), b.mvt_sched.to_numpy(float),
                          h_all, b.band.to_numpy())
    rows = []
    for lab in LAB:
        q = b[b.band == lab]
        if not len(q):
            continue
        rows.append({"band": lab, "n2026": len(q),
                     "p_v7_median": float(q.p_v7.median()),
                     "p_v7_mean": float(q.p_v7.mean()),
                     "p_band_all": h_all[lab][0], "p_matched": h_mat[lab][0],
                     "v7_mean": float(q.v7.mean()),
                     "mvt_sched_mean": float(q.mvt_sched.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "implied_p.csv", index=False)
    print("=== what odds v7 already sits at, band by band ===")
    print(t.round(3).to_string(index=False))
    b.to_parquet(EXP / "work" / "noac_2026_d.parquet")

    # --- repair 2: the correction on the 1-2h band only ---------------------
    v = pd.read_parquet(EXP / "work" / "fold_arms.parquet")
    for bands, tag in ((["1-2h"], "1-2h only"), (["1-2h", "2-4h"], "1-2h + 2-4h"),
                       (LAB, "every band")):
        m = v.band.isin(bands)
        arm = v.y_pred.to_numpy(float).copy()
        arm[m] = v.delta.to_numpy(float)[m]
        for scope in ("matched", "all"):
            s = np.flatnonzero(v._m.to_numpy() if scope == "matched" else np.ones(len(v), bool))
            lo, hi = paired_rmse_diff(v.y_true.to_numpy()[s], v.y_pred.to_numpy()[s],
                                      arm[s], v.day.to_numpy()[s])
            g = rmse(v.y_true.to_numpy()[s], v.y_pred.to_numpy()[s]) - rmse(v.y_true.to_numpy()[s], arm[s])
            n26 = int(b.band.isin(bands).sum())
            sh = float(np.sqrt(np.mean((arm[s] - v.y_pred.to_numpy()[s]) ** 2)))
            print(f"{tag:14s} scope={scope:7s} n={len(s):4d} gain {g:+7.2f} "
                  f"[{lo:+7.2f}, {hi:+7.2f}]  fold shift RMS {sh:6.1f}  (2026 rows touched {n26})")

    # --- repair 3: a shrinkage curve on the correction ----------------------
    print("\nshrinkage curve (lambda * the per-band correction), scope=matched:")
    s = np.flatnonzero(v._m.to_numpy())
    base = v.y_pred.to_numpy(float)
    corr = v.delta.to_numpy(float) - base
    for lam in (0.25, 0.5, 1.0, 1.5, 2.0):
        arm = base + lam * corr
        lo, hi = paired_rmse_diff(v.y_true.to_numpy()[s], base[s], arm[s], v.day.to_numpy()[s])
        g = rmse(v.y_true.to_numpy()[s], base[s]) - rmse(v.y_true.to_numpy()[s], arm[s])
        print(f"  lambda {lam:4.2f}: gain {g:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]")


if __name__ == "__main__":
    main()
