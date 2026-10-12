"""Step 3d: where the base sits, 2025 against 2026, and the honest band selection.

Two things decide whether the fold gain of §3c transfers.

1. **The base's own odds.**  A correction that lowers the placeholder weight
   pays only where the base is above the matched rate.  stack-5's OUT-OF-FOLD
   prediction on the 2025 noac rows and its RANKING prediction on the 388 are
   two different objects; if they sit at different odds, a gain measured on the
   first cannot be claimed for the second.  This is the exact shape of the
   failure BOARD READINGS 4 and 5 recorded.
2. **Selection.**  §3c chose the two bands after seeing the every-band result.
   Here the bands are chosen INSIDE each fold, on its training months only, so
   the choice is out of fold and the gain is not a selection artefact.
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
from why import implied_p  # noqa: E402

OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof" / "stack5-{fold}.parquet"


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)
    h_all, h_mat = fit_head(a), fit_head(a[a._m])
    v = pd.read_parquet(EXP / "work" / "fold_arms.parquet")

    # --- 1. the base's odds, 2025 out-of-fold against 2026 ranking ----------
    v["p_base"] = implied_p(v.y_pred.to_numpy(float), v.mvt_sched.to_numpy(float),
                            h_all, v.band.to_numpy())
    rows = []
    for lab in LAB:
        s25, s26 = v[v.band == lab], b[b.band == lab]
        if not len(s26):
            continue
        rows.append({"band": lab, "n_2025_fold": len(s25), "n2026": len(s26),
                     "p_base_2025_median": float(s25.p_base.median()) if len(s25) else np.nan,
                     "p_v7_2026_median": float(s26.p_v7.median()),
                     "gap_2025_minus_2026": (float(s25.p_base.median()) - float(s26.p_v7.median()))
                     if len(s25) else np.nan,
                     "p_all": h_all[lab][0], "p_matched": h_mat[lab][0]})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "base_odds.csv", index=False)
    print("=== the odds the base already sits at: stack-5 OOF on 2025 vs v7 on the 388 ===")
    print(t.round(3).to_string(index=False))

    # --- 2. band choice made inside the fold -------------------------------
    print("\n=== band set chosen out of fold (on the training months only) ===")
    chosen, arm = {}, v.y_pred.to_numpy(float).copy()
    for fold in ("july_loo", "december", "january"):
        tr = a[a.month.isin(D.FOLDS[fold].train_months)].copy()
        oof = pd.concat([pd.read_parquet(str(OOF).format(fold=f))[["MVT_ID_mvt", "y_pred"]]
                         .assign(_f=f) for f in ("july_loo", "december", "january")])
        # the training months a fold can see the base on: the OTHER folds' months
        seen = tr.merge(oof, on="MVT_ID_mvt", how="inner")
        ha, hm = fit_head(tr), fit_head(tr[tr._m])
        seen["corr"] = apply_head(hm, seen) - apply_head(ha, seen)
        pick = []
        for lab in LAB:
            s = seen[seen.band == lab]
            if len(s) < 10:
                continue
            g = rmse(s.y_true, s.y_pred) - rmse(s.y_true, s.y_pred + s["corr"])
            if g > 0:
                pick.append(lab)
        chosen[fold] = pick
        m = (v.fold == fold).to_numpy() & v.band.isin(pick).to_numpy()
        arm[m] = v.delta.to_numpy(float)[m]
        print(f"  {fold}: base rows seen {len(seen)}, bands chosen {pick}")
    for scope in ("matched", "all"):
        s = np.flatnonzero(v._m.to_numpy() if scope == "matched" else np.ones(len(v), bool))
        lo, hi = paired_rmse_diff(v.y_true.to_numpy()[s], v.y_pred.to_numpy()[s],
                                  arm[s], v.day.to_numpy()[s])
        g = rmse(v.y_true.to_numpy()[s], v.y_pred.to_numpy()[s]) - rmse(v.y_true.to_numpy()[s], arm[s])
        print(f"  out-of-fold band choice, scope={scope:7s}: gain {g:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]")

    # --- 3. the shift the arms make on the 388 ------------------------------
    b["corr_all"] = apply_head(h_mat, b) - apply_head(h_all, b)
    print("\n=== shift on the 388, against v7 ===")
    out = []
    for bands, tag in ((["1-2h"], "1-2h only"), (["1-2h", "2-4h"], "1-2h + 2-4h"),
                       (LAB, "every band")):
        c = np.where(b.band.isin(bands), b.corr_all, 0.0)
        out.append({"arm": tag, "n_moved": int((c != 0).sum()),
                    "shift_rms_388": float(np.sqrt(np.mean(c ** 2))),
                    "shift_mean": float(c.mean()),
                    "budget_500": bool(np.sqrt(np.mean(c ** 2)) <= 500)})
        print(f"  {tag:14s} moves {out[-1]['n_moved']:3d} rows, shift RMS "
              f"{out[-1]['shift_rms_388']:6.1f} s, mean {out[-1]['shift_mean']:+7.1f} s, "
              f"budget {'OK' if out[-1]['budget_500'] else 'BUST'}")
    pd.DataFrame(out).to_csv(EXP / "analysis" / "shift_388.csv", index=False)
    b.to_parquet(EXP / "work" / "noac_2026_d.parquet")


if __name__ == "__main__":
    main()
