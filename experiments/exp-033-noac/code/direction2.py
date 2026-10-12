"""Step 3b: the shipped base, and what the matched-vs-all correction actually moves.

Instrument 2 scores against the head that is on the board: stack-5's own
out-of-fold prediction on the noac rows of july_loo / december / january
(`experiments/exp-026-rebuild/oof/stack5-*.parquet`; v7 serves stack-5's bytes
on all 388 noac rows, `docs/SUBMISSIONS.md` row 7).

Three arms, each fitted on the fold's TRAINING months only:

    band-all      the per-band mixture head, fit set = every noac row
    band-matched  the same head, fit set = the matched rows only
    delta         the shipped prediction PLUS (band-matched - band-all), i.e.
                  the shipped row-level model kept and only the matched-vs-all
                  correction applied.  This is the smallest arm that carries
                  step 2's finding, and the one the 500 s movement budget can
                  hold.
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
from direction import (K, SPACE, apply_head, fit_head, matched_mask,  # noqa: E402
                       paired_rmse_diff, rmse)

OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof" / "stack5-{fold}.parquet"


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)

    # --- diagnostic: what the two fit sets say, band by band, on all twelve --
    h_all, h_mat = fit_head(a), fit_head(a[a._m])
    rows = []
    for lab in LAB:
        pa, opa, ona = h_all[lab]
        pm, opm, onm = h_mat[lab]
        q = b[b.band == lab]
        ms = float(q.mvt_sched.median()) if len(q) else np.nan
        rows.append({"band": lab, "n2026": len(q),
                     "p_all": pa, "p_matched": pm, "dp": pm - pa,
                     "off_all": pa * opa + (1 - pa) * ona,
                     "off_matched": pm * opm + (1 - pm) * onm,
                     "pred_shift_median_row": (pa * opa + (1 - pa) * ona)
                                              - (pm * opm + (1 - pm) * onm),
                     "mvt_sched_med_2026": ms})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "analysis" / "head_bands.csv", index=False)
    print("=== the head's two fit sets, all twelve months ===")
    print(t.round(3).to_string(index=False))
    shift = t.set_index("band").pred_shift_median_row
    n = t.set_index("band").n2026
    print("\nRMS of the per-band correction over the 388: %.1f s"
          % np.sqrt((n * shift ** 2).sum() / n.sum()))

    # --- instrument 2: the three shipped folds ------------------------------
    out, keep = [], []
    for fold in ("july_loo", "december", "january"):
        vm = D.FOLDS[fold].valid_month
        oof = pd.read_parquet(str(OOF).format(fold=fold))[["MVT_ID_mvt", "y_pred"]]
        va = a[a.month == vm].merge(oof, on="MVT_ID_mvt", how="left")
        assert va.y_pred.notna().all(), fold
        fit = a[a.month.isin(D.FOLDS[fold].train_months)]
        ha, hm = fit_head(fit), fit_head(fit[fit._m])
        va["band_all"] = apply_head(ha, va)
        va["band_matched"] = apply_head(hm, va)
        va["delta"] = va.y_pred + (va.band_matched - va.band_all)
        va["fold"] = fold
        keep.append(va)
    v = pd.concat(keep, ignore_index=True)
    v.to_parquet(EXP / "work" / "fold_arms.parquet")
    for scope in ("matched", "all"):
        s = v[v._m] if scope == "matched" else v
        for arm in ("band_all", "band_matched", "delta"):
            lo, hi = paired_rmse_diff(s.y_true, s.y_pred, s[arm], s.day)
            out.append({"instrument": "shipped folds", "scope": scope, "arm": arm,
                        "n": len(s), "rmse_shipped": rmse(s.y_true, s.y_pred),
                        "rmse_arm": rmse(s.y_true, s[arm]),
                        "gain": rmse(s.y_true, s.y_pred) - rmse(s.y_true, s[arm]),
                        "lo": lo, "hi": hi,
                        "shift_rms": float(np.sqrt(np.mean((s[arm] - s.y_pred) ** 2)))})
    o = pd.DataFrame(out)
    o.to_csv(EXP / "analysis" / "shipped_fold_arms.csv", index=False)
    print("\n=== instrument 2: against stack-5's own out-of-fold prediction ===")
    print(o.round(2).to_string(index=False))
    print("\nper fold, arm=delta, scope=matched:")
    for fold in ("july_loo", "december", "january"):
        s = v[(v.fold == fold) & v._m]
        print(f"  {fold}: n={len(s)} shipped {rmse(s.y_true, s.y_pred):.1f} "
              f"delta {rmse(s.y_true, s.delta):.1f} "
              f"gain {rmse(s.y_true, s.y_pred) - rmse(s.y_true, s.delta):+.1f}")


if __name__ == "__main__":
    main()
