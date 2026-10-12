"""Step 4: the one table, and the declared multiplicity family.

Band x {n2026, n2025 matched, P(placeholder) matched vs all, fold direction,
shift RMS}.  Every column comes from one canonical source and the source is
named in the column comment, so the record and the CSV cannot drift (Sol round
1, S11).

**The correction family, declared once** (Sol round 1, S7).  The primary
analysis is seven bands at the registered settings (k = 5, the row-level space,
the distinct-row estimand), tested twice: once on the label difference (step 2)
and once on the fold gain (step 3).  That is 7 tests per step, and Holm is
applied within each step.  The k in {1, 3, 5, 10} and space in {row-level, full}
sweeps are SENSITIVITY, reported in `within_band_*.csv` and `robust.py`, and
they are not additional tests of a new hypothesis -- but the record says
explicitly that k and space were looked at before the primary settings were
fixed, so no p-value here should be read as if the settings were pre-registered.
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
from describe import LAB  # noqa: E402
from direction import apply_head, fit_head, matched_mask  # noqa: E402

N_BOARD = 344841
V7_SCORE = 263.4623
SCOPE = "matched (real rows, fixed)"


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values, in the input order."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["_m"] = matched_mask(a, b)
    h_all, h_mat = fit_head(a), fit_head(a[a._m])
    b["corr"] = apply_head(h_mat, b) - apply_head(h_all, b)

    wb = pd.read_csv(EXP / "analysis" / "within_band_rowlevel.csv")     # step 2
    wb = wb[wb.k == 5].set_index("band")
    p2 = pd.read_csv(EXP / "analysis" / "placebo_step2.csv").set_index("band")
    p3 = pd.read_csv(EXP / "analysis" / "placebo.csv")                  # step 3
    p3 = p3[p3.scope == SCOPE].set_index("band")

    rows = []
    for lab in LAB:
        q = b[b.band == lab]
        if not len(q):
            continue
        w, s2, s3 = wb.loc[lab], p2.loc[lab], p3.loc[lab]
        c = np.where(b.band == lab, b["corr"], 0.0)
        rows.append({
            "band": lab, "n2026": len(q), "n2025_band": int(w.n_2025_band),
            "n2025_matched": int(w.n_matched_distinct),
            # step 2, distinct-row estimand: within_band_rowlevel.csv
            "P_ph_matched": w.P_ph_matched, "P_ph_all": w.P_ph_all,
            "P_ph_diff": w["diff"], "diff_lo": w.lo, "diff_hi": w.hi,
            "diff_vs_placebo_p": s2.add_one_p_two_sided,
            # step 3, matched scope at a fixed scoring population: placebo.csv
            "fold_gain_s": s3.gain_real, "placebo_mean_s": s3.placebo_mean,
            "placebo_sd_s": s3.placebo_sd, "gain_vs_placebo_p": s3.add_one_p,
            # the arm's own size
            "shift_rms_388": float(np.sqrt(np.mean(c ** 2))),
            "price_of_moving_board_s": float(
                np.sqrt(V7_SCORE ** 2 + (c ** 2).sum() / N_BOARD) - V7_SCORE),
        })
    t = pd.DataFrame(rows)
    t["diff_p_holm"] = holm(t.diff_vs_placebo_p.to_numpy())
    t["gain_p_holm"] = holm(t.gain_vs_placebo_p.to_numpy())
    # the aggregate row the record shows, so it lives in the file too (Sol R2-7)
    allc = b["corr"].to_numpy(float)
    ag = p3.loc["ALL"]
    t.loc[len(t)] = {
        "band": "ALL", "n2026": int(t.n2026.sum()),
        "n2025_band": int(t.n2025_band.sum()), "n2025_matched": int(a._m.sum()),
        "P_ph_matched": np.nan, "P_ph_all": np.nan, "P_ph_diff": np.nan,
        "diff_lo": np.nan, "diff_hi": np.nan, "diff_vs_placebo_p": np.nan,
        "fold_gain_s": ag.gain_real, "placebo_mean_s": ag.placebo_mean,
        "placebo_sd_s": ag.placebo_sd, "gain_vs_placebo_p": ag.add_one_p,
        "shift_rms_388": float(np.sqrt(np.mean(allc ** 2))),
        "price_of_moving_board_s": float(
            np.sqrt(V7_SCORE ** 2 + (allc ** 2).sum() / N_BOARD) - V7_SCORE),
        "diff_p_holm": np.nan, "gain_p_holm": np.nan}
    t.to_csv(EXP / "analysis" / "band_table.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    allc = b["corr"].to_numpy(float)
    per = t[t.band != "ALL"]
    print("\ntotals: 2026 rows %d, 2025 matched %d of %d"
          % (per.n2026.sum(), per.n2025_matched.sum(), per.n2025_band.sum()))
    print("every-band arm: shift RMS %.1f s, price of moving %.3f s of board"
          % (np.sqrt(np.mean(allc ** 2)),
             np.sqrt(V7_SCORE ** 2 + (allc ** 2).sum() / N_BOARD) - V7_SCORE))
    sig = 3542.0
    print("board arithmetic: 1 s of board = %.0f s of noac RMS at sigma=%.0f"
          % (sig - np.sqrt(max(sig ** 2 - ((V7_SCORE + 1.0) ** 2 - V7_SCORE ** 2)
                               * N_BOARD / 388, 0)), sig))


if __name__ == "__main__":
    main()
