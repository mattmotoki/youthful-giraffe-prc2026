"""Step 2: the 2025 labels conditioned on the match, WITHIN BAND.

Sol round 1, S5: step 2 and step 3 must estimate the same object.  Step 3 fits a
head on a SET of rows, unweighted, so the primary estimand here is the
**distinct-row** rate: the mean of `ph` over the distinct 2025 rows of the band
that are a within-band k-nearest neighbour of one of the 388.  The
multiplicity-weighted rate (each 2026 row's five neighbour slots, so a row that
several 2026 rows match counts several times) is what a k-NN predictor would
average and is reported beside it as sensitivity.  The first version of this
module made the weighted rate primary while `direction.py` used the distinct-row
mask, which mixed the two.

The interval is on the DIFFERENCE and it is paired: one calendar-day resample
per replicate, both means recomputed on it, so the two estimates share their
sampling noise instead of being compared as two independent intervals.  The
noac rows are one airport's feed and are day-concentrated, so an unclustered
interval would overstate significance.

Matching itself is fixed (it is a deterministic nearest-neighbour rule on
label-free features); this interval prices the label sampling, not the match.
`placebo2.py` prices the match.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from describe import LAB  # noqa: E402

KS = (1, 3, 5, 10)
NBOOT = 4000


def paired_diff(ph, day, w, nboot=NBOOT, seed=17):
    """95 % interval for (mean over the matched rows) - (mean over the band)."""
    rng = np.random.default_rng(seed)
    days, inv = np.unique(day, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(days))]
    out = np.full(nboot, np.nan)
    for i in range(nboot):
        sel = np.concatenate([idx[j] for j in rng.integers(0, len(days), len(days))])
        ww = w[sel]
        if ww.sum() <= 0:
            continue
        out[i] = np.average(ph[sel], weights=ww) - ph[sel].mean()
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    ph_all, day_all = a.ph.to_numpy(float), a.day.to_numpy()

    for space in ("rowlevel", "full"):
        d_ba = np.load(EXP / "work" / f"d_ba_{space}.npy")
        rows = []
        for lab in LAB:
            q = np.flatnonzero((b.band == lab).to_numpy())
            p = np.flatnonzero((a.band == lab).to_numpy())
            if not len(q) or not len(p):
                continue
            ordr = np.argsort(d_ba[np.ix_(q, p)], axis=1)
            ph, day = ph_all[p], day_all[p]
            for k in KS:
                nb = ordr[:, :min(k, len(p))]
                mult = np.bincount(nb.ravel(), minlength=len(p)).astype(float)
                hit = (mult > 0).astype(float)          # the distinct-row set
                m_set, m_w, m_all = (float(np.average(ph, weights=hit)),
                                     float(np.average(ph, weights=mult)),
                                     float(ph.mean()))
                lo, hi = paired_diff(ph, day, hit)
                lo_w, hi_w = paired_diff(ph, day, mult)
                rows.append({"space": space, "band": lab, "k": k, "n2026": len(q),
                             "n_2025_band": len(p),
                             "n_matched_distinct": int(hit.sum()),
                             "P_ph_matched": m_set, "P_ph_all": m_all,
                             "diff": m_set - m_all, "lo": lo, "hi": hi,
                             "clear_of_zero": bool(lo > 0 or hi < 0),
                             "P_ph_matched_weighted": m_w,
                             "diff_weighted": m_w - m_all,
                             "lo_weighted": lo_w, "hi_weighted": hi_w,
                             "clear_weighted": bool(lo_w > 0 or hi_w < 0)})
        t = pd.DataFrame(rows)
        t.to_csv(EXP / "analysis" / f"within_band_{space}.csv", index=False)
        print(f"\n=== {space}, k=5, primary estimand = distinct rows ===")
        print(t[t.k == 5][["band", "n2026", "n_2025_band", "n_matched_distinct",
                           "P_ph_matched", "P_ph_all", "diff", "lo", "hi",
                           "clear_of_zero", "diff_weighted", "clear_weighted"]]
              .round(3).to_string(index=False))
        print("diff (distinct rows) by k:")
        print(t.pivot_table(index="band", columns="k", values="diff").reindex(LAB).round(3).to_string())


if __name__ == "__main__":
    main()
