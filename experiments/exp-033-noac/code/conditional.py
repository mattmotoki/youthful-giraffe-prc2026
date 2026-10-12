"""Step 2: condition the 2025 labels on the match.

The label is the record's own: a row is a PLACEHOLDER when |mvt_sched - y| <
`prc.tail_model.PLACEHOLDER_TOL` (300 s).  That definition reproduces the
`P_ph_2025` column of `experiments/exp-026-rebuild/analysis/noac_cells.csv` to
three decimals on every band, so it is the one the record means.

Two estimators, because "the matched sub-population" has two honest readings:

  set   the DISTINCT 2025 rows that are some 2026 row's k nearest neighbours,
        each counted once -- the rate of the sub-population itself;
  qw    each 2026 row's own neighbourhood rate, averaged over the 2026 rows of
        the band -- what a k-NN head would actually predict.

Intervals are a calendar-day-clustered bootstrap over 2025 days: the noac rows
are one airport's feed and are day-concentrated, so an unclustered interval
overstates significance (exp-022's gate, and exp-007 review-2 before it).
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
from prc.tail_model import PLACEHOLDER_TOL  # noqa: E402

KS = (1, 3, 5, 10)
NBOOT = 2000


def day_boot(ph: np.ndarray, day: np.ndarray, w: np.ndarray | None = None,
             nboot: int = NBOOT, seed: int = 7) -> tuple[float, float]:
    """95 % interval for a weighted mean of `ph`, resampling calendar days."""
    if not len(ph):
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    days, inv = np.unique(day, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(days))]
    w = np.ones(len(ph)) if w is None else w
    out = np.empty(nboot)
    for i in range(nboot):
        pick = rng.integers(0, len(days), len(days))
        sel = np.concatenate([idx[j] for j in pick])
        out[i] = np.average(ph[sel], weights=w[sel]) if w[sel].sum() > 0 else np.nan
    return tuple(np.nanpercentile(out, [2.5, 97.5]))


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    a["ph"] = (np.abs(a.mvt_sched - a.y_true) < PLACEHOLDER_TOL).astype(float)
    a.to_parquet(EXP / "work" / "noac_2025_lab.parquet")
    day = a.day.to_numpy()
    ph = a.ph.to_numpy()

    for space in ("rowlevel", "full"):
        d_ba = np.load(EXP / "work" / f"d_ba_{space}.npy")
        order = np.argsort(d_ba, axis=1)
        rows = []
        for k in KS:
            nb = order[:, :k]                              # (388, k)
            mult = np.bincount(nb.ravel(), minlength=len(a)).astype(float)
            for lab in LAB:
                q = (b.band == lab).to_numpy()
                if not q.sum():
                    continue
                # set estimator: distinct matched rows of any band, weight 1
                sel = np.unique(nb[q].ravel())
                m_set = float(ph[sel].mean())
                lo_s, hi_s = day_boot(ph[sel], day[sel])
                # query-weighted: each 2026 row's own neighbourhood rate
                per_q = ph[nb[q]].mean(1)
                # its interval clusters on the 2025 days that supply the weight
                mq = np.bincount(nb[q].ravel(), minlength=len(a)).astype(float)
                use = mq > 0
                m_qw = float(np.average(ph[use], weights=mq[use]))
                lo_q, hi_q = day_boot(ph[use], day[use], mq[use])
                # the whole 2025 set, same band
                whole = (a.band == lab).to_numpy()
                m_all = float(ph[whole].mean())
                lo_a, hi_a = day_boot(ph[whole], day[whole])
                rows.append({
                    "space": space, "k": k, "band": lab, "n2026": int(q.sum()),
                    "n_matched_distinct": int(len(sel)),
                    "P_ph_matched_set": m_set, "set_lo": lo_s, "set_hi": hi_s,
                    "P_ph_matched_qw": m_qw, "qw_lo": lo_q, "qw_hi": hi_q,
                    "n_2025_band": int(whole.sum()),
                    "P_ph_all_band": m_all, "all_lo": lo_a, "all_hi": hi_a,
                    "diff_qw_minus_all": m_qw - m_all,
                    "matched_band_share": float((a.band.to_numpy()[nb[q].ravel()] == lab).mean()),
                    "per_query_mean": float(per_q.mean()),
                })
        t = pd.DataFrame(rows)
        t.to_csv(EXP / "analysis" / f"conditional_{space}.csv", index=False)
        show = t[t.k == 5]
        print(f"\n=== {space}, k=5 ===")
        print(show[["band", "n2026", "n_matched_distinct", "P_ph_matched_set", "set_lo", "set_hi",
                    "P_ph_matched_qw", "qw_lo", "qw_hi", "n_2025_band", "P_ph_all_band",
                    "all_lo", "all_hi", "diff_qw_minus_all"]].round(3).to_string(index=False))
        print("k stability, diff (qw - all) per band:")
        print(t.pivot_table(index="band", columns="k", values="diff_qw_minus_all")
              .reindex(LAB).round(3).to_string())


if __name__ == "__main__":
    main()
