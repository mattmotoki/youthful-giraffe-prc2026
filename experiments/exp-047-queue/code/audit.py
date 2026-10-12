"""exp-047-queue: the leakage audit and the anchor diagnostic.

Three assertions and one diagnosis, all written to `analysis/audit.json`.

**A. No column reads the row's own segment.**  Every count in `state()` subtracts the
query row's own event by index.  The audit re-runs the block with the self-exclusion
switched off and reports, per column, how many rows move and by how much: the counts
that can contain the row's own event must move by exactly one on the rows where it
falls in the window, and by nothing anywhere else.  A column that does not move at all
when the row's own event is put back was never self-excluding anything.

**B. No column is a function of the row's own taxi span.**  For a matched row
`MVT_TIME − t_ref` is `exp-046-surfacefit`'s observed taxi.  The audit reports the
Spearman correlation of every block column with that span; the block's claim is not
that the correlation is zero (a busy runway makes both the queue and the taxi large --
that is the signal) but that no column is a *deterministic function* of it, which is
tested by asking whether the span is recoverable from the column: within bins of the
column, the span must retain most of its variance.

**C. No label anywhere.**  `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are read on
ARRIVAL rows only.  The audit asserts the two columns are entirely null on the
departure rows of the frame the block is built from, which `mask_like_ranking`
guarantees and which is therefore a check that the masking ran.

**D. Is the anchor change cosmetic?**  The correlation between the same state read at
the take-off instant and at the off-block instant, per column.  If they agree to 0.95
the anchor is not a new observable and the `Fob` reading is explained.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
import state as S                                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1])


def main(month: str = "2025-07") -> None:
    raw = D.load_month(month, columns=BK.FRAME_COLS)
    masked, _ = D.mask_like_ranking(raw)
    del raw
    dep = masked[masked.PHASE_mvt == "DEP"]
    rep = {"month": month, "rows": int(len(dep))}

    # ---- C. no label on the departure side
    rep["C_label_free"] = {
        "dep_block_time_nonnull": int(dep.BLOCK_TIME_UTC_mvt.notna().sum()),
        "dep_taxitime_nonnull": int(dep.TAXITIME_SEC_mvt.notna().sum()),
        "arr_taxitime_nonnull": int(masked.loc[masked.PHASE_mvt == "ARR",
                                               "TAXITIME_SEC_mvt"].notna().sum())}
    if rep["C_label_free"]["dep_block_time_nonnull"] or \
            rep["C_label_free"]["dep_taxitime_nonnull"]:
        raise SystemExit("audit C: a departure label survived the ranking mask")

    b, _ = BK.build(masked, sources=("F",))

    # ---- A. the self-exclusion moves the columns it should, and only those
    k = BK.keys(masked)
    ev = BK.file_events(masked, k)
    isdep = masked.PHASE_mvt.to_numpy() == "DEP"
    mvt = BK._epoch(masked.MVT_TIME_UTC_mvt)[isdep]
    t_file, _ = BK.anchor(masked, None)
    qa, qr = k["apt"][isdep], k["rwy"][isdep]
    idx = np.arange(int(isdep.sum()))
    a_rep = {}
    for pre, t in (("Fto_", mvt), ("Fob_", t_file)):
        on = S.state(ev, qa, qr, t, idx, pre)
        off = S.state(ev, qa, qr, t, None, pre)
        for c in on.columns:
            d = off[c].to_numpy("float64") - on[c].to_numpy("float64")
            fin = np.isfinite(d)
            a_rep[c] = {"rows_moved": int((fin & (d != 0)).sum()),
                        "max_abs_move": float(np.nanmax(np.abs(d))) if fin.any() else 0.0}
    rep["A_self_exclusion"] = a_rep
    counts = [c for c in a_rep if c.split("_", 1)[1] in S.COLS_COUNT
              and not c.endswith(("_oth",))]
    bad = {c: a_rep[c] for c in counts if a_rep[c]["max_abs_move"] > 1.0}
    if bad:
        raise SystemExit(f"audit A: a count moved by more than one own event: {bad}")
    never = [c for c in counts if a_rep[c]["rows_moved"] == 0]
    rep["A_columns_that_never_contain_the_row"] = never

    # ---- B. the span is not recoverable from any column
    span = mvt - t_file
    cols = [c for c in b.columns if c != "MVT_ID_mvt" and b[c].dtype.kind in "fiu"]
    br = {}
    for c in cols:
        v = b[c].to_numpy("float64")
        rho = spearman(v, span)
        # variance of the span that survives conditioning on 20 quantile bins of `v`
        ok = np.isfinite(v) & np.isfinite(span)
        keep = float("nan")
        if ok.sum() > 1000:
            q = pd.qcut(pd.Series(v[ok]).rank(method="first"), 20, labels=False)
            within = pd.Series(span[ok]).groupby(q).transform(
                lambda s: s - s.mean()).to_numpy()
            keep = float(np.var(within) / np.var(span[ok]))
        br[c] = {"spearman_with_own_span": round(rho, 4),
                 "span_variance_left_within_bins": round(keep, 4)}
    rep["B_span"] = br
    worst = min(br, key=lambda c: br[c]["span_variance_left_within_bins"])
    rep["B_worst_column"] = {worst: br[worst]}
    if br[worst]["span_variance_left_within_bins"] < 0.5:
        raise SystemExit(f"audit B: {worst} recovers half the row's own span")

    # ---- D. is the anchor change cosmetic?
    d_rep = {}
    for c in S.COLS:
        x = b.get(f"Fto_{c}")
        y = b.get(f"Fob_{c}")
        if x is None or y is None:
            continue
        d_rep[c] = {"pearson": round(float(pd.Series(x).corr(pd.Series(y))), 4),
                    "mean_abs_diff": round(float(np.nanmean(np.abs(
                        x.to_numpy("float64") - y.to_numpy("float64")))), 3)}
    rep["D_anchor"] = d_rep
    (EXP / "analysis/audit.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps({"A_never_self_excluding": never,
                      "B_worst": rep["B_worst_column"],
                      "D_median_anchor_correlation": round(float(np.median(
                          [v["pearson"] for v in d_rep.values()])), 4)}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
