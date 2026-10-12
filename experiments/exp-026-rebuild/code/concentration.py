"""Where in the fold does blend-5's gain over blend-4 actually sit?

RMSE differences do not decompose, but the sum of squared error does, and that
is what a gain is made of.  For each fold this splits

    MSE(blend-5) - MSE(blend-4)   over all rows

into the contribution of each regime and of each error class, in seconds of
RMSE-equivalent as well as in share.  A gain concentrated on a few hundred
high-error rows is a different object from a gain spread over the bulk: the
first has to reproduce on a handful of ranking rows to transfer at all, and the
ranking file holds 388 `noac` rows against 344,841 total.

Rows the clean convention excludes (`prc.data.UNREACHED_ROLLOVER`) are dropped
first, so this decomposes the CLEAN gain -- the one the board predictor uses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import ladder as L  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402

#: exp-014's ranking regime census: how many rows of each regime the board holds.
RANKING_N = {"nm_bulk": 323468, "nm_late": 16090, "nonm": 4895, "noac": 388}


def split(data: dict, a: dict, b: dict, cache: dict) -> pd.DataFrame:
    """Contribution of each row group to MSE(b) - MSE(a), per fold."""
    rows = []
    for f in F.FOLDS:
        m = data[f]
        keep = ~np.isin(m["MVT_ID_mvt"].to_numpy("float64"), list(D.excluded_ids(f)))
        y = m["y_true"].to_numpy("float64")[keep]
        pa, pb = a[f]["pred"][keep], b[f]["pred"][keep]
        ea, eb = (pa - y) ** 2, (pb - y) ** 2
        n = len(y)
        tot = (eb.sum() - ea.sum()) / n
        groups = {f"regime:{r}": m["regime"].to_numpy()[keep] == r for r in
                  ("nm_bulk", "nm_late", "nonm", "noac")}
        groups["class:y>3600"] = y > 3600.0
        groups["class:y<=3600"] = y <= 3600.0
        for g, sel in groups.items():
            d = (eb[sel].sum() - ea[sel].sum()) / n
            rows.append(dict(fold=f, group=g, n=int(sel.sum()),
                             share_of_rows=sel.mean(), d_mse=d,
                             share_of_gain=(d / tot if tot else np.nan)))
        rows.append(dict(fold=f, group="ALL", n=n, share_of_rows=1.0, d_mse=tot,
                         share_of_gain=1.0))
        rows[-1]["rmse_a"] = float(np.sqrt(ea.mean()))
        rows[-1]["rmse_b"] = float(np.sqrt(eb.mean()))
    return pd.DataFrame(rows)


def _show(T: pd.DataFrame) -> None:
    for f in F.FOLDS:
        s = T[T.fold == f]
        tot = s[s.group == "ALL"].iloc[0]
        print(f"{f:9s} RMSE {tot.rmse_a:8.3f} -> {tot.rmse_b:8.3f}   " + "  ".join(
            f"{r.group.split(':')[1]}={r.share_of_gain*100:+6.1f}%"
            for r in s[s.group.str.startswith('regime')].itertuples()))


def main() -> None:
    cache: dict = {}
    names5 = L.B5
    a = L.run(L.B4, cache)
    b = L.run(names5, cache)
    # the two rungs are fitted on different member sets, so rebuild both frames
    data4 = {f: F.fold_frame(f, L.B4) for f in F.FOLDS}
    data5 = {f: F.fold_frame(f, names5) for f in F.FOLDS}
    for f in F.FOLDS:
        if not np.array_equal(data4[f]["MVT_ID_mvt"].to_numpy(),
                              data5[f]["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: the two member sets do not align row for row")
    parts = []
    for label, names in L.SINGLE + [L.SWAP[-1]]:
        r = L.run(names, cache)
        t = split(data5, a, r, cache)
        t.insert(0, "config", label)
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T.to_csv(F.EXP / "analysis" / "concentration.csv", index=False)
    for label in T["config"].unique():
        print(f"\n########## {label} vs blend-4 ##########")
        _show(T[T.config == label])
    T = T[T.config == L.SWAP[-1][0]]

    print("\n=== share of the CLEAN gain, blend-4 -> blend-5, by row group ===")
    for f in F.FOLDS:
        s = T[T.fold == f]
        tot = s[s.group == "ALL"].iloc[0]
        print(f"\n{f}: RMSE {tot.rmse_a:.3f} -> {tot.rmse_b:.3f}  "
              f"(dMSE {tot.d_mse:+.1f} over {tot.n} rows)")
        for r in s[s.group != "ALL"].itertuples():
            print(f"   {r.group:18s} n={r.n:7d} ({r.share_of_rows*100:6.3f} % of rows)  "
                  f"dMSE {r.d_mse:+9.2f}  {r.share_of_gain*100:+7.1f} % of the gain")

    print("\n=== the same groups in the ranking file (exp-014 census) ===")
    for r, n in RANKING_N.items():
        print(f"   regime:{r:9s} {n:7d} rows ({n/344841*100:6.3f} %)")
    (F.EXP / "analysis" / "concentration.json").write_text(
        T.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
