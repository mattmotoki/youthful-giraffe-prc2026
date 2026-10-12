"""Which regimes the window block should be served on.

`where.py` decomposes the `depx` change by regime and finds the same shape on
all three folds: `nm_bulk` gains (+0.383 / +0.237 / +0.066 s of fold RMSE) and
the aggregate loss on december is `nm_late` (-0.227) plus a handful of tail
rows -- december's most-moved 1 % of rows carries 366 % of that fold's total
squared-error change, so the sign of the fold is set by a few rows and not by
the bulk.

`stack-5 keep-noac` already serves one regime from the anchor for exactly this
reason.  This extends the same idea: serve the window block only where it
transfers, and stack-5's own value everywhere else.  Every variant below is
arithmetic on the saved rotation vectors -- no model is refitted, so nothing
here can be a different fit dressed as a restriction.

**Stated plainly: the regime set is chosen after seeing the regime table, which
is a selection on the same three folds that carry the scored row.**  What bounds
it is that the chosen slice passes the three-fold gate INSIDE the slice on each
fold independently, which the aggregate does not, and that `nm_bulk` is where
`docs/STRUCTURAL-STEP.md` measured the transient in the first place (SS 8a-8c are
all `nm_bulk` statistics).  It is a candidate, not a discovery.

    python restrict.py [variant]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel as P  # noqa: E402
import where as W  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

VAR = sys.argv[1] if len(sys.argv) > 1 else "depx"
TAG = sys.argv[2] if len(sys.argv) > 2 else "w600"
W.TAG = TAG
SETS = {
    "all-but-noac": ("nm_bulk", "nm_late", "nonm"),
    "bulk+late": ("nm_bulk", "nm_late"),
    "bulk+nonm": ("nm_bulk", "nonm"),
    "bulk": ("nm_bulk",),
}


def main() -> None:
    data = {f: W.load(f) for f in P.FOLDS}
    rows = []
    for name, regs in SETS.items():
        per, boot, boot_nn = {}, {}, {}
        for f, d in data.items():
            y = d["y_true"].to_numpy("float64")
            a = d["y_pred_stack5"].to_numpy("float64")
            b = d[f"y_pred_{VAR}"].to_numpy("float64")
            ids = d["MVT_ID_mvt"].to_numpy("float64")
            p = np.where(np.isin(d["regime"].to_numpy(), regs), b, a)
            per[f] = D.rmse_both(y, p, ids, f)["rmse_clean"]
            cl = P.clusters(d)
            # Sol round 1, Standards 2: the point estimate above is clean, so the
            # interval is bootstrapped on the same rows; `no_noac` drops the rows
            # the block never touches.  On january the two excluded rows are worth
            # 0.16 s of the delta on their own.
            ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
            nn = ok & (d["regime"].to_numpy() != "noac")
            boot[f] = P.boot_delta(y[ok], p[ok], a[ok], pd.factorize(cl[ok])[0])
            boot_nn[f] = P.boot_delta(y[nn], p[nn], a[nn], pd.factorize(cl[nn])[0])
        base = {f: D.rmse_both(data[f]["y_true"], data[f]["y_pred_stack5"],
                               data[f]["MVT_ID_mvt"], f)["rmse_clean"] for f in P.FOLDS}
        g = {f: base[f] - per[f] for f in P.FOLDS}
        rows.append(dict(variant=name, regimes="+".join(regs),
                         **{f: round(per[f], 4) for f in P.FOLDS},
                         **{f"gain_{f}": round(g[f], 4) for f in P.FOLDS},
                         clean_board=round(P.board(per), 4),
                         gain_board=round(P.board(base) - P.board(per), 4),
                         three_folds=bool(all(v > 0 for v in g.values())),
                         **{f"ci_{f}": f"[{boot[f][1]:+.3f}, {boot[f][2]:+.3f}]"
                            for f in P.FOLDS},
                         **{f"ci_no_noac_{f}": f"[{boot_nn[f][1]:+.3f}, {boot_nn[f][2]:+.3f}]"
                            for f in P.FOLDS}))
    rows.append(dict(variant="stack-5", regimes="-",
                     **{f: round(base[f], 4) for f in P.FOLDS},
                     **{f"gain_{f}": 0.0 for f in P.FOLDS},
                     clean_board=round(P.board(base), 4), gain_board=0.0,
                     three_folds=False, **{f"ci_{f}": "-" for f in P.FOLDS},
                     **{f"ci_no_noac_{f}": "-" for f in P.FOLDS}))
    T = pd.DataFrame(rows)
    T.to_csv(P.EXP / "analysis" / f"restrict_{VAR}_{TAG}.csv", index=False)
    print(T[["variant", "regimes"] + list(P.FOLDS)
            + [f"gain_{f}" for f in P.FOLDS] + ["clean_board", "gain_board", "three_folds"]]
          .to_string(index=False))
    print()
    for r in rows[:-1]:
        print(f"  {r['variant']:14s} " + "  ".join(f"{f} {r[f'gain_{f}']:+.3f} {r[f'ci_{f}']}"
                                                   for f in P.FOLDS))
    print("\n  the same, with the noac rows dropped (the block never touches them)")
    for r in rows[:-1]:
        print(f"  {r['variant']:14s} " + "  ".join(f"{f} {r[f'ci_no_noac_{f}']}"
                                                   for f in P.FOLDS))
    (P.EXP / "analysis" / f"restrict_{VAR}_{TAG}.json").write_text(
        json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
