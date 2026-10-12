"""Where the window block's gain sits, and whether december's loss is a shape.

The `both-folds` gate fails on december by 0.04 s, which is smaller than the
construction's own run-to-run spread (0.02 s from feature order alone, RESULTS
§5).  A number that small is not a finding on its own; the question the gate is
really asking is the transfer question exp-026 §5 asked of blend-5: **is the
gain carried by a handful of rows whose sign flips between folds, or is it
broad and same-signed?**

So this reads the saved rotation vectors and decomposes the change, with no
refitting:

  * the required decomposition (`<= 3600`, `> 3600`, no-NM, noac) per fold;
  * the gain by regime and by airport, per fold, with the sign pattern;
  * concentration: what share of the squared-error change the most-moved 1 %
    of rows carries.

    python where.py [variant]      # default depx
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
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

VAR = sys.argv[1] if len(sys.argv) > 1 else "depx"
TAG = sys.argv[2] if len(sys.argv) > 2 else "w600"


def load(fold: str) -> pd.DataFrame:
    o = pd.read_parquet(P.EXP / "oof" / f"exp-028-stack5w-{TAG}-{fold}.parquet")
    p = P.load_fold(fold)[["MVT_ID_mvt", "airport", "regime", "nm_missing", "noac", "t"]]
    m = o.merge(p, on="MVT_ID_mvt", how="inner")
    if len(m) != len(o):
        raise SystemExit(f"{fold}: panel covers {len(m)} of {len(o)} rows")
    return m


def rmse_of(y, p, m=None) -> float:
    e = (np.asarray(p, "float64") - np.asarray(y, "float64")) ** 2
    e = e if m is None else e[m]
    return float(np.sqrt(e.mean())) if len(e) else float("nan")


def main() -> None:
    data = {f: load(f) for f in P.FOLDS}
    dec, by_reg, by_apt, conc = [], [], [], []
    for f, d in data.items():
        y = d["y_true"].to_numpy("float64")
        a = d["y_pred_stack5"].to_numpy("float64")
        b = d[f"y_pred_{VAR}"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        for lab, p in (("stack-5", a), (f"stack-5+{VAR}", b)):
            dec.append(dict(fold=f, variant=lab,
                            rmse=D.rmse_both(y, p, ids, f)["rmse"],
                            clean=D.rmse_both(y, p, ids, f)["rmse_clean"],
                            **{k: v for k, v in P.decomp(y, p, d).items() if k != "rmse"}))
        # the gain per group, as the drop in that group's contribution to the
        # fold's mean squared error, expressed in seconds of the fold's RMSE
        n = len(d)
        r0 = rmse_of(y, a)
        for key, col in (("regime", "regime"), ("airport", "airport")):
            for g, idx in d.groupby(col).groups.items():
                m = np.zeros(n, bool); m[d.index.get_indexer(idx)] = True
                d2 = float(((a[m] - y[m]) ** 2).sum() - ((b[m] - y[m]) ** 2).sum())
                row = dict(fold=f, key=key, group=str(g), n=int(m.sum()),
                           share=float(m.mean()),
                           gain_s=float(r0 - np.sqrt(max((((a - y) ** 2).sum() - d2) / n, 0.0))))
                (by_reg if key == "regime" else by_apt).append(row)
        # concentration of the change
        dd = (a - y) ** 2 - (b - y) ** 2
        o = np.sort(np.abs(b - a))[::-1]
        k = max(int(0.01 * n), 1)
        top = np.abs(b - a) >= o[k - 1]
        conc.append(dict(fold=f, total_sse_drop=float(dd.sum()),
                         top1pct_share=float(dd[top].sum() / dd.sum()) if dd.sum() else np.nan,
                         n_moved_gt_60s=int((np.abs(b - a) > 60).sum()),
                         rms_move=float(np.sqrt(np.mean((b - a) ** 2))),
                         gain_s=float(r0 - rmse_of(y, b))))
    Dc, Rg, Ap, Cn = (pd.DataFrame(x) for x in (dec, by_reg, by_apt, conc))
    for name, T in (("decomp", Dc), ("by_regime", Rg), ("by_airport", Ap),
                    ("concentration", Cn)):
        T.round(4).to_csv(P.EXP / "analysis" / f"where_{VAR}_{TAG}_{name}.csv", index=False)

    print(f"=== required decomposition, {VAR} ===")
    print(Dc.round(2).to_string(index=False))
    print(f"\n=== gain in seconds of fold RMSE, by regime ===")
    print(Rg.pivot(index="group", columns="fold", values="gain_s")[list(P.FOLDS)]
          .round(3).to_string())
    print(f"\n=== by airport ===")
    A = Ap.pivot(index="group", columns="fold", values="gain_s")[list(P.FOLDS)]
    A["folds_positive"] = (A > 0).sum(axis=1)
    print(A.round(3).to_string())
    print(f"\n=== concentration of the change ===")
    print(Cn.round(4).to_string(index=False))
    (P.EXP / "analysis" / f"where_{VAR}_{TAG}.json").write_text(json.dumps(
        {"variant": VAR,
         "airports_positive_on_all_three": int((A["folds_positive"] == 3).sum()),
         "airports": int(len(A)),
         "regime_gain": json.loads(Rg.pivot(index="group", columns="fold",
                                            values="gain_s").round(4).to_json()),
         "concentration": json.loads(Cn.round(4).to_json(orient="records"))}, indent=2))


if __name__ == "__main__":
    main()
