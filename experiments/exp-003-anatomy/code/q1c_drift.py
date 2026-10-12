"""Does the month-to-month drift of the offset matter out of sample?

The `month` row of the decomposition in q1b_decomp.md is uninformative by
construction: the validation month's level is never in the training table, so
the predictor always falls back to the global median and reproduces it exactly.
This script asks the question the right way: fit the airport (and stand) offset
on different windows of training months and see which window predicts the
validation month best.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

FOLDS = {"july": ([f"2025-{m:02d}" for m in range(1, 7)], "2025-07"),
         "december": ([f"2025-{m:02d}" for m in range(1, 12)], "2025-12"),
         "january": ([f"2025-{m:02d}" for m in range(2, 13)], "2025-01")}


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    out = []; W = out.append
    W("# Q1c — does the seasonal drift of the offset matter?\n")

    W("## Range of the trimmed-mean offset across the twelve 2025 months, per airport\n")
    tm = lambda s: float(s[(s >= s.quantile(.1)) & (s <= s.quantile(.9))].mean())
    b = dep[(~dep.is_ranking) & dep.has_nm & dep.TAXITIME_SEC_mvt.le(3600)]
    piv = b.pivot_table(index="apt", columns="month", values="blk_aobt", aggfunc=tm, observed=True)
    t = pd.DataFrame({"min": piv.min(axis=1), "max": piv.max(axis=1),
                      "range": piv.max(axis=1) - piv.min(axis=1),
                      "argmin month": piv.idxmin(axis=1), "argmax month": piv.idxmax(axis=1)})
    W(C.md_table(t.sort_values("range", ascending=False).rename_axis("airport"), floatfmt="{:.0f}"))
    W("")

    W("## Which training window predicts the validation month best?\n")
    W("Predictor: `taxi = (MVT - AOBT_3) - offset`, offset = the median of "
      "`BLOCK - AOBT_3` over the named window, per airport and per stand "
      "(stand falling back to airport, minimum 30 rows). Bulk RMSE.\n")
    rows = []
    for fold, (trm, vam) in FOLDS.items():
        va = dep[dep.month == vam]
        mv = va.has_nm & va.TAXITIME_SEC_mvt.le(3600)
        VA = va[mv]
        y = VA.TAXITIME_SEC_mvt.to_numpy(float)
        nm = VA.nm_taxi.to_numpy(float)
        windows = {"all training months": trm,
                   "last 3 training months": trm[-3:],
                   "last 1 training month": trm[-1:],
                   "first 3 training months": trm[:3]}
        for wname, w in windows.items():
            tr = dep[dep.month.isin(w)]
            mt = tr.has_nm & tr.TAXITIME_SEC_mvt.le(3600)
            TR = tr[mt]
            g0 = TR.blk_aobt.median()
            amed = TR.groupby("apt", observed=True)["blk_aobt"].median()
            smed = TR.groupby("apt_stand", observed=True)["blk_aobt"].agg(["median", "size"])
            smed = smed[smed["size"] >= 30]["median"]
            off_a = VA.apt.astype(object).map(amed).astype(float).fillna(g0).to_numpy()
            off_s = (VA.apt_stand.astype(object).map(smed).astype(float)
                     .fillna(pd.Series(off_a, index=VA.index))).to_numpy()
            rows.append({"fold": fold, "window": wname, "months": len(w),
                         "airport offset": d.rmse(y, nm - off_a),
                         "stand offset": d.rmse(y, nm - off_s)})
    T = pd.DataFrame(rows)
    W(C.md_table(T.set_index("fold"), floatfmt="{:.2f}"))
    W("")
    C.write("q1c_drift.md", "\n".join(out))
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
