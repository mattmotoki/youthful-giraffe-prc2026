"""Why the bulk model beats both of its ingredients: two independent estimates of one number."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d

FOLDS = {"july": ([f"2025-{m:02d}" for m in range(1, 7)], "2025-07"),
         "december": ([f"2025-{m:02d}" for m in range(1, 12)], "2025-12"),
         "january": ([f"2025-{m:02d}" for m in range(2, 13)], "2025-01")}


def group_pred(tr, va, keys, col, fallback_keys, min_n=30):
    """Median of `col` per level of `keys`, fitted on tr, with a fallback chain."""
    pred = pd.Series(np.nan, index=va.index)
    for k in [keys] + fallback_keys:
        g = tr.groupby(k, observed=True)[col].agg(["median", "size"])
        g = g[g["size"] >= min_n]["median"]
        idx = pd.MultiIndex.from_frame(va[k]) if isinstance(k, list) and len(k) > 1 else va[k[0] if isinstance(k, list) else k].astype(object)
        vals = pd.Series(idx.map(g).to_numpy() if hasattr(idx, "map") else np.nan, index=va.index)
        pred = pred.where(pred.notna(), vals)
    return pred.fillna(tr[col].median())


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    dep["srt"] = dep.apt_stand + "|" + dep.RUNWAY_mvt.astype(str) + "|" + dep.AIRCRAFT_TYPE_mvt.astype(str)
    dep["sr"] = dep.apt_stand + "|" + dep.RUNWAY_mvt.astype(str)
    out = []; W = out.append
    W("# Two independent estimates of the same number\n")
    W("`taxi = nm_taxi - offset`. There are two ways to guess it:\n")
    W("- **A (the NM path)**: take the observed `nm_taxi = MVT - AOBT_3` and subtract a "
      "group median of the offset. Its error is the unpredictable part of `BLOCK - AOBT_3`.\n"
      "- **I (the identity path)**: ignore the NM columns and predict the group median taxi "
      "time for this stand, runway and aircraft type. Its error is the unpredictable part of "
      "the taxi time itself.\n")
    W("If the two errors were independent, the inverse-variance blend would score "
      "`1/sqrt(1/sA^2 + 1/sI^2)`. Measured, out of sample, on the bulk of each fold:\n")
    rows = []
    for fold, (trm, vam) in FOLDS.items():
        tr = dep[dep.month.isin(trm)]
        va = dep[dep.month == vam]
        mt = tr.has_nm & tr.TAXITIME_SEC_mvt.le(3600)
        mv = va.has_nm & va.TAXITIME_SEC_mvt.le(3600)
        TR, VA = tr[mt], va[mv]
        y = VA.TAXITIME_SEC_mvt.to_numpy(float)
        # A: offset from the stand x runway median, falling back to stand then airport
        offA = group_pred(TR, VA, "srt", "blk_aobt", ["sr", "apt_stand", "apt"])
        predA = VA.nm_taxi.to_numpy(float) - offA.to_numpy(float)
        # I: median taxi of stand x runway x type, same fallback chain
        predI = group_pred(TR, VA, "srt", "TAXITIME_SEC_mvt", ["sr", "apt_stand", "apt"]).to_numpy(float)
        eA, eI = predA - y, predI - y
        sA, sI = eA.std(ddof=0), eI.std(ddof=0)
        rho = float(np.corrcoef(eA, eI)[0, 1])
        # optimal 2-term linear blend (least squares, with intercept)
        X = np.column_stack([np.ones_like(predA), predA, predI])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        blend = X @ beta
        # naive inverse-variance blend
        w = (1 / sA ** 2) / (1 / sA ** 2 + 1 / sI ** 2)
        inv = w * predA + (1 - w) * predI
        rows.append({"fold": fold, "n": len(y),
                     "RMSE A (NM path)": d.rmse(y, predA), "RMSE I (identity path)": d.rmse(y, predI),
                     "corr(eA,eI)": rho, "w on A": w,
                     "inverse-variance blend": d.rmse(y, inv),
                     "independent-blend prediction": 1 / np.sqrt(1 / sA ** 2 + 1 / sI ** 2),
                     "least-squares blend": d.rmse(y, blend)})
    W(C.md_table(pd.DataFrame(rows).set_index("fold"), floatfmt="{:.3f}"))
    W("")
    C.write("q0b_blend.md", "\n".join(out))
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
