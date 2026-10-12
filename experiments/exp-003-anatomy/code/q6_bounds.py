"""Q6 (desk note): stand-occupancy bounds — validity, tightness, and what clipping buys."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d
import prc.stand_features as sf
from features import ART

FOLDS = {"july": ([f"2025-{m:02d}" for m in range(1, 7)], "2025-07"),
         "december": ([f"2025-{m:02d}" for m in range(1, 12)], "2025-12"),
         "january": ([f"2025-{m:02d}" for m in range(2, 13)], "2025-01")}


def main():
    out = []; W = out.append
    W("# Q6 — stand occupancy as a bound on the hidden block time\n")
    W("A stand holds one aircraft at a time and arrivals keep their on-block time in the "
      "ranking frame, so an arrival at stand S bounds the departure's hidden off-block `B`. "
      "In taxi terms `taxi = MVT - B`, so an arrival that parks *after* B gives a **lower "
      "bound on taxi** and an arrival that parked *before* B gives an **upper bound**. "
      "The anchors below differ in how 'the next arrival' is defined without knowing `B`.\n")

    W("## Validity and tightness, measured on labelled months\n")
    W("`coverage` = share of departures the bound is defined on; `valid` = share of those it "
      "does not violate; `slack` = bound minus truth in seconds (>= 0 when the bound holds).\n")
    rows = []
    for m in ["2025-07", "2025-12", "2025-01"]:
        full = d.load_month(m)
        for key in ["key_raw", "key_base"]:
            v = sf.validate_bounds(full, key=key)
            v["month"] = m; v["stand key"] = key
            rows.append(v)
    V = pd.concat(rows, ignore_index=True)
    V = V.set_index(["month", "stand key", "bound"])[
        ["coverage", "valid", "slack_p10", "slack_p50", "slack_p90", "n"]]
    W(C.md_table(V.reset_index().set_index("month"), floatfmt="{:.4f}"))
    W("")
    W("Reading: `taxi_lb_nm` (first arrival on-block after the departure's *take-off*) is a "
      "bound by construction and never violated. `taxi_lb_na` (first arrival after the "
      "departure's *AOBT*) is tighter and violated on 0.05 % of rows with the raw stand name "
      "and 0.3 % with the MARS parent name — the parent key is wrong here because two "
      "aircraft legitimately occupy 519L and 519R at once. `taxi_ub_pm` (last arrival before "
      "take-off) is violated on 12 % of rows: at a busy stand the next aircraft frequently "
      "parks between our off-block and our take-off, which is the phenomenon the bound is "
      "meant to catch, not a bound at all. **Use `key_raw`, use the next-arrival bounds as "
      "hard bounds, and use the previous-arrival ones as features only.**\n")

    W("## What the bound is worth: clipping a predictor at `taxi >= MVT - A_next`\n")
    W("Predictor is the reference naive one (`taxi = MVT - AOBT_3 - stand median offset`, "
      "stand medians fitted on the fold's training months). This isolates the bound's value "
      "from the model's.\n")
    rows = []
    for fold, (trm, vam) in FOLDS.items():
        tr = pd.concat([pd.read_parquet(ART / f"{m}.parquet") for m in trm], ignore_index=True)
        va = pd.read_parquet(ART / f"{vam}.parquet")
        for f in (tr, va):
            f["off"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
            f["y"] = f.TAXITIME_SEC_mvt.astype(float)
        mt = tr.has_nm & tr.y.le(3600)
        g = tr[mt].groupby("key_raw", observed=True)["off"].agg(["median", "size"])
        g = g[g["size"] >= 30]["median"]
        gm = tr.loc[mt].groupby("apt", observed=True)["off"].median()
        off = va.key_raw.map(g).fillna(va.apt.map(gm)).fillna(tr.loc[mt, "off"].median())
        pred = (va.nm_taxi - off).to_numpy()
        ymed = tr.loc[tr.y.notna()].groupby("apt", observed=True)["y"].median()
        pred = np.where(va.has_nm.to_numpy(), pred, va.apt.map(ymed).fillna(tr.y.median()).to_numpy())
        y = va.y.to_numpy()
        lb = va["taxi_lb_na"].to_numpy()
        lb_safe = va["taxi_lb_nm"].to_numpy()
        for bname, b in [("taxi_lb_na (after AOBT)", lb), ("taxi_lb_nm (after take-off)", lb_safe)]:
            clipped = np.where(np.isfinite(b) & (pred < b), b, pred)
            binds = np.isfinite(b) & (pred < b)
            for seg, m in [("all", np.ones(len(y), bool)),
                           ("bulk (NM, y<=3600)", (va.has_nm & va.y.le(3600)).to_numpy()),
                           ("y>3600", (va.y > 3600).to_numpy()),
                           ("no NM", (~va.has_nm).to_numpy())]:
                rows.append({"fold": fold, "bound": bname, "segment": seg, "n": int(m.sum()),
                             "binds %": 100 * binds[m].mean(),
                             "RMSE before": d.rmse(y[m], pred[m]),
                             "RMSE after clip": d.rmse(y[m], clipped[m]),
                             "gain": d.rmse(y[m], pred[m]) - d.rmse(y[m], clipped[m])})
    T = pd.DataFrame(rows)
    W(C.md_table(T.set_index("fold"), floatfmt="{:.2f}"))
    W("")
    C.write("q6_bounds.md", "\n".join(out))
    print(T.to_string())


if __name__ == "__main__":
    main()
