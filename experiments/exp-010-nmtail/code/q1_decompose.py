"""Q1: where the NM-joined tail error lives in the current best models."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-010-nmtail/code")
import common as C
import prc.data as d

R1 = d.ROOT / "experiments/exp-001-retro-v1/oof"
T2 = d.ROOT / "experiments/exp-002-tail/oof/exp-002-tail-oof.parquet"
MODELS = {
    "retro-v1 R0 (guarded, linear leaf)": {
        "july_loo": R1 / "R0-jloo-july_loo.parquet",
        "december": R1 / "R0-dec-december.parquet",
        "january": R1 / "R0-jan-january.parquet"},
    "retro-v1 A0 (guarded, plain leaf)": {"july_loo": R1 / "A0-plain-jloo-july_loo.parquet"},
}


def load_models():
    out = {}
    for name, files in MODELS.items():
        for fold, p in files.items():
            if p.exists():
                o = pd.read_parquet(p, columns=["MVT_ID_mvt", "y_pred"])
                out[(name, fold)] = o
    tm = pd.read_parquet(T2)
    for fold, g in tm.groupby("fold"):
        out[("prc.tail_model router", fold)] = g[["MVT_ID_mvt", "y_pred"]]
    return out


def main():
    preds = load_models()
    frames = {m: C.add_regime(C.month(m)) for m in
              sorted({C.VALID_MONTH[f] for (_, f) in preds})}
    out = []; W = out.append
    W("# Q1 — where the NM-joined tail error lives\n")
    W("NM tail = `AOBT_3_flt` present and `TAXITIME_SEC_mvt > 3600`. "
      "`taxi = nm_taxi - phi * gap` with `gap = LOBT - AOBT` and "
      "`phi = 0` block-on-AOBT, `phi = 1` block-on-LOBT.\n")

    # ---------- the segment itself
    W("## The segment\n")
    rows = []
    for m in d.MONTHS:
        f = C.add_regime(C.month(m))
        nm = f[f.has_nm]
        t = nm[nm.TAXITIME_SEC_mvt > 3600]
        tt = t[t.gap.abs() > 300]
        rows.append({"month": m, "departures": len(f), "NM rows": len(nm),
                     "NM tail": len(t), "NM tail %": 100 * len(t) / len(f),
                     "share block-on-LOBT": 100 * (t.anchor_true == "lobt").mean(),
                     "median phi (|gap|>300)": float(np.nanmedian(tt.phi)) if len(tt) else np.nan,
                     "frac |phi-1|<0.1": 100 * float(((tt.phi - 1).abs() < 0.1).mean()) if len(tt) else np.nan})
    W(C.md_table(pd.DataFrame(rows).set_index("month"), floatfmt="{:.2f}"))
    W("")

    # ---------- model error on the segment, and the oracle split
    W("## Model error on the NM tail, against the two anchors and the oracle\n")
    W("`oracle anchor` picks, per row, whichever of `nm_taxi` and `mvt_lobt` is closer "
      "to the label: its RMSE is the magnitude error that survives a perfect regime "
      "call. `model implied anchor` is whichever of the two the model's own prediction "
      "is closer to.\n")
    rows = []
    for (name, fold), o in sorted(preds.items()):
        f = frames[C.VALID_MONTH[fold]]
        j = f.merge(o, on="MVT_ID_mvt", how="inner")
        t = j[j.has_nm & j.TAXITIME_SEC_mvt.gt(3600)]
        y = t.TAXITIME_SEC_mvt.to_numpy(float)
        imp = np.where((t.y_pred - t.pred_lobt).abs() < (t.y_pred - t.pred_aobt).abs(),
                       "lobt", "aobt")
        ok = imp == t.anchor_true.to_numpy()
        rows.append({
            "model": name, "fold": fold, "n": len(t),
            "model": name,
            "RMSE model": d.rmse(y, t.y_pred),
            "RMSE always-AOBT": d.rmse(y, t.pred_aobt),
            "RMSE always-LOBT": d.rmse(y, t.pred_lobt),
            "RMSE oracle anchor": d.rmse(y, t.oracle_pred),
            "anchor call correct %": 100 * ok.mean(),
            "RMSE where correct": d.rmse(y[ok], t.y_pred.to_numpy()[ok]),
            "RMSE where wrong": d.rmse(y[~ok], t.y_pred.to_numpy()[~ok]) if (~ok).any() else np.nan,
            "SSE share of wrong %": 100 * float(((t.y_pred.to_numpy()[~ok] - y[~ok]) ** 2).sum()
                                                / ((t.y_pred.to_numpy() - y) ** 2).sum()),
        })
    T = pd.DataFrame(rows).set_index("fold")
    W(C.md_table(T, floatfmt="{:.1f}"))
    W("")
    C.write("q1_decompose.md", "\n".join(out))
    print(T.to_string())


if __name__ == "__main__":
    main()
