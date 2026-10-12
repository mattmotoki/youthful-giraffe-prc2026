"""Calibration of the leave-one-month-out `p` and `phi`, and what each is worth."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-010-nmtail/code")
import common as C
import prc.data as d

OOF = d.ROOT / "experiments/exp-010-nmtail/oof/loo_2025.parquet"


def main():
    o = pd.read_parquet(OOF)
    o = o[o.has_nm & o.TAXITIME_SEC_mvt.notna()].copy()
    y = o.TAXITIME_SEC_mvt.to_numpy(float)
    o["lobt_true"] = np.abs(y - o.mvt_lobt) < np.abs(y - o.nm_taxi)
    o["pred_phi"] = np.maximum(o.nm_taxi - o.phi_hat * o.gap, 0)
    o["pred_p"] = np.maximum(o.nm_taxi - o.p_hat * o.gap, 0)
    tail = y > 3600
    out = []; W = out.append
    W("# Leave-one-month-out over all twelve 2025 months\n")
    W(f"{len(o):,} NM-joined departures, each scored by a model fitted on the other "
      "eleven months (`code/loo.py`, 400 rounds, lr 0.05, 63 leaves).\n")

    W("## What each parameterisation is worth\n")
    rows = []
    for label, p in [("phi regression (weight gap^2)", o.pred_phi),
                     ("p classifier (two-point)", o.pred_p),
                     ("always block-on-AOBT", o.nm_taxi),
                     ("always block-on-LOBT", o.mvt_lobt)]:
        rows.append({"predictor": label,
                     "RMSE all NM": d.rmse(y, p),
                     "RMSE NM tail": d.rmse(y[tail], p[tail]),
                     "RMSE NM bulk": d.rmse(y[~tail], p[~tail])})
    W(C.md_table(pd.DataFrame(rows).set_index("predictor"), floatfmt="{:.1f}"))
    W("")

    W("## Calibration of `p`, on the rows where the two anchors differ (|gap| > 600 s)\n")
    sub = o[o.gap.abs() > 600].copy()
    sub["bin"] = pd.cut(sub.p_hat, [0, .05, .1, .2, .3, .5, .7, .9, 1.0], include_lowest=True)
    g = sub.groupby("bin", observed=True)
    t = pd.DataFrame({"n": g.size(),
                      "mean p_hat": g.p_hat.mean(),
                      "observed block-on-LOBT": g.lobt_true.mean(),
                      "mean phi_hat": g.phi_hat.mean(),
                      "observed mean phi": g.apply(
                          lambda x: float(np.nanmean((x.nm_taxi - x.TAXITIME_SEC_mvt) / x.gap))),
                      "P(taxi>3600) %": 100 * g.apply(lambda x: float((x.TAXITIME_SEC_mvt > 3600).mean()))})
    W(C.md_table(t.rename_axis("p_hat bin"), floatfmt="{:.3f}"))
    W("")
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        W(f"AUC of `p_hat` for block-on-LOBT on those rows: "
          f"**{roc_auc_score(sub.lobt_true, sub.p_hat):.3f}**; Brier score "
          f"{brier_score_loss(sub.lobt_true, sub.p_hat):.4f} against "
          f"{brier_score_loss(sub.lobt_true, np.full(len(sub), sub.lobt_true.mean())):.4f} "
          f"for the base rate.\n")
        W(f"AUC of `p_hat` for `taxi > 3600 s` on the same rows: "
          f"**{roc_auc_score(sub.TAXITIME_SEC_mvt > 3600, sub.p_hat):.3f}**; of `phi_hat`: "
          f"**{roc_auc_score(sub.TAXITIME_SEC_mvt > 3600, sub.phi_hat):.3f}**.\n")
    except Exception as e:  # pragma: no cover
        W(f"(sklearn metrics unavailable: {e})\n")

    W("## By month\n")
    g = o.groupby("month", observed=True)
    t = pd.DataFrame({"NM rows": g.size(),
                      "NM tail": g.apply(lambda x: int((x.TAXITIME_SEC_mvt > 3600).sum())),
                      "RMSE phi, tail": g.apply(lambda x: d.rmse(
                          x.TAXITIME_SEC_mvt[x.TAXITIME_SEC_mvt > 3600],
                          x.pred_phi[x.TAXITIME_SEC_mvt > 3600])),
                      "RMSE p, tail": g.apply(lambda x: d.rmse(
                          x.TAXITIME_SEC_mvt[x.TAXITIME_SEC_mvt > 3600],
                          x.pred_p[x.TAXITIME_SEC_mvt > 3600])),
                      "RMSE phi, all NM": g.apply(lambda x: d.rmse(x.TAXITIME_SEC_mvt, x.pred_phi))})
    W(C.md_table(t.rename_axis("month"), floatfmt="{:.1f}"))
    C.write("q2_calibration.md", "\n".join(out))


if __name__ == "__main__":
    main()
