"""Q1(b): the best OOF's bulk residuals, binned by prediction and by airport,
and Q1(a) recomputed inside the validation month so the two are comparable."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C
from q1a_pairs import pooled_sigma, COLS

FEAT = d.ROOT / "artifacts/exp-003-anatomy/feat"
BEST = {"july_loo": ("SB-L2hub-july_loo", "exp-008-diversity"),
        "december": ("F1-phifull-december", "exp-001-retro-v1"),
        "july": ("F1-phifull-july", "exp-001-retro-v1"),
        "january": ("L1-philirf-guard-january", "exp-001-retro-v1")}


def load_best(fold):
    name, src = BEST[fold]
    p = d.ROOT / f"experiments/{src}/oof/{name}.parquet"
    return pd.read_parquet(p, columns=["MVT_ID_mvt", "y_true", "y_pred"]), name


def month_frame(m):
    f = pd.read_parquet(FEAT / f"{m}.parquet", columns=COLS)
    f["h2"] = (f.hour // 2).astype(int)
    f["gap_bin"] = (f.aobt_lobt.fillna(0) / 120).round().astype(int)
    f["q_bin"] = (f.nbr_apt_30m.fillna(0) / 4).round().astype(int)
    return f


def main():
    out = []; W = out.append
    W("# Q1(b) — where the bulk residual sits, and the bound inside the same month\n")
    TIGHT = ["stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "gap_bin", "q_bin"]
    MID = ["stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2"]
    for fold in ("july_loo", "december"):
        m = d.FOLDS[fold].valid_month
        f = month_frame(m)
        o, name = load_best(fold)
        j = f.merge(o, on="MVT_ID_mvt")
        b = j[j.has_nm & j.noac.ne(1) & j.y_true.le(3600)].copy()
        b["res"] = b.y_pred - b.y_true
        W(f"## fold `{fold}` (validation month {m}), best OOF `{name}`\n")
        W(f"{len(b):,} bulk rows, RMSE **{d.rmse(b.y_true, b.y_pred):.1f}**.\n")
        W("### by predicted value\n")
        b["pbin"] = pd.qcut(b.y_pred, 10, duplicates="drop")
        g = b.groupby("pbin", observed=True)
        t = pd.DataFrame({"n": g.size(), "mean pred": g.y_pred.mean(),
                          "mean actual": g.y_true.mean(),
                          "bias (pred-actual)": g.res.mean(),
                          "residual sd": g.res.std(),
                          "RMSE": g.apply(lambda x: d.rmse(x.y_true, x.y_pred)),
                          "share of bulk SSE %": 100 * g.apply(lambda x: (x.res ** 2).sum())
                          / (b.res ** 2).sum()})
        W(C.md_table(t.rename_axis("predicted taxi, decile"), floatfmt="{:.1f}"))
        W("")
        W("### by airport, against the same-month near-duplicate bound\n")
        rows = []
        for a, ga in b.groupby("apt", observed=True):
            fa = f[(f.apt == a) & f.has_nm & f.noac.ne(1) & f.TAXITIME_SEC_mvt.le(3600)]
            rows.append({"airport": a, "n": len(ga),
                         "model RMSE": d.rmse(ga.y_true, ga.y_pred),
                         "bias": float(ga.res.mean()),
                         "bound, stand x rwy x op x 2h (same month)": pooled_sigma(fa, MID)["sigma"],
                         "bound, + gap and queue (same month)": pooled_sigma(fa, TIGHT)["sigma"],
                         "coverage of the tight bound %": pooled_sigma(fa, TIGHT)["share"],
                         "share of bulk SSE %": 100 * float((ga.res ** 2).sum() / (b.res ** 2).sum())})
        T = pd.DataFrame(rows).set_index("airport").sort_values("share of bulk SSE %", ascending=False)
        W(C.md_table(T, floatfmt="{:.1f}"))
        W("")
        fa = f[f.has_nm & f.noac.ne(1) & f.TAXITIME_SEC_mvt.le(3600)]
        W(f"All airports, same month: model {d.rmse(b.y_true, b.y_pred):.1f}; bound at "
          f"stand x runway x operator x 2h {pooled_sigma(fa, MID)['sigma']:.1f} "
          f"(coverage {pooled_sigma(fa, MID)['share']:.0f} %); bound with the gap and queue "
          f"bins {pooled_sigma(fa, TIGHT)['sigma']:.1f} "
          f"(coverage {pooled_sigma(fa, TIGHT)['share']:.0f} %).\n")
    C.write("q1b_residuals.md", "\n".join(out))


if __name__ == "__main__":
    main()
