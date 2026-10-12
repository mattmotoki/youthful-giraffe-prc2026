"""Q1b: the NM-tail error sliced by the observables the desk named."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-010-nmtail/code")
import common as C
import prc.data as d

R1 = d.ROOT / "experiments/exp-001-retro-v1/oof"
FOLD_FILE = {"july_loo": R1 / "R0-jloo-july_loo.parquet",
             "december": R1 / "R0-dec-december.parquet",
             "january": R1 / "R0-jan-january.parquet"}


def slices(t):
    s = {}
    s["airport"] = t.apt.astype(str)
    s["AOBT-LOBT band"] = pd.cut(-t.gap, [-np.inf, -600, 0, 600, 1800, 3600, 7200, np.inf],
                                 labels=["<-10m", "-10..0", "0..10m", "10..30m",
                                         "30..60m", "1..2h", ">2h"]).astype(str)
    s["LOBT == SCHED"] = np.where(t.lobt_sched.abs() <= 60, "yes (<=60 s)", "no")
    s["stand prefix"] = (t.apt.astype(str) + "|" + t.stand_pfx.astype(str))
    s["hour band"] = pd.cut(t.hour, [-1, 5, 9, 13, 17, 21, 24],
                            labels=["00-05", "06-09", "10-13", "14-17", "18-21", "22-23"]).astype(str)
    s["operator (top)"] = t.AIRCRAFT_OPERATOR_flt.astype(str).str[:8]
    return s


def table(t, key, vals, min_n=8):
    y = t.TAXITIME_SEC_mvt.to_numpy(float)
    imp = np.where((t.y_pred - t.pred_lobt).abs() < (t.y_pred - t.pred_aobt).abs(), "lobt", "aobt")
    ok = imp == t.anchor_true.to_numpy()
    df = pd.DataFrame({"g": vals, "y": y, "p": t.y_pred.to_numpy(),
                       "orc": t.oracle_pred.to_numpy(), "lob": t.pred_lobt.to_numpy(),
                       "ok": ok, "lobt_true": (t.anchor_true == "lobt").to_numpy()})
    g = df.groupby("g")
    out = pd.DataFrame({
        "n": g.size(),
        "block-on-LOBT %": 100 * g.lobt_true.mean(),
        "RMSE model": g.apply(lambda x: d.rmse(x.y, x.p)),
        "RMSE always-LOBT": g.apply(lambda x: d.rmse(x.y, x.lob)),
        "RMSE oracle": g.apply(lambda x: d.rmse(x.y, x.orc)),
        "anchor correct %": 100 * g.ok.mean(),
        "SSE share %": 100 * g.apply(lambda x: ((x.p - x.y) ** 2).sum()) / ((df.p - df.y) ** 2).sum(),
    })
    return out[out.n >= min_n].sort_values("SSE share %", ascending=False)


def main():
    out = []; W = out.append
    W("# Q1b — the NM-tail error by slice\n")
    W("Model is retro-v1's guarded linear-leaf line (`R0`), the current best. "
      "Slices with fewer than 8 rows are dropped.\n")
    for fold, p in FOLD_FILE.items():
        f = C.add_regime(C.month(C.VALID_MONTH[fold]))
        o = pd.read_parquet(p, columns=["MVT_ID_mvt", "y_pred"])
        j = f.merge(o, on="MVT_ID_mvt")
        t = j[j.has_nm & j.TAXITIME_SEC_mvt.gt(3600)].copy()
        W(f"## fold `{fold}` — {len(t)} NM-tail rows\n")
        for key, vals in slices(t).items():
            W(f"### by {key}\n")
            W(C.md_table(table(t, key, vals).rename_axis(key), floatfmt="{:.1f}"))
            W("")
    C.write("q1b_slices.md", "\n".join(out))


if __name__ == "__main__":
    main()
