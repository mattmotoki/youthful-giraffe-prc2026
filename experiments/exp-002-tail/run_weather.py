"""it12: ERA5 weather on the regime-routed model, aimed at the January tail.

January's remaining error is 60 % in rows above 3600 s that are *not* noac, and
they cluster on the 5 % worst airport-days. Airport-day aggregates of
`MVT - AOBT_3` (it11) took only 1 % off. `prc.weather_features` carries the
winter set the de-icing story predicts: `subzero`, `snowing`, `freezing_precip`,
`snow_3h_mm`, `low_visibility`, plus runway-aligned wind.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd
import prc.data as D
sys.path.insert(0, str(D.ROOT / "experiments" / "exp-002-tail"))
from run_all import EXP, ART, CATS, load, harmonise, reg, metrics
from run_final import DAYF, FULL2, NOFLT2
from noac_model import fit_predict as noac_fit_predict


def load_wx(months):
    return pd.concat([pd.read_parquet(ART / f"weather_{m}.parquet") for m in months], ignore_index=True)


def attach(sig, months):
    wx = load_wx(months).set_index("MVT_ID_mvt")
    wx = wx.reindex(sig.MVT_ID_mvt.to_numpy())
    wx.index = sig.index
    assert wx.notna().any(axis=1).mean() > 0.95, "weather join lost rows"
    return pd.concat([sig, wx], axis=1), list(wx.columns)


def run(fold_name, rows):
    f = D.FOLDS[fold_name]
    tr = load(f.train_months); va = load([f.valid_month])
    tr, WX = attach(tr, f.train_months)
    va, _ = attach(va, [f.valid_month])
    tr, va = harmonise(tr, va)
    tr = tr[tr.y.notna()].reset_index(drop=True)
    yv = va.y.to_numpy(float)
    nm_tr = tr[tr.nonm == 0].reset_index(drop=True)
    all_tr = tr[tr.noac == 0].reset_index(drop=True)
    nn_va = (va.nonm.to_numpy() == 1) & (va.noac.to_numpy() == 0)
    na_va = va.noac.to_numpy() == 1
    o_nm = nm_tr.mvt_aobt.to_numpy(float) - nm_tr.y.to_numpy(float)
    noac_out = noac_fit_predict(tr[tr.noac == 1].reset_index(drop=True), va[na_va].reset_index(drop=True))
    P = {}
    t0 = time.time()
    for tag, F, NF in (("day-only", FULL2, NOFLT2), ("day+weather", FULL2 + WX, NOFLT2 + WX)):
        pnm = va.mvt_aobt.to_numpy(float) - reg(nm_tr, F, o_nm).predict(va[F])
        pnn = reg(all_tr, NF, all_tr.y.to_numpy(float)).predict(va[NF])
        p = pnm.copy(); p[nn_va] = pnn[nn_va]
        for na_tag in ("mixture-expectation", "lgb-direct-offset"):
            q = p.copy(); q[na_va] = noac_out[na_tag]
            P[f"it12-{tag}|noac:{na_tag}"] = np.clip(q, 60.0, None)
        q = p.copy(); q[na_va] = yv[na_va]
        P[f"it12-{tag}|noac:ORACLE"] = np.clip(q, 60.0, None)
        print(f"  {tag} {time.time()-t0:.0f}s", flush=True)
    for k, p in P.items():
        rows.append(metrics(yv, p, va, k, fold_name))
    np.savez_compressed(EXP / "work" / f"wx_preds_{fold_name}.npz", **P)
    best = "it12-day+weather|noac:mixture-expectation"
    pd.DataFrame({"MVT_ID_mvt": va.MVT_ID_mvt.to_numpy(), "fold": fold_name, "y_true": yv,
                  "y_pred": P[best]}).to_parquet(EXP / "work" / f"wx_oof_{fold_name}.parquet", index=False)
    print(f"[{fold_name}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    rows = []
    for fd in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["january", "july", "december"]):
        run(fd, rows)
        pd.DataFrame(rows).to_csv(EXP / "work" / "metrics_weather.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
