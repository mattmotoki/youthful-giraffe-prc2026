"""Q1: out-of-range inputs on the ranking months, and what the candidate predicts there.

Training range = the twelve 2025 months, which is what the submission fit sees.
Two ranges per numeric column: the observed min-max, and the 0.1-99.9 percentile
band. Categoricals: levels the twelve months never contain.

For every material exposure the script reports what the SHIPPED candidate
predicts on those rows against in-range rows MATCHED on (airport, take-off hour)
-- the match is a reweighting of the in-range rows by the exposure rows' own
(airport, hour) cell counts, so the comparison is not confounded by the exposure
sitting at one airport or one time of day.

  python q1_range.py v2      # exp-008's frozen 54-column frame (C2-cat, L2-hub)
  python q1_range.py tail    # prc.tail_features.SIGNALS (T-tailv2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402

EXP = F.EXP
OUT = EXP / "analysis"
MATERIAL_ROWS = 200
MATERIAL_SHARE = 0.005


def _spec(which: str):
    if which == "v2":
        tr = F.v2_training()
        rk = F.v2_ranking()
        drop = set(F.V2_DROP)
        apt, hour = "airport", "local_hour"
    elif which == "tail":
        tr = F.tail_training()
        rk = F.tail_ranking()
        drop = {"MVT_ID_mvt", "month_str"}
        apt, hour = "apt", "hour"
    else:
        raise SystemExit(which)
    cols = [c for c in rk.columns if c not in drop and c in tr.columns]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(rk[c])]
    cat = [c for c in cols if c not in num]
    return tr, rk, num, cat, apt, hour


def matched_mean(values, mask, cells) -> tuple[float, float]:
    """(mean over `mask`, mean over ~mask reweighted to `mask`'s cell counts)."""
    v = np.asarray(values, dtype="float64")
    inn = ~mask
    a = float(np.nanmean(v[mask])) if mask.any() else np.nan
    if not mask.any() or not inn.any():
        return a, np.nan
    df = pd.DataFrame({"cell": cells, "v": v})
    tgt = df.loc[mask].groupby("cell", observed=True).size()
    ref = df.loc[inn].groupby("cell", observed=True)["v"].mean()
    common = tgt.index.intersection(ref.index)
    if not len(common):
        return a, np.nan
    w = tgt.loc[common].to_numpy("float64")
    return a, float(np.average(ref.loc[common].to_numpy("float64"), weights=w))


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "v2"
    tr, rk, num, cat, APT, HOUR = _spec(which)
    cand = F.candidate()
    pred = cand.reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    month = rk["month_str"].to_numpy()
    airport = rk[APT].astype("object").to_numpy()
    cells = pd.Series(airport).astype(str) + "|" + pd.Series(rk[HOUR].to_numpy()).astype(str)
    cells = cells.to_numpy()
    # the observable the residual target is anchored on
    obs_col = "d_takeoff_aobt" if which == "v2" else "mvt_aobt"
    obs = rk[obs_col].to_numpy("float64") if obs_col in rk.columns else np.full(len(rk), np.nan)

    print(f"[{which}] train {tr.shape} ranking {rk.shape}  "
          f"{len(num)} numeric, {len(cat)} categorical", flush=True)

    # ---------------------------------------------------------- numeric ----
    rows, per_apt, exposure_masks = [], [], {}
    for c in num:
        t = tr[c].to_numpy("float64")
        t = t[np.isfinite(t)]
        if not len(t):
            continue
        lo, hi = float(t.min()), float(t.max())
        plo, phi = (float(np.quantile(t, 0.001)), float(np.quantile(t, 0.999)))
        v = rk[c].to_numpy("float64")
        fin = np.isfinite(v)
        out_mm = fin & ((v < lo) | (v > hi))
        out_pq = fin & ((v < plo) | (v > phi))
        for m in ("2026-01", "2026-07"):
            sel = month == m
            n = int(sel.sum())
            rows.append(dict(frame=which, feature=c, month=m, n=n,
                             n_null=int((~fin & sel).sum()),
                             n_out_minmax=int((out_mm & sel).sum()),
                             n_out_p999=int((out_pq & sel).sum()),
                             share_out_minmax=float((out_mm & sel).sum()) / n,
                             share_out_p999=float((out_pq & sel).sum()) / n,
                             train_min=lo, train_max=hi, train_p001=plo, train_p999=phi,
                             rank_min=float(np.nanmin(v[sel])) if fin[sel].any() else np.nan,
                             rank_max=float(np.nanmax(v[sel])) if fin[sel].any() else np.nan))
            for a in np.unique(airport[sel]):
                s2 = sel & (airport == a)
                k = int((out_mm & s2).sum())
                if k:
                    per_apt.append(dict(frame=which, feature=c, month=m, airport=a,
                                        n=int(s2.sum()), n_out_minmax=k,
                                        share=k / int(s2.sum()),
                                        n_out_p999=int((out_pq & s2).sum())))
        if out_mm.sum() >= MATERIAL_ROWS or (out_mm.sum() and out_mm.sum() / len(v) >= MATERIAL_SHARE):
            exposure_masks[f"num:{c}"] = out_mm

    R = pd.DataFrame(rows)
    R.to_csv(OUT / f"q1_numeric_{which}.csv", index=False)
    A = pd.DataFrame(per_apt)
    A.to_csv(OUT / f"q1_numeric_by_airport_{which}.csv", index=False)

    # ------------------------------------------------------ categorical ----
    crows, clevels = [], []
    for c in cat:
        seen = set(pd.unique(tr[c].astype("object").dropna()))
        v = rk[c].astype("object")
        unseen = v.notna() & ~v.isin(seen)
        um = unseen.to_numpy()
        for m in ("2026-01", "2026-07"):
            sel = month == m
            crows.append(dict(frame=which, feature=c, month=m,
                              n_levels_train=len(seen),
                              n_levels_rank=int(v[sel].nunique()),
                              n_unseen_levels=int(v[sel & um].nunique()),
                              n_unseen_rows=int((um & sel).sum()),
                              share_unseen=float((um & sel).sum()) / int(sel.sum()),
                              n_null=int((v.isna().to_numpy() & sel).sum())))
        if um.any():
            g = pd.DataFrame({"level": v[um].to_numpy(), "month": month[um],
                              "airport": airport[um]})
            gg = g.groupby(["level", "month", "airport"], observed=True).size()
            for (lv, m, a), k in gg.items():
                clevels.append(dict(frame=which, feature=c, level=lv, month=m,
                                    airport=a, n_rows=int(k)))
        if um.sum() >= MATERIAL_ROWS or (um.sum() and um.sum() / len(v) >= MATERIAL_SHARE):
            exposure_masks[f"cat:{c}"] = um

    C = pd.DataFrame(crows)
    C.to_csv(OUT / f"q1_categorical_{which}.csv", index=False)
    CL = pd.DataFrame(clevels)
    CL.to_csv(OUT / f"q1_unseen_levels_{which}.csv", index=False)

    # ------------------------------- what the candidate predicts there ----
    prows = []
    for key, mask in exposure_masks.items():
        for m in ("2026-01", "2026-07", "both"):
            sel = np.ones(len(rk), bool) if m == "both" else (month == m)
            mm = mask & sel
            if not mm.any():
                continue
            sub = np.where(sel)[0]
            p_out, p_in = matched_mean(pred[sub], mm[sub], cells[sub])
            o_out, o_in = matched_mean(obs[sub], mm[sub], cells[sub])
            r = obs - pred
            r_out, r_in = matched_mean(r[sub], mm[sub], cells[sub])
            prows.append(dict(frame=which, exposure=key, month=m, n=int(mm.sum()),
                              pred_out=p_out, pred_in_matched=p_in, d_pred=p_out - p_in,
                              obs_out=o_out, obs_in_matched=o_in, d_obs=o_out - o_in,
                              r_out=r_out, r_in_matched=r_in, d_r=r_out - r_in))
    P = pd.DataFrame(prows)
    P.to_csv(OUT / f"q1_prediction_on_exposure_{which}.csv", index=False)

    top = R[R.n_out_minmax > 0].sort_values("n_out_minmax", ascending=False).head(25)
    print(top[["feature", "month", "n_out_minmax", "share_out_minmax",
               "train_min", "train_max", "rank_min", "rank_max"]].to_string(index=False))
    print()
    print(C[C.n_unseen_rows > 0].sort_values("n_unseen_rows", ascending=False).to_string(index=False))
    print()
    if len(P):
        print(P[P.month == "both"].sort_values("n", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
