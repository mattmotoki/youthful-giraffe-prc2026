"""Round-1 review repairs, as measurements rather than edits.

M8  the day-state exposure count was one-sided (`v > hi` only), which dropped
    the `ds_nm_rate_z` collapse that `docs/shift-2026.md` calls EHAM January's
    signature. Recount symmetrically.
M2  the "worst 5 % of days" slice contained the out-of-envelope days, so the
    contrast was between a set and its own subset. Recompute it disjointly.
M11 the claim "correct ids and order" was never checked. Check it.
supplied-input rule  the assignment's 5,290 no-NM rows against the audit's 4,895 `nonm`: find
    the remainder and say what the candidate does there.
compute allocation  per-airport 0.1-99.9 shares for every numeric column, not only the ones
    with a min-max exceedance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
sys.path.insert(0, str(F.ROOT / "experiments" / "exp-012-blend2"))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
from prc.day_state_features import Z_STATS  # noqa: E402

OUT = F.EXP / "analysis"
CACHE = F.ROOT / "artifacts" / "exp-014" / "frames"


def m8_daystate_symmetric() -> None:
    tr = pd.read_parquet(CACHE / "ds-train.parquet")
    rk = pd.read_parquet(CACHE / "ds-ranking.parquet")
    z = [f"{c}_z" for c in Z_STATS]
    lim = tr.groupby("apt", observed=True)[z].agg(["min", "max"])
    over = np.zeros(len(rk), int)
    under = np.zeros(len(rk), int)
    for c in z:
        hi = rk["apt"].map(lim[(c, "max")]).to_numpy("float64")
        lo = rk["apt"].map(lim[(c, "min")]).to_numpy("float64")
        v = rk[c].to_numpy("float64")
        over += (np.isfinite(v) & (v > hi)).astype(int)
        under += (np.isfinite(v) & (v < lo)).astype(int)
    rk = rk.assign(n_over=over, n_under=under, n_out=over + under)
    by = rk.groupby(["month_str", "apt"], observed=True).agg(
        n=("n_out", "size"),
        n_over=("n_over", lambda s: int((s > 0).sum())),
        n_under=("n_under", lambda s: int((s > 0).sum())),
        n_out=("n_out", lambda s: int((s > 0).sum())),
        max_out=("n_out", "max")).reset_index()
    by["share_out"] = by["n_out"] / by["n"]
    by.to_csv(OUT / "r1_daystate_by_airport_symmetric.csv", index=False)
    print("=== M8: day-state `_z` exposure, counting BOTH tails ===")
    print(by[by.n_out > 0].sort_values("n_out", ascending=False).round(3).to_string(index=False))
    for m in ("2026-01", "2026-07"):
        s = rk["month_str"] == m
        print(f"{m}: over-only {int((rk.loc[s,'n_over']>0).sum())}, "
              f"either tail {int((rk.loc[s,'n_out']>0).sum())} of {int(s.sum())} "
              f"({100*(rk.loc[s,'n_out']>0).mean():.1f} %)")
    day = rk[rk["n_out"] > 0].groupby(["month_str", "apt", "day"], observed=True).agg(
        n=("n_out", "size"), z_over=("n_over", "max"), z_out=("n_out", "max")).reset_index()
    day.to_csv(OUT / "r1_daystate_exposed_days_symmetric.csv", index=False)
    for m in ("2026-01", "2026-07"):
        d = day[day.month_str == m]
        print(f"{m}: {len(d)} exposed airport-days; "
              f"{int((d.z_over>=6).sum())} with 6+ columns ABOVE the maximum")
    print(day[day.z_over >= 6].sort_values("z_over", ascending=False).to_string(index=False))


def m2_disjoint_slices() -> None:
    FD = pd.read_csv(OUT / "q2b_fold_days.csv", parse_dates=["day"])
    rows = []
    for fold, g in FD.groupby("fold"):
        g = g.copy()
        g["rank"] = g.groupby("apt")["obs_mean"].rank(pct=True)
        ex = g["over_tr_max"] > 0
        for label, m in (("whole fold", np.ones(len(g), bool)),
                         ("worst 5 % of days, INCLUDING out-of-envelope", g["rank"] > 0.95),
                         ("worst 5 % of days, EXCLUDING out-of-envelope", (g["rank"] > 0.95) & ~ex),
                         ("past the training maximum", ex)):
            gg = g[np.asarray(m)]
            if not len(gg):
                continue
            w = gg["n"].to_numpy("float64")
            rows.append(dict(fold=fold, slice=label, days=len(gg), rows=int(w.sum()),
                             bias=float(np.average(gg["bias"], weights=w)),
                             rmse=float(np.sqrt(np.average(gg["rmse"] ** 2, weights=w)))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "r1_disjoint_slices.csv", index=False)
    print("\n=== M2: the severe-but-in-range slice with the out-of-envelope days removed ===")
    print(T.round(1).to_string(index=False))


def m11_id_order() -> None:
    tpl = D.load_submitting()
    print("\n=== M11: candidate ids against the submission template ===")
    for name, path in (("candidate 1", "submissions/candidates/exp-012-blend-1.parquet"),
                       ("candidate 2", "submissions/candidates/exp-012-blend-2.parquet")):
        d = pd.read_parquet(F.ROOT / path)
        same_set = set(d[D.ID].to_numpy("float64")) == set(tpl[D.ID].to_numpy("float64"))
        same_order = bool(np.array_equal(d[D.ID].to_numpy("float64"),
                                         tpl[D.ID].to_numpy("float64")))
        print(f"  {name}: rows {len(d)} (template {len(tpl)}), id set identical {same_set}, "
              f"template order {same_order}, dtype {d[D.TARGET].dtype}, "
              f"finite {bool(np.isfinite(d[D.TARGET].to_numpy()).all())}")


def d1_no_nm_reconciliation() -> None:
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AOBT_3_flt", "LOBT_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt"])
    dep = rk[rk["PHASE_mvt"] == "DEP"].copy()
    dep["regime"] = B.regimes_from_raw(dep, B.LATE_CUT)
    no_aobt = dep["AOBT_3_flt"].isna()
    no_fid = dep["FLIGHT_ID_mvt"].isna()
    print("\n=== D1: reconciling the assignment's 5,290 no-NM rows ===")
    print(f"  departures with AOBT_3_flt null: {int(no_aobt.sum())}")
    print(f"  departures with FLIGHT_ID_mvt null: {int(no_fid.sum())}  "
          f"(= nonm {int((dep.regime=='nonm').sum())} + noac {int((dep.regime=='noac').sum())})")
    odd = dep[no_aobt & ~no_fid]
    print(f"  no AOBT but a live NM FLIGHT_ID: {len(odd)} rows, regimes "
          f"{dict(odd['regime'].value_counts())}")
    if len(odd):
        c1 = F.candidate().reindex(odd["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
        c2 = F.candidate("submissions/candidates/exp-012-blend-2.parquet") \
            .reindex(odd["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
        o = odd.assign(cand1=c1, cand2=c2,
                       airport=odd["ADEP_mvt"].astype(str),
                       t=odd["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d %H:%M"))
        o[["MVT_ID_mvt", "airport", "t", "regime", "cand1", "cand2"]] \
            .to_csv(OUT / "r1_no_aobt_with_nm.csv", index=False)
        print(o[["airport", "t", "regime", "cand1", "cand2"]].round(0).to_string(index=False))
        print("  the residual guard is a no-op on these rows (it needs a finite AOBT), "
              "and no slice of the audit named them.")


def d3_per_airport_percentiles() -> None:
    for which, apt_col in (("v2", "airport"), ("tail", "apt")):
        if which == "v2":
            tr, rk = F.v2_training(), F.v2_ranking()
            drop = set(F.V2_DROP)
        else:
            tr, rk = F.tail_training(), F.tail_ranking()
            drop = {"MVT_ID_mvt", "month_str"}
        cols = [c for c in rk.columns if c not in drop and c in tr.columns]
        num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c])
               and pd.api.types.is_numeric_dtype(rk[c])]
        month = rk["month_str"].to_numpy()
        apt = rk[apt_col].astype("object").to_numpy()
        rows = []
        for c in num:
            t = tr[c].to_numpy("float64")
            t = t[np.isfinite(t)]
            if not len(t):
                continue
            lo, hi = float(t.min()), float(t.max())
            plo, phi = float(np.quantile(t, 0.001)), float(np.quantile(t, 0.999))
            v = rk[c].to_numpy("float64")
            fin = np.isfinite(v)
            mm = fin & ((v < lo) | (v > hi))
            pq = fin & ((v < plo) | (v > phi))
            for m in ("2026-01", "2026-07"):
                sm = month == m
                for a in np.unique(apt[sm]):
                    s2 = sm & (apt == a)
                    n = int(s2.sum())
                    rows.append(dict(frame=which, feature=c, month=m, airport=a, n=n,
                                     n_out_minmax=int((mm & s2).sum()),
                                     share_out_minmax=float((mm & s2).sum()) / n,
                                     n_out_p999=int((pq & s2).sum()),
                                     share_out_p999=float((pq & s2).sum()) / n))
        T = pd.DataFrame(rows)
        T.to_csv(OUT / f"r1_per_airport_full_{which}.csv", index=False)
        top = T.sort_values("share_out_p999", ascending=False).head(12)
        print(f"\n=== D3: per-airport 0.1-99.9 shares, {which} frame "
              f"({len(T)} rows, every numeric column) — worst cells ===")
        print(top[["feature", "month", "airport", "n", "n_out_p999",
                   "share_out_p999"]].round(4).to_string(index=False))


if __name__ == "__main__":
    m8_daystate_symmetric()
    m2_disjoint_slices()
    m11_id_order()
    d1_no_nm_reconciliation()
    d3_per_airport_percentiles()
