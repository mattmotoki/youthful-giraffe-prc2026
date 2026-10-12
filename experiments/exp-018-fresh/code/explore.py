"""exp-018: the pre-modelling exploration behind RESULTS §1 and appendix A.

Every number in RESULTS §1 comes from here. Committed after review round 1
found that these lived only in a scratch directory (finding 8).

  PYTHONPATH=. .venv/bin/python experiments/exp-018-fresh/code/explore.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from prc import data

pd.set_option("display.width", 220)
JULY = "2025-07"


def load_all_dep() -> pd.DataFrame:
    out = []
    for m in data.MONTHS:
        d = data.load_month(m)
        out.append(d[d.PHASE_mvt == "DEP"])
    return pd.concat(out, ignore_index=True)


def s1_tail(dep: pd.DataFrame) -> None:
    """§1.1 the metric is a tail metric; §1.2 the class is identifiable."""
    y = dep.TAXITIME_SEC_mvt.astype(float)
    nonm = dep.AOBT_3_flt.isna(); lirf = dep.ADEP_mvt.eq("LIRF")
    tot = ((y - 948) ** 2).sum()
    print(f"§1.1 all departures {len(dep)}")
    for name, m in [("y > 3600", y > 3600), ("LIRF", lirf), ("no NM join", nonm),
                    ("LIRF and no NM", lirf & nonm)]:
        print(f"   {name:16s} n={int(m.sum()):8d} ({m.mean()*100:6.2f} %)  "
              f"share of SS about the median {((y[m]-948)**2).sum()/tot*100:5.1f} %")
    print(f"§1.2 null AIRCRAFT_TYPE_mvt: {int(dep.AIRCRAFT_TYPE_mvt.isna().sum())} rows, "
          f"by airport {dep[dep.AIRCRAFT_TYPE_mvt.isna()].ADEP_mvt.value_counts().to_dict()}")


def s3_ghost(dep: pd.DataFrame) -> None:
    """§1.3 the noac class: how much of it is schedule-anchored, and the anchors."""
    g = dep[dep.AIRCRAFT_TYPE_mvt.isna()].copy()
    g["y"] = g.TAXITIME_SEC_mvt.astype(float)
    g["ms"] = (g.MVT_TIME_UTC_mvt - g.SCHED_TIME_UTC_mvt).dt.total_seconds()
    g = g[g.ms > 60]
    ratio = g.y / g.ms
    print(f"\n§1.3 noac with MVT-SCHED > 60 s: n={len(g)}")
    print(f"   |BLOCK-SCHED| <= 6 s          {(np.abs(g.y-g.ms)<=6).mean()*100:.1f} %")
    print(f"   ratio y/(MVT-SCHED) > 0.5     {(ratio>0.5).mean()*100:.1f} %")
    print(f"   ratio quantiles               {np.percentile(ratio,[5,25,50,75,95]).round(3)}")
    print(f"   RMSE of MVT-SCHED on the class {data.rmse(g.y, g.ms):.0f}")
    a = np.abs(g.y - g.ms) <= 6
    mub = g.y[~a].mean()
    print(f"   RMSE of an oracle split (exact -> MVT-SCHED, else class mean {mub:.0f}) "
          f"{data.rmse(g.y, np.where(a, g.ms, mub)):.0f}")


def s4_sched(dep: pd.DataFrame) -> None:
    """§1.4 the schedule-placeholder class over the WHOLE population."""
    y = dep.TAXITIME_SEC_mvt.astype(float)
    bs = (dep.BLOCK_TIME_UTC_mvt - dep.SCHED_TIME_UTC_mvt).dt.total_seconds()
    ex = bs == 0; win = bs.abs() <= 6
    print(f"\n§1.4 BLOCK == SCHED exactly {int(ex.sum())} ({ex.mean()*100:.2f} %); "
          f"|BLOCK-SCHED| <= 6 s {int(win.sum())} ({win.mean()*100:.3f} %); "
          f"non-noac members {int((win & dep.AIRCRAFT_TYPE_mvt.notna()).sum())}")
    print("   histogram of BLOCK-SCHED over -8..+8 s:",
          {int(v): int((bs == v).sum()) for v in range(-8, 9)})
    print("   by airport (|d| <= 6 s):",
          dep.assign(w=win).groupby("ADEP_mvt")["w"].mean().mul(100).round(2).to_dict())
    r = (dep.MVT_TIME_UTC_mvt - dep.AOBT_3_flt).dt.total_seconds()
    res = (y - r)[r.notna()]; w = win[r.notna()]
    print(f"   twelve months, NM rows: RMSE(y-r) on the class {np.sqrt((res[w]**2).mean()):.0f} "
          f"vs {np.sqrt((res[~w]**2).mean()):.0f} elsewhere; the class is {w.mean()*100:.2f} % "
          f"of NM rows and {(res[w]**2).sum()/(res**2).sum()*100:.2f} % of the squared error")


def s5_screens() -> None:
    """Appendix A: the three screens, all negative. July plus three months."""
    rows = [data.load_month(m) for m in ("2025-01", "2025-04", "2025-07", "2025-10")]
    dep = pd.concat([d[d.PHASE_mvt == "DEP"] for d in rows], ignore_index=True)
    arr = pd.concat([d[d.PHASE_mvt == "ARR"] for d in rows], ignore_index=True)
    y = dep.TAXITIME_SEC_mvt.astype(float)
    r = (dep.MVT_TIME_UTC_mvt - dep.AOBT_3_flt).dt.total_seconds()
    res = y - r
    nz = dep.AOBT_3_flt.dt.second.fillna(0) != 0
    ok = dep.AOBT_3_flt.notna()
    print(f"\nA(a) AOBT second != 0 on {nz[ok].mean()*100:.1f} % of NM rows, "
          f"{dep.ADEP_mvt[nz & ok].value_counts(normalize=True).head(1).to_dict()} dominant; "
          f"|y-r| <= 2 s on {(res.abs()<=2)[nz & ok].mean()*100:.1f} % there vs "
          f"{(res.abs()<=2)[(~nz) & ok].mean()*100:.1f} % elsewhere")
    B = dep.BLOCK_TIME_UTC_mvt
    lab = np.zeros(len(dep), dtype=int)
    for k, col in enumerate(["SCHED_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt"], start=1):
        m = ((B - dep[col]).dt.total_seconds().abs() <= 6).fillna(False) & (lab == 0)
        lab[m] = k
        print(f"A(b) disjoint class {k} ({col}): n={int(m.sum())} ({m.mean()*100:.2f} %), "
              f"share of the baseline's squared error {(res[m]**2).sum()/(res**2).sum()*100:.2f} %, "
              f"its own RMSE(y-r) {np.sqrt((res[m]**2).mean()):.0f}")
    co = (arr.MVT_TIME_UTC_mvt - arr.ARVT_3_flt).dt.total_seconds()
    print(f"A(c) arrivals: |MVT - ARVT_3| <= 60 s on {(co.abs()<=60).mean()*100:.1f} % of rows; "
          f"per-airport median {arr.groupby('ADES_mvt')[ 'MVT_TIME_UTC_mvt'].size().pipe(lambda _: '')}"
          f"{arr.assign(co=co).groupby('ADES_mvt')['co'].median().round(0).to_dict()}")


if __name__ == "__main__":
    dep = load_all_dep()
    s1_tail(dep); s3_ghost(dep); s4_sched(dep)
    del dep
    s5_screens()
