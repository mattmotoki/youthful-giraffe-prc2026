"""exp-165 check 35 (bound-check of C73 and C74, `rules` exp-171 unit 2), my own code. No fit.
LIRF departures; late = take-off - SCHED >= 1,200 s, or no plan; placeholder = |BLOCK - SCHED| <= 6 s; route = airline
(first 3 characters of the flight string) x destination. Route set: rate >= 0.98 on >= 5 late rows (of the months that
choose it). 1. the routes of 2025 (all months) and their airline rates; 2. quarter stability (a route >= 0.98 in a
quarter, its rate in the next quarter); 3. fold G of the rule 'MVT - SCHED' on the late rows of the routes chosen on the
ten other months, on B; 4. 2026: late rows on the routes chosen on all of 2025, stack4 distance from MVT - SCHED by route;
5. C74's cells on long delays (take-off - SCHED >= 4,500 s): U = plan and |LOBT - SCHED| > 3,612 s (a real block for
certain), A = no plan, S = the other rows with a plan; 2025: placeholders in U; 2026 counts on the chosen routes."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prc import data as D  # noqa: E402

spec = importlib.util.spec_from_file_location("rho_screen", ROOT / "experiments/exp-165-levers/code/rho_screen.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
ID = "MVT_ID_mvt"
N = rs.N_RANKING
COLS = [ID, "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt"]
sec = lambda a, b: (a - b).dt.total_seconds()


def lirf(x):
    d = x[(x.PHASE_mvt == "DEP") & (x.ADEP_mvt == "LIRF")].drop_duplicates(ID).copy()
    d["al"] = d.FLIGHT_mvt.astype(str).str[:3]
    d["route"] = d.al + " " + d.ADES_mvt.astype(str)
    d["ms"] = sec(d.MVT_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt)
    d["plan"] = d.LOBT_flt.notna()
    d["late"] = (d.ms >= 1200) | ~d.plan
    d["P"] = d.ms.clip(lower=60)
    d["mon"] = d.MVT_TIME_UTC_mvt.dt.month
    ls = sec(d.LOBT_flt, d.SCHED_TIME_UTC_mvt).abs()
    d["cell"] = np.where(~d.plan, "A", np.where(ls > 3612, "U", "S"))
    return d


def chosen(v, min_rows=5, rate=0.98):
    g = v.groupby("route").ph.agg(["size", "mean"])
    return g[(g["size"] >= min_rows) & (g["mean"] >= rate)]


y = lirf(pd.concat([D.load_month(f"2025-{m:02d}", columns=COLS) for m in range(1, 13)]))
y = y[y.MVT_TIME_UTC_mvt.dt.year == 2025]
y["y"] = sec(y.MVT_TIME_UTC_mvt, y.BLOCK_TIME_UTC_mvt)
y["ph"] = (y.y - y.ms).abs() <= 6
late = y[y.late]
al = late.groupby("al").ph.agg(["size", "mean"])
R = chosen(late)
print(f"## 1. 2025 LIRF late rows {len(late):,}; placeholder share {late.ph.mean():.3f}; routes >= 0.98 on >= 5 late rows: {len(R)}")
print(R.assign(airline_rate=[round(al.loc[r.split()[0], 'mean'], 3) for r in R.index]).sort_values("size", ascending=False).round(3).to_string())
print(f"precision on the chosen routes: {int(late[late.route.isin(R.index)].ph.sum())} of {int(late.route.isin(R.index).sum())}")

print("\n## 2. quarter stability: a route >= 0.98 on >= 5 late rows in a quarter, its rate in the next quarter (>= 3 late rows)")
late = late.assign(q=(late.mon - 1) // 3)
nxt = []
for q in range(3):
    a = chosen(late[late.q == q])
    b = late[(late.q == q + 1) & late.route.isin(a.index)].groupby("route").ph.agg(["size", "mean"])
    b = b[b["size"] >= 3]
    nxt += [(q, r, round(m, 3), int(n)) for r, (n, m) in b.iterrows()]
nm = np.array([m for _, _, m, _ in nxt])
print(f"pairs {len(nxt)}; mean next-quarter rate {nm.mean():.3f}; share >= 0.9: {np.mean(nm >= 0.9):.2f}; below 0.9: {[x for x in nxt if x[2] < 0.9]}")

print("\n## 3. fold on B: routes chosen on the ten other months; rule MVT - SCHED on their late rows")
f = rs.fold_frame()
B, _ = rs.stack_vector(f, rs.B_FORM, rs.B_SHA)
pos = pd.Series(np.arange(len(f)), index=f[ID].to_numpy("int64"))
yv, wv = f.y.to_numpy(), f.w26.to_numpy()
R10 = chosen(late[~late.mon.isin([1, 7])])
fz = late[late.mon.isin([1, 7]) & late.route.isin(R10.index)]
fz = fz[fz[ID].astype("int64").isin(pos.index)]
p = pos.reindex(fz[ID].astype("int64")).to_numpy().astype(int)
g = wv[p] * ((B[p] - yv[p]) ** 2 - (fz.P.to_numpy() - yv[p]) ** 2)
pl = ((B[p] - yv[p]) ** 2 - (fz.P.to_numpy() - yv[p]) ** 2)
print(f"routes chosen on the ten other months: {len(R10)}; fold late rows on them {len(p)} (placeholders {int(fz.ph.sum())}); differ from B by > 60 s: "
      f"{int((np.abs(B[p] - fz.P.to_numpy()) > 60).sum())}; G {g.sum():+.1f}; plain sum {pl.sum() / len(f) * 1:+.2f} per fold row; by route G "
      f"{pd.Series(g, index=fz.route.to_numpy()).groupby(level=0).sum().round(1).sort_values(ascending=False).head(8).to_dict()}")
print(f"  without no-plan rows {g[fz.plan.to_numpy()].sum():+.1f}; no-plan rows {g[~fz.plan.to_numpy()].sum():+.1f}; by month {pd.Series(g, index=fz.mon.to_numpy()).groupby(level=0).sum().round(1).to_dict()}")

print("\n## 4. 2026: routes chosen on all of 2025")
r = lirf(pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=COLS))
s4 = pd.read_parquet(ROOT / "experiments/exp-169-stack-plan/exp169-stack4-mlp-bag6-ext-aobt-c55.parquet").set_index(ID).TAXITIME_SEC_mvt
r["E"] = s4.reindex(r[ID]).to_numpy("float64")
z = r[r.late & r.route.isin(R.index)].copy()
z["d"] = (z.E - z.P) ** 2 / N
t = z.groupby("route").agg(rows=("d", "size"), distance=("d", "sum"), E=("E", "mean"), P=("P", "mean"), months=("mon", lambda v: sorted(set(v))))
print(f"2026 late rows on the chosen routes: {len(z)} on {z.route.nunique()} routes; distance of stack4 from MVT - SCHED {z.d.sum():.1f}")
print(t.sort_values("distance", ascending=False).round(1).to_string())

print("\n## 5. C74's cells on long delays (take-off - SCHED >= 4,500 s)")
ld = y[y.ms >= 4500]
print(f"2025 LIRF long delays {len(ld):,}: placeholders in U {int((ld.ph & (ld.cell == 'U')).sum())} of {int((ld.cell == 'U').sum())}; "
      f"share of placeholders in A {ld[ld.ph].cell.eq('A').mean():.2f}; real rows in U {ld[~ld.ph].cell.eq('U').mean():.2f}")
ld26 = r[(r.ms >= 4500) & r.route.isin(R.index)]
print("2026 chosen routes, long delays by cell:", ld26.groupby(["route", "cell"]).size().unstack(fill_value=0).to_dict("index"))

print("\n## 6. 2026 concentration: the WMT routes LEIB and LGKR (78 % of the distance)")
for rt in ("WMT LEIB", "WMT LGKR"):
    w = z[z.route == rt].sort_values("d", ascending=False)
    print(f"{rt}: rows {len(w)}; distance {w.d.sum():.1f}; top 5 rows {w.d.head(5).sum():.1f}; by day {w.groupby(w.MVT_TIME_UTC_mvt.dt.strftime('%m-%d')).d.sum().round(1).to_dict()}; "
          f"by cell {w.groupby('cell').d.sum().round(1).to_dict()}; E - P median {np.median(w.E - w.P):.0f} s")
wl = r[(r.al == "WMT")]
print("2026 WMT LIRF departures by month and cell (all routes, late rows):", wl[wl.late].groupby(["mon", "cell"]).size().unstack(fill_value=0).to_dict("index"))
print("2025 WMT LIRF late rows by quarter and cell, chosen WMT routes / other WMT routes:")
wy = late[late.al == "WMT"].assign(ch=lambda v: v.route.isin(R.index))
print(wy.groupby(["ch", "q", "cell"]).size().unstack(fill_value=0).to_string())
d0 = z.d.sum()
for rt in ("WMT LEIB", "WMT LGKR", "WMT HESX"):
    print(f"if {rt} flips to real blocks (label at stack4): 2026 gain {d0 - 2 * z[z.route == rt].d.sum():+.1f}")
