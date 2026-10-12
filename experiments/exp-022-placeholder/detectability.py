"""Why the detector's posterior cannot climb, and what would fix it.

Two candidate repairs, both measured before spending fits on them:
  A. a deterministic high-p subset from anchor coincidences
  B. day-level structure the detector cannot see
"""
import numpy as np, pandas as pd, prc.data as D

dep = D.load_month("2025-07").query("PHASE_mvt == 'DEP'")
blk, sch, lobt, aobt = (dep.BLOCK_TIME_UTC_mvt, dep.SCHED_TIME_UTC_mvt,
                        dep.LOBT_flt, dep.AOBT_3_flt)
mn = lambda x: x.dt.floor("min")
same = (mn(blk) == mn(sch))
sel = dep.AIRCRAFT_TYPE_mvt.notna()          # non-noac: noac has its own head
base = same[sel].mean()
print(f"non-noac july departures {int(sel.sum()):,}, class rate {base*100:.2f}%\n")

print("A. is there a deterministic subset?")
for name, cond in (("LOBT == SCHED", mn(lobt) == mn(sch)),
                   ("AOBT missing", aobt.isna()),
                   ("AOBT == LOBT", mn(aobt) == mn(lobt)),
                   ("AOBT == SCHED", mn(aobt) == mn(sch)),
                   ("AOBT == LOBT == SCHED", (mn(aobt) == mn(lobt)) & (mn(lobt) == mn(sch)))):
    c = cond & sel
    print(f"   {name:24s} n={int(c.sum()):7,}  P(class)={same[c].mean()*100:6.2f}%  "
          f"lift={same[c].mean()/base:5.2f}")
print("   -> best is 14.6 % at lift 3.7. No deterministic subset exists.\n")

print("B. how much could day-level information add?")
d = pd.DataFrame({"apt": dep.ADEP_mvt[sel], "day": dep.MVT_TIME_UTC_mvt[sel].dt.floor("D"),
                  "c": same[sel].astype(int)})
g = d.groupby(["apt", "day"])["c"].agg(["mean", "size"])
g = g[g["size"] >= 30]
obs = np.average((g["mean"] - base) ** 2, weights=g["size"])
exp = np.average(base * (1 - base) / g["size"], weights=g["size"])
d["adm"] = d.groupby(["apt", "day"])["c"].transform("mean")
d["am"] = d.groupby("apt")["c"].transform("mean")
print(f"   airport-day rate variance: observed {obs:.5f} vs binomial {exp:.5f} "
      f"({obs/exp:.1f}x over-dispersed)")
print(f"   between-airport-day share of ROW variance : {d.adm.var()/d.c.var()*100:.1f}%")
print(f"   of which between-AIRPORT (have it in 2026): {d.am.var()/d.c.var()*100:.1f}%")
print(f"   day residual (no labels in 2026, so lost) : "
      f"{(d.adm.var()-d.am.var())/d.c.var()*100:.1f}%")
print("   -> the class is over-dispersed by day but that is 1 % of row variance,")
print("      so day-level information is NOT the missing piece. The ceiling is")
print("      a 3.9 % base rate against an AUC-0.86 ranking: the p-rule, not a bug.")
