"""exp-168 unit 12: mirror rows. A LIRF departure row whose destination is LIRF is a copy of an arrival record: an arrival
row of the same flight string with the same movement time, SCHED and stand exists. Its block time is the airport's
default: SCHED when SCHED is before the movement time, else the movement time - 600 s. So the label is label-free:
take-off - SCHED when that is positive (above 30 s: the times carry a dither of 6 s), else 600 s.
  1. census: departures with ADES = the airport, with and without a mirror arrival row, by airport, 2025 and 2026;
  2. LIRF 2025: hold rate of the rule, by month and by plan; the rows that do not follow it;
  3. the same default on all LIRF rows: the share of the label 600 s by take-off - SCHED (why the rule has two cases);
  4. fold on base B;
  5. 2026 (label-free): the rows, the arrival side of each mirror (its block time is in the ranking file), the stack's
     vector against the rule; a forecast with the 2025 hold pattern;
  6. departures with ADES = the airport at the other airports (no mirror row there): a constant from the other months.
Usage: u12_mirror.py <fold5.parquet>  > checks/u12_mirror.out"""
import sys
import numpy as np, pandas as pd
from u11_lib import raw2025, rank2026, secs, near, AP, ROOT
from u9_lib import frame, boot, compose

d = raw2025(); r = rank2026()


def mirror(x):
    """Departure rows with ADES = the airport; mir = an arrival row at the airport from the airport itself with the same
    flight string (or both missing) and a movement time within 12 s; the arrival's SCHED and block time are kept."""
    dep = x[(x.PHASE_mvt == "DEP") & (x.ADES_mvt == x.ap)].copy(); arr = x[(x.PHASE_mvt == "ARR") & (x.ADEP_mvt == x.ap)].copy()
    dep["fs"] = dep.FLIGHT_mvt.fillna("?"); arr["fs"] = arr.FLIGHT_mvt.fillna("?")
    a = arr[["ap", "fs", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt"]].rename(columns={"MVT_TIME_UTC_mvt": "a_mvt", "BLOCK_TIME_UTC_mvt": "a_block", "SCHED_TIME_UTC_mvt": "a_sched", "STAND_mvt": "a_stand"})
    m = dep.merge(a, on=["ap", "fs"], how="left"); m["gap"] = secs(m.a_mvt, m.MVT_TIME_UTC_mvt).abs(); m = m.sort_values("gap").drop_duplicates("MVT_ID_mvt")
    m["mir"] = m.gap <= 12; m["ds"] = secs(m.MVT_TIME_UTC_mvt, m.SCHED_TIME_UTC_mvt); m["rule"] = np.where(m.ds > 30, m.ds, 600.0)
    m["a_in"] = secs(m.a_block, m.a_mvt); m["a_ds"] = secs(m.a_sched, m.a_mvt)                         # arrival: in-block - landing; SCHED - landing
    m["a_rule"] = np.where(m.a_ds > 30, m.a_ds, 600.0); m["a_ok"] = (m.a_in - m.a_rule).abs() <= 24
    m["lob"] = secs(m.LOBT_flt, m.SCHED_TIME_UTC_mvt); m["own"] = m.plan & (m.lob >= -600)                # a plan of the row's own time: a real flight
    return m.sort_values("MVT_TIME_UTC_mvt")


m25, m26 = mirror(d), mirror(r)
print("## 1. departures with ADES = the airport: rows, rows with a mirror arrival row, median label; 2026 rows and mirrors")
g = m25.groupby("ap").agg(rows=("y", "size"), mirror=("mir", "sum"), plan=("plan", "mean"), med_y=("y", "median")); g2 = m26.groupby("ap").agg(rows26=("mir", "size"), mirror26=("mir", "sum")); print(g.join(g2).round(2).to_string())
print("mirror rows by airport and month, 2025:"); print(m25[m25.mir].groupby(["ap", "mon"]).size().unstack(fill_value=0).to_string())
for ap in AP:
    z = m25[(m25.ap == ap) & m25.mir]
    if len(z) >= 5: print(f"  {ap}: mirror rows {len(z)}; label within 24 s of the rule on {np.mean((z.y - z.rule).abs() <= 24):.3f}; the arrival side follows its rule on {z.a_ok.mean():.3f}; same stand {np.mean(z.STAND_mvt.astype(str) == z.a_stand.astype(str)):.3f}; same SCHED {np.mean(z.SCHED_TIME_UTC_mvt == z.a_sched):.3f}")

print("\n## 2. LIRF 2025: the rule label = take-off - SCHED when above 30 s, else 600 s")
L = m25[m25.ap == "LIRF"].copy(); L["err"] = L.y - L.rule; L["ok"] = L.err.abs() <= 24
print(f"rows {len(L)}; with a mirror {int(L.mir.sum())}; rule within 24 s on {int(L.ok.sum())} ({L.ok.mean():.3f}); placeholder case (take-off after SCHED) {int((L.ds > 30).sum())} rows, ok {L.ok[L.ds > 30].mean():.3f}; 600 s case {int((L.ds <= 30).sum())} rows, ok {L.ok[L.ds <= 30].mean():.3f}")
print("by month: rows, ok:", {int(k): (len(v), int(v.ok.sum())) for k, v in L.groupby("mon")}); print("by plan:", {bool(k): (len(v), int(v.ok.sum())) for k, v in L.groupby("plan")})
print(f"error of the rule: RMS {np.sqrt(np.mean(L.err ** 2)):.0f} s; a constant 600 s: {np.sqrt(np.mean((L.y - 600) ** 2)):.0f} s; all other LIRF departures of 2025 have a median label of {d[(d.PHASE_mvt == 'DEP') & (d.ap == 'LIRF')].y.median():.0f} s")
print("rows that do not follow the rule:"); print(L[~L.ok][["FLIGHT_mvt", "plan", "RUNWAY_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "ds", "y", "rule", "mir"]].to_string(index=False))
print("rows with a plan: LOBT - SCHED, label, rule:", [(int(a), int(b), int(c)) for a, b, c in zip(L.lob[L.plan], L.y[L.plan], L.rule[L.plan])])
print("rows with a plan: ADEP of the plan, own flag:", [(str(a), bool(b)) for a, b in zip(L.ADEP_flt[L.plan], L.own[L.plan])])
print("by the plan class (own = a plan with LOBT at or after SCHED - 600 s): rows, ok:", {bool(k): (len(v), int(v.ok.sum())) for k, v in L.groupby("own")})
print("by the arrival side (a_ok = the mirror arrival row has its default in-block time): rows, ok:", {bool(k): (len(v), int(v.ok.sum())) for k, v in L.groupby("a_ok")})
print("by take-off - SCHED, rows without an own plan: rows, ok:", {str(k): (len(v), int(v.ok.sum())) for k, v in L[~L.own].groupby(pd.cut(L.ds[~L.own], [-1e9, 30, 1800, 1e9]), observed=True)})
K = L[~L.own]; print(f"rule set (no own plan): {len(K)} rows, ok {int(K.ok.sum())} ({K.ok.mean():.3f}); RMS error of the rule {np.sqrt(np.mean(K.err ** 2)):.0f} s")
for c in (1025.0, 1650.0):
    for nm, z in (("no own plan", K), ("no own plan, take-off - SCHED at most 1,800 s", K[K.ds <= 1800])):
        print(f"  proxy base = constant {c:.0f} s, {nm}: {len(z)} rows; share of the squared distance base-rule that is gain {float(((c - z.y) ** 2 - (z.rule - z.y) ** 2).sum() / ((c - z.rule) ** 2).sum()):.2f}")

print("\n## 3. all LIRF departures 2025: share of the label 600 s (+-12 s) by take-off - SCHED")
x = d[(d.PHASE_mvt == "DEP") & (d.ap == "LIRF")].copy(); x["ds"] = secs(x.MVT_TIME_UTC_mvt, x.SCHED_TIME_UTC_mvt); x["k"] = near(x.y, 600, 12)
print(x.groupby(pd.cut(x.ds, [-1e9, -600, -60, 60, 600, 1200, 3600, 1e9]), observed=True).k.agg(["size", "mean"]).round(3).T.to_string())

print("\n## 4. fold on base B")
f = frame(sys.argv[1]); N = len(f); Bv, yv = f.B.to_numpy(), f.y.to_numpy(); fm = m25[(m25.ap == "LIRF") & m25.mon.isin([1, 7])]
pos = pd.Series(np.arange(N), index=f.MVT_ID_mvt.to_numpy()).reindex(fm.MVT_ID_mvt.astype("int64")).to_numpy(); v = Bv.copy(); v[pos] = fm.rule.to_numpy()
g_, lo, hi, g5 = boot(f, (Bv - yv) ** 2 - (v - yv) ** 2)
print(pd.DataFrame(dict(flight=fm.FLIGHT_mvt.to_numpy(), ds=fm.ds.to_numpy(), y=yv[pos], B=Bv[pos].round(0), rule=fm.rule.to_numpy(), w=(f.w26.to_numpy()[pos] * N).round(2))).to_string(index=False))
print(f"fold: {len(fm)} rows; G of the rule on B {g_:+.1f} [{lo:+.1f}, {hi:+.1f}]; mass of these rows in B {float(f.gB.to_numpy()[pos].sum()):.1f}")

print("\n## 5. 2026 (label-free)")
rb = pd.read_parquet(f"{ROOT}/experiments/exp-169-stack-plan/inputs/unit2-ready/bundle-ranking.parquet"); rb["B26"] = compose(rb, route="R4mk", noplan="v3cal", practice="selected"); rb = rb.set_index("MVT_ID_mvt")
Z = m26[m26.ap == "LIRF"].copy(); Z["B"] = rb.B26.reindex(Z.MVT_ID_mvt).to_numpy(); n26 = int((r.PHASE_mvt == "DEP").sum())
print(Z[["mon", "FLIGHT_mvt", "plan", "RUNWAY_mvt", "STAND_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "ds", "lob", "own", "mir", "a_in", "a_rule", "a_ok", "rule", "B"]].round(0).to_string(index=False))
print(f"rows {len(Z)} (January {int((Z.mon == 1).sum())}, July {int((Z.mon == 7).sum())}); with a mirror arrival row {int(Z.mir.sum())}; the arrival side has its default (in-block = SCHED when after the landing, else landing + 600 s) on {int(Z.a_ok[Z.mir].sum())} of {int(Z.mir.sum())} (2025: {int(L.a_ok[L.mir].sum())} of {int(L.mir.sum())})")
dist = lambda z: float(((z.B - z.rule) ** 2).sum() / n26)
for nm, z in (("all rows", Z), ("no own plan", Z[~Z.own]), ("no own plan, take-off - SCHED at most 1,800 s", Z[~Z.own & (Z.ds <= 1800)]), ("the same, 600 s case", Z[~Z.own & (Z.ds <= 30)]), ("the same, placeholder case", Z[~Z.own & (Z.ds > 30) & (Z.ds <= 1800)]),
              ("no own plan, take-off - SCHED at most 1,800 s, arrival side at its default", Z[~Z.own & (Z.ds <= 1800) & Z.a_ok])):
    print(f"  {nm}: {len(z)} rows; mean of the stack {z.B.mean():.0f} s, of the rule {z.rule.mean():.0f} s; the stack is above the rule on {int((z.B > z.rule).sum())}; distance {dist(z):.1f} s2 per ranking row")
# forecast: each 2026 row holds with the 2025 rate p; a row that does not hold has an ordinary label (the 2025 rows that did not hold: 351 to 1,198 s; 800 s here)
z = Z[~Z.own & (Z.ds <= 1800)]; kk = K[K.ds <= 1800]; p = kk.ok.mean(); miss = 800.0
gain = lambda p: float((p * (z.B - z.rule) ** 2 + (1 - p) * ((z.B - miss) ** 2 - (z.rule - miss) ** 2)).sum() / n26)
print(f"forecast on the {len(z)} rows (no own plan, take-off - SCHED at most 1,800 s): 2025 hold rate {int(kk.ok.sum())} of {len(kk)} = {p:.3f} -> {gain(p):+.1f}; at a hold rate of 0.80: {gain(0.8):+.1f}; of 0.60: {gain(0.6):+.1f}; break-even hold rate {next((q / 100 for q in range(0, 101) if gain(q / 100) > 0), 1.0):.2f}")
print("2026 rows with a plan: ADEP of the plan, LOBT - SCHED, own flag:", [(str(a), int(c), bool(b)) for a, c, b in zip(Z.ADEP_flt[Z.plan], Z.lob[Z.plan], Z.own[Z.plan])])
zz = Z[~Z.own & (Z.ds > 1800)]; print("rows above 1,800 s (left to the stack):", [(str(a), int(b), int(c)) for a, b, c in zip(zz.FLIGHT_mvt, zz.ds, zz.B)])

print("\n## 6. other airports: departures with ADES = the airport (no mirror row); constant = median label of the other months, by airport and plan")
allm = m25.assign(fold=m25.mon.isin([1, 7])); f["loc"] = (f.ADES_mvt == f.airport).to_numpy()
for ap in ("EHAM", "LSZH", "LEBL"):
    for pl in (False, True):
        o = allm[(allm.ap == ap) & (allm.plan == pl) & ~allm.fold]; sel = (f["loc"] & (f.airport == ap) & (f.nm == pl)).to_numpy()
        if sel.sum() < 5 or len(o) < 20: continue
        c = float(o.y.median()); v = Bv.copy(); v[sel] = c; g_, lo, hi, g5 = boot(f, (Bv - yv) ** 2 - (v - yv) ** 2)
        z6 = m26[(m26.ap == ap) & (m26.plan == pl)]; z6 = z6.assign(B=rb.B26.reindex(z6.MVT_ID_mvt).to_numpy())
        print(f"  {ap} plan={pl}: 2026 rows {len(z6)}, mean of the stack {z6.B.mean():.0f} s, distance from the constant {float(((z6.B - c) ** 2).sum() / n26):.1f} s2 per ranking row")
        print(f"  {ap} plan={pl}: other months {len(o)} rows, median label {c:.0f} s (share within 60 s of 0: {np.mean(o.y.abs() <= 60):.2f}); fold {int(sel.sum())} rows, mass in B {float(f.gB.to_numpy()[sel].sum()):.1f}, mean B {Bv[sel].mean():.0f}, median label {np.median(yv[sel]):.0f}; G of the constant {g_:+.1f} [{lo:+.1f}, {hi:+.1f}], without the 5 largest airport-days {g5:+.1f}")
