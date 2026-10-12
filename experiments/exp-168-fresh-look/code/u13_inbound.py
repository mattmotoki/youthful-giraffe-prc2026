"""exp-168 unit 13, item A: the inbound aircraft at the stand as an upper bound of the label (no fit).
The aircraft of a departure came in at its stand before it left: BLOCK >= in-block of the inbound, so
label <= g = take-off - in-block. The arrival rows keep their in-block time in the ranking file.
Link (label-free): the last arrival at the stand before the take-off with the same aircraft type and the same airline
prefix, with no other departure of the stand between the in-block time and the take-off. A same-type arrival can also
be the next aircraft (it came in after we left: label >= g, the C35 case), so the read is by band of g and by tier:
  tier 1 = the inbound came from our destination (a return flight), tier 2 = the other linked rows.
LIRF is left out (default in-block times; units 1 and 4).
  1. 2025, all months: link rate, the time at the stand (g - label) by airport, tier and band of g; hold rate;
  2. fold: rows where the entry (held stack4) or B is above the bound g - c; G of the clip;
  3. 2026: rows where stack4 is above the bound; distance;
  4. rows with a plan and a take-off more than 2 h after LOBT (C67), and the EHAM long-taxi days.
Usage: u13_inbound.py <fold5.parquet>  > checks/u13_inbound.out"""
import sys
import numpy as np, pandas as pd
from u11_lib import raw2025, rank2026, secs, AP, ROOT
from u9_lib import frame, boot

S4 = f"{ROOT}/experiments/exp-169-stack-plan"; W = 3606.0


def inbound(d):
    dd = d[d.STAND_mvt.notna() & (d.ap != "LIRF")]
    dep = dd[dd.PHASE_mvt == "DEP"].sort_values("MVT_TIME_UTC_mvt").copy(); dep["prev_dep"] = dep.groupby(["ap", "STAND_mvt"]).MVT_TIME_UTC_mvt.shift()
    arr = dd[(dd.PHASE_mvt == "ARR") & dd.BLOCK_TIME_UTC_mvt.notna()]
    a = arr[["ap", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "pre", "ADEP_mvt", "FLIGHT_mvt"]].rename(columns={"BLOCK_TIME_UTC_mvt": "a_t", "AIRCRAFT_TYPE_mvt": "a_type", "pre": "a_pre", "ADEP_mvt": "a_from", "FLIGHT_mvt": "a_fl"}).sort_values("a_t"); a["k"] = a.a_t
    m = pd.merge_asof(dep, a, left_on="MVT_TIME_UTC_mvt", right_on="k", by=["ap", "STAND_mvt"], direction="backward", allow_exact_matches=False)
    m["g"] = secs(m.MVT_TIME_UTC_mvt, m.a_t); m["link"] = m.a_t.notna() & (m.a_type == m.AIRCRAFT_TYPE_mvt) & (m.a_pre == m.pre) & (m.prev_dep.isna() | (m.a_t > m.prev_dep)) & (m.g < 43200)
    m["tier"] = np.where(m.a_from == m.ADES_mvt, 1, 2); m["dl"] = secs(m.MVT_TIME_UTC_mvt, m.LOBT_flt); m["ds"] = secs(m.MVT_TIME_UTC_mvt, m.SCHED_TIME_UTC_mvt); m["tn"] = secs(m.MVT_TIME_UTC_mvt, m.AOBT_3_flt)
    return m


d = raw2025(); r = rank2026(); n26 = int((r.PHASE_mvt == "DEP").sum())
m5 = inbound(d); m6 = inbound(r); m5["stay"] = m5.g - m5.y                       # time at the stand of the linked aircraft; under 0: the linked arrival is the next aircraft
gb = [0, 1200, 1800, 2400, 3000, 3600, 5400, 7200, 43200]
print("## 1. 2025, all months (no LIRF)")
t = m5.groupby("ap").agg(dep=("link", "size"), linked=("link", "sum")); t["share"] = (t.linked / t.dep).round(3); t6 = m6.groupby("ap").agg(dep26=("link", "size"), linked26=("link", "sum")); t6["share26"] = (t6.linked26 / t6.dep26).round(3); print(t.join(t6).T.to_string())
L = m5[m5.link].copy(); L["gband"] = pd.cut(L.g, gb)
q = lambda v: pd.Series(dict(rows=len(v), fail=float((v.stay < -6).mean()), under_10min=float((v.stay < 600).mean()), q01=v.stay.quantile(.01), q05=v.stay.quantile(.05), q25=v.stay.quantile(.25), med=v.stay.median()))
print("time at the stand (g - label) by band of g: rows, share with the label above g (the linked arrival is the next aircraft), share with under 10 min at the stand, quantiles")
for tier in (1, 2): print(f" tier {tier}:"); print(L[L.tier == tier].groupby("gband", observed=True).apply(q).round(3).to_string())
print("by airport, g of 2,400 s and more:"); print(L[L.g >= 2400].groupby(["ap", "tier"]).apply(q).round(3).to_string())
print("by plan, g of 2,400 s and more:"); print(L[L.g >= 2400].groupby(["plan", "tier"]).apply(q).round(3).to_string())

print("\n## 2. fold (January and July 2025): the clip min(entry, g - c) on linked rows with g of 2,400 s and more")
f = frame(sys.argv[1]); h = pd.read_parquet(f"{S4}/results/stack4-held-014.parquet").set_index("MVT_ID_mvt"); f["P"] = h.stack4.reindex(f.MVT_ID_mvt.astype("int64")).to_numpy()
k = m5[m5.link].set_index(m5[m5.link].MVT_ID_mvt.astype("int64")); f["g"] = k.g.reindex(f.MVT_ID_mvt.astype("int64")).to_numpy(); f["tier"] = k.tier.reindex(f.MVT_ID_mvt.astype("int64")).to_numpy()
y = f.y.to_numpy()
for base in ("P", "B"):
    Pv = f[base].to_numpy()
    for c in (0, 600, 900, 1200):
        for nm_, sel in (("all linked", f.g.notna()), ("tier 1", f.tier == 1), ("no EHAM", f.g.notna() & (f.airport != "EHAM")), ("rows without a plan", f.g.notna() & ~f.nm)):
            s = (sel & (f.g >= 2400) & (f[base] > f.g - c)).to_numpy(); v = Pv.copy(); v[s] = np.maximum(f.g.to_numpy()[s] - c, 60); G, lo, hi, g5 = boot(f, (Pv - y) ** 2 - (v - y) ** 2)
            print(f"  {base}, c = {c}, {nm_}: {int(s.sum())} rows above the bound; G {G:+.1f} [{lo:+.1f}, {hi:+.1f}], {g5:+.1f} without the 5 largest airport-days; rows that gain {int((((Pv - y) ** 2 - (v - y) ** 2)[s] > 0).sum())}; distance {float((f.w26.to_numpy() * (Pv - v) ** 2).sum()):.1f}")
s = (f.g.notna() & (f.g >= 2400) & (f.P > f.g - 600)).to_numpy(); print("the fold rows where the held stack4 is above g - 600 s (airport, flight, plan, g, label, stack4, B):")
print(f[s].assign(Pr=f.P.round(0), Br=f.B.round(0))[["airport", "FLIGHT_mvt", "nm", "tier", "g", "y", "Pr", "Br", "cls"]].sort_values("g").to_string(index=False))

print("\n## 3. 2026 (label-free): stack4 against the bound")
s4 = pd.read_parquet(f"{S4}/exp169-stack4-mlp-bag6-ext-aobt-c55.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt; m6["P"] = s4.reindex(m6.MVT_ID_mvt.astype(float)).to_numpy(); L6 = m6[m6.link & (m6.g >= 2400)].copy()
for c in (0, 600, 900, 1200):
    z = L6[L6.P > L6.g - c]; print(f"  c = {c}: stack4 above the bound on {len(z)} rows (tier 1: {int((z.tier == 1).sum())}; without a plan: {int((~z.plan).sum())}); distance {float(((z.P - (z.g - c)) ** 2).sum() / n26):.1f} s2 per ranking row; by airport {z.ap.value_counts().to_dict()}")
z = L6[L6.P > L6.g - 600].assign(mvt=lambda v: v.MVT_TIME_UTC_mvt.dt.strftime("%m-%d %H:%M")); print(z[["ap", "FLIGHT_mvt", "a_fl", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "mvt", "plan", "tier", "g", "dl", "ds", "tn", "P"]].sort_values(["ap", "mvt"]).round(0).to_string(index=False))

print("\n## 4. rows with a plan and a take-off more than 2 h after LOBT (C67): the in-block time against LOBT")
F5 = m5[m5.plan & (m5.dl > 7200) & (m5.dl < 72000)].copy(); F5["bl"] = F5.dl - F5.y; F5["in_l"] = F5.dl - F5.g       # in-block of the inbound - LOBT
print(f"2025: far rows {len(F5)}, linked {int(F5.link.sum())}; linked rows by in-block - LOBT: rows, median BLOCK - LOBT, share with BLOCK - LOBT above 1,800 s")
Fl = F5[F5.link]; print(Fl.groupby(pd.cut(Fl.in_l, [-1e6, -3600, 0, 1800, 3606, 1e6])).apply(lambda v: pd.Series(dict(rows=len(v), med_bl=v.bl.median(), late=float((v.bl > 1800).mean()), hold=float((v.bl >= v.in_l - 6).mean())))).round(2).to_string())
F6 = m6[m6.plan & (m6.dl > 7200) & (m6.dl < 72000)].copy(); F6["in_l"] = F6.dl - F6.g; F6["floor"] = F6.dl - W
print(f"2026: far rows {len(F6)}, linked {int(F6.link.sum())}; in-block after LOBT on {int((F6.link & (F6.in_l > 0)).sum())}; the bound g under stack4 + 600 s on {int((F6.link & (F6.P > F6.g - 600)).sum())}")
print("  by airport: far rows, linked:", {a_: (len(v), int(v.link.sum())) for a_, v in F6.groupby("ap")})
print("EHAM long-taxi days of January 2026 (2 to 7, 9, 29): rows without a plan, linked, stack4 above g - 600 s")
E6 = m6[(m6.ap == "EHAM") & m6.MVT_TIME_UTC_mvt.dt.strftime("%m-%d").isin(["01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-09", "01-29"])]
for nm_, v in (("without a plan", E6[~E6.plan]), ("with a plan", E6[E6.plan])):
    l = v[v.link]; print(f"  {nm_}: rows {len(v)}, linked {len(l)}, g quartiles {l.g.quantile([.25, .5, .75]).round(0).tolist()}; stack4 above g - 600 s on {int((l.P > l.g - 600).sum())}; above g - 1,200 s on {int((l.P > l.g - 1200).sum())}")
E5 = m5[(m5.ap == "EHAM") & m5.MVT_TIME_UTC_mvt.dt.strftime("%m-%d").isin(["01-05", "01-18", "11-21"]) & m5.link]
print("2025 EHAM long-taxi days, linked rows: time at the stand by plan:"); print(E5.groupby("plan").apply(q).round(3).to_string())

print("\n## 5. the other reading: the last arrival at the stand is the NEXT aircraft, so label >= g (C35 and C55 use it for another aircraft type at EDDF, EGLL, LFPG, LTFM)")
print("An own inbound has 28 min and more at the stand (section 1: 1 % quantile 1,678 s). So an arrival of the same type that came in under 20 to 25 min before the take-off is the next aircraft, at any airport.")
for nm_, m in (("2025", m5), ("2026", m6)):
    m["nodep"] = m.a_t.notna() & (m.prev_dep.isna() | (m.a_t > m.prev_dep)); m["same"] = (m.a_type == m.AIRCRAFT_TYPE_mvt) & (m.a_pre == m.pre); m["sametype"] = m.a_type == m.AIRCRAFT_TYPE_mvt
    m["kind"] = np.select([m.same, m.sametype], ["same type and airline", "same type, other airline"], "other type")
A5 = m5[m5.nodep & (m5.g < 7200)].copy(); A5["gband"] = pd.cut(A5.g, [0, 600, 900, 1200, 1500, 1800, 2400, 3600, 7200]); A5["hold"] = A5.y >= A5.g - 6
print("2025 hold rate of label >= g by kind and band of g (rows, hold):"); t = A5.groupby(["kind", "gband"], observed=True).hold.agg(["size", "mean"]).round(3); print(t.unstack(0).to_string())
print("2025 hold rate by airport, g under 1,500 s: rows, hold"); t = A5[A5.g < 1500].groupby(["ap", "kind"]).hold.agg(["size", "mean"]).round(3); print(t.unstack(1).to_string())
print("2025, same type and airline, by airport and band: hold"); print(A5[A5.kind == "same type and airline"].pivot_table(index="ap", columns="gband", values="hold", aggfunc="mean", observed=True).round(3).to_string())
k = m5[m5.nodep].set_index(m5[m5.nodep].MVT_ID_mvt.astype("int64")); ids = f.MVT_ID_mvt.astype("int64")
f["g2"] = k.g.reindex(ids).to_numpy(); f["kind2"] = k.kind.reindex(ids).to_numpy(); f["tn2"] = f.taxi_nm
C55AP = ["EDDF", "EGLL", "LFPG", "LTFM"]
def G2(sel, title, base="P"):
    Pv = f[base].to_numpy(); s = (sel & (f[base] < f.g2)).to_numpy(); v = Pv.copy(); v[s] = f.g2.to_numpy()[s]; d_ = (Pv - y) ** 2 - (v - y) ** 2; G, lo, hi, g5 = boot(f, d_)
    print(f"  {title} [{base}]: window rows {int(sel.sum())}, entry below g on {int(s.sum())}; hold of label >= g on the moved rows {float((y[s] >= f.g2.to_numpy()[s] - 6).mean()) if s.sum() else float('nan'):.3f}; G {G:+.1f} [{lo:+.1f}, {hi:+.1f}], {g5:+.1f} without the 5 largest airport-days; rows that gain {int((d_[s] > 0).sum())}")
print("fold: the clip max(entry, g)")
for base in ("P", "B"):
    for hi in (900, 1200, 1500):
        G2((f.kind2 == "same type and airline") & (f.g2 < hi), f"same type and airline, g under {hi}", base)
    G2((f.kind2 == "same type and airline") & (f.g2 < 1200) & ~f.airport.isin(C55AP), "same type and airline, g under 1,200, the five airports outside C55 (no LIRF)", base)
    G2((f.kind2 != "other type") & (f.g2 < 1200), "same type, any airline, g under 1,200", base)
    G2((f.kind2 == "other type") & (f.g2 < 7200) & f.airport.isin(C55AP), "other type at the C55 airports, g under 7,200 (C55's class, without its window)", base)
    G2((f.kind2 == "other type") & (f.g2 < 1200) & ~f.airport.isin(C55AP), "other type at the other airports, g under 1,200", base)
    G2((f.kind2 == "same type and airline") & (f.g2 >= 1200) & (f.g2 < 7200) & (f.late_l - W > f.g2 - 1500), "same type and airline, g of 1,200 s and more, join floor above g - 1,500 s (an own inbound is not possible)", base)
    G2((f.kind2 == "same type and airline") & (f.g2 >= 1200) & (f.g2 < 7200) & (f.tn2 > f.g2 - 900), "same type and airline, g of 1,200 s and more, NM off-block under 15 min after the in-block", base)
m6["P"] = s4.reindex(m6.MVT_ID_mvt.astype(float)).to_numpy(); B6 = m6[m6.nodep].copy()
print("2026: rows where stack4 is below g; distance (s2 per ranking row)")
for title, sel in (("same type and airline, g under 900", (B6.kind == "same type and airline") & (B6.g < 900)), ("same type and airline, g under 1,200", (B6.kind == "same type and airline") & (B6.g < 1200)), ("same type and airline, g under 1,500", (B6.kind == "same type and airline") & (B6.g < 1500)),
                   ("same, g under 1,200, outside the C55 airports", (B6.kind == "same type and airline") & (B6.g < 1200) & ~B6.ap.isin(C55AP)), ("other type at the C55 airports, g under 7,200", (B6.kind == "other type") & (B6.g < 7200) & B6.ap.isin(C55AP)),
                   ("same, g 1,200 and more, floor above g - 1,500", (B6.kind == "same type and airline") & (B6.g >= 1200) & (B6.g < 7200) & (B6.dl - W > B6.g - 1500)), ("same, g 1,200 and more, NM off-block under 15 min after the in-block", (B6.kind == "same type and airline") & (B6.g >= 1200) & (B6.g < 7200) & (B6.tn > B6.g - 900))):
    z = B6[sel & (B6.P < B6.g)]; print(f"  {title}: window rows {int(sel.sum())}, stack4 below g on {len(z)}; distance {float(((z.g - z.P) ** 2).sum() / n26):.1f}; by airport {z.ap.value_counts().to_dict()}; mean gap {float((z.g - z.P).mean()) if len(z) else 0:.0f} s")

print("\n## 6. C55 outside its window: another aircraft type came in at the stand (EDDF, EGLL, LFPG, LTFM), no departure of the stand between, g of 7,200 s and less")
print("C55's window (stack, stand_floor.py): with a plan g > NM taxi + 300 s and g < take-off - EOBT_1 + 900 s; without a plan g > 300 s and g < take-off - SCHED + 900 s. Parts outside it:")
def parts(m):
    m = m[m.ap.isin(C55AP) & (m.kind == "other type") & m.a_t.notna() & (m.g > 0) & (m.g <= 7200)].copy(); fe = secs(m.MVT_TIME_UTC_mvt, m.EOBT_1_flt)
    m["part"] = np.select([m.plan & m.tn.isna(), m.plan & (m.g > m.tn + 300) & (m.g < fe + 900), m.plan & (m.g <= m.tn + 300) & (m.g > m.tn), m.plan & (m.g <= m.tn), m.plan, ~m.plan & (m.g > 300) & (m.g < m.ds + 900), ~m.plan & (m.g <= 300)],
                          ["plan, no AOBT_3", "plan, in the window", "plan, in-block 0 to 300 s before AOBT_3", "plan, in-block after AOBT_3", "plan, in-block more than 900 s before EOBT_1", "no plan, in the window", "no plan, g of 300 s and less"], "no plan, in-block more than 900 s before SCHED")
    return m
p5 = parts(m5); p5["hold"] = p5.y >= p5.g - 6; p5["ph"] = secs(p5.BLOCK_TIME_UTC_mvt, p5.SCHED_TIME_UTC_mvt).abs() <= 6
print("2025, all months: rows, hold of label >= g, by part and the no-departure-between flag"); print(p5.groupby(["part", "nodep"]).hold.agg(["size", "mean"]).round(3).unstack().to_string())
print("2025, no departure between, by part and airport: hold"); print(p5[p5.nodep].pivot_table(index="part", columns="ap", values="hold", aggfunc="mean").round(3).to_string())
print("2025, no departure between, by part and band of g: hold (rows)"); p5["gband"] = pd.cut(p5.g, [0, 300, 600, 900, 1200, 1800, 3600, 7200]); t = p5[p5.nodep].groupby(["part", "gband"], observed=True).hold.agg(["size", "mean"]).round(3); print(t.unstack(1).to_string())
kp = p5.set_index(p5.MVT_ID_mvt.astype("int64")); f["part"] = kp.part.reindex(ids).to_numpy(); f["nodep"] = kp.nodep.reindex(ids).to_numpy(); f["g3"] = kp.g.reindex(ids).to_numpy(); f["pre55"] = h.stack4_pre_C55.reindex(ids).to_numpy()
print("fold: clip max(entry, g) by part (held stack4, which holds C55 in its window; pre = stack4 before C55)")
def G3(sel, title, base="P"):
    Pv = f[base].to_numpy(); s = (sel & (f[base] < f.g3)).to_numpy(); v = Pv.copy(); v[s] = f.g3.to_numpy()[s]; d_ = (Pv - y) ** 2 - (v - y) ** 2; G, lo, hi, g5 = boot(f, d_)
    print(f"  {title} [{base}]: rows {int(sel.sum())}, entry below g on {int(s.sum())}; hold on the moved rows {float((y[s] >= f.g3.to_numpy()[s] - 6).mean()) if s.sum() else float('nan'):.3f}; G {G:+.1f} [{lo:+.1f}, {hi:+.1f}], {g5:+.1f} without the 5 largest airport-days; rows that gain {int((d_[s] > 0).sum())}; mean gap {float((f.g3.to_numpy()[s] - Pv[s]).mean()) if s.sum() else 0:.0f} s")
    return s
for part in sorted(p5.part.unique()):
    G3((f.part == part) & (f.nodep == True), part + ", no departure between"); G3((f.part == part) & (f.nodep == False), part + ", a departure between")
G3((f.part == "plan, in the window"), "plan, in the window, all", "pre55"); G3((f.part == "no plan, in the window"), "no plan, in the window, all", "pre55")
s = G3(f.part.isin(["plan, in-block 0 to 300 s before AOBT_3", "plan, in-block after AOBT_3", "plan, no AOBT_3", "no plan, g of 300 s and less"]) & (f.nodep == True), "the four parts near or after the NM off-block, no departure between")
print(f[s].assign(Pr=f.P.round(0), gain=(f.w26 * ((f.P - f.y) ** 2 - (f.g3 - f.y) ** 2)).round(1))[["airport", "FLIGHT_mvt", "nm", "part", "g3", "taxi_nm", "y", "Pr", "gain", "cls"]].sort_values("gain").to_string(index=False))
p6 = parts(m6); print("2026: rows by part, rows with stack4 below g, distance (s2 per ranking row), mean gap")
for (part, nd), v in p6.groupby(["part", "nodep"]):
    z = v[v.P < v.g]; print(f"  {part}, {'no departure between' if nd else 'a departure between'}: rows {len(v)}, stack4 below g on {len(z)}, distance {float(((z.g - z.P) ** 2).sum() / n26):.1f}, mean gap {float((z.g - z.P).mean()) if len(z) else 0:.0f} s, by airport {z.ap.value_counts().to_dict()}")

print("\n## 7. the two parts that hold: (a) in-block of the other type after AOBT_3 - 300 s; (b) in-block more than 900 s before EOBT_1, g of 3,600 s and less; no departure between")
NEAR = ["plan, in-block 0 to 300 s before AOBT_3", "plan, in-block after AOBT_3"]; EARLY = "plan, in-block more than 900 s before EOBT_1"
e5 = p5[(p5.part == EARLY) & p5.nodep]; print("(b) 2025 all months by band of g: rows, hold, placeholders (BLOCK = SCHED):", {str(k_): (len(v), round(float(v.hold.mean()), 3), int(v.ph.sum())) for k_, v in e5.groupby(pd.cut(e5.g, [0, 1800, 3600, 5400, 7200]), observed=True)})
print("    by month (rows, hold):", {int(k_): (len(v), round(float(v.hold.mean()), 2)) for k_, v in e5[e5.g <= 3600].groupby("mon")}, "; failures:", [(a_, str(fl), int(g_), int(y_)) for a_, fl, g_, y_ in zip(e5[~e5.hold & (e5.g <= 3600)].ap, e5[~e5.hold & (e5.g <= 3600)].FLIGHT_mvt, e5[~e5.hold & (e5.g <= 3600)].g, e5[~e5.hold & (e5.g <= 3600)].y)])
print("    label above g on the rows that hold (g of 3,600 s and less): quartiles", (e5[e5.hold & (e5.g <= 3600)].y - e5[e5.hold & (e5.g <= 3600)].g).quantile([.25, .5, .75]).round(0).tolist())
for cap in (3600, 5400, 7200):
    s = G3((f.part == EARLY) & (f.nodep == True) & (f.g3 <= cap), f"(b) g of {cap} s and less")
s = G3((f.part == EARLY) & (f.nodep == True) & (f.g3 <= 3600), "(b) g of 3,600 s and less, on stack4 before C55", "pre55"); s = G3((f.part == EARLY) & (f.nodep == True) & (f.g3 <= 3600), "(b) on B", "B")
s = G3((f.part == EARLY) & (f.nodep == True) & (f.g3 <= 3600), "(b) again, for the row list")
print(f[s].assign(Pr=f.P.round(0), gain=(f.w26 * ((f.P - f.y) ** 2 - (f.g3 - f.y) ** 2)).round(1), e1=secs(f.MVT_TIME_UTC_mvt, f.EOBT_1_flt))[["airport", "FLIGHT_mvt", "month", "g3", "e1", "taxi_nm", "late_l", "y", "Pr", "gain", "cls"]].sort_values("gain").to_string(index=False))
G3((f.part.isin(NEAR) & (f.nodep == True)) | ((f.part == EARLY) & (f.nodep == True) & (f.g3 <= 3600)), "(a) and (b) together")
for title, sel in (("(a)", p6.part.isin(NEAR) & p6.nodep), ("(b) g of 3,600 s and less", (p6.part == EARLY) & p6.nodep & (p6.g <= 3600)), ("(b) g above 3,600 s", (p6.part == EARLY) & p6.nodep & (p6.g > 3600))):
    z = p6[sel & (p6.P < p6.g)]; print(f"2026 {title}: rows {int(sel.sum())}, stack4 below g on {len(z)}, distance {float(((z.g - z.P) ** 2).sum() / n26):.1f}")
    if len(z) <= 25: print(z.assign(mvt=z.MVT_TIME_UTC_mvt.dt.strftime("%m-%d %H:%M"), e1=secs(z.MVT_TIME_UTC_mvt, z.EOBT_1_flt))[["ap", "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "a_type", "mvt", "g", "e1", "tn", "dl", "P"]].round(0).to_string(index=False))

print("\n## 8. the same aircraft type as the next aircraft, at the airports and bands with a 2025 hold of 0.99 and more (EGLL under 1,800 s, LFPG under 1,200 s, LTFM under 1,200 s, EDDF under 600 s)")
lim = {"EGLL": 1800, "LFPG": 1200, "LTFM": 1200, "EDDF": 600}
x5 = m5[m5.nodep & m5.sametype & m5.ap.isin(list(lim)) & (m5.g < m5.ap.map(lim))]; print("2025 all months: rows", len(x5), "; hold", round(float((x5.y >= x5.g - 6).mean()), 4), "; by airport", {a_: (len(v), round(float((v.y >= v.g - 6).mean()), 4)) for a_, v in x5.groupby("ap")})
f["same2"] = k.sametype.reindex(ids).to_numpy(); f["g3"] = f.g2; sel = (f.same2 == True) & f.airport.isin(list(lim)) & (f.g2 < f.airport.map(lim).astype(float)); G3(sel, "fold")
x6 = B6[B6.sametype & B6.ap.isin(list(lim)) & (B6.g < B6.ap.map(lim))]; z = x6[x6.P < x6.g]; print(f"2026: rows {len(x6)}, stack4 below g on {len(z)}, distance {float(((z.g - z.P) ** 2).sum() / n26):.1f}, by airport {z.ap.value_counts().to_dict()}")
