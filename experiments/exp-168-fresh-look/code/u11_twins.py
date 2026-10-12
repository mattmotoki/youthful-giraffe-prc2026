"""exp-168 unit 11: twin departure rows. One flight has two departure rows at the airport: the same flight string and
a take-off time within 120 s, and exactly one of the two rows has a plan. Seen at LFPG only (a second off-block after a
return to a stand or a stand change): the row with the plan keeps the first off-block time (a long label), the row
without a plan has the second one. The pair is label-free (flight string, take-off time, LOBT).
A wider sign of the same event: LFPG times are whole minutes (+-6 s of dither), and the first record of such a flight
has a take-off time with real seconds (filled from another source). U = P rows and rows with real seconds.
  1. census 2025 and 2026; hold rates of "the row with the plan is long";
  2. the first off-block time against the plan times (which anchor is nearest);
  3. the set U: counts, hold rates, the error of the rule label = take-off - LOBT - c (constants from the other months);
  4. fold on base B (January and July 2025): G of the rule, constants from the ten other months;
  5. 2026: the U rows and the distance of the stack's vector from the rule.
Usage: u11_twins.py <fold5.parquet>  > checks/u11_twins.out"""
import sys
import numpy as np, pandas as pd
from u11_lib import raw2025, rank2026, secs, AP, ROOT
from u9_lib import frame, boot, compose

Q = [0, .1, .25, .5, .75, .9, 1]
q = lambda s: {k: round(float(v)) for k, v in s.quantile(Q).items()}


def twins(x, tol=120):
    """Pairs of departure rows of one airport with the same flight string and a take-off within tol s, one with a plan.
    Returns one line per pair: the id of the row with the plan (P) and of the row without (T)."""
    x = x[(x.PHASE_mvt == "DEP") & x.FLIGHT_mvt.notna()].copy(); x["t"] = x.MVT_TIME_UTC_mvt.astype("int64") // 10**6
    x = x.sort_values(["ap", "FLIGHT_mvt", "t", "MVT_ID_mvt"]).reset_index(drop=True)
    same = (x.ap.eq(x.ap.shift()) & x.FLIGHT_mvt.eq(x.FLIGHT_mvt.shift()) & ((x.t - x.t.shift()) <= tol)).to_numpy()
    i = np.flatnonzero(same); a, b = x.iloc[i - 1].reset_index(drop=True), x.iloc[i].reset_index(drop=True)
    one = a.plan.to_numpy() ^ b.plan.to_numpy(); a, b = a[one].reset_index(drop=True), b[one].reset_index(drop=True); ap = a.plan.to_numpy()
    P = pd.concat([a[ap], b[~ap]]).sort_index(); T = pd.concat([b[ap], a[~ap]]).sort_index()   # row-wise choice: P has the plan
    o = pd.DataFrame(dict(ap=P.ap, mon=P.mon, flight=P.FLIGHT_mvt, idP=P.MVT_ID_mvt, idT=T.MVT_ID_mvt, tP=P.MVT_TIME_UTC_mvt, tT=T.MVT_TIME_UTC_mvt, sched=P.SCHED_TIME_UTC_mvt,
                          lobt=P.LOBT_flt, eobt=P.EOBT_1_flt, iobt=P.IOBT_flt, a3=P.AOBT_3_flt, yP=P.y, yT=T.y, bP=P.BLOCK_TIME_UTC_mvt, bT=T.BLOCK_TIME_UTC_mvt,
                          standP=P.STAND_mvt.astype(str), standT=T.STAND_mvt.astype(str), rwyP=P.RUNWAY_mvt.astype(str), rwyT=T.RUNWAY_mvt.astype(str), schedT=T.SCHED_TIME_UTC_mvt))
    for c in ("tP", "tT", "sched", "lobt", "eobt", "iobt", "a3", "bP", "bT", "schedT"): o[c] = pd.to_datetime(o[c], utc=True)
    o["dl"] = secs(o.tP, o.lobt); o["taxi_nm"] = secs(o.tP, o.a3); return o


d = raw2025(); r = rank2026(); t = twins(d); t26 = twins(r)
print("## 1. census: pairs by airport and month"); print("2025"); print(t.groupby(["ap", "mon"]).size().unstack(fill_value=0).to_string()); print("2026"); print(t26.groupby(["ap", "mon"]).size().unstack(fill_value=0).to_string())
t = t[t.ap == "LFPG"].copy(); t26 = t26[t26.ap == "LFPG"].copy()
print(f"LFPG 2025: {len(t)} pairs; same SCHED {np.mean(t.sched == t.schedT):.3f}; same stand {np.mean(t.standP == t.standT):.3f}; same runway {np.mean(t.rwyP == t.rwyT):.3f}; id of T = id of P + 1: {np.mean(t.idT - t.idP == 1):.3f}")
print("label of P (plan):", q(t.yP)); print("label of T (no plan):", q(t.yT)); print("second off-block minus first (s):", q(secs(t.bT, t.bP)))
print(f"hold rates: P longer than T {np.mean(t.yP > t.yT):.3f}; P of 1,800 s and more {np.mean(t.yP >= 1800):.3f}; P of 3,600 s and more {np.mean(t.yP >= 3600):.3f}; T under 1,800 s {np.mean(t.yT < 1800):.3f}")
dep = d[(d.PHASE_mvt == "DEP") & (d.ap == "LFPG")]; print(f"all LFPG departures 2025 with a plan: label of 1,800 s and more {np.mean(dep.y[dep.plan] >= 1800):.4f}, 3,600 s and more {np.mean(dep.y[dep.plan] >= 3600):.4f}; rows with a plan and a label of 3,600 s and more: {int((dep.y[dep.plan] >= 3600).sum())}, of them P rows: {int((t.yP >= 3600).sum())}")
print("take-off minus LOBT of P (s):", q(t.dl), "; 2026:", q(t26.dl)); print("NM taxi time of P (s):", q(t.taxi_nm), "; 2026:", q(t26.taxi_nm))

print("\n## 2. the first off-block time of P against the plan times (s; BLOCK minus the time), 2025")
for c in ("sched", "lobt", "eobt", "iobt"):
    e = secs(t.bP, t[c]); print(f"  {c:6s} {q(e)}  mean {e.mean():7.0f}  sd {e.std():6.0f}  median abs dev from the median {np.median(np.abs(e - e.median())):5.0f}")
e = secs(t.bP, t.lobt); print("  by the lateness of the plan (LOBT - SCHED):"); print(t.assign(e=e, late=pd.cut(secs(t.lobt, t.sched), [-1e9, 60, 1200, 3600, 1e9])).groupby("late", observed=True).e.describe().round(0).to_string())
print("  T: label minus the NM taxi time of P (s):", q(t.yT - t.taxi_nm), "; label of T by quantile:", q(t.yT))

print("\n## 3. the wider set U at LFPG: the P rows and every row whose take-off time has real seconds (more than 6 s from a whole minute)")


def uset(x, tw):
    z = x[(x.PHASE_mvt == "DEP") & (x.ap == "LFPG")].copy(); s = z.MVT_TIME_UTC_mvt.dt.second + z.MVT_TIME_UTC_mvt.dt.microsecond / 1e6
    z["odd"] = np.minimum(s, 60 - s) > 6.5; z["isP"] = z.MVT_ID_mvt.isin(set(tw.idP)); z["isT"] = z.MVT_ID_mvt.isin(set(tw.idT))
    u = z[z.odd | z.isP].copy(); u["dl"] = secs(u.MVT_TIME_UTC_mvt, u.LOBT_flt); u["ls"] = secs(u.LOBT_flt, u.SCHED_TIME_UTC_mvt); u["taxi_nm"] = secs(u.MVT_TIME_UTC_mvt, u.AOBT_3_flt); u["late"] = u.ls > 60; u["es"] = secs(u.EOBT_1_flt, u.LOBT_flt)
    return z, u


z, u = uset(d, t); z26, u26 = uset(r, t26)
print(f"2025: rows with real seconds {int(z.odd.sum())} (with a plan {int((z.odd & z.plan).sum())}; P rows among them {int((z.odd & z.isP).sum())}; T rows {int((z.odd & z.isT).sum())}); P rows {int(z.isP.sum())}; U {len(u)}; by month {u.groupby('mon').size().to_dict()}")
print(f"2026: rows with real seconds {int(z26.odd.sum())} (with a plan {int((z26.odd & z26.plan).sum())}; P rows among them {int((z26.odd & z26.isP).sum())}); P rows {int(z26.isP.sum())}; U {len(u26)}; by month {u26.groupby('mon').size().to_dict()}")
for nm, m in (("real seconds and P", u.odd & u.isP), ("real seconds, no twin", u.odd & ~u.isP), ("P on a whole minute", ~u.odd & u.isP), ("U", u.odd | u.isP)):
    v = u[m]; print(f"  {nm:22s} rows {len(v):3d}; label {q(v.y)}; 1,800 s and more {np.mean(v.y >= 1800):.3f}; 3,600 s and more {np.mean(v.y >= 3600):.3f}; BLOCK - LOBT median {np.median(secs(v.BLOCK_TIME_UTC_mvt, v.LOBT_flt)):.0f}")
print(f"  all LFPG rows with a plan: 3,600 s and more on {int((z.y[z.plan] >= 3600).sum())} rows, of them in U {int((u.y >= 3600).sum())}; 1,800 s and more on {int((z.y[z.plan] >= 1800).sum())}, in U {int((u.y >= 1800).sum())}")
print("  BLOCK - LOBT of U by the lateness of the plan (LOBT - SCHED):"); print(u.assign(e=secs(u.BLOCK_TIME_UTC_mvt, u.LOBT_flt), band=pd.cut(u.ls, [-1e9, 60, 1200, 3600, 1e9])).groupby("band", observed=True).e.describe().round(0).to_string())


def rule(o, h, form):
    """Label of the U rows h with constants from the U rows o. one: take-off - LOBT - c. two: c by LOBT - SCHED above 60 s or not.
    ls: least squares on take-off - LOBT, LOBT - SCHED (0 to 3,600 s) and EOBT_1 - LOBT (0 to 7,200 s)."""
    e = o.dl - o.y
    if form == "one": return np.clip(h.dl - np.median(e), 60, None).to_numpy()
    if form == "ls":
        X = lambda v: np.c_[np.ones(len(v)), v.dl, v.ls.clip(0, 3600), v.es.clip(0, 7200)]
        return np.clip(X(h) @ np.linalg.lstsq(X(o), o.y.to_numpy(), rcond=None)[0], 60, None)
    return np.clip(np.where(h.late, h.dl - np.median(e[o.late]), h.dl - np.median(e[~o.late])), 60, None)


print("rule error on U by month, constants from the 11 other months: month, rows, RMS error of form one, of form two, RMS of the label about 900 s")
E = {"one": [], "two": [], "ls": []}
for m in range(1, 13):
    o, h = u[u.mon != m], u[u.mon == m]; e1 = rule(o, h, "one") - h.y; e2 = rule(o, h, "two") - h.y; E["one"].append(e1); E["two"].append(e2); E["ls"].append(rule(o, h, "ls") - h.y)
    print(f"   {m:2d} {len(h):3d} {np.sqrt(np.mean(e1 ** 2)):6.0f} {np.sqrt(np.mean(e2 ** 2)):6.0f} {np.sqrt(np.mean((900 - h.y) ** 2)):6.0f}")
for k in E: e = np.concatenate([np.asarray(x) for x in E[k]]); print(f"  form {k}: RMS {np.sqrt(np.mean(e ** 2)):.0f} s, median abs {np.median(np.abs(e)):.0f} s, mean {e.mean():+.0f} s")

print("\n## 4. fold on base B")
f = frame(sys.argv[1]); N = len(f); fi = f.set_index("MVT_ID_mvt"); other = u[~u.mon.isin([1, 7])]; fold = u[u.mon.isin([1, 7])].copy()
fold["B"] = fi.B.reindex(fold.MVT_ID_mvt).to_numpy(); fold["w"] = fi.w26.reindex(fold.MVT_ID_mvt).to_numpy(); fold["one"] = rule(other, fold, "one"); fold["two"] = rule(other, fold, "two"); fold["lsq"] = rule(other, fold, "ls")
e = other.dl - other.y; print(f"constants from the ten other months ({len(other)} rows): c {np.median(e):.0f} s; plan on time {np.median(e[~other.late]):.0f} s ({int((~other.late).sum())} rows), plan late {np.median(e[other.late]):.0f} s ({int(other.late.sum())} rows)")
print(fold.assign(w=fold.w * N)[["FLIGHT_mvt", "MVT_TIME_UTC_mvt", "odd", "isP", "ls", "dl", "y", "B", "one", "two", "w"]].round({"ls": 0, "dl": 0, "y": 0, "B": 0, "one": 0, "two": 0, "w": 2}).to_string())
print(f"U rows of the fold: {len(fold)}; RMS error of B {np.sqrt(np.mean((fold.B - fold.y) ** 2)):.0f} s, form one {np.sqrt(np.mean((fold.one - fold.y) ** 2)):.0f} s, form two {np.sqrt(np.mean((fold.two - fold.y) ** 2)):.0f} s; mean of label - B {np.mean(fold.y - fold.B):+.0f} s; B below the label on {int((fold.B < fold.y).sum())} rows")
pos = pd.Series(np.arange(N), index=f.MVT_ID_mvt.to_numpy()).reindex(fold.MVT_ID_mvt).to_numpy(); yv = f.y.to_numpy(); Bv = f.B.to_numpy(); wv = f.w26.to_numpy()


def G(new, title, rows=None):
    rows = np.ones(len(fold), bool) if rows is None else np.asarray(rows); v = Bv.copy(); v[pos[rows]] = np.asarray(new)[rows]
    g, lo, hi, g5 = boot(f, (Bv - yv) ** 2 - (v - yv) ** 2)                                # boot applies w26 (the weights sum to 0.934 on the fold)
    per = wv[pos] * ((Bv[pos] - yv[pos]) ** 2 - (np.where(rows, new, Bv[pos]) - yv[pos]) ** 2)
    print(f"  {title}: G {g:+.1f} [{lo:+.1f}, {hi:+.1f}], without the 5 largest airport-days {g5:+.1f}; rows {int(rows.sum())}, gain {int((per > 0).sum())}, lose {int((per < 0).sum())}; January {per[fold.mon.to_numpy() == 1].sum():+.1f}, July {per[fold.mon.to_numpy() == 7].sum():+.1f}")


print(f"G against B (sum of w26 x fall of the squared error, s2 per ranking row; airport-day bootstrap); mass of the U rows in B {float((fold.w * (fold.B - fold.y) ** 2).sum()):.1f} of {float((wv * (Bv - yv) ** 2).sum()):.0f}")
G(fold.one, "form one, all U rows"); G(fold.two, "form two, all U rows")
G(fold.two, "form two, P rows only", fold.isP); G(fold.two, "form two, rows with real seconds only", fold.odd); G(fold.two, "form two, real seconds without a twin", fold.odd & ~fold.isP)
for w in (0.25, 0.5, 0.75): G(fold.B + w * (fold.two - fold.B), f"form two at weight {w}")
G(np.maximum(fold.two, fold.B), "the larger of form two and B"); G(np.clip(fold.two, fold.dl - 3606, fold.dl + 3606), "form two inside the join window")
G(fold.lsq, "least-squares form, all U rows"); G(fold.B + 0.5 * (fold.lsq - fold.B), "least-squares form at weight 0.5")
G(fold.y, "label oracle on the U rows")

print("\n## 5. 2026 (label-free)")
rb = pd.read_parquet(f"{ROOT}/experiments/exp-169-stack-plan/inputs/unit2-ready/bundle-ranking.parquet"); rb["B26"] = compose(rb, route="R4mk", noplan="v3cal", practice="selected"); rb = rb.set_index("MVT_ID_mvt")
u26["B"] = rb.B26.reindex(u26.MVT_ID_mvt).to_numpy(); u26["one"] = rule(u, u26, "one"); u26["two"] = rule(u, u26, "two"); u26["lsq"] = rule(u, u26, "ls"); e = u.dl - u.y
print(f"constants from all 2025 U rows: c {np.median(e):.0f} s; plan on time {np.median(e[~u.late]):.0f} s, plan late {np.median(e[u.late]):.0f} s")
print(u26[["mon", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "odd", "isP", "ls", "dl", "taxi_nm", "B", "one", "two", "lsq"]].round(0).to_string())
n26 = int((r.PHASE_mvt == "DEP").sum()); print(f"U rows 2026: {len(u26)}; take-off - LOBT {q(u26.dl)} (2025: {q(u.dl)}); plan late {u26.late.mean():.2f} (2025: {u.late.mean():.2f})")
print(f"the stack's vector on the U rows of 2026: mean {u26.B.mean():.0f} s; form two: mean {u26.two.mean():.0f} s; rows where the stack is below form two: {int((u26.B < u26.two).sum())}; least-squares form: mean {u26.lsq.mean():.0f} s, stack below it on {int((u26.B < u26.lsq).sum())} rows")
print(f"distance of the stack's vector from the rule, s2 per ranking row: form one {float(((u26.B - u26.one) ** 2).sum() / n26):.1f}, form two {float(((u26.B - u26.two) ** 2).sum() / n26):.1f}, least squares {float(((u26.B - u26.lsq) ** 2).sum() / n26):.1f}; fold analogue (unweighted): form one {float(((fold.B - fold.one) ** 2).sum() / N):.1f}, form two {float(((fold.B - fold.two) ** 2).sum() / N):.1f}, least squares {float(((fold.B - fold.lsq) ** 2).sum() / N):.1f}")
