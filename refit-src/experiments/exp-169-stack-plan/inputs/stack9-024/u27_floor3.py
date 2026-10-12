"""exp-168 unit 27, third addendum: the soft stand floor of the row itself (EDDF, EGLL, LFPG, LTFM).
g = take-off - the latest in-block of an arrival of another aircraft type at the departure's stand in the 90 min before
the take-off. r = g - stack5. The label is at least g, so a floor under the stack value still raises the expected label.
  A. fold (held stack5): step forms (1) m(r) = mean of the other month's residuals of the airport above r, less their
     mean (no fitted size); (2) calibrated: mean e by band of r over the rows with a floor of the four airports in the
     other month; (3) hard floor max(r, 0) (the form of C55 and C71 on this g). G, interval over airport-days, without
     the 5 largest airport-days, rows moved, by airport and month. The bands of r with rows, mean e and mean m;
  B. 2026 (ranking stack5 vector): rows with a floor by airport and month, bands of r against the fold, the bands on the
     rows with a surface clock (EDDF, EGLL), mean step and distance (sum of step^2 / ranking rows).
Usage: u27_floor3.py <fold5.parquet>  > checks/u27_floor3.out"""
import sys
import numpy as np, pandas as pd
from u11_lib import raw2025, rank2026, secs, ROOT
from u9_lib import frame, boot

S4 = f"{ROOT}/experiments/exp-169-stack-plan"; pd.set_option("display.width", 330); FOUR = ["EDDF", "EGLL", "LFPG", "LTFM"]; BANDS = [-1e9, -1500, -900, -600, -450, -300, -200, -100, 0, 1e9]


def floors(d, T0):
    d = d[d.ap.isin(FOUR)].copy(); d["t"] = secs(d.MVT_TIME_UTC_mvt, T0); d["k_st"] = d.ap + "|" + d.STAND_mvt.astype(str)
    x = d[d.PHASE_mvt == "DEP"].copy().reset_index(drop=True); a = d[(d.PHASE_mvt == "ARR") & d.STAND_mvt.notna() & (secs(d.BLOCK_TIME_UTC_mvt, d.SCHED_TIME_UTC_mvt).abs() > 6)].copy(); a["ib"] = secs(a.BLOCK_TIME_UTC_mvt, T0); a = a[a.ib.notna()]
    g = np.full(len(x), np.nan); gs = dict(tuple(a.groupby("k_st"))); xt = x.t.to_numpy(); xty = x.AIRCRAFT_TYPE_mvt.astype(str).to_numpy(); has_st = x.STAND_mvt.notna().to_numpy()
    for k, idx in x.groupby("k_st").indices.items():
        v = gs.get(k)
        if v is None: continue
        ib = v.ib.to_numpy(); o = np.argsort(ib, kind="stable"); ib = ib[o]; ty = v.AIRCRAFT_TYPE_mvt.astype(str).to_numpy()[o]
        j = np.searchsorted(ib, xt[idx], "left") - 1; ok = (j >= 0) & has_st[idx]; jj = np.clip(j, 0, None); gg = xt[idx] - ib[jj]
        ok &= (gg > 0) & (gg <= 5400) & (ty[jj] != xty[idx]) & (ty[jj] != "nan") & (xty[idx] != "nan"); g[idx[ok]] = gg[ok]
    x["g"] = g; x["id"] = x.MVT_ID_mvt.astype("int64"); return x


def mexc(eo, r):
    eo = np.sort(eo); cs = np.concatenate([[0.0], np.cumsum(eo[::-1])])[::-1]; i = np.searchsorted(eo, r, "left"); n = len(eo) - i
    return np.where(n >= 20, cs[np.clip(i, 0, len(eo))] / np.maximum(n, 1), np.maximum(r, 0)) - eo.mean()


# ---- A. fold
f = frame(sys.argv[1]); w26 = pd.Series(f.w26.to_numpy(), index=f.MVT_ID_mvt.astype("int64")); x = floors(raw2025([1, 7]), pd.Timestamp("2025-01-01", tz="UTC"))
h = pd.read_parquet(f"{S4}/results/stack5-held-016.parquet").set_index("MVT_ID_mvt"); x["s5"] = h.stack5.reindex(x.id).to_numpy(); x["w26"] = w26.reindex(x.id).to_numpy(); x = x[x.s5.notna() & x.w26.notna()].reset_index(drop=True)
x["airport"] = x.ap; x["day"] = (x.t // 86400).astype(int); x["e"] = x.y - x.s5; x["fit"] = x.plan & (x.ADES_mvt != x.ap) & (x.y.abs() < 21600); x["has"] = x.g.notna() & x.plan & (x.ADES_mvt != x.ap); x["r"] = x.g - x.s5
e = x.e.to_numpy(); fitm = x.fit.to_numpy(); mon = x.mon.to_numpy(); apv = x.ap.to_numpy(); has = x.has.to_numpy(); r = x.r.to_numpy(); x["band"] = pd.cut(x.r, BANDS)
m = np.zeros(len(x)); cal = np.zeros(len(x))
for mth, o in ((1, 7), (7, 1)):
    tb = x[fitm & has & (mon == o)].groupby("band", observed=False).e.mean().fillna(0.0); te = has & (mon == mth); cal[te] = x.band[te].map(tb).astype(float).to_numpy()
    for ap in FOUR:
        te = np.flatnonzero(has & (mon == mth) & (apv == ap)); m[te] = mexc(e[fitm & (mon == o) & (apv == ap)], r[te])
hard = np.where(has, np.maximum(r, 0), 0.0); x["m"] = m; x["cal"] = cal
print("## A. fold, held stack5; rows in scope by airport:", x[fitm].groupby("ap").size().to_dict(), "; with a floor:", x[fitm & has].groupby("ap").size().to_dict(), "; label more than 30 s under g:", int((x.y[fitm & has] < x.g[fitm & has] - 30).sum()))
print("bands of r = g - stack5 (rows with a floor, the four airports): rows, mean e, mean m, mean calibrated step, by month:")
print(x[fitm & has].groupby(["band", "mon"], observed=False).agg(rows=("e", "size"), e_mean=("e", "mean"), m_mean=("m", "mean"), cal_mean=("cal", "mean")).round(1).unstack("mon").to_string())


def show(name, step):
    dS = np.where(fitm, e ** 2 - (e - step) ** 2, 0.0); G = boot(x, dS); by = (x.w26 * dS).groupby([x.ap, x.mon]).sum().round(1); n = int((np.abs(step[fitm]) >= 1).sum()); gain = int((dS[fitm & (np.abs(step) >= 1)] > 0).sum()); days = x[fitm & (np.abs(step) >= 1)].groupby(["ap", "day"]).ngroups
    print(f"  {name}: G {G[0]:+.1f} [{G[1]:+.1f}, {G[2]:+.1f}], without the 5 largest airport-days {G[3]:+.1f}; rows moved by 1 s and more {n} (gain {gain}, loss {n - gain}) on {days} airport-days; by airport and month {by.to_dict()}")
    scope_all = x.plan & (x.ADES_mvt != x.ap); dA = np.where(scope_all, e ** 2 - (e - step) ** 2, 0.0); print(f"      with the labels of 6 h and more: {float((x.w26 * dA).sum()):+.1f}")


show("(1) m(r), no fitted size", m); show("(1) with rows of r above -600 only", np.where(r > -600, m, 0.0)); show("(1) with rows of r above -300 only", np.where(r > -300, m, 0.0)); show("(2) calibrated bands from the other month (pooled)", cal); show("(2) calibrated, rows of r above -600 only", np.where(r > -600, cal, 0.0)); show("(3) hard floor max(r, 0)", hard); show("(1) at half size", 0.5 * m); show("larger of (1) and (3)", np.maximum(m, hard))

# ---- B. 2026
q = floors(rank2026(), pd.Timestamp("2026-01-01", tz="UTC")); rk = pd.read_parquet(f"{S4}/exp169-stack5-Gmixed-bag6r6-C73-ext-aobt-c55.parquet").set_index("MVT_ID_mvt"); n26 = len(rk)
q["s5"] = rk.TAXITIME_SEC_mvt.reindex(q.id).to_numpy(); q = q[q.s5.notna()].copy(); q["has"] = q.g.notna() & q.plan & (q.ADES_mvt != q.ap); q["r"] = q.g - q.s5; q["band"] = pd.cut(q.r, BANDS)
ck = set(pd.read_parquet(f"{S4}/inputs/HB1-jjrank.parquet").MVT_ID_mvt.astype("int64")); q["clock"] = q.id.isin(ck)
print("\n## B. 2026 (ranking vector of stack5); ranking rows", n26)
t5 = x[x.plan & (x.ADES_mvt != x.ap)].groupby(["ap", "mon"]).agg(rows=("has", "size"), floor=("has", "sum")); t6 = q[q.plan & (q.ADES_mvt != q.ap)].groupby(["ap", "mon"]).agg(rows=("has", "size"), floor=("has", "sum"), clock=("clock", "sum"))
print("rows with a plan and rows with a floor, 2025 fold:"); print(t5.assign(share=(t5.floor / t5.rows).round(4)).T.to_string()); print("2026:"); print(t6.assign(share=(t6.floor / t6.rows).round(4)).T.to_string())
b5 = x[x.has].groupby(["ap", "band"], observed=False).size().unstack("ap"); b6 = q[q.has].groupby(["ap", "band"], observed=False).size().unstack("ap")
print("share of the rows with a floor by band of r: 2025 fold, then 2026:"); print((b5 / b5.sum()).round(4).to_string()); print((b6 / b6.sum()).round(4).to_string())
c6 = q[q.has & q.ap.isin(["EDDF", "EGLL"])].groupby(["ap", "clock", "band"], observed=False).size().unstack(["ap", "clock"]); print("2026, EDDF and EGLL, rows with a floor by surface clock (counts by band):"); print(c6.to_string())
m6 = np.zeros(len(q)); r6 = q.r.to_numpy()
for ap in FOUR:
    te = np.flatnonzero(q.has.to_numpy() & (q.ap.to_numpy() == ap)); m6[te] = mexc(e[fitm & (apv == ap)], r6[te])
tb = x[fitm & has].groupby("band", observed=False).e.mean().fillna(0.0); cal6 = np.where(q.has, q.band.map(tb).astype(float).fillna(0.0), 0.0); hard6 = np.where(q.has, np.maximum(r6, 0), 0.0); q["m"] = m6; q["cal"] = cal6
for nm, st in (("(1) m(r)", m6), ("(1), r above -600 only", np.where(r6 > -600, m6, 0.0)), ("(2) calibrated bands of the fold", cal6), ("(3) hard floor", hard6)):
    mv = np.abs(st) >= 1; print(f"  {nm}: rows moved {int(mv.sum())}; mean step on them {st[mv].mean():+.1f} s; largest {np.abs(st).max():.0f} s; distance {float((st ** 2).sum() / n26):.1f}; by airport (rows, distance) {{" + ", ".join(f"{ap}: ({int((mv & (q.ap.to_numpy() == ap)).sum())}, {float((st[q.ap.to_numpy() == ap] ** 2).sum() / n26):.1f})" for ap in FOUR) + "}")
cq = q[q.has & q.clock]; print(f"2026 rows with a floor and a surface clock: {len(cq)}; share with g above stack5 {float((cq.r > 0).mean()):.4f} (rows without a clock: {float((q[q.has & ~q.clock].r > 0).mean()):.4f}); share with r above -300: {float((cq.r > -300).mean()):.4f} against {float((q[q.has & ~q.clock].r > -300).mean()):.4f}")
