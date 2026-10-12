"""LD reads (SPEC-LD.md): isolated-5's LD step (ORD-S also applies on the rows of long-taxi days; the fit is unchanged)
on chain f1's R9 tree, against f1's R9 tree.

Held (fold rows, the release scorer iso/score.py of the LD tree): G of the LD ORD-S values against f1's, and per
airport-day of the fold's long days: rows, apply rows, changed rows, G, mean and rms of the change, mean residual. The
fit check: every row outside the long-day apply rows keeps f1's value (fold and 2026).
2026: changed rows per month and per long-taxi airport-day (rows, apply rows, changed rows, mean and rms of the
change, mean s7 - c5), the share of the change per airport-month; T11b's flag on the raw 2026 rows must give the
chain's long_day column. The dh rows that move. 2025: the long days of each training month (T11b's flag on the
twelve raw training files together), and the same flag against the fold's long_day column (January and July).
No pass line.

    $PY ld_read.py <f1 R9 run tree> <LD run tree> <f1 dh file> <LD dh file> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
A, N = (Path(x).resolve() for x in sys.argv[1:3])
dh_a, dh_n, out = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
assert not out.exists(), f"{out} exists (write once)"
sys.path.insert(0, str(N / "w2")); sys.path.insert(0, str(N / "iso")); sys.path.append(str(N / "release"))
import base as B  # noqa: E402
import finalmode as FM  # noqa: E402
import score as S  # noqa: E402
import t11b_stage as T  # noqa: E402

assert Path(T.__file__).resolve().parent == (N / "release").resolve(), T.__file__
assert Path(S.__file__).resolve().parent == (N / "iso").resolve() and Path(B.__file__).resolve().parent == (N / "w2").resolve()
assert FM.is_final()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r3(x):
    return round(float(x), 3)


rec = {"f1_run": str(A), "ld_run": str(N), "code": {"ld_read.py": sha(Path(__file__))},
       "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in
                 (("f1_v", A / "outputs/w2-l2c-s7c.parquet"), ("ld_v", N / "outputs/w2-l2c-s7c.parquet"),
                  ("f1_dh", dh_a), ("ld_dh", dh_n), ("npz", N / "work/w2/s7c-final.npz"))}}
assert sha(A / "work/w2/s7c-final.npz") == rec["files"]["npz"]["sha256"]
z = np.load(N / "work/w2/s7c-final.npz")

# ---------------------------------------------------------------- held (fold)
d = S.fold()
bf = B.load("fold")
fid = d[ID].to_numpy("int64")
assert np.array_equal(bf[ID].to_numpy("int64"), fid)
fa, fn = np.load(A / "work/w2/ords_fold.npy").astype(float), np.load(N / "work/w2/ords_fold.npy").astype(float)
c5 = bf.c5.to_numpy(float); v = z["fold"].astype(float); y = d.y.to_numpy(float); w = d.w.to_numpy(float)
ld = bf.long_day.to_numpy(bool); month = bf.month.to_numpy(); ap = bf.airport.astype(str).to_numpy()
elig = bf.scope.to_numpy(bool) & (bf.cls == "ordinary").to_numpy() & np.isfinite(v) & ld
chg = fn != fa
assert not (chg & ~elig).any(), "a fold row outside the long-day apply rows moved"
assert np.array_equal(fn[~elig].view(np.uint64), fa[~elig].view(np.uint64))
res = S.score(fn, base=fa, d=d)
rec["held"] = {k: (r3(x) if isinstance(x, float) else x) for k, x in res.items()}
rec["held"]["apply_rows"] = int(elig.sum())
rec["held"]["long_day_rows_by_month"] = {int(m): int((ld & (month == m)).sum()) for m in (1, 7)}
contrib = w * ((y - fa) ** 2 - (y - fn) ** 2)
e = fn - fa
rows = []
for k in sorted(set(d.aday[ld])):
    m = (d.aday == k).to_numpy()
    mc = m & chg
    rows.append({"aday": k, "month": int(month[m][0]), "rows": int(m.sum()), "long_day_rows": int((m & ld).sum()),
                 "apply_rows": int((m & elig).sum()), "changed": int(mc.sum()), "G": r3(contrib[m].sum()),
                 "mean_change": r3(e[mc].mean()) if mc.any() else 0.0,
                 "rms_change": r3(np.sqrt((e[mc] ** 2).mean())) if mc.any() else 0.0,
                 "mean_y_minus_f1": r3((y - fa)[mc].mean()) if mc.any() else 0.0,
                 "mean_s7_minus_c5": r3((v - c5)[mc].mean()) if mc.any() else 0.0,
                 "weight_sum": r3(w[mc].sum())})
rec["held_days"] = rows
assert sum(r["changed"] for r in rows) == int(chg.sum())

# ---------------------------------------------------------------- 2026
rk = B.load("ranking")
rid = rk[ID].to_numpy("int64")
ra, rn = np.load(A / "work/w2/ords_ranking.npy").astype(float), np.load(N / "work/w2/ords_ranking.npy").astype(float)
c5r = rk.c5.to_numpy(float); vr = z["ranking"].astype(float)
ldr = rk.long_day.to_numpy(bool); mr = rk.month.to_numpy(); apr = rk.airport.astype(str).to_numpy()
eligr = rk.scope.to_numpy(bool) & (rk.cls == "ordinary").to_numpy() & np.isfinite(vr) & ldr
chr_ = rn != ra
assert not (chr_ & ~eligr).any(), "a 2026 row outside the long-day apply rows moved"
assert np.array_equal(rn[~eligr].view(np.uint64), ra[~eligr].view(np.uint64))
raw = pd.read_parquet(FM.ranking_path(N / "input-root/data/raw"), columns=T.COLS)
keys = T.row_keys(rid, raw).to_numpy()
_, g = T.flags(raw)
snow = g[g.snow].copy()
skeys = set(snow.ap + "|" + snow.dstr)
flag = pd.Series(keys).isin(skeys).to_numpy()
rec["flag_2026_equals_chain_long_day"] = bool(np.array_equal(flag, ldr))
rec["flag_2026_mismatch_rows"] = int((flag != ldr).sum())
er = rn - ra
mo = {1: "2026-01", 2: "2026-02", 6: "2026-06", 7: "2026-07"}
rec["moved_2026_by_month"] = {
    mo[m]: {"long_day_rows": int((ldr & (mr == m)).sum()), "apply_rows": int((eligr & (mr == m)).sum()),
            "changed": int((chr_ & (mr == m)).sum()),
            "rms_on_changed": r3(np.sqrt((er[chr_ & (mr == m)] ** 2).mean())) if (chr_ & (mr == m)).any() else 0.0,
            "mean_on_changed": r3(er[chr_ & (mr == m)].mean()) if (chr_ & (mr == m)).any() else 0.0}
    for m in mo}
rec["moved_2026"] = {"changed": int(chr_.sum()), "rms_on_changed": r3(np.sqrt((er[chr_] ** 2).mean())) if chr_.any() else 0.0,
                     "mean_on_changed": r3(er[chr_].mean()) if chr_.any() else 0.0,
                     "rms_all_rows": r3(np.sqrt((er ** 2).mean()))}
days = []
for _, s in snow.sort_values(["dstr", "ap"]).iterrows():
    k = s.ap + "|" + s.dstr
    m = keys == k
    mc = m & chr_
    days.append({"aday": k, "month": int(s.dstr[:2]), "departures": int(s.rows), "n_long": int(s.n_long),
                 "share_long": r3(s.share), "rows": int(m.sum()), "apply_rows": int((m & eligr).sum()),
                 "changed": int(mc.sum()), "mean_change": r3(er[mc].mean()) if mc.any() else 0.0,
                 "rms_change": r3(np.sqrt((er[mc] ** 2).mean())) if mc.any() else 0.0,
                 "mean_s7_minus_c5": r3((vr - c5r)[mc].mean()) if mc.any() else 0.0,
                 "lirf_jul_scope": bool(s.ap == "LIRF" and int(s.dstr[:2]) in FM.JUL_SCOPE)})
rec["days_2026"] = days
assert sum(r["changed"] for r in days) == int(chr_.sum())
am = pd.DataFrame({"k": pd.Series(apr).str.cat(pd.Series(mr).astype(str), sep="-"), "e2": er ** 2, "c": chr_.astype(int)})[chr_]
tot = am.groupby("k").agg(rows=("c", "sum"), e2=("e2", "sum"))
rec["share_2026_by_airport_month"] = {str(k): {"rows": int(r.rows), "row_share": r3(r.rows / max(1, chr_.sum())),
                                               "sq_change_share": r3(r.e2 / max(1e-12, tot.e2.sum()))}
                                      for k, r in tot.iterrows()}

# ---------------------------------------------------------------- dh rows
va, vn = pd.read_parquet(A / "outputs/w2-l2c-s7c.parquet"), pd.read_parquet(N / "outputs/w2-l2c-s7c.parquet")
ha, hn = pd.read_parquet(dh_a), pd.read_parquet(dh_n)
for f in (va, vn, ha, hn):
    assert np.array_equal(f[ID].to_numpy("float64").astype("int64"), rid)
va, vn, ha, hn = (f[Y].to_numpy(float) for f in (va, vn, ha, hn))
assert np.array_equal(vn, rn) and np.array_equal(va, ra)
dhm = va != ha
assert int(dhm.sum()) == 14587 and set(apr[dhm]) == {"EDDF"}
mv = hn != ha
rec["dh"] = {"dh_rows": int(dhm.sum()), "dh_rows_moved_by_ld": int((mv & dhm).sum()),
             "dh_rows_on_long_days": int((dhm & ldr).sum()), "dh_rows_apply_rows": int((dhm & eligr).sum()),
             "moved_outside_dh_rows_equal_v_move": bool(np.array_equal((hn - ha)[~dhm], (vn - va)[~dhm])),
             "ld_dh_equals_ld_v_outside_dh_rows": bool(np.array_equal(hn[~dhm], vn[~dhm])),
             "dh_file_moved_rows": int(mv.sum())}

# ---------------------------------------------------------------- 2025 training months
files = sorted((N / "input-root/data/raw").glob("training_2025-*.parquet"))
assert len(files) == 12, files
raw25 = pd.concat([pd.read_parquet(f, columns=T.COLS) for f in files], ignore_index=True)
_, g25 = T.flags(raw25)
s25 = g25[g25.snow].copy()
s25["month"] = s25.dstr.str[:2].astype(int)
rec["long_days_2025"] = {
    "by_month": {int(m): int((s25.month == m).sum()) for m in range(1, 13)},
    "by_airport": {str(a): int(n) for a, n in s25.groupby("ap").size().items()},
    "days": [{"aday": a + "|" + t, "departures": int(r), "n_long": int(n), "share_long": r3(sh)}
             for a, t, r, n, sh in s25.sort_values(["dstr", "ap"])[["ap", "dstr", "rows", "n_long", "share"]].itertuples(index=False)],
    "february_june": [a + "|" + t for a, t in s25[s25.month.isin([2, 6])][["ap", "dstr"]].itertuples(index=False)],
    "ltfm": [t for t in s25[s25.ap == "LTFM"].dstr]}
fkeys = T.row_keys(fid, raw25).to_numpy()
fflag = pd.Series(fkeys).isin(set(s25.ap + "|" + s25.dstr)).to_numpy()
rec["flag_2025_year_vs_fold_long_day"] = {"equal": bool(np.array_equal(fflag, ld)), "mismatch_rows": int((fflag != ld).sum())}

out.write_text(json.dumps(rec, indent=1))
h = rec["held"]
print(f"LDHELD G {h['G']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}] P {h['p_le0']:.3f} wo5 {h['wo5']:+.3f} changed {h['changed']} "
      f"apply {h['apply_rows']} month {h['month']} airport {h['airport']} long-day rows by month {h['long_day_rows_by_month']}")
for r in rows:
    print("LDHELDDAY", json.dumps(r))
print("LD26", json.dumps(rec["moved_2026"]), json.dumps(rec["moved_2026_by_month"]))
print("LD26FLAG equal", rec["flag_2026_equals_chain_long_day"], "mismatch", rec["flag_2026_mismatch_rows"])
for r in days:
    print("LD26DAY", json.dumps(r))
print("LD26SHARE", json.dumps(rec["share_2026_by_airport_month"]))
print("LDDH", json.dumps(rec["dh"]))
print("LD25", json.dumps({k: rec["long_days_2025"][k] for k in ("by_month", "by_airport", "february_june", "ltfm")}),
      json.dumps(rec["flag_2025_year_vs_fold_long_day"]))
