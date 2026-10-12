"""R9 reads on the final base (SPEC-R9.md): the R9 chain against the base chain (the released rule through the
same code; its V file is run (iii)'s 6de02372 and its dh file final-iii-dh c4511650).

Held (fold rows, the release scorer `iso/score.py` of the R9 run tree: the pack's exposure weights, 2,000 airport-day
draws, seed 601, wo5, G by month, class and airport):
- the R9 step at each assembly stage (route ... ROLL of the Z chain, from r9_build.py's fold-stages npz);
- the R9 step at each run stage: c5, L2C, ORD-S (= the V file's fold values);
- each step within its own chain: L2C against c5 and ORD-S against L2C, base and R9.
2026: rows moved by R9 at each stage (zabd-stages parts, the run trees, the V and dh files), by month, class and
airport, with the rms of the move; the move on the 14,587 dh rows. ORD-S betas of both trees (port_read.py form).
No pass line.

    $PY r9_read.py <base run tree> <r9 run tree> <base parts dir> <r9 parts dir> <base dh file> <r9 dh file> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
A, N, PA, PN = (Path(x).resolve() for x in sys.argv[1:5])
dh_a, dh_n, out = Path(sys.argv[5]), Path(sys.argv[6]), Path(sys.argv[7])
assert not out.exists(), f"{out} exists (write once)"
sys.path.insert(0, str(N / "iso")); sys.path.insert(0, str(N / "w2"))
import base as B  # noqa: E402
import ords as O  # noqa: E402
import score as S  # noqa: E402

V_III, DH_III = ("6de02372de5d1aea062300a88af7af51606ac2d6aee645c384b09398c9c57378",
                 "c45116502cad96dbb270467b9f2ae26fb6e06d9a50ca34ff214253609f44612d")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rec = {"base_run": str(A), "r9_run": str(N), "files": {}}
for k, p in (("base_v", A / "outputs/w2-l2c-s7c.parquet"), ("r9_v", N / "outputs/w2-l2c-s7c.parquet"),
             ("base_dh", dh_a), ("r9_dh", dh_n)):
    rec["files"][k] = {"path": str(p), "sha256": sha(p)}
rec["base_is_run_iii"] = {"v": rec["files"]["base_v"]["sha256"] == V_III, "dh": rec["files"]["base_dh"]["sha256"] == DH_III}
for s in ("fold", "ranking"):
    assert sha(A / f"work/w2/chain_ids_{s}.parquet") == sha(N / f"work/w2/chain_ids_{s}.parquet")

d = S.fold()
fa = pd.read_parquet(A / "work/chain/fold.parquet", columns=[ID, "y", "w"])
for c in (ID, "y", "w"):
    assert np.array_equal(fa[c].to_numpy(), d[c].to_numpy()), c   # same rows, labels and weights in both trees
fid = d[ID].to_numpy("int64")


def sc(cand, base):
    r = S.score(np.asarray(cand, float), base=np.asarray(base, float), d=d)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


# 1. assembly stages (fold): the R9 step at each Z stage
za, zn = np.load(PA / "fold-stages.npz"), np.load(PN / "fold-stages.npz")
assert list(za["names"]) == list(zn["names"]) and np.array_equal(za["ids"], zn["ids"])
pos = pd.Series(np.arange(len(za["ids"])), index=za["ids"]).reindex(fid).to_numpy()
assert not np.isnan(pos.astype(float)).any() and len(set(za["ids"].tolist())) == len(fid)
names = [str(x) for x in za["names"]]
stage = {"base": {n: za[f"s{i:02d}"][pos] for i, n in enumerate(names)}, "r9": {n: zn[f"s{i:02d}"][pos] for i, n in enumerate(names)}}
rec["held_assembly"] = {n: sc(stage["r9"][n], stage["base"][n]) for n in names}

# 2. run stages (fold): c5, L2C, ORD-S
run = {}
for name, t in (("base", A), ("r9", N)):
    bf = pd.read_parquet(t / "work/w2/base_fold.parquet", columns=[ID, "v64", "c5"])
    assert np.array_equal(bf[ID].to_numpy("int64"), fid)
    run[name] = {"v64_chain": bf.v64.to_numpy(float), "c5": bf.c5.to_numpy(float),
                 "l2c": np.load(t / "work/w2/l2c_fold.npy").astype(float), "ords": np.load(t / "work/w2/ords_fold.npy").astype(float)}
    v64n = [n for n in names if n.startswith("T11b")][0]
    rec.setdefault("link_v64_chain_vs_assembly_max_abs_s", {})[name] = float(np.max(np.abs(run[name]["v64_chain"] - stage[name][v64n])))
rec["held_run"] = {k: sc(run["r9"][k], run["base"][k]) for k in ("v64_chain", "c5", "l2c", "ords")}
rec["held_steps"] = {f"{k}_{t}": {"base": sc(run["base"][t], run["base"][k]), "r9": sc(run["r9"][t], run["r9"][k])}
                     for k, t in (("v64_chain", "c5"), ("c5", "l2c"), ("l2c", "ords"))}

# 3. 2026 rows moved
rk = B.load("ranking")
month = rk.month.to_numpy(); ap = rk.airport.astype(str).to_numpy(); cls = rk.cls.astype(str).to_numpy()
rid = rk[ID].to_numpy("int64")
va, vn = pd.read_parquet(A / "outputs/w2-l2c-s7c.parquet"), pd.read_parquet(N / "outputs/w2-l2c-s7c.parquet")
ha, hn = pd.read_parquet(dh_a), pd.read_parquet(dh_n)
for f in (va, vn, ha, hn):
    assert np.array_equal(f[ID].to_numpy("float64").astype("int64"), rid)
dhm = va[T].to_numpy(float) != ha[T].to_numpy(float)
assert dhm.sum() == 14587 and set(ap[dhm]) == {"EDDF"}, int(dhm.sum())


def groups(m):
    s = pd.Series(1, index=pd.MultiIndex.from_arrays([month[m], cls[m], ap[m]], names=["month", "cls", "airport"]))
    return {"by_month": {int(k): int(v) for k, v in s.groupby(level=0).sum().items()},
            "by_class": {str(k): int(v) for k, v in s.groupby(level=1).sum().items()},
            "by_airport": {str(k): int(v) for k, v in s.groupby(level=2).sum().items()}}


def moved(old, new):
    old, new = np.asarray(old, float), np.asarray(new, float)
    m = old != new; e = new - old
    r = {"rows": int(m.sum()), **groups(m), "rms_on_moved": round(float(np.sqrt((e[m] ** 2).mean())), 3) if m.any() else 0.0,
         "mean_on_moved": round(float(e[m].mean()), 3) if m.any() else 0.0}
    md = m & dhm
    r["dh_rows"] = {"rows": int(md.sum()), **groups(md),
                    "rms_on_moved": round(float(np.sqrt((e[md] ** 2).mean())), 3) if md.any() else 0.0}
    return r


sa, sn = pd.read_parquet(PA / "zabd-stages.parquet"), pd.read_parquet(PN / "zabd-stages.parquet")
for f in (sa, sn):
    assert np.array_equal(f[ID].to_numpy("int64"), rid)
rec["moved_2026"] = {f"zabd/{k}": moved(sa[k], sn[k]) for k in sa.columns
                     if k not in (ID, "member_scope", "cls", "month")}
rec["moved_2026"]["member_scope_equal"] = bool(np.array_equal(sa.member_scope.to_numpy(), sn.member_scope.to_numpy()))
ba, bn = pd.read_parquet(A / "work/w2/base_ranking.parquet", columns=[ID, "c5"]), pd.read_parquet(N / "work/w2/base_ranking.parquet", columns=[ID, "c5"])
rec["moved_2026"]["c5"] = moved(ba.c5, bn.c5)
for k in ("l2c", "ords"):
    rec["moved_2026"][k] = moved(np.load(A / f"work/w2/{k}_ranking.npy"), np.load(N / f"work/w2/{k}_ranking.npy"))
rec["moved_2026"]["v_file"] = moved(va[T], vn[T])
rec["moved_2026"]["dh_file"] = moved(ha[T], hn[T])
inc = {t: np.load(t / "work/w2/ords_ranking.npy") - np.load(t / "work/w2/l2c_ranking.npy") for t in (A, N)}
rec["moved_2026"]["ords_step"] = moved(inc[A], inc[N])
# the dh step's own move on the dh rows: (dh - V) in each chain
rec["dh_step"] = {"base_rms_s": round(float(np.sqrt(np.mean((ha[T].to_numpy(float) - va[T].to_numpy(float))[dhm] ** 2))), 3),
                  "r9_rms_s": round(float(np.sqrt(np.mean((hn[T].to_numpy(float) - vn[T].to_numpy(float))[dhm] ** 2))), 3),
                  "r9_minus_base_on_dh_rows_over_v_move": None}
dv = (vn[T].to_numpy(float) - va[T].to_numpy(float))[dhm]; dd = (hn[T].to_numpy(float) - ha[T].to_numpy(float))[dhm]
if np.any(dv != 0):
    rec["dh_step"]["r9_minus_base_on_dh_rows_over_v_move"] = round(float(np.sum(dd * dv) / np.sum(dv * dv)), 4)

# 4. ORD-S betas (port_read.py form; the mask reads the R9 tree's fold frame, as port_read.py reads its port tree's)
dfold = B.load("fold")
assert np.array_equal(dfold[ID].to_numpy("int64"), fid)
y = d.y.to_numpy(float); w = d.w.to_numpy(float); fm = dfold.month.to_numpy(); fap = dfold.airport.astype(str).to_numpy()
rec["betas"] = {}
for name, t in (("base", A), ("r9", N)):
    c5f = run[name]["c5"]
    z = np.load(t / "work/w2/s7c-final.npz"); v = z["fold"].astype(float)
    m = O.mask(dfold, v); x = v - c5f; r = y - c5f; ok = m & (w > 0) & (np.abs(r) < O.RCLIP)
    bs, bp = O.betas(x, r, w, ok, fap)
    held = {int(h): {k: round(b, 4) for k, b in O.betas(x, r, w, ok & (fm != h), fap)[0].items()} for h in (1, 7)}
    rec["betas"][name] = {"npz_sha256": sha(t / "work/w2/s7c-final.npz"), "2026": {k: round(b, 4) for k, b in bs.items()},
                          "pooled": round(bp, 4), "held_from_other_month": held}
rec["betas"]["npz_equal"] = rec["betas"]["base"]["npz_sha256"] == rec["betas"]["r9"]["npz_sha256"]
rec["code"] = {"r9_read.py": sha(Path(__file__))}
out.write_text(json.dumps(rec, indent=1))
print("base is run (iii):", rec["base_is_run_iii"], "| link v64 max abs:", rec["link_v64_chain_vs_assembly_max_abs_s"])
def line(r):
    return (f"G {r['G']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] P {r['p_le0']:.3f} wo5 {r['wo5']:+.2f} changed "
            f"{r['changed']} month {r['month']}")


for n in names:
    print("held R9 at", n, line(rec["held_assembly"][n]))
for k in ("v64_chain", "c5", "l2c", "ords"):
    print("held R9 at", k, line(rec["held_run"][k]))
for k, v in rec["held_steps"].items():
    print("step", k, "base", line(v["base"]), "| r9", line(v["r9"]))
for k in ("zabd/primary", "zabd/final", "c5", "l2c", "ords", "v_file", "dh_file"):
    r = rec["moved_2026"][k]
    print(f"moved {k}: {r['rows']} {r['by_month']} rms {r['rms_on_moved']} | dh rows {r['dh_rows']['rows']} rms {r['dh_rows']['rms_on_moved']}")
print("betas base", rec["betas"]["base"]["2026"], rec["betas"]["base"]["pooled"])
print("betas r9  ", rec["betas"]["r9"]["2026"], rec["betas"]["r9"]["pooled"])
print("dh step", rec["dh_step"])
