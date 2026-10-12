"""exp-194 ords_ld.py: a copy of finalv's w2/ords.py (d2b5c7cd) with one flag, --long-days (isolated-5's LD step,
docs/assignments/2026-10-10-ld-port.md): the APPLY masks (te, mr and the LIRF July-scope rows) also take the rows of
long-taxi days; the FIT mask (ok) keeps ~long_day. Without the flag the copy is the released ORD-S.

Step ORD-S: on ordinary stack-scope rows, add the mean of the airport-context sequence members (s1, s2)
as a linear column with one coefficient per airport.

value = c5 + beta_a (s - c5), beta_a = weighted least squares of (y - c5) on (s - c5) over the airport's held
rows with |y - c5| < 3,000 s (pooled beta for an airport with fewer than 300 rows), floored at 0. Held read: beta from
the other held month. 2026: beta from both held months. LIRF July rows keep their input (1,647 held rows
stand for that 2026 cell; their read is noise-dominated), and so do the rows of long-taxi days. With --lirf-jul K,
LIRF July rows get c5 + K beta_LIRF (s - c5), beta_LIRF fitted without them (LIRF January rows); the LIRF July beta
of its own 1,531 held rows is 0.284 [0.038, 0.494] (day bootstrap), and 2026 LIRF July has no clock rows, like
LIRF January 2025. Clock and noac rows keep their input (c5 or the L2C values). Clipped to c5's bounds, floor 60 s.

    python3 w2/ords.py --on work/w2/l2c --names s1,s2 --save-npz w2/inputs/s12.npz [--out ...]   (from the member files)
    python3 w2/ords.py --on work/w2/l2c --npz w2/inputs/s12.npz --out outputs/w2-l2c-s.parquet  (rebuild)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso")); sys.path.insert(0, str(ROOT / "w2"))
import base as B  # noqa: E402
import finalmode as FM  # noqa: E402
import out as O  # noqa: E402
import score as S  # noqa: E402
from extra_screen import load_member  # noqa: E402

RCLIP = 3000.0
MIN_ROWS = 300
NONNEG = True   # a member column with a negative weight is noise; floor each airport's beta at 0


def mask(d, v, long_days=False):
    m = d.scope.to_numpy() & (d.cls == "ordinary").to_numpy() & np.isfinite(v)
    if not long_days:
        m &= ~d.long_day.to_numpy(bool)   # long-taxi days: 3 held days, and the 2026 ones lie outside them
    return m & ~lirf_jul(d)


def lirf_jul(d):
    return (d.airport.eq("LIRF") & d.month.isin(FM.JUL_SCOPE)).to_numpy()   # season map: Jun as Jul


def lirf_jul_rows(d, v, long_days=False):
    m = d.scope.to_numpy() & (d.cls == "ordinary").to_numpy() & np.isfinite(v)
    if not long_days:
        m &= ~d.long_day.to_numpy(bool)
    return m & lirf_jul(d)


def betas(x, r, w, ok, ap):
    bp = float((w[ok] * x[ok] * r[ok]).sum() / (w[ok] * x[ok] ** 2).sum())
    out = {}
    for a in np.unique(ap):
        t = ok & (ap == a)
        out[a] = float((w[t] * x[t] * r[t]).sum() / (w[t] * x[t] ** 2).sum()) if t.sum() >= MIN_ROWS else bp
        out[a] = max(0.0, out[a]) if NONNEG else out[a]
    return out, bp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--on", default=None, help="prefix of <prefix>_{fold,ranking}.npy values to start from (default c5)")
    p.add_argument("--names", default="s1,s2")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--allow-negative", action="store_true")
    p.add_argument("--npz", type=Path, default=None, help="member means saved by --save-npz (arrays fold, ranking)")
    p.add_argument("--save-npz", type=Path, default=None)
    p.add_argument("--lirf-jul", type=float, default=0.0, help="share K of the LIRF beta for LIRF July rows")
    p.add_argument("--long-days", action="store_true", help="apply (not fit) also on the rows of long-taxi days")
    a = p.parse_args()
    global NONNEG
    NONNEG = not a.allow_negative
    names = a.names.split(",")
    d = B.load("fold"); f = S.fold(); y = f.y.to_numpy(float); w = f.w.to_numpy(float)
    c5 = d.c5.to_numpy(float); lo = d.c5_lower.to_numpy(float); hi = d.c5_upper.to_numpy(float)
    start = np.load(ROOT / f"{a.on}_fold.npy") if a.on else c5.copy()
    if a.npz:
        z = np.load(ROOT / a.npz); v = z["fold"].astype(float); vr_saved = z["ranking"].astype(float)
    else:
        v = np.mean([load_member(n, "fold") for n in names], 0).astype(np.float32).astype(float); vr_saved = None
    m = mask(d, v); ma = mask(d, v, a.long_days); month = d.month.to_numpy(); ap = d.airport.astype(str).to_numpy()
    x = v - c5; r = y - c5; ok = m & (w > 0) & (np.abs(r) < RCLIP)
    cand = start.copy()
    for held in (1, 7):
        bs, _ = betas(x, r, w, ok & (month != held), ap)
        te = ma & (month == held)
        b = np.array([bs.get(z, 0.0) for z in ap[te]])
        cand[te] = np.maximum(60, np.clip(c5[te] + b * x[te], lo[te], hi[te]))
        lj = lirf_jul_rows(d, v, a.long_days) & (month == held)
        if a.lirf_jul and lj.any():
            cand[lj] = np.maximum(60, np.clip(c5[lj] + a.lirf_jul * bs["LIRF"] * x[lj], lo[lj], hi[lj]))
    res = S.score(cand, base=start); print("held vs start", S.fmt(res)); print("  airport", res["airport"])
    print("held vs c5", S.fmt(S.score(cand, base=c5)), flush=True)
    np.save(ROOT / "work/w2/ords_fold.npy", cand)
    held_sq = float((w * (cand - start) ** 2).sum())
    try:
        vr = vr_saved if vr_saved is not None else np.mean([load_member(n, "ranking") for n in names], 0).astype(np.float32).astype(float)
    except FileNotFoundError as e:
        print("no 2026 member file:", e); return
    if a.save_npz:
        np.savez_compressed(ROOT / a.save_npz, fold=v.astype(np.float32), ranking=vr.astype(np.float32))
        print("saved", a.save_npz)
    bs, bp = betas(x, r, w, ok, ap)
    print("2026 betas", json.dumps({k: round(z, 3) for k, z in bs.items()}), "pooled", round(bp, 3))
    rk = B.load("ranking")
    c5r = rk.c5.to_numpy(float); lor = rk.c5_lower.to_numpy(float); hir = rk.c5_upper.to_numpy(float)
    startr = np.load(ROOT / f"{a.on}_ranking.npy") if a.on else c5r.copy()
    mr = mask(rk, vr, a.long_days); apr = rk.airport.astype(str).to_numpy()
    b = np.array([bs.get(z, bp) for z in apr[mr]])
    candr = startr.copy()
    candr[mr] = np.maximum(60, np.clip(c5r[mr] + b * (vr[mr] - c5r[mr]), lor[mr], hir[mr]))
    ljr = lirf_jul_rows(rk, vr, a.long_days)
    if a.lirf_jul:
        candr[ljr] = np.maximum(60, np.clip(c5r[ljr] + a.lirf_jul * bs["LIRF"] * (vr[ljr] - c5r[ljr]), lor[ljr], hir[ljr]))
        mr = mr | ljr
    ch = candr - startr
    print("2026 changed", int((ch != 0).sum()), "mean change %.2f" % ch.mean(),
          "squared change per row %.1f (held, weighted: %.1f)" % ((ch ** 2).mean(), held_sq))
    t = pd.DataFrame({"k": rk.airport.astype(str) + "-" + rk.month.astype(str), "c": ch})[mr]
    print("2026 rms change by airport-month", json.dumps(t.groupby("k").c.agg(lambda z: round(float(np.sqrt((z ** 2).mean())), 1)).to_dict()))
    np.save(ROOT / "work/w2/ords_ranking.npy", candr)
    if a.out:
        print("sha256", O.write(rk.MVT_ID_mvt.to_numpy(), candr, a.out), a.out)


if __name__ == "__main__":
    main()
