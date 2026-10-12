"""Compose candidate steps on top of v64 and write the 2026 file.

Steps (each switchable):
- ext:  the January 2026 extension rows (LEMD, LFPG clock rows without 2025 support) get the
        pooled clock stack of five members on the head base instead of the fallback pair.
        No held rows exist; the held proxy is iso/proxy_ext.py.
- long: the long-route rows (typed, AOBT_3, route >= 2,400 s, outside the stack scope) get a
        pooled member stack: weights fitted on the other held month (2026: both months), half step.
Every step changes the primary blend; the original rule chain then runs (iso/chain.py).

    .venv/bin/python iso/candidate.py --steps ext,long --out outputs/cand.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso"))
import chain  # noqa: E402
import finalmode as FM  # noqa: E402
import score as S  # noqa: E402

ID, COL = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
RAW5 = ["raw_g", "raw_mlp", "raw_sequence021", "raw_sequence039", "m_dark18"]
LONG_STEP = 0.5


def raw_members(split):
    return pd.read_parquet(ROOT / f"work/chain/{split}.parquet", columns=RAW5).to_numpy(float)


def step_ext(st, base, prim):
    c = st["contexts"]["ranking"]; b = c["b"]
    e = b.extension.to_numpy(bool)
    rec = st["pp"]["clock"]
    assert rec["columns"] == ["g", "mlp", "sequence021", "sequence_new", "dark"]
    coef = np.asarray(rec["coefficients"], float)
    M = raw_members("ranking")
    hb = base["ranking"]["head"]
    assert np.isfinite(M[e]).all() and (M[e] >= 60).all()
    new = prim["ranking"].copy()
    new[e] = np.maximum(60, hb[e] + (M[e] - hb[e, None]) @ coef)
    prim["ranking"] = new
    return {"rows_2026": int(e.sum()), "coefficients": coef.tolist()}


def lirf_cell(st):
    """The LIRF January 2026 clock rows outside the scope and the extension (route only in v64).
    Final mode: February takes January's scope (season map)."""
    c = st["contexts"]["ranking"]; b = c["b"]
    return ((b.airport.eq("LIRF") & b.month.isin(FM.JAN_SCOPE) & b.cls.eq("clock")).to_numpy()
            & ~c["scope"] & ~b.extension.to_numpy(bool))


def long_mask(st, split):
    """Long-route rows; on 2026 the LIRF January clock cell belongs to step_lirf (29 rows overlap)."""
    c = st["contexts"][split]; b = c["b"]
    feat = pd.read_parquet(ROOT / f"work/feat/{split}.parquet", columns=["nm_taxi"])
    has_aobt = np.isfinite(feat.nm_taxi.to_numpy())
    m = (~c["scope"] & ~b.extension.to_numpy(bool) & b.cls.ne("noac").to_numpy() & has_aobt
         & (b.R4mk.to_numpy() >= 2400) & ~b.noplan.to_numpy(bool))
    return m & ~lirf_cell(st) if split == "ranking" else m


def _wls(X, r, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)[0]


def step_long(st, base, prim, ycap=20000.0):
    c = st["contexts"]["fold"]; b = c["b"]
    y = st["rframe"].y.to_numpy(float); w = st["exposure"].weight.to_numpy(float) * 1e6
    m = long_mask(st, "fold"); month = b.month.to_numpy()
    hb = base["fold"]["head"]; X = raw_members("fold") - hb[:, None]; r = y - hb
    fit = m & (w > 0) & (y < ycap)
    betas = {}
    new = prim["fold"].copy()
    for held in (1, 7):
        beta = _wls(X[fit & (month != held)], r[fit & (month != held)], w[fit & (month != held)])
        te = m & (month == held)
        new[te] = np.maximum(60, hb[te] + LONG_STEP * (X[te] @ beta))
        betas[str(held)] = beta.tolist()
    prim["fold"] = new
    pooled = _wls(X[fit], r[fit], w[fit]); betas["pooled"] = pooled.tolist()
    mr = long_mask(st, "ranking"); hr = base["ranking"]["head"]; Xr = raw_members("ranking") - hr[:, None]
    assert np.isfinite(Xr[mr]).all()
    new = prim["ranking"].copy()
    new[mr] = np.maximum(60, hr[mr] + LONG_STEP * (Xr[mr] @ pooled))
    prim["ranking"] = new
    return {"rows_fold": int(m.sum()), "rows_2026": int(mr.sum()), "step": LONG_STEP, "betas": betas}


LIRF_STEP = 0.5


def step_lirf(st, base, prim):
    """LIRF January 2026 clock rows (no 2025 support; the clock head misreads LIRF placeholders):
    the ordinary stack of six members on the route base, pooled ordinary weights, half step."""
    m = lirf_cell(st)
    rec = st["pp"]["ordinary"]
    assert rec["columns"] == ["g", "mlp", "sequence021", "sequence_new", "dark", "control"]
    coef = np.asarray(rec["coefficients"], float)
    M = pd.read_parquet(ROOT / "work/chain/ranking.parquet", columns=RAW5 + ["raw_row6"]).to_numpy(float)
    route = base["ranking"]["route"]
    assert np.isfinite(M[m]).all() and np.array_equal(prim["ranking"][m], route[m])
    new = prim["ranking"].copy()
    new[m] = np.maximum(60, route[m] + LIRF_STEP * ((M[m] - route[m, None]) @ coef))
    prim["ranking"] = new
    return {"rows_2026": int(m.sum()), "step": LIRF_STEP, "coefficients": coef.tolist()}


STEPS = {"ext": step_ext, "long": step_long, "lirf": step_lirf}

HELD_FILES = ("training_2025-01-01_2025-02-01", "training_2025-07-01_2025-08-01")


def heli_groups(exclude_held: bool) -> pd.DataFrame:
    """(airport, actype) groups of typed no-join departures with labels near zero."""
    import glob
    parts = [pd.read_parquet(p, columns=["airport", "actype", "joined", "y", "mvt_month", "dom", "file"])
             for p in sorted(glob.glob(str(ROOT / "work/feat/months/*.parquet")))]
    t = pd.concat(parts, ignore_index=True)
    t = t[(t.joined == 0) & (t.actype != "None") & t.y.notna()]
    if exclude_held:
        t = t[~t.file.isin(HELD_FILES) & ~t.mvt_month.isin([1, 7])]
    g = t.groupby(["airport", "actype"]).y.agg(n="size", mean="mean", q90=lambda v: v.quantile(0.9))
    return g[(g.n >= 5) & (g.q90 < 60)]


def post_heli(split, out_split, base):
    """Typed no-join rows of a near-zero group take the group mean label (floor 0 s, not 60 s)."""
    feat = pd.read_parquet(ROOT / f"work/feat/{split}.parquet", columns=["airport", "actype", "joined"])
    g = heli_groups(exclude_held=split == "fold")
    key = pd.MultiIndex.from_arrays([feat.airport, feat.actype])
    val = g["mean"].reindex(key).to_numpy()
    v = out_split["v64"].copy()
    m = (feat.joined.to_numpy() == 0) & np.isfinite(val) & (out_split["lower"] <= 60)
    v[m] = np.clip(val[m], 0.0, out_split["upper"][m])
    out_split["v64"] = v
    return {"rows": int(m.sum()), "groups": [f"{a}|{t}" for a, t in g.index]}


NOAC_R, NOAC_STEP = 900.0, 0.25


def _noac_mix(split, v):
    """LIRF no-type rows whose value is a mixture between the real mode and the placeholder point."""
    st = chain.load(); b = st["contexts"][split]["b"]
    ms = pd.read_parquet(ROOT / f"work/feat/{split}.parquet", columns=["mvt_sched"]).mvt_sched.to_numpy(float)
    den = np.where(np.abs(ms - NOAC_R) < 1, 1.0, ms - NOAC_R)
    p = (v - NOAC_R) / den
    mix = b.cls.eq("noac").to_numpy() & b.airport.eq("LIRF").to_numpy() & (p > 0.03) & (p < 0.97) & (ms > NOAC_R + 600)
    return mix


def _shrink_fit(v, y, mask):
    x = v[mask] - NOAC_R; t = y[mask] - NOAC_R
    return float((x * t).sum() / (x * x).sum())


def post_noac(split, out_split, base):
    """v = R + lam (v - R) on the mixture rows; lam = 1 - STEP (1 - lam_fit), lam_fit from the other held month (2026: both).
    lam_fit is fitted on the held v64 values (no other step changes the LIRF no-type rows)."""
    st = chain.load()
    vf = base["fold"]["v64"]; y = st["rframe"].y.to_numpy(float); bf = st["contexts"]["fold"]["b"]
    mixf = _noac_mix("fold", vf) & (y < 80000)
    month = bf.month.to_numpy()
    v = out_split["v64"].copy()
    mix = _noac_mix(split, v)
    lams = {}
    if split == "fold":
        for held in (1, 7):
            lam_fit = _shrink_fit(vf, y, mixf & (month != held))
            lam = 1 - NOAC_STEP * (1 - lam_fit); lams[str(held)] = lam
            te = mix & (month == held)
            v[te] = NOAC_R + lam * (v[te] - NOAC_R)
    else:
        lam_fit = _shrink_fit(vf, y, mixf)
        lam = 1 - NOAC_STEP * (1 - lam_fit); lams["pooled"] = lam
        v[mix] = NOAC_R + lam * (v[mix] - NOAC_R)
    v[mix] = np.maximum(60, np.clip(v[mix], out_split["lower"][mix], out_split["upper"][mix]))
    out_split["v64"] = v
    return {"rows": int(mix.sum()), "lambda": lams}


def post_roll(split, out_split, base):
    """LIRF no-type rows with 50,000 <= ms < 86,400 s and a take-off one Rome-local date after the
    schedule: expectation over the placeholder and rollover points, p(rollover | ms) from
    iso/roll_step.py (held: ten non-held months; 2026: all twelve)."""
    import roll_step as R
    st = chain.load(); b = st["contexts"][split]["b"]
    ms = pd.read_parquet(ROOT / f"work/feat/{split}.parquet", columns=["mvt_sched"]).mvt_sched.to_numpy(float)
    lr, off, _ = R.model(exclude_held=split == "fold")
    cross = R.cross_rows(split, b[ID].to_numpy())
    m = (b.cls.eq("noac") & b.airport.eq("LIRF")).to_numpy() & cross & (ms >= R.ROLL_LO) & (ms < R.DAY)
    v = out_split["v64"].copy()
    p, val = R.value(lr, off, ms[m])
    v[m] = np.maximum(60, np.clip(val, out_split["lower"][m], out_split["upper"][m]))
    out_split["v64"] = v
    return {"rows": int(m.sum()), "offset": off, "p": np.round(p, 3).tolist(), "ms": ms[m].tolist()}


POST = {"heli": post_heli, "noac": post_noac, "roll": post_roll}


def build(steps):
    st = chain.load(); base = chain.run()
    prim = {s: base[s]["primary"].copy() for s in ("fold", "ranking")}
    info = {}
    for name in steps:
        if name in STEPS:
            info[name] = STEPS[name](st, base, prim)
    out = chain.run(primary=prim)
    for name in steps:
        if name in POST:
            info[name] = {s: POST[name](s, out[s], base) for s in ("fold", "ranking")}
    return base, out, info


def write(out, path: Path):
    st = chain.load()
    ids = st["contexts"]["ranking"]["b"][ID].to_numpy("float64")
    tmpl = pd.read_parquet(FM.submitting_path(ROOT / "input-root/data/raw"))
    assert np.array_equal(tmpl[ID].to_numpy("float64"), ids)
    v = out["ranking"]["v64"]
    assert np.isfinite(v).all() and len(v) == FM.n_rows() and v.dtype == np.float64
    q = pd.DataFrame({ID: ids, COL: v.astype("float64")})
    buf = io.BytesIO(); q.to_parquet(buf, index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="ext,long")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    steps = [s for s in a.steps.split(",") if s]
    base, out, info = build(steps)
    d = S.fold()
    res = S.score(out["fold"]["v64"], base=base["fold"]["v64"], d=d)
    print("held", S.fmt(res))
    print("airport", res["airport"])
    dv = out["ranking"]["v64"] - base["ranking"]["v64"]
    ch = dv != 0
    print("2026 changed rows", int(ch.sum()), "squared change per ranking row %.2f" % float((dv ** 2).mean()),
          "mean change on changed %.2f" % (float(dv[ch].mean()) if ch.any() else 0.0))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "betas"} for k, v in info.items()}))
    if a.out:
        print("sha256", write(out, a.out), a.out)


if __name__ == "__main__":
    main()
