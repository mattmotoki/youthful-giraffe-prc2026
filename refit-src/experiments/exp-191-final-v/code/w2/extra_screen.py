"""Screens of a new member column against c5 (held, cross-month):
(a) L2C with the new member as an extra input, against L2C without it;
(b) on ordinary scope rows, c5 + beta (m - c5) with beta fitted on the other held month.

    python3 w2/extra_screen.py t1 [s1 ...]      (reads work/w2/<name>_fold.parquet, column <name>)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "iso")); sys.path.insert(0, str(ROOT / "w2"))
import base as B  # noqa: E402
import l2  # noqa: E402
import l2c  # noqa: E402
import score as S  # noqa: E402


def load_member(name, split):
    t = pd.read_parquet(ROOT / f"work/w2/{name}_{split}.parquet")
    ids = pd.read_parquet(ROOT / f"work/w2/chain_ids_{split}.parquet").MVT_ID_mvt.to_numpy()
    assert np.array_equal(t.MVT_ID_mvt.to_numpy(), ids)
    return t[name].to_numpy(float)


def main():
    names = sys.argv[1:]
    d = B.load("fold"); f = S.fold(); y = f.y.to_numpy(float); w = f.w.to_numpy(float)
    base = d.c5.to_numpy(float); lo = d.c5_lower.to_numpy(float); hi = d.c5_upper.to_numpy(float)
    month = d.month.to_numpy()
    mem = {n: load_member(n, "fold") for n in names}
    for n, v in mem.items():
        for cls in ("clock", "ordinary"):
            m = d.scope.to_numpy() & (d.cls == cls).to_numpy() & np.isfinite(v) & (w > 0)
            r1 = np.sqrt((w[m] * (y[m] - v[m]) ** 2).sum() / w[m].sum())
            r0 = np.sqrt((w[m] * (y[m] - base[m]) ** 2).sum() / w[m].sum())
            c = np.corrcoef((y - v)[m], (y - base)[m])[0, 1]
            print(f"{n} {cls}: wRMSE {r1:.1f} (c5 {r0:.1f}), residual corr with c5 {c:.3f}", flush=True)
        # (b) linear column on ordinary scope rows
        m = d.scope.to_numpy() & (d.cls == "ordinary").to_numpy() & np.isfinite(v)
        x = v - base; r = y - base; ok = m & (w > 0) & (np.abs(r) < 3000)
        cand = base.copy()
        betas = {}
        for held in (1, 7):
            tr = ok & (month != held); te = m & (month == held)
            beta = (w[tr] * x[tr] * r[tr]).sum() / (w[tr] * x[tr] ** 2).sum(); betas[held] = round(float(beta), 3)
            cand[te] = np.maximum(60, np.clip(base[te] + beta * x[te], lo[te], hi[te]))
        res = S.score(cand, base=base)
        print(f"{n} ordinary linear column betas {betas}:", S.fmt(res), flush=True)
        print("   airport", res["airport"], flush=True)
        for cls in ("clock",):
            m = d.scope.to_numpy() & (d.cls == cls).to_numpy() & np.isfinite(v)
            ok = m & (w > 0) & (np.abs(r) < 3000)
            cand = base.copy(); betas = {}
            for held in (1, 7):
                tr = ok & (month != held); te = m & (month == held)
                beta = (w[tr] * x[tr] * r[tr]).sum() / (w[tr] * x[tr] ** 2).sum(); betas[held] = round(float(beta), 3)
                cand[te] = np.maximum(60, np.clip(base[te] + beta * x[te], lo[te], hi[te]))
            res = S.score(cand, base=base)
            print(f"{n} clock linear column betas {betas}:", S.fmt(res), flush=True)
    # (a) L2C with and without the new members
    X = l2.frame(d)
    mask = l2c.masks(d, "fold")
    res_y = np.clip(y - base, -2000, 2000)
    for with_new in (False, True):
        Xw = X.copy()
        if with_new:
            for n, v in mem.items():
                Xw["d_" + n] = v - base
        corr = np.zeros(len(d))
        for held in (1, 7):
            tr = mask & (month != held) & (w > 0); te = mask & (month == held)
            corr[te] = l2c.fit_predict(Xw[tr], res_y[tr], w[tr], Xw[te])
        cand = np.maximum(60, np.clip(base + l2c.K * corr, lo, hi))
        res = S.score(cand, base=base)
        print(f"L2C with {names if with_new else 'none'}:", S.fmt(res), flush=True)
        print("   airport", res["airport"], flush=True)


if __name__ == "__main__":
    main()
