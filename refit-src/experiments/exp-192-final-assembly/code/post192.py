"""NOAC and ROLL of c5 on an assembly chain value (exp-192, finalplan, 2026-10-08).

The Z and R files apply these two steps on top of their chain value (desk answer to finalv's need 3: the Z part
holds v64 as released; the Z FILE applies NOAC and ROLL). The lines follow c5 in the export,
`iso/candidate.py` `_noac_mix`, `post_noac` and `post_roll` (finallirf ported the same steps in
`experiments/exp-187-final-lirf/code/final_lirf.py`):
- NOAC: rows of class noac at LIRF whose value is a mixture between the real mode and the placeholder point
  (p = (v - 900) / (ms - 900), 0.03 < p < 0.97, ms > 1,500 s). v = 900 + lam (v - 900), then
  max(60, clip(v, lower, upper)). lam = 1 - 0.25 (1 - lam_fit), with lam_fit fitted on the held fold values of the
  same chain on its mixture rows with y < 80,000 s: pooled for 2026; for a held month, on the other held month.
- ROLL: noac LIRF rows whose take-off is one Rome-local date after the schedule, with 50,000 <= ms < 86,400 s.
  v = max(60, clip(p (86,400 + offset) + (1 - p) ms, lower, upper)), with p and offset from `iso/roll_step.py`
  (ten non-held 2025 months for the held fold, all twelve for 2026).
ms = MVT - SCHED in seconds (c5 feature `mvt_sched`); lower and upper are the exploit frame bounds (c5 `chain.py`).
"""
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
NOAC_R, NOAC_STEP = 900.0, 0.25


def ms_of(table):
    return (table.MVT_TIME_UTC_mvt - table.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy(float)


def lirf_noac(b):
    return b.cls.eq('noac').to_numpy() & b.airport.eq('LIRF').to_numpy()


def mixture(b, v, ms):
    den = np.where(np.abs(ms - NOAC_R) < 1, 1.0, ms - NOAC_R)
    p = (v - NOAC_R) / den
    return lirf_noac(b) & (p > 0.03) & (p < 0.97) & (ms > NOAC_R + 600)


def shrink_fit(v, y, mask):
    x = v[mask] - NOAC_R; t = y[mask] - NOAC_R
    return float((x * t).sum() / (x * x).sum())


def noac(split, v, b, ms, lower, upper, fold):
    """`fold`: dict v, y, b, ms of the held fold of the same chain. Returns the values, the record and the rows the
    step serves (its eligible rows)."""
    mixf = mixture(fold['b'], fold['v'], fold['ms']) & (fold['y'] < 80000)
    month_f = fold['b'].month.to_numpy()
    out = np.asarray(v, float).copy()
    mix = mixture(b, out, ms)
    lams = {}
    if split == 'fold':
        for held in (1, 7):
            fit = mixf & (month_f != held)
            lam_fit = shrink_fit(fold['v'], fold['y'], fit)
            lam = 1 - NOAC_STEP * (1 - lam_fit)
            lams[str(held)] = {'fit_rows': int(fit.sum()), 'lam_fit': lam_fit, 'lam': lam}
            te = mix & (b.month.to_numpy() == held)
            out[te] = NOAC_R + lam * (out[te] - NOAC_R)
    else:
        lam_fit = shrink_fit(fold['v'], fold['y'], mixf)
        lam = 1 - NOAC_STEP * (1 - lam_fit)
        lams['pooled'] = {'fit_rows': int(mixf.sum()), 'lam_fit': lam_fit, 'lam': lam}
        out[mix] = NOAC_R + lam * (out[mix] - NOAC_R)
    out[mix] = np.maximum(60, np.clip(out[mix], lower[mix], upper[mix]))
    return out, {'rows': int(mix.sum()), 'lambda': lams}, mix


def cross(RS, paths, ids):
    """`roll_step.cross_rows` with explicit files: True for a LIRF no-type cross row, in ids order."""
    f = RS._frame(paths, with_y=False)
    s = pd.Series(f.cross.to_numpy(), index=f[ID].astype('int64').to_numpy())
    s = s[~s.index.duplicated()]
    return s.reindex(np.asarray(ids).astype('int64')).astype('boolean').fillna(False).to_numpy(bool)


def roll(split, v, b, ms, lower, upper, RS, is_cross):
    lr, off, t = RS.model(exclude_held=split == 'fold')
    m = lirf_noac(b) & is_cross & (ms >= RS.ROLL_LO) & (ms < RS.DAY)
    out = np.asarray(v, float).copy()
    p, val = RS.value(lr, off, ms[m])
    out[m] = np.maximum(60, np.clip(val, lower[m], upper[m]))
    return out, {'rows': int(m.sum()), 'offset': off, 'fit_rows': int(len(t)), 'rollovers': int(t.roll.sum()),
                 'coef': float(lr.coef_[0, 0]), 'intercept': float(lr.intercept_[0])}, m
