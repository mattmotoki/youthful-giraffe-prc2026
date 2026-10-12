"""One saved-vector assembly path for held rows, 10-04 stacks and the final entry.

The bundle supplies approved route/head vectors and row masks. No score or board value
enters this module. A final refit must replace the input bundle, not these transforms.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ID, COL = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'
FLOOR = 60.0


def head_blend(base, head, rule, weight):
    out = np.asarray(base, dtype=np.float64).copy()
    assert np.isfinite(head[rule]).all()
    out[rule] = np.maximum(FLOOR, (1-weight)*out[rule] + weight*np.maximum(FLOOR, head[rule]))
    return out


def clip(pred, lo, hi):
    joined = np.isfinite(lo)
    assert np.array_equal(joined, np.isfinite(hi))
    assert (hi[joined] >= FLOOR).all(), 'The floor and join bracket conflict.'
    out = pred.copy()
    out[joined] = np.maximum(FLOOR, np.clip(out[joined], lo[joined], hi[joined]))
    assert ((out[joined] >= lo[joined]) & (out[joined] <= hi[joined])).all()
    return out


def compose(b, *, route='R4', form='HB1', extension=True, noplan='v1', bet=False, c33=True):
    """Route -> head -> no-plan form -> optional bet -> C33; no whole-file overlay."""
    assert not b[ID].duplicated().any()
    rule = b.rule.to_numpy(bool)
    ext = b.extension.to_numpy(bool)
    assert not (rule & ext).any()
    assert not (extension and form != 'HB1'), 'Extension requires HB1.'
    p = b['F'].to_numpy(float).copy()
    if route in ('R4A3', 'R4HB1', 'R4HB1mk'):
        # These admitted clock-member routes serve the head-rule rows only.
        rr = rule | (ext if extension else False)
        p[rr] = b[route].to_numpy(float)[rr]
    elif route != 'R4':
        p = b[route].to_numpy(float).copy()
    if form != 'F':
        weight = {'HA3': .3, 'HB1': .5}[form]
        p = head_blend(p, b[form].to_numpy(float), rule | (ext if extension else False), weight)
    if noplan != 'none':
        n = b.noplan.to_numpy(bool)
        assert not np.isfinite(b.lo.to_numpy(float)[n]).any(), 'No-plan row has a plan.'
        assert np.array_equal(p[n], b.np_base.to_numpy(float)[n]), 'No-plan cache base changed; owner must rebuild it.'
        p[n] = b['np_'+noplan].to_numpy(float)[n]
        if bet:
            q = b.bet.to_numpy(bool)
            assert not (q & ~n).any()
            p[q] = b.np_bet.to_numpy(float)[q]
    else:
        assert not bet, 'Bet requires the no-plan form.'
    if c33:
        p = clip(p, b.lo.to_numpy(float), b.hi.to_numpy(float))
    assert np.isfinite(p).all() and (p >= FLOOR).all()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--bundle-sha256', required=True)
    ap.add_argument('--route', default='R4', choices=['R4','R4mk','R4A3','R4HB1','R4HB1mk'])
    ap.add_argument('--form', default='HB1', choices=['F','HA3','HB1'])
    ap.add_argument('--no-extension', action='store_true')
    ap.add_argument('--noplan', default='v1', choices=['none','v1','v2'])
    ap.add_argument('--bet', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), 'Write-once output exists.'
    assert hashlib.sha256(a.bundle.read_bytes()).hexdigest() == a.bundle_sha256, 'Bundle hash changed.'
    b = pd.read_parquet(a.bundle)
    assert len(b) == 344841, 'A ranking entry must have 344,841 rows.'
    p = compose(b, route=a.route, form=a.form, extension=not a.no_extension, noplan=a.noplan, bet=a.bet)
    # The reviewed files use float64 IDs and values. Preserve the canonical bundle ID type.
    out = pd.DataFrame({ID:b[ID], COL:p})
    out.to_parquet(a.out, index=False)
    back = pd.read_parquet(a.out)
    assert np.array_equal(back[ID], b[ID]) and np.array_equal(back[COL], p)


if __name__ == '__main__':
    main()
