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
from diverse_blend import load_member
from pair_blend import PairedBlend
from joint_blend import JointBlend
from stand_floor import apply_floor

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


def compose(b, *, route='R4', form='HB1', extension=True, noplan='v1', bet=False,
            practice='none', blend=None, c33=True, c55=False, pair_extension=False, head_swap=None):
    """Route -> head -> member blend -> no-plan -> bet -> practice -> C33 -> C55."""
    assert not b[ID].duplicated().any()
    assert isinstance(pair_extension,bool)
    assert not pair_extension or isinstance(blend,(PairedBlend,JointBlend)), 'Pair extension requires a signed pair or joint blend.'
    rule = b.rule.to_numpy(bool)
    ext = b.extension.to_numpy(bool)
    assert not (rule & ext).any()
    assert not (extension and form not in ('HB1','HB3u')), 'Extension requires HB1 or HB3u.'
    assert (form == 'HB3u') == (head_swap is not None), 'HB3u needs its checked swap manifest.'
    if head_swap is not None:
        assert route == 'R4mk' and extension and not bet
    p = b['F'].to_numpy(float).copy()
    if route in ('R4A3', 'R4HB1', 'R4HB1mk'):
        # These admitted clock-member routes serve the head-rule rows only.
        assert np.array_equal(b[route].to_numpy(float)[ext], p[ext]), 'Clock-only route changes extension rows; scope needs admission.'
        rr = rule | (ext if extension else False)
        p[rr] = b[route].to_numpy(float)[rr]
    elif route != 'R4':
        p = b[route].to_numpy(float).copy()
    if head_swap is not None:
        p = head_swap.bases(b,p)
    elif form != 'F':
        weight = {'HA3': .3, 'HB1': .5}[form]
        p = head_blend(p, b[form].to_numpy(float), rule | (ext if extension else False), weight)
    if blend is not None:
        if head_swap is not None:
            p = head_swap.blend(blend,b[ID].to_numpy(),p,pair_extension)
        elif isinstance(blend,(PairedBlend,JointBlend)):
            p = blend.apply(b[ID].to_numpy(),p,enable_extension=pair_extension)
        else:
            p = blend.apply(b[ID].to_numpy(),p)
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
    if practice != 'none':
        assert practice in ('selected','confirmed','selected_c73')
        eligible = b.practice.to_numpy(bool).copy()
        assert np.isfinite(b.lo.to_numpy(float)[eligible]).all(), 'Practice row lacks a plan.'
        assert not (eligible & b.noplan.to_numpy(bool)).any()
        weights = b.practice_weight.to_numpy(float)
        points = b.practice_point.to_numpy(float)
        if practice == 'confirmed':
            eligible &= b.practice_status.eq('confirmed').to_numpy()
        assert np.isfinite(weights[eligible]).all() and ((weights[eligible] >= 0) & (weights[eligible] <= 1)).all()
        assert np.isfinite(points[eligible]).all() and (points[eligible] >= FLOOR).all()
        p[eligible] = np.maximum(FLOOR, p[eligible] + weights[eligible]*(points[eligible]-p[eligible]))
    if practice == 'selected_c73':
        route_key=b.c73.to_numpy(bool)
        assert not (route_key & b.practice.to_numpy(bool)).any(), 'Practice keys overlap.'
        assert (b.airport.to_numpy()[route_key]=='LIRF').all()
        assert not b.c73_U.to_numpy(bool)[route_key].any(), 'Certain-real row in route practice.'
        assert (b.c73_U_count.to_numpy(float)[route_key]<2).all()
        assert (b.c73_rate.to_numpy(float)[route_key]>=.98).all()
        assert (b.c73_n.to_numpy(float)[route_key]>=5).all()
        assert (b.c73_airline_rate.to_numpy(float)[route_key]<.9).all()
        w=b.c73_weight.to_numpy(float);point=b.c73_point.to_numpy(float)
        assert np.isfinite(w[route_key]).all() and ((w[route_key]>0)&(w[route_key]<=1)).all()
        assert np.isfinite(point[route_key]).all() and (point[route_key]>=FLOOR).all()
        p[route_key]=np.maximum(FLOOR,p[route_key]+w[route_key]*(point[route_key]-p[route_key]))
    if c33:
        p = clip(p, b.lo.to_numpy(float), b.hi.to_numpy(float))
    if c55:
        assert c33, 'C55 requires C33 first.'
        p = apply_floor(p,b.stand_floor.to_numpy(float),b.airport.to_numpy(),
                        b.lo.to_numpy(float),b.hi.to_numpy(float))
    assert np.isfinite(p).all() and (p >= FLOOR).all()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--bundle-sha256', required=True)
    ap.add_argument('--route', default='R4', choices=['R4','R4mk','R4A3','R4HB1','R4HB1mk'])
    ap.add_argument('--form', default='HB1', choices=['F','HA3','HB1','HB3u'])
    ap.add_argument('--no-extension', action='store_true')
    ap.add_argument('--noplan', default='v1', choices=['none','v1','v2','v3cal'])
    ap.add_argument('--bet', action='store_true')
    ap.add_argument('--practice', default='none', choices=['none','selected','confirmed','selected_c73'])
    ap.add_argument('--blend-manifest', type=Path)
    ap.add_argument('--blend-manifest-sha256')
    ap.add_argument('--pair-extension', choices=['off','proxy-2'], default='off',
                    help='Enable pair extension only with a pinned positive valid proxy read 2 and admitted scope.')
    ap.add_argument('--c55', action='store_true', help='Apply the admitted stand re-use floor after C33.')
    ap.add_argument('--head-swap-manifest', type=Path)
    ap.add_argument('--head-swap-manifest-sha256')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    assert not a.out.exists(), 'Write-once output exists.'
    assert hashlib.sha256(a.bundle.read_bytes()).hexdigest() == a.bundle_sha256, 'Bundle hash changed.'
    b = pd.read_parquet(a.bundle)
    assert len(b) == 344841, 'A ranking entry must have 344,841 rows.'
    template = pd.read_parquet(Path(__file__).resolve().parents[3]/'data/raw/submitting.parquet', columns=[ID])
    assert np.array_equal(b[ID].to_numpy(), template[ID].to_numpy()), 'Ranking template order differs.'
    assert b[ID].dtype == np.dtype('float64')
    assert bool(a.blend_manifest) == bool(a.blend_manifest_sha256)
    blend = load_member(a.blend_manifest, a.blend_manifest_sha256, b, 'ranking') if a.blend_manifest else None
    assert bool(a.head_swap_manifest) == bool(a.head_swap_manifest_sha256)
    swap = None
    if a.head_swap_manifest:
        from head_swap import load_head_swap
        from stack6 import check_build_ready
        check_build_ready(a.out)
        assert a.form == 'HB3u' and a.route == 'R4mk' and not a.no_extension
        assert a.noplan == 'v3cal' and a.practice == 'selected_c73' and a.c55 and a.pair_extension == 'proxy-2' and not a.bet
        swap = load_head_swap(a.head_swap_manifest,a.head_swap_manifest_sha256,b,blend,'ranking')
    p = compose(b, route=a.route, form=a.form, extension=not a.no_extension, noplan=a.noplan,
                bet=a.bet, practice=a.practice, blend=blend, c55=a.c55,pair_extension=a.pair_extension=='proxy-2',head_swap=swap)
    # The reviewed files use float64 IDs and values. Preserve the canonical bundle ID type.
    out = pd.DataFrame({ID:b[ID], COL:p})
    out.to_parquet(a.out, index=False)
    back = pd.read_parquet(a.out)
    assert np.array_equal(back[ID], b[ID]) and np.array_equal(back[COL], p)
    assert back[ID].dtype == back[COL].dtype == np.dtype('float64')
    assert np.array_equal(clip(back[COL].to_numpy(), b.lo.to_numpy(), b.hi.to_numpy()), back[COL].to_numpy())
    if a.c55:
        assert np.array_equal(apply_floor(p,b.stand_floor.to_numpy(),b.airport.to_numpy(),
                                         b.lo.to_numpy(),b.hi.to_numpy()),p)


if __name__ == '__main__':
    main()
