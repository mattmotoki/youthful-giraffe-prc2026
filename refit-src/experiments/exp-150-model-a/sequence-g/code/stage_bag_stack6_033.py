"""Freeze stack6 held terms and prove parent parity; no labels or score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

E=Path(__file__).resolve().parents[1]
ID='MVT_ID_mvt'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def tail(b,p,clip,apply_floor):
    """Frozen assembly order: no-plan, C39, C73, C33, C55."""
    p=p.copy();n=b.noplan.to_numpy(bool)
    p[n]=b.np_v3cal.to_numpy(float)[n]
    for mask,w,point in [('practice','practice_weight','practice_point'),('c73','c73_weight','c73_point')]:
        on=b[mask].to_numpy(bool);weight=b[w].to_numpy(float);value=b[point].to_numpy(float)
        p[on]=np.maximum(60,p[on]+weight[on]*(value[on]-p[on]))
    p=clip(p,b.lo.to_numpy(float),b.hi.to_numpy(float))
    return apply_floor(p,b.stand_floor.to_numpy(float),b.airport.to_numpy(),
                       b.lo.to_numpy(float),b.hi.to_numpy(float))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('Stage output exists')
    review=json.loads(a.review.read_text())
    if review['verdict']!='PASS' or review['stage_source_sha256']!=sha(Path(__file__)):
        raise ValueError('Stage source not reviewed')
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    pins=json.loads((E/'results/BAG-STACK-PINS-033.json').read_text())
    expected_repo=Path(os.path.commonpath(list(pins['source_sha256']))).parents[1]
    if a.repo.resolve()!=expected_repo.resolve():raise ValueError('Parent repository differs from pins')
    archive=E/'bag-stack-source-033';source=a.repo/'experiments/exp-169-stack-plan'
    # The exact archived modules are used, even if stack7 changes shared code.
    for p,h in pins['source_sha256'].items():
        p=Path(p);candidate=archive/p.name if p.parent.name=='code' else p
        if sha(candidate)!=h:raise ValueError('Frozen stack6 input changed: '+str(candidate))
    sys.path.insert(0,str(archive));sys.dont_write_bytecode=True
    from assembly import compose,clip
    from stand_floor import apply_floor
    from diverse_blend import load_member
    import head_swap as H
    H.ROOT=a.repo.resolve()
    joint=source/'inputs/stack5-016/joint.json';swap=source/'inputs/stack6-021/head-swap.json'
    b=pd.read_parquet(source/'inputs/stack5-016/bundle-fold.parquet',use_threads=False)
    member=load_member(joint,sha(joint),b,'fold')
    head=H.load_head_swap(swap,sha(swap),b,member,'fold')
    cfg=dict(route='R4mk',form='HB3u',noplan='v3cal',practice='selected_c73',
             c55=True,pair_extension=True,head_swap=head,blend=member)
    parent=compose(b,**cfg)
    saved=pd.read_parquet(source/'results/stack6-held-021.parquet',use_threads=False)
    np.testing.assert_array_equal(b[ID],saved[ID]);np.testing.assert_array_equal(parent,saved.stack6)
    rebound=H._rebase(member,b[ID].to_numpy(),head.old_base,head.new_base)
    fallback=rebound.fallback.apply(b[ID].to_numpy(),head.new_base,enable_extension=True)
    blend=fallback.copy();scope=member.scope
    delta=np.zeros(len(b))
    for j in range(4):
        active=scope&(member.weights[:,j]!=0)
        delta[active]+=member.weights[active,j]*(member.values[active,j]-head.new_base[active])
    blend[scope]=np.maximum(60,head.new_base[scope]+delta[scope])
    np.testing.assert_array_equal(blend,rebound.apply(b[ID].to_numpy(),head.new_base,enable_extension=True))
    np.testing.assert_array_equal(tail(b,blend,clip,apply_floor),parent)
    b=b.copy();b['G_base']=head.new_base;b['G_fallback']=fallback;b['G_parent']=parent
    b['G_scope']=scope
    # Fit convention from stack5: C39/C33, but not C73/C55. Store that separately.
    fit_base=head.new_base.copy();n=b.noplan.to_numpy(bool);fit_base[n]=b.np_v3cal.to_numpy(float)[n]
    on=b.practice.to_numpy(bool);weight=b.practice_weight.to_numpy(float);point=b.practice_point.to_numpy(float)
    fit_base[on]=np.maximum(60,fit_base[on]+weight[on]*(point[on]-fit_base[on]))
    b['G_fit_base']=clip(fit_base,b.lo.to_numpy(float),b.hi.to_numpy(float))
    for j,name in enumerate(['g','mlp','sequence','control']):
        b['G_'+name]=member.values[:,j];b['G_weight_'+name]=member.weights[:,j]
    if len(b)!=344419 or not b[ID].is_unique:raise ValueError('Parent IDs differ')
    a.out.mkdir(parents=True);path=a.out/'TERMS.parquet';b.to_parquet(path,index=False)
    rec=dict(status='PASS',rows=len(b),primary_rows=int(scope.sum()),parent_parity_exact=True,
             independent_member_and_tail_exact=True,labels_loaded=False,quality_score_read=False,
             source_sha256=sha(Path(__file__)),review_sha256=sha(a.review),
             stack_pins_sha256=sha(E/'results/BAG-STACK-PINS-033.json'),
             plan_sha256=sha(E/'BAG-PLAN-033.json'),terms_sha256=sha(path))
    (a.out/'STAGE.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
