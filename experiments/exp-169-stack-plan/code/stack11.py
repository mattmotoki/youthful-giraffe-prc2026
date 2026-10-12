"""Unit026: same C71/C95/C98 recipes, residuals on fixed stack10, one freeze."""
import argparse,hashlib,io,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import stack10 as P
import stack9 as Q
import exploit_rules as R
from assembly_exploit import compose_exploit
from verdict026 import require_pass
from assembly import ID,COL
from diverse_blend import vector_digest
E=P.E;ROOT=P.R;D=E/'inputs/stack11-026';O=E/'results'
SPEC=E/'SPEC-026.md';READ=O/'stack11-read-026.json';FORECAST=O/'stack11-forecast-frozen-026.json';MANIFEST=D/'MANIFEST.json'
CANDIDATE=E/'exp169-stack11-R4mks-C71-C95-C98.parquet';PARENT=P.CANDIDATE
PARENT_SHA='f7c4904d3193ce5072820f944f04aca8449e7ebd196c696c49f627f07f2c16fb'
PARENT_REVIEW=ROOT/'docs/status/2026-10-03-exp169-stack10-review/review-stack10.log'
CODE=list(dict.fromkeys([*P.CODE,*Q.CODE,'build_stack10_025.py','verdict026.py','stack11.py']))
sha=P.sha;js=P.js;now=P.now;write=P.write;record=P.record;PINS={}

def pin(p):
    p=Path(p).resolve();PINS[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=sha(p);return p

def parquet(f,p):
    buf=io.BytesIO();f.to_parquet(buf,index=False);body=buf.getvalue()
    if p.exists():assert p.read_bytes()==body
    else:p.write_bytes(body)
    pin(p)

def compose11(b,obj,**kw):
    z=compose_exploit(b,obj,**kw)
    return {('stack10' if k=='stack8' else 'stack11' if k=='stack9' else k):v for k,v in z.items()}

def context(split):
    b,n,m,a,t,old=P.load_route(split);kw=dict(blend=m,head_swap=t,**P.S8.CFG)
    parent=P.compose(n,**kw)
    assert vector_digest(n[ID],parent)==js(P.READ)['prediction_digests'][split]
    return n,parent,kw

def compact(r):
    return {k:r[k] for k in ['G','interval','views','G_without_top_days','without5_interval','n_rows','n_changed','step_min','step_max','top_airport_days']}

def score(f,old,new):return compact(Q.registered(f,old,new))

def no_lirf(f,old,new):
    z=np.asarray(new).copy();mask=f.ap.eq('LIRF').to_numpy();z[mask]=old[mask]
    return score(f,old,z)

def old_unchanged():
    files=js(O/'stack11-before-026.json')['files']
    for p,h in files.items():assert sha(ROOT/p)==h,p
    return len(files)

def sources():
    assert sha(PARENT)==PARENT_SHA;P.sources()
    require_pass(ROOT/'docs/status/2026-10-02-exp169-stack9-review/review-stack9.log')
    for p in [SPEC,PARENT,P.READ,P.FORECAST,P.MANIFEST,Q.CANDIDATE,Q.READ,Q.FORECAST,Q.MANIFEST,
              ROOT/'docs/status/2026-10-02-exp169-stack9-review/review-stack9.log',O/'stack9-c71-hold-024.json',O/'stack10-held-025.parquet',O/'stack9-held-024.parquet',Q.D/'statistics.json',ROOT/'data/raw/ranking.parquet']:
        pin(p)
    for p,h in js(P.READ)['source_sha256'].items():assert sha(ROOT/p)==h;pin(ROOT/p)
    for p,h in js(Q.MANIFEST)['source_sha256'].items():assert sha(ROOT/p)==h;pin(ROOT/p)
    for n,h in js(Q.READ)['code_sha256'].items():assert sha(E/'code'/n)==h
    for p,h in js(P.READ)['code_sha256'].items():assert sha(E/'code'/p)==h
    for n in CODE:pin(E/'code'/n)
    for split in ['fold','ranking']:pin(ROOT/js(Q.MANIFEST)['splits'][split]['path'])
    hold=js(O/'stack9-c71-hold-024.json');assert hold['passed']
    for p,h in hold['source_sha256'].items():assert sha(ROOT/p)==h;pin(ROOT/p)
    for name,rate in [('a',.998),('b_outside_a',.95)]:assert hold['parts'][name]['six_second_rate']>=rate
    return hold

def old_rule_parity(f,rank):
    stats=js(Q.D/'statistics.json');out={}
    for split,frame in [('fold',f),('ranking',rank)]:
        b,parent,kw=Q.parent_context(split)
        obj,st,v,n=R.prepare_rule(f,Q.parent_context('fold')[1],frame,parent,b,pooled=split=='ranking')
        old=Q.load_rule(split,parent);saved=pd.read_parquet(ROOT/js(Q.MANIFEST)['splits'][split]['path'])
        pd.testing.assert_frame_equal(Q.table(obj,v,n),saved,check_exact=True)
        assert st==stats[split]
        pred=compose_exploit(b,obj,**kw)['stack9'];assert vector_digest(b[ID],pred)==js(Q.READ)['prediction_digests'][split]
        if split=='ranking':
            buf=io.BytesIO();pd.DataFrame({ID:b[ID].to_numpy('float64'),COL:pred.astype('float64')}).to_parquet(buf,index=False)
            assert hashlib.sha256(buf.getvalue()).hexdigest()==sha(Q.CANDIDATE)
        else:
            got=score(f,parent,pred);want=js(Q.READ)['reads']['joint_vs_stack8']
            assert got['G']==want['G'] and got['interval']==want['interval'] and got['G_without_top_days']==want['G_without_top_days']
            out['joint_G']=got['G']
        out[split]={'rule_vector_exact':True,'statistics_exact':True,'prediction_digest_exact':True}
    return out

def load_rule(split,parent):
    rec=js(MANIFEST)['splits'][split];p=ROOT/rec['path'];assert sha(p)==rec['sha256']
    q=pd.read_parquet(p);assert vector_digest(q[ID],parent)==rec['parent_digest']
    return R.Exploit(q[ID].to_numpy(),rec['parent_digest'],q.c71.to_numpy(),q.g71.to_numpy(),q.step95.to_numpy(),q.scope95.to_numpy(),q.step98.to_numpy(),q.scope98.to_numpy(),q.lower.to_numpy(),q.upper.to_numpy())

def prepare():
    assert not READ.exists() and not FORECAST.exists() and not CANDIDATE.exists();D.mkdir(exist_ok=True)
    start=now();hold=sources();b,parent,kw=context('fold');q,parent26,kw26=context('ranking')
    saved=pd.read_parquet(PARENT);np.testing.assert_array_equal(saved[ID],q[ID]);np.testing.assert_array_equal(saved[COL],parent26)
    buf=io.BytesIO();pd.DataFrame({ID:saved[ID],COL:parent26}).to_parquet(buf,index=False);assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT_SHA
    held=pd.read_parquet(O/'stack10-held-025.parquet');np.testing.assert_array_equal(held[ID],b[ID]);np.testing.assert_array_equal(held.stack10,parent)
    f=pd.read_parquet(Q.D/'frame.parquet');np.testing.assert_array_equal(f[ID],b[ID]);np.testing.assert_array_equal(f.y,Q.GS.labels(b[ID].to_numpy('int64')))
    raw=pd.read_parquet(Q.D/'raw-fold.parquet');g,g71,a,c,_=Q.stand_inputs(raw,f[ID])
    for actual,want in [(g,f.g),(g71,f.g71),(a|c,f.c71)]:np.testing.assert_array_equal(actual,want)
    rank=Q.ranking_frame(q);parquet(rank,D/'frame-ranking.parquet')
    proof=old_rule_parity(f,rank);print('PARENT/RULE PARITY',proof,flush=True)
    obj,stats,view,n=R.prepare_rule(f,parent,f,parent,b);steps=compose11(b,obj,**kw)
    np.testing.assert_array_equal(compose11(b,obj,enabled=False,**kw)['stack11'],parent)
    independent=np.clip(np.maximum(60,np.maximum(60,R.c71_parent(f,parent)+obj.step95)+obj.step98),obj.lower,obj.upper)
    np.testing.assert_array_equal(independent,steps['stack11'])
    reads={};previous=parent
    for key in ['C71','C95','C98','stack11']:
        reads[key+'_step']=score(f,previous,steps[key]);previous=steps[key]
    reads['C98_direct']=score(f,parent,np.maximum(60,parent+obj.step98))
    reads['joint_vs_stack10']=score(f,parent,steps['stack11'])
    old8=held.stack8.to_numpy();old9=pd.read_parquet(O/'stack9-held-024.parquet').stack9.to_numpy()
    reads['route_vs_stack8']=score(f,old8,parent)
    reads['stack11_vs_stack9']=score(f,old9,steps['stack11']);reads['stack11_vs_stack8']=score(f,old8,steps['stack11'])
    for key,base,new in [('joint_vs_stack10',parent,steps['stack11']),('route_vs_stack8',old8,parent),('stack11_vs_stack9',old9,steps['stack11']),('stack11_vs_stack8',old8,steps['stack11'])]:
        reads[key+'_without_LIRF']=no_lirf(f,base,new)
    np.testing.assert_array_equal(steps['stack11'][f.ap.eq('LIRF')],parent[f.ap.eq('LIRF')])
    assert reads['joint_vs_stack10']==reads['joint_vs_stack10_without_LIRF']
    # The matched shadow has the same non-LIRF parent and the same rule increment.
    shadow_old=held.masked_stack10.to_numpy();shadow_new=steps['stack11'].copy()
    lirf=f.ap.eq('LIRF').to_numpy();shadow_new[lirf]=shadow_old[lirf]
    reads['shadow_rule_increment']=score(f,shadow_old,shadow_new)
    for key in ['G','G_without_top_days']:assert reads['shadow_rule_increment'][key]==reads['joint_vs_stack10'][key]
    # G intervals are identical; their derived RMSE gain fields have a different
    # local reference RMSE when the parent carries the LIRF shadow.
    for key in ['se','lo95','hi95','p_G_le_0']:
        assert reads['shadow_rule_increment']['interval'][key]==reads['joint_vs_stack10']['interval'][key]
    r95=reads['C95_step'];r98=reads['C98_direct'];positive=[k for k,v in r98['views']['airport'].items() if v>0]
    gates={'C71_hold':hold['passed'],'C95_without5_ge20':r95['G_without_top_days']>=20,'C95_lower95_gt0':r95['interval']['lo95']>0,
           'C98_direct_G_ge30':r98['G']>=30,'C98_direct_without5_gt0':r98['G_without_top_days']>0,
           'C98_not_EDDF_only_weak':not(positive==['EDDF'] and r98['without5_interval']['lo95']<=0),
           'C98_ordered_without5_gt0':reads['C98_step']['G_without_top_days']>0,'joint_without5_gt0':reads['joint_vs_stack10']['G_without_top_days']>0}
    gates['pass']=all(gates.values())
    rule26,stats26,v26,n26=R.prepare_rule(f,parent,rank,parent26,q,pooled=True);out26=compose11(q,rule26,**kw26)
    np.testing.assert_array_equal(compose11(q,rule26,enabled=False,**kw26)['stack11'],parent26)
    ind26=np.clip(np.maximum(60,np.maximum(60,R.c71_parent(rank,parent26)+rule26.step95)+rule26.step98),rule26.lower,rule26.upper)
    np.testing.assert_array_equal(ind26,out26['stack11'])
    splits={};counts={}
    for split,base,rule,v,nn,bundle,out in [('fold',parent,obj,view,n,b,steps),('ranking',parent26,rule26,v26,n26,q,out26)]:
        path=D/f'rules-{split}.parquet';parquet(Q.table(rule,v,nn),path)
        splits[split]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'parent_digest':rule.parent_digest}
        masks={'C71':rule.c71,'C95':rule.scope95,'C98':rule.scope98};changed=out['stack11']!=base
        counts[split]={'changed':int(changed.sum()),'scopes':{k:int(z.sum()) for k,z in masks.items()},'overlap':{a+'/'+c:int((masks[a]&masks[c]).sum()) for a,c in [('C71','C95'),('C71','C98'),('C95','C98')]},
          'outside_union_changed':int((changed&~(rule.c71|rule.scope95|rule.scope98)).sum()),
          'parent_mask_overlap':{k:int((changed&bundle[k].to_numpy(bool)).sum()) for k in ['extension','noplan','bet','practice','c73']}}
        assert counts[split]['outside_union_changed']==0
    parquet(pd.DataFrame({ID:f[ID],**steps}),O/'stack11-held-026.parquet')
    residual=f[[ID,'ap','mon','cls','fit']].copy();residual['e71']=f.y.to_numpy()-R.c71_parent(f,parent);residual['e10']=f.y.to_numpy()-parent;residual['view98']=view
    parquet(residual,D/'training-residuals.parquet');write(D/'statistics.json',{'fold':stats,'ranking':stats26,'C98_base':'unchanged stack10','ranking_training_months':[1,7]});pin(D/'statistics.json')
    manifest={'schema':1,'parent_sha256':PARENT_SHA,'spec_sha256':sha(SPEC),'splits':splits,'source_sha256':PINS.copy()};write(MANIFEST,manifest)
    result={'started_at_utc':start,'read_at_utc':now(),'source_rows':[899,901],'spec_sha256':sha(SPEC),'parent_sha256':PARENT_SHA,'parent_byte_exact':True,'old_rule_parity':proof,'C71_hold':hold['parts'],'reads':reads,'gates':gates,'counts':counts,'LIRF_shadow_increment_exact':True,'prediction_digests':{'fold':vector_digest(f[ID],steps['stack11']),'ranking':vector_digest(q[ID],out26['stack11'])},'ranking_distance':float(np.mean((out26['stack11']-parent26)**2)),'old_artifacts_unchanged':old_unchanged(),'manifest_sha256':sha(MANIFEST),'code_sha256':{n:sha(E/'code'/n) for n in CODE}}
    for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,p
    write(READ,result)
    for key in ['C71_step','C95_step','C98_step','C98_direct','joint_vs_stack10','stack11_vs_stack9_without_LIRF','route_vs_stack8_without_LIRF']:
        z=reads[key];print(key,z['G'],z['interval'],'without5',z['G_without_top_days'],flush=True)
    print('GATES',gates,flush=True)
    if not gates['pass']:return
    joint=reads['joint_vs_stack10'];pf=js(P.FORECAST);anchor=pf['central'];ratio=1.24
    forecast={'frozen_at_utc':now(),'candidate_exists':False,'parent':anchor,'parent_forecast':record(P.FORECAST),'ratio':ratio,'source_rows':[899,901],'increment_G':joint['G'],'central':float(np.sqrt(anchor**2-ratio*joint['G'])),'conditional_range':[float(np.sqrt(anchor**2-ratio*joint['interval'][k])) for k in ['hi95','lo95']],'unmasked_parent_sensitivity':float(np.sqrt(pf['unmasked_sensitivity']**2-ratio*joint['G'])),'read_sha256':sha(READ),'spec_sha256':sha(SPEC),'manifest_sha256':sha(MANIFEST),'limits':'Same-row rule/source/form and gate selection; conditional day interval excludes selection, seed and2026 transfer; C98 population and LEBL slope risk; C73 upper600 with corrected centre536-576; no repeated route/member/clip credit.'}
    write(FORECAST,forecast);print('FROZEN',forecast['central'],forecast['conditional_range'],flush=True)

def build(parent_review=None):
    assert not CANDIDATE.exists();r=js(READ);f=js(FORECAST);m=js(MANIFEST)
    review=Path(parent_review) if parent_review else PARENT_REVIEW
    lines=require_pass(review);assert PARENT_SHA in review.read_text()
    assert r['gates']['pass'] and sha(PARENT)==PARENT_SHA and sha(SPEC)==r['spec_sha256']==f['spec_sha256']==m['spec_sha256']
    assert sha(READ)==f['read_sha256'] and sha(MANIFEST)==r['manifest_sha256']==f['manifest_sha256']
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    for n,h in r['code_sha256'].items():assert sha(E/'code'/n)==h,n
    for p in [SPEC,READ,FORECAST,MANIFEST,*[E/'code'/n for n in CODE]]:P.S8.s7.committed(p)
    q,parent,kw=context('ranking');rule=load_rule('ranking',parent);out=compose11(q,rule,**kw)['stack11']
    assert vector_digest(q[ID],out)==r['prediction_digests']['ranking']
    ids=pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID];np.testing.assert_array_equal(ids,q[ID]);assert len(ids)==344841 and ids.is_unique
    result=pd.DataFrame({ID:ids.to_numpy('float64'),COL:out.astype('float64')});assert np.isfinite(result.to_numpy()).all() and result[COL].min()>=60
    old_unchanged();parquet(result,CANDIDATE)
    write(O/'stack11-build-026.json',{'candidate':record(CANDIDATE),'parent_review':record(review),'verdict_lines':lines,'spec':record(SPEC),'read':record(READ),'forecast':record(FORECAST),'manifest':record(MANIFEST),'built_at_utc':now(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'review':'desk D123 pending','upload':False})
    print('BUILT',sha(CANDIDATE),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-read','build']);ap.add_argument('--parent-review',type=Path);a=ap.parse_args()
    prepare() if a.stage=='prepare-read' else build(a.parent_review)
