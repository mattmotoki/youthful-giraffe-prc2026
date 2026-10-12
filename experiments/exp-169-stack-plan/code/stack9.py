"""Unit 024: source parity, fixed joint read, frozen forecast and one gated build."""
import argparse,ast,datetime,hashlib,io,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import stack8 as S
from assembly import ID,COL
from assembly_exploit import compose_exploit
from diverse_blend import vector_digest,load_member
from head_swap import load_head_swap,JOINT
from stack8_transition import load_stack8
from strict_verdict import require_strict_pass
from read_stacks import GS
import exploit_rules as R
ROOT=S.ROOT;EXP=S.EXP;FEATURE=ROOT/'experiments/exp-160-feature-worker'
D=EXP/'inputs/stack9-024';O=EXP/'results';SPEC=EXP/'SPEC-024.md'
PARENT=EXP/'exp169-stack8-HB3u-laneC-mix12-c73-c55.parquet'
PARENT_SHA='fec00f357276aeb07d33a19955e5c639c544f04547d5e7869ddac43d4f9c4a50'
PARENT_REVIEW=ROOT/'docs/status/2026-10-02-exp169-stack8-review/review-stack8.log'
CANDIDATE=EXP/'exp169-stack9-C71-C95-C98.parquet'
READ=O/'stack9-read-024.json';FORECAST=O/'stack9-forecast-frozen-024.json'
MANIFEST=D/'MANIFEST.json';PINS={};T0=pd.Timestamp('2025-01-01',tz='UTC')
CODE=['stack9.py','exploit_rules.py','assembly_exploit.py','strict_verdict.py']
RAWCOLS=[ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt',
         'SCHED_TIME_UTC_mvt','LOBT_flt','EOBT_1_flt','AOBT_3_flt','STAND_mvt','RUNWAY_mvt',
         'AIRCRAFT_TYPE_mvt','FLIGHT_mvt']
sha=S.sha;js=S.js;now=S.now

def pin(p):
    p=Path(p);PINS[str(p.relative_to(ROOT))]=sha(p);return p

def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    if p.exists():assert p.read_text()==text,f'Write-once record differs: {p}'
    else:p.write_text(text)

def stage(p,name=None):
    p=pin(p);q=D/(name or p.name);body=p.read_bytes()
    assert not q.exists() or q.read_bytes()==body,q
    if not q.exists():q.write_bytes(body)
    pin(q);return q

def parquet(f,p):
    p=Path(p);buf=io.BytesIO();f.to_parquet(buf,index=False);body=buf.getvalue()
    assert not p.exists() or p.read_bytes()==body,p
    if not p.exists():p.write_bytes(body)
    pin(p)

def source_function(file,name):
    path=D/file
    ns={'np':np,'pd':pd,'secs':lambda a,b:(a-b).dt.total_seconds(),'FOUR':list(R.FOUR),
        'AP':['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM'],
        'source_function':source_function,'ID':ID,'GS':GS}
    defs=[x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name]
    assert len(defs)==1
    exec(compile(ast.Module(body=defs,type_ignores=[]),str(path),'exec'),ns)
    return ns[name]

def sources():
    D.mkdir(exist_ok=True)
    for rel in ['REPORT-014.md','SPEC-014.md','ADDENDUM-014-STACK8-037.md',
                'review/EXPLOIT-014-STACK8-037-REVIEW.json','results/exploit-014-stack8-037/C95.json',
                'results/exploit-014-stack8-037/C98.json','results/exploit-014/C95-source-parity.json',
                'results/exploit-014/C98-source-parity.json','code/exploit_014.py',
                'inputs/exploit-014/frame.parquet','inputs/exploit-014/raw-fold.parquet',
                'inputs/exploit-014/source/u27_floor3.py','inputs/exploit-014/source/u13_inbound.py',
                'results/exploit-014-stack8-037/C95-steps.parquet',
                'results/exploit-014-stack8-037/C95-after71-steps.parquet',
                'results/exploit-014-stack8-037/C98-steps.parquet']:
        stage(FEATURE/rel)
    assert js(D/'EXPLOIT-014-STACK8-037-REVIEW.json')['status']=='PASS'
    for rule in ['C95','C98']:
        assert js(D/f'{rule}.json')['fixed_line_pass'] is True
        assert js(D/f'{rule}.json')['default']['G_without_top_days']>0
        assert js(D/f'{rule}-source-parity.json')['status']=='PASS'
    assert hashlib.sha256(subprocess.check_output(['git','show','45ec5f21:experiments/exp-160-feature-worker/REPORT-014.md'],cwd=ROOT)).hexdigest()==sha(D/'REPORT-014.md')
    assert sha(pin(PARENT))==PARENT_SHA
    pin(SPEC)
    for code in [*S.CODE,*CODE]:pin(EXP/'code'/code)
    pin(EXP/'results/stack8-read-023-037.json');pin(EXP/'results/stack8-forecast-frozen-023-037.json')


def parent_context(split):
    b=pd.read_parquet(pin(EXP/f'inputs/stack5-016/bundle-{split}.parquet'))
    m=load_member(S.OLD,JOINT,b,split);h=load_head_swap(S.HEAD,sha(S.HEAD),b,m,split)
    manifest=pin(EXP/'inputs/stack8-023/037/stack8.json')
    t=load_stack8(manifest,sha(manifest),b,m,h,split)
    kwargs=dict(blend=m,head_swap=t,**S.CFG)
    parent=S.compose(b,**kwargs)
    return b,parent,kwargs


def rawframe(d):
    d=d.copy();d['ap']=np.where(d.PHASE_mvt.eq('DEP'),d.ADEP_mvt,d.ADES_mvt).astype(str)
    d['mon']=d.MVT_TIME_UTC_mvt.dt.month;d['plan']=d.LOBT_flt.notna()
    p=d.FLIGHT_mvt.fillna('').astype(str);pre=p.str.extract(r'^([A-Z]{1,3})')[0];d['pre']=pre.where(pre.notna(),p.str[:2])
    return d


def stand_inputs(raw,ids):
    raw=rawframe(raw)
    floors=source_function('u27_floor3.py','floors')(raw,T0).set_index(ID).reindex(ids)
    link=source_function('u13_inbound.py','inbound')(raw).set_index(ID).reindex(ids)
    a,b,br=R.c71_parts(link)
    # Preserve ambiguous-time evidence; never let input order pick different types.
    arr=raw[raw.PHASE_mvt.eq('ARR')&raw.STAND_mvt.notna()&raw.BLOCK_TIME_UTC_mvt.notna()]
    amb=arr.groupby(['ap','STAND_mvt','BLOCK_TIME_UTC_mvt'],dropna=False).AIRCRAFT_TYPE_mvt.nunique(dropna=False)
    bad=amb[amb>1].index
    if len(bad):
        key=pd.MultiIndex.from_arrays([link.ap,link.STAND_mvt,link.a_t]);assert not ((a|b)&key.isin(bad)).any(),'Ambiguous C71 link.'
    return floors.g.to_numpy(),link.g.to_numpy(),a,b,br


def audit_c71():
    out=O/'stack9-c71-hold-024.json'
    if out.exists():
        audit=js(out)
        for p,h in audit['source_sha256'].items():assert sha(ROOT/p)==h,p
        assert audit['passed'];PINS.update(audit['source_sha256']);pin(out);return audit
    parts=[];rawpins={}
    for month in range(1,13):
        p=next((ROOT/'data/raw').glob(f'training_2025-{month:02d}-*.parquet'));rawpins[str(p.relative_to(ROOT))]=sha(p)
        d=pd.read_parquet(p,columns=RAWCOLS+['TAXITIME_SEC_mvt'])
        ap=np.where(d.PHASE_mvt.eq('DEP'),d.ADEP_mvt,d.ADES_mvt)
        parts.append(d[np.isin(ap,R.FOUR)])
    raw=rawframe(pd.concat(parts,ignore_index=True));assert raw[ID].is_unique
    dep=raw[raw.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    _,g,a,b,br=stand_inputs(raw,dep[ID]);y=dep.TAXITIME_SEC_mvt.to_numpy(float)
    audit={'definition':'Fresh/features links; part b excludes part a for the part audit; union unchanged; y >= g - 6.',
           'source_sha256':rawpins,'parts':{},'passed':True}
    for name,m,threshold in [('a',a,.998),('b_outside_a',b,.95),('b_including_a',br,.95)]:
        z=y[m]-g[m];rate=float((z>=-6).mean());audit['parts'][name]={'rows':int(m.sum()),'strict_rate':float((z>=0).mean()),
          'six_second_rate':rate,'fails':int((z < -6).sum()),'required':threshold}
        audit['passed'] &= rate>=threshold
    write(out,audit);assert audit['passed'],audit
    PINS.update(rawpins);pin(out)
    return audit


def registered(f,base,cand):
    return source_function('exploit_014.py','registered')(f,base,cand)[0]


def parity(f,parent):
    out={};ref95=source_function('exploit_014.py','c95_steps');ref98=source_function('exploit_014.py','c98_step')
    p71=R.c71_parent(f,parent)
    for name,base in [('C95',parent),('C95-after71',p71)]:
        s,mask,_=R.c95(f,base,f,base)
        np.testing.assert_array_equal(s,ref95(f,base,classed=True,soft=True))
        saved=pd.read_parquet(D/f'{name}-steps.parquet');np.testing.assert_array_equal(saved[ID],f[ID])
        if name=='C95-after71':np.testing.assert_array_equal(saved.parent71,p71)
        else:np.testing.assert_array_equal(saved.parent,parent)
        np.testing.assert_array_equal(np.maximum(60,base+s)-base,saved.step)
        read=registered(f,base,np.maximum(60,base+s))
        want=js(D/'C95.json')['default' if name=='C95' else 'after_C71']
        for k in ['G','G_without_top_days']:assert abs(read[k]-want[k])<1e-8
        out[name]={'saved_step_exact':True,'G':read['G'],'without5':read['G_without_top_days']}
    s,mask,coefs,v,n=R.c98(f,parent,f,parent);rs,rv,rn,rc=ref98(f,parent)
    np.testing.assert_array_equal(s,rs);np.testing.assert_array_equal(v,rv);np.testing.assert_array_equal(n,rn)
    saved=pd.read_parquet(D/'C98-steps.parquet');np.testing.assert_array_equal(saved[ID],f[ID]);np.testing.assert_array_equal(saved.parent,parent)
    np.testing.assert_array_equal(np.maximum(60,parent+s)-parent,saved.step)
    read=registered(f,parent,np.maximum(60,parent+s));want=js(D/'C98.json')['default']
    for k in ['G','G_without_top_days']:assert abs(read[k]-want[k])<1e-8
    out['C98']={'saved_step_exact':True,'G':read['G'],'without5':read['G_without_top_days']}
    # Source source-base controls retain their original source bootstrap.
    boot=source_function('exploit_014.py','source_boot');keep=f.ap.isin(R.FOUR).to_numpy();fit=f.fit.to_numpy();base=f.stack5.to_numpy()
    src=base+np.where(fit,ref95(f,base),0.)
    source95=boot(f[keep],((f.y.to_numpy()-base)**2-(f.y.to_numpy()-src)**2)[keep])
    ss,_,_,_=ref98(f,base,source_support=True);src98=base+np.where(fit,ss,0.)
    source98=boot(f,(f.y.to_numpy()-base)**2-(f.y.to_numpy()-src98)**2)
    b4=f.stack4.to_numpy();b471=R.c71_parent(f,b4);order=f.sort_values(['ap','day','t',ID],kind='stable').index.to_numpy()
    source71=boot(f.loc[order],((f.y.to_numpy()-b4)**2-(f.y.to_numpy()-b471)**2)[order])
    for name,got,want in [('C95_source',source95,{'G':151.,'lo95':71.2,'hi95':257.,'without5':54.}),
                          ('C98_source',source98,{'G':62.5,'lo95':33.,'hi95':95.3,'without5':34.1}),
                          ('C71_source',source71,{'G':27.3,'lo95':5.7,'hi95':57.3})]:
        assert all(abs(got[k]-v)<=.11 for k,v in want.items()),(name,got,want);out[name]=got
    write(O/'stack9-source-parity-024.json',out)
    return out


def ranking_frame(b):
    raw=pd.read_parquet(pin(ROOT/'data/raw/ranking.parquet'),columns=RAWCOLS)
    raw=rawframe(raw);dep=raw[raw.PHASE_mvt.eq('DEP')].set_index(ID).loc[b[ID]].reset_index()
    f=dep.copy();f['cls']=b.cls.to_numpy();f['F']=b.F.to_numpy();f['t']=(f.MVT_TIME_UTC_mvt-T0).dt.total_seconds()
    f['rw']=f.RUNWAY_mvt.astype(str);f['scope']=f.plan & f.ADES_mvt.ne(f.ap)
    g,g71,a,c,_=stand_inputs(raw,f[ID]);f['g']=g;f['g71']=g71;f['c71']=a|c
    assert not (f.c71.to_numpy()&b.stand_floor.notna().to_numpy()).any()
    return f


def table(obj,view,n):
    return pd.DataFrame({ID:obj.ids,'c71':obj.c71,'g71':obj.g71,'step95':obj.step95,'scope95':obj.scope95,
                         'step98':obj.step98,'scope98':obj.scope98,'lower':obj.lower,'upper':obj.upper,'view98':view,'neighbour_n':n})


def load_rule(split,parent):
    manifest=js(MANIFEST);record=manifest['splits'][split];path=ROOT/record['path'];assert sha(path)==record['sha256']
    q=pd.read_parquet(path)
    return R.Exploit(q[ID].to_numpy(),record['parent_digest'],q.c71.to_numpy(),q.g71.to_numpy(),q.step95.to_numpy(),
                     q.scope95.to_numpy(),q.step98.to_numpy(),q.scope98.to_numpy(),q.lower.to_numpy(),q.upper.to_numpy())


def check_old():
    old=js(O/'stack9-before-024.json')['files'];assert all(sha(ROOT/p)==h for p,h in old.items())
    return len(old)


def prepare():
    assert not READ.exists() and not FORECAST.exists() and not CANDIDATE.exists()
    start=now();sources();hold=audit_c71()
    b,parent,kw=parent_context('fold');q,parent26,kw26=parent_context('ranking')
    saved=pd.read_parquet(pin(PARENT));np.testing.assert_array_equal(saved[ID],q[ID]);np.testing.assert_array_equal(saved[COL],parent26)
    h=pd.read_parquet(pin(O/'stack8-held-023-037.parquet'));np.testing.assert_array_equal(h[ID],b[ID]);np.testing.assert_array_equal(h.stack8,parent)
    assert vector_digest(b[ID],parent)==js(O/'stack8-read-023-037.json')['prediction_digests']['fold']
    f=pd.read_parquet(D/'frame.parquet');np.testing.assert_array_equal(f[ID],b[ID]);np.testing.assert_array_equal(f.y,GS.labels(b[ID].to_numpy('int64')))
    raw=pd.read_parquet(D/'raw-fold.parquet');g,g71,a,c,_=stand_inputs(raw,f[ID])
    np.testing.assert_array_equal(g,f.g);np.testing.assert_array_equal(g71,f.g71);np.testing.assert_array_equal(a|c,f.c71)
    assert not (f.c71.to_numpy()&b.stand_floor.notna().to_numpy()).any()
    proof=parity(f,parent)
    rule,stats,view,n=R.prepare_rule(f,parent,f,parent,b)
    steps=compose_exploit(b,rule,**kw);np.testing.assert_array_equal(steps['stack8'],parent)
    np.testing.assert_array_equal(compose_exploit(b,rule,enabled=False,**kw)['stack9'],parent)
    # Independent final arithmetic, separate from the wrapper/apply method.
    independent=np.clip(np.maximum(60,np.maximum(60,R.c71_parent(f,parent)+rule.step95)+rule.step98),rule.lower,rule.upper)
    np.testing.assert_array_equal(independent,steps['stack9'])
    reads={};previous=parent
    for key in ['C71','C95','C98','stack9']:
        reads[key+'_step']=registered(f,previous,steps[key]);previous=steps[key]
    reads['joint_vs_stack8']=registered(f,parent,steps['stack9'])
    assert reads['C95_step']['G_without_top_days']>=20 and reads['C95_step']['interval']['lo95']>0
    joint=reads['joint_vs_stack8'];assert joint['G_without_top_days']>0
    assert 60<=joint['G']<=170,'Joint result outside SPEC prior: diagnostic and desk report required.'
    # Every new rule is outside LIRF; this increment is identical for the shadow parent.
    lirf=f.ap.eq('LIRF').to_numpy();np.testing.assert_array_equal(steps['stack9'][lirf],parent[lirf])
    rank=ranking_frame(q);rule26,stats26,view26,n26=R.prepare_rule(f,parent,rank,parent26,q,pooled=True)
    out26=compose_exploit(q,rule26,**kw26);np.testing.assert_array_equal(compose_exploit(q,rule26,enabled=False,**kw26)['stack9'],parent26)
    ind26=np.clip(np.maximum(60,np.maximum(60,R.c71_parent(rank,parent26)+rule26.step95)+rule26.step98),rule26.lower,rule26.upper)
    np.testing.assert_array_equal(ind26,out26['stack9'])
    # Keep no new official ranking file before the reviewed-parent/commit gate.
    splits={};counts={}
    for split,base,obj,v,nn in [('fold',parent,rule,view,n),('ranking',parent26,rule26,view26,n26)]:
        path=D/f'rules-{split}.parquet';parquet(table(obj,v,nn),path)
        splits[split]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'parent_digest':obj.parent_digest}
        masks={'C71':obj.c71,'C95':obj.scope95,'C98':obj.scope98}
        counts[split]={'scopes':{k:int(x.sum()) for k,x in masks.items()},'overlap':{a+'/'+b:int((masks[a]&masks[b]).sum()) for a,b in [('C71','C95'),('C71','C98'),('C95','C98')]},
                       'changed':int(np.count_nonzero((steps if split=='fold' else out26)['stack9']!=base))}
    parquet(pd.DataFrame({ID:f[ID],**steps}),O/'stack9-held-024.parquet')
    # Held residual training samples make pooled and cross-month statistics inspectable.
    residuals=f[[ID,'ap','mon','cls','fit']].copy();residuals['e71']=f.y.to_numpy()-R.c71_parent(f,parent);residuals['e8']=f.y.to_numpy()-parent;residuals['view98']=view
    parquet(residuals,D/'training-residuals.parquet')
    write(D/'statistics.json',{'fold':stats,'ranking':stats26,'ranking_training_months':[1,7],'C98_base':'unchanged stack8'})
    manifest={'schema':1,'parent_sha256':PARENT_SHA,'spec_sha256':sha(SPEC),'splits':splits,'source_sha256':PINS.copy()}
    write(MANIFEST,manifest)
    result={'started_at_utc':start,'read_at_utc':now(),'source_rows':[899,901],'spec_sha256':sha(SPEC),'parent_sha256':PARENT_SHA,
      'source_parity':proof,'C71_hold':hold['parts'],'reads':reads,'counts':counts,'LIRF_shadow_increment_exact':True,
      'prediction_digests':{'fold':vector_digest(f[ID],steps['stack9']),'ranking':vector_digest(q[ID],out26['stack9'])},
      'ranking_distance':float(np.mean((out26['stack9']-parent26)**2)),
      'additivity_difference_vs_source_direct_marginals':float(joint['G']-js(D/'C95.json')['default']['G']-js(D/'C98.json')['default']['G']-reads['C71_step']['G']),
      'old_artifacts_unchanged':check_old(),'manifest_sha256':sha(MANIFEST),'code_sha256':{n:sha(EXP/'code'/n) for n in CODE}}
    write(READ,result)
    pf=js(O/'stack8-forecast-frozen-023-037.json');anchor=pf['central'];ratio=1.24
    forecast={'frozen_at_utc':now(),'candidate_exists':False,'parent':anchor,'ratio':ratio,'source_rows':[899,901],
      'increment_G':joint['G'],'central':float(np.sqrt(anchor**2-ratio*joint['G'])),
      'conditional_range':[float(np.sqrt(anchor**2-ratio*joint['interval'][k])) for k in ['hi95','lo95']],
      'unmasked_parent_sensitivity':float(np.sqrt(pf['unmasked_sensitivity']**2-ratio*joint['G'])),
      'read_sha256':sha(READ),'spec_sha256':sha(SPEC),'manifest_sha256':sha(MANIFEST),
      'limits':'Same-row rule selection; day interval excludes selection, seed and 2026 population uncertainty; C98 ordinary-population transfer risk; C73 upper600 remains, corrected centre536–576; no repeated old-rule credit.'}
    write(FORECAST,forecast)
    print(json.dumps({'G':joint['G'],'without5':joint['G_without_top_days'],'forecast':forecast['central'],'counts':counts},indent=2),flush=True)


def build():
    assert not CANDIDATE.exists();r=js(READ);f=js(FORECAST);m=js(MANIFEST)
    lines=require_strict_pass(PARENT_REVIEW);assert PARENT_SHA in PARENT_REVIEW.read_text()
    assert sha(PARENT)==PARENT_SHA and sha(SPEC)==r['spec_sha256']==f['spec_sha256']==m['spec_sha256']
    assert sha(READ)==f['read_sha256'] and sha(MANIFEST)==r['manifest_sha256']==f['manifest_sha256']
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    for n,h in r['code_sha256'].items():assert sha(EXP/'code'/n)==h,n
    for p in [SPEC,READ,FORECAST,MANIFEST,*[EXP/'code'/n for n in CODE]]:S.s7.committed(p)
    q,parent,kw=parent_context('ranking');rule=load_rule('ranking',parent)
    out=compose_exploit(q,rule,**kw)['stack9'];assert vector_digest(q[ID],out)==r['prediction_digests']['ranking']
    ids=pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID];np.testing.assert_array_equal(ids,q[ID]);assert len(ids)==344841 and ids.is_unique
    result=pd.DataFrame({ID:ids.to_numpy('float64'),COL:out.astype('float64')});parquet(result,CANDIDATE)
    assert np.isfinite(result.to_numpy()).all() and (result[COL]>=60).all();check_old()
    reviewcopy=stage(PARENT_REVIEW,'PARENT-D123-PASS.log')
    write(O/'stack9-build-024.json',{'candidate':S.record(CANDIDATE),'parent_review':S.record(reviewcopy),'verdict_lines':lines,
      'spec':S.record(SPEC),'read':S.record(READ),'forecast':S.record(FORECAST),'manifest':S.record(MANIFEST),
      'built_at_utc':now(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'review':'desk D123 pending','upload':False})
    print('BUILT',sha(CANDIDATE),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-read','build']);a=ap.parse_args()
    {'prepare-read':prepare,'build':build}[a.stage]()
