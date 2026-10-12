"""Unit028: one039 replacement in stack11, fixed residual recipes, two forecasts."""
import argparse,hashlib,io,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import stack11 as S
from lane_c_swap import make_transition,matrix_apply,NAMES
from verdict026 import require_pass
P=S.P;Q=S.Q;R=S.R;S8=P.S8;E=S.E;ROOT=S.ROOT;O=S.O;D=E/'inputs/stack12-028';DIV=S8.DIV
ID=S.ID;COL=S.COL;sha=S.sha;js=S.js;write=S.write;record=S.record;now=S.now;digest=S.vector_digest
SPEC=E/'SPEC-028.md';SPEC_SHA='9b969c26d22eb3161182c1759c37e447c0dfec7f3fe453ce482bdf51d9b8cfd7'
READ=O/'stack12-read-028.json';FORECAST=O/'stack12-forecast-frozen-028.json';RATIO=O/'stack12-forecast-ratio083-028.json';MANIFEST=D/'MANIFEST.json'
CANDIDATE=E/'exp169-stack12-R4mks-C039-C71-C95-C98.parquet'
PARENT=S.CANDIDATE;PARENT_SHA='3c7e0c5ffefe97c5b66df7dc9222c2d606225f2fe3e30c46a854d5d68eda4e20'
REVIEW=ROOT/'docs/status/2026-10-03-exp169-stack11-review/review-stack11.log'
PACK_SHA='51cb861243917ee2a93d4cdb1b99c69c7151f7694ab3c0394a3abdd460cb0a01'
VECTOR_SHA='2b608e3ceb602908b5b47aff29cfec27fd0cc9f5f30affac9296f0a13bc0400e'
CODE=list(dict.fromkeys([*S.CODE,'build_stack11_026.py','stack12.py','check_stack12_028.py']))
PINS={}

def pin(p):
    p=Path(p).resolve();k=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p);h=sha(p)
    if k in PINS:assert PINS[k]==h,k
    PINS[k]=h;return p

def parquet(f,p):
    buf=io.BytesIO();f.to_parquet(buf,index=False);body=buf.getvalue()
    if p.exists():assert p.read_bytes()==body,p
    else:p.write_bytes(body)
    pin(p)

def stage(src,name,expected=None):
    src=pin(src)
    if expected:assert sha(src)==expected,src
    out=D/name
    if out.exists():assert out.read_bytes()==src.read_bytes(),out
    else:out.write_bytes(src.read_bytes())
    pin(out);return out

def unchanged():
    files=js(O/'stack12-before-028.json')['files']
    for p,h in files.items():assert sha(ROOT/p)==h,p
    return len(files)

def historical(p,h):
    p=Path(p)
    if sha(p)!=h:
        rec=js(O/'verdict-checks-027.json');name=p.name
        assert p.parent==E/'code' and name in rec['code_before'],p
        assert h==rec['code_before'][name] and sha(p)==rec['code_after'][name],p
        old=pin(E/'inputs/verdict-027/before'/name);assert sha(old)==h
    pin(p)

def sources():
    assert sha(SPEC)==SPEC_SHA and sha(PARENT)==PARENT_SHA
    require_pass(REVIEW);assert PARENT_SHA in REVIEW.read_text()
    for p in [SPEC,PARENT,REVIEW,S.READ,S.FORECAST,S.MANIFEST,P.READ,P.MANIFEST,P.FORECAST,
              O/'stack11-forecast-ratio083-026.json',O/'stack11-held-026.parquet',O/'stack10-held-025.parquet',
              O/'verdict-checks-027.json',O/'stack12-before-028.json',S.D/'statistics.json',Q.D/'frame.parquet',
              S.D/'frame-ranking.parquet',Q.D/'raw-fold.parquet',ROOT/'data/raw/submitting.parquet']:
        pin(p)
    for rec in [js(S.MANIFEST)['source_sha256'],js(P.READ)['source_sha256']]:
        for path,h in rec.items():historical(ROOT/path,h)
    for rec in [js(S.READ)['code_sha256'],js(P.READ)['code_sha256']]:
        for name,h in rec.items():historical(E/'code'/name,h)
    for n in CODE:pin(E/'code'/n)
    for split in ['fold','ranking']:pin(ROOT/js(S.MANIFEST)['splits'][split]['path'])
    pack=js(stage(DIV/'results/pack-close30r6-pair-039/PACK.json','PACK.json',PACK_SHA))
    review=stage(DIV/'REVIEW-PACK-039.md','SOURCE-PASS.md');require_pass(review)
    assert PACK_SHA in review.read_text() and VECTOR_SHA in review.read_text()
    stage(DIV/'results/pack-close30r6-pair-039/members-final.parquet','pack-ranking.parquet',VECTOR_SHA)
    for split,typ,filename in [('fold','held','jjrefit.parquet'),('ranking','annual','ranking2026.parquet')]:
        rec=pack['records']['sequence'];folder=DIV/'results'/(Path(rec[typ+'_fit_path']).name+'-fit')
        stage(folder/filename,f'sequence-{split}.parquet',rec['annual_raw_vector_sha256' if split=='ranking' else 'held_vector_sha256'])
        stage(folder/'FIT.json',f'sequence-{split}-FIT.json',rec[typ+'_fit_sha256'])
    oldpack=js(pin(E/'inputs/stack8-023/037/PACK.json'))
    for key in ['scope_pack_sha256','F_2026_sha256','template_sha256','control_bag_seeds','control_bag_weights']:
        assert pack[key]==oldpack[key],key
    assert pack['records']['control']==oldpack['records']['control']
    stage(DIV/'results/paired-close30-038-B/PAIRED.json','SOURCE-PAIR.json',pack['read_sha256'])
    stage(DIV/'results/paired-close30-038-B/paired_ls.parquet','source-pair-held.parquet','9bffcffc418fd8ff4cee94e48d6af089df240757524a04b706f065a4f92d319e')
    stage(DIV/'results/paired-mixed30-037-B/paired_ls.parquet','parent-source-pair-held.parquet','54b270d8f93be46ddeb9c4a566af9fa00277f8debb5144a97482b1c062d4db28')
    stage(DIV/'results/STEP-close30-over-mixed30-038.json','SOURCE-STEP.json')
    stage(DIV/'results/masked-close30-LIRF-039-fit/jjrefit.parquet','masked-new.parquet','d3eca63ea70db0fd32d83d5fb5273b8fe3e147f2674d4d27c5094860608a408f')
    pr=js(stage(DIV/'results/masked-close30-LIRF-039-fit/PREDICT.json','masked-new-PREDICT.json'))
    assert pr['label_free'] and pr['unaffected_predictions_exact'] and pr['fit_sha256']==pack['records']['sequence']['held_fit_sha256']
    assert pr['files']['jjrefit.parquet']==sha(D/'masked-new.parquet')
    checker=pin(DIV/'code/check_pack_039.py')
    run=subprocess.run([str(ROOT/'.venv/bin/python'),str(checker),'close30'],cwd=ROOT,text=True,capture_output=True,check=True)
    (D/'source-check.log').write_text(run.stdout);pin(D/'source-check.log')
    assert run.stdout.startswith('PASS close30 ') and PACK_SHA in run.stdout and VECTOR_SHA in run.stdout
    hold=js(pin(O/'stack9-c71-hold-024.json'));assert hold['passed']
    return pack,hold

def new_frame(split,b,m):
    old=pd.read_parquet(pin(E/f'inputs/stack8-023/037/matrix-{split}.parquet'));f=old.copy()
    f['sequence_new']=S8.aligned(D/f'sequence-{split}.parquet',b[ID])
    assert f[ID].is_unique
    for key in [ID,'month','cls']:np.testing.assert_array_equal(f[key],b[key])
    np.testing.assert_array_equal(f.member_scope,m.scope)
    for key in old.columns:
        if key!='sequence_new':np.testing.assert_array_equal(f[key],old[key])
    assert np.isfinite(f.loc[m.scope,'sequence_new']).all()
    if split=='ranking':
        annual=pd.read_parquet(D/'pack-ranking.parquet')
        for key,want in [(ID,b[ID]),('member_scope',m.scope),('control_scope',m.scope&m.ordinary),('pred_row170',f.control),('pred_seq170',f.sequence_new)]:np.testing.assert_array_equal(annual[key],want)
    return old,f

def composed(b,m,t,rule=None):
    kw=dict(blend=m,head_swap=t,**S8.CFG);parent=P.compose(b,**kw)
    if rule is None:return parent
    z=S.compose11(b,rule,**kw)
    return {('pre_exploit' if k=='stack10' else 'stack12' if k=='stack11' else k):v for k,v in z.items()}

def independent_rule(f,parent,obj,steps):
    out=np.clip(np.maximum(60,np.maximum(60,R.c71_parent(f,parent)+obj.step95)+obj.step98),obj.lower,obj.upper)
    np.testing.assert_array_equal(out,steps['stack12']);np.testing.assert_array_equal(out,np.clip(out,obj.lower,obj.upper))
    np.testing.assert_array_equal(out[~(obj.c71|obj.scope95|obj.scope98)],parent[~(obj.c71|obj.scope95|obj.scope98)])

def parent_parity(contexts,f,rank):
    out={};parents={};finals={};stats=js(S.D/'statistics.json')
    for split,frame in [('fold',f),('ranking',rank)]:
        ob,b,m,a,t,_=contexts[split];parent=composed(b,m,t);parents[split]=parent
        obj,st,v,n=R.prepare_rule(f,parents['fold'],frame,parent,b,pooled=split=='ranking')
        pd.testing.assert_frame_equal(Q.table(obj,v,n),pd.read_parquet(ROOT/js(S.MANIFEST)['splits'][split]['path']),check_exact=True)
        assert st==stats[split]
        old=S.load_rule(split,parent);finals[split]=composed(b,m,t,old)['stack12']
        assert digest(b[ID],finals[split])==js(S.READ)['prediction_digests'][split]
        if split=='ranking':
            candidate=pd.DataFrame({ID:b[ID].to_numpy('float64'),COL:finals[split].astype('float64')});buf=io.BytesIO();candidate.to_parquet(buf,index=False)
            assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT_SHA
        else:
            held=pd.read_parquet(O/'stack11-held-026.parquet');np.testing.assert_array_equal(held[ID],b[ID]);np.testing.assert_array_equal(held.stack11,finals[split]);np.testing.assert_array_equal(held.stack10,parent)
        out[split]={'rule_vector_exact':True,'residual_statistics_exact':True,'prediction_exact':True}
    ob,b,m,a,t,_=contexts['fold'];oldframe,_=new_frame('fold',b,m)
    fw,pw=S8.s7.fit(b,oldframe,a.new_base,P.base_tail(b,a),f.y.to_numpy(),P.weights(b));manifest=js(P.MANIFEST)
    assert fw==manifest['fold_weights'] and pw==manifest['ranking_weights']
    out['normal_equations_exact']=True;out['ranking_byte_exact']=True
    return parents,finals,out

def shadow(b,m,a,t,final,new=False):
    source=b.airport.eq('LIRF').to_numpy()&b.month.eq(7).to_numpy();use=source&m.ordinary;v=t.values.copy()
    paths=[E/'inputs/stack8-023/037/masked-old021.parquet',D/'masked-new.parquet' if new else E/'inputs/stack8-023/037/masked-new.parquet']
    for col,path in zip([2,3],paths):
        vals=S8.aligned(pin(path),b[ID]);np.testing.assert_array_equal(vals[~source],v[~source,col]);v[use,col]=vals[use]
    prior=a.blend(m,b[ID].to_numpy(),a.new_base,True)
    masked=S8.s7.independent_tail(b,matrix_apply(a.new_base,prior,v,t.weights,m.scope,m.ordinary))
    if not new:np.testing.assert_array_equal(masked,pd.read_parquet(O/'stack10-held-025.parquet').masked_stack10)
    result=final.copy();lirf=b.airport.eq('LIRF').to_numpy();result[lirf]=masked[lirf];return result

def scope(b,m,a,t,oldpre,newpre,old,new,obj):
    primary=S8.s7.scope_check(b,m,a,t,oldpre,newpre)
    np.testing.assert_array_equal(new[b.noplan],old[b.noplan]);np.testing.assert_array_equal(new[b.bet],old[b.bet])
    union=m.scope|obj.c71|obj.scope95|obj.scope98;np.testing.assert_array_equal(new[~union],old[~union])
    return {'pre_exploit':primary,'final_changed':int((new!=old).sum()),'outside_primary_final_changed':int(((new!=old)&~m.scope).sum()),
            'outside_allowed_union_changed':0,'overlap':{k:{'rows':int(b[k].sum()),'changed':int(((new!=old)&b[k].to_numpy(bool)).sum())} for k in ['extension','noplan','bet','practice','c73']}}

def rule_gates(reads,hold):
    r95=reads['C95_step'];r98=reads['C98_direct'];positive=[k for k,v in r98['views']['airport'].items() if v>0]
    return {'C71_hold':hold['passed'],'C95_without5_ge20':r95['G_without_top_days']>=20,'C95_lower95_gt0':r95['interval']['lo95']>0,
            'C98_direct_G_ge30':r98['G']>=30,'C98_direct_without5_gt0':r98['G_without_top_days']>0,
            'C98_not_EDDF_only_weak':not(positive==['EDDF'] and r98['without5_interval']['lo95']<=0),
            'C98_ordered_without5_gt0':reads['C98_step']['G_without_top_days']>0,'joint_rules_without5_gt0':reads['rules_joint']['G_without_top_days']>0,
            'replacement_G_gt0':reads['full_vs_stack11']['G']>0}

def load(split):
    rec=js(MANIFEST);part=rec['splits'][split];ob,b,m,a,old,_=P.load_route(split)
    f=pd.read_parquet(ROOT/part['matrix']['path']);assert sha(ROOT/part['matrix']['path'])==part['matrix']['sha256']
    expected_old,expected_new=new_frame(split,b,m);pd.testing.assert_frame_equal(f,expected_new,check_exact=True)
    t=make_transition(a,m,f,rec['fold_weights'],rec['ranking_weights'],split)
    assert t.matrix_digest==part['matrix_digest'] and digest(b[ID],a.new_base)==part['base_digest']
    parent=composed(b,m,t);rp=ROOT/part['rules']['path'];assert sha(rp)==part['rules']['sha256'];q=pd.read_parquet(rp)
    assert digest(q[ID],parent)==part['parent_digest']
    obj=R.Exploit(q[ID].to_numpy(),part['parent_digest'],q.c71.to_numpy(),q.g71.to_numpy(),q.step95.to_numpy(),q.scope95.to_numpy(),q.step98.to_numpy(),q.scope98.to_numpy(),q.lower.to_numpy(),q.upper.to_numpy())
    return b,m,a,t,parent,obj

def prepare():
    assert not READ.exists() and not FORECAST.exists() and not RATIO.exists() and not CANDIDATE.exists();start=now();D.mkdir(exist_ok=True)
    pack,hold=sources();contexts={s:P.load_route(s) for s in ['fold','ranking']}
    f=pd.read_parquet(Q.D/'frame.parquet');rank=pd.read_parquet(S.D/'frame-ranking.parquet')
    np.testing.assert_array_equal(f[ID],contexts['fold'][1][ID]);np.testing.assert_array_equal(f.y,Q.GS.labels(f[ID].to_numpy('int64')))
    raw=pd.read_parquet(Q.D/'raw-fold.parquet');g,g71,aa,bb,_=Q.stand_inputs(raw,f[ID])
    for x,y in [(g,f.g),(g71,f.g71),(aa|bb,f.c71)]:np.testing.assert_array_equal(x,y)
    parents,finals,proof=parent_parity(contexts,f,rank);print('PARENT PARITY',proof,flush=True)
    frames={s:new_frame(s,c[1],c[2])[1] for s,c in contexts.items()};ob,b,m,a,old,_=contexts['fold'];y=f.y.to_numpy()
    parity=S8.s7.source_parity(ob,m,frames['fold'],pack,D,y)
    source_old=S8.aligned(D/'parent-source-pair-held.parquet',f[ID]);source_new=S8.aligned(D/'source-pair-held.parquet',f[ID])
    pair_step=P.score(f[ID].to_numpy('int64'),source_old,source_new,y);want=js(D/'SOURCE-STEP.json')['step']
    assert abs(pair_step['G']-want['G'])<1e-7
    for k in ['lo95','hi95']:assert abs(pair_step['interval'][k]-want['interval'][k])<1e-7
    print('SOURCE PARITY',parity,'pair step',pair_step['G'],flush=True)
    fw,pw=S8.s7.fit(b,frames['fold'],a.new_base,P.base_tail(b,a),y,P.weights(b))
    transitions={s:make_transition(c[3],c[2],frames[s],fw,pw,s) for s,c in contexts.items()}
    pre={s:composed(c[1],c[2],transitions[s]) for s,c in contexts.items()};steps={};rules={};stats={};splits={};counts={};risk={}
    for split,frame in [('fold',f),('ranking',rank)]:
        ob,b,m,a,old,_=contexts[split];t=transitions[split]
        obj,st,v,n=R.prepare_rule(f,pre['fold'],frame,pre[split],b,pooled=split=='ranking');rules[split]=obj;stats[split]=st
        steps[split]=composed(b,m,t,obj);independent_rule(frame,pre[split],obj,steps[split])
        # C71 identity and inherited bounds stay exact, even as residual statistics change.
        oldobj=S.load_rule(split,parents[split])
        for key in ['ids','c71','g71','lower','upper']:np.testing.assert_array_equal(getattr(obj,key),getattr(oldobj,key))
        counts[split]=scope(b,m,a,t,parents[split],pre[split],finals[split],steps[split]['stack12'],obj)
        mp=D/f'matrix-{split}.parquet';rp=D/f'rules-{split}.parquet';parquet(frames[split],mp);parquet(Q.table(obj,v,n),rp)
        splits[split]={'matrix':record(mp),'rules':record(rp),'parent_digest':obj.parent_digest,'base_digest':digest(b[ID],a.new_base),'matrix_digest':t.matrix_digest}
        risk[split]=S8.sequence_risk(b,m,t)
    reads={};previous=pre['fold']
    for key in ['C71','C95','C98','stack12']:
        reads[key+'_step']=S.score(f,previous,steps['fold'][key]);previous=steps['fold'][key]
    reads['C98_direct']=S.score(f,pre['fold'],np.maximum(60,pre['fold']+rules['fold'].step98))
    reads['rules_joint']=S.score(f,pre['fold'],steps['fold']['stack12'])
    reads['member_only']=S.score(f,parents['fold'],pre['fold'])
    reads['full_vs_stack11']=S.score(f,finals['fold'],steps['fold']['stack12'])
    reads['full_vs_stack11_without_LIRF']=S.no_lirf(f,finals['fold'],steps['fold']['stack12'])
    ob,b,m,a,old,_=contexts['fold'];oldshadow=shadow(b,m,a,old,finals['fold']);newshadow=shadow(b,m,a,transitions['fold'],steps['fold']['stack12'],True)
    reads['matched_shadow_vs_stack11']=S.score(f,oldshadow,newshadow)
    np.testing.assert_array_equal(steps['fold']['stack12'][f.ap.eq('LIRF')],pre['fold'][f.ap.eq('LIRF')])
    gates=rule_gates(reads,hold);gates['pass']=all(gates.values())
    parquet(pd.DataFrame({ID:f[ID],'stack11_pre':parents['fold'],'stack11':finals['fold'],**steps['fold'],'masked_stack11':oldshadow,'masked_stack12':newshadow}),O/'stack12-held-028.parquet')
    residual=f[[ID,'ap','mon','cls','fit']].copy();residual['e71']=f.y.to_numpy()-R.c71_parent(f,pre['fold']);residual['e_direct']=f.y.to_numpy()-pre['fold'];parquet(residual,D/'training-residuals.parquet')
    write(D/'statistics.json',stats);pin(D/'statistics.json')
    manifest={'schema':1,'change':'sequence037_to039_only_with_same_residual_recipes','weight_rule':'other_month_then_pooled','spec_sha256':SPEC_SHA,'parent_sha256':PARENT_SHA,'pack_sha256':PACK_SHA,'vector_sha256':VECTOR_SHA,'fold_weights':fw,'ranking_weights':pw,'splits':splits,'source_sha256':PINS.copy()};write(MANIFEST,manifest)
    for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,p
    result={'started_at_utc':start,'read_at_utc':now(),'spec_sha256':SPEC_SHA,'parent_sha256':PARENT_SHA,'source_rows':[899,901,902],'ledger_end':902,'parent_parity':proof,'source_parity':parity,'source_pair_step':pair_step,'reads':reads,'gates':gates,'scope':counts,'sequence_difference_risk':risk,'prediction_digests':{s:digest(contexts[s][1][ID],steps[s]['stack12']) for s in steps},'ranking_distance':float(np.mean((steps['ranking']['stack12']-finals['ranking'])**2)),'old_artifacts_unchanged':unchanged(),'manifest_sha256':sha(MANIFEST),'code_sha256':{n:sha(E/'code'/n) for n in CODE}}
    write(READ,result)
    for key in ['member_only','rules_joint','full_vs_stack11','full_vs_stack11_without_LIRF','matched_shadow_vs_stack11']:
        z=reads[key];print(key,z['G'],z['interval'],'without5',z['G_without_top_days'],flush=True)
    print('GATES',gates,flush=True)
    if not gates['pass']:return
    limits='Conditional day interval after reused-fold source and form selection; two-fold parent-label dependence; no seed/search or2026 transfer coverage; opposing sequence coefficients; C98 population/LEBL risk; C73 upper600, corrected centre536-576; no repeated route/member/rule credit.'
    r=reads['matched_shadow_vs_stack11'];anchor=js(S.FORECAST)['central'];assert anchor==221.78546683365872
    frozen={'central':float(np.sqrt(anchor**2-1.24*r['G'])),'conditional_range':[float(np.sqrt(anchor**2-1.24*r['interval'][k])) for k in ['hi95','lo95']],'parent':anchor,'parent_forecast':record(S.FORECAST),'ratio':1.24,'source_rows':[899,901],'increment_G':r['G'],'increment_interval':r['interval'],'unmasked_increment_G':reads['full_vs_stack11']['G'],'unmasked_step_sensitivity':float(np.sqrt(anchor**2-1.24*reads['full_vs_stack11']['G'])),'frozen_at_utc':now(),'candidate_exists':False,'read_sha256':sha(READ),'manifest_sha256':sha(MANIFEST),'spec_sha256':SPEC_SHA,'limits':limits};write(FORECAST,frozen)
    prior=389.7378388913742;board=227.7205;ratio=.83
    alternative={'central':float(np.sqrt(board**2-ratio*(prior+r['G']))),'conditional_range':[float(np.sqrt(board**2-ratio*(prior+r['interval'][k]))) for k in ['hi95','lo95']],'anchor_v52':board,'ratio':ratio,'source_rows':[902],'prior_matched_G_since_v52':prior,'increment_G':r['G'],'total_matched_G_since_v52':prior+r['G'],'increment_interval':r['interval'],'frozen_at_utc':now(),'candidate_exists':False,'read_sha256':sha(READ),'manifest_sha256':sha(MANIFEST),'spec_sha256':SPEC_SHA,'limits':limits+' This step interval fixes the board anchor, prior gain and ratio; no ratio uncertainty.'};write(RATIO,alternative)
    print('FROZEN',frozen['central'],frozen['conditional_range'],'RATIO083',alternative['central'],alternative['conditional_range'],flush=True)

def validate_inputs():
    r=js(READ);f=js(FORECAST);alt=js(RATIO);m=js(MANIFEST)
    require_pass(REVIEW);assert PARENT_SHA in REVIEW.read_text();require_pass(D/'SOURCE-PASS.md')
    assert PACK_SHA in (D/'SOURCE-PASS.md').read_text() and VECTOR_SHA in (D/'SOURCE-PASS.md').read_text()
    assert r['gates']['pass'] and sha(PARENT)==PARENT_SHA and sha(SPEC)==SPEC_SHA==r['spec_sha256']==f['spec_sha256']==alt['spec_sha256']==m['spec_sha256']
    assert sha(READ)==f['read_sha256']==alt['read_sha256'] and sha(MANIFEST)==r['manifest_sha256']==f['manifest_sha256']==alt['manifest_sha256']
    for p,h in m['source_sha256'].items():assert sha(ROOT/p)==h,p
    for n,h in r['code_sha256'].items():assert sha(E/'code'/n)==h,n
    return r

def build():
    assert not CANDIDATE.exists();r=validate_inputs();checks=js(O/'stack12-checks-028.json');assert checks['pass'] and checks['read']['sha256']==sha(READ)
    paths=[SPEC,READ,FORECAST,RATIO,MANIFEST,O/'stack12-checks-028.json',*[E/'code'/n for n in CODE]]
    paths += [ROOT/p for p in js(MANIFEST)['source_sha256'] if 'stack12-028/' in p and not p.endswith('.parquet')]
    for p in paths:S8.s7.committed(p)
    b,m,a,t,parent,obj=load('ranking');out=composed(b,m,t,obj)['stack12'];assert digest(b[ID],out)==r['prediction_digests']['ranking']
    ids=pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID];np.testing.assert_array_equal(ids,b[ID]);assert len(ids)==344841 and ids.is_unique
    result=pd.DataFrame({ID:ids.to_numpy('float64'),COL:out.astype('float64')});assert np.isfinite(result.to_numpy()).all() and result[COL].min()>=60
    unchanged();parquet(result,CANDIDATE);pd.testing.assert_frame_equal(pd.read_parquet(CANDIDATE),result,check_exact=True)
    write(O/'stack12-build-028.json',{'candidate':record(CANDIDATE),'spec':record(SPEC),'read':record(READ),'forecast':record(FORECAST),'ratio083':record(RATIO),'manifest':record(MANIFEST),'checks':record(O/'stack12-checks-028.json'),'parent_review':record(REVIEW),'source_review':record(D/'SOURCE-PASS.md'),'built_at_utc':now(),'commit_before_build':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'review':'desk D123 pending','upload':False})
    print('BUILT',sha(CANDIDATE),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare-read','build']);args=parser.parse_args()
    prepare() if args.stage=='prepare-read' else build()
