"""SPEC030: replace only the served ranking route; retain all held statistics."""
import argparse, hashlib, io, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import exploit029 as X
from route_swap import make_route
from lane_c_swap import make_transition, matrix_apply
from verdict026 import require_pass
S=X.S;P=S.P;R=S.R;Q=S.Q;E=X.E;ROOT=X.ROOT;ID=X.ID;COL=X.COL
D=E/'inputs/final-route-030';O=E/'results';SPEC=E/'SPEC-030.md'
SOURCE=D/'SOURCE.json';MANIFEST=D/'MANIFEST.json'
PARENT=E/'exp169-stack12-exploit-bounds-029.parquet'
PARENT_SHA='22b71023e6729b117c2508d6d7146e8873a664632da49f164f1f42e12a1be54a'
CANDIDATE=E/'exp169-stack12-exploit-final-R4mks-030.parquet'
REVIEW=ROOT/'docs/status/2026-10-03-exp169-exploit-stack12-review/review-exploit-stack12.log'
REVIEW_PROMPT=REVIEW.parent/'REVIEW-PROMPT-exploit-stack12.md'
CODE=['final_route030.py','check_final_route030.py']
V32_SHA='f09f609429c81fd888e19e75e16f51b57978d55cc2639d6161c894da0baaa0de'
FILES={'b50':'ranking-final-R4mks-B50.parquet','b50_receipt':'ranking-final-R4mks-B50.json','kno':'ranking-final-R4mks-kno.parquet','kno_receipt':'ranking-final-R4mks-kno.json','v32':'v32-exp137-proxy.parquet','raw':'R4mks-final.parquet','fit':'R4mks-final.json','check':'check-final-route-R4mks.json','review':'SOURCE-REVIEW.md'}
sha=X.sha;js=X.js;write=X.write;record=X.record;digest=X.digest;now=X.now

def unchanged():
    files=js(O/'final-route-before-030.json')['files']
    for p,h in files.items():assert sha(ROOT/p)==h,p
    return len(files)

def save_parquet(q,p):
    buf=io.BytesIO();q.to_parquet(buf,index=False);body=buf.getvalue()
    if p.exists():assert p.read_bytes()==body,p
    else:p.write_bytes(body)
    return record(p)

def require_bound_review(review,full_hash,prompt=None):
    require_pass(review);text=Path(review).read_text()
    if full_hash not in text:
        assert prompt is not None and full_hash in Path(prompt).read_text(),'Review lacks full artifact binding.'
        assert full_hash[:8] in text.splitlines()[0],'Review header is not bound to its exact prompt.'

def schema(q,ids):
    assert list(q.columns)==[ID,COL] and len(q)==344841 and q[ID].is_unique
    assert all(q[c].dtype==np.dtype('float64') for c in [ID,COL])
    np.testing.assert_array_equal(q[ID],ids)
    assert np.isfinite(q.to_numpy()).all() and q[COL].min()>=60

def training_fields(stats):
    # Ranking support may change for C95; the fitting data and coefficients may not.
    return {k:[{a:b for a,b in row.items() if a!='n_serve'} for row in rows] for k,rows in stats.items()}

def parent_checks():
    X.validate('stack12');assert sha(PARENT)==PARENT_SHA
    require_bound_review(REVIEW,PARENT_SHA,REVIEW_PROMPT)
    old=js(E/'FREEZE-029.json')
    for p,h in old['records'].items():assert sha(ROOT/p)==h,p
    return {'file':record(PARENT),'review':record(REVIEW),'review_prompt':record(REVIEW_PROMPT),'freeze029':record(E/'FREEZE-029.json')}

def held():
    b,m,a,t,pre,obj=S.load('fold');steps=S.composed(b,m,t,obj)
    saved=pd.read_parquet(O/'stack12-held-028.parquet')
    for k in ['pre_exploit','stack12']:np.testing.assert_array_equal(steps[k],saved[k])
    rules=pd.read_parquet(E/'inputs/exploit-029/rules-fold.parquet');np.testing.assert_array_equal(rules[ID],b[ID])
    ex=X.K.apply(rules,steps['stack12'],include67=False)['final']
    want=pd.read_parquet(E/'inputs/exploit-029/stack12-fold-values.parquet')
    np.testing.assert_array_equal(ex,want.final)
    assert digest(b[ID],ex)==js(X.paths('stack12')['read'])['prediction_digests']['fold']
    shadow=pd.read_parquet(E/'inputs/exploit-029/stack12-shadow.parquet')
    np.testing.assert_array_equal(X.K.apply(rules,shadow.parent_shadow.to_numpy(),include67=False)['final'],shadow.candidate_shadow)
    return {'bundle':b,'pre':pre,'stack12':steps['stack12'],'exploit':ex,'digests':{'route':digest(b[ID],b.R4mk.to_numpy()),'pre':digest(b[ID],pre),'stack12':digest(b[ID],steps['stack12']),'exploit':digest(b[ID],ex),'shadow':digest(shadow[ID],shadow.candidate_shadow.to_numpy())}}

def ranking(route,h):
    """Pure ranking rebuild. No source gate bypass in the separate build command."""
    b,m,head,old_transition,_,_=P.context('ranking')
    rec10=js(P.MANIFEST);rec12=js(S.MANIFEST)
    n,a=make_route(b,np.asarray(route,float),old_transition,head,m,'ranking',rec10['pair_fold_weights'],rec10['pair_ranking_weights'])
    frame=pd.read_parquet(ROOT/rec12['splits']['ranking']['matrix']['path'])
    t=make_transition(a,m,frame,rec12['fold_weights'],rec12['ranking_weights'],'ranking')
    assert t.matrix_digest==rec12['splits']['ranking']['matrix_digest']
    pd.testing.assert_frame_equal(n.drop(columns='R4mk'),pd.read_parquet(E/'inputs/stack10-025/bundle-ranking.parquet').drop(columns='R4mk'),check_exact=True)
    assert int(n.rule.sum())==173539 and int(n.extension.sum())==17017 and int(n.noplan.sum())==383
    pre=S.composed(n,m,t)
    train=pd.read_parquet(Q.D/'frame.parquet');test=pd.read_parquet(S.S.D/'frame-ranking.parquet')
    obj,stats,v,nn=R.prepare_rule(train,h['pre'],test,pre,n,pooled=True)
    assert training_fields(stats)==training_fields(js(S.D/'statistics.json')['ranking'])
    steps=S.composed(n,m,t,obj);S.independent_rule(test,pre,obj,steps)
    exrules=pd.read_parquet(E/'inputs/exploit-029/rules-ranking.parquet');np.testing.assert_array_equal(exrules[ID],n[ID])
    ex=X.K.apply(exrules,steps['stack12'],include67=False)['final']
    # Independent member equation and unchanged tail.
    prior=a.pair.apply(m.ids,a.new_base,enable_extension=True)
    member=matrix_apply(a.new_base,prior,t.values,t.weights,m.scope,m.ordinary)
    np.testing.assert_array_equal(P.S8.s7.independent_tail(n,member),pre)
    # Source missing-AOBT exclusions apply to the pair term, not to the route.
    oldpair=P.load_member(E/'inputs/stack3-ext-009/pair.json',sha(E/'inputs/stack3-ext-009/pair.json'),b,'ranking')
    excluded=oldpair.scope&oldpair.extension&~m.fallback.scope;assert excluded.sum()==60
    np.testing.assert_array_equal(prior[excluded],a.new_base[excluded])
    oldroute=pd.read_parquet(P.D/'ranking-fold-R4mks-B50.parquet')[COL].to_numpy()
    np.testing.assert_array_equal(route[n.noplan],oldroute[n.noplan])
    # The complete parent already has C55 floors on nine no-plan rows.
    np.testing.assert_array_equal(pre[n.noplan],pd.read_parquet(S.CANDIDATE)[COL].to_numpy()[n.noplan])
    # C66 and other admitted downstream rules can change these cached outputs.
    final0=pd.read_parquet(PARENT)[COL].to_numpy();np.testing.assert_array_equal(ex[n.noplan],final0[n.noplan])
    np.testing.assert_array_equal(ex[n.bet],final0[n.bet])
    assert (exrules.lower<=exrules.upper).all()
    np.testing.assert_array_equal(ex,np.clip(ex,exrules.lower,exrules.upper))
    return {'bundle':n,'base':a.new_base,'pre':pre,'steps':steps,'final':ex,'rule':obj,'table':Q.table(obj,v,nn),'statistics':stats,'matrix_digest':t.matrix_digest,'pair_weights':rec10['pair_ranking_weights'],'ranking_weights':rec12['ranking_weights'],'excluded_pair_rows':int(excluded.sum()),'parent_digest':obj.parent_digest,'digests':{'route':digest(n[ID],route),'base':digest(n[ID],a.new_base),'pre':digest(n[ID],pre),'stack12':digest(n[ID],steps['stack12']),'exploit':digest(n[ID],ex)}}

def keep_mask(bundle,legal):
    cells=P.GS.load_cells()
    unsupported={(int(mon),str(ap),str(cls)) for mon,ap,cls,supported in cells[['month','airport','cls','supported']].itertuples(index=False,name=None) if not supported}
    unsup=np.array([(int(mon),str(ap),str(cls)) in unsupported for mon,ap,cls in zip(bundle.month,bundle.airport,bundle.cls)])
    return unsup|bundle.cls.eq('noac').to_numpy()|(bundle.cls.eq('ordinary').to_numpy()&legal.AOBT_3_flt.isna().to_numpy())

def fit_contract(fit):
    assert fit['stage']=='R4mks' and fit['owner']=='qfinal' and fit['inputs']=='r4mks'
    assert fit['train_rows']==1540605 and fit['scored_rows']==fit['rows']==344841
    assert fit['params']['num_threads']==16 and set(map(int,fit['train_rows_by_month']))=={1,2,3,6,7,8,10,11,12}
    assert fit['loss_mask_sets']==['noac','ordna']
    assert len(fit['cv'])==9 and fit['model_sha256'] and fit['cat_model_sha256']

def rms_admitted(rms):
    assert np.isfinite(rms) and 19<=rms<=34,'Final-route RMS is outside the desk-admitted 19–34 range.'


def source_checks(paths):
    b50=pd.read_parquet(paths['b50']);ids=pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID];schema(b50,ids)
    require_bound_review(paths['review'],sha(paths['b50']))
    fit=js(paths['fit']);c=js(paths['check']);br=js(paths['b50_receipt']);kr=js(paths['kno_receipt'])
    fit_contract(fit)
    assert fit['output_sha256']==sha(paths['raw'])
    assert c['fails']==[] and c['record']['unexpected']==[] and c['record']['train_rows']==1540605 and c['record']['scored_rows']==344841
    assert c['record']['threads']==16 and c['time']['rc']==0 and c['time']['seconds']>0
    assert c['vectors']['ids_equal_reference'] and c['vectors']['finite'] and c['vectors']['floor_60'] and c['vectors']['anchor_max_abs_vs_reference']<=1e-6
    assert c['served']['kno_rows_from_v32']==kr['rows_from_v32']==26310
    for k in ['b50','kno','raw']:assert sha(paths[k]) in c['inputs_sha256'].values(),k
    assert P.ROUTE_HASH['ranking'] in c['inputs_sha256'].values()
    assert br['w_on_a']==.5 and br['rows']==344841 and br['output_sha256']==sha(paths['b50'])
    assert br['inputs_sha256']['a']==sha(paths['kno'])==kr['output_sha256']
    assert br['inputs_sha256']['b']==sha(paths['v32'])==V32_SHA==kr['inputs_sha256']['v32']
    assert kr['inputs_sha256']['final']==sha(paths['raw']) and kr['rows']==344841
    assert kr['choice']['form_column']=='pred_Sb' and kr['choice']['w']==1 and kr['choice']['keep_sets']==['noac','ordna']
    kno=pd.read_parquet(paths['kno']);v32=pd.read_parquet(paths['v32']);schema(kno,ids);schema(v32,ids)
    np.testing.assert_allclose(b50[COL],.5*kno[COL]+.5*v32[COL],rtol=0,atol=1e-8)
    raw=pd.read_parquet(paths['raw']);assert raw[ID].is_unique and set(raw[ID])==set(ids)
    raw=raw.set_index(ID).loc[ids];assert np.isfinite(raw.pred_Sb).all() and (raw.pred_Sb>=60).all()
    # The old bundle has the already-fixed class and AOBT availability flag.
    bundle=pd.read_parquet(E/'inputs/stack10-025/bundle-ranking.parquet')
    # Actual keep mask is read from legal ranking fields, not inferred from equal predictions.
    legal=pd.read_parquet(ROOT/'data/raw/ranking.parquet',columns=[ID,'AOBT_3_flt']).set_index(ID).loc[ids]
    keep=keep_mask(bundle,legal)
    assert int(keep.sum())==kr['rows_from_v32']
    np.testing.assert_allclose(kno[COL],np.where(keep,v32[COL],raw.pred_Sb),rtol=0,atol=1e-8)
    fold=pd.read_parquet(P.D/'ranking-fold-R4mks-B50.parquet');schema(fold,ids)
    rms=float(np.sqrt(np.mean((b50[COL].to_numpy()-fold[COL].to_numpy())**2)))
    rms_admitted(rms)
    assert abs(rms-c['served']['B50']['rms_vs_own_fold_form'])<1e-8
    return b50[COL].to_numpy(),{'rms_vs_fold':rms,'rows_kept':int(keep.sum()),'B50_identity':True,'owner_qfinal':True,'source_review_PASS':True,'check':record(paths['check'])}

def stage(source_file):
    assert not SOURCE.exists();r=js(source_file);assert set(r['files'])==set(FILES)
    paths={}
    for key,value in r['files'].items():
        p=Path(value['path']);p=p if p.is_absolute() else ROOT/p
        assert sha(p)==value['sha256'],key;paths[key]=p
    _,proof=source_checks(paths)
    dest={}
    for k,p in paths.items():
        q=D/FILES[k]
        if q.exists():assert q.read_bytes()==p.read_bytes()
        else:q.write_bytes(p.read_bytes())
        dest[k]=record(q)
    write(SOURCE,{'files':dest,'original':r['files'],'intake':record(source_file),'checks':proof,'staged_at_utc':now()})

def change(b,old,new):
    delta=new-old
    def view(q):
        d=delta[q];return {'rows':int(q.sum()),'changed':int((d!=0).sum()),'mean':float(d.mean()),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d)))}
    allrows=np.ones(len(b),bool)
    masks={k:b[k].to_numpy(bool) for k in ['rule','extension','noplan','practice','c73','bet']}
    masks.update({f'class_{k}':b.cls.eq(k).to_numpy() for k in b.cls.unique()})
    masks.update({f'month_{k}':b.month.eq(k).to_numpy() for k in b.month.unique()})
    return {'all':view(allrows),'views':{k:view(v) for k,v in masks.items() if v.any()}}

def prepare():
    assert not MANIFEST.exists() and not CANDIDATE.exists();parent=parent_checks();unchanged()
    source=js(SOURCE);paths={k:ROOT/v['path'] for k,v in source['files'].items()}
    for k,p in paths.items():assert sha(p)==source['files'][k]['sha256'],k
    route,proof=source_checks(paths);h=held();z=ranking(route,h)
    oldroute=pd.read_parquet(P.D/'ranking-fold-R4mks-B50.parquet')[COL].to_numpy();old=ranking(oldroute,h)
    np.testing.assert_array_equal(old['final'],pd.read_parquet(PARENT)[COL])
    buf=io.BytesIO();pd.DataFrame({ID:z['bundle'][ID],COL:old['final']}).to_parquet(buf,index=False)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT_SHA
    artifacts={}
    for k,q in [('bundle',z['bundle']),('rules',z['table']),('ranking-values',pd.DataFrame({ID:z['bundle'][ID],'route':route,'base':z['base'],**z['steps'],'final_exploit':z['final']}))]:
        artifacts[k]=save_parquet(q,D/f'{k}.parquet')
    write(D/'statistics.json',z['statistics'])
    held_check=held();assert held_check['digests']==h['digests']
    diagnostic={k:change(z['bundle'],a,b) for k,a,b in [('route',oldroute,route),('base',old['base'],z['base']),('pre_exploit',old['pre'],z['pre']),('stack12',old['steps']['stack12'],z['steps']['stack12']),('final_exploit',old['final'],z['final'])]}
    data={'spec':record(SPEC),'source':record(SOURCE),'parent':parent,'source_checks':proof,'held_digests':h['digests'],'ranking_digests':z['digests'],'matrix_digest':z['matrix_digest'],'pair_ranking_weights':z['pair_weights'],'ranking_weights':z['ranking_weights'],'artifacts':artifacts,'statistics':record(D/'statistics.json'),'diagnostics':diagnostic,'excluded_pair_rows':60,'measured_new_held_G':None,'new_held_score':False,'bets':False,'published_means':'not built','code_sha256':{n:sha(E/'code'/n) for n in CODE},'snapshot':record(O/'final-route-before-030.json'),'old_artifacts_unchanged':unchanged(),'prepared_at_utc':now()}
    write(MANIFEST,data)
    for kind,key in [('frozen','forecast'),('ratio083','ratio')]:
        parentpath=X.paths('stack12')[key];f=js(parentpath)
        write(O/f'final-route-forecast-{kind}-030.json',{'central':f['central'],'conditional_range':f['conditional_range'],'ratio':f['ratio'],'parent_forecast':record(parentpath),'source_rows':f['source_rows'],'spec':record(SPEC),'manifest':record(MANIFEST),'additional_gain_credit':0,'meaning':'Inherited held-stack forecast for the final ranking model; final refit has no held score of its own.','limits':f['limits']+' Final-refit uncertainty is not covered.','frozen_at_utc':now(),'candidate_exists':False})
    print('PREPARED final route; inherited forecasts; no candidate',flush=True)

def build():
    from check_final_route030 import actual_checks
    assert not CANDIDATE.exists();actual_checks();m=js(MANIFEST)
    # The desk authorized building on source arrival after the fixed gates pass.
    q=pd.read_parquet(D/'ranking-values.parquet');result=q[[ID]].copy();result[COL]=q.final_exploit.to_numpy('float64')
    schema(result,pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID])
    assert digest(result[ID],result[COL].to_numpy())==m['ranking_digests']['exploit']
    for p in [O/'final-route-forecast-frozen-030.json',O/'final-route-forecast-ratio083-030.json']:
        f=js(p);assert f['manifest']==record(MANIFEST) and f['spec']==record(SPEC)
    unchanged();save_parquet(result,CANDIDATE)
    write(O/'final-route-build-030.json',{'candidate':record(CANDIDATE),'manifest':record(MANIFEST),'spec':record(SPEC),'checks':record(O/'final-route-final-checks-030.json'),'forecasts':[record(O/f'final-route-forecast-{k}-030.json') for k in ['frozen','ratio083']],'built_at_utc':now(),'HEAD_at_receipt':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'D123':'pending desk','upload':False})
    print('BUILT',sha(CANDIDATE),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','prepare','build']);ap.add_argument('--source',type=Path);args=ap.parse_args()
    if args.action=='stage':stage(args.source)
    elif args.action=='prepare':prepare()
    else:build()
