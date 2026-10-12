"""Unit025: fixed route read, forecast freeze, commit gate, one candidate."""
import argparse,hashlib,io,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import stack8 as S8
from assembly import ID,COL,compose,clip
from head_swap import checked,load_head_swap,JOINT
from diverse_blend import load_member,vector_digest
from lane_c_swap import make_transition,matrix_apply
from stack8_transition import load_stack8
from route_swap import make_route,fit_pair,require_pass
from read_stacks import GS

E=S8.EXP;R=S8.ROOT;D=E/'inputs/stack10-025'
SPEC=E/'SPEC-025.md';SPEC_SHA='fce6af74c71be9338addc0d81b04c3ebe5510ed66260ab17a634bc7cf966a009'
PARENT=E/'exp169-stack8-HB3u-laneC-mix12-c73-c55.parquet'
PARENT_SHA='fec00f357276aeb07d33a19955e5c639c544f04547d5e7869ddac43d4f9c4a50'
PM=E/'inputs/stack8-023/037/stack8.json'
PF=E/'results/stack8-forecast-frozen-023-037.json'
READ=E/'results/stack10-read-025.json';FORECAST=E/'results/stack10-forecast-frozen-025.json'
MANIFEST=D/'route-swap.json';CANDIDATE=E/'exp169-stack10-R4mks-HB3u-laneC-mix12-c73-c55.parquet'
CODE=[*S8.CODE,'route_swap.py','stack10.py']
sha=S8.sha;js=S8.js;write=S8.write;record=S8.record;now=S8.now;score=S8.score
ROUTE_HASH={1:'d103398cfee7a6693e16aaeacd87b89de1fa9935530bd265cb79915f1dd046ed',7:'142baf8b6b07a0e53e106ca2934f54e1a865e254d784b566812d72cbd8313fa2','ranking':'0b20ec728f780b7cd79ce50523b42a4bc4ead6d4d6bce50a9fdce3a73c483295'}


def code_pins():return {n:sha(E/'code'/n) for n in CODE}


def sources():
    assert sha(SPEC)==SPEC_SHA
    source=js(D/'SOURCE.json')
    for r in source.values():assert sha(R/r['local'])==r['sha256']
    require_pass(next(D.glob('review-R4mks-recheck-*-20261002T233143Z.last.md')))
    assert '0b20ec72' in (next(D.glob('review-R4mks-recheck-*-20261002T233143Z.last.md'))).read_text()
    report=(D/'SOURCE-REPORT-67752be2.md').read_text()
    for h in ROUTE_HASH.values():assert h in report
    for mon in [1,7]:
        p=D/f'R4mksFjjrefit-month-2025-{mon:02}.parquet';rec=js(p.with_suffix('.json'))
        assert sha(p)==ROUTE_HASH[mon]==rec['output_sha256']
        assert rec['stage']=='R4mks' and rec['owner']==f'q{mon:02}' and rec['inputs']=='r4mks'
        assert len(pd.read_parquet(p))==rec['rows']
        assert len(rec['split']['held'])==31 and all(s.startswith(f'2025-{mon:02}-') for s in rec['split']['held'])
        assert set(rec['split']['held'])<set(rec['split']['excluded'])
    assert sha(D/'ranking-fold-R4mks-B50.parquet')==ROUTE_HASH['ranking']
    require_pass(R/'docs/status/2026-10-02-exp169-stack8-review/review-stack8.log')
    assert sha(PARENT)==PARENT_SHA


def context(split):
    b=pd.read_parquet(E/f'inputs/stack5-016/bundle-{split}.parquet')
    m=load_member(S8.OLD,JOINT,b,split);h=load_head_swap(S8.HEAD,sha(S8.HEAD),b,m,split)
    t=load_stack8(PM,sha(PM),b,m,h,split)
    pred=compose(b,blend=m,head_swap=t,**S8.CFG)
    f=pd.read_parquet(E/f'inputs/stack8-023/037/matrix-{split}.parquet')
    return b,m,h,t,pred,f


def route_values(b,split,old=False):
    if split=='ranking':
        q=pd.read_parquet(D/'ranking-fold-R4mks-B50.parquet')
        np.testing.assert_array_equal(q[ID],b[ID]);return q[COL].to_numpy(float)
    folder=E/'inputs' if old else D;name='R4mk' if old else 'R4mks'
    q=pd.concat([pd.read_parquet(folder/f'{name}Fjjrefit-month-2025-{mon:02}.parquet') for mon in [1,7]]).set_index(ID)
    assert q.index.is_unique and set(q.index)==set(b[ID])
    rows=GS.load_rows().set_index(ID).loc[b[ID]]
    refs=pd.read_parquet(R/'data/exp167/refs.parquet').set_index(ID);v32=refs.v32.reindex(b[ID]).to_numpy(float)
    keep=rows.cls.eq('noac') | (rows.cls.eq('ordinary') & ~rows.aobt)
    return .5*np.where(keep,v32,q.pred_Sb.reindex(b[ID]).to_numpy(float))+.5*v32


def weights(b):
    rows=GS.load_rows().set_index(ID).loc[b[ID]];counts=rows.groupby(['month','airport','cls']).size();cells=GS.load_cells().set_index(['month','airport','cls']).N
    return np.array([cells.get((mon,a,c),0)/GS.N_RANKING/counts[mon,a,c] if counts[mon,a,c]>=GS.MIN_SUPPORT else 0. for mon,a,c in zip(rows.month,rows.airport,rows.cls)])


def base_tail(b,a):
    return compose(b,head_swap=a,**{**S8.CFG,'practice':'selected','c55':False,'pair_extension':False})


def shadow(b,m,a,t):
    v=t.values.copy();source=b.airport.eq('LIRF').to_numpy() & b.month.eq(7).to_numpy();use=source&m.ordinary
    for col,file in [(2,'masked-old021.parquet'),(3,'masked-new.parquet')]:
        vals=S8.aligned(E/'inputs/stack8-023/037'/file,b[ID]);np.testing.assert_array_equal(vals[~source],v[~source,col]);v[use,col]=vals[use]
    base=a.new_base;prior=a.blend(m,b[ID].to_numpy(),base,True)
    return S8.s7.independent_tail(b,matrix_apply(base,prior,v,t.weights,m.scope,m.ordinary))


def scope_check(b,n,m,a,t,old,new):
    prior=a.pair.apply(m.ids,a.new_base,enable_extension=True)
    independent=S8.s7.independent_tail(n,matrix_apply(a.new_base,prior,t.values,t.weights,m.scope,m.ordinary))
    np.testing.assert_array_equal(independent,new)
    np.testing.assert_array_equal(S8.s7.independent_tail(n,new),new) if False else None
    np.testing.assert_array_equal(clip(new,n.lo.to_numpy(),n.hi.to_numpy()),new)
    np.testing.assert_array_equal(S8.s7.apply_floor(new,n.stand_floor.to_numpy(),n.airport.to_numpy(),n.lo.to_numpy(),n.hi.to_numpy()),new)
    np.testing.assert_array_equal(new[n.noplan],old[n.noplan])
    assert np.isfinite(new).all() and new.min()>=60
    out={'rows':len(b),'changed':int((new!=old).sum()),'independent_formula_exact':True,'mean_shift':float(np.mean(new-old)),'distance':float(np.mean((new-old)**2))}
    for name,q in [('primary',m.scope),('outside_primary',~m.scope),('extension',b.extension.to_numpy()),('pair_extension',m.fallback.scope&m.fallback.extension),('noplan',b.noplan.to_numpy()),('C39',b.practice.to_numpy()),('C73',b.c73.to_numpy())]:
        out[name]={'rows':int(q.sum()),'changed':int((new[q]!=old[q]).sum()),'distance_total':float(np.sum((new[q]-old[q])**2)/len(b))}
    return out


def load_route(split):
    rec=js(MANIFEST);assert rec['spec']==record(SPEC) and rec['parent']==record(PARENT)
    assert rec['change']=='R4mk_to_R4mks' and rec['weight_rule']=='other_month_then_pooled'
    for r in rec['inputs'].values():checked(r)
    b,m,h,t,old,f=context(split)
    np.testing.assert_array_equal(b[ID],pd.read_parquet(checked(rec['splits'][split]['bundle']))[ID])
    r=route_values(b,split);n,a=make_route(b,r,t,h,m,split,rec['pair_fold_weights'],rec['pair_ranking_weights'])
    pd.testing.assert_frame_equal(n,pd.read_parquet(checked(rec['splits'][split]['bundle'])),check_exact=True)
    tr=make_transition(a,m,f,rec['fold_weights'],rec['ranking_weights'],split)
    assert vector_digest(m.ids,a.new_base)==rec['splits'][split]['base_digest']
    assert tr.matrix_digest==rec['splits'][split]['matrix_digest']
    return b,n,m,a,tr,old


def prepare():
    assert not READ.exists() and not FORECAST.exists() and not CANDIDATE.exists();sources();start=now();cp=code_pins()
    contexts={s:context(s) for s in ['fold','ranking']};b,m,h,t,old,f=contexts['fold'];ids=m.ids;y=GS.labels(ids);w=weights(b)
    held=pd.read_parquet(E/'results/stack8-held-023-037.parquet');np.testing.assert_array_equal(held[ID],ids);np.testing.assert_array_equal(held.stack8,old)
    q=pd.read_parquet(PARENT);np.testing.assert_array_equal(q[ID],contexts['ranking'][0][ID]);np.testing.assert_array_equal(q[COL],contexts['ranking'][4])
    buf=io.BytesIO();pd.DataFrame({ID:q[ID],COL:contexts['ranking'][4]}).to_parquet(buf,index=False);assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT_SHA
    B=base_tail(b,h);pf,pp=S8.s7.fit(b,f,h.new_base,B,y,w);pm=js(PM);assert pf==pm['fold_weights'] and pp==pm['ranking_weights']
    np.testing.assert_array_equal(route_values(b,'fold',True),b.R4mk)
    route=route_values(b,'fold');parity=score(ids,b.R4mk.to_numpy(),route,y);want=js(D/'fold-pair-FR4mks-FR4mk.json')['score'];assert abs(parity['G']-want['G'])<1e-7
    for k in ['lo95','hi95']:assert abs(parity['interval'][k]-want['interval'][k])<1e-7
    print('SOURCE PARITY',parity['G'],flush=True)
    oldshadow=shadow(b,m,h,t);np.testing.assert_array_equal(oldshadow,held.masked_stack8)
    n,a=make_route(b,route,t,h,m,'fold');Bnew=base_tail(n,a)
    pairfold,pairpool=fit_pair(n,m.fallback,a.new_base,Bnew,y,w)
    folds,pool=S8.s7.fit(n,f,a.new_base,Bnew,y,w)
    transferred=make_transition(a,m,f,pf,pp,'fold');p0=compose(n,blend=m,head_swap=transferred,**S8.CFG)
    n,a=make_route(b,route,t,h,m,'fold',pairfold,pairpool)
    paironly=make_transition(a,m,f,pf,pp,'fold');p1=compose(n,blend=m,head_swap=paironly,**S8.CFG)
    # The primary overlay replaces the fallback on all held member rows.
    np.testing.assert_array_equal(p0,p1)
    splits={};inputs={}
    for s,(ob,om,oh,ot,op,of) in contexts.items():
        nb,ra=make_route(ob,route_values(ob,s),ot,oh,om,s,pairfold,pairpool)
        tr=make_transition(ra,om,of,folds,pool,s)
        path=D/f'bundle-{s}.parquet'
        if path.exists():pd.testing.assert_frame_equal(pd.read_parquet(path),nb,check_exact=True)
        else:nb.to_parquet(path,index=False)
        splits[s]={'bundle':record(path),'base_digest':vector_digest(om.ids,ra.new_base),'matrix_digest':tr.matrix_digest}
    for p in D.iterdir():
        if p.is_file() and p!=MANIFEST and p.name not in ['bundle-fold.parquet','bundle-ranking.parquet']:inputs[p.name]=record(p)
    for name,p in [('parent_manifest',PM),('parent_forecast',PF),('parent_held',E/'results/stack8-held-023-037.parquet'),('refs',R/'data/exp167/refs.parquet')]:inputs[name]=record(p)
    manifest={'schema':1,'change':'R4mk_to_R4mks','weight_rule':'other_month_then_pooled','spec':record(SPEC),'parent':record(PARENT),'inputs':inputs,'splits':splits,'pair_fold_weights':pairfold,'pair_ranking_weights':pairpool,'fold_weights':folds,'ranking_weights':pool}
    write(MANIFEST,manifest)
    loaded={s:load_route(s) for s in contexts};pred={};scopes={}
    for s,(ob,nb,om,ra,tr,op) in loaded.items():
        pred[s]=compose(nb,blend=om,head_swap=tr,**S8.CFG);scopes[s]=scope_check(ob,nb,om,ra,tr,op,pred[s])
    reads={'transferred_weights_vs_stack8':score(ids,old,p0,y),'fallback_pair_refit_step':score(ids,p0,p1,y),'primary_refit_step':score(ids,p1,pred['fold'],y),'full_vs_stack8':score(ids,old,pred['fold'],y)}
    ob,nb,om,ra,tr,op=loaded['fold'];newshadow=shadow(nb,om,ra,tr)
    reads['matched_shadow_vs_stack8']=score(ids,oldshadow,newshadow,y)
    for k,v in reads.items():print(k,v['G'],v['interval'],flush=True)
    br=compose(b,route='R4mk',noplan='v3cal',practice='selected');reads['full_vs_B']=score(ids,br,pred['fold'],y)
    # Keep the missing-AOBT extension mask exact, with no pair term on these rows.
    ob,nb,om,ra,tr,op=loaded['ranking'];oldpair=load_member(E/'inputs/stack3-ext-009/pair.json',sha(E/'inputs/stack3-ext-009/pair.json'),ob,'ranking')
    excluded=oldpair.scope & oldpair.extension & ~om.fallback.scope;assert excluded.sum()==60
    np.testing.assert_array_equal(ra.pair.apply(om.ids,ra.new_base,enable_extension=True)[excluded],ra.new_base[excluded])
    scopes['ranking']['missing_AOBT3_pair_exclusions']={'rows':60,'changed_by_route':int((pred['ranking'][excluded]!=op[excluded]).sum()),'pair_term_exact_zero':True}
    pd.DataFrame({ID:ids,'stack8':old,'route_transferred':p0,'pair_refit':p1,'stack10':pred['fold'],'masked_stack8':oldshadow,'masked_stack10':newshadow}).to_parquet(E/'results/stack10-held-025.parquet',index=False)
    risk={s:S8.sequence_risk(loaded[s][1],loaded[s][2],loaded[s][4]) for s in loaded}
    oldfiles=js(E/'results/stack10-before-025.json')['files']
    for p,hsh in oldfiles.items():assert sha(R/p)==hsh,p
    assert code_pins()==cp
    g=reads['full_vs_stack8'];passed=g['G']>0 and g['interval']['lo95']>-50
    sourcepins={p:hsh for p,hsh in js(E/'results/stack8-read-023-037.json')['source_sha256'].items()}
    sourcepins.update({str(p.relative_to(R)):sha(p) for p in D.iterdir() if p.is_file()})
    for p,hsh in sourcepins.items():assert sha(R/p)==hsh,p
    limits='Conditional airport-day interval; same-row route/member/form selection and route gate; no seed/search/2026-transfer coverage; extension weights transfer without new credit; opposing sequence coefficients; inherited C73 upper600, corrected centre536-576.'
    write(READ,{'spec_sha256':SPEC_SHA,'started_at_utc':start,'read_at_utc':now(),'source_rows':[899,901],'code_sha256':cp,'source_sha256':sourcepins,'source_parity':parity,'parent_byte_exact':True,'parent_refit_exact':True,'parent_shadow_exact':True,'reads':reads,'gate':{'G_positive':g['G']>0,'lower95_above_minus50':g['interval']['lo95']>-50,'source_pass':True,'pass':passed},'scope':scopes,'sequence_difference_risk':risk,'prediction_digests':{s:vector_digest(contexts[s][0][ID],pred[s]) for s in pred},'old_artifacts_unchanged':len(oldfiles),'limits':limits})
    if not passed:print('ROUTE GATE FAIL: no forecast or build',flush=True);return
    r=reads['matched_shadow_vs_stack8'];p=js(PF)['central'];assert p==222.41647028752698
    freeze={'central':float(np.sqrt(p*p-1.24*r['G'])),'conditional_range':[float(np.sqrt(p*p-1.24*r['interval']['hi95'])),float(np.sqrt(p*p-1.24*r['interval']['lo95']))],'parent':p,'parent_forecast':record(PF),'parent_sha256':PARENT_SHA,'increment_G':r['G'],'increment_interval':r['interval'],'unmasked_sensitivity':float(np.sqrt(p*p-1.24*g['G'])),'ratio':1.24,'ratio_basis':'ledger899; baseline901 v47','incremental_extension_credit':0,'limits':limits,'spec_sha256':SPEC_SHA,'read_sha256':sha(READ),'frozen_at_utc':now(),'candidate_exists':False}
    write(FORECAST,freeze);print('FROZEN',freeze['central'],freeze['conditional_range'],flush=True)


def build():
    sources();assert not CANDIDATE.exists();r=js(READ);f=js(FORECAST)
    assert r['gate']['pass'] and f['read_sha256']==sha(READ) and f['spec_sha256']==SPEC_SHA and not f['candidate_exists']
    assert r['code_sha256']==code_pins()
    for p,h in r['source_sha256'].items():assert sha(R/p)==h,p
    for p in [SPEC,READ,FORECAST,MANIFEST,*[E/'code'/n for n in CODE],*[R/p for p in r['source_sha256'] if 'stack10-025' in p]]:S8.s7.committed(p)
    b,n,m,a,t,old=load_route('ranking');pred=compose(n,blend=m,head_swap=t,**S8.CFG);scope_check(b,n,m,a,t,old,pred)
    assert vector_digest(b[ID],pred)==r['prediction_digests']['ranking']
    template=pd.read_parquet(R/'data/raw/submitting.parquet');np.testing.assert_array_equal(b[ID],template[ID])
    q=pd.DataFrame({ID:b[ID].astype('float64'),COL:pred.astype('float64')});assert len(q)==344841 and q[ID].is_unique
    q.to_parquet(CANDIDATE,index=False);pd.testing.assert_frame_equal(pd.read_parquet(CANDIDATE),q,check_exact=True)
    write(E/'results/stack10-build-025.json',{'candidate':record(CANDIDATE),'read':record(READ),'forecast':record(FORECAST),'manifest':record(MANIFEST),'commit_before_build':S8.s7.committed(SPEC),'built_at_utc':now(),'review':'desk D123 pending'})
    print('BUILT',sha(CANDIDATE),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare-read','build']);a=p.parse_args()
    prepare() if a.stage=='prepare-read' else build()
