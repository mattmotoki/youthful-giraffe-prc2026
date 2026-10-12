"""Unit030 parity and adverse-input checks; no new scored read or candidate."""
import argparse,copy,hashlib,io,tempfile
from pathlib import Path
import numpy as np,pandas as pd
import final_route030 as F
E=F.E;O=F.O;D=F.D;ROOT=F.ROOT;ID=F.ID;COL=F.COL
FIT_FIELD = 'owner'  # receipt field that names the fit

def reject(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,TypeError):return
    raise AssertionError('Bad input was accepted.')

def parity_checks():
    F.parent_checks();F.unchanged();h=F.held()
    route=pd.read_parquet(F.P.D/'ranking-fold-R4mks-B50.parquet')[COL].to_numpy()
    z=F.ranking(route,h)
    stack=pd.read_parquet(F.S.CANDIDATE);exploit=pd.read_parquet(F.PARENT)
    np.testing.assert_array_equal(z['steps']['stack12'],stack[COL]);np.testing.assert_array_equal(z['final'],exploit[COL])
    for name,values,path in [('stack12',z['steps']['stack12'],F.S.CANDIDATE),('exploit',z['final'],F.PARENT)]:
        buf=io.BytesIO();pd.DataFrame({ID:z['bundle'][ID],COL:values}).to_parquet(buf,index=False)
        assert hashlib.sha256(buf.getvalue()).hexdigest()==F.sha(path),name
    pd.testing.assert_frame_equal(z['table'],pd.read_parquet(F.S.D/'rules-ranking.parquet'),check_exact=True)
    assert z['statistics']==F.js(F.S.D/'statistics.json')['ranking']
    masks={k:z['bundle'][k].to_numpy().copy() for k in ['rule','extension','noplan','practice','c73','bet']}
    # Fixed in-memory correctness probe, not a model/candidate or held score.
    # Cover primary, ordinary and extension paths with +1s on route-served rows.
    probe=route.copy();scope=z['bundle'].cls.ne('noac').to_numpy()&~masks['noplan']&~masks['bet'];probe[scope]+=1.
    perturbed=F.ranking(probe,h)
    assert perturbed['digests']['exploit']!=z['digests']['exploit']
    assert perturbed['matrix_digest']==z['matrix_digest'] and perturbed['pair_weights']==z['pair_weights'] and perturbed['ranking_weights']==z['ranking_weights']
    assert F.training_fields(perturbed['statistics'])==F.training_fields(z['statistics'])
    for k,m in masks.items():np.testing.assert_array_equal(perturbed['bundle'][k],m)
    assert F.held()['digests']==h['digests']
    # C95 scope and C98 neighbour steps must be recomputed, not copied deltas.
    assert np.any(perturbed['rule'].step98!=z['rule'].step98)
    reject(lambda:z['rule'].apply(z['bundle'][ID].to_numpy(),perturbed['pre']))
    bad=h.copy();bad['pre']=h['pre']+1
    reject(lambda:F.ranking(route,bad))
    rejects=['wrong rule parent','changed held training residuals']
    legal=pd.read_parquet(ROOT/'data/raw/ranking.parquet',columns=[ID,'AOBT_3_flt']).set_index(ID).loc[z['bundle'][ID]]
    keep=F.keep_mask(z['bundle'],legal);assert int(keep.sum())==26310
    # Receipt-schema tests do not claim the pending source receipt has passed.
    fake={'stage':'R4mks',FIT_FIELD:'qfinal','inputs':'r4mks','train_rows':1540605,'scored_rows':344841,'rows':344841,'params':{'num_threads':16},'train_rows_by_month':{str(k):1 for k in [1,2,3,6,7,8,10,11,12]},'loss_mask_sets':['noac','ordna'],'cv':{str(k):{} for k in range(9)},'model_sha256':'fixture','cat_model_sha256':'fixture'}
    F.fit_contract(fake)
    for key,value in [('stage','R4'),('owner','q01'),('inputs','r4mk'),('train_rows',1540604),('rows',344840),('cv',{}),('loss_mask_sets',['noac'])]:
        bad=copy.deepcopy(fake);bad[key]=value;reject(lambda:F.fit_contract(bad));rejects.append('fit '+key)
    for rms in [0.,18.999,34.001,float('nan')]:reject(lambda:F.rms_admitted(rms));rejects.append('RMS '+str(rms))
    F.rms_admitted(19.);F.rms_admitted(34.)

    for text in ['VERDICT: FAILED','VERDICT FAIL','Verdict = FAIL','Overall verdict: FAIL','| Verdict | FAIL |','Verdict: [FAIL]','VERDICT — FAIL','VERDICT: PASS\nFINAL VERDICT: FAIL','```\nVERDICT: PASS\n```','> VERDICT: PASS','No verdict']:
        with tempfile.TemporaryDirectory(dir=D,prefix='verdict-test-') as tmp:
            p=Path(tmp)/'review.md';p.write_text(F.PARENT_SHA+'\n'+text)
            reject(lambda:F.require_bound_review(p,F.PARENT_SHA))
        rejects.append(text)
    q=exploit.copy();F.schema(q,q[ID])
    for label,change in [('order',lambda q:q.iloc[::-1].reset_index(drop=True)),('dtype',lambda q:q.astype({ID:'int64'})),('missing row',lambda q:q.iloc[:-1]),('nan',lambda q:q.assign(**{COL:np.full(len(q),np.nan)})),('floor',lambda q:q.assign(**{COL:np.full(len(q),59.)}))]:
        badq=change(q);reject(lambda:F.schema(badq,q[ID]));rejects.append('schema '+label)
    rec={'pass':True,'spec':F.record(F.SPEC),'code_sha256':{n:F.sha(E/'code'/n) for n in F.CODE},'checked_at_utc':F.now(),'parent':F.record(F.PARENT),'old_route_stack12_byte_exact':True,'old_route_exploit_byte_exact':True,'old_ranking_rules_statistics_exact':True,'held_digests':h['digests'],'in_memory_probe_changed_ranking_only':True,'weights_member_values_masks_fixed':True,'training_statistics_fixed':True,'ranking_C98_recomputed':True,'rejected':rejects,'old_artifacts_unchanged':F.unchanged(),'actual_final_source_tested':False,'source_checks_pending':True,'fixed_source_keep_rows':26310,'candidate_exists':F.CANDIDATE.exists()}
    F.write(O/'final-route-adapter-checks-030.json',rec);print('PASS adapter parity/probe/adverse checks; source pending',flush=True)

def actual_checks():
    m=F.js(F.MANIFEST);assert m['spec']==F.record(F.SPEC)
    for n,s in m['code_sha256'].items():assert F.sha(E/'code'/n)==s,n
    proof=F.js(O/'final-route-adapter-checks-030.json');assert proof['pass']
    assert proof['code_sha256']==m['code_sha256'] and proof['spec']==m['spec']
    assert m['source']==F.record(F.SOURCE) and m['snapshot']==F.record(O/'final-route-before-030.json')
    assert m['parent']==F.parent_checks()
    source=F.js(F.SOURCE);paths={k:ROOT/v['path'] for k,v in source['files'].items()}
    for k,p in paths.items():assert F.sha(p)==source['files'][k]['sha256'],k
    route,sc=F.source_checks(paths);assert sc==m['source_checks']
    h=F.held();assert h['digests']==m['held_digests']==proof['held_digests']
    z=F.ranking(route,h);assert z['digests']==m['ranking_digests'] and z['matrix_digest']==m['matrix_digest']
    assert z['pair_weights']==m['pair_ranking_weights'] and z['ranking_weights']==m['ranking_weights']
    for key,r in m['artifacts'].items():assert F.sha(ROOT/r['path'])==r['sha256'],key
    assert F.record(D/'statistics.json')==m['statistics'] and z['statistics']==F.js(D/'statistics.json')
    pd.testing.assert_frame_equal(z['bundle'],pd.read_parquet(D/'bundle.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(z['table'],pd.read_parquet(D/'rules.parquet'),check_exact=True)
    values=pd.read_parquet(D/'ranking-values.parquet');np.testing.assert_array_equal(z['final'],values.final_exploit)
    # Independently replay the unit029 point sequence followed by its sole projection.
    f=pd.read_parquet(E/'inputs/exploit-029/rules-ranking.parquet');out=z['steps']['stack12']+f.order_delta.to_numpy()
    on=f.C66.to_numpy(bool);out[on]=f.point66.to_numpy()[on]
    on=f.C58.to_numpy(bool);out[on]+=.5*(f.point58.to_numpy()[on]-out[on])
    out=np.minimum(np.maximum(out,f.lower.to_numpy()),f.upper.to_numpy());np.testing.assert_array_equal(out,z['final'])
    result=pd.DataFrame({ID:z['bundle'][ID],COL:out});F.schema(result,pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID])
    for kind,key in [('frozen','forecast'),('ratio083','ratio')]:
        f=F.js(O/f'final-route-forecast-{kind}-030.json');old=F.js(F.X.paths('stack12')[key])
        for k in ['central','conditional_range','ratio','source_rows']:assert f[k]==old[k],k
        assert f['manifest']==F.record(F.MANIFEST) and f['parent_forecast']==F.record(F.X.paths('stack12')[key]) and f['additional_gain_credit']==0
    rec={'pass':True,'spec':F.record(F.SPEC),'manifest':F.record(F.MANIFEST),'held_digests_exact':h['digests'],'ranking_digests_exact':z['digests'],'independent_exploit_arithmetic_exact':True,'source_checks':sc,'old_artifacts_unchanged':F.unchanged(),'checked_at_utc':F.now(),'candidate_exists':F.CANDIDATE.exists(),'D123':'pending desk'}
    p=O/'final-route-final-checks-030.json'
    if p.exists():
        old=F.js(p)
        for k in rec:
            if k not in ['checked_at_utc','candidate_exists']:assert old[k]==rec[k],k
    else:F.write(p,rec)
    print('PASS actual final source, composition, unchanged held digests and forecasts',flush=True)
    return rec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','actual']);a=p.parse_args()
    parity_checks() if a.mode=='parity' else actual_checks()
