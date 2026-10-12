"""SPEC-023: timed source decision, two fixed member steps, one forecast and build."""
import argparse
from dataclasses import replace
import datetime,hashlib,io,json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import stack7 as s7
from assembly import ID,COL,compose
from head_swap import checked,fingerprint,load_head_swap,JOINT
from diverse_blend import vector_digest,load_member
from lane_c_swap import NAMES,make_transition,load_lane_c_swap,matrix_apply
from stack8_transition import load_stack8,sequence_receipt,PARENT
from review_verdict import require_pass
from read_stacks import GS

EXP=s7.EXP;ROOT=s7.ROOT;DIV=s7.DIV;MLP=s7.MLP
OUT=EXP/'inputs/stack8-023';SPEC=EXP/'SPEC-023.md'
SPEC_SHA='4e30eb76850c29e4d83b218f0400148c89ffed3da2ca65d01fa0a6ea2aed9632'
CUTOFF=datetime.datetime.fromisoformat('2026-10-02T23:30:00+00:00')
PARENT_FILE=EXP/'exp169-stack7-HB3u-laneC-c73-c55.parquet'
PARENT_MANIFEST=EXP/'inputs/stack7-022/030/lane-c-swap.json'
PARENT_FORECAST=EXP/'results/stack7-forecast-frozen-022-030.json'
PARENT_REVIEW=ROOT/'docs/status/2026-10-02-exp169-stack7-review/review-stack7.log'
CANDIDATE=EXP/'exp169-stack8-HB3u-laneC-mix12-c73-c55.parquet'
DECISION=EXP/'results/stack8-source-choice-023.json'
COMMAND=EXP/'results/stack8-build-command-023.json'
OLD=s7.OLD;HEAD=s7.HEAD;CFG=s7.CFG
CODE=[*s7.CODE,'stack8.py','stack8_transition.py','review_verdict.py']
sha=s7.sha;js=s7.js;now=s7.now;write=s7.write;record=s7.record;pin=s7.pin;stage=s7.stage;aligned=s7.aligned;score=s7.score
PINS=s7.PINS
PACKS={'033b':('pack-mixed18r6-pair-033b','paired-mixed18-033a-B','masked-mixed18-LIRF-033a-fit'),
       '037':('pack-mixed30r6-pair-037','paired-mixed30-037-B','masked-mixed30-LIRF-037-fit')}


def code_pins():return {n:sha(EXP/'code'/n) for n in CODE}


def paths(key):return OUT/key,EXP/f'results/stack8-read-023-{key}.json',EXP/f'results/stack8-forecast-frozen-023-{key}.json'


def source_record(key):
    pack=DIV/'results'/PACKS[key][0]/'PACK.json';p=js(pack)
    vector=pack.parent/'members-final.parquet';assert sha(vector)==p['vector_sha256']
    reviews=[DIV/'REVIEW-MIXED18-033.md'] if key=='033b' else sorted(DIV.glob('REVIEW*037*.md'))
    for review in reviews:
        text=review.read_text()
        if sha(pack) not in text or p['vector_sha256'] not in text:continue
        verdict=require_pass(review)
        return {'key':key,'pack':record(pack),'vector':record(vector),'review':record(review),'verdict_lines':verdict,'observed_at_utc':now()}
    raise AssertionError(f'No explicit reviewed pack identity for {key}')


def observe():
    OUT.mkdir(exist_ok=True)
    r=source_record('037');p=OUT/'reviewed-037-before-cutoff.json'
    if not p.exists():
        current=datetime.datetime.now(datetime.timezone.utc)
        evidence={'kind':'observed explicit PASS','time_utc':r['observed_at_utc']}
        if current>CUTOFF:
            review=checked(r['review']);rel=str(review.relative_to(ROOT));evidence=None
            log=subprocess.check_output(['git','log','--format=%H %cI','--',rel],cwd=ROOT,text=True)
            for line in reversed(log.splitlines()):
                commit,stamp=line.split(' ',1)
                if datetime.datetime.fromisoformat(stamp)>CUTOFF:continue
                content=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT)
                if hashlib.sha256(content).hexdigest()==r['review']['sha256']:
                    evidence={'kind':'exact review committed before cutoff','commit':commit,'time_utc':stamp};break
            assert evidence,'No record proves037PASS by cutoff.'
        r['deadline_evidence']=evidence
        write(p,r)
    else:
        old=js(p)
        for k in ['key','pack','vector','review']:assert old[k]==r[k]
    return js(p)


def choose():
    if DECISION.exists():
        r=js(DECISION)
        for key in ['pack','vector','review']:checked(r['source'][key])
        require_pass(checked(r['source']['review']));return r
    observed=OUT/'reviewed-037-before-cutoff.json'
    if not observed.exists():
        try:observe()
        except (FileNotFoundError,AssertionError):pass
    if observed.exists():
        r=js(observed);assert datetime.datetime.fromisoformat(r['deadline_evidence']['time_utc'])<=CUTOFF
        for key in ['pack','vector','review']:checked(r[key])
        require_pass(checked(r['review']))
    else:
        assert datetime.datetime.now(datetime.timezone.utc)>=CUTOFF,'Wait for037PASS or23:30UTC before final source choice.'
        # A late first observation needs an exact pre-cutoff commit to qualify.
        r=source_record('033b')
    out={'source':r,'rule':'037 explicit PASS observed by2026-10-02 23:30UTC else033b','chosen_at_utc':now(),'spec_sha256':SPEC_SHA}
    write(DECISION,out);return out


def source_inputs(key,dst,bundles,members,parents):
    choice=source_record(key)
    if key=='037':observe()
    pack=js(stage(checked(choice['pack']),dst/'PACK.json'));stage(checked(choice['review']),dst/'SOURCE-PASS.md')
    annual=pd.read_parquet(stage(checked(choice['vector']),dst/'pack-ranking.parquet'))
    np.testing.assert_array_equal(annual[ID],bundles['ranking'][ID]);np.testing.assert_array_equal(annual.member_scope,members['ranking'].scope)
    np.testing.assert_array_equal(annual.control_scope,members['ranking'].scope & members['ranking'].ordinary)
    np.testing.assert_array_equal(annual.pred_row170,parents['ranking'].values[:,-1])
    mp=stage(MLP/'results/stack-pack-012/MEMBER.json',dst/'MLP-MEMBER.json');mlp_member=js(mp)
    mr=stage(MLP/'results/pack-012/REVIEW-012.md',dst/'MLP-PASS.md');require_pass(mr)
    assert sha(mp)=='5ffc057af6fd5ccde5ba479420817b494e4fbbc09ee394641b8d77830b557de0'
    assert all(mlp_member['vectors'][k]['sha256'][:8] in mr.read_text() for k in ['fold','ranking'])
    reviewed=subprocess.check_output(['git','show','c105eb0b:experiments/exp-164-double-down/results/stack-pack-012/MEMBER.json'],cwd=ROOT)
    assert hashlib.sha256(reviewed).hexdigest()==sha(mp)
    write(dst/'MLP-REVIEW-BINDING.json',{'reviewed_commit':'c105eb0b','member_sha256':sha(mp),'review_sha256':sha(mr),'vector_hashes':{k:mlp_member['vectors'][k]['sha256'] for k in ['fold','ranking']}})
    assert mlp_member['member_training_months']['fold']==[2,3,4,5,6,8,9,10,11,12]
    assert mlp_member['member_training_months']['ranking_final']==list(range(1,13))
    stage(ROOT/mlp_member['bag_json']['path'],dst/'MLP-BAG.json',mlp_member['bag_json']['sha256'])
    frames={};mlps={};splits={};receipts={}
    for split in ['fold','ranking']:
        b=bundles[split];m=members[split];old=parents[split];typ='annual' if split=='ranking' else 'held'
        folder=DIV/'results'/(Path(pack['records']['sequence'][typ+'_fit_path']).name+'-fit')
        seq=stage(folder/('ranking2026.parquet' if split=='ranking' else 'jjrefit.parquet'),dst/f'sequence-{split}.parquet',pack['records']['sequence']['annual_raw_vector_sha256' if split=='ranking' else 'held_vector_sha256'])
        receipt=stage(folder/'FIT.json',dst/f'sequence-{split}-FIT.json',pack['records']['sequence'][typ+'_fit_sha256'])
        receipts[split]=sequence_receipt(pack,key,split,seq,receipt)
        f=b[[ID,'month','cls']].copy();f['member_scope']=m.scope
        for j,name in enumerate(NAMES):f[name]=old.values[:,j]
        f['sequence_new']=aligned(seq,b[ID]);frames[split]=f
        if split=='ranking':np.testing.assert_array_equal(f.sequence_new,annual.pred_seq170)
        vpath=stage(MLP/'results/stack-pack-012'/mlp_member['vectors'][split]['path'],dst/f'mlp-{split}.parquet',mlp_member['vectors'][split]['sha256'])
        v=pd.read_parquet(vpath)
        for k in [ID,'month','cls']:np.testing.assert_array_equal(v[k],b[k])
        assert v.mlp_scope.to_numpy(bool)[m.scope].all()
        src=mlp_member['sources']['ranking_final' if split=='ranking' else 'fold']
        raw=stage(ROOT/src['path'],dst/f'mlp-{split}-raw.parquet',src['sha256'])
        np.testing.assert_array_equal(v.pred_mlp164,aligned(raw,b[ID],'prediction'))
        assert src['bag_record']['weights']=={'drop0':.5,'e80':.5}
        for recipe,r in src['bag_record']['members'].items():assert r['seeds']==list(range(701,707))
        mlps[split]=v.pred_mlp164.to_numpy(float)
        splits[split]={'sequence_new':record(seq),'sequence_receipt':record(receipt),'mlp_vector':record(vpath)}
    folder=DIV/'results'/PACKS[key][1]
    stage(folder/'PAIRED.json',dst/'SOURCE-PAIR.json',pack['read_sha256']);stage(folder/'paired_ls.parquet',dst/'source-pair-held.parquet')
    folder=DIV/'results'/PACKS[key][2]
    stage(folder/'jjrefit.parquet',dst/'masked-new.parquet');mask=js(stage(folder/'PREDICT.json',dst/'masked-new-PREDICT.json'))
    assert mask['label_free'] and mask['unaffected_predictions_exact']
    assert mask['fit_sha256']==pack['records']['sequence']['held_fit_sha256'] and mask['files']['jjrefit.parquet']==sha(dst/'masked-new.parquet')
    for old,new in [('masked-old.parquet','masked-old021.parquet'),('masked-new.parquet','masked-parent030.parquet')]:stage(EXP/'inputs/stack7-022/030'/old,dst/new)
    return choice,pack,frames,mlps,splits,receipts


def matched_shadow(b,m,h,parent,new,dst):
    source=b.airport.eq('LIRF').to_numpy() & b.month.eq(7).to_numpy();use=source & m.ordinary
    old021=aligned(dst/'masked-old021.parquet',b[ID]);old030=aligned(dst/'masked-parent030.parquet',b[ID]);newseq=aligned(dst/'masked-new.parquet',b[ID])
    for values,actual in [(old021,parent.values[:,2]),(old030,parent.values[:,3]),(newseq,new.values[:,3])]:np.testing.assert_array_equal(values[~source],actual[~source])
    prior=h.blend(m,b[ID].to_numpy(),h.new_base,True);outs=[]
    for t,seq in [(parent,old030),(new,newseq)]:
        v=t.values.copy();v[use,2]=old021[use];v[use,3]=seq[use]
        z=matrix_apply(h.new_base,prior,v,t.weights,m.scope,m.ordinary);outs.append(s7.independent_tail(b,z))
    return outs


def sequence_risk(b,m,t):
    out=[]
    for cls in ['ordinary','clock']:
        for mon in [1,7]:
            q=m.scope & b.cls.eq(cls).to_numpy() & b.month.eq(mon).to_numpy()
            diff=t.values[q,3]-t.values[q,2]
            out.append({'class':cls,'month':mon,'rows':int(q.sum()),'old_weight':float(t.weights[q,2][0]),'new_weight':float(t.weights[q,3][0]),'difference_rms':float(np.sqrt(np.mean(diff**2)))})
    return out


def prepare(key):
    assert sha(SPEC)==SPEC_SHA and not CANDIDATE.exists()
    OUT.mkdir(exist_ok=True);dst,result,forecast=paths(key);dst.mkdir(exist_ok=True)
    assert not result.exists() and not forecast.exists()
    code=code_pins();pin(SPEC,SPEC_SHA);pin(PARENT_FILE,PARENT);pin(PARENT_FORECAST)
    start=now();bundles={};members={};heads={};parents={};parentpred={}
    review=stage(PARENT_REVIEW,dst/'PARENT-PASS.log');require_pass(review)
    for split in ['fold','ranking']:
        b=pd.read_parquet(pin(EXP/f'inputs/stack5-016/bundle-{split}.parquet'))
        m=load_member(pin(OLD,JOINT),JOINT,b,split);h=load_head_swap(pin(HEAD),sha(HEAD),b,m,split)
        old=load_lane_c_swap(pin(PARENT_MANIFEST),sha(PARENT_MANIFEST),b,m,h,split)
        bundles[split]=b;members[split]=m;heads[split]=h;parents[split]=old;parentpred[split]=compose(b,blend=m,head_swap=old,**CFG)
    q=pd.read_parquet(PARENT_FILE);np.testing.assert_array_equal(q[ID],bundles['ranking'][ID]);np.testing.assert_array_equal(q[COL],parentpred['ranking'])
    buf=io.BytesIO();pd.DataFrame({ID:q[ID],COL:parentpred['ranking']}).to_parquet(buf,index=False);assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT
    held7=pd.read_parquet(pin(EXP/'results/stack7-held-022-030.parquet'));np.testing.assert_array_equal(held7[ID],bundles['fold'][ID]);np.testing.assert_array_equal(held7.stack7,parentpred['fold'])
    choice,pack,frames,mlps,splits,receipts=source_inputs(key,dst,bundles,members,parents)
    b=bundles['fold'];m=members['fold'];h=heads['fold'];ids=b[ID].to_numpy('int64');y=GS.labels(ids)
    parity=s7.source_parity(b,m,frames['fold'],pack,dst,y)
    rows=GS.load_rows().set_index(ID).loc[ids];counts=rows.groupby(['month','airport','cls']).size();cells=GS.load_cells().set_index(['month','airport','cls']).N
    w=np.array([cells.get((mon,a,c),0)/GS.N_RANKING/counts[mon,a,c] if counts[mon,a,c]>=GS.MIN_SUPPORT else 0. for mon,a,c in zip(rows.month,rows.airport,rows.cls)])
    B3=compose(b,head_swap=h,**{**CFG,'practice':'selected','c55':False,'pair_extension':False})
    # Refit the parent before any new G; it must recover the original normal records.
    parentframe=pd.read_parquet(EXP/'inputs/stack7-022/030/matrix-fold.parquet')
    pf,pp=s7.fit(b,parentframe,h.new_base,B3,y,w);parentrec=js(PARENT_MANIFEST)
    assert pf==parentrec['fold_weights'] and pp==parentrec['ranking_weights']
    folds,pool=s7.fit(b,frames['fold'],h.new_base,B3,y,w)
    lane=make_transition(parents['fold'],m,frames['fold'],folds,pool,'fold');lane_pred=compose(b,blend=m,head_swap=lane,**CFG)
    reads={'lane_c_vs_stack7':score(ids,parentpred['fold'],lane_pred,y)}
    print('lane C',key,reads['lane_c_vs_stack7']['G'],reads['lane_c_vs_stack7']['interval'],flush=True)
    for split,f in frames.items():
        use=members[split].scope & ~members[split].ordinary;f.loc[use,'mlp']=mlps[split][use]
    ff,fp=s7.fit(b,frames['fold'],h.new_base,B3,y,w);assert ff['ordinary']==folds['ordinary'] and fp['ordinary']==pool['ordinary']
    folds,pool=ff,fp
    write(dst/'desk-approval.json',{'approved':True,'spec_sha256':SPEC_SHA,'parent_sha256':PARENT,'desk_order':'21:28UTC; timed037/033b rule and fixed clock mix12; one build after commit'})
    manifest={'schema':3,'form':'stack8','source_key':key,'source_observed':choice,'parent':record(PARENT_FILE),'parent_transition':record(PARENT_MANIFEST),'parent_review':record(review),
              'spec':record(SPEC),'desk_approval':record(dst/'desk-approval.json'),'pack':record(dst/'PACK.json'),'source_review':record(dst/'SOURCE-PASS.md'),
              'mlp_pack':record(dst/'MLP-MEMBER.json'),'mlp_review':record(dst/'MLP-PASS.md'),'fold_weights':folds,'ranking_weights':pool,'splits':splits}
    for split in bundles:
        matrix=dst/f'matrix-{split}.parquet';frames[split].to_parquet(matrix,index=False)
        t=make_transition(parents[split],members[split],frames[split],folds,pool,split)
        splits[split].update(matrix=record(matrix),matrix_digest=t.matrix_digest,bundle=record(EXP/f'inputs/stack5-016/bundle-{split}.parquet'),parent_transition_fingerprint=fingerprint(parents[split]),base_digest=vector_digest(members[split].ids,heads[split].new_base))
    write(dst/'stack8.json',manifest)
    ts={k:load_stack8(dst/'stack8.json',sha(dst/'stack8.json'),bundles[k],members[k],heads[k],k) for k in bundles}
    pred={k:compose(bundles[k],blend=members[k],head_swap=ts[k],**CFG) for k in bundles};full=pred['fold']
    np.testing.assert_array_equal(lane_pred[m.ordinary],full[m.ordinary])
    reads['clock_mix12_step']=score(ids,lane_pred,full,y);reads['full_vs_stack7']=score(ids,parentpred['fold'],full,y)
    print('clock mix12',reads['clock_mix12_step']['G'],reads['clock_mix12_step']['interval'],flush=True)
    B=compose(b,route='R4mk',noplan='v3cal',practice='selected');v47=compose(b,route='R4',form='HA3',extension=False,noplan='none',c33=False)
    reads['full_vs_B']=score(ids,B,full,y);reads['full_vs_v47']=score(ids,v47,full,y)
    maskedold,maskednew=matched_shadow(b,m,h,parents['fold'],ts['fold'],dst)
    np.testing.assert_array_equal(maskedold,held7.masked_stack7)
    reads['matched_shadow_vs_stack7']=score(ids,maskedold,maskednew,y)
    scopes={k:s7.scope_check(bundles[k],members[k],heads[k],ts[k],parentpred[k],pred[k]) for k in bundles}
    assert scopes['ranking']['extension_rows_exact']==17017 and scopes['ranking']['pair_extension_exact']==16748
    oldpair=load_member(EXP/'inputs/stack3-ext-009/pair.json',sha(EXP/'inputs/stack3-ext-009/pair.json'),bundles['ranking'],'ranking')
    excluded=oldpair.scope & oldpair.extension & ~members['ranking'].fallback.scope;assert excluded.sum()==60
    np.testing.assert_array_equal(pred['ranking'][excluded],parentpred['ranking'][excluded]);scopes['ranking']['missing_AOBT3_pair_exclusions_exact']=60
    pd.DataFrame({ID:ids,'stack7':parentpred['fold'],'lane_c':lane_pred,'stack8':full,'masked_stack7':maskedold,'masked_stack8':maskednew}).to_parquet(EXP/f'results/stack8-held-023-{key}.parquet',index=False)
    risk={k:sequence_risk(bundles[k],members[k],ts[k]) for k in bundles}
    for p in dst.iterdir():
        if p.is_file():pin(p)
    for p,hsh in js(EXP/'results/stack7-read-022-030.json')['source_sha256'].items():
        if Path(p).suffix in ('.parquet','.json','.npz','.pt'):pin(ROOT/p,hsh)
    oldfiles=js(EXP/'results/stack8-before-023.json')['files']
    for p,hsh in oldfiles.items():assert sha(ROOT/p)==hsh,p
    limits='Recipes and class/form choices used the same held rows; conditional day interval excludes seed noise, source/form selection and2026 transfer. Opposing sequence weights depend on similar-vector differences. Parent C73 upper600 scenario remains, corrected centre536-576. No extension credit.'
    assert code_pins()==code
    write(result,{'source_rows':[899,901],'source_key':key,'source':choice,'code_sha256':code,'source_sha256':PINS,'parent_byte_exact':True,'parent_refit_exact':True,'source_parity':parity,'receipts':receipts,
                  'reads':reads,'scope':scopes,'sequence_difference_risk':risk,'prediction_digests':{k:vector_digest(bundles[k][ID],pred[k]) for k in bundles},'old_artifacts_unchanged':len(oldfiles),'limits':limits,'spec_sha256':SPEC_SHA,'started_at_utc':start,'read_at_utc':now()})
    low,high=(30,180) if key=='033b' else (60,320)
    r=reads['matched_shadow_vs_stack7'];assert low<=reads['full_vs_stack7']['G']<=high and reads['clock_mix12_step']['G']>0 and r['G']>0,'Outside written prior or nonpositive step: report desk before freeze.'
    p=js(PARENT_FORECAST)['central'];assert p==222.87231371089865
    freeze={'central':float(np.sqrt(p*p-1.24*r['G'])),'conditional_range':[float(np.sqrt(p*p-1.24*r['interval']['hi95'])),float(np.sqrt(p*p-1.24*r['interval']['lo95']))],
            'parent':p,'parent_forecast':record(PARENT_FORECAST),'parent_sha256':PARENT,'source_key':key,'increment_G':r['G'],'increment_interval':r['interval'],'unmasked_sensitivity':float(np.sqrt(p*p-1.24*reads['full_vs_stack7']['G'])),
            'ratio':1.24,'ratio_basis':'ledger899 v44/v45 versus v43; baseline901 v47','incremental_extension_credit':0,'limits':limits,'spec_sha256':SPEC_SHA,'read_sha256':sha(result),'frozen_at_utc':now(),'candidate_exists':False}
    write(forecast,freeze)
    argv=[str(ROOT/'.venv/bin/python'),str(EXP/'code/assembly.py'),'--bundle',str(EXP/'inputs/stack5-016/bundle-ranking.parquet'),'--bundle-sha256',sha(EXP/'inputs/stack5-016/bundle-ranking.parquet'),'--route','R4mk','--form','HB3u','--noplan','v3cal','--practice','selected_c73','--blend-manifest',str(OLD),'--blend-manifest-sha256',JOINT,'--head-swap-manifest',str(HEAD),'--head-swap-manifest-sha256',sha(HEAD),'--stack8-manifest',str(dst/'stack8.json'),'--stack8-manifest-sha256',sha(dst/'stack8.json'),'--pair-extension','proxy-2','--c55','--out',str(CANDIDATE)]
    write(dst/'build-command.json',{'argv':argv,'key':key,'code_sha256':code,'spec_sha256':SPEC_SHA,'forecast_sha256':sha(forecast),'read_sha256':sha(result),'written_at_utc':now()})
    print('FROZEN',key,freeze['central'],freeze['conditional_range'],'matched G',r['G'],flush=True)


def check_build_ready(out):
    assert Path(out).resolve()==CANDIDATE and not CANDIDATE.exists()
    decision=choose();key=decision['source']['key'];dst,result,forecast=paths(key);r=js(result);f=js(forecast);cmd=js(dst/'build-command.json')
    for k in ['key','pack','vector','review']:assert decision['source'][k]==r['source'][k]
    assert sha(SPEC)==SPEC_SHA==r['spec_sha256']==cmd['spec_sha256']
    assert code_pins()==r['code_sha256']==cmd['code_sha256']
    assert sha(result)==cmd['read_sha256'] and sha(forecast)==cmd['forecast_sha256']
    for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert not f['candidate_exists'] and f['frozen_at_utc']<now()
    for p in [SPEC,forecast,checked(decision['source']['pack']),checked(decision['source']['review']),dst/'SOURCE-PASS.md',MLP/'results/pack-012/REVIEW-012.md',dst/'MLP-PASS.md',DECISION]:s7.committed(p)
    return decision


def build():
    decision=check_build_ready(CANDIDATE);key=decision['source']['key'];dst,result,forecast=paths(key);cmd=js(dst/'build-command.json');write(COMMAND,cmd)
    commit=s7.committed(SPEC);subprocess.run(cmd['argv'],cwd=ROOT,check=True)
    q=pd.read_parquet(CANDIDATE);assert len(q)==344841 and q[ID].is_unique and list(q.columns)==[ID,COL]
    assert q[ID].dtype==q[COL].dtype==np.dtype('float64')
    np.testing.assert_array_equal(q[ID],pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID]);assert np.isfinite(q[COL]).all() and q[COL].min()>=60
    assert vector_digest(q[ID],q[COL])==js(result)['prediction_digests']['ranking']
    write(EXP/'results/stack8-build-023.json',{'candidate':record(CANDIDATE),'source_decision':record(DECISION),'read':record(result),'forecast':record(forecast),'manifest':record(dst/'stack8.json'),'commit_before_build':commit,'built_at_utc':now(),'review':'desk D123 pending'})
    print('BUILT',sha(CANDIDATE),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-read','observe','choose','build']);ap.add_argument('--source',choices=['033b','037']);a=ap.parse_args()
    if a.stage=='prepare-read':assert a.source;prepare(a.source)
    elif a.stage=='observe':print(json.dumps(observe(),indent=2))
    elif a.stage=='choose':print(json.dumps(choose(),indent=2))
    else:build()
