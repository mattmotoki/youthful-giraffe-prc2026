"""SPEC-016 saved-vector stack5 read, frozen forecast, and common assembly build."""
import argparse
from dataclasses import replace
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from assembly import ID,COL,compose,clip
from diverse_blend import load_member,vector_digest
from joint_blend import JointBlend,G_NAMES
from read_stacks import GS
from stand_floor import apply_floor
from c73 import raw_frame,add_key

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
DIV=ROOT/'experiments/exp-170-diverse/results'
GROOT=ROOT.parent/'prc-data-challenge-2026-exp150/experiments/exp-150-model-a/sequence-g'
OUT=EXP/'inputs/stack5-016';RESULT=EXP/'results/stack5-read-016.json'
CANDIDATE=EXP/'exp169-stack5-Gmixed-bag6r6-C73-ext-aobt-c55.parquet'
CFG=dict(route='R4mk',noplan='v3cal',practice='selected',pair_extension=True)
PINS={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()


def pin(p,expected=None):
    p=Path(p).resolve();h=sha(p)
    if expected:assert h==expected,(str(p),h,expected)
    PINS[str(p)]=h;return p


def copy_source(p,dst,expected=None):
    p=pin(p,expected);shutil.copyfile(p,dst);pin(dst,sha(p));return dst


def write(p,obj):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,indent=2)+'\n')


def code_pins():
    return {n:sha(EXP/'code'/n) for n in ['stack5.py','c73.py','assembly.py','joint_blend.py','diverse_blend.py','pair_blend.py','stand_floor.py']}


def aligned(p,ids,col):
    q=pd.read_parquet(p).set_index(ID);assert q.index.is_unique and set(q.index).issubset(set(ids))
    return q[col].reindex(ids).to_numpy(float)


def fit(b,values,on,pre,B,y,w):
    scale=np.where(b.practice,1-b.practice_weight,1.)
    X=(values-pre[:,None])*scale[:,None];folds={};pooled={}
    for cls,names in [('ordinary',G_NAMES),('clock',G_NAMES[:-1])]:
        folds[cls]={}
        for month in [1,7,None]:
            mask=on & b.cls.eq(cls).to_numpy()
            if month:mask &= b.month.eq(month).to_numpy()
            A=X[mask,:len(names)]*np.sqrt(w[mask,None]);r=(y-B)[mask]*np.sqrt(w[mask])
            beta=np.linalg.lstsq(A,r,rcond=None)[0]
            rec={'method':'practice_scaled_wls','columns':names,'coefficients':beta.tolist(),'trained_on_months':[month] if month else [1,7],
                 'fit_rows':int(mask.sum()),'xtwx':(A.T@A).tolist(),'xtwr':(A.T@r).tolist()}
            if month:folds[cls][str(8-month)]=rec
            else:pooled[cls]=rec
    return folds,pooled


def member(b,frame,pre,fallback,folds,pooled,split):
    on=frame.member_scope.to_numpy(bool);w=np.zeros((len(b),4))
    for cls in ['ordinary','clock']:
        for month in [1,7]:
            rec=folds[cls][str(month)] if split=='fold' else pooled[cls]
            beta=np.array(rec['coefficients']);beta=np.r_[beta,0.] if cls=='clock' else beta
            w[on & b.cls.eq(cls).to_numpy() & b.month.eq(month).to_numpy()]=beta
    return JointBlend(b[ID].to_numpy('int64'),frame[G_NAMES].to_numpy(float),w,on,b.cls.eq('ordinary').to_numpy(),vector_digest(b[ID].to_numpy('int64'),pre),fallback)


def score(ids,B,z,y):
    s=GS.score(ids,B,z,y=y,require_complete=True)
    return {k:s[k] for k in ['G','interval','views','G_without_top_days']}


def control_receipts(pack):
    checks={}
    for split,folder,filename,months in [('held','row-bag6-021-fit','jjrefit.parquet',[2,3,4,5,6,8,9,10,11,12]),('annual','row-bag6-final-021-fit','ranking2026.parquet',list(range(1,13)))]:
        r=js(pin(DIV/folder/'FIT.json',pack['records']['control'][split+'_fit_sha256']))
        q=pd.read_parquet(pin(DIV/folder/filename,r['files'][filename]));assert q[ID].is_unique
        assert r['train_months']==months and r['n_seeds']==6 and r['args']['seed']==list(range(17001,17007))
        parts=[]
        for p in r['parents']:
            assert p['weight']==1/6 and p['fit']['train_months']==months and p['fit']['args']['seed']==p['seed']
            assert p['fit']['args']['mode']=='row' and p['fit']['args']['epochs']==12
            if p['seed']==17001:
                if split=='held':
                    path=pin(DIV/'vectors/row-001/jjrefit.parquet',p['vector_sha256']);v=aligned(path,q[ID],'pred_seq170')
                    pin(DIV/'fit-records/row-001/FIT.json',p['fit_sha256'])
                else:
                    # The old reviewed pack preserves the annual seed001 member column.
                    oldpack=js(pin(DIV/'pack-bag6-pair-020/PACK.json'))
                    path=pin(DIV/'pack-bag6-pair-020/members-final.parquet',oldpack['vector_sha256'])
                    v=aligned(path,q[ID],'pred_row170')
                    pin(DIV/'row-final-005-FIT.json',p['fit_sha256'])
                    assert oldpack['records']['control']['annual_raw_vector_sha256']==p['vector_sha256']
            else:
                loc=DIV/(Path(p['path']).name+'-fit')
                assert js(pin(loc/'FIT.json',p['fit_sha256']))==p['fit']
                pin(loc/'model.pt',p['model_sha256'])
                v=aligned(pin(loc/filename,p['vector_sha256']),q[ID],'pred_seq170')
            parts.append(v)
        np.testing.assert_array_equal(np.mean(np.stack(parts),axis=0),q.pred_seq170)
        checks[split]={'rows':len(q),'seeds':list(range(17001,17007)),'months':months,'mean_exact':True,
                       'seed001_annual_source':'reviewed pack020 column' if split=='annual' else 'raw held vector'}
    return checks


def prepare():
    assert not OUT.exists() and not CANDIDATE.exists();OUT.mkdir()
    code=code_pins();pin(EXP/'SPEC-016.md');pin(ROOT/'docs/LEDGER.md')
    parent_manifest=pin(EXP/'inputs/stack4-014/joint.json','5b204d7c867611b02de7abb5e40668146fe79ede7062050393f7d74960cca914')
    parent_file=pin(EXP/'exp169-stack4-mlp-bag6-ext-aobt-c55.parquet','4c0d4dc131596d173fe08d2c68c1a55fd3214dbea7948173c6d89906f0aa93dd')
    pin(EXP/'results/stack4-read-014.json');parent_forecast=js(pin(EXP/'results/stack4-forecast-frozen-014.json'))
    pack=js(pin(DIV/'pack-bag6r6-pair-021/PACK.json','a29c99a9f3424b88e9843be8245ce16f1ffa5a9a0550ae3d0dc62af6cdf0303a'))
    annpath=pin(DIV/'pack-bag6r6-pair-021/members-final.parquet',pack['vector_sha256'])
    receipts=control_receipts(pack)
    copy_source(DIV.parent/'REVIEW-BAG6R6-021.md',OUT/'REVIEW-BAG6R6-021.md')
    gm=js(copy_source(GROOT/'results/ANNUAL-MEMBER-032.json',OUT/'G-MEMBER.json'))
    gr=js(copy_source(GROOT/'REVIEW-ANNUAL-OUTPUT-032.json',OUT/'G-REVIEW.json'))
    assert gr['status']=='PASS' and gr['member_sha256']==sha(OUT/'G-MEMBER.json')
    copy_source(GROOT/'results/ANNUAL-OUTPUT-AUDIT-032.json',OUT/'G-AUDIT.json',gr['audit_sha256'])
    for split,src,h in [('fold',gm['held']['path'],gm['held']['sha256']),('ranking',gm['annual']['ranking_path'],gm['annual']['ranking_sha256'])]:
        copy_source(src,OUT/f'G-{split}.parquet',h)
        fitpath=Path(src).parent/'FIT.json';expected=gm['held']['FIT_sha256'] if split=='fold' else gm['annual']['FIT_sha256']
        copy_source(fitpath,OUT/f'G-{split}-FIT.json',expected)
    assert gm['held']['train_months']==[2,3,4,5,6,8,9,10,11,12] and gm['annual']['train_months']==list(range(1,13))
    pin(gm['annual']['model_path'],gm['annual']['model_sha256'])
    source=js(copy_source(GROOT/'results/stack4-joint-read-001/READ.json',OUT/'G-SOURCE-READ.json'))
    sourcevec=copy_source(GROOT/'results/stack4-joint-read-001/SERVED.parquet',OUT/'G-SOURCE-SERVED.parquet',source['saved_predictions_sha256'])
    for path,name in [(ROOT/'experiments/exp-171-rules/LIRF-STEP2-UNIT2.md','C73-SOURCE.md'),(ROOT/'experiments/exp-171-rules/code/unit2_g_read.py','C73-source-code.txt'),(ROOT/'experiments/exp-165-levers/checks/c73_route_practice.out','C73-BOUND.txt')]:copy_source(path,OUT/name)
    raw=raw_frame(pin);bundles={};frames={};parents={};pres={};c73={};fold_detail=None
    for split in ['fold','ranking']:
        b=pd.read_parquet(pin(EXP/f'inputs/stack3-007/bundle-{split}.parquet'))
        b,d,rates,cs=add_key(b,raw,split);bundles[split]=b;c73[split]=cs
        rates.to_csv(OUT/f'C73-rates-{split}.csv');d.to_parquet(OUT/f'C73-rows-{split}.parquet')
        if split=='fold':fold_detail=d
        par=load_member(parent_manifest,sha(parent_manifest),b,split);parents[split]=par
        pre=compose(b,route='R4mk',noplan='none',practice='none',c33=False);pres[split]=pre
        q=b[[ID,'month','cls']].copy();q['member_scope']=par.scope;q['g']=aligned(OUT/f'G-{split}.parquet',b[ID],'pred_Gmixed')
        q['mlp']=par.values[:,0];q['sequence']=par.values[:,1]
        if split=='fold':q['control']=aligned(pin(DIV/'row-bag6-021-fit/jjrefit.parquet',pack['records']['control']['held_vector_sha256']),b[ID],'pred_seq170')
        else:
            annual=pd.read_parquet(annpath);np.testing.assert_array_equal(annual[ID],b[ID]);np.testing.assert_array_equal(annual.member_scope,q.member_scope)
            np.testing.assert_array_equal(annual.pred_seq170,q.sequence);q['control']=annual.pred_row170.to_numpy()
            np.testing.assert_array_equal(q.control,aligned(DIV/'row-bag6-final-021-fit/ranking2026.parquet',b[ID],'pred_seq170'))
        frames[split]=q
    b=bundles['fold'];ids=b[ID].to_numpy('int64');pre=pres['fold'];on=parents['fold'].scope
    rows=GS.load_rows().set_index(ID).loc[ids];counts=rows.groupby(['month','airport','cls']).size();cells=GS.load_cells().set_index(['month','airport','cls']).N
    w=np.array([cells.get((m,a,z),0)/GS.N_RANKING/counts[m,a,z] if counts[m,a,z]>=GS.MIN_SUPPORT else 0. for m,a,z in zip(rows.month,rows.airport,rows.cls)])
    np.testing.assert_array_equal(on,(b.F.to_numpy()<2400)&rows.aobt.to_numpy(bool)&b.cls.ne('noac').to_numpy()&(w>0)&~b.noplan.to_numpy(bool))
    B=compose(b,route='R4mk',noplan='v3cal',practice='selected');y=GS.labels(ids)
    # Independent G source parity on the old single-control stack4.
    oldframe=frames['fold'].copy();oldframe['control']=parents['fold'].values[:,-1]
    oldfold,oldpool=fit(b,oldframe[G_NAMES].to_numpy(),on,pre,B,y,w)
    for cls in ['ordinary','clock']:
        for mon in ['1','7']:
            src=source['weights'][cls][mon]['coefficients'];want=[src[k] for k in (['mixed','mlp','runway','control'] if cls=='ordinary' else ['mixed','mlp','runway'])]
            np.testing.assert_allclose(oldfold[cls][mon]['coefficients'],want,rtol=0,atol=1e-10)
    oldg=member(b,oldframe,pre,parents['fold'].fallback,oldfold,oldpool,'fold')
    oldserved=compose(b,blend=oldg,**CFG);oldfull=compose(b,blend=oldg,**CFG,c55=True)
    src=pd.read_parquet(sourcevec).set_index(ID).loc[ids]
    np.testing.assert_allclose(oldserved,src.joint,rtol=0,atol=1e-8);np.testing.assert_allclose(oldfull,src.joint_C55,rtol=0,atol=1e-8)
    assert abs(float(w@((y-B)**2-(y-oldserved)**2))-source['scores']['joint_vs_B']['G'])<1e-7
    # Pair-only parity: pack021 coefficients have apply-month keys, and clock=.38.
    weights=np.zeros((len(b),3));values=np.column_stack([np.zeros(len(b)),frames['fold'].sequence,frames['fold'].control])
    for cls in ['ordinary','clock']:
        for mon in [1,7]:
            use=on & b.cls.eq(cls).to_numpy() & b.month.eq(mon).to_numpy();rec=pack['weights'][cls][str(mon)]
            weights[use]=[0,rec['weights']['sequence'],rec['weights']['control']] if cls=='ordinary' else [0,rec['weight'],0]
    values[:,0]=pre
    pair=JointBlend(ids,values,weights,on,b.cls.eq('ordinary').to_numpy(),vector_digest(ids,pre),parents['fold'].fallback)
    pairpred=compose(b,blend=pair,**CFG)
    pairsource=pin(DIV/'paired-bag6r6-021-B/paired_ls.parquet')
    np.testing.assert_allclose(pairpred,aligned(pairsource,ids,'pred_seq170'),rtol=0,atol=1e-8)
    # Source C73 read has all qualifying routes and no guard; preserve it as parity only.
    broader=B.copy();d=fold_detail[fold_detail.source_scope];loc=pd.Index(ids).get_indexer(d.index)
    broader[loc]=d.point.to_numpy();sourcec73=score(ids,B,broader,y)
    assert abs(sourcec73['G']-244.7)<.1,sourcec73['G']
    # First new joint read: no weight search or changes after it.
    folds,pooled=fit(b,frames['fold'][G_NAMES].to_numpy(),on,pre,B,y,w)
    spec={'schema':5,'lane':'B+C+G','member':'Gmixed032 + MLP007/008 + six sequence and six control seeds; C73 separate',
          'weight_basis':'two_fold_2025','serving_point':'after_head','member_training_months':{'fold':[2,3,4,5,6,8,9,10,11,12],'ranking':list(range(1,13))},
          'fold_weights':folds,'ranking_weights':pooled,'fallback':{'path':'../stack3-aobt-011/pair.json','sha256':sha(EXP/'inputs/stack3-aobt-011/pair.json')},'vectors':{},
          'extension_rule':'old single-pair on16748 AOBT_3 rows only'}
    for split in ['fold','ranking']:
        frames[split].to_parquet(OUT/f'members-{split}.parquet',index=False)
        bundles[split].to_parquet(OUT/f'bundle-{split}.parquet',index=False)
        spec['vectors'][split]={'path':f'members-{split}.parquet','sha256':sha(OUT/f'members-{split}.parquet'),'base_digest':vector_digest(bundles[split][ID].to_numpy('int64'),pres[split])}
        pin(OUT/f'bundle-{split}.parquet');pin(OUT/f'members-{split}.parquet')
    write(OUT/'joint.json',spec);pin(OUT/'joint.json')
    new=load_member(OUT/'joint.json',sha(OUT/'joint.json'),b,'fold')
    base4=compose(b,blend=parents['fold'],**CFG,c55=True)
    members=compose(b,blend=new,**CFG,c55=True);full=compose(b,blend=new,**{**CFG,'practice':'selected_c73'},c55=True)
    c73B=compose(b,route='R4mk',noplan='v3cal',practice='selected_c73')
    mask=b.airport.eq('LIRF').to_numpy()&b.month.eq(7).to_numpy()&b.cls.eq('ordinary').to_numpy()
    mp=pin(DIV/'masked-bag6-LIRF-020/jjrefit.parquet','792c9f3236370a6970372bbbd76bc2650c1f05b11cb8f6d771c47b21d37cade7')
    mv=aligned(mp,ids,'pred_seq170');nv=new.values.copy();nv[mask,2]=mv[mask];ov=parents['fold'].values.copy();ov[mask,1]=mv[mask]
    maskednew=compose(b,blend=replace(new,values=nv),**CFG,c55=True);maskedold=compose(b,blend=replace(parents['fold'],values=ov),**CFG,c55=True)
    v47=compose(b,route='R4',form='HA3',extension=False,noplan='none',c33=False)
    reads={'C73_source_all_routes_on_B':sourcec73}
    for name,base,z in [('G_source_on_B',B,oldserved),('pack021_pair_on_B',B,pairpred),('C73_on_B',B,c73B),('members_on_B',B,members),('member_step_on_stack4',base4,members),('masked_member_step_on_stack4',maskedold,maskednew),('C73_on_members',members,full),('full_on_B',B,full),('full_on_v47',v47,full),('full_step_on_stack4',base4,full)]:
        reads[name]=score(ids,base,z,y);print(name,reads[name]['G'],reads[name]['interval'],flush=True)
    assert reads['C73_on_B']['G']>=100,'C73 failed the declared gate; desk decision needed.'
    assert reads['member_step_on_stack4']['G']>0,'Joint member step failed; desk decision needed.'
    annualb=bundles['ranking'];annual=load_member(OUT/'joint.json',sha(OUT/'joint.json'),annualb,'ranking')
    oldranking=compose(annualb,blend=parents['ranking'],**CFG,c55=True);np.testing.assert_array_equal(oldranking,pd.read_parquet(parent_file)[COL])
    annualmembers=compose(annualb,blend=annual,**CFG,c55=True);annualfull=compose(annualb,blend=annual,**{**CFG,'practice':'selected_c73'},c55=True)
    c73['ranking']['moved_rows']=int((annualfull!=annualmembers).sum());c73['ranking']['squared_move']=float(np.mean((annualfull-annualmembers)**2))
    c73['ranking']['guarded_distance_to_point']=float(np.nansum((annualmembers[annualb.c73]-annualb.c73_point[annualb.c73])**2)/len(annualb))
    for split,z in [('fold',full),('ranking',annualfull)]:
        bb=bundles[split];c73[split]['overlap']={k:int((bb.c73 & bb[k]).sum()) for k in ['noplan','bet','practice','rule','extension']}
        c73[split]['overlap']['member']=int((bb.c73.to_numpy()&frames[split].member_scope.to_numpy()).sum())
    pd.DataFrame({ID:ids,'B':B,'stack4':base4,'members':members,'stack5':full,'masked_stack4':maskedold,'masked_members':maskednew,'C73_on_B':c73B}).to_parquet(EXP/'results/stack5-held-016.parquet',index=False)
    step=reads['masked_member_step_on_stack4'];p=parent_forecast['central']
    forecast={'central':float(np.sqrt(p*p-1.24*step['G']-600)),
      'conditional_range':[float(np.sqrt(p*p-1.24*step['interval']['hi95']-650)),float(np.sqrt(p*p-1.24*step['interval']['lo95']-550))],
      'C73_zero_scenario':float(np.sqrt(p*p-1.24*step['G'])),
      'unmasked_sensitivity':float(np.sqrt(p*p-1.24*reads['member_step_on_stack4']['G']-600)),
      'parent':p,'ratio':1.24,'ratio_basis':'ledger899 v44/v45 vs v43; baseline901 v47','member_delta_G':step['G'],'C73_ranking_MSE_scenario':600,
      'rule':'SPEC016; sequence-only matched LIRF July ordinary mask, weights fixed; inherit parent C57/C55, no duplicate credit',
      'limits':'G/control have no masked vector; conditional development interval fixes parent; route regime and annual transfer uncertainty remain',
      'frozen_at_utc':now(),'candidate_exists':CANDIDATE.exists()}
    assert not CANDIDATE.exists();write(EXP/'results/stack5-forecast-frozen-016.json',forecast)
    for p,h in PINS.items():assert sha(p)==h,p
    assert code_pins()==code
    rec={'source_rows':[899,901],'code_sha256':code,'source_sha256':PINS,'receipt_checks':receipts,'reads':reads,'c73':c73,'forecast':forecast,'weights':{'fold':folds,'ranking':pooled},
         'parity':{'G_source_weights_and_served':True,'pack021_pair_exact':True,'C73_source_G':sourcec73['G'],'stack4_ranking_exact':True},'prepared_at_utc':now()}
    write(RESULT,rec);print('FORECAST',forecast,flush=True)


def build():
    r=js(RESULT);assert not CANDIDATE.exists() and code_pins()==r['code_sha256']
    for p,h in r['source_sha256'].items():assert sha(p)==h,p
    manifest=OUT/'joint.json';bundle=OUT/'bundle-ranking.parquet'
    argv=[str(ROOT/'.venv/bin/python'),str(EXP/'code/assembly.py'),'--bundle',str(bundle),'--bundle-sha256',sha(bundle),'--route','R4mk','--noplan','v3cal','--practice','selected_c73','--blend-manifest',str(manifest),'--blend-manifest-sha256',sha(manifest),'--pair-extension','proxy-2','--c55','--out',str(CANDIDATE)]
    write(EXP/'results/stack5-build-command-016.json',{'argv':argv,'code_sha256':code_pins(),'forecast_sha256':sha(EXP/'results/stack5-forecast-frozen-016.json'),'spec_sha256':sha(EXP/'SPEC-016.md')})
    subprocess.run(argv,cwd=ROOT,check=True)
    b=pd.read_parquet(bundle);m=load_member(manifest,sha(manifest),b,'ranking');z=pd.read_parquet(CANDIDATE)
    actual=compose(b,blend=m,**{**CFG,'practice':'selected_c73'},c55=True)
    np.testing.assert_array_equal(actual,z[COL]);np.testing.assert_array_equal(z[ID],pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID])
    old=pd.read_parquet(EXP/'exp169-stack4-mlp-bag6-ext-aobt-c55.parquet')[COL].to_numpy()
    off=~(m.scope|b.c73.to_numpy());np.testing.assert_array_equal(actual[off],old[off])
    extension=m.fallback.scope&m.fallback.extension;np.testing.assert_array_equal(actual[extension],old[extension])
    assert int(extension.sum())==16748
    assert list(z.columns)==[ID,COL] and z[ID].dtype==z[COL].dtype==np.dtype('float64') and z[ID].is_unique and len(z)==344841
    pre=compose(b,blend=m,**{**CFG,'practice':'selected_c73'},c33=False);bracket=clip(pre,b.lo.to_numpy(),b.hi.to_numpy())
    np.testing.assert_array_equal(apply_floor(bracket,b.stand_floor.to_numpy(),b.airport.to_numpy(),b.lo.to_numpy(),b.hi.to_numpy()),actual)
    members=compose(b,blend=m,**CFG,c55=True)
    np.testing.assert_array_equal(actual[b.c73_U],members[b.c73_U])
    checks={'rows':len(z),'changed_from_stack4':int((actual!=old).sum()),'C33_clips':int((pre!=bracket).sum()),'C55_clips':int((bracket!=actual).sum()),'C73_rows':int(b.c73.sum()),'C73_moved':int((actual!=members).sum()),'extension_exact_rows':int(extension.sum()),'outside_scope_exact':True,'U_rows_keep_member':True,'C39_overlap':int((b.c73&b.practice).sum()),'bet_rows_keep_parent':bool(np.array_equal(actual[b.bet],old[b.bet]))}
    assert checks['bet_rows_keep_parent']
    write(EXP/'results/stack5-build-016.json',{'candidate':str(CANDIDATE.relative_to(ROOT)),'sha256':sha(CANDIDATE),'manifest_sha256':sha(manifest),'forecast_sha256':sha(EXP/'results/stack5-forecast-frozen-016.json'),'read_sha256':sha(RESULT),'checks':checks,'built_at_utc':now(),'review':'pending'})
    print('BUILT',sha(CANDIDATE),checks,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','build']);{'prepare':prepare,'build':build}[ap.parse_args().stage]()
