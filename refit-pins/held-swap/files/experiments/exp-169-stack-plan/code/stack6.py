"""SPEC-021 saved-vector head swap, paired read, forecast freeze, common build."""
import argparse
from dataclasses import replace
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from assembly import ID, COL, compose, clip, head_blend
from diverse_blend import load_member, vector_digest
from head_swap import load_head_swap, aligned_head, fingerprint, _rebase, PARENT, PACK, SPEC, JOINT
from read_stacks import GS
from stand_floor import apply_floor

def id_digest(value):  # row IDs and callsigns stay out of the public source
    return hashlib.sha256(np.int64(value).tobytes() if not isinstance(value, str) else value.encode()).hexdigest()

EXP = Path(__file__).resolve().parents[1]; ROOT = EXP.parents[1]
OUT = EXP/'inputs/stack6-021'
RESULT = EXP/'results/stack6-read-021.json'
FORECAST = EXP/'results/stack6-forecast-frozen-021.json'
COMMAND = EXP/'results/stack6-build-command-021.json'
CANDIDATE = EXP/'exp169-stack6-HB3u-Gmixed-bag6r6-C73-ext-aobt-c55.parquet'
PARENT_FILE = EXP/'exp169-stack5-Gmixed-bag6r6-C73-ext-aobt-c55.parquet'
CFG = dict(route='R4mk',noplan='v3cal',practice='selected_c73',c55=True,pair_extension=True)
CODE = ['assembly.py','head_swap.py','stack6.py','joint_blend.py','pair_blend.py','diverse_blend.py','stand_floor.py','read_stacks.py']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text())
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, obj):
    assert not path.exists(), f'Write-once file exists: {path}'
    path.write_text(json.dumps(obj,indent=2)+'\n')


def record(path):
    p=Path(path).resolve();return {'path':str(p.relative_to(ROOT)), 'sha256':sha(p)}


def code_pins():return {name:sha(EXP/'code'/name) for name in CODE}


def committed(path):
    rel=str(Path(path).relative_to(ROOT))
    body=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT,stderr=subprocess.PIPE)
    assert hashlib.sha256(body).hexdigest()==sha(path), f'Commit required before build: {rel}'
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def check_build_ready(out):
    assert Path(out).resolve()==CANDIDATE and not CANDIDATE.exists()
    r=js(RESULT);f=js(FORECAST);cmd=js(COMMAND)
    assert code_pins()==r['code_sha256']==cmd['code_sha256']
    assert sha(EXP/'SPEC-021.md')==SPEC==cmd['spec_sha256']
    assert sha(FORECAST)==cmd['forecast_sha256'] and sha(RESULT)==cmd['read_sha256']
    assert f['candidate_exists'] is False and f['frozen_at_utc'] < now()
    for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
    committed(EXP/'SPEC-021.md');committed(FORECAST)


def original_formula(member, base):
    """Independent equation for primary and pair-only rows, with the serving floors."""
    z=base.copy();fb=member.fallback
    for mask,values,weights in [(fb.scope & fb.extension, np.column_stack([fb.sequence,fb.control]), np.column_stack([fb.sequence_weight,fb.control_weight])),
                               (member.scope,member.values,member.weights)]:
        delta=np.zeros(len(base))
        for k in range(values.shape[1]):
            use=mask & (weights[:,k]!=0)
            delta[use]+=weights[use,k]*(values[use,k]-base[use])
        z[mask]=np.maximum(60.,base[mask]+delta[mask])
    return z


def interface_checks(b,m,s):
    ids=b[ID].to_numpy(); checks=[]
    def rejects(name,fn):
        try:fn()
        except (AssertionError,ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Required rejection failed: '+name)
    for name,base in [('actual',s.new_base),('identity',s.old_base),('floor_case',np.full(len(ids),60.))]:
        z=_rebase(m,ids,s.old_base,base).apply(ids,base,enable_extension=True)
        np.testing.assert_array_equal(z,original_formula(m,base))
        if name=='identity':np.testing.assert_array_equal(z,m.apply(ids,s.old_base,enable_extension=True))
        if name=='floor_case':assert (z[m.scope]==60).any()
    rejects('old API rejects new base',lambda:m.apply(ids,s.new_base,enable_extension=True))
    rejects('wrong IDs',lambda:s.blend(m,ids[::-1],s.new_base,True))
    bad=m.weights.copy();bad[np.flatnonzero(m.scope)[0],0]+=.001
    rejects('changed coefficients',lambda:s.blend(replace(m,weights=bad),ids,s.new_base,True))
    rejects('wrong old base hash',lambda:_rebase(replace(m,base_digest='0'*64),ids,s.old_base,s.new_base))
    bb=b.copy();bb.loc[bb.index[0],'R4mk']+=1
    rejects('changed route or unsupported row',lambda:s.bases(bb,bb.R4mk.to_numpy(float)))
    head=s.head.copy();head[np.flatnonzero(s.scope)[0]]=np.nan
    rejects('missing active head',lambda:replace(s,head=head).bases(b,b.R4mk.to_numpy(float)))
    rejects('wrong split',lambda:load_head_swap(OUT/'head-swap.json',sha(OUT/'head-swap.json'),b,m,'ranking'))
    # Altered receipt fixtures are durable and contain no score or prediction.
    for name,patch in [('missing approval',{'desk_approval':{'path':'experiments/exp-169-stack-plan/inputs/stack6-021/NO-APPROVAL','sha256':'0'*64}}),
                       ('missing source PASS',{'source_review':{'path':'experiments/exp-169-stack-plan/inputs/stack6-021/NO-REVIEW','sha256':'0'*64}})]:
        rec=js(OUT/'head-swap.json');rec.update(patch)
        p=OUT/('fixture-'+name.replace(' ','-')+'.json');write(p,rec)
        rejects(name,lambda p=p:load_head_swap(p,sha(p),b,m,'fold'))
    return {'rejections':checks,'identity_exact':True,'formula_exact':True,'floor_case_exercised':True}


def score(ids,base,z,y):
    return GS.score(ids,base,z,y=y,require_complete=True)


def source_parity(bundles,heads,y):
    hroot=ROOT/'experiments/exp-159-clock-head'
    pin={};reads={}
    for ext,suffix in [(False,''),(True,'-x')]:
        b=bundles['ranking'];mask=b.rule.to_numpy(bool)|(b.extension.to_numpy(bool) if ext else False)
        z=head_blend(b.F.to_numpy(float),heads['ranking'],mask,.5)
        source=hroot/f'candidates/exp159-fold-F-HB3u-w50{suffix}.parquet'
        want=pd.read_parquet(source)
        np.testing.assert_array_equal(b[ID],want[ID]);np.testing.assert_array_equal(z,want[COL])
        buf=io.BytesIO();pd.DataFrame({ID:b[ID],COL:z}).to_parquet(buf,index=False)
        assert hashlib.sha256(buf.getvalue()).hexdigest()==sha(source)
        pin[str(source.relative_to(ROOT))]=sha(source)
    b=bundles['fold'];ids=b[ID].to_numpy('int64')
    old=head_blend(b.F.to_numpy(float),b.HB1.to_numpy(float),b.rule.to_numpy(bool),.5)
    new=head_blend(b.F.to_numpy(float),heads['fold'],b.rule.to_numpy(bool),.5)
    r=score(ids,old,new,y);src=js(hroot/'results/pair-HB3u-w50-vs-HB1-w50.json')['pairs']['HB3u against HB1']
    for a,c in [(r['G'],src['G']),(r['interval']['lo95'],src['lo95']),(r['interval']['hi95'],src['hi95'])]:assert abs(a-c)<1e-8
    pin[str((hroot/'results/pair-HB3u-w50-vs-HB1-w50.json').relative_to(ROOT))]=sha(hroot/'results/pair-HB3u-w50-vs-HB1-w50.json')
    return {'entries_byte_exact':True,'raw_packed_value_parity':True,'source_G':r},pin


def scope_report(b,m,old,new,pre,bracket):
    d=new-old;head=(b.rule|b.extension).to_numpy(bool);ext=b.extension.to_numpy(bool)
    np.testing.assert_array_equal(old[~head],new[~head])
    masks={k:b[k].to_numpy(bool) for k in ['rule','extension','noplan','bet','practice','c73']}
    masks.update(member=m.scope,pair_extension=m.fallback.scope & m.fallback.extension,C33=pre!=bracket,C55=bracket!=new)
    groups={k:{'rows':int(mask.sum()),'head_overlap':int((mask&head).sum()),'moved':int(((d!=0)&mask).sum()),'mean_shift':float(np.mean(d[mask])) if mask.any() else 0.,'distance_all_rows':float(np.sum(d[mask]**2)/len(b))} for k,mask in masks.items()}
    cells=[]
    for (a,mon),loc in b.loc[ext].groupby(['airport','month']).groups.items():
        k=b.index.get_indexer(loc);cells.append({'airport':a,'month':int(mon),'rows':len(k),'mean_shift':float(np.mean(d[k])),'distance_all_rows':float(np.sum(d[k]**2)/len(b))})
    top=np.flatnonzero(ext);top=top[np.argsort(d[top]**2)[-5:][::-1]]
    risks=np.flatnonzero(b[ID].map(id_digest).eq('5f4a4a3ca4990c2ceb4354b92bc9759cb2912c57cbdf1309e4d3f9f2b20fce1b'))
    detail=lambda i:{'id':int(b[ID].iloc[i]),'airport':b.airport.iloc[i],'old':float(old[i]),'new':float(new[i]),'delta':float(d[i]),'distance_all_rows':float(d[i]**2/len(b)),'pair_on':bool(m.fallback.scope[i]&ext[i])}
    return {'changed':int((d!=0).sum()),'off_head_changed':0,'groups':groups,'extension_cells':cells,'extension_top5':[detail(i) for i in top],'risk_row':[detail(i) for i in risks]}


def prepare_read(a):
    assert not RESULT.exists() and not FORECAST.exists() and not CANDIDATE.exists()
    assert a.spec.resolve()==EXP/'SPEC-021.md' and sha(a.spec)==SPEC
    assert a.parent_manifest.resolve()==EXP/'inputs/stack5-016/joint.json' and sha(a.parent_manifest)==JOINT
    assert a.out_dir.resolve()==OUT and a.head_pack.resolve()==OUT/'PACK.json'
    assert a.source_review.resolve()==OUT/'HB3u-D123-PASS.log' and a.desk_approval.resolve()==OUT/'DESK-APPROVAL.json'
    assert sha(a.head_pack)==PACK and 'VERDICT: PASS' in a.source_review.read_text()
    approval=js(a.desk_approval);assert approval['approved'] and approval['spec_sha256']==SPEC and approval['source_review_sha256']==sha(a.source_review)
    assert sha(PARENT_FILE)==PARENT
    code=code_pins();bundles={};members={};heads={};splits={}
    for split,name in [('fold','jjrefit'),('ranking','jjrank')]:
        bpath=EXP/f'inputs/stack5-016/bundle-{split}.parquet';b=pd.read_parquet(bpath)
        m=load_member(a.parent_manifest,JOINT,b,split)
        head=aligned_head(a.head_pack,OUT/f'HB3u-{name}.json',OUT/f'raw-HB3u-{name}.parquet',OUT/f'raw-HB3u-{name}.json',b,split)
        mask=(b.rule|b.extension).to_numpy(bool);ids=b[ID].to_numpy('int64')
        old=head_blend(b.R4mk.to_numpy(float),b.HB1.to_numpy(float),mask,.5)
        new=head_blend(b.R4mk.to_numpy(float),head,mask,.5)
        assert vector_digest(ids,old)==m.base_digest
        splits[split]={'source_split':name,'bundle':record(bpath),'vector':record(OUT/f'HB3u-{name}.parquet'),'receipt':record(OUT/f'HB3u-{name}.json'),
                       'raw_vector':record(OUT/f'raw-HB3u-{name}.parquet'),'raw_receipt':record(OUT/f'raw-HB3u-{name}.json'),
                       'mask_digest':vector_digest(ids,mask),'old_base_digest':vector_digest(ids,old),'new_base_digest':vector_digest(ids,new),'member_fingerprint':fingerprint(m)}
        bundles[split]=b;members[split]=m;heads[split]=head
    manifest={'schema':1,'change':'HB1_to_HB3u','weight_policy':'inherit_stack5_exact','route':'R4mk','head_weight':.5,'extension':True,'normal_equations_basis':'original HB1 base; unchanged, no refit',
              'parent':record(PARENT_FILE),'spec':record(a.spec),'joint':record(a.parent_manifest),'pack':record(a.head_pack),
              'source_review':record(a.source_review),'desk_approval':record(a.desk_approval),'plan_review':record(EXP/'review/review-021-plan-astra.log'),'splits':splits,'written_at_utc':now()}
    write(OUT/'head-swap.json',manifest)
    swaps={s:load_head_swap(OUT/'head-swap.json',sha(OUT/'head-swap.json'),bundles[s],members[s],s) for s in bundles}
    # Parent parity is mandatory before a new score. Preserve the old candidate bytes.
    old_rank=compose(bundles['ranking'],blend=members['ranking'],**CFG)
    np.testing.assert_array_equal(old_rank,pd.read_parquet(PARENT_FILE)[COL])
    buf=io.BytesIO();pd.DataFrame({ID:bundles['ranking'][ID],COL:old_rank}).to_parquet(buf,index=False)
    assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT
    b=bundles['fold'];m=members['fold'];s=swaps['fold'];ids=b[ID].to_numpy('int64')
    checks=interface_checks(b,m,s)
    parent=compose(b,blend=m,**CFG)
    held5=pd.read_parquet(EXP/'results/stack5-held-016.parquet');np.testing.assert_array_equal(ids,held5[ID]);np.testing.assert_array_equal(parent,held5.stack5)
    y=GS.labels(ids);parity,pins=source_parity(bundles,heads,y)
    # The first new G read follows fixed source, parent and contract parity.
    reads={};stages={}
    for name,cfg in [('head',dict(noplan='none',practice='none',c33=False,c55=False)),
                     ('member',dict(noplan='none',practice='none',c33=False,c55=False)),
                     ('practice',dict(c33=False,c55=False)),('C33',dict(c55=False)),('full',{})]:
        member=None if name=='head' else m
        params={**CFG,**cfg,'pair_extension':member is not None}
        old=compose(b,blend=member,**params)
        new=compose(b,form='HB3u',head_swap=s,blend=member,**params)
        np.testing.assert_array_equal(old[~s.scope],new[~s.scope])
        reads[name]=score(ids,old,new,y);stages[name]=(old,new)
        print(name,reads[name]['G'],reads[name]['interval'],flush=True)
    full=stages['full'][1]
    B=compose(b,route='R4mk',noplan='v3cal',practice='selected')
    v47=compose(b,route='R4',form='HA3',extension=False,noplan='none',c33=False)
    reads['full_on_B']=score(ids,B,full,y);reads['full_on_v47']=score(ids,v47,full,y)
    assert -50<=reads['full']['G']<=200,'Outside prior: complete defect check and report desk before forecast.'
    scope={};predictions={}
    for split in ['fold','ranking']:
        bb=bundles[split];mm=members[split];ss=swaps[split]
        old=compose(bb,blend=mm,**CFG)
        pre=compose(bb,blend=mm,form='HB3u',head_swap=ss,**{**CFG,'c33':False,'c55':False})
        bracket=clip(pre,bb.lo.to_numpy(float),bb.hi.to_numpy(float))
        new=compose(bb,blend=mm,form='HB3u',head_swap=ss,**CFG)
        np.testing.assert_array_equal(new,apply_floor(bracket,bb.stand_floor.to_numpy(float),bb.airport.to_numpy(),bb.lo.to_numpy(float),bb.hi.to_numpy(float)))
        assert np.array_equal(clip(new,bb.lo.to_numpy(float),bb.hi.to_numpy(float)),new)
        np.testing.assert_array_equal(old[bb.bet],new[bb.bet]);np.testing.assert_array_equal(old[bb.noplan],new[bb.noplan])
        scope[split]=scope_report(bb,mm,old,new,pre,bracket);predictions[split]=vector_digest(bb[ID],new)
    assert scope['ranking']['groups']['extension']['rows']==17017
    assert scope['ranking']['groups']['pair_extension']['rows']==16748
    # Explicitly preserve the previously removed 60 missing-AOBT_3 pair rows.
    oldpair=load_member(EXP/'inputs/stack3-ext-009/pair.json',sha(EXP/'inputs/stack3-ext-009/pair.json'),bundles['ranking'],'ranking')
    removed=oldpair.scope & oldpair.extension & ~members['ranking'].fallback.scope
    assert removed.sum()==60
    bb=bundles['ranking'];ss=swaps['ranking'];mm=members['ranking']
    before=compose(bb,form='HB3u',head_swap=ss,blend=mm,**{**CFG,'noplan':'none','practice':'none','c33':False,'c55':False})
    np.testing.assert_array_equal(before[removed],ss.new_base[removed])
    scope['ranking']['missing_AOBT3_pair_exclusions_exact']=60
    held=pd.DataFrame({ID:ids,'stack5':parent,'stack6':full})
    for name,(_,z) in stages.items():held[name]=z
    held.to_parquet(EXP/'results/stack6-held-021.parquet',index=False)
    # Numeric freeze is before the ranking candidate write; no incremental extension credit.
    p=js(EXP/'results/stack5-forecast-frozen-016.json')['central'];r=reads['full'];ratio=1.24
    transfer=61.81082282563272/136.42476480185903
    forecast={'central':float(np.sqrt(p*p-ratio*r['G'])),
              'conditional_range':[float(np.sqrt(p*p-ratio*r['interval']['hi95'])),float(np.sqrt(p*p-ratio*r['interval']['lo95']))],
              'parent':p,'parent_sha256':PARENT,'parent_forecast_sha256':sha(EXP/'results/stack5-forecast-frozen-016.json'),
              'increment_G':r['G'],'increment_interval':r['interval'],'ratio':ratio,'ratio_basis':'ledger899 v44/v45 vs v43; baseline901 v47',
              'seven_month_transfer_sensitivity':{'source_season_mean_G':61.81082282563272,'source_development_G':136.42476480185903,'factor':transfer,'increment_scenario_G':r['G']*transfer,'forecast':float(np.sqrt(p*p-ratio*r['G']*transfer)),'measured_stack6_interval':False},
              'incremental_extension_credit':0.,'C73_credit_added':0.,'limits':'Conditional on frozen stack5 forecast, including old C73 upper600 scenario (corrected centre536-576). Interval covers2025 days only. Source was selected on Jan/July; seven-month proxy is sensitivity, not a stack6 read. No new extension gain credit.',
              'frozen_at_utc':now(),'candidate_exists':CANDIDATE.exists(),'spec_sha256':SPEC}
    write(FORECAST,forecast)
    # Pin immutable inputs and all unchanged original production dependencies.
    oldpins=js(EXP/'results/stack5-read-016.json')['source_sha256']
    for path,h in oldpins.items():
        pth=Path(path)
        # Living source prose can change; the stack5 bundle/manifest/member bytes cannot.
        if pth.suffix in ('.parquet','.json','.npz','.pt'):
            assert sha(pth)==h,path;pins[str(pth.relative_to(ROOT)) if pth.is_relative_to(ROOT) else str(pth)] = h
    # Absolute external paths remain absolute; Path(ROOT)/absolute resolves correctly.
    for key in list(pins):
        if key.startswith('../'):raise AssertionError(key)
    for pth in OUT.iterdir():
        if pth.is_file():pins[str(pth.relative_to(ROOT))]=sha(pth)
    pins[str(PARENT_FILE.relative_to(ROOT))]=PARENT
    pins[str((EXP/'results/stack5-held-016.parquet').relative_to(ROOT))]=sha(EXP/'results/stack5-held-016.parquet')
    pins[str((EXP/'results/stack5-forecast-frozen-016.json').relative_to(ROOT))]=sha(EXP/'results/stack5-forecast-frozen-016.json')
    assert code_pins()==code
    result={'source_rows':[899,901],'parent_sha256':PARENT,'code_sha256':code,'source_sha256':pins,'parity':parity,'contract_checks':checks,'reads':reads,'scope':scope,'forecast':forecast,'prediction_digests':predictions,'prepared_at_utc':now()}
    write(RESULT,result)
    argv=[str(ROOT/'.venv/bin/python'),str(EXP/'code/assembly.py'),'--bundle',str(EXP/'inputs/stack5-016/bundle-ranking.parquet'),'--bundle-sha256',splits['ranking']['bundle']['sha256'],'--route','R4mk','--form','HB3u','--noplan','v3cal','--practice','selected_c73','--blend-manifest',str(a.parent_manifest),'--blend-manifest-sha256',JOINT,'--head-swap-manifest',str(OUT/'head-swap.json'),'--head-swap-manifest-sha256',sha(OUT/'head-swap.json'),'--pair-extension','proxy-2','--c55','--out',str(CANDIDATE)]
    write(COMMAND,{'argv':argv,'code_sha256':code,'spec_sha256':SPEC,'forecast_sha256':sha(FORECAST),'read_sha256':sha(RESULT),'written_at_utc':now()})
    print('FROZEN',forecast,flush=True)


def build():
    check_build_ready(CANDIDATE)
    commits={'spec':committed(EXP/'SPEC-021.md'),'forecast':committed(FORECAST)}
    subprocess.run(js(COMMAND)['argv'],cwd=ROOT,check=True)
    q=pd.read_parquet(CANDIDATE);b=pd.read_parquet(EXP/'inputs/stack5-016/bundle-ranking.parquet')
    assert len(q)==344841 and list(q.columns)==[ID,COL] and q[ID].is_unique
    assert q[ID].dtype==q[COL].dtype==np.dtype('float64')
    np.testing.assert_array_equal(q[ID],pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID])
    assert np.isfinite(q[COL]).all() and q[COL].min()>=60
    assert vector_digest(q[ID],q[COL])==js(RESULT)['prediction_digests']['ranking']
    assert sha(PARENT_FILE)==PARENT
    write(EXP/'results/stack6-build-021.json',{'candidate':str(CANDIDATE.relative_to(ROOT)),'sha256':sha(CANDIDATE),'forecast_sha256':sha(FORECAST),'read_sha256':sha(RESULT),'manifest_sha256':sha(OUT/'head-swap.json'),'scope':js(RESULT)['scope']['ranking'],'spec_forecast_commits_before_build':commits,'built_at_utc':now(),'review':'desk D123 pending'})
    print('BUILT',sha(CANDIDATE),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='stage',required=True)
    prep=sub.add_parser('prepare-read')
    for name in ['spec','parent-manifest','head-pack','source-review','desk-approval','out-dir']:prep.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('build');a=ap.parse_args()
    if a.stage=='prepare-read':prepare_read(a)
    else:build()
