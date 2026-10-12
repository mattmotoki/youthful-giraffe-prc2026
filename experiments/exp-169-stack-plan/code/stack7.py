"""SPEC-022: one conditional lane C build and a clock-only e80 step."""
import argparse
from dataclasses import replace
import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from assembly import ID,COL,compose,clip
from diverse_blend import load_member,vector_digest
from head_swap import load_head_swap,fingerprint,JOINT,checked
from joint_blend import JointBlend
from lane_c_swap import NAMES,PARENT,make_transition,load_lane_c_swap,check_sequence,matrix_apply
from read_stacks import GS
from stand_floor import apply_floor

EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
DIV=ROOT/'experiments/exp-170-diverse';MLP=ROOT/'experiments/exp-164-double-down'
OUT=EXP/'inputs/stack7-022';SPEC=EXP/'SPEC-022.md'
SPEC_SHA='cc4cb23342b3d0505716b98f50556774b7b9c9fd2db631ceac8887465799a23e'
CANDIDATE=EXP/'exp169-stack7-HB3u-laneC-c73-c55.parquet'
PARENT_FILE=EXP/'exp169-stack6-HB3u-Gmixed-bag6r6-C73-ext-aobt-c55.parquet'
HEAD=EXP/'inputs/stack6-021/head-swap.json';OLD=EXP/'inputs/stack5-016/joint.json'
COMMAND=EXP/'results/stack7-build-command-022.json'
CFG=dict(route='R4mk',form='HB3u',noplan='v3cal',practice='selected_c73',c55=True,pair_extension=True)
CODE=['stack7.py','lane_c_swap.py','assembly.py','head_swap.py','joint_blend.py','pair_blend.py','diverse_blend.py','stand_floor.py','read_stacks.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
PINS={}


def write(p,obj):
    if p.exists():
        assert js(p)==obj,f'Write-once file differs: {p}'
        return
    p.write_text(json.dumps(obj,indent=2)+'\n')


def record(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def pin(p,h=None):
    p=Path(p).resolve();v=sha(p)
    if h:assert v==h,(p,v,h)
    PINS[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=v;return p


def stage(p,dst,h=None):
    p=pin(p,h)
    if dst.exists():assert sha(dst)==sha(p)
    else:shutil.copyfile(p,dst)
    return pin(dst,sha(p))


def code_pins():return {n:sha(EXP/'code'/n) for n in CODE}


def aligned(p,ids,col='pred_seq170'):
    q=pd.read_parquet(p).set_index(ID);assert q.index.is_unique
    return q[col].reindex(ids).to_numpy(float)


def source_choice():
    for key,folder in [('033b','pack-mixed18r6-pair-033b'),('030','pack-fast-last-bag12r6-pair-030')]:
        pack=DIV/'results'/folder/'PACK.json'
        if not pack.exists():continue
        r=js(pack);vector=pack.parent/'members-final.parquet'
        if not vector.exists() or sha(vector)!=r['vector_sha256']:continue
        candidates=sorted(DIV.glob('REVIEW*033*.md'))+sorted(pack.parent.glob('*REVIEW*.md')) if key=='033b' else [DIV/'REVIEW-BAG12-030.md']
        for review in candidates:
            text=review.read_text()
            if 'PASS' in text and sha(pack)[:8] in text and r['vector_sha256'][:8] in text:
                return {'key':key,'pack':record(pack),'vector':record(vector),'review':record(review),'checked_at_utc':now()}
    raise AssertionError('No exact reviewed lane C pack is present.')


def source_inputs(choice,dst,bundles,members):
    key=choice['key'];pack_path=stage(checked(choice['pack']),dst/'PACK.json');pack=js(pack_path)
    stage(checked(choice['review']),dst/'SOURCE-PASS.md');annual=pd.read_parquet(stage(checked(choice['vector']),dst/'pack-ranking.parquet'))
    np.testing.assert_array_equal(annual[ID],bundles['ranking'][ID])
    np.testing.assert_array_equal(annual.member_scope,members['ranking'].scope)
    np.testing.assert_array_equal(annual.control_scope,members['ranking'].scope & members['ranking'].ordinary)
    np.testing.assert_array_equal(annual.pred_row170,members['ranking'].values[:,-1])
    seqchecks={};frames={};mlps={};splitrec={}
    mp=stage(MLP/'results/stack-pack-011/MEMBER.json',dst/'MLP-MEMBER.json');mr=stage(MLP/'results/pack-011/REVIEW-011.md',dst/'MLP-PASS.md')
    mlp_member=js(mp);assert mr.read_text().startswith('VERDICT: PASS') and sha(mp)[:8] in mr.read_text()
    assert mlp_member['member_training_months']['fold']==[2,3,4,5,6,8,9,10,11,12]
    assert mlp_member['member_training_months']['ranking_final']==list(range(1,13))
    for split in ['fold','ranking']:
        b=bundles[split];m=members[split];typ='annual' if split=='ranking' else 'held'
        folder=DIV/'results'/(Path(pack['records']['sequence'][typ+'_fit_path']).name+'-fit')
        fname='ranking2026.parquet' if split=='ranking' else 'jjrefit.parquet'
        seq=stage(folder/fname,dst/f'sequence-{split}.parquet',pack['records']['sequence']['annual_raw_vector_sha256' if split=='ranking' else 'held_vector_sha256'])
        receipt=stage(folder/'FIT.json',dst/f'sequence-{split}-FIT.json',pack['records']['sequence'][typ+'_fit_sha256'])
        seqchecks[split]=check_sequence(pack,split,seq,receipt)
        values=aligned(seq,b[ID]);assert np.isfinite(values[m.scope]).all()
        if split=='ranking':np.testing.assert_array_equal(annual.pred_seq170,values)
        f=b[[ID,'month','cls']].copy();f['member_scope']=m.scope
        for name,j in [('g',0),('mlp',1),('sequence021',2),('control',3)]:f[name]=m.values[:,j]
        f['sequence_new']=values;frames[split]=f
        vpath=stage(MLP/'results/stack-pack-011'/mlp_member['vectors'][split]['path'],dst/f'mlp-{split}.parquet',mlp_member['vectors'][split]['sha256'])
        v=pd.read_parquet(vpath)
        for col in [ID,'month','cls']:np.testing.assert_array_equal(v[col],b[col])
        assert v.mlp_scope.to_numpy(bool)[m.scope].all()
        src=mlp_member['sources']['ranking_final' if split=='ranking' else 'fold']
        raw=stage(ROOT/src['path'],dst/f'mlp-{split}-raw.parquet',src['sha256'])
        np.testing.assert_array_equal(v.pred_mlp164,aligned(raw,b[ID],'prediction'))
        assert src['bag_record']['seeds']==list(range(701,707))
        mlps[split]=v.pred_mlp164.to_numpy(float)
        splitrec[split]={'sequence_new':record(seq),'sequence_receipt':record(receipt),'mlp_vector':record(vpath)}
    pairfolder='paired-mixed18-033a-B' if key=='033b' else 'paired-fast-last-bag12-030-B'
    pair=stage(DIV/'results'/pairfolder/'PAIRED.json',dst/'SOURCE-PAIR.json',pack['read_sha256'])
    stage(DIV/'results'/pairfolder/'paired_ls.parquet',dst/'source-pair-held.parquet')
    maskfolder='masked-mixed18-LIRF-033a-fit' if key=='033b' else 'masked-fast-last-bag12-LIRF-030b-fit'
    stage(DIV/'results'/maskfolder/'jjrefit.parquet',dst/'masked-new.parquet')
    pr=js(stage(DIV/'results'/maskfolder/'PREDICT.json',dst/'masked-new-PREDICT.json'))
    assert pr['label_free'] and pr['unaffected_predictions_exact']
    assert pr['fit_sha256']==pack['records']['sequence']['held_fit_sha256']
    assert pr['files']['jjrefit.parquet']==sha(dst/'masked-new.parquet')
    stage(DIV/'results/masked-bag6-LIRF-020/jjrefit.parquet',dst/'masked-old.parquet','792c9f3236370a6970372bbbd76bc2650c1f05b11cb8f6d771c47b21d37cade7')
    return pack,frames,mlps,splitrec,seqchecks


def fit(b,frame,base,B,y,w):
    on=frame.member_scope.to_numpy(bool);scale=np.where(b.practice,1-b.practice_weight,1.)
    X=(frame[NAMES].to_numpy(float)-base[:,None])*scale[:,None];folds={};pooled={}
    for cls in ['ordinary','clock']:
        names=NAMES if cls=='ordinary' else NAMES[:-1];folds[cls]={}
        for month in [1,7,None]:
            mask=on & b.cls.eq(cls).to_numpy()
            if month:mask &= b.month.eq(month).to_numpy()
            A=X[mask,:len(names)]*np.sqrt(w[mask,None]);r=(y-B)[mask]*np.sqrt(w[mask])
            assert np.isfinite(A).all() and np.isfinite(r).all()
            beta,_,rank,_=np.linalg.lstsq(A,r,rcond=None);assert rank==len(names)
            normal=A.T@A;rhs=A.T@r
            np.testing.assert_allclose(np.linalg.solve(normal,rhs),beta,rtol=0,atol=1e-10)
            rec={'method':'practice_scaled_wls','columns':names,'coefficients':beta.tolist(),'trained_on_months':[month] if month else [1,7],
                 'fit_rows':int(mask.sum()),'xtwx':normal.tolist(),'xtwr':rhs.tolist(),'condition_number':float(np.linalg.cond(normal))}
            if month:folds[cls][str(8-month)]=rec
            else:pooled[cls]=rec
    return folds,pooled


def score(ids,old,new,y):
    r=GS.score(ids,old,new,y=y,require_complete=True)
    return {k:r[k] for k in ['G','interval','views','G_without_top_days']}


def independent_tail(b,z):
    z=z.copy();n=b.noplan.to_numpy(bool);z[n]=b.np_v3cal.to_numpy(float)[n]
    for prefix in ['practice','c73']:
        q=b[prefix].to_numpy(bool);weight=b[prefix+'_weight'].to_numpy(float);point=b[prefix+'_point'].to_numpy(float)
        z[q]=np.maximum(60,z[q]+weight[q]*(point[q]-z[q]))
    lo=b.lo.to_numpy(float);hi=b.hi.to_numpy(float);q=np.isfinite(lo)
    z[q]=np.maximum(60,np.maximum(lo[q],np.minimum(hi[q],z[q])))
    # The C55 implementation owns the stand/bracket consistency checks.
    return apply_floor(z,b.stand_floor.to_numpy(float),b.airport.to_numpy(),lo,hi)


def shadow(b,m,h,transition,dst):
    mask=b.airport.eq('LIRF').to_numpy() & b.month.eq(7).to_numpy() & m.ordinary
    old=aligned(dst/'masked-old.parquet',b[ID]);new=aligned(dst/'masked-new.parquet',b[ID])
    source_mask=b.airport.eq('LIRF').to_numpy() & b.month.eq(7).to_numpy()
    np.testing.assert_array_equal(old[~source_mask],m.values[~source_mask,2])
    np.testing.assert_array_equal(new[~source_mask],transition.values[~source_mask,3])
    # Diagnostic arithmetic only; production fingerprint checks remain unchanged.
    base=h.new_base;prior=h.blend(m,b[ID].to_numpy(),base,True)
    pv=m.values.copy();pv[mask,2]=old[mask]
    po=matrix_apply(base,prior,pv,m.weights,m.scope,m.ordinary)
    nv=transition.values.copy();nv[mask,2]=old[mask];nv[mask,3]=new[mask]
    no=matrix_apply(base,prior,nv,transition.weights,m.scope,m.ordinary)
    return independent_tail(b,po),independent_tail(b,no)


def source_parity(b,m,frame,pack,dst,y):
    ids=b[ID].to_numpy('int64');pre=compose(b,route='R4mk',noplan='none',practice='none',c33=False)
    B=compose(b,route='R4mk',noplan='v3cal',practice='selected')
    w=np.zeros((len(b),3));v=np.column_stack([pre,frame.sequence_new,frame.control])
    for cls in ['ordinary','clock']:
        for mon in [1,7]:
            use=m.scope & b.cls.eq(cls).to_numpy() & b.month.eq(mon).to_numpy();r=pack['weights'][cls][str(mon)]
            assert r['fit_month']==8-mon
            w[use]=[0,r['weights']['sequence'],r['weights']['control']] if cls=='ordinary' else [0,r['weight'],0]
    p=JointBlend(ids,v,w,m.scope,m.ordinary,vector_digest(ids,pre),m.fallback)
    got=compose(b,route='R4mk',noplan='v3cal',practice='selected',blend=p,pair_extension=True)
    target=aligned(dst/'source-pair-held.parquet',ids)
    np.testing.assert_allclose(got,target,rtol=0,atol=1e-8)
    r=score(ids,B,got,y);source_score=js(dst/'SOURCE-PAIR.json')['forms']['paired_ls']['score']
    assert abs(r['G']-source_score['G'])<1e-7,(r['G'],source_score['G'])
    return {'complete_source_pair_exact_atol':1e-8,'G':r['G'],'owner_G':source_score['G']}


def scope_check(b,m,h,t,old,new):
    np.testing.assert_array_equal(old[~m.scope],new[~m.scope])
    for k in ['extension','noplan','bet']:np.testing.assert_array_equal(old[b[k]],new[b[k]])
    np.testing.assert_array_equal(clip(new,b.lo.to_numpy(),b.hi.to_numpy()),new)
    np.testing.assert_array_equal(apply_floor(new,b.stand_floor.to_numpy(),b.airport.to_numpy(),b.lo.to_numpy(),b.hi.to_numpy()),new)
    prior=h.blend(m,b[ID].to_numpy(),h.new_base,True)
    delta=np.zeros(len(b))
    for j in range(5):
        on=m.scope & (t.weights[:,j]!=0)
        delta[on]+=t.weights[on,j]*(t.values[on,j]-h.new_base[on])
    prior[m.scope]=np.maximum(60,h.new_base[m.scope]+delta[m.scope])
    np.testing.assert_array_equal(independent_tail(b,prior),new)
    return {'rows':len(b),'changed':int((old!=new).sum()),'off_gate_changed':0,'extension_rows_exact':int(b.extension.sum()),
            'pair_extension_exact':int((m.fallback.scope&m.fallback.extension).sum()),'independent_formula_exact':True,
            'mean_shift':float(np.mean(new-old)),'distance_all_rows':float(np.mean((new-old)**2))}


def paths(key):
    return OUT/key,EXP/f'results/stack7-read-022-{key}.json',EXP/f'results/stack7-forecast-frozen-022-{key}.json'


def prepare_read():
    assert sha(SPEC)==SPEC_SHA and not CANDIDATE.exists()
    choice=source_choice();key=choice['key'];dst,result,forecast=paths(key)
    assert not result.exists() and not forecast.exists();dst.mkdir(exist_ok=True)
    if (dst/'lane-c-swap.json').exists():
        prior_choice=js(dst/'lane-c-swap.json')['chosen_source']
        for k in ['key','pack','vector','review']:assert prior_choice[k]==choice[k]
        choice=prior_choice
    start=now();code=code_pins();pin(SPEC,SPEC_SHA);pin(PARENT_FILE,PARENT)
    stage(ROOT/'docs/status/2026-10-02-exp169-stack6-review/review-stack6.log',dst/'PARENT-PASS.log','9038f20a69816233c44d4e4f24fc24ef44c77d02eb89cab5418e6b835131413a')
    bundles={};members={};heads={};parents={}
    for split in ['fold','ranking']:
        bp=pin(EXP/f'inputs/stack5-016/bundle-{split}.parquet');b=pd.read_parquet(bp)
        m=load_member(pin(OLD,JOINT),JOINT,b,split);h=load_head_swap(pin(HEAD),sha(HEAD),b,m,split)
        bundles[split]=b;members[split]=m;heads[split]=h
        parents[split]=compose(b,blend=m,head_swap=h,**CFG)
    q=pd.read_parquet(PARENT_FILE);np.testing.assert_array_equal(q[ID],bundles['ranking'][ID]);np.testing.assert_array_equal(parents['ranking'],q[COL])
    buf=io.BytesIO();pd.DataFrame({ID:q[ID],COL:parents['ranking']}).to_parquet(buf,index=False);assert hashlib.sha256(buf.getvalue()).hexdigest()==PARENT
    held=pd.read_parquet(pin(EXP/'results/stack6-held-021.parquet'));np.testing.assert_array_equal(held[ID],bundles['fold'][ID]);np.testing.assert_array_equal(held.stack6,parents['fold'])
    pack,frames,mlps,splits,receipts=source_inputs(choice,dst,bundles,members)
    b=bundles['fold'];m=members['fold'];h=heads['fold'];ids=b[ID].to_numpy('int64');y=GS.labels(ids)
    parity=source_parity(b,m,frames['fold'],pack,dst,y)
    rows=GS.load_rows().set_index(ID).loc[ids];counts=rows.groupby(['month','airport','cls']).size();cells=GS.load_cells().set_index(['month','airport','cls']).N
    w=np.array([cells.get((mon,a,c),0)/GS.N_RANKING/counts[mon,a,c] if counts[mon,a,c]>=GS.MIN_SUPPORT else 0. for mon,a,c in zip(rows.month,rows.airport,rows.cls)])
    np.testing.assert_array_equal(m.scope,(b.F.to_numpy()<2400)&rows.aobt.to_numpy(bool)&b.cls.ne('noac').to_numpy()&(w>0)&~b.noplan.to_numpy(bool))
    B3=compose(b,head_swap=h,**{**CFG,'practice':'selected','c55':False,'pair_extension':False})
    folds,pool=fit(b,frames['fold'],h.new_base,B3,y,w)
    lane=make_transition(h,m,frames['fold'],folds,pool,'fold');lane_pred=compose(b,blend=m,head_swap=lane,**CFG)
    reads={'lane_c_vs_stack6':score(ids,parents['fold'],lane_pred,y)}
    print('lane C',key,reads['lane_c_vs_stack6']['G'],reads['lane_c_vs_stack6']['interval'],flush=True)
    eframes={s:f.copy() for s,f in frames.items()}
    for s in eframes:
        use=members[s].scope & ~members[s].ordinary;eframes[s].loc[use,'mlp']=mlps[s][use]
    efold,epool=fit(b,eframes['fold'],h.new_base,B3,y,w)
    assert efold['ordinary']==folds['ordinary'] and epool['ordinary']==pool['ordinary']
    e=make_transition(h,m,eframes['fold'],efold,epool,'fold');epred=compose(b,blend=m,head_swap=e,**CFG)
    np.testing.assert_array_equal(epred[m.ordinary],lane_pred[m.ordinary])
    reads['clock_mlp_step']=score(ids,lane_pred,epred,y);use_e80=reads['clock_mlp_step']['G']>0
    print('clock e80',reads['clock_mlp_step']['G'],reads['clock_mlp_step']['interval'],'include',use_e80,flush=True)
    if use_e80:frames=eframes;folds,pool=efold,epool
    write(dst/'desk-approval.json',{'approved':True,'desk_orders':'2026-10-02 17:59,18:07,18:09 UTC; one build after exact commit','spec_sha256':SPEC_SHA,'pack_sha256':choice['pack']['sha256'],'parent_sha256':PARENT})
    manifest={'schema':2,'form':'J','weight_policy':'other_month_practice_scaled_wls_then_pooled_2026','chosen_source':choice,'clock_mlp_e80':use_e80,
              'parent':record(PARENT_FILE),'spec':record(SPEC),'pack':record(dst/'PACK.json'),'source_review':record(dst/'SOURCE-PASS.md'),
              'mlp_pack':record(dst/'MLP-MEMBER.json'),'mlp_review':record(dst/'MLP-PASS.md'),'parent_review':record(dst/'PARENT-PASS.log'),
              'desk_approval':record(dst/'desk-approval.json'),'head_swap':record(HEAD),'fold_weights':folds,'ranking_weights':pool,'splits':splits}
    for s in ['fold','ranking']:
        bp=EXP/f'inputs/stack5-016/bundle-{s}.parquet';matrix=dst/f'matrix-{s}.parquet'
        if matrix.exists():pd.testing.assert_frame_equal(pd.read_parquet(matrix),frames[s],check_exact=True)
        else:frames[s].to_parquet(matrix,index=False)
        t=make_transition(heads[s],members[s],frames[s],folds,pool,s)
        splits[s].update(bundle=record(bp),matrix=record(matrix),matrix_digest=t.matrix_digest,parent_member_fingerprint=fingerprint(members[s]),base_digest=vector_digest(members[s].ids,heads[s].new_base))
    write(dst/'lane-c-swap.json',manifest)
    swaps={s:load_lane_c_swap(dst/'lane-c-swap.json',sha(dst/'lane-c-swap.json'),bundles[s],members[s],heads[s],s) for s in bundles}
    predictions={s:compose(bundles[s],blend=members[s],head_swap=swaps[s],**CFG) for s in bundles}
    full=predictions['fold'];reads['full_vs_stack6']=score(ids,parents['fold'],full,y)
    B=compose(b,route='R4mk',noplan='v3cal',practice='selected')
    v47=compose(b,route='R4',form='HA3',extension=False,noplan='none',c33=False)
    reads['full_vs_B']=score(ids,B,full,y);reads['full_vs_v47']=score(ids,v47,full,y)
    maskedold,maskednew=shadow(b,m,h,swaps['fold'],dst);reads['matched_shadow_vs_stack6']=score(ids,maskedold,maskednew,y)
    scopes={s:scope_check(bundles[s],members[s],heads[s],swaps[s],parents[s],predictions[s]) for s in bundles}
    assert scopes['ranking']['extension_rows_exact']==17017 and scopes['ranking']['pair_extension_exact']==16748
    exclusions=load_member(EXP/'inputs/stack3-ext-009/pair.json',sha(EXP/'inputs/stack3-ext-009/pair.json'),bundles['ranking'],'ranking')
    removed=exclusions.scope & exclusions.extension & ~members['ranking'].fallback.scope;assert removed.sum()==60
    np.testing.assert_array_equal(predictions['ranking'][removed],parents['ranking'][removed]);scopes['ranking']['missing_AOBT3_pair_exclusions_exact']=60
    held7=pd.DataFrame({ID:ids,'stack6':parents['fold'],'lane_c':lane_pred,'clock_e80':epred,'stack7':full,'masked_stack6':maskedold,'masked_stack7':maskednew})
    heldpath=EXP/f'results/stack7-held-022-{key}.parquet'
    if heldpath.exists():pd.testing.assert_frame_equal(pd.read_parquet(heldpath),held7,check_exact=True)
    else:held7.to_parquet(heldpath,index=False)
    limits='Selected lane C and e80 recipe/class choice on these held rows; conditional day interval, no seed/source-selection/2026-transfer coverage. Parent C73 upper600 scenario stays; corrected centre536-576. Ratio1.24 fixed. No new extension credit.'
    for p in dst.iterdir():
        if p.is_file():pin(p)
    for p,hsh in js(EXP/'results/stack6-read-021.json')['source_sha256'].items():
        if Path(p).suffix in ('.parquet','.json','.npz','.pt'):pin(ROOT/p,hsh)
    oldcheck={p:hsh for p,hsh in js(EXP/'results/stack7-before-022.json')['files'].items() if Path(p).suffix=='.parquet' or 'forecast-frozen' in p}
    for p,hsh in oldcheck.items():assert sha(ROOT/p)==hsh
    resultbody={'source_rows':[899,901],'choice':choice,'clock_mlp_e80':use_e80,'code_sha256':code,'source_sha256':PINS,'source_parity':parity,'receipts':receipts,
                'parent_byte_exact':True,'old_artifacts_unchanged':len(oldcheck),'reads':reads,'scope':scopes,'prediction_digests':{s:vector_digest(bundles[s][ID],predictions[s]) for s in bundles},
                'limits':limits,'spec_sha256':SPEC_SHA,'started_at_utc':start,'prepared_at_utc':now()}
    assert code_pins()==code
    write(result,resultbody)
    r=reads['matched_shadow_vs_stack6'];p=224.21270981951176;ratio=1.24
    # A source or scope failure stops above; an unexpected gain leaves the read for desk review.
    low,high=([280,840] if key=='030' else [360,940]) if use_e80 else ([300,700] if key=='030' else [380,800])
    assert low<=reads['full_vs_stack6']['G']<=high and r['G']>0,'Outside SPEC prior: report desk with read before freezing.'
    freeze={'central':float(np.sqrt(p*p-ratio*r['G'])),'conditional_range':[float(np.sqrt(p*p-ratio*r['interval']['hi95'])),float(np.sqrt(p*p-ratio*r['interval']['lo95']))],
            'unmasked_sensitivity':float(np.sqrt(p*p-ratio*reads['full_vs_stack6']['G'])),'parent':p,'parent_sha256':PARENT,'ratio':ratio,'ratio_basis':'ledger899 v44/v45 versus v43; baseline901 v47',
            'increment_G':r['G'],'increment_interval':r['interval'],'choice':choice,'clock_mlp_e80':use_e80,'limits':limits,'incremental_extension_credit':0.,'frozen_at_utc':now(),'candidate_exists':False,'spec_sha256':SPEC_SHA,'read_sha256':sha(result)}
    write(forecast,freeze)
    argv=[str(ROOT/'.venv/bin/python'),str(EXP/'code/assembly.py'),'--bundle',str(EXP/'inputs/stack5-016/bundle-ranking.parquet'),'--bundle-sha256',sha(EXP/'inputs/stack5-016/bundle-ranking.parquet'),'--route','R4mk','--form','HB3u','--noplan','v3cal','--practice','selected_c73','--blend-manifest',str(OLD),'--blend-manifest-sha256',JOINT,'--head-swap-manifest',str(HEAD),'--head-swap-manifest-sha256',sha(HEAD),'--lane-c-swap-manifest',str(dst/'lane-c-swap.json'),'--lane-c-swap-manifest-sha256',sha(dst/'lane-c-swap.json'),'--pair-extension','proxy-2','--c55','--out',str(CANDIDATE)]
    write(dst/'build-command.json',{'argv':argv,'code_sha256':code,'key':key,'spec_sha256':SPEC_SHA,'forecast_sha256':sha(forecast),'read_sha256':sha(result),'written_at_utc':now()})
    print('FROZEN',key,freeze['central'],freeze['conditional_range'],'matched G',r['G'],flush=True)


def committed(p):
    rel=str(Path(p).relative_to(ROOT));body=subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT,stderr=subprocess.PIPE)
    assert hashlib.sha256(body).hexdigest()==sha(p),f'Commit required before build: {rel}'
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def check_build_ready(out):
    assert Path(out).resolve()==CANDIDATE and not CANDIDATE.exists()
    choice=source_choice();dst,result,forecast=paths(choice['key']);r=js(result);f=js(forecast);cmd=js(dst/'build-command.json')
    for k in ['key','pack','vector','review']:assert choice[k]==r['choice'][k],f'Chosen source changed: {k}'
    assert sha(SPEC)==SPEC_SHA==r['spec_sha256']==cmd['spec_sha256']
    assert sha(forecast)==cmd['forecast_sha256'] and sha(result)==cmd['read_sha256']
    assert code_pins()==r['code_sha256']==cmd['code_sha256']
    for p,h in r['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert not f['candidate_exists'] and f['frozen_at_utc']<now()
    for p in [SPEC,forecast,checked(choice['review']),checked(choice['pack']),dst/'SOURCE-PASS.md']:
        committed(p)
    if r['clock_mlp_e80']:
        committed(MLP/'results/pack-011/REVIEW-011.md');committed(dst/'MLP-PASS.md')
    return choice


def build():
    choice=check_build_ready(CANDIDATE);dst,result,forecast=paths(choice['key']);cmd=js(dst/'build-command.json')
    if not COMMAND.exists():write(COMMAND,cmd)
    else:assert js(COMMAND)==cmd
    commit=committed(SPEC);subprocess.run(cmd['argv'],cwd=ROOT,check=True)
    q=pd.read_parquet(CANDIDATE);r=js(result)
    assert len(q)==344841 and q[ID].is_unique and list(q.columns)==[ID,COL]
    assert q[ID].dtype==q[COL].dtype==np.dtype('float64')
    np.testing.assert_array_equal(q[ID],pd.read_parquet(ROOT/'data/raw/submitting.parquet')[ID])
    assert np.isfinite(q[COL]).all() and q[COL].min()>=60
    assert vector_digest(q[ID],q[COL])==r['prediction_digests']['ranking']
    write(EXP/'results/stack7-build-022.json',{'candidate':record(CANDIDATE),'choice':choice,'read':record(result),'forecast':record(forecast),'manifest':record(dst/'lane-c-swap.json'),'commit_before_build':commit,'built_at_utc':now(),'review':'desk D123 pending'})
    print('BUILT',sha(CANDIDATE),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare-read','build']);a=ap.parse_args()
    prepare_read() if a.stage=='prepare-read' else build()
