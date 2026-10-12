"""Read saved two-fold stacks; create ID-aligned bundles for the common assembly path."""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ID, COL, compose, clip
ROOT=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parents[1]
VALID=ROOT.parent/'prc-data-challenge-2026-exp147/experiments/exp-148-short-route/results/fold'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from prc import gscore as GS

INPUTS={}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def read(path,**kw):
    path=Path(path);INPUTS[str(path)]=sha(path)
    return pd.read_parquet(path,**kw)

def align(path,col,ids):
    a=read(path).set_index(ID)
    a.index=a.index.astype('int64')
    assert a.index.is_unique
    return a[col].reindex(ids).to_numpy(float)

def add_bracket(b,raw):
    raw=raw[raw.PHASE_mvt=='DEP'].set_index(ID)
    assert raw.index.is_unique
    m=(raw.MVT_TIME_UTC_mvt-raw.LOBT_flt).dt.total_seconds().reindex(b[ID]).to_numpy(float)
    b['lo'],b['hi']=m-3606,m+3606
    b['airport']=raw.ADEP_mvt.reindex(b[ID]).to_numpy()
    b['month']=raw.MVT_TIME_UTC_mvt.dt.month.reindex(b[ID]).to_numpy()
    b['joined']=np.isfinite(m)

def add_np(b,path,ranking):
    n=read(path)
    if ranking:
        n=n.rename(columns={'base':'np_base','candidate':'np_v1','candidate_bet':'np_bet'})
        v=read(ROOT/'experiments/exp-167-target/candidates/noplan-form-v2-001.parquet').set_index(ID)
        n['np_v2']=v.candidate.reindex(n[ID]).to_numpy()
        n['bet']=n.np_bet!=n.np_v1
    n=n.set_index(ID)
    assert n.index.is_unique
    b['noplan']=b[ID].isin(n.index)
    for k in ['np_base','np_v1','np_v2','np_bet']:
        b[k]=n[k].reindex(b[ID]).to_numpy()
    b['bet']=n.bet.reindex(b[ID]).eq(True).to_numpy(bool)
    assert not (b.noplan & (b.rule | b.extension | b.joined)).any()
    assert np.array_equal(b.F[b.noplan].to_numpy(),b.np_base[b.noplan].to_numpy())


def prepare():
    r=GS.load_rows();r=r[r.month.isin([1,7])].reset_index(drop=True)
    b=pd.DataFrame({ID:r[ID].astype('int64')})
    ids=b[ID].to_numpy()
    v32=align(ROOT/'data/exp167/refs.parquet','v32',ids)
    keep=(r.cls=='noac').to_numpy()|((r.cls=='ordinary').to_numpy() & ~r.aobt.to_numpy(bool))
    for stage in ['R4','R4mk','R4A3']:
        folder=ROOT/'data/exp167' if stage=='R4' else EXP/'inputs'
        parts=[read(folder/f'{stage}Fjjrefit-month-2025-{m}.parquet').set_index(ID) for m in ['01','07']]
        a=pd.concat(parts);a.index=a.index.astype('int64')
        assert a.index.is_unique and set(a.index)==set(ids)
        pred=a.pred_Sb.reindex(ids).to_numpy(float)
        b['F' if stage=='R4' else stage]=.5*np.where(keep,v32,pred)+.5*v32
    b['HA3']=align(ROOT/'data/exp167/HA3-jjrefit.parquet','pred_HA3',ids)
    b['HB1']=align(EXP/'inputs/HB1-jjrefit.parquet','pred_HB1',ids)
    unsupported={('LEMD',1),('LFPG',1),('LIRF',1)}
    b['rule']=(r.cls=='clock').to_numpy() & np.array([(a,int(m)) not in unsupported for a,m in zip(r.airport,r.month)])
    b['extension']=False
    columns=[ID,'PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','ADEP_mvt']
    raw=pd.concat([read(next((ROOT/'data/raw').glob(f'training_2025-{m}-*.parquet')),columns=columns) for m in ['01','07']],ignore_index=True)
    add_bracket(b,raw)
    add_np(b,EXP/'inputs/noplan-fold.parquet',False)
    b['cls']=r.cls.to_numpy()
    b['pair']=r.pair.to_numpy()
    y=GS.labels(ids)
    assert ((y[b.joined]>=b.lo[b.joined]) & (y[b.joined]<=b.hi[b.joined])).all()
    c=read(ROOT/'submissions/uploaded/youthful-giraffe_v43.parquet').rename(columns={COL:'F'})
    ids26=c[ID].to_numpy().astype('int64')
    for stage in ['R4mk','R4A3']:
        c[stage]=align(VALID/f'ranking-fold-{stage}-B50.parquet',COL,ids26)
    for head in ['HA3','HB1']:
        c[head]=align(EXP/f'inputs/{head}-jjrank.parquet','pred_'+head,ids26)
    raw=read(ROOT/'data/raw/ranking.parquet',columns=columns)
    add_bracket(c,raw)
    clock=np.isfinite(c.HB1)
    assert np.array_equal(clock,np.isfinite(c.HA3))
    c['rule']=clock & np.array([(a,int(m)) not in unsupported for a,m in zip(c.airport,c.month)])
    c['extension']=clock & c.airport.isin(['LEMD','LFPG']) & (c.month==1)
    add_np(c,ROOT/'experiments/exp-167-target/candidates/noplan-form-001.parquet',True)
    template=read(ROOT/'data/raw/submitting.parquet',columns=[ID])
    assert np.array_equal(template[ID],c[ID]) and c[ID].is_unique and len(c)==344841
    assert int(c.rule.sum())==173539 and int(c.extension.sum())==17017 and int(c.noplan.sum())==383
    for x,label in [(b,'fold'),(c,'ranking')]: x.to_parquet(EXP/f'inputs/bundle-{label}.parquet',index=False)
    return b,c,y


def score(b,base,pred,y):
    z=GS.score(b[ID].to_numpy(),base,pred,y=y,groups={'pair':b.pair.to_numpy()},require_complete=True)
    return {'G':z['G'],'lo95':z['interval']['lo95'],'hi95':z['interval']['hi95'],
            'month':z['views']['month'],'class':z['views']['class'],'G_without_top_days':z['G_without_top_days'],
            'rows_changed':int((pred!=base).sum())}


def config(**kwargs):
    return dict(route='R4',form='HB1',extension=True,noplan='v1',bet=False,c33=False)|kwargs


def parity(b,c,y):
    checks={}
    f=b.F.to_numpy(float)
    baseline=compose(b,**config(form='HA3',extension=False,noplan='none'))
    hb=compose(b,**config(noplan='none'))
    np1=compose(b,**config(form='F',extension=False))
    np2=compose(b,**config(form='F',extension=False,noplan='v2'))
    for name,base,pred,want in [('v47_vs_F',f,baseline,807.4),('HB1_vs_F',f,hb,1265.3),
                              ('noplan_v1_vs_F',f,np1,1567.5),('noplan_v2_vs_F',f,np2,1653.5)]:
        z=score(b,base,pred,y);print(name,z,flush=True);assert round(z['G'],1)==want,(name,z,want);checks[name]=z
    for name,cfg,path in [
        ('v47',config(form='HA3',extension=False,noplan='none'),ROOT/'submissions/uploaded/youthful-giraffe_v47.parquet'),
        ('HB1',config(extension=False,noplan='none'),ROOT/'submissions/candidates/exp159-fold-F-HB1-w50.parquet'),
        ('HB1x',config(noplan='none'),ROOT/'submissions/candidates/exp159-fold-F-HB1-w50-x.parquet'),
        ('noplan_v1',config(form='HA3',extension=False),ROOT/'submissions/candidates/exp167-noplan-form-001-v47.parquet'),
        ('R4mk_HB1x_C33',config(route='R4mk',noplan='none',c33=True),ROOT/'submissions/candidates/exp148-fold-FR4mk-HB1xform-c33.parquet'),
        ('R4A3_v47_C33',config(route='R4A3',form='HA3',extension=False,noplan='none',c33=True),ROOT/'submissions/candidates/exp148-fold-FR4A3-rule-v47form-c33.parquet'),
        ('bet_e8802b41',config(form='HA3',extension=False,bet=True),ROOT/'data/exp167/noplan-form-001-bet-entry.parquet')]:
        p=compose(c,**cfg);want=align(path,COL,c[ID].to_numpy('int64'))
        delta=float(np.max(np.abs(p-want)));assert delta==0,(name,delta)
        checks[name]={'max_abs_s':delta,'source_sha256':sha(path)}
    return checks,baseline


def overlap(b,tag):
    # These patches describe each component's actual changed rows in its named context.
    # They are for conflict analysis only. compose() rebuilds the route inside the blend.
    v47=compose(b,**config(form='HA3',extension=False,noplan='none'))
    hb=compose(b,**config(extension=False,noplan='none'))
    hbx=compose(b,**config(noplan='none'))
    mk=compose(b,**config(route='R4mk',noplan='none'))
    a3=compose(b,**config(route='R4A3',noplan='none'))
    np1=compose(b,**config(form='HA3',extension=False))
    full=compose(b,**config(route='R4mk'))
    final=compose(b,**config(route='R4mk',c33=True))
    patches={'HB1':(hb!=v47,hb),'extension':(hbx!=hb,hbx),'noplan_v1':(np1!=v47,np1),
             'R4mk':(mk!=hbx,mk),'R4A3_rule':(a3!=hbx,a3),'C33_on_stack':(final!=full,final)}
    rows=[];ids=b[[ID]].copy()
    for name,(mask,p) in patches.items():ids[name]=mask
    ids.to_parquet(EXP/f'results/overlap-masks-{tag}.parquet',index=False)
    for A,B in itertools.combinations(patches,2):
        ma,pa=patches[A];mb,pb=patches[B];both=ma & mb
        conflicts=both & (pa!=pb)
        rows.append({'A':A,'B':B,'A_changed':int(ma.sum()),'B_changed':int(mb.sum()),
                     'both_change':int(both.sum()),'different_values':int(conflicts.sum()),
                     'max_order_difference_s':float(np.max(np.abs(pa[conflicts]-pb[conflicts]))) if conflicts.any() else 0,
                     'A_then_B':B+' value on intersection','B_then_A':A+' value on intersection'})
        if conflicts.any():
            pd.DataFrame({ID:b[ID][conflicts],A+'_then_'+B:pb[conflicts],B+'_then_'+A:pa[conflicts]}).to_parquet(
                EXP/f'results/conflict-{tag}-{A}-{B}.parquet',index=False)
    # True operator check: C33 before a later route/head write can reopen the bracket.
    early=clip(v47,b.lo.to_numpy(),b.hi.to_numpy())
    for name,(mask,p) in patches.items():
        if name=='C33_on_stack':continue
        q=early.copy();q[mask]=p[mask]
        bad=b.joined.to_numpy() & ((q<b.lo.to_numpy()) | (q>b.hi.to_numpy()))
        rows.append({'A':'C33_first','B':name,'bracket_violations_after_late_patch':int(bad.sum())})
    for name,(mask,p) in patches.items():
        if name=='C33_on_stack':continue
        first=v47.copy();first[mask]=p[mask]
        after=clip(first,b.lo.to_numpy(),b.hi.to_numpy())
        before=clip(v47,b.lo.to_numpy(),b.hi.to_numpy());before[mask]=p[mask]
        changed=after!=before
        rows.append({'A':name,'B':'C33_operator','different_values':int(changed.sum()),
                     'max_order_difference_s':float(np.max(np.abs(after-before))),
                     'A_then_B':'Clip of the patched value','B_then_A':'Unclipped patch value'})
        if changed.any():
            pd.DataFrame({ID:b[ID][changed],'patch_then_C33':after[changed],
                          'C33_then_patch':before[changed]}).to_parquet(
                EXP/f'results/order-{tag}-{name}-C33.parquet',index=False)
    return rows


def bet_table(c):
    n=read(ROOT/'experiments/exp-167-target/candidates/noplan-form-001.parquet')
    n=n[n.candidate_bet!=n.candidate]
    assert len(n)==2 and hashlib.sha256(np.sort(n[ID].to_numpy('int64')).tobytes()).hexdigest()=='7e7c5db9fb220b500e758603d83995e607e9a224fcde3ed5f08aaa58de0fb9df'
    rec=json.loads((ROOT/'experiments/exp-167-target/results/noplan-form-001.json').read_text())
    tb=rec['rows_2026']['tables_twelve_months'];p=tb['rate'];r=tb['r_mid']
    base=n.candidate.to_numpy(float);cand=n.candidate_bet.to_numpy(float);ms=n.ms.to_numpy(float)
    table=[]
    for outcome in itertools.product('PR',repeat=2):
        y=np.array([r if k=='R' else ms[i] for i,k in enumerate(outcome)])
        gain=((y-base)**2-(y-cand)**2).sum()/344841
        prob=float(np.prod([p if k=='R' else 1-p for k in outcome]))
        table.append({'outcome':''.join(outcome),'gain_s2_per_row':float(gain),'probability_if_independent':prob})
    pp=table[0]['gain_s2_per_row'];rr=table[-1]['gain_s2_per_row']
    return {'rows':n[[ID,'ms','candidate','candidate_bet']].to_dict('records'),
            'p':p,'rollover_point':r,'rate_source':'8 of 11 in 2025; Jeffreys mean (8+.5)/(11+1)',
            'fold_rate':'Ten other months: (4+.5)/(6+1)', 'outcomes':table,
            'expected_gain':sum(z['gain_s2_per_row']*z['probability_if_independent'] for z in table),
            'common_rate_break_even':-pp/(rr-pp),
            'note':'Joint outcome probabilities assume independence. Expected gain depends only on marginal rates. No board value selects this arm.'}


def main():
    source_hashes={p.name:sha(p) for p in (EXP/'code').glob('*.py')}
    b,c,y=prepare();checks,v47=parity(b,c,y)
    records={};vectors={}
    cases=[('HB1',config(extension=False,noplan='none')),('HB1x',config(noplan='none')),
           ('HB1x_np1',config()),('HB1x_np1_R4mk',config(route='R4mk')),
           ('HB1x_np1_R4mk_C33',config(route='R4mk',c33=True)),
           ('HB1x_np1_R4A3',config(route='R4A3')),('HB1x_np1_R4A3_C33',config(route='R4A3',c33=True)),
           ('HB1x_np2_R4mk_C33',config(route='R4mk',noplan='v2',c33=True)),
           ('HB1x_np1_R4mk_bet_C33',config(route='R4mk',bet=True,c33=True))]
    parents={'HB1':'v47','HB1x':'HB1','HB1x_np1':'HB1x','HB1x_np1_R4mk':'HB1x_np1',
             'HB1x_np1_R4mk_C33':'HB1x_np1_R4mk','HB1x_np1_R4A3':'HB1x_np1',
             'HB1x_np1_R4A3_C33':'HB1x_np1_R4A3','HB1x_np2_R4mk_C33':'HB1x_np1_R4mk_C33',
             'HB1x_np1_R4mk_bet_C33':'HB1x_np1_R4mk_C33'}
    vectors['v47']=v47
    for name,cfg in cases:
        print('Read',name,flush=True)
        p=compose(b,**cfg);vectors[name]=p
        z=score(b,v47,p,y);m=score(b,vectors[parents[name]],p,y)
        pc=compose(c,**cfg)
        z.update({'marginal':m,'parent':parents[name],'config':cfg,'forecast_local_only':float(np.sqrt(237.7071**2-1.24*z['G']))})
        z['ranking_rows_changed_from_v47']=int((pc!=align(ROOT/'submissions/uploaded/youthful-giraffe_v47.parquet',COL,c[ID])).sum())
        if cfg['c33']:
            unclip=compose(c,**(cfg|{'c33':False}))
            z['c33_ranking_rows']=int((pc!=unclip).sum())
            z['c33_guaranteed_s2_per_row']=float(np.mean((pc-unclip)**2))
            pre=score(b,v47,compose(b,**(cfg|{'c33':False})),y)
            z['forecast_with_ranking_clip_bound']=float(np.sqrt(237.7071**2-1.24*pre['G']-z['c33_guaranteed_s2_per_row']))
            out=EXP/f'results/{name}.parquet'
            pd.DataFrame({ID:c[ID],COL:pc}).to_parquet(out,index=False)
            z['output_sha256']=sha(out)
        records[name]=z;print(name,'G',round(z['G'],6),'marginal',round(m['G'],6),flush=True)
    # Single component gains on a common v47 comparator, without C33 in the route terms.
    singles={}
    for name,cfg in [('noplan',config(form='HA3',extension=False)),
                     ('R4mk',config(route='R4mk',form='HA3',extension=False,noplan='none')),
                     ('R4A3_rule',config(route='R4A3',form='HA3',extension=False,noplan='none')),
                     ('C33',config(form='HA3',extension=False,noplan='none',c33=True))]:
        singles[name]=score(b,v47,compose(b,**cfg),y)
    additivity={}
    for stage,key in [('R4mk','R4mk'),('R4A3','R4A3_rule')]:
        name=f'HB1x_np1_{stage}_C33'
        total=records['HB1']['G']+singles['noplan']['G']+singles[key]['G']+singles['C33']['G']
        additivity[stage]={'sum_common_base_singles':total,'stacked':records[name]['G'],
                           'stack_minus_sum':records[name]['G']-total,
                           'route_interaction':records[f'HB1x_np1_{stage}']['marginal']['G']-singles[key]['G'],
                           'clip_interaction':records[name]['marginal']['G']-singles['C33']['G']}
    out={'parity':checks,'stacks':records,'single_reads':singles,'additivity':additivity,
         'overlap_fold':overlap(b,'fold'),'overlap_ranking':overlap(c,'ranking'),'bet':bet_table(c),
         'counts':{'fold':len(b),'fold_rule':int(b.rule.sum()),'fold_noplan':int(b.noplan.sum()),'ranking':len(c)},
         'extension':{'ranking_rows':17017,'local_rows':0,'proxy_per_clock_row':3187.7142879834305,
                      'proxy_whole_entry':3187.7142879834305*17017/344841,
                      'source':'exp-159 results/ext-HB1.json by_weight 0.5 january_six; proxy only'},
         'forecast_note':'1.24 from ledger row 899, v44 and v45 against v43; v47 vs F ratio 1.41 is a separate check. Forecasts never enter predictions.',
         'inputs_sha256':INPUTS,'code_sha256':source_hashes}
    assert source_hashes=={p.name:sha(p) for p in (EXP/'code').glob('*.py')}, 'Source changed during the read.'
    (EXP/'results/stack-reads.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
    for name in ['fold','ranking']:
        pd.DataFrame(out['overlap_'+name]).to_csv(EXP/f'results/overlap-{name}.csv',index=False)
    print('DONE',flush=True)

if __name__=='__main__': main()
