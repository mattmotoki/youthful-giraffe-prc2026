"""Three-arm decision evidence from saved predictions, raw inputs and 2025 rates only."""
from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np,pandas as pd
from assembly import ID,COL,compose,clip
ROOT=Path(__file__).resolve().parents[3];EXP=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'experiments/exp-167-target/code'))
import common as C
import practice_arm_r1 as P
sys.path.insert(0,str(ROOT))
from prc import gscore as GS
from read_stacks import score,config
N=344841;R=87361.0
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={}
def read(p,**kwargs):
    inputs[str(p)]=sha(p);return pd.read_parquet(p,**kwargs)

def raw_lirf():
    cols=[ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt',
          'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt','LOBT_flt','AOBT_3_flt','AIRCRAFT_TYPE_mvt']
    parts=[]
    for path in sorted((ROOT/'data/raw').glob('training_2025-*.parquet')):
        t=read(path,columns=cols)
        parts.append(t[((t.PHASE_mvt=='DEP')&(t.ADEP_mvt=='LIRF'))|((t.PHASE_mvt=='ARR')&(t.ADES_mvt=='LIRF'))])
    return pd.concat(parts,ignore_index=True)

def far_rows(t):
    d=t[(t.PHASE_mvt=='DEP')&(t.ADEP_mvt=='LIRF')].copy()
    d[ID]=d[ID].astype('int64');d['ms']=C.secs(d.MVT_TIME_UTC_mvt,d.SCHED_TIME_UTC_mvt)
    d['month']=d.MVT_TIME_UTC_mvt.dt.month;d['pre']=d.FLIGHT_mvt.str[:3];d['day']=d.MVT_TIME_UTC_mvt.dt.floor('D')
    tb=d.ms.to_numpy()/3600-.3;wrapped=tb-24*np.round(tb/24)
    d['zone']=np.where(np.abs(wrapped)>10.55,'forced',np.where(np.abs(wrapped)>10.2,'edge','open'))
    share=P.V2.arrival_share(t);key=pd.MultiIndex.from_arrays([d.pre,d.month])
    d['pre_def']=share.pre_def.reindex(key).to_numpy();d['pre_n']=share.pre_n.reindex(key).to_numpy()
    return d

def outcomes(base,cand,yp,yr):
    a=((yp-base)**2-(yp-cand)**2)/N;b=((yr-base)**2-(yr-cand)**2)/N
    den=a-b
    return a,b,np.divide(-b,den,out=np.full(np.shape(den),np.nan),where=den!=0)

def one_table(d,base,cand,kind='rollover'):
    p,r=d.ms.to_numpy(),np.full(len(d),R)
    gp,gr,_=outcomes(base,cand,p,r)
    den=gr-gp;be=np.divide(-gp,den,out=np.full(len(d),np.nan),where=den!=0)
    out=d[[ID,'FLIGHT_mvt','ms']].copy();out['base']=base;out['candidate']=cand
    out['base_weight_R']=(base-p)/(r-p);out['candidate_weight_R']=(cand-p)/(r-p)
    out['G_if_P']=gp;out['G_if_R']=gr;out['break_even_rate_R']=be
    for rate in [.15,.25,.5,.7083333333333334,.75]:out[f'G_at_p{rate:.3f}']=(1-rate)*gp+rate*gr
    return out

def all_outcomes(d,base,cand,name):
    p=d.ms.to_numpy();out=[]
    for states in itertools.product('PR',repeat=len(d)):
        yy=np.array([p[i] if k=='P' else R for i,k in enumerate(states)])
        out.append({'outcome':''.join(states),'gain':float(((yy-base)**2-(yy-cand)**2).sum()/N)})
    pd.DataFrame(out).to_csv(EXP/f'results/{name}-outcomes.csv',index=False)
    return {'min':min(z['gain'] for z in out),'max':max(z['gain'] for z in out),'row_order':d[ID].tolist()}

def main():
    t=raw_lirf();d=far_rows(t);b=read(EXP/'inputs/bundle-fold.parquet');c=read(EXP/'inputs/bundle-ranking.parquet')
    y=GS.labels(b[ID].to_numpy('int64'));v47=compose(b,**config(form='HA3',extension=False,noplan='none'))
    base25=compose(b,**config(route='R4mk',c33=True));base26=read(EXP/'exp169-HB1x-noplan-v1-R4mk-c33.parquet')[COL].to_numpy()
    bidx=pd.Index(b[ID]);cidx=pd.Index(c[ID]);rec={}
    # C45 source replay: exact source comparator, uniform weight, point, rate and day bootstrap.
    ph=C.secs(d.BLOCK_TIME_UTC_mvt,d.SCHED_TIME_UTC_mvt).abs()<=6
    d['real']=(~ph & ((d[COL]-600).abs()>12) & (d[COL]<=80000)).astype(float)
    mid=d[(d.ms>7200)&(d.ms<=34200)&~d.month.isin([1,7])]
    allr=mid.real.mean();tab=mid.groupby('pre').real.agg(pr='mean',pn='size')
    f=d[d.LOBT_flt.isna()&d.month.isin([1,7])&(d.ms>10.6*3600)].copy().join(tab,on='pre')
    f=f[f[ID].isin(bidx)].reset_index(drop=True)
    practice=((f.pr*f.pn.fillna(0)+allr*5)/(f.pn.fillna(0)+5)).fillna(allr).to_numpy()
    few=f.FLIGHT_mvt.isna()|(f.pre_n.fillna(0)<10)|(f.pre_def>.6)
    pos=bidx.get_indexer(f[ID]);F=b.F.to_numpy()[pos];pp=f.ms.to_numpy();lo=np.minimum(pp,87168);hi=np.maximum(pp,88392)
    Fc=np.clip(F,lo,hi);qq=np.where(few,.05,.7*practice)
    zone=np.where(f.zone=='forced',pp,np.where(f.zone=='edge',Fc,np.clip(qq*87300+(1-qq)*pp,lo,hi)))
    band=np.where((pp>57600)&(pp<=86400),np.clip(.708*87300+.292*pp,lo,hi),Fc)
    gy=(Fc-f[COL].to_numpy())**2-(zone-f[COL].to_numpy())**2
    sg=pd.Series(gy).groupby(f.day.to_numpy()).sum().to_numpy();rng=np.random.default_rng(7)
    draws=np.array([sg[rng.integers(0,len(sg),len(sg))].sum() for _ in range(2000)])/344419
    rec['C45_source_replay']={'rows':len(f),'open_rows':int((f.zone=='open').sum()),'G_uniform':float(gy.sum()/344419),
       'interval_uniform':np.percentile(draws,[2.5,97.5]).tolist(),
       'zone_minus_band_uniform':float((((band-f[COL].to_numpy())**2-(zone-f[COL].to_numpy())**2).sum()/344419)),
       'note':'Source uses uniform 1/344419, day bootstrap seed 7, R=87300, band .708, and F clipped to two points; it is not the GS cell-weighted C20 read.'}
    assert round(rec['C45_source_replay']['G_uniform'])==1827
    # Matched readings: target source control on the current stack; local changes use no new fit.
    bet25=compose(b,**config(route='R4mk',bet=True,c33=True));c45local=bet25.copy()
    openfar=(f.zone=='open').to_numpy() & (pp>57600)
    unknown=f.FLIGHT_mvt.isna().to_numpy()|(f.pre_n.fillna(0).to_numpy()<10)
    proposed=openfar & ((pp>86400)|unknown)
    # The held point must use only the ten training months.
    replay_path=EXP/'results/noplan-replay-inputs.json';inputs[str(replay_path)]=sha(replay_path)
    held_R=json.loads(replay_path.read_text())['training_tables']['r_mid']
    c45local[pos[proposed]]=.5*held_R+.5*pp[proposed]
    fullzone25=base25.copy();fullzone25[pos]=zone
    rec['C45_matched']={'target_bet_over_stack':score(b,base25,bet25,y),
       'levers_rule_over_target_bet':score(b,bet25,c45local,y),
       'levers_rule_over_stack':score(b,base25,c45local,y),
       'source_zone_over_stack':score(b,base25,fullzone25,y),
       'source_zone_over_target_bet':score(b,bet25,fullzone25,y),
       'held_rollover_point':held_R,
       'warning':'Levers rule is selected after these months were inspected; no independent confirmation set. Its unknown-airline adjustment has no held qualifying row. The held rule is a cohort proxy; the final 2026 rule is restricted to the fixed IDs in BET-DECISION.md.'}
    # All five 2026 open rows with material stake; R harmonized to target's fixed point.
    raw26=read(ROOT/'data/raw/ranking.parquet');d26=far_rows(raw26)
    names=['TAY9LA','WMTT3508','SVII3006','ITYTY672','NOSOS431']
    o=d26[d26.LOBT_flt.isna()&(d26.zone=='open')&(d26.ms>10.6*3600)&d26.FLIGHT_mvt.isin(names)].copy()
    assert len(o)==5 and set(o.FLIGHT_mvt)==set(names)
    o['order']=o.FLIGHT_mvt.map(dict(zip(names,range(5))));o=o.sort_values('order').reset_index(drop=True)
    po=cidx.get_indexer(o[ID]);bs=base26[po];target=compose(c,**config(route='R4mk',bet=True,c33=True))[po]
    chosen=target.copy();chosen[o.FLIGHT_mvt.isin(['SVII3006','NOSOS431'])]=.5*(R+o.ms[o.FLIGHT_mvt.isin(['SVII3006','NOSOS431'])])
    variants={'target_bet':target,'levers_modified':chosen,
       'fresh_named_proposal':np.array([.35,.35,.25,17/24,.5])*R+(1-np.array([.35,.35,.25,17/24,.5]))*o.ms.to_numpy()}
    rec['C45_outcomes']={}
    for name,vals in variants.items():
        per=one_table(o,bs,vals);per.to_csv(EXP/f'results/C45-{name}-rows.csv',index=False)
        rec['C45_outcomes'][name]={'rows':per.to_dict('records'),'all_32':all_outcomes(o,bs,vals,'C45-'+name)}
    inc=one_table(o,target,chosen);inc.to_csv(EXP/'results/C45-change-from-C20.csv',index=False)
    rec['C45_change_from_C20']=inc.to_dict('records')
    rates=np.array([1/6,1/6,.5,17/24,.5])
    selected=one_table(o,bs,chosen);control=one_table(o,bs,target)
    def expected(tab):return float(((1-rates)*tab.G_if_P+rates*tab.G_if_R).sum())
    rec['C45_selected_scenario']={'rates_R':dict(zip(names,rates)),
        'chosen_expected_G':expected(selected),'control_at_same_rates_G':expected(control),
        'increment_over_control_G':expected(inc)}
    assert np.isclose(expected(selected)-expected(control),expected(inc))
    # C39 source parity and recomposition on the assembled route.
    d25=P.rows(t,True);table,rate=P.status_table(d25[d25.month.isin(P.R1.TEN)])
    f39=d25[d25[ID].isin(bidx)].copy().reset_index(drop=True);p39=bidx.get_indexer(f39[ID])
    local,st=P.apply(f39,v47[p39],table,rate,True);pv=v47.copy();pv[p39]=local
    rec['C39_fold_v47']=score(b,v47,pv,y)
    assert round(rec['C39_fold_v47']['G'],1)==44.2
    pre25=compose(b,**config(route='R4mk',c33=False))
    local,st=P.apply(f39,pre25[p39],table,rate,True);pl=pre25.copy();pl[p39]=local;pl=clip(pl,b.lo.to_numpy(),b.hi.to_numpy())
    rec['C39_fold_stack']=score(b,base25,pl,y)
    rec['C39_fold_counts']={'THY':int((f39.pre=='THY').sum()),'new':int((st=='new').sum())}
    s39=read(ROOT/'experiments/exp-167-target/candidates/practice-arm-r1-002.parquet')
    table12,rate12=P.status_table(d25);p26=cidx.get_indexer(s39[ID])
    pre26=compose(c,**config(route='R4mk',c33=False));rp,st=P.apply(s39,pre26[p26],table12,rate12,True)
    old,oldst=P.apply(s39,s39.base.to_numpy(),table12,rate12,True)
    assert np.array_equal(old,s39.candidate.to_numpy())
    post=pre26.copy();post[p26]=rp;final=clip(post,c.lo.to_numpy(),c.hi.to_numpy())
    # Same point surrogate as target: AOBT_3 taxi time plus 66 s, or 1018 s without AOBT_3.
    sc=json.loads((ROOT/'experiments/exp-167-target/results/practice-arm-r1-001.json').read_text())
    xx=P.rows(raw26,False).set_index(ID).exot.reindex(s39[ID]).to_numpy()
    real=np.where(np.isfinite(xx),xx+sc['rows_2026']['real_point_offset_s'],sc['rows_2026']['real_point_without_aobt_s'])
    gp,gr,be=outcomes(base26[p26],final[p26],s39.point.to_numpy(),real)
    z=s39[[ID,'pre','month','status']].copy();z['base']=base26[p26];z['candidate']=final[p26];z['point']=s39.point.to_numpy();z['real_surrogate']=real;z['G_if_P']=gp;z['G_if_real']=gr
    z.to_parquet(EXP/'results/C39-outcome-rows.parquet',index=False)
    groups=[]
    for (pre,month,status),g in z.groupby(['pre','month','status']):
        a=float(g.G_if_P.sum());rr=float(g.G_if_real.sum());den=a-rr
        phrate=1689/1699 if status=='confirmed' else rate12 if status=='new' else 0
        groups.append({'pre':pre,'month':int(month),'status':status,'rows':len(g),'G_if_P':a,'G_if_real_surrogate':rr,
                       'break_even_placeholder_rate':-rr/den if den else None,'rate_used':phrate,'expected_G':phrate*a+(1-phrate)*rr,
                       'G_at_THY_rate_236of238':236/238*a+2/238*rr if pre=='THY' else None})
    rec['C39_outcomes']={'groups':groups,'total_expected_G':sum(v['expected_G'] for v in groups),
       'clip_changed_rows':int((post!=final).sum()),'clip_changed_practice_rows':int((post[p26]!=final[p26]).sum()),'real_point_note':'Scenario surrogate only: MVT-AOBT_3+66 s, else 1018 s. It is not an observed 2026 label.',
       'rate_new':rate12,'parity_source_rows_max_abs':float(np.max(np.abs(old-s39.candidate.to_numpy())))}
    # Exact overlaps in ranking space.
    m20=c.bet.to_numpy(bool);m45=np.zeros(len(c),bool);m45[po]=True;m39=final!=base26
    head=c.rule.to_numpy(bool);ext=c.extension.to_numpy(bool)
    no_route=compose(c,**config(noplan='none'));with_route=compose(c,**config(route='R4mk',noplan='none'));route=with_route!=no_route
    preclip=compose(c,**config(route='R4mk'));clipmask=base26!=preclip
    masks={'C20b':m20,'C45_open_five':m45,'C39b':m39,'HB1':head,'extension':ext,'R4mk':route,'C33':clipmask,'no_plan':c.noplan.to_numpy(bool)}
    rec['overlap']={a+' & '+bb:int((ma&mb).sum()) for (a,ma),(bb,mb) in itertools.combinations(masks.items(),2)}
    rec['C39_base_changed_from_source']=int((base26[p26]!=s39.base.to_numpy()).sum())
    rec['C45_rates_2025']={'open_all':'10/19 includes one joined placeholder','open_noplan':'10/18',
        'open_noplan_14to16':'1/6','open_noplan_over16':'9/12','unknown_few':'0/1; no-plan open-zone subset; tiny sample',
        'few_or_high_default':'1/4; no-plan open-zone subset includes CHH rollover',
        'checked_counts':'bet-checks.json independently recomputes these denominators'}
    rec['sources_sha256']=inputs
    rec['source_code_sha256']={p.name:sha(p) for p in [Path(__file__),ROOT/'experiments/exp-167-target/code/practice_arm_r1.py',ROOT/'experiments/exp-168-fresh-look/code/u4_far_fold.py']}
    (EXP/'results/bet-arms.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
    for k in ['C45_source_replay','C45_matched','C39_fold_v47','C39_fold_stack','C39_fold_counts','C39_outcomes','overlap']:
        print(k,json.dumps(rec[k],default=str),flush=True)
    print('DONE',flush=True)

if __name__=='__main__':main()
