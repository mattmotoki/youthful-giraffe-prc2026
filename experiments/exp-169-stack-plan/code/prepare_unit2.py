"""Prepare new bundles from target's saved tables and rows; no fit and no old overwrite."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from assembly import ID,COL,compose
from bets import P,ROOT,EXP

inputs={}
def read(path,**kw):
    path=Path(path);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path,**kw)
def record(path):
    path=Path(path);inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())

def add_practice(b,raw,table,rate):
    d=P.rows(raw,False);d=d[d[ID].isin(b[ID])].copy().set_index(ID)
    status=d.pre.map(table.status).fillna('new')
    share=d.pre.map(table.s)
    weight=np.select([status.eq('confirmed'),status.eq('mixed'),status.eq('new')],[1.,share.fillna(0),rate],0.)
    d['practice_status']=status;d['practice_weight']=weight;d['practice_point']=d.point
    b['practice']=b[ID].isin(d.index)
    for key in ['practice_status','practice_weight','practice_point']:
        b[key]=d[key].reindex(b[ID]).to_numpy()
    return {'eligible':len(d),'status':status.value_counts().to_dict(),'rate_new':rate}

def main():
    out=EXP/'inputs/unit2-ready';out.mkdir(exist_ok=True)
    rank=read(EXP/'inputs/bundle-ranking.parquet');fold=read(EXP/'inputs/bundle-fold.parquet')
    rows=read(ROOT/'experiments/exp-167-target/candidates/noplan-form-v3-cal-002.parquet').set_index(ID)
    noplan_record=record(ROOT/'experiments/exp-167-target/results/noplan-form-v3-cal-002.json')
    source=ROOT/noplan_record['entry']['file'];entry=read(source)
    assert inputs[str(source)]=='6f3b1ebdbee2b2f9a30ca8b9113dd856a92df07c5915c9b4b5912ce0f9cea807'
    n=rank.noplan.to_numpy(bool)
    assert set(rows.index)==set(rank.loc[n,ID]) and np.array_equal(rows.base.reindex(rank.loc[n,ID]),rank.loc[n,'np_base'])
    rank['np_v3cal']=rows.cal.reindex(rank[ID]).to_numpy()
    held_path=ROOT/'experiments/exp-167-target/results/noplan-form-v3-fold-001.parquet'
    held=read(held_path).set_index(ID)
    assert inputs[str(held_path)]=='5b6855d93e35b5ba03c2d33adfa1df88c6119c0b726b79814e2535b906726bff'
    nf=fold.noplan.to_numpy(bool)
    assert held.index.is_unique and set(held.index)==set(fold.loc[nf,ID])
    assert np.array_equal(held.base.reindex(fold.loc[nf,ID]),fold.loc[nf,'np_base'])
    assert np.array_equal(held.v2.reindex(fold.loc[nf,ID]),fold.loc[nf,'np_v2'])
    fold['np_v3cal']=held.cal.reindex(fold[ID]).to_numpy()
    hb=compose(rank,route='R4',form='HB1',extension=True,noplan='v3cal',c33=False)
    assert np.array_equal(hb,entry[COL].to_numpy()), 'Cannot rebuild target v3-cal HB1-x entry.'
    sc=record(ROOT/'experiments/exp-167-target/results/practice-arm-r1-001.json')
    cols=[ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt',COL,'LOBT_flt','AOBT_3_flt','AIRCRAFT_TYPE_mvt']
    parts=[]
    for m in ['01','07']:
        t=read(next((ROOT/'data/raw').glob('training_2025-'+m+'-*.parquet')),columns=cols)
        parts.append(t[((t.PHASE_mvt=='DEP')&(t.ADEP_mvt=='LIRF'))|((t.PHASE_mvt=='ARR')&(t.ADES_mvt=='LIRF'))])
    t25=pd.concat(parts,ignore_index=True)
    t26=read(ROOT/'data/raw/ranking.parquet',columns=cols)
    s25=sc['fold'];tab25=pd.DataFrame(s25['status_table_ten_months']).set_index('pre')
    s26=sc['rows_2026'];tab26=pd.DataFrame(s26['status_table_twelve_months']).set_index('pre')
    counts={'fold':add_practice(fold,t25,tab25,s25['rate_new_ten_months']),
            'ranking':add_practice(rank,t26,tab26,s26['rate_new_twelve_months'])}
    for name,b in [('fold',fold),('ranking',rank)]:
        dest=out/f'bundle-{name}.parquet';assert not dest.exists();b.to_parquet(dest,index=False)
    rec={'inputs':inputs,'practice':counts,'target_v3cal_parity_max_abs':0.0,
         'fold_v3cal_status':'Target ac3698b2 exported all 397 rows; IDs, base, v2 and hash match.',
         'bundles':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('bundle-*.parquet')}}
    (EXP/'results/unit2-inputs.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec['practice']),flush=True)
if __name__=='__main__':main()
