"""Training-only route practice rates and label-free U guards (SPEC-016)."""
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ID
ROOT=Path(__file__).resolve().parents[3]
EXPECTED={'WMT HESX','WMT LEIB','WMT LGKR','EXS EGBB','EXS EGPH','RYR LIRA','ABR LLBG'}
TIMES={'MVT_TIME_UTC_mvt':'mvt','SCHED_TIME_UTC_mvt':'sched','LOBT_flt':'lobt','AOBT_3_flt':'aobt'}


def raw_frame(pin):
    parts=[]
    for path in [*sorted((ROOT/'data/raw').glob('training_2025-*.parquet')),ROOT/'data/raw/ranking.parquet']:
        annual=path.name=='ranking.parquet'
        cols=[ID,'FLIGHT_mvt','ADES_mvt',*TIMES]
        if not annual:cols+=['BLOCK_TIME_UTC_mvt']
        d=pd.read_parquet(pin(path),columns=cols,filters=[('ADEP_mvt','==','LIRF'),('PHASE_mvt','==','DEP')])
        d['year']=2026 if annual else 2025
        for src,dst in {**TIMES,**({} if annual else {'BLOCK_TIME_UTC_mvt':'block'})}.items():
            # Match the source whole-second definition without a NaT sentinel.
            d[dst]=(d[src].dt.as_unit('us').astype('int64')//1000000).astype(float).where(d[src].notna())
            d.drop(columns=src,inplace=True)
        d['ph']=False if annual else (d.block-d.sched).abs().le(6)
        d['month']=pd.to_datetime(d.mvt,unit='s').dt.month
        d['present']=d.lobt.notna();d['ms']=d.mvt-d.sched
        d['late']=np.where(d.present,d.aobt-d.sched,d.mvt-((d.mvt+30)%60-30)-900-d.sched)
        d['airline']=d.FLIGHT_mvt.astype(str).str[:3]
        d['route']=d.airline+' '+d.ADES_mvt.astype(str)
        d['U']=d.present & (d.lobt-d.sched).abs().gt(3612)
        parts.append(d)
    out=pd.concat(parts,ignore_index=True).set_index(ID)
    assert out.index.is_unique
    return out


def add_key(bundle,raw,split):
    train=raw[(raw.year==2025)&(raw.late>=900)]
    if split=='fold':train=train[~train.month.isin([1,7])]
    rates=train.groupby('route').ph.agg(n='size',hits='sum',rate='mean')
    ar=train.groupby('airline').ph.mean()
    rates['airline_rate']=[ar[k.split()[0]] for k in rates.index]
    rates['selected']=(rates.rate>=.98)&(rates.n>=5)&(rates.airline_rate<.9)
    if split=='ranking':assert set(rates.index[rates.selected])==EXPECTED
    evalrows=raw[(raw.year==(2025 if split=='fold' else 2026))&raw.index.isin(bundle[ID])].copy()
    # The route-level kill uses every input row in the evaluation file, not only late rows.
    ucounts=evalrows.groupby('route').U.sum()
    d=evalrows.join(rates,on='route')
    d['U_count']=d.route.map(ucounts)
    d['late_serve']=(d.present&(d.late>=1200))|(~d.present&(d.ms>=3600))
    d['source_scope']=(d.rate>=.98)&(d.n>=5)&d.late_serve
    d['active']=d.source_scope&d.selected.fillna(False)&~d.U&(d.U_count<2)
    d['weight']=np.where(d.route=='WMT LGKR',.9,1.)
    d['point']=np.maximum(60.,d.ms)
    b=bundle.copy();a=d.reindex(b[ID]);idx=b.index
    b['c73']=a.active.fillna(False).to_numpy(bool)
    for dst,src in [('c73_U','U'),('c73_U_count','U_count'),('c73_rate','rate'),('c73_n','n'),('c73_airline_rate','airline_rate'),('c73_weight','weight'),('c73_point','point'),('c73_route','route')]:
        b[dst]=a[src].fillna(False).to_numpy(bool) if dst=='c73_U' else a[src].to_numpy()
    assert not (b.c73 & b.practice).any()
    selected=d[d.selected.fillna(False)]
    stats={'training_rows':len(train),'training_months':sorted(train.month.unique().tolist()),'selected_routes':rates[rates.selected].reset_index().to_dict('records'),
           'source_scope':int(d.source_scope.sum()),'selected_late_rows':int((selected.late_serve).sum()),'active_rows':int(b.c73.sum()),
           'U_excluded_late':int((selected.late_serve&selected.U).sum()),'route_U_counts':selected.groupby('route').U.sum().astype(int).to_dict(),
           'killed_routes':sorted(set(selected.route[selected.U_count>=2])),
           'no_plan_active':int((d.active&~d.present).sum()),'training_hold_hits':int(rates.loc[rates.selected,'hits'].sum()),'training_hold_n':int(rates.loc[rates.selected,'n'].sum())}
    return b,d,rates,stats
