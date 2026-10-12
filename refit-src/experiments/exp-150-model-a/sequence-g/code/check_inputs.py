"""Independent payload checks before any lane G model sees data."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa

EXP=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(EXP/'code'))
import r4_admission_runtime as R
import prepare as P

def main():
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('Check record exists; write-once')
    rec=json.loads((a.inputs/'MANIFEST.json').read_text())
    for name,want in rec['files'].items():
        if R.sha(a.inputs/name)!=want:raise ValueError('Input hash differs: '+name)
    if {'TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'} & set(rec['input_columns']):raise ValueError('Label in whitelist')
    f=pd.read_parquet(a.inputs/'events.parquet',use_threads=False);ids=f[P.ID].to_numpy('int64')
    if len(np.unique(ids))!=len(ids):raise ValueError('Event keys repeat')
    ts=pd.to_datetime(f[P.TIME],utc=True).astype('datetime64[ns, UTC]')
    if not f.day.eq(ts.dt.strftime('%Y-%m-%d')).all():raise ValueError('Wrong day')
    # Derive airport and runway from source fields, rather than trust builder groups.
    raw=[]
    for path,want in rec['sources_sha256'].items():
        p=Path(path)
        if R.sha(p)!=want:raise ValueError('Raw source hash differs')
        raw.append(pd.read_parquet(p,columns=[P.ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','RUNWAY_mvt',P.TIME],use_threads=False))
    r=pd.concat(raw,ignore_index=True).drop_duplicates(P.ID).set_index(P.ID).reindex(ids)
    airport=r.ADEP_mvt.where(r.PHASE_mvt.eq('DEP'),r.ADES_mvt).astype(str).str.strip().str.upper()
    runway=r.RUNWAY_mvt.astype('string').str.strip().str.upper()
    missing=runway.isna() | runway.eq('').fillna(False);runway=runway.mask(missing,'<MISSING>')
    if not np.array_equal(airport.to_numpy(),f.airport.to_numpy()):raise ValueError('Arrival/departure airport differs')
    if not np.array_equal(runway.to_numpy(),f.runway.to_numpy()):raise ValueError('Runway differs')
    if not np.array_equal(pd.to_datetime(r[P.TIME],utc=True).astype('datetime64[ns, UTC]').astype('int64').to_numpy(),ts.astype('int64').to_numpy()):raise ValueError('Source event time differs')
    def load(n):return np.load(a.inputs/(n+'.npy'),mmap_mode='r',allow_pickle=False)
    centres,nb,elapsed=load('departure-centres'),load('neighbours'),load('elapsed-seconds')
    if nb.shape!=(len(centres),17) or elapsed.shape!=nb.shape:raise ValueError('Window shape differs')
    if not np.array_equal(centres,np.flatnonzero(f.PHASE_mvt.eq('DEP'))):raise ValueError('Departure index differs')
    if not np.array_equal(nb[:,8],centres):raise ValueError('Central token is not departure')
    # Source-based key, with each unknown runway given its own key.
    key=airport.reset_index(drop=True)+'|'+f.day+'|'+runway.reset_index(drop=True)
    key=key.where(~missing.reset_index(drop=True),key+'|singleton|'+f[P.ID].astype(str))
    g=pd.factorize(key)[0];t=ts.astype('int64').to_numpy()/1e9
    for start in range(0,len(centres),50000):
        c=centres[start:start+50000];n=nb[start:start+50000];ok=n>=0;safe=np.maximum(n,0)
        if (n[ok]>=len(f)).any():raise ValueError('Neighbour index out of range')
        if ((g[safe]!=g[c,None]) & ok).any():raise ValueError('Neighbour crossed source group')
        d=t[safe]-t[c,None]
        if (np.abs(d[ok])>1800).any():raise ValueError('Neighbour beyond 30 minutes')
        if not np.array_equal(elapsed[start:start+50000],np.where(ok,d,0).astype('float32')):raise ValueError('Elapsed time differs')
        if not np.array_equal(n, np.where((c[:,None]+np.arange(-8,9)>=0) &
            (c[:,None]+np.arange(-8,9)<len(f)) &
            (g[np.clip(c[:,None]+np.arange(-8,9),0,len(f)-1)]==g[c,None]) &
            (np.abs(t[np.clip(c[:,None]+np.arange(-8,9),0,len(f)-1)]-t[c,None])<=1800),
            c[:,None]+np.arange(-8,9),-1).astype('int32')):raise ValueError('Window skips or adds a movement')
    st,_,_,_,_=R.load();cons=st.P.consumer('janjul')
    train=load('train-indices');y=load('train-labels');valid=load('train-development-valid')
    if len(train)!=len(y) or not np.isfinite(y).all() or len(np.unique(train))!=len(train):raise ValueError('Bad loss arrays')
    loss=f.iloc[train][[P.TIME,'PHASE_mvt']].copy();loss['target']=y
    cons.assert_training_frame(loss,'target',what='checked lane G loss payload')
    if not loss.PHASE_mvt.eq('DEP').all() or loss[P.TIME].dt.month.isin([1,7]).any():raise ValueError('Wrong loss ownership')
    allowed=(ts.dt.year.eq(2025) & ~ts.dt.month.isin([1,7]) & ~f.day.isin(cons.excluded)).to_numpy()
    if not np.array_equal(allowed,f.encoder_train.to_numpy()):raise ValueError('Encoder ownership differs')
    from prc import gscore as GS
    want=GS.load_rows();want=want[want.month.isin([1,7])][P.ID].to_numpy('int64')
    held=load('held-indices');held_ids=load('held-ids')
    if not np.array_equal(held_ids,want) or not np.array_equal(ids[held],want):raise ValueError('Held order/IDs differ')
    if np.isin(held,train).any():raise ValueError('Held label in loss indices')
    if not np.array_equal(valid,f.development_valid.to_numpy()[train]):raise ValueError('Development owner differs')
    checks={'input_hashes':True,'source_group_boundaries':True,'short_mixed_order':True,
        'held_ids_and_order':True,'label_split_admission':True,'no_held_loss_indices':True}
    for stage,mask in [('development',allowed & ~f.development_valid.to_numpy()),('full',allowed)]:
        enc=json.loads((a.inputs/f'encoder-{stage}.json').read_text())
        if enc['fit_event_rows']!=int(mask.sum()):raise ValueError('Encoder fit count differs')
        num,cat=load('numeric-'+stage),load('category-'+stage)
        if num.shape[0]!=len(f) or cat.shape!=(len(f),len(P.CATS)):raise ValueError('Encoded shape differs')
        for start in range(0,len(f),50000):
            if not np.isfinite(num[start:start+50000]).all():raise ValueError('Nonfinite feature')
        if (cat<0).any():raise ValueError('Negative category')
    # Poison both labels and BLOCK times in a real mixed sample through the exact feature entry path.
    sample=pd.read_parquet(next(iter(rec['sources_sha256'])),columns=P.COLS,use_threads=False).iloc[:20000].copy()
    sample['TAXITIME_SEC_mvt']=0.;sample['BLOCK_TIME_UTC_mvt']=pd.NaT
    poison=sample.copy();poison.TAXITIME_SEC_mvt=1e12
    poison.BLOCK_TIME_UTC_mvt=pd.Timestamp('1901-01-01',tz='UTC')
    first,last=P.events(sample),P.events(poison)
    pd.testing.assert_frame_equal(first,last)
    for x,z in zip(P.neighbours(first),P.neighbours(last)):np.testing.assert_array_equal(x,z)
    x,an=P.numerics(first);z,bn=P.numerics(last)
    pd.testing.assert_frame_equal(x,z);np.testing.assert_array_equal(an,bn)
    checks['real_label_poison_same_features_and_indices']=True
    result={'status':'PASS','checks':checks,'events':len(f),'train_loss_rows':len(train),
        'held_rows':len(held),'manifest_sha256':R.sha(a.inputs/'MANIFEST.json'),
        'checker_sha256':R.sha(Path(__file__)),'no_neural_fit':True}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
