"""Check the actual prepared arrays and row contract; never load held labels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from io_utils import W, ID, sha, artifact, write, lane

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='CHECK-ARRAYS.json');args=ap.parse_args()
    lane(); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    inp=W/'inputs/mlp-007'; result=W/'results/mlp-007'
    m=json.loads((inp/'INPUTS.json').read_text())
    pp=json.loads((inp/'preprocessing.json').read_text())
    checks=[]
    def check(name,ok,**facts):
        if not bool(ok): raise RuntimeError(f'{name}: {facts}')
        checks.append({'name':name,'state':'pass',**facts})
    check('spec pin',sha(result/'SPEC-007.md')==m['spec_sha256'])
    for k,v in m['artifacts'].items():
        check('artifact pin '+k,sha(v['path'])==v['sha256'])
    check('ten training months',set(m['train_months'])=={f'2025-{i:02d}' for i in range(1,13) if i not in [1,7]})
    nn,nc=len(pp['numeric']),len(pp['categorical'])
    check('feature ordering',m['features']['full']['numeric']==pp['numeric'] and m['features']['full']['categorical']==pp['categorical'])
    check('numeric preprocessing widths',all(len(pp[k])==nn for k in ['median','mean','std']))
    check('finite preprocessing and scale floor',all(np.isfinite(pp[k]).all() for k in ['median','mean','std']) and min(pp['std'])>=np.float32(1e-6))
    check('category sizes',pp['category_sizes']==[len(pp['vocab'][c])+1 for c in pp['categorical']])
    meta={}
    for p in ['train','held']:
        meta[p]=pd.read_parquet(inp/f'{p}-metadata.parquet')
        rows=m[p+'_rows']; x=meta[p]
        check(p+' metadata contract',len(x)==rows and list(x)==[ID,'month','utc_day','airport','anchor'] and x[ID].is_unique and np.isfinite(x.anchor).all())
        check(p+' integer IDs',np.array_equal(x[ID].to_numpy(),x[ID].to_numpy('int64')))
        for kind,width,dtype in [('num',nn,'float32'),('missing',nn,'bool'),('cat',nc,'int64')]:
            a=np.load(inp/f'{p}-{kind}.npy',mmap_mode='r',allow_pickle=False)
            check(p+' '+kind+' shape/type',a.shape==(rows,width) and str(a.dtype)==dtype,shape=list(a.shape))
            if kind=='num':
                check(p+' finite clipped numeric',all(np.isfinite(a[i:i+8192]).all() and np.abs(a[i:i+8192]).max()<=20 for i in range(0,rows,8192)))
            if kind=='cat':
                for j,c in enumerate(pp['categorical']):
                    check(p+' category range '+c,int(a[:,j].min())>=0 and int(a[:,j].max())<pp['category_sizes'][j])
                if p=='held':
                    check('held month embedding unseen',not a[:,pp['categorical'].index('emb_month')].any())
    tr,held=meta['train'],meta['held']
    check('actual training months',set(tr.month)==set(m['train_months']))
    check('actual training buffers',not tr.utc_day.astype(str).isin(m['split']['excluded']).any())
    check('held months',set(held.month)=={'2025-01','2025-07'})
    check('train held IDs disjoint',not tr[ID].isin(held[ID]).any())
    target=np.load(inp/'train-target.npy',mmap_mode='r',allow_pickle=False)
    check('target row type',target.shape==(len(tr),) and str(target.dtype)=='float32')
    check('target finite',np.isfinite(target).all())
    sys.path.insert(0,str(W/'code/rho-source'))
    from prc import gscore as GS
    frozen=GS.load_rows(); frozen=frozen[frozen.month.isin([1,7])]
    check('frozen full fold count',len(frozen)==344419 and frozen[ID].is_unique)
    check('exact fitted-scope fold IDs',set(held[ID])==set(frozen.loc[frozen.cls.ne('noac'),ID]))
    check('no training full-fold IDs',not tr[ID].isin(frozen[ID]).any())
    small=m['features']['physical']
    check('physical numeric indices',[pp['numeric'][j] for j in small['numeric_indices']]==small['numeric'])
    check('physical category indices',[pp['categorical'][j] for j in small['categorical_indices']]==small['categorical'])
    check('physical seven raw categories',small['categorical']==['emb_airport','emb_stand','emb_runway','emb_operator','emb_type','emb_hour','emb_month'])
    check('physical no target encodings',not any(c.startswith('te_') for c in small['numeric']))
    check('physical calendar cycle',set(['hour','dow','dom','doy','sin_h','cos_h'])<=set(small['numeric']))
    check('physical LIRF clocks and traffic',set(['lirf_mvt_anchor_250','lirf_al_gap','lirf_day_n_dep','lirf_stand_in_block_gap'])<=set(small['numeric']))
    check('physical no LIRF identifier/admin',not set(small['numeric']) & {'lirf_flight_len','lirf_ades_is_adep','lirf_stand_num','lirf_stand_retired'})
    rec={'state':'pass','n_checks':len(checks),'checks':checks,'manifest_sha256':sha(inp/'INPUTS.json'),
         'code':artifact(Path(__file__)),'train_rows':len(tr),'held_rows':len(held),
         'full_fold_rows':len(frozen),'F_pass_through':len(frozen)-len(held),'no_held_labels_loaded':True}
    write(result/args.out,rec)
    print(json.dumps({k:rec[k] for k in ['state','n_checks','train_rows','held_rows','full_fold_rows','F_pass_through']}))

if __name__=='__main__':main()
