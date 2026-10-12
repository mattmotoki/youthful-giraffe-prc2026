"""Unit 008: check one fit's actual prepared arrays and row contract; never load a held label.

Unit 007's check_inputs.py (sha 7193c6a1) for the full arm, with the months and scored rows of --fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from io_utils import W, ID, sha, artifact, write, lane

ALL = [f'2025-{i:02d}' for i in range(1, 13)]
RANK = ['2026-01', '2026-07']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',required=True);a=ap.parse_args()
    lane(); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    inp=W/'inputs/mlp-008'/a.fit; result=W/'results/mlp-008'
    m=json.loads((inp/'INPUTS.json').read_text())
    pp=json.loads((inp/'preprocessing.json').read_text())
    ref=json.loads((W/'inputs/mlp-007/INPUTS.json').read_text())
    checks=[]
    def check(name,ok,**facts):
        if not bool(ok): raise RuntimeError(f'{name}: {facts}')
        checks.append({'name':name,'state':'pass',**facts})
    check('fit name',m['fit']==a.fit)
    check('spec pin',sha(result/'SPEC-008.md')==m['spec_sha256'])
    check('input build check passed',json.loads((result/f'CHECK-INPUTS-{a.fit}.json').read_text())['manifest_sha256']==sha(inp/'INPUTS.json'))
    for k,v in m['artifacts'].items():
        check('artifact pin '+k,sha(v['path'])==v['sha256'])
    if a.fit=='final':
        train_months,held_months=ALL,RANK
    elif a.fit=='jjrank':
        train_months,held_months=[x for x in ALL if x not in ('2025-01','2025-07')],['2025-01','2025-07',*RANK]
    else:
        mm='2025-'+a.fit[-2:]
        train_months,held_months=[x for x in ALL if x not in ('2025-01','2025-07',mm)],[mm]
    check('training months',m['train_months']==train_months,months=m['train_months'])
    check('held months',m['held_months']==held_months,months=m['held_months'])
    check('unit 007 full feature order',m['features']['full']['numeric']==ref['features']['full']['numeric']==pp['numeric']
          and m['features']['full']['categorical']==ref['features']['full']['categorical']==pp['categorical'])
    nn,nc=len(pp['numeric']),len(pp['categorical'])
    check('210 numeric and 18 categorical',nn==210 and nc==18)
    check('numeric preprocessing widths',all(len(pp[k])==nn for k in ['median','mean','std']))
    check('finite preprocessing and scale floor',all(np.isfinite(pp[k]).all() for k in ['median','mean','std']) and min(pp['std'])>=np.float32(1e-6))
    check('category sizes',pp['category_sizes']==[len(pp['vocab'][c])+1 for c in pp['categorical']])
    meta={}
    unseen={}
    for p in ['train','held']:
        meta[p]=pd.read_parquet(inp/f'{p}-metadata.parquet')
        rows=m[p+'_rows']; x=meta[p]
        check(p+' metadata contract',len(x)==rows and list(x)==[ID,'month','utc_day','airport','anchor'] and x[ID].is_unique and np.isfinite(x.anchor).all())
        check(p+' integer IDs',np.array_equal(x[ID].to_numpy(),x[ID].to_numpy('int64')))
        for kind,width,dtype in [('num',nn,'float32'),('missing',nn,'bool'),('cat',nc,'int64')]:
            arr=np.load(inp/f'{p}-{kind}.npy',mmap_mode='r',allow_pickle=False)
            check(p+' '+kind+' shape/type',arr.shape==(rows,width) and str(arr.dtype)==dtype,shape=list(arr.shape))
            if kind=='num':
                check(p+' finite clipped numeric',all(np.isfinite(arr[i:i+8192]).all() and np.abs(arr[i:i+8192]).max()<=20 for i in range(0,rows,8192)))
            if kind=='cat':
                for j,c in enumerate(pp['categorical']):
                    check(p+' category range '+c,int(arr[:,j].min())>=0 and int(arr[:,j].max())<pp['category_sizes'][j])
                if p=='held':
                    unseen={c:float((arr[:,j]==0).mean()) for j,c in enumerate(pp['categorical'])}
                    month_code=arr[:,pp['categorical'].index('emb_month')]
                    if a.fit=='final':
                        check('held month embedding seen in training (final)',(month_code>0).all())
                    else:
                        check('held month embedding unseen',not month_code.any())
    tr,held=meta['train'],meta['held']
    check('actual training months',sorted(set(tr.month))==train_months)
    check('actual held months',sorted(set(held.month))==held_months)
    check('actual training buffers',not tr.utc_day.astype(str).isin(m['split']['excluded']).any())
    check('train held IDs disjoint',not tr[ID].isin(held[ID]).any())
    target=np.load(inp/'train-target.npy',mmap_mode='r',allow_pickle=False)
    check('target row type',target.shape==(len(tr),) and str(target.dtype)=='float32')
    check('target finite',np.isfinite(target).all())
    if a.fit!='final':
        check('no January or July 2025 training row',not tr.month.isin(['2025-01','2025-07']).any())
    rec={'state':'pass','fit':a.fit,'n_checks':len(checks),'checks':checks,'manifest_sha256':sha(inp/'INPUTS.json'),
         'code':artifact(Path(__file__)),'train_rows':len(tr),'held_rows':len(held),
         'held_rows_by_month':held.groupby('month').size().to_dict(),'held_unknown_category_share':unseen,
         'A3_hit_share':m['A3_hit_share'],'no_held_labels_loaded':True}
    write(result/f'CHECK-ARRAYS-{a.fit}.json',rec)
    print(json.dumps({k:rec[k] for k in ['state','fit','n_checks','train_rows','held_rows','held_rows_by_month','A3_hit_share']}))

if __name__=='__main__':main()
