"""Matched CPU seed bags; fixed train-owned epochs, no held labels."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn

import fit as F
from model import Model

LANES = ({4,5,6,7,68,69,70,71}, {60,61,62,63,124,125,126,127})
RECIPES = {'mixed64': (64,2,128), 'mixed128': (128,3,256)}


class WideModel(Model):
    def __init__(self, numeric, vocab):
        nn.Module.__init__(self)
        self.arm = 'sequence'
        self.embeds = nn.ModuleList([nn.Embedding(v,8) for v in vocab])
        self.numeric = nn.Linear(numeric,128)
        self.category = nn.Linear(8*len(vocab),128)
        self.elapsed = nn.Linear(1,128)
        layer = nn.TransformerEncoderLayer(128,4,256,dropout=.1,batch_first=True)
        self.context = nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        self.head = nn.Linear(128,1)


def make(data, recipe):
    if recipe == 'mixed64': return Model(data.n_numeric,data.vocab_sizes,'sequence')
    if recipe == 'mixed128': return WideModel(data.n_numeric,data.vocab_sizes)
    raise ValueError('Unknown recipe')


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def save(path, model, data, recipe, value):
    torch.save(dict(state_dict=model.state_dict(), recipe=recipe, seed=value,
                    n_numeric=data.n_numeric, vocab_sizes=data.vocab_sizes,
                    encoder_stage='full'),path)


def load(path, data, recipe, value):
    r=torch.load(path,map_location='cpu',weights_only=True)
    if (r['recipe'],r['seed'],r['encoder_stage'],r['n_numeric'],r['vocab_sizes']) != (
            recipe,value,'full',data.n_numeric,data.vocab_sizes):
        raise ValueError('Model contract differs')
    model=make(data,recipe); model.load_state_dict(r['state_dict']); return model


def inputs(root, checks):
    c=json.loads(checks.read_text()); m=json.loads((root/'MANIFEST.json').read_text())
    if c['status']!='PASS' or c['manifest_sha256']!=F.sha(root/'MANIFEST.json'):
        raise ValueError('Frozen input checks differ')
    if F.sha(root/'MANIFEST.json')!='5f44ada3df9f5a5335b29bc60f72747e60038214d25e44334118f2e8dc8501f0':
        raise ValueError('Input manifest is not frozen001')
    for n,h in m['files'].items():
        if F.sha(root/n)!=h: raise ValueError('Input hash differs: '+n)
    ix=np.load(root/'train-indices.npy',allow_pickle=False)
    y=np.load(root/'train-labels.npy',allow_pickle=False)
    held=np.load(root/'held-indices.npy',allow_pickle=False)
    ids=np.load(root/'held-ids.npy',allow_pickle=False)
    if len(ix)!=1723505 or len(held)!=344419 or len(ids)!=len(held):
        raise ValueError('Wrong split counts')
    if len(np.unique(ix))!=len(ix) or len(np.intersect1d(ix,held)):
        raise ValueError('Train/held overlap or duplicate losses')
    if len(np.unique(ids))!=len(ids) or len(y)!=len(ix) or not np.isfinite(y).all():
        raise ValueError('Bad labels or held IDs')
    months=pd.read_parquet(root/'events.parquet',columns=['MVT_TIME_UTC_mvt'],use_threads=False)
    months=pd.to_datetime(months.MVT_TIME_UTC_mvt,utc=True)
    if months.iloc[ix].dt.month.isin([1,7]).any() or not months.iloc[ix].dt.year.eq(2025).all():
        raise ValueError('Loss rows include scored month or other year')
    if not months.iloc[held].dt.month.isin([1,7]).all():
        raise ValueError('Held rows differ from Jan/Jul')
    return ix,y,held,ids,m


def smoke(model,data,ix,y,value,out,recipe):
    seed(value); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    take=np.random.default_rng(value).choice(len(ix),size=10*F.BATCH,replace=False)
    times=[]
    for k in range(10):
        t=time.monotonic(); rows=take[k*F.BATCH:(k+1)*F.BATCH]
        result=F.run_epoch(model,data,ix[rows],y[rows],torch.device('cpu'),opt,np.random.default_rng(value+k))
        if result['rows']!=F.BATCH or not np.isfinite(result['huber']):
            raise ValueError('Invalid benchmark loss')
        times.append(time.monotonic()-t)
    model.eval(); n,c,e,p,anchor=data.batch(ix[take[:F.BATCH]],torch.device('cpu'))
    mask=p.clone(); mask[:,[0,16]]=True
    nn_=n.clone();cc=c.clone();ee=e.clone()
    nn_[mask]=1000; cc[mask]=0; ee[mask]=-1000
    with torch.no_grad():
        clean=model(n,c,e,mask); poisoned=model(nn_,cc,ee,mask)
    torch.testing.assert_close(clean,poisoned,rtol=0,atol=0)
    save(out/'model.pt',model,data,recipe,value)
    replay=load(out/'model.pt',data,recipe,value).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(n,c,e,p),replay(n,c,e,p),rtol=0,atol=0)
    median=float(np.median(times[2:])); full_batches=(len(ix)+F.BATCH-1)//F.BATCH
    return dict(status='PASS',mode='training-row benchmark, no held quality read',
                train_batches=10,median_seconds_per_batch=median,
                estimated_12_epoch_fit_seconds=12*full_batches*median,
                parameters=sum(q.numel() for q in model.parameters()),
                padding_invariant=True,model_reload_exact=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--checks',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--recipe',choices=RECIPES,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--pins',type=Path)
    a=ap.parse_args()
    if a.out.exists(): raise ValueError('Write-once output exists')
    if os.sched_getaffinity(0) not in LANES or os.getpriority(os.PRIO_PROCESS,0)!=19:
        raise ValueError('CPU lane/nice differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CUDA must be hidden')
    resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
    torch.set_num_threads(6);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    plan=json.loads(a.plan.read_text()); cfg=plan['recipes'][a.recipe]
    if (cfg['width'],cfg['layers'],cfg['feedforward'])!=RECIPES[a.recipe] or cfg['heads']!=4 or cfg['dropout']!=.1:
        raise ValueError('Recipe differs from frozen plan')
    if a.seed not in plan['seeds'] or plan['fixed_epochs']!=12: raise ValueError('Seed/epochs differ')
    code=Path(__file__).resolve().parent
    pins={n:F.sha(code/n) for n in ['fit_bag_033.py','fit.py','model.py']}
    plan_sha=F.sha(a.plan); started=time.monotonic()
    run_pins_sha=None
    if not a.benchmark:
        if a.pins is None:raise ValueError('Full fit requires reviewed frozen pins')
        frozen=json.loads(a.pins.read_text());run_pins_sha=F.sha(a.pins)
        if frozen['plan_sha256']!=plan_sha:raise ValueError('Plan pin differs')
        for n,h in pins.items():
            if frozen['source_sha256'][n]!=h:raise ValueError('Reviewed fit source differs: '+n)
        review=code.parent/'REVIEW-BAG-SOURCE-033.json'
        if F.sha(review)!=frozen['review_sha256'] or json.loads(review.read_text())['verdict']!='PASS':
            raise ValueError('Source review missing or changed')
        old=a.inputs.parent/'sequence-g-kaggle-v2-001/sequence-g-001/sequence'
        if F.sha(old/'FIT.json')!=frozen['epoch_proof']['FIT_sha256'] or F.sha(old/'SELECTION.json')!=frozen['epoch_proof']['SELECTION_sha256']:
            raise ValueError('Fixed epoch proof changed')
        if json.loads((old/'FIT.json').read_text())['selected_epochs']!=12 or json.loads((old/'SELECTION.json').read_text())['selected_epochs']!=12:
            raise ValueError('Prior train-owned selection was not12')
    ix,y,held,ids,manifest=inputs(a.inputs,a.checks)
    a.out.mkdir(parents=True); data=F.Arrays(a.inputs,'full','sequence');seed(a.seed)
    model=make(data,a.recipe)
    common=dict(recipe=a.recipe,seed=a.seed,epochs=12,manifest_sha256=F.sha(a.inputs/'MANIFEST.json'),
                checks_sha256=F.sha(a.checks),plan_sha256=plan_sha,run_pins_sha256=run_pins_sha,source_sha256=pins,
                train_labels=len(ix),train_months=manifest['summary']['train_rows_by_UTC_month'],
                held_labels_in_payload=False,held_labels_used_for_selection=False,
                device='cpu',threads=6,nice=19,affinity=sorted(os.sched_getaffinity(0)))
    if a.benchmark:
        rec={**common,**smoke(model,data,ix,y,a.seed,a.out,a.recipe)}
        target=a.out/'BENCHMARK.json'
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(a.seed);history=[]
        for epoch in range(1,13):
            row=dict(epoch=epoch,train=F.run_epoch(model,data,ix,y,torch.device('cpu'),opt,rng),
                     elapsed_seconds=time.monotonic()-started)
            if not np.isfinite(row['train']['huber']): raise ValueError('Non-finite train loss')
            history.append(row); print(json.dumps(row),flush=True)
        save(a.out/'model.pt',model,data,a.recipe,a.seed)
        pred=F.predict(model,data,held,torch.device('cpu'))
        replay=F.predict(load(a.out/'model.pt',data,a.recipe,a.seed),data,held[:1024],torch.device('cpu'))
        np.testing.assert_array_equal(pred[:1024],replay)
        pd.DataFrame({'MVT_ID_mvt':ids,'pred_Gmixed':pred}).to_parquet(a.out/'held.parquet',index=False)
        rec={**common,'status':'completed','full_history':history,'held_rows':len(ids),
             'model_reload_exact_first_1024':True,'loss':'SmoothL1 beta2 in300s; AdamW.001/.0001; clip1',
             'floor':60,'upper_cap':None,'fixed_epoch_source':'Gmixed028 train-only selection',
             'files':{p.name:F.sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}}
        target=a.out/'FIT.json'
    for n,h in pins.items():
        if F.sha(code/n)!=h: raise ValueError('Fit source changed: '+n)
    if F.sha(a.plan)!=plan_sha: raise ValueError('Frozen plan changed')
    if a.pins is not None and F.sha(a.pins)!=run_pins_sha:raise ValueError('Frozen run pins changed')
    rec.update(seconds=time.monotonic()-started,
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch=torch.__version__)
    target.write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
