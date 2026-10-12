"""Two fixed MLP seeds; reads no held labels and writes no score."""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from io_utils import W, ID, sha, write, artifact, lane

class Net(nn.Module):
    def __init__(self, sizes, nnum):
        super().__init__()
        dims = [min(32, max(4, round(math.sqrt(s)))) for s in sizes]
        self.emb = nn.ModuleList([nn.Embedding(s, d, padding_idx=0) for s,d in zip(sizes,dims)])
        layers, prev = [], sum(dims)+2*nnum
        for h in [512,512,256]:
            layers.extend([nn.Linear(prev,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(.1)])
            prev = h
        layers.append(nn.Linear(prev,1))
        self.body = nn.Sequential(*layers)

    def forward(self, cat, num, missing):
        e = [emb(cat[:,i]) for i,emb in enumerate(self.emb)]
        return self.body(torch.cat([num,missing,*e],dim=1)).squeeze(1)

def smi():
    fields = 'index,uuid,memory.total,memory.used,memory.free,utilization.gpu'
    out = subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
    return [x.strip().split(', ') for x in out.strip().splitlines()]

def own_memory():
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    values = [float(x.split(',')[1]) for x in out.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
    return max(values,default=0.)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['full','physical'],required=True)
    a=ap.parse_args()
    cores=lane(); torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('GPU 1 only under the current grant')
    inp=W/'inputs/mlp-007';out=W/'out/mlp-007'/a.arm;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inp/'INPUTS.json').read_text())
    if sha(W/'results/mlp-007/SPEC-007.md') != manifest['spec_sha256']:
        raise RuntimeError('spec differs from prepared inputs')
    for r in manifest['artifacts'].values():
        if sha(r['path'])!=r['sha256']:
            raise RuntimeError('input artifact pin differs: '+r['path'])
    setting=manifest['features'][a.arm];ni=setting['numeric_indices'];ci=setting['categorical_indices']
    pp=json.loads((inp/'preprocessing.json').read_text())
    sizes=[pp['category_sizes'][i] for i in ci]
    arrays={p:{kind:np.load(inp/f'{p}-{kind}.npy',mmap_mode='r',allow_pickle=False) for kind in ['num','missing','cat']} for p in ['train','held']}
    target=np.load(inp/'train-target.npy',mmap_mode='r',allow_pickle=False)
    meta=pd.read_parquet(inp/'held-metadata.parquet')
    if 'y' in meta or len(meta)!=len(arrays['held']['num']):
        raise RuntimeError('held metadata differs or carries a label')
    n=len(target);batch=8192;epochs=40;preds=[];records=[]
    def tensors(p,idx):
        z=arrays[p]
        # CPU arrays and bounded GPU batches; no data-loader processes.
        num=np.array(z['num'][idx][:,ni],dtype='float32',copy=True)
        missing=np.array(z['missing'][idx][:,ni],dtype='float32',copy=True)
        cat=np.array(z['cat'][idx][:,ci],dtype='int64',copy=True)
        return tuple(torch.from_numpy(v).to('cuda:0') for v in [cat,num,missing])
    for seed in [701,702]:
        before=out/f'seed-{seed}-before.json'
        if before.exists():
            raise RuntimeError('fit already has a launch record')
        gpu=smi();g=[r for r in gpu if r[0]=='1']
        if len(g)!=1 or float(g[0][4])<8192:
            raise RuntimeError('valid first call: GPU 1 lacks the reserved eight GB')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA runtime unavailable; no CPU fit fallback')
        torch.cuda.set_per_process_memory_fraction((6*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
        torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark=False
        model=Net(sizes,len(ni)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        steps=math.ceil(n/batch)
        def rate(step):
            progress=step/steps
            if progress<5:
                return .1+.9*progress/5
            return .01+.99*.5*(1+math.cos(math.pi*min(1,(progress-5)/(epochs-5))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,rate)
        rec={'seed':seed,'arm':a.arm,'epochs':epochs,'batch':batch,'loss':'MSE','target_scale':1000,
             'hidden':[512,512,256],'normalization':'LayerNorm','activation':'SiLU','dropout':.1,
             'optimizer':'AdamW','peak_lr':.001,'weight_decay':.0001,'gradient_norm_cap':5,
             'category_sizes':sizes,'features':setting,'inputs_sha256':sha(inp/'INPUTS.json'),
             'spec_sha256':manifest['spec_sha256'],'code_sha256':sha(__file__),'split':manifest['split'],
             'pid':os.getpid(),'cores':cores,'nice':19,'torch':torch.__version__,'cuda':torch.version.cuda,
             'gpu_before':gpu,'GPU_allocator_cap_GiB':6,'actual_GPU_limit_MiB':7500,'train_rows':n,
             'held_rows':len(meta),'no_held_labels':True,'no_held_early_stop':True,'no_score_in_fit':True}
        write(before,rec)
        rng=np.random.default_rng(seed);t0=time.monotonic();step=0;curve=[]
        for epoch in range(epochs):
            model.train();order=rng.permutation(n);total=0.;seen=0
            for start in range(0,n,batch):
                idx=order[start:start+batch];c,x,m=tensors('train',idx)
                y=torch.from_numpy(np.array(target[idx],copy=True)).to('cuda:0')
                opt.zero_grad(set_to_none=True);p=model(c,x,m)
                loss=nn.functional.mse_loss(p,y)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step()
                total+=float(loss.detach())*len(idx);seen+=len(idx);step+=1
            mem=own_memory()
            if mem>7500:
                raise RuntimeError(f'own GPU memory {mem} MiB exceeds the grant')
            r={'epoch':epoch+1,'train_rmse_s':1000*math.sqrt(total/seen),'rows_seen':seen,
               'lr':opt.param_groups[0]['lr'],'GPU_MiB':mem,'elapsed_s':time.monotonic()-t0}
            if seen!=n:
                raise RuntimeError('an epoch omitted training rows')
            curve.append(r);write(out/f'seed-{seed}-progress.json',r,overwrite=True)
            print(json.dumps({'event':'epoch','seed':seed,**r}),flush=True)
        model.eval();z=np.empty(len(meta),dtype='float64')
        with torch.inference_mode():
            for start in range(0,len(meta),batch):
                sl=slice(start,start+batch);c,x,m=tensors('held',sl)
                z[sl]=model(c,x,m).cpu().numpy().astype('float64')*1000
        pred=np.maximum(60,meta.anchor.to_numpy('float64')+z)
        if not np.isfinite(pred).all():
            raise RuntimeError('nonfinite held prediction')
        p=out/f'seed-{seed}-held.parquet';f=meta[[ID,'month','utc_day','airport']].copy();f['prediction']=pred;f.to_parquet(p,index=False)
        model_path=out/f'seed-{seed}-model.pt'
        torch.save({'state_dict':model.cpu().state_dict(),'category_sizes':sizes,'numeric_count':len(ni)},model_path)
        rec.update(curve=curve,seconds=time.monotonic()-t0,prediction=artifact(p),model=artifact(model_path),
                   allocator_peak_MiB=torch.cuda.max_memory_reserved()/1024**2,actual_peak_MiB=max(r['GPU_MiB'] for r in curve))
        write(out/f'seed-{seed}.json',rec);records.append(artifact(out/f'seed-{seed}.json'));preds.append(pred)
        del model,opt,scheduler;torch.cuda.empty_cache()
    f=meta[[ID,'month','utc_day','airport']].copy();f['prediction']=np.mean(preds,axis=0)
    p=out/'mean-held.parquet';f.to_parquet(p,index=False)
    write(out/'FIT.json',{'arm':a.arm,'seeds':[701,702],'seed_records':records,'prediction':artifact(p),
                         'inputs_sha256':sha(inp/'INPUTS.json'),'code_sha256':sha(__file__),'no_score':True})
    print(json.dumps({'event':'fit-ready','arm':a.arm,'prediction':artifact(p)}),flush=True)

if __name__=='__main__':
    main()
