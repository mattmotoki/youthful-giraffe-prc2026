"""Unit 009: more fixed MLP seeds for one prepared unit 008 fit; reads no held labels and writes no score.

fit.py (sha 1f528e92) with three changes only: the seeds are an argument (default 703-706), the fit may
be `jjrank` (unit 007's training arrays; its held part is the jjrefit rows then the ranking rows), and
the training arrays stay on GPU 1, so a batch is a GPU index of the same rows in the same order
(`rng.permutation`, batch 8192) instead of a CPU copy from the memory-mapped files. The model, loss,
optimizer, schedule, epochs and the input values are unchanged; `--stop-after` (development only)
stops after that many epochs of the 40-epoch schedule. A development run (`--tag`) writes to
out/mlp-009-dev/<fit>-<tag>; with `--stop-after` it writes only the curve.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from io_utils import W, ID, sha, write, artifact, lane
from fit import Net, smi, own_memory

FITS = ('final', 'jjrank', 'jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',choices=FITS,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[703,704,705,706]);ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--tag',default='');a=ap.parse_args()
    cores=lane(); torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('GPU 1 only under the current grant')
    dev=bool(a.tag)
    if a.stop_after and not dev:
        raise RuntimeError('--stop-after is for a development run (--tag) only')
    inp=W/'inputs/mlp-008'/a.fit
    out=(W/'out/mlp-009-dev'/f'{a.fit}-{a.tag}') if dev else (W/'out/mlp-009'/a.fit)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'{out} is not empty; write-once')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((inp/'INPUTS.json').read_text())
    if manifest['fit']!=a.fit:
        raise RuntimeError('inputs belong to another fit')
    check=json.loads((W/'results/mlp-008'/f'CHECK-ARRAYS-{a.fit}.json').read_text())
    if check['state']!='pass' or check['manifest_sha256']!=sha(inp/'INPUTS.json'):
        raise RuntimeError('actual input check missing or for other inputs')
    for r in manifest['artifacts'].values():
        if sha(r['path'])!=r['sha256']:
            raise RuntimeError('input artifact pin differs: '+r['path'])
    setting=manifest['features']['full'];ni=setting['numeric_indices'];ci=setting['categorical_indices']
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
        num=np.array(z['num'][idx][:,ni],dtype='float32',copy=True)
        missing=np.array(z['missing'][idx][:,ni],dtype='float32',copy=True)
        cat=np.array(z['cat'][idx][:,ci],dtype='int64',copy=True)
        return tuple(torch.from_numpy(v).to('cuda:0') for v in [cat,num,missing])
    gpu=smi();g=[r for r in gpu if r[0]=='1']
    if len(g)!=1 or float(g[0][4])<8192:
        raise RuntimeError('valid first call: GPU 1 lacks the reserved eight GB')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA runtime unavailable; no CPU fit fallback')
    torch.cuda.set_per_process_memory_fraction((6*1024**3)/torch.cuda.get_device_properties(0).total_memory,0)
    # the training arrays on GPU 1: the same values the CPU path copies (bool missing becomes 0/1 per batch)
    z=arrays['train']
    g_num=torch.from_numpy(np.ascontiguousarray(z['num'][:,ni],dtype='float32')).to('cuda:0')
    g_missing=torch.from_numpy(np.ascontiguousarray(z['missing'][:,ni])).to('cuda:0')
    g_cat=torch.from_numpy(np.ascontiguousarray(z['cat'][:,ci],dtype='int64')).to('cuda:0')
    g_target=torch.from_numpy(np.array(target,copy=True)).to('cuda:0')
    resident_MiB=sum(t.element_size()*t.nelement() for t in [g_num,g_missing,g_cat,g_target])/1024**2
    for seed in a.seeds:
        before=out/f'seed-{seed}-before.json'
        if before.exists():
            raise RuntimeError('fit already has a launch record')
        gpu=smi();g=[r for r in gpu if r[0]=='1']
        if len(g)!=1 or float(g[0][4])<8192:
            raise RuntimeError('valid first call: GPU 1 lacks the reserved eight GB')
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
        rec={'seed':seed,'fit':a.fit,'arm':'full','epochs':epochs,'batch':batch,'loss':'MSE','target_scale':1000,
             'hidden':[512,512,256],'normalization':'LayerNorm','activation':'SiLU','dropout':.1,
             'optimizer':'AdamW','peak_lr':.001,'weight_decay':.0001,'gradient_norm_cap':5,
             'category_sizes':sizes,'features':setting,'inputs_sha256':sha(inp/'INPUTS.json'),
             'spec_sha256':manifest['spec_sha256'],'code_sha256':sha(__file__),'fit_code_sha256':sha(W/'code/mlp008/fit.py'),
             'split':manifest['split'],'loader':'training arrays resident on GPU 1','resident_MiB':resident_MiB,
             'pid':os.getpid(),'cores':cores,'nice':19,'torch':torch.__version__,'cuda':torch.version.cuda,
             'gpu_before':gpu,'GPU_allocator_cap_GiB':6,'actual_GPU_limit_MiB':7500,'train_rows':n,
             'held_rows':len(meta),'no_held_labels':True,'no_held_early_stop':True,'no_score_in_fit':True,
             'development_stop_after':a.stop_after or None}
        write(before,rec)
        rng=np.random.default_rng(seed);t0=time.monotonic();step=0;curve=[]
        for epoch in range(a.stop_after or epochs):
            model.train();order=rng.permutation(n);total=0.;seen=0
            for start in range(0,n,batch):
                idx=order[start:start+batch];gi=torch.from_numpy(idx).to('cuda:0')
                c,x,m=g_cat[gi],g_num[gi],g_missing[gi].float();y=g_target[gi]
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
        if a.stop_after:
            rec.update(curve=curve,seconds=time.monotonic()-t0);write(out/f'seed-{seed}.json',rec)
            del model,opt,scheduler;torch.cuda.empty_cache();continue
        model.eval();zh=np.empty(len(meta),dtype='float64')
        with torch.inference_mode():
            for start in range(0,len(meta),batch):
                sl=slice(start,start+batch);c,x,m=tensors('held',sl)
                zh[sl]=model(c,x,m).cpu().numpy().astype('float64')*1000
        pred=np.maximum(60,meta.anchor.to_numpy('float64')+zh)
        if not np.isfinite(pred).all():
            raise RuntimeError('nonfinite held prediction')
        p=out/f'seed-{seed}-held.parquet';f=meta[[ID,'month','utc_day','airport']].copy();f['prediction']=pred;f.to_parquet(p,index=False)
        model_path=out/f'seed-{seed}-model.pt'
        torch.save({'state_dict':model.cpu().state_dict(),'category_sizes':sizes,'numeric_count':len(ni)},model_path)
        rec.update(curve=curve,seconds=time.monotonic()-t0,prediction=artifact(p),model=artifact(model_path),
                   allocator_peak_MiB=torch.cuda.max_memory_reserved()/1024**2,actual_peak_MiB=max(r['GPU_MiB'] for r in curve))
        write(out/f'seed-{seed}.json',rec);records.append(artifact(out/f'seed-{seed}.json'));preds.append(pred)
        del model,opt,scheduler;torch.cuda.empty_cache()
    if a.stop_after:
        print(json.dumps({'event':'dev-ready','fit':a.fit,'tag':a.tag}),flush=True);return
    f=meta[[ID,'month','utc_day','airport']].copy();f['prediction']=np.mean(preds,axis=0)
    p=out/'mean-held.parquet';f.to_parquet(p,index=False)
    write(out/'FIT.json',{'fit':a.fit,'arm':'full','seeds':a.seeds,'seed_records':records,'prediction':artifact(p),
                         'held_part':'ranking rows' if a.fit=='final' else ('jjrefit rows then ranking rows' if a.fit=='jjrank' else 'held month'),
                         'inputs_sha256':sha(inp/'INPUTS.json'),'code_sha256':sha(__file__),'no_score':True})
    print(json.dumps({'event':'fit-ready','fit':a.fit,'prediction':artifact(p)}),flush=True)

if __name__=='__main__':
    main()
