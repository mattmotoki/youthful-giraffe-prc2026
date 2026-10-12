"""CPU annual sequence shape, padding and full-model reload checks; no quality score."""
import argparse,json,os,resource,time
from pathlib import Path
import pyarrow as pa
import numpy as np
import torch
from torch import nn
from fit import Arrays,sha,seed,save_model,load_model,SEED,BATCH

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--checks',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--threads',type=int,default=6);a=ap.parse_args()
    if a.out.exists():raise ValueError('Smoke root exists')
    if not 1<=a.threads<=6:raise ValueError('Bad thread cap')
    if os.sched_getaffinity(0) not in ({4,5,6,7,68,69,70,71},{60,61,62,63,124,125,126,127}):raise ValueError('Smoke lane differs')
    if os.getpriority(os.PRIO_PROCESS,0)!=19:raise ValueError('Smoke requires nice19')
    resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    device=torch.device('cpu')
    check=json.loads(a.checks.read_text())
    if check['status']!='PASS' or check['manifest_sha256']!=sha(a.inputs/'MANIFEST.json'):raise ValueError('Wrong input check')
    train=np.load(a.inputs/'train-indices.npy',allow_pickle=False)
    labels=np.load(a.inputs/'train-labels.npy',allow_pickle=False)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    rng=np.random.default_rng(SEED);pool=np.arange(len(train))
    take=rng.choice(pool,size=10*BATCH,replace=False)
    a.out.mkdir(parents=True);results=[]
    for arm in ('sequence',):
        seed();data=Arrays(a.inputs,'annual',arm);model=data.model().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        times=[]
        for k in range(10):
            batch=take[k*BATCH:(k+1)*BATCH];t=time.monotonic()
            n,c,e,p,anchor=data.batch(train[batch],device)
            target=torch.from_numpy(labels[batch]);corr=model(n,c,e,p)
            if corr.shape!=(BATCH,) or not torch.isfinite(corr).all():raise ValueError('Bad output shape/values')
            loss=nn.functional.smooth_l1_loss(corr,(target-anchor)/300.,beta=2)
            opt.zero_grad(set_to_none=True);loss.backward()
            if any(q.grad is not None and not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('Bad gradient')
            nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();times.append(time.monotonic()-t)
        model.eval();n,c,e,p,anchor=data.batch(train[take[:BATCH]],device)
        with torch.no_grad():before=model(n,c,e,p)
        padding_checked=False
        if arm=='sequence':
            # Give every row artificial padding on the two outer tokens and
            # alter those inputs. Masked inputs cannot change the centre output.
            mask=p.clone();mask[:,[0,16]]=True
            with torch.no_grad():base=model(n,c,e,mask)
            nn_=n.clone();cc=c.clone();ee=e.clone()
            nn_[mask]=1e6;cc[mask]=0;ee[mask]=-1e6
            with torch.no_grad():poison=model(nn_,cc,ee,mask)
            torch.testing.assert_close(poison,base,rtol=0,atol=0)
            padding_checked=True
        path=a.out/(arm+'.pt');save_model(path,model,data,'annual')
        with torch.no_grad():after=load_model(path,device).eval()(n,c,e,p)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        ranking=np.load(a.inputs/'ranking-indices.npy',allow_pickle=False)[:1024]
        from fit import predict
        a0=predict(model,data,ranking,device);a1=predict(load_model(path,device),data,ranking,device)
        np.testing.assert_array_equal(a0,a1)
        results.append({'arm':arm,'parameters':sum(q.numel() for q in model.parameters()),
            'batch_rows':BATCH,'train_batches':10,'warm_median_seconds_per_batch':float(np.median(times[2:])),
            'CPU_rows_per_second':float(BATCH/np.median(times[2:])),
            'shape_gradient_finite':True,'reload_exact':True,'ranking_first1024_reload_exact':True,'padding_invariant':padding_checked,
            'model_sha256':sha(path)})
        del model,opt,data
    out={'status':'PASS','what':'annual CPU smoke only; no ranking quality score','seed':SEED,
         'rows_from_all2025_training_only':len(take),'manifest_sha256':sha(a.inputs/'MANIFEST.json'),
         'checks_sha256':sha(a.checks),'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['model.py','fit.py','annual_job.py','smoke_annual.py']},
         'torch':torch.__version__,'device':'cpu','threads':a.threads,'affinity':sorted(os.sched_getaffinity(0)),
         'nice':19,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'arms':results}
    (a.out/'SMOKE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':main()
