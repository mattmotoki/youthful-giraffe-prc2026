"""One grind-only twelve-month mixed fit, frozen12epochs,90min/10GiB."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from fit import Arrays,sha,seed,run_epoch,predict,save_model,load_model,SEED,BATCH

BUDGET=5400
GPU_CAP=10*1024**3
SOURCES=['model.py','fit.py','annual_job.py','smoke_annual.py']

def verify(inputs,checks,smoke,review):
    manifest=json.loads((inputs/'MANIFEST.json').read_text());manifest_hash=sha(inputs/'MANIFEST.json')
    for receipt in [checks,smoke]:
        r=json.loads(receipt.read_text())
        if r['status']!='PASS' or r['manifest_sha256']!=manifest_hash:raise ValueError('Missing matching annual PASS')
    r=json.loads(review.read_text())
    if r['status']!='PASS':raise ValueError('Fresh annual source review failed')
    required=set(SOURCES+['prepare.py','prepare_annual.py','check_annual.py'])
    if not required.issubset(r['source_sha256']):raise ValueError('Incomplete annual source review')
    if json.loads(checks.read_text())['checker_sha256']!=sha(Path(__file__).with_name('check_annual.py')):raise ValueError('Annual checker changed')
    for name,want in manifest['source_sha256'].items():
        if sha(Path(__file__).with_name(name))!=want:raise ValueError('Annual preparation source changed')
    for rec in [r,json.loads(smoke.read_text())]:
        for name,want in rec['source_sha256'].items():
            if sha(Path(__file__).with_name(name))!=want:raise ValueError('Reviewed source changed: '+name)
    for name,want in manifest['files'].items():
        if sha(inputs/name)!=want:raise ValueError('Annual input changed: '+name)
    if manifest['split']!={'loss_year':2025,'loss_months':list(range(1,13)),'encoder_year':2025,'ranking_year':2026}:raise ValueError('Annual split differs')
    return manifest

def device_setup():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    if not torch.cuda.is_available():raise ValueError('Annual fit belongs to grind GPU job')
    total=torch.cuda.get_device_properties(0).total_memory
    if total<GPU_CAP:raise ValueError('GPU memory below10GiB')
    torch.cuda.set_per_process_memory_fraction(GPU_CAP/total,0)
    return torch.device('cuda')

def benchmark(inputs,device):
    ix=np.load(inputs/'train-indices.npy',allow_pickle=False);y=np.load(inputs/'train-labels.npy',allow_pickle=False)
    data=Arrays(inputs,'annual','sequence');seed();model=data.model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    take=np.random.default_rng(SEED).choice(len(ix),32*BATCH,replace=False);times=[]
    for k in range(32):
        q=take[k*BATCH:(k+1)*BATCH];torch.cuda.synchronize();started=time.monotonic()
        n,c,e,p,anchor=data.batch(ix[q],device);target=torch.from_numpy(y[q]).to(device)
        loss=torch.nn.functional.smooth_l1_loss(model(n,c,e,p),(target-anchor)/300.,beta=2)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        torch.cuda.synchronize();times.append(time.monotonic()-started)
    seconds=float(np.median(times[8:]));ranking=len(np.load(inputs/'ranking-ids.npy',allow_pickle=False))
    # Charge both predictions as training batches, add50% plus5min; benchmark weights are discarded.
    estimate=1.5*seconds*(12*len(ix)+2*ranking)/BATCH+300
    return {'batch_seconds':seconds,'rows_per_batch':BATCH,'fixed_epochs':12,'estimate_seconds':estimate,
        'safety_multiplier':1.5,'benchmark_not_used_as_final_model':True}

def fit(a,manifest,device):
    a.out.mkdir(parents=True);started=time.monotonic();seed()
    data=Arrays(a.inputs,'annual','sequence');model=data.model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(SEED)
    ix=np.load(a.inputs/'train-indices.npy',allow_pickle=False);y=np.load(a.inputs/'train-labels.npy',allow_pickle=False)
    if len(ix)!=len(y) or not np.isfinite(y).all():raise ValueError('Bad annual loss')
    history=[]
    for epoch in range(1,13):
        row={'epoch':epoch,'train':run_epoch(model,data,ix,y,device,opt,rng),'seconds':time.monotonic()-started}
        history.append(row);print(json.dumps(row),flush=True)
    save_model(a.out/'model.pt',model,data,'annual')
    ranking=np.load(a.inputs/'ranking-indices.npy',allow_pickle=False);ids=np.load(a.inputs/'ranking-ids.npy',allow_pickle=False)
    if len(ids)!=344841:raise ValueError('Wrong ranking count')
    pred=predict(model,data,ranking,device)
    pd.DataFrame({'MVT_ID_mvt':ids,'pred_Gmixed':pred}).to_parquet(a.out/'ranking.parquet',index=False)
    replay=predict(load_model(a.out/'model.pt',device),data,ranking,device)
    if not np.array_equal(replay,pred):raise ValueError('Full ranking reload differs')
    record={'status':'completed','arm':'Gmixed','owner':('final'),'encoder_stage':'annual','seed':SEED,
        'fixed_epochs':12,'epoch_selection':'frozen from screen FIT5eaff7fd; no new selection','history':history,
        'train_labels':len(ix),'train_months':manifest['summary']['train_rows_by_UTC_month'],'train_split':manifest['split'],
        'ranking_rows':len(ids),'ranking_order':'v37 and official submitting exact IDs',
        'ranking_ID_bytes_sha256':__import__('hashlib').sha256(ids.tobytes()).hexdigest(),
        'ranking_labels_in_payload':False,'ranking_labels_used':False,'floor':60,'upper_cap':None,
        'loss':'smooth-L1 beta2 correction300s; AdamWlr.001wd.0001batch256clip1',
        'model_reload_exact_all_344841':True,'manifest_sha256':sha(a.inputs/'MANIFEST.json'),
        'checks_sha256':sha(a.checks),'review_sha256':sha(a.review),'smoke_sha256':sha(a.smoke),
        'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in SOURCES},'seconds':time.monotonic()-started,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,
        'files':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}}
    (a.out/'FIT.json').write_text(json.dumps(record,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser()
    for n in ['inputs','checks','smoke','review','out']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--fit',action='store_true');ap.add_argument('--benchmark',action='store_true');a=ap.parse_args()
    if a.out.exists():raise ValueError('Annual GPU output exists')
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    started=time.monotonic();manifest=verify(a.inputs,a.checks,a.smoke,a.review)
    if a.fit:fit(a,manifest,device_setup());return
    if a.benchmark:
        a.out.mkdir(parents=True);bench=benchmark(a.inputs,device_setup())
        (a.out/'BENCHMARK.json').write_text(json.dumps(bench,indent=2)+'\n');return
    a.out.mkdir(parents=True);source_hashes={n:sha(Path(__file__).with_name(n)) for n in SOURCES}
    result={'status':'running','GPU_budget_seconds':BUDGET,'GPU_memory_cap_bytes':GPU_CAP,'source_sha256':source_hashes}
    try:
        # Isolate the GPU benchmark so its allocator/context is released before the final fit.
        common=['--inputs',str(a.inputs),'--checks',str(a.checks),'--smoke',str(a.smoke),'--review',str(a.review)]
        with (a.out/'benchmark.log').open('x') as stream:
            run=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--benchmark',*common,'--out',str(a.out/'preflight')],
                stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,BUDGET-(time.monotonic()-started)))
        if run.returncode:raise RuntimeError('Annual benchmark failed; see benchmark.log')
        bench=json.loads((a.out/'preflight/BENCHMARK.json').read_text());result['benchmark']=bench
        if bench['estimate_seconds']>BUDGET-(time.monotonic()-started):
            result.update(status='needs-decision',reason='Annual preflight exceeds90min; no real fit started')
        else:
            log=a.out/'annual.log'
            cmd=[sys.executable,'-u',str(Path(__file__)),'--fit','--inputs',str(a.inputs),'--checks',str(a.checks),
                '--smoke',str(a.smoke),'--review',str(a.review),'--out',str(a.out/'annual')]
            with log.open('x') as stream:
                run=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,BUDGET-(time.monotonic()-started)))
            result.update(returncode=run.returncode,log_sha256=sha(log))
            if run.returncode:raise RuntimeError('Annual fit failed; see annual.log')
            result['status']='completed'
        for name,want in source_hashes.items():
            if sha(Path(__file__).with_name(name))!=want:raise ValueError('Annual source changed')
    except Exception as ex:result.update(status='failed',error=repr(ex))
    result['seconds']=time.monotonic()-started
    (a.out/'JOB.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='completed':raise SystemExit(75 if result['status']=='needs-decision' else 1)

if __name__=='__main__':main()
