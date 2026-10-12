"""Unit 008: check one fit's model receipts and a CPU replay of fixed scored rows; load no held label.

Unit 007's check_fit_outputs.py (sha 6197f8e3) for the full arm and the fit named by --fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from fit import Net
from io_utils import W, ID, sha, artifact, write, lane

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',required=True);a=ap.parse_args()
    lane();torch.set_num_threads(1);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    inp=W/'inputs/mlp-008'/a.fit;out=W/'out/mlp-008'/a.fit;result=W/'results/mlp-008'
    m=json.loads((inp/'INPUTS.json').read_text());fit=json.loads((out/'FIT.json').read_text())
    pp=json.loads((inp/'preprocessing.json').read_text());feature=m['features']['full']
    ni,ci=feature['numeric_indices'],feature['categorical_indices'];sizes=[pp['category_sizes'][i] for i in ci]
    checks=[]
    def check(name,ok,**facts):
        if not bool(ok):raise RuntimeError(f'{name}: {facts}')
        checks.append({'name':name,'state':'pass',**facts})
    check('fixed fit seeds and inputs',fit['fit']==a.fit and fit['arm']=='full' and fit['seeds']==[701,702] and fit['inputs_sha256']==sha(inp/'INPUTS.json'))
    check('fit code is the current fit.py',fit['code_sha256']==sha(Path(__file__).with_name('fit.py')))
    meta=pd.read_parquet(inp/'held-metadata.parquet');idx=np.linspace(0,len(meta)-1,257,dtype=int)
    arrays={k:np.load(inp/f'held-{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['num','missing','cat']}
    x=torch.from_numpy(np.array(arrays['num'][idx][:,ni],dtype='float32',copy=True))
    missing=torch.from_numpy(np.array(arrays['missing'][idx][:,ni],dtype='float32',copy=True))
    cats=torch.from_numpy(np.array(arrays['cat'][idx][:,ci],dtype='int64',copy=True))
    check('two seed receipts',len(fit['seed_records'])==2)
    preds=[];replays={};actual_seeds=[]
    for receipt in fit['seed_records']:
        check('seed record pin '+receipt['path'],sha(receipt['path'])==receipt['sha256'])
        rr=json.loads(Path(receipt['path']).read_text());seed=rr['seed'];actual_seeds.append(seed)
        fixed={'fit':a.fit,'arm':'full','epochs':40,'batch':8192,'loss':'MSE','target_scale':1000,'hidden':[512,512,256],'normalization':'LayerNorm','activation':'SiLU','dropout':.1,'optimizer':'AdamW','peak_lr':.001,'weight_decay':.0001,'gradient_norm_cap':5,'train_rows':m['train_rows'],'held_rows':m['held_rows'],'no_held_labels':True,'no_held_early_stop':True,'no_score_in_fit':True,'spec_sha256':m['spec_sha256'],'code_sha256':fit['code_sha256']}
        check('complete fixed receipt settings '+str(seed),all(rr[k]==v for k,v in fixed.items()))
        check('spec pin '+str(seed),rr['spec_sha256']==sha(result/'SPEC-008.md'))
        check('seed settings '+str(seed),rr['inputs_sha256']==sha(inp/'INPUTS.json') and len(rr['curve'])==40 and rr['features']==feature and rr['category_sizes']==sizes)
        check('all epochs cover train rows '+str(seed),[r['epoch'] for r in rr['curve']]==list(range(1,41)) and all(r['rows_seen']==m['train_rows'] and np.isfinite(r['train_rmse_s']) and r['GPU_MiB']<=7500 for r in rr['curve']))
        for key in ['model','prediction']:check(key+' pin '+str(seed),sha(rr[key]['path'])==rr[key]['sha256'])
        state=torch.load(rr['model']['path'],map_location='cpu',weights_only=True)
        check('saved architecture '+str(seed),state['category_sizes']==sizes and state['numeric_count']==len(ni))
        model=Net(sizes,len(ni));model.load_state_dict(state['state_dict'],strict=True);model.eval()
        check('padding vectors remain zero '+str(seed),all(not emb.weight[0].detach().numpy().any() for emb in model.emb))
        pred=pd.read_parquet(rr['prediction']['path'])
        check('prediction rows IDs scope '+str(seed),np.array_equal(pred[ID],meta[ID]) and len(pred)==m['held_rows'] and pred[ID].is_unique)
        check('finite floored predictions '+str(seed),np.isfinite(pred.prediction).all() and pred.prediction.min()>=60)
        with torch.inference_mode():replay=np.maximum(60,meta.anchor.to_numpy()[idx]+model(cats,x,missing).numpy().astype(float)*1000)
        error=np.abs(replay-pred.prediction.to_numpy()[idx]);replays[str(seed)]={'rows':len(idx),'max_abs_s':float(error.max()),'rms_s':float(np.sqrt(np.mean(error**2))),'tolerance_s':.1}
        check('CPU replay '+str(seed),error.max()<=.1,**replays[str(seed)])
        preds.append(pred.prediction.to_numpy());del model,state
    check('actual ordered seed identities',actual_seeds==[701,702])
    check('mean prediction pin',sha(fit['prediction']['path'])==fit['prediction']['sha256'])
    mean=pd.read_parquet(fit['prediction']['path'])
    check('mean exact rows IDs and equal weighting',np.array_equal(mean[ID],meta[ID]) and np.array_equal(mean.prediction.to_numpy(),np.mean(preds,axis=0)))
    curves={str(s):[r['train_rmse_s'] for r in json.loads(Path(rc['path']).read_text())['curve']][-1] for s,rc in zip(actual_seeds,fit['seed_records'])}
    write(result/f'CHECK-FIT-{a.fit}.json',{'state':'pass','fit':a.fit,'n_checks':len(checks),'checks':checks,'replay':replays,
        'final_train_rmse_s':curves,'fit_receipt':artifact(out/'FIT.json'),'helper':artifact(Path(__file__)),'no_held_labels_loaded':True,'no_G_read':True})
    print(json.dumps({'fit':a.fit,'state':'pass','n_checks':len(checks),'CPU_replay':replays,'final_train_rmse_s':curves}))

if __name__=='__main__':main()
