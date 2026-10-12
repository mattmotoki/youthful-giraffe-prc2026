"""Unit 011: check one fit_sweep fit (an arm's seed receipts, models and a CPU replay of fixed held rows); load no held label.

check_bag.py (unit 009) for out/mlp-010/<arm>/<sub> (fit_sweep.py; sub is the fit name, or
<fit>-seeds-S1-S2-... for more seeds of the arm). The settings are the arm's (fit_sweep ARMS over
BASE). fit_sweep code: one of the three committed versions below; they differ only in the docstring,
the FITS list and the folder of non-default seeds, so each fit runs the same model code. A receipt's
code hash is read from disk when it is written, so a fit that ran while the file changed can name the
newer version (e80 jjx-10 seed 702 and its FIT.json; REPORT-011).

    python3 check_sweep.py --arm e80 --sub jjx-10-seeds-703-704-705-706
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from io_utils import W, ID, sha, artifact, write, lane
from fit_sweep import Net, ARMS, BASE, FITS

KNOWN_CODE = {'965e4101f032a8870d628498d3c45e767fa2807a0ccd07c9a3d3e8b7b722da3c': '69d651d2 (unit 010 sweep)',
              'ef00d5115ebaed283837140731e43e627e32246bb4dfca1328cd44d9e3d117c7': '44350b36 (extra-seed folder)',
              '95bd3e4a15472bacac2c52d00925ffd6ac46586cdb3d32e44f051db90cf99c29': 'b58ffb51 (all nine fits)'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=list(ARMS));ap.add_argument('--sub',required=True);a=ap.parse_args()
    lane();torch.set_num_threads(1);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    out=W/'out/mlp-010'/a.arm/a.sub;result=W/'results/mlp-011';result.mkdir(parents=True,exist_ok=True)
    dest=result/f'CHECK-{a.arm}-{a.sub}.json'
    if dest.exists():raise RuntimeError(f'{dest} exists; write-once')
    fit=json.loads((out/'FIT.json').read_text());fit_name=fit['fit']
    if fit_name not in FITS:raise RuntimeError('unknown fit')
    cfg={**BASE,**ARMS[a.arm]};seeds=fit['seeds']
    inp=W/'inputs/mlp-008'/fit_name
    m=json.loads((inp/'INPUTS.json').read_text())
    pp=json.loads((inp/'preprocessing.json').read_text());feature=m['features']['full']
    ni,ci=feature['numeric_indices'],feature['categorical_indices'];sizes=[pp['category_sizes'][i] for i in ci]
    here=Path(__file__).parent
    checks=[]
    def check(name,ok,**facts):
        if not bool(ok):raise RuntimeError(f'{name}: {facts}')
        checks.append({'name':name,'state':'pass',**facts})
    want_sub=fit_name if seeds==[701,702] else fit_name+'-seeds-'+'-'.join(map(str,seeds))
    check('fit, arm, seeds, folder and inputs',fit['arm']==a.arm and fit['config']==cfg and a.sub==want_sub and fit['inputs_sha256']==sha(inp/'INPUTS.json'))
    check('fit code is a committed fit_sweep.py',fit['code_sha256'] in KNOWN_CODE,code=fit['code_sha256'][:8],commit=KNOWN_CODE.get(fit['code_sha256']))
    meta=pd.read_parquet(inp/'held-metadata.parquet');idx=np.linspace(0,len(meta)-1,257,dtype=int)
    arrays={k:np.load(inp/f'held-{k}.npy',mmap_mode='r',allow_pickle=False) for k in ['num','missing','cat']}
    x=torch.from_numpy(np.array(arrays['num'][idx][:,ni],dtype='float32',copy=True))
    missing=torch.from_numpy(np.array(arrays['missing'][idx][:,ni],dtype='float32',copy=True))
    cats=torch.from_numpy(np.array(arrays['cat'][idx][:,ci],dtype='int64',copy=True))
    check('one receipt per seed',len(fit['seed_records'])==len(seeds) and len(set(seeds))==len(seeds))
    preds=[];replays={};actual_seeds=[];curves={};codes={}
    E=cfg['epochs']
    for receipt in fit['seed_records']:
        check('seed record pin '+receipt['path'],sha(receipt['path'])==receipt['sha256'])
        rr=json.loads(Path(receipt['path']).read_text());seed=rr['seed'];actual_seeds.append(seed);codes[str(seed)]=rr['code_sha256'][:8]
        fixed={'fit':fit_name,'arm':a.arm,'arm_change':ARMS[a.arm],'epochs':E,'batch':8192,'loss':'MSE','target_scale':1000,'hidden':cfg['hidden'],
               'normalization':'LayerNorm','activation':'SiLU','dropout':cfg['dropout'],'emb_mult':cfg['emb_mult'],'optimizer':'AdamW',
               'peak_lr':cfg['peak_lr'],'weight_decay':cfg['weight_decay'],'gradient_norm_cap':5,'train_rows':m['train_rows'],'held_rows':m['held_rows'],
               'no_held_labels':True,'no_held_early_stop':True,'no_score_in_fit':True,'spec_sha256':m['spec_sha256'],
               'fit_code_sha256':sha(here/'fit.py'),'development_stop_after':None}
        check('complete fixed receipt settings '+str(seed),all(rr[k]==v for k,v in fixed.items()),bad=[k for k,v in fixed.items() if rr.get(k)!=v])
        check('receipt code is a committed fit_sweep.py '+str(seed),rr['code_sha256'] in KNOWN_CODE)
        check('seed settings '+str(seed),rr['inputs_sha256']==sha(inp/'INPUTS.json') and len(rr['curve'])==E and rr['features']==feature and rr['category_sizes']==sizes and rr['split']==m['split'])
        check('all epochs cover train rows '+str(seed),[r['epoch'] for r in rr['curve']]==list(range(1,E+1)) and all(r['rows_seen']==m['train_rows'] and np.isfinite(r['train_rmse_s']) and r['GPU_MiB']<=7500 for r in rr['curve']))
        for key in ['model','prediction']:check(key+' pin '+str(seed),sha(rr[key]['path'])==rr[key]['sha256'])
        state=torch.load(rr['model']['path'],map_location='cpu',weights_only=True)
        check('saved architecture '+str(seed),state['category_sizes']==sizes and state['numeric_count']==len(ni) and state['arm']==a.arm and state['config']==cfg)
        model=Net(sizes,len(ni),cfg['hidden'],cfg['dropout'],cfg['emb_mult']);model.load_state_dict(state['state_dict'],strict=True);model.eval()
        check('padding vectors remain zero '+str(seed),all(not emb.weight[0].detach().numpy().any() for emb in model.emb))
        pred=pd.read_parquet(rr['prediction']['path'])
        check('prediction rows IDs scope '+str(seed),np.array_equal(pred[ID],meta[ID]) and len(pred)==m['held_rows'] and pred[ID].is_unique)
        check('finite floored predictions '+str(seed),np.isfinite(pred.prediction).all() and pred.prediction.min()>=60)
        with torch.inference_mode():replay=np.maximum(60,meta.anchor.to_numpy()[idx]+model(cats,x,missing).numpy().astype(float)*1000)
        error=np.abs(replay-pred.prediction.to_numpy()[idx]);replays[str(seed)]={'rows':len(idx),'max_abs_s':float(error.max()),'rms_s':float(np.sqrt(np.mean(error**2))),'tolerance_s':.1}
        check('CPU replay '+str(seed),error.max()<=.1,**replays[str(seed)])
        preds.append(pred.prediction.to_numpy());curves[str(seed)]=rr['curve'][-1]['train_rmse_s'];del model,state
    check('actual ordered seed identities',actual_seeds==seeds)
    check('mean prediction pin',sha(fit['prediction']['path'])==fit['prediction']['sha256'])
    mean=pd.read_parquet(fit['prediction']['path'])
    check('mean exact rows IDs and equal weighting',np.array_equal(mean[ID],meta[ID]) and np.array_equal(mean.prediction.to_numpy(),np.mean(preds,axis=0)))
    write(dest,{'state':'pass','arm':a.arm,'fit':fit_name,'sub':a.sub,'seeds':seeds,'n_checks':len(checks),'checks':checks,'replay':replays,
        'receipt_code':codes,'fit_code':fit['code_sha256'][:8],'final_train_rmse_s':curves,'fit_receipt':artifact(out/'FIT.json'),
        'helper':artifact(Path(__file__)),'no_held_labels_loaded':True,'no_G_read':True})
    print(json.dumps({'arm':a.arm,'sub':a.sub,'state':'pass','n_checks':len(checks),'max_replay_s':max(v['max_abs_s'] for v in replays.values()),'final_train_rmse_s':curves}))

if __name__=='__main__':main()
