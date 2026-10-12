"""The original039 thirty annual recipes on selected Z; exact fixed queue only."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from check_time_day_052 import sha

FAST = '4e86dc93cf5a698b1bc87fc875ccb998a7243eb4c91df228551d1c51b6724714'
INPUT = 'd052ad0239f6ebca4c05320eac7ba0766dbbfcd158afc06a318fa880ce92d3b1'
AVERAGE = '05385518b5310bd9c724d192f44fd6402cfdb7573f448cf4d4d7843f7198d63a'
SCOPE = 'e4199e5938b6ce932e3b4fa30e1d88e866e528b333d007a794b19ce264063554'
ORIGINAL = '6b21654958e980719b3db950deb2881508383b756e5d2a2c5fbe55d933da0a82'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec-sha', required=True)
    ap.add_argument('--legacy-source-sha', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    assert sha('SPEC-FIXED039-ANNUAL-061.md') == a.spec_sha
    assert sha('code/fit_runway_fast_029.py') == FAST and sha('code/average_mixed_033.py') == AVERAGE
    assert sha('REVIEW-TIME-DAY-INPUTS-ACTUAL-052.md') == 'af9c2b25732f89ff8233c666a3f41b3d38c964abc0d4e11159513acbf3fd0da3'
    assert sha('results/STACK12-TIME-SCOPE-055.json') == SCOPE
    assert sha('out/sequence-close30-final-039/FIT.json') == ORIGINAL
    assert sha('code/rebuild_legacy_final_060.py') == a.legacy_source_sha
    inputs = Path('inputs/time-day-Z-052')
    assert sha(inputs/'MANIFEST.json') == INPUT
    manifest = json.loads((inputs/'MANIFEST.json').read_text())
    assert manifest['time_day_form'] == 'Z' and manifest['changed_numeric_columns'] == [57,116]
    for name,digest in manifest['files'].items():assert sha(inputs/name) == digest
    meta = pd.read_parquet(inputs/'rows.parquet',columns=['MVT_ID_mvt','year','month'])
    assert len(meta) == 2429888 and meta.MVT_ID_mvt.dtype == np.int64 and meta.MVT_ID_mvt.is_unique
    train_ids = meta.loc[meta.year.eq(2025),'MVT_ID_mvt'].to_numpy('int64')
    rank_ids = meta.loc[meta.year.eq(2026),'MVT_ID_mvt'].to_numpy('int64')
    assert len(train_ids) == 2085047 and len(rank_ids) == 344841
    train_sha = hashlib.sha256(train_ids.tobytes()).hexdigest()
    scope = next(m for m in json.loads(Path('results/STACK12-TIME-SCOPE-055.json').read_text())['members'] if m['role'] == 'sequence039')
    references = {}
    for p in scope['parents']:
        assert sha(p['fit']) == p['sha256']
        ref = json.loads(Path(p['fit']).read_text())
        s = ref['args'];key=(s['seed'],s['width'],s['epochs'])
        assert key not in references and ref['code_sha256'] == FAST and ref['weights'] == 'last iterate'
        assert s['layers'] == 3 and s['batch_size'] == 512 and s['mode'] == 'sequence-runway' and s['loss'] == 'huber'
        assert s['final'] is True and s['ema'] == .999 and s['ema2'] == .9997
        references[key] = p
    expected = {(s,256,24) for s in range(17001,17013)}|{(s,128,48) for s in range(17001,17013)}|{(s,256,48) for s in range(17001,17007)}
    assert set(references) == expected and len(references) == 30
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ[key] == '2'
    assert os.getpriority(os.PRIO_PROCESS,0) == 19 and set(os.sched_getaffinity(0)) == {14,15}
    if a.preflight_only:
        print('PREFLIGHT_PASS: exact Z input, all thirty original annual recipes and sources',flush=True)
        return
    Path('logs/fixed039-annual-061.claim').mkdir(exist_ok=False)
    Path('logs/fixed039-annual-061.pid').write_text(str(os.getpid())+'\n')
    result = dict(status='WAITING_057_AND060',source_row=902,source_sha256=sha(__file__),spec_sha256=a.spec_sha,
                  input_manifest_sha256=INPUT,original039_FIT_sha256=ORIGINAL,legacy_source_sha256=a.legacy_source_sha,
                  stage='LAST only; retain bothEMA stages',annual=True,quality_read=False,pack_admission=False,streams={})
    receipt = Path('results/FIXED039-ANNUAL-061.json');assert not receipt.exists()
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    def wait_for(path, timeout_hours, source_sha, input_sha=None):
        stop = time.monotonic()+timeout_hours*3600
        while True:
            try:
                prior=json.loads(path.read_text())
                if prior.get('status') in ['PASS','FAIL']:break
            except (FileNotFoundError,json.JSONDecodeError):pass
            assert time.monotonic() < stop, f'wait timeout:{path}'
            time.sleep(30)
        assert prior['status'] == 'PASS', f'prior queue failed:{path}'
        assert prior['source_sha256'] == source_sha
        if input_sha:assert prior['input_manifest_sha256'] == input_sha
        return sha(path)
    def run(argv, name, gpu):
        used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used <= 14000, f'GPU{gpu} has{used}MiB used'
        with Path('logs',name+'.log').open('x') as log:
            p = subprocess.Popen([sys.executable,'-u',*argv],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            print('PROCESS',gpu,p.pid,argv,flush=True)
            assert p.wait() == 0, f'failed:{argv}'
    def fit(seed, width, epochs, recipe, gpu):
        folder = Path('out',f'fixed039-{recipe}-{seed}-annual-061')
        assert not folder.exists()
        ref = references[(seed,width,epochs)]
        run(['code/fit_runway_fast_029.py','--inputs',str(inputs),'--out',str(folder),'--mode','sequence-runway','--loss','huber',
             '--epochs',str(epochs),'--batch-size','512','--seed',str(seed),'--width',str(width),'--layers','3',
             '--ema','.999','--ema2','.9997','--device','cuda','--final'],folder.name,gpu)
        stages = []
        for stage,weights in [('last','last iterate'),('ema2','exponential mean, decay 0.9997'),('','exponential mean, decay 0.999')]:
            root = folder/stage;rec=json.loads((root/'FIT.json').read_text())
            for key,value in dict(seed=seed,width=width,epochs=epochs,layers=3,batch_size=512,loss='huber',mode='sequence-runway',ema=.999,ema2=.9997,final=True,device='cuda').items():assert rec['args'][key] == value
            assert rec['weights'] == weights and rec['code_sha256'] == FAST and rec['inputs_manifest_sha256'] == INPUT
            assert rec['train_rows'] == len(train_ids) and rec['train_ids_sha256'] == train_sha and rec['train_months'] == list(range(1,13))
            assert rec['window'] == dict(group='airport-day-runway',halo=4,core=4)
            assert [h['epoch'] for h in rec['history']] == list(range(1,epochs+1)) and all(np.isfinite(h['train_loss']) for h in rec['history'])
            assert rec['peak_gpu_bytes'] <= 10*(1<<30)
            for name,digest in rec['files'].items():assert sha(root/name) == digest
            vector = pd.read_parquet(root/'ranking2026.parquet')
            assert vector.MVT_ID_mvt.dtype == np.int64 and vector.MVT_ID_mvt.is_unique and np.array_equal(vector.MVT_ID_mvt.to_numpy(),rank_ids)
            assert np.isfinite(vector.pred_seq170).all() and vector.pred_seq170.min() >= 60
            stages.append(dict(stage=stage or 'ema',FIT_sha256=sha(root/'FIT.json'),model_sha256=sha(root/'model.pt'),vector_sha256=sha(root/'ranking2026.parquet')))
        return dict(path=str((folder/'last').resolve()),FIT_sha256=stages[0]['FIT_sha256'],stages=stages,
                    seed=seed,width=width,epochs=epochs,reference_FIT_sha256=ref['sha256'])
    def stream(gpu):
        prior = wait_for(Path('results/FIXED-PACK-REBUILD-057.json'),6,'e7d4aa96e21624af76ef393d5ed0bf005b8c32a356483955ff7231f26267e70c',INPUT)
        legacy = None
        if gpu == 0:legacy = wait_for(Path('results/FINAL-LEGACY-REBUILD-060.json'),8,a.legacy_source_sha)
        groups = [('wide24',256,24,range(17001,17013))] if gpu == 1 else [('r48',128,48,range(17001,17013)),('wide48',256,48,range(17001,17007))]
        parents = {name:[fit(seed,width,epochs,name,gpu) for seed in seeds] for name,width,epochs,seeds in groups}
        return dict(prior057_sha256=prior,prior060_sha256=legacy,parents=parents)
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {str(gpu):pool.submit(stream,gpu) for gpu in [0,1]}
        for gpu,job in futures.items():
            try:result['streams'][gpu]=job.result()
            except Exception as exc:errors.append(f'GPU{gpu}:{type(exc).__name__}:{exc}')
    if not errors:
        groups = {**result['streams']['1']['parents'],**result['streams']['0']['parents']}
        result['means'] = {}
        for name in ['wide24','r48','wide48']:
            folder = Path('out',f'fixed039-{name}-annual-mean-061')
            run(['code/average_mixed_033.py','--parents',*[p['path'] for p in groups[name]],'--domain','ranking2026','--out',str(folder)],folder.name,0)
            result['means'][name]=dict(path=str(folder.resolve()),FIT_sha256=sha(folder/'FIT.json'),vector_sha256=sha(folder/'ranking2026.parquet'))
        parents = groups['wide24']+groups['r48']+groups['wide48']
        assert len(parents) == len({p['path'] for p in parents}) == 30
        folder = Path('out/fixed-close30-annual-061')
        run(['code/average_mixed_033.py','--parents',*[p['path'] for p in parents],'--domain','ranking2026','--out',str(folder)],folder.name,0)
        result['mixed']=dict(path=str(folder.resolve()),FIT_sha256=sha(folder/'FIT.json'),vector_sha256=sha(folder/'ranking2026.parquet'))
    result['status']='FAIL' if errors else 'PASS';result['errors']=errors;result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],errors=errors)),flush=True)
    assert not errors,errors

if __name__ == '__main__':
    try:main()
    except Exception as exc:
        path=Path('results/FIXED039-ANNUAL-061.json')
        if path.exists():
            value=json.loads(path.read_text());value['status']='FAIL';value['error']=f'{type(exc).__name__}: {exc}'
            temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
        raise
