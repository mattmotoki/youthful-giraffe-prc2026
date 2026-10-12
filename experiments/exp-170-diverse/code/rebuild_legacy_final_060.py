"""Exact fixed-Z legacy/row held and annual fits; no label score or selection."""
import argparse
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

PINS = {
 'fit.py': '79ec20fea289f93cbe43b9259fc986a0b0edf8aa07932a56fb24af3d69bbac77',
 'fit_runway.py': '7207612fc686b192725349c57a75b3a644135f0ed5085ba86929b45b810d0c90',
 'predict_mask_legacy_060.py': '293f27bafaa0c0bcda7606598f7799c8737c5b030815b32ff1dab0cb6042707c',
 'average_rows_021.py': 'c70449e9e087f25ed25c6451d08dc82ab0b316487bdecc40db94f28c42429286',
 'average_seeds_n.py': 'fa5c7e9a1bf0d7992551f3a8c4cc0ddb2bef3f5be896c303283f8b7026a50319',
 'join_masks.py': 'bbe13f6209fc31aa4945a32eba844fe74fff54535553e2cd4e01e7058350ae3d',
}
SCOPE_SHA = 'e4199e5938b6ce932e3b4fa30e1d88e866e528b333d007a794b19ce264063554'
GAP = 'd052ad0239f6ebca4c05320eac7ba0766dbbfcd158afc06a318fa880ce92d3b1'
ROW = 'cd2c84e0bbfa46579be23baf597cd6e7d19694daa1d444cfa68c7a625ffa3c2a'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runway-manifest-sha', required=True)
    ap.add_argument('--runway-review', type=Path, required=True)
    ap.add_argument('--runway-review-sha', required=True)
    ap.add_argument('--spec-sha', required=True)
    ap.add_argument('--row-mask-mean-sha', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    assert sha('SPEC-FINAL-LEGACY-060.md') == a.spec_sha
    assert sha(a.runway_review) == a.runway_review_sha
    assert 'PASS' in a.runway_review.read_text() and a.runway_manifest_sha in a.runway_review.read_text()
    assert sha('REVIEW-TIME-DAY-INPUTS-ACTUAL-052.md') == 'af9c2b25732f89ff8233c666a3f41b3d38c964abc0d4e11159513acbf3fd0da3'
    assert sha('REVIEW-ROW-TIME-INPUT-ACTUAL-054.md') == '035a126bc316ff384573cddf88bcd0f9c366f8131547889085dfd61471926d70'
    for name, digest in PINS.items():
        assert sha(Path('code') / name) == digest, name
    assert sha('code/mean_row_masks_060.py') == a.row_mask_mean_sha
    assert sha('results/STACK12-TIME-SCOPE-055.json') == SCOPE_SHA
    scope = {m['role']: m for m in json.loads(Path('results/STACK12-TIME-SCOPE-055.json').read_text())['members']}
    assert scope['fallback_control011']['remote_fit'] == scope['control6']['parents'][0]['fit']
    assert scope['fallback_control011']['fit_sha256'] == scope['control6']['parents'][0]['sha256']
    inputs = {'sequence021': (Path('inputs/time-day-Z-052'), GAP),
              'control6': (Path('inputs/row-time-day-Z-054'), ROW),
              'fallback_sequence011': (Path('inputs/runway-time-day-Z-059'), a.runway_manifest_sha)}
    metadata, originals = {}, {}
    for role, (root, digest) in inputs.items():
        assert sha(root / 'MANIFEST.json') == digest
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        assert manifest.get('fixed_time_form', manifest.get('time_day_form')) == 'Z'
        assert manifest['changed_numeric_columns'] == [57,116]
        for name, file_sha in manifest['files'].items():
            assert sha(root / name) == file_sha
        meta = pd.read_parquet(root / 'rows.parquet', columns=['MVT_ID_mvt','year','month'])
        assert len(meta) == 2429888 and meta.MVT_ID_mvt.dtype == np.int64 and meta.MVT_ID_mvt.is_unique
        metadata[role] = meta
        member = scope[role]
        assert sha(member['remote_fit']) == member['fit_sha256']
        originals[role] = member['parents'] or [{'fit': member['remote_fit'], 'sha256': member['fit_sha256']}]
        assert len(originals[role]) == (1 if role == 'fallback_sequence011' else 6)
        for i, pin in enumerate(originals[role]):
            assert sha(pin['fit']) == pin['sha256']
            ref = json.loads(Path(pin['fit']).read_text())
            source = 'fit.py' if role == 'control6' else 'fit_runway.py'
            assert ref['code_sha256'] == PINS[source]
            assert ref['args']['mode'] == ('row' if role == 'control6' else 'sequence-runway')
            assert ref['args']['loss'] == 'huber' and ref['args']['epochs'] == 12
            assert ref['args']['batch_size'] == (128 if role == 'control6' else 512)
            assert ref['args']['seed'] == 17001+i and ref['args']['final'] is True
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        assert os.environ[key] == '2'
    assert os.getpriority(os.PRIO_PROCESS,0) == 19 and set(os.sched_getaffinity(0)) == {14,15}
    if a.preflight_only:
        print('PREFLIGHT_PASS: exact three inputs, thirteen original parents, recipes, sources and reuse',flush=True)
        return
    Path('logs/final-legacy-060.claim').mkdir(exist_ok=False)
    Path('logs/final-legacy-060.pid').write_text(str(os.getpid())+'\n')
    result = dict(status='WAITING_GPU0', source_row=902, source_sha256=sha(__file__), spec_sha256=a.spec_sha,
                  input_manifests={k:v[1] for k,v in inputs.items()}, runway_review_sha256=a.runway_review_sha,
                  source_pins={**PINS,'mean_row_masks_060.py':a.row_mask_mean_sha}, held={}, annual={},
                  held_complete=False, annual_complete=False, quality_read=False, pack_admission=False)
    receipt = Path('results/FINAL-LEGACY-REBUILD-060.json')
    assert not receipt.exists()
    def save():
        temp = receipt.with_suffix('.json.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n')
        temp.replace(receipt)
    save()
    # Desk 2026-10-03 11:20: reviewed inputs plus a free GPU0 release this queue.
    # If057 later fails, these outputs stay outside every pack until its cause is known.
    result['release_rule'] = 'desk11:20: GPU0 free plus all three actual-reviewed inputs'
    result['hold_if057_fails'] = True
    result['prior057_expected_source_sha256'] = 'e7d4aa96e21624af76ef393d5ed0bf005b8c32a356483955ff7231f26267e70c'
    deadline = time.monotonic()+6*3600
    while True:
        gpu = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        used, util = [int(v.strip()) for v in gpu.split(',')]
        if used <= 100 and util == 0:break
        assert time.monotonic() < deadline, 'GPU0 free wait exceeded six hours'
        time.sleep(30)
    result['gpu0_release_check'] = gpu
    result['released_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['status'] = 'RUNNING'
    save()
    def run(argv, name):
        used = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used <= 14000, f'GPU0 has{used}MiB used'
        with Path('logs',name+'.log').open('x') as log:
            p = subprocess.Popen([sys.executable,'-u',*argv],env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
            print('PROCESS',0,p.pid,argv,flush=True)
            assert p.wait() == 0, f'failed:{argv}'
    def mask(parent, input_root, prefix):
        records = {}
        for airport in ['LIRF','EDDF']:
            folder = Path('out',prefix+'-mask-'+airport.lower())
            run(['code/predict_mask_legacy_060.py','--inputs',str(input_root),'--fit',str(parent),'--airport',airport,'--out',str(folder)],folder.name)
            rec = json.loads((folder/'PREDICT.json').read_text())
            assert rec['fit_sha256'] == sha(parent/'FIT.json') and rec['label_free'] and rec['unaffected_predictions_exact']
            assert rec['peak_gpu_bytes'] <= 10*(1<<30)
            assert sha(folder/'jjrefit.parquet') == rec['files']['jjrefit.parquet']
            records[airport] = str(folder.resolve())
        joint = Path('out',prefix+'-mask-joint')
        run(['code/join_masks.py','--fit',str(parent),'--lirf',records['LIRF'],'--eddf',records['EDDF'],'--out',str(joint)],joint.name)
        records['joint'] = str(joint.resolve())
        records['joint_PREDICT_sha256'] = sha(joint/'PREDICT.json')
        return records
    def fit(role, pin, final):
        reference = json.loads(Path(pin['fit']).read_text())
        settings = reference['args']
        seed = settings['seed']
        domain = 'annual' if final else 'held'
        folder = Path('out',f'fixed-{role}-{seed}-{domain}-060')
        assert not folder.exists()
        source = 'fit.py' if role == 'control6' else 'fit_runway.py'
        root, input_sha = inputs[role]
        cmd = ['code/'+source,'--inputs',str(root),'--out',str(folder),'--mode',settings['mode'],'--loss','huber',
               '--epochs','12','--batch-size',str(settings['batch_size']),'--seed',str(seed),'--device','cuda']
        if final:cmd.append('--final')
        run(cmd,folder.name)
        rec = json.loads((folder/'FIT.json').read_text())
        for key in ['mode','loss','epochs','batch_size','seed']:
            assert rec['args'][key] == settings[key]
        assert rec['args']['final'] == final and rec['args']['device'] == 'cuda'
        assert rec['code_sha256'] == PINS[source] and rec['inputs_manifest_sha256'] == input_sha
        meta = metadata[role]
        train = (meta.year == 2025) & (True if final else ~meta.month.isin([1,7]))
        ids = meta.loc[train,'MVT_ID_mvt'].to_numpy('int64')
        assert rec['train_rows'] == len(ids) and rec['train_ids_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest()
        assert rec['train_months'] == (list(range(1,13)) if final else [2,3,4,5,6,8,9,10,11,12])
        assert [h['epoch'] for h in rec['history']] == list(range(1,13))
        assert all(np.isfinite(h['train_loss']) for h in rec['history']) and rec['peak_gpu_bytes'] <= 10*(1<<30)
        for name,digest in rec['files'].items():assert sha(folder/name) == digest
        vector = 'ranking2026.parquet' if final else 'jjrefit.parquet'
        frame = pd.read_parquet(folder/vector)
        score_rows = meta.year.eq(2026) if final else meta.year.eq(2025)&meta.month.isin([1,7])
        assert frame.MVT_ID_mvt.dtype == np.int64 and frame.MVT_ID_mvt.is_unique
        assert np.array_equal(frame.MVT_ID_mvt.to_numpy(),meta.loc[score_rows,'MVT_ID_mvt'].to_numpy())
        assert np.isfinite(frame.pred_seq170).all() and frame.pred_seq170.min() >= 60
        record = dict(path=str(folder.resolve()),FIT_sha256=sha(folder/'FIT.json'),model_sha256=sha(folder/'model.pt'),
                      vector_sha256=sha(folder/vector),seed=seed,reference_FIT_sha256=pin['sha256'],seconds=rec['seconds'])
        if not final:record['masks'] = mask(folder,root,folder.name)
        return record
    def mean(role, parents, final):
        if len(parents) == 1:return {**parents[0], 'parents':parents, 'single_parent':True}
        domain = 'ranking2026' if final else 'jjrefit'
        folder = Path('out',f'fixed-{role}-bag6-'+('annual' if final else 'held')+'-060')
        helper = 'average_rows_021.py' if role == 'control6' else 'average_seeds_n.py'
        argv = ['code/'+helper,'--parents',*[p['path'] for p in parents],'--domain',domain,'--out',str(folder)]
        if role != 'control6':argv += ['--kind','fits']
        run(argv,folder.name)
        fit_rec = json.loads((folder/'FIT.json').read_text())
        assert fit_rec['n_seeds'] == 6 and fit_rec['args']['seed'] == list(range(17001,17007))
        assert sha(folder/(domain+'.parquet')) == fit_rec['files'][domain+'.parquet']
        record = dict(path=str(folder.resolve()),FIT_sha256=sha(folder/'FIT.json'),vector_sha256=sha(folder/(domain+'.parquet')),parents=parents)
        if not final:
            record['masks'] = {}
            for airport in ['LIRF','EDDF']:
                out = Path(str(folder)+'-mask-'+airport.lower())
                helper = 'mean_row_masks_060.py' if role == 'control6' else 'average_seeds_n.py'
                argv = ['code/'+helper,'--fit',str(folder),'--parents',*[p['masks'][airport] for p in parents],'--out',str(out)]
                if role != 'control6':argv += ['--kind','masks']
                run(argv,out.name)
                record['masks'][airport] = str(out.resolve())
            joint = Path(str(folder)+'-mask-joint')
            run(['code/join_masks.py','--fit',str(folder),'--lirf',record['masks']['LIRF'],'--eddf',record['masks']['EDDF'],'--out',str(joint)],joint.name)
            record['masks']['joint'] = str(joint.resolve())
            record['masks']['joint_PREDICT_sha256'] = sha(joint/'PREDICT.json')
        return record
    try:
        for final in [False,True]:
            domain = 'annual' if final else 'held'
            for role in ['sequence021','fallback_sequence011','control6']:
                parents = []
                result[domain][role] = dict(parents=parents,status='RUNNING')
                for pin in originals[role]:
                    parents.append(fit(role,pin,final));save()
                result[domain][role] = {**mean(role,parents,final), 'status':'COMPLETE'}
                save()
            first = result[domain]['control6']['parents'][0]
            result[domain]['fallback_control011'] = {**first,'single_parent':True,'reused_from':'control6-seed17001','status':'COMPLETE'}
            result[domain+'_complete'] = True
            save()
            print(domain.upper()+'_COMPLETE',flush=True)
        result['status'] = 'PASS'
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    except Exception as exc:
        result['status'] = 'FAIL';result['error'] = f'{type(exc).__name__}: {exc}';save();raise

if __name__ == '__main__':
    try:main()
    except Exception as exc:
        path=Path('results/FINAL-LEGACY-REBUILD-060.json')
        if path.exists():
            value=json.loads(path.read_text());value['status']='FAIL';value['error']=f'{type(exc).__name__}: {exc}'
            temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
        raise
