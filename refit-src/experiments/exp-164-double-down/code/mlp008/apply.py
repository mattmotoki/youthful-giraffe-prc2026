"""Unit 008 jjrank: unit 007's two saved full models score the jjrank arrays (jjrefit rows, then the 2026
ranking rows). The jjrefit part must reproduce unit 007's saved held predictions; no label is read.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fit import Net
from io_utils import W, ID, sha, write, artifact, lane

REF_FIT_SHA = '34c5b7334ca7b1cd9b2a7df54d85f2f588d0e5e0fd257cd43f77e5c76fa7a9e6'
TOL = 1e-3

def main():
    cores = lane(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('GPU 1 only under the current grant')
    inp = W / 'inputs/mlp-008/jjrank'; ref_out = W / 'out/mlp-007/full'; out = W / 'out/mlp-008/jjrank'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'APPLY.json').exists():
        raise RuntimeError('write-once')
    m = json.loads((inp / 'INPUTS.json').read_text())
    check = json.loads((W / 'results/mlp-008/CHECK-ARRAYS-jjrank.json').read_text())
    if check['state'] != 'pass' or check['manifest_sha256'] != sha(inp / 'INPUTS.json'):
        raise RuntimeError('actual input check missing or for other inputs')
    if sha(ref_out / 'FIT.json') != REF_FIT_SHA:
        raise RuntimeError('unit 007 fit receipt differs')
    for r in m['artifacts'].values():
        if sha(r['path']) != r['sha256']:
            raise RuntimeError('input artifact pin differs: ' + r['path'])
    fit = json.loads((ref_out / 'FIT.json').read_text())
    pp = json.loads((inp / 'preprocessing.json').read_text())
    sizes = pp['category_sizes']; nnum = len(pp['numeric'])
    arrays = {k: np.load(inp / f'held-{k}.npy', mmap_mode='r', allow_pickle=False) for k in ['num', 'missing', 'cat']}
    meta = pd.read_parquet(inp / 'held-metadata.parquet')
    fold = ~meta.month.isin(['2026-01', '2026-07']).to_numpy()
    nf = int(fold.sum())
    if not fold[:nf].all():
        raise RuntimeError('jjrefit rows are not first')
    batch = 8192; preds = []; seeds = []; replay = {}
    for receipt in fit['seed_records']:
        if sha(receipt['path']) != receipt['sha256']:
            raise RuntimeError('seed receipt differs')
        rr = json.loads(Path(receipt['path']).read_text()); seed = rr['seed']; seeds.append(seed)
        if sha(rr['model']['path']) != rr['model']['sha256'] or sha(rr['prediction']['path']) != rr['prediction']['sha256']:
            raise RuntimeError('unit 007 model or prediction differs')
        state = torch.load(rr['model']['path'], map_location='cpu', weights_only=True)
        if state['category_sizes'] != sizes or state['numeric_count'] != nnum:
            raise RuntimeError('saved architecture differs from the jjrank preprocessing')
        model = Net(sizes, nnum); model.load_state_dict(state['state_dict'], strict=True); model = model.cuda().eval()
        z = np.empty(len(meta), dtype='float64')
        with torch.inference_mode():
            for start in range(0, len(meta), batch):
                sl = slice(start, start + batch)
                c = torch.from_numpy(np.array(arrays['cat'][sl], dtype='int64', copy=True)).to('cuda:0')
                x = torch.from_numpy(np.array(arrays['num'][sl], dtype='float32', copy=True)).to('cuda:0')
                mi = torch.from_numpy(np.array(arrays['missing'][sl], dtype='float32', copy=True)).to('cuda:0')
                z[sl] = model(c, x, mi).cpu().numpy().astype('float64') * 1000
        pred = np.maximum(60, meta.anchor.to_numpy('float64') + z)
        if not np.isfinite(pred).all():
            raise RuntimeError('nonfinite prediction')
        old = pd.read_parquet(rr['prediction']['path'])
        if not np.array_equal(old[ID].to_numpy('int64'), meta[ID].to_numpy('int64')[:nf]):
            raise RuntimeError('jjrefit IDs differ from unit 007 held order')
        err = np.abs(pred[:nf] - old.prediction.to_numpy())
        replay[str(seed)] = {'rows': nf, 'max_abs_s': float(err.max()), 'rms_s': float(np.sqrt(np.mean(err ** 2))), 'tolerance_s': TOL}
        if err.max() > TOL:
            raise RuntimeError(f'seed {seed} replay error {err.max()} s')
        f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = pred
        p = out / f'seed-{seed}-held.parquet'; f.to_parquet(p, index=False)
        preds.append(pred); del model, state; torch.cuda.empty_cache()
    if seeds != [701, 702]:
        raise RuntimeError('seed order differs')
    mean = np.mean(preds, axis=0)
    ref_mean = pd.read_parquet(fit['prediction']['path'])
    merr = np.abs(mean[:nf] - ref_mean.prediction.to_numpy())
    if merr.max() > TOL:
        raise RuntimeError(f'mean replay error {merr.max()} s')
    f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = mean
    p = out / 'mean-held.parquet'; f.to_parquet(p, index=False)
    r = f.loc[~fold].reset_index(drop=True); pr = out / 'mean-ranking.parquet'; r.to_parquet(pr, index=False)
    write(out / 'APPLY.json', {'fit': 'jjrank', 'models': 'unit 007 full seeds 701, 702 (ten-month jjrank training)',
        'unit007_fit': artifact(ref_out / 'FIT.json'), 'inputs_sha256': sha(inp / 'INPUTS.json'), 'code_sha256': sha(__file__),
        'replay': replay, 'mean_replay_max_abs_s': float(merr.max()), 'jjrefit_rows': nf, 'ranking_rows': int((~fold).sum()),
        'prediction_all': artifact(p), 'prediction_ranking': artifact(pr), 'cores': cores, 'no_label_read': True})
    print(json.dumps({'event': 'apply-ready', 'replay': replay, 'mean_replay_max_abs_s': float(merr.max()),
                      'ranking_rows': int((~fold).sum())}), flush=True)

if __name__ == '__main__':
    main()
