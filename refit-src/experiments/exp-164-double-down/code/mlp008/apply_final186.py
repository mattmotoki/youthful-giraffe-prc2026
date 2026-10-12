"""exp-186: the saved MLP008 `final` models (seeds 701, 702) score the 2026 rows of prepared `final` inputs.

apply.py replays unit 007's models on the jjrank arrays; this entry does the same for the two models of the
`final` fit (out/mlp-008/final). Ranking mode: the inputs are inputs/mlp-008/final, and each seed and the mean
must reproduce the saved predictions within 1e-3 s. Final mode (prepare.py --fit final with
PRC_RANKING_MODE=final, inputs/mlp-008-final/final): the training arrays and the preprocessing must equal those
of inputs/mlp-008/final byte for byte, so the models meet the encoding they were fitted on; the Jan/Jul rows
are compared with the saved predictions as a record (the final file moves some Jan/Jul inputs). No label is
read; no weight changes. CPU by default (an MLP of 512-512-256 on 0.7 M rows).

    PRC_RANKING_MODE=final EXP164_WORK=<work> python code/mlp008/apply_final186.py [--device cuda] [--threads 2]
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fit import Net
from io_utils import W, ID, sha, write, artifact

TOL = 1e-3
SEEDS = [701, 702]
TRAIN_SIDE = ['train-num', 'train-missing', 'train-cat', 'train-target', 'train-metadata', 'preprocessing']


def difference(pred, ids, months, old_path, strict, what, tol=TOL):
    old = pd.read_parquet(old_path)
    s = old.set_index(ID).prediction
    shared = np.isin(ids, s.index.to_numpy())
    d = pred[shared] - s.reindex(ids[shared]).to_numpy('float64')
    rec = {'shared_rows': int(shared.sum()), 'saved_rows': int(len(s)),
           'max_abs_s': float(np.abs(d).max()) if len(d) else None,
           'rms_s_by_month': {m: float(np.sqrt(np.mean(d[months[shared] == m] ** 2)))
                              for m in sorted(set(months[shared]))}}
    if strict:
        if not np.array_equal(old[ID].to_numpy('int64'), ids):
            raise RuntimeError(what + ': row order differs from the saved prediction')
        if rec['max_abs_s'] > tol:
            raise RuntimeError(f"{what}: replay difference {rec['max_abs_s']} s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tol', type=float, default=TOL, help='ranking mode: largest replay difference in s')
    a = ap.parse_args()
    if not 1 <= a.threads <= 4:
        raise RuntimeError('thread cap out of range')
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    device = torch.device(a.device)
    mode = os.environ.get('PRC_RANKING_MODE') or 'ranking'
    final_mode = mode != 'ranking'
    ref_inp = W / 'inputs/mlp-008/final'
    inp = W / ('inputs/mlp-008-final' if final_mode else 'inputs/mlp-008') / 'final'
    fit_dir = W / 'out/mlp-008/final'
    out = W / ('out/mlp-008-final186' if final_mode else 'out/mlp-008-replay186') / 'final'
    if (out / 'APPLY.json').exists():
        raise RuntimeError('write-once')
    m = json.loads((inp / 'INPUTS.json').read_text())
    if m['fit'] != 'final' or (final_mode and m.get('ranking_mode') != mode):
        raise RuntimeError('inputs belong to another fit or mode')
    for r in m['artifacts'].values():
        if sha(r['path']) != r['sha256']:
            raise RuntimeError('input artifact pin differs: ' + r['path'])
    ref = json.loads((ref_inp / 'INPUTS.json').read_text())
    for k in TRAIN_SIDE:
        if m['artifacts'][k]['sha256'] != ref['artifacts'][k]['sha256']:
            raise RuntimeError('training side differs from the inputs of the saved models: ' + k)
    fit = json.loads((fit_dir / 'FIT.json').read_text())
    if fit['fit'] != 'final' or fit['seeds'] != SEEDS:
        raise RuntimeError('fit receipt differs')
    setting = m['features']['full']; ni = setting['numeric_indices']; ci = setting['categorical_indices']
    pp = json.loads((inp / 'preprocessing.json').read_text())
    sizes = [pp['category_sizes'][i] for i in ci]
    arrays = {k: np.load(inp / f'held-{k}.npy', mmap_mode='r', allow_pickle=False) for k in ['num', 'missing', 'cat']}
    meta = pd.read_parquet(inp / 'held-metadata.parquet')
    if 'y' in meta or len(meta) != len(arrays['num']):
        raise RuntimeError('held metadata differs or carries a label')
    months = meta.month.astype(str).to_numpy()
    if final_mode and sorted(set(months)) != sorted(m['rank_months']):
        raise RuntimeError(f'held months {sorted(set(months))} differ from the mode months {m["rank_months"]}')
    ids = meta[ID].to_numpy('int64')
    out.mkdir(parents=True, exist_ok=True)
    batch = 8192; preds = []; seeds = {}
    for receipt in fit['seed_records']:
        if sha(receipt['path']) != receipt['sha256']:
            raise RuntimeError('seed receipt differs')
        rr = json.loads(Path(receipt['path']).read_text()); seed = rr['seed']
        if sha(rr['model']['path']) != rr['model']['sha256'] or rr['inputs_sha256'] != sha(ref_inp / 'INPUTS.json'):
            raise RuntimeError('saved model or its inputs differ')
        state = torch.load(rr['model']['path'], map_location='cpu', weights_only=True)
        if state['category_sizes'] != sizes or state['numeric_count'] != len(ni):
            raise RuntimeError('saved architecture differs from the preprocessing')
        model = Net(sizes, len(ni)); model.load_state_dict(state['state_dict'], strict=True); model = model.to(device).eval()
        z = np.empty(len(meta), dtype='float64')
        with torch.inference_mode():
            for start in range(0, len(meta), batch):
                sl = slice(start, start + batch)
                c = torch.from_numpy(np.array(arrays['cat'][sl][:, ci], dtype='int64', copy=True)).to(device)
                x = torch.from_numpy(np.array(arrays['num'][sl][:, ni], dtype='float32', copy=True)).to(device)
                mi = torch.from_numpy(np.array(arrays['missing'][sl][:, ni], dtype='float32', copy=True)).to(device)
                z[sl] = model(c, x, mi).cpu().numpy().astype('float64') * 1000
        pred = np.maximum(60, meta.anchor.to_numpy('float64') + z)
        if not np.isfinite(pred).all():
            raise RuntimeError('nonfinite prediction')
        seeds[str(seed)] = difference(pred, ids, months, rr['prediction']['path'], not final_mode, f'seed {seed}', a.tol)
        f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = pred
        f.to_parquet(out / f'seed-{seed}-held.parquet', index=False)
        preds.append(pred); del model, state
    if [int(s) for s in seeds] != SEEDS:
        raise RuntimeError('seed order differs')
    mean = np.mean(preds, axis=0)
    mean_rec = difference(mean, ids, months, fit['prediction']['path'], not final_mode, 'mean', a.tol)
    f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = mean
    p = out / 'mean-ranking.parquet'; f.to_parquet(p, index=False)
    write(out / 'APPLY.json', {'fit': 'final', 'ranking_mode': mode, 'models': 'MLP008 final seeds 701, 702 (twelve-month fit)',
        'fit_receipt': artifact(fit_dir / 'FIT.json'), 'inputs_sha256': sha(inp / 'INPUTS.json'),
        'training_side_equal_to': artifact(ref_inp / 'INPUTS.json'), 'code_sha256': sha(__file__),
        'rows': len(meta), 'rows_by_month': pd.Series(months).value_counts().sort_index().to_dict(),
        'against_saved': {'seeds': seeds, 'mean': mean_rec}, 'prediction_ranking': artifact(p),
        'device': a.device, 'threads': a.threads, 'tol_s': a.tol, 'torch': torch.__version__, 'no_label_read': True})
    print(json.dumps({'event': 'apply-ready', 'mode': mode, 'rows': len(meta), 'mean': mean_rec}), flush=True)


if __name__ == '__main__':
    main()
