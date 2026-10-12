"""exp-186: the saved mix12 models (mlp-010 arms e80 and drop0, `final` fit, seeds 701-706 each) score the 2026 rows.

mix012_vectors.py's `final` vector is 0.5 x e80 bag + 0.5 x drop0 bag; each bag (vectors011.py) is the equal mean
of seeds 701-706 from out/mlp-010/<arm>/final and out/mlp-010/<arm>/final-seeds-703-704-705-706. fit_sweep.py
fitted all twelve models on inputs/mlp-008/final (INPUTS.json 230f9eb8, the inputs of the MLP008 `final` models);
each FIT.json and seed receipt must name that manifest. This entry replays them as apply_final186.py replays
MLP008. Ranking mode: inputs/mlp-008/final; each seed, each bag and the mix must reproduce the saved files
within 1e-3 s. Final mode (prepare.py --fit final with PRC_RANKING_MODE=final, inputs/mlp-008-final/final): the
training arrays and the preprocessing must equal those of inputs/mlp-008/final byte for byte; the Jan/Jul rows
against the saved files are a record. No label is read; no weight changes. CPU by default.

    PRC_RANKING_MODE=final EXP164_WORK=<work> python code/mlp008/apply_mix12_186.py [--device cuda] [--threads 2]
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from apply_final186 import TOL, TRAIN_SIDE, difference
from fit_sweep import Net
from io_utils import W, ID, sha, write, artifact

ARMS = ('e80', 'drop0')
SEEDS = [701, 702, 703, 704, 705, 706]
SUBS = ('final', 'final-seeds-703-704-705-706')


def need(name, ok):
    if not ok:
        raise RuntimeError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tol', type=float, default=TOL, help='ranking mode: largest replay difference in s')
    a = ap.parse_args()
    need('thread cap out of range', 1 <= a.threads <= 4)
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    device = torch.device(a.device)
    mode = os.environ.get('PRC_RANKING_MODE') or 'ranking'
    final_mode = mode != 'ranking'
    ref_inp = W / 'inputs/mlp-008/final'
    inp = W / ('inputs/mlp-008-final' if final_mode else 'inputs/mlp-008') / 'final'
    out = W / ('out/mlp-012-final186' if final_mode else 'out/mlp-012-replay186') / 'final'
    need('write-once', not (out / 'APPLY.json').exists())
    m = json.loads((inp / 'INPUTS.json').read_text())
    need('inputs belong to another fit or mode', m['fit'] == 'final' and (not final_mode or m.get('ranking_mode') == mode))
    for r in m['artifacts'].values():
        need('input artifact pin differs: ' + r['path'], sha(r['path']) == r['sha256'])
    ref = json.loads((ref_inp / 'INPUTS.json').read_text())
    for k in TRAIN_SIDE:
        need('training side differs from the inputs of the saved models: ' + k, m['artifacts'][k]['sha256'] == ref['artifacts'][k]['sha256'])
    ref_sha = sha(ref_inp / 'INPUTS.json')
    mix_rec = json.loads((W / 'out/mlp-012/vectors/BAG-mix.json').read_text())
    saved_mix = [v for v in mix_rec['vectors'] if v['name'] == 'final']
    need('one saved final mix vector', len(saved_mix) == 1 and sha(saved_mix[0]['path']) == saved_mix[0]['sha256'])
    saved_mix = saved_mix[0]
    setting = m['features']['full']; ni = setting['numeric_indices']; ci = setting['categorical_indices']
    pp = json.loads((inp / 'preprocessing.json').read_text())
    sizes = [pp['category_sizes'][i] for i in ci]
    arrays = {k: np.load(inp / f'held-{k}.npy', mmap_mode='r', allow_pickle=False) for k in ['num', 'missing', 'cat']}
    meta = pd.read_parquet(inp / 'held-metadata.parquet')
    need('held metadata differs or carries a label', 'y' not in meta and len(meta) == len(arrays['num']))
    months = meta.month.astype(str).to_numpy()
    need(f'held months {sorted(set(months))} differ from the mode months',
         not final_mode or sorted(set(months)) == sorted(m['rank_months']))
    ids = meta[ID].to_numpy('int64')
    out.mkdir(parents=True, exist_ok=True)
    batch = 8192; bags = {}; recs = {}
    for arm in ARMS:
        member = saved_mix['members'][arm]
        need(f'{arm}: saved bag pin and seeds', sha(member['path']) == member['sha256'] and member['seeds'] == SEEDS)
        receipts, fits = [], []
        for sub in SUBS:
            fit_path = W / 'out/mlp-010' / arm / sub / 'FIT.json'
            fit = json.loads(fit_path.read_text())
            need(f'{arm}/{sub}: fit receipt differs', fit['fit'] == 'final' and fit['arm'] == arm and fit['inputs_sha256'] == ref_sha)
            fits.append(artifact(fit_path))
            for receipt in fit['seed_records']:
                need('seed receipt differs', sha(receipt['path']) == receipt['sha256'])
                receipts.append((fit['config'], json.loads(Path(receipt['path']).read_text())))
        need(f'{arm}: seeds 701-706 in order', [rr['seed'] for _, rr in receipts] == SEEDS)
        need(f'{arm}: seed files are the saved bag members',
             [(f['seed'], f['path'], f['sha256']) for f in member['seed_files']]
             == [(rr['seed'], rr['prediction']['path'], rr['prediction']['sha256']) for _, rr in receipts])
        preds, seeds = [], {}
        for cfg, rr in receipts:
            seed = rr['seed']
            need('saved model or its inputs differ', sha(rr['model']['path']) == rr['model']['sha256'] and rr['inputs_sha256'] == ref_sha
                 and rr['arm'] == arm)
            state = torch.load(rr['model']['path'], map_location='cpu', weights_only=True)
            need('saved architecture differs from the preprocessing or the fit config',
                 state['category_sizes'] == sizes and state['numeric_count'] == len(ni) and state['arm'] == arm and state['config'] == cfg)
            model = Net(sizes, len(ni), cfg['hidden'], cfg['dropout'], cfg['emb_mult'])
            model.load_state_dict(state['state_dict'], strict=True); model = model.to(device).eval()
            z = np.empty(len(meta), dtype='float64')
            with torch.inference_mode():
                for start in range(0, len(meta), batch):
                    sl = slice(start, start + batch)
                    c = torch.from_numpy(np.array(arrays['cat'][sl][:, ci], dtype='int64', copy=True)).to(device)
                    x = torch.from_numpy(np.array(arrays['num'][sl][:, ni], dtype='float32', copy=True)).to(device)
                    mi = torch.from_numpy(np.array(arrays['missing'][sl][:, ni], dtype='float32', copy=True)).to(device)
                    z[sl] = model(c, x, mi).cpu().numpy().astype('float64') * 1000
            pred = np.maximum(60, meta.anchor.to_numpy('float64') + z)
            need('nonfinite prediction', np.isfinite(pred).all())
            seeds[str(seed)] = difference(pred, ids, months, rr['prediction']['path'], not final_mode, f'{arm} seed {seed}', a.tol)
            f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = pred
            f.to_parquet(out / f'{arm}-seed-{seed}-held.parquet', index=False)
            preds.append(pred); del model, state
        bags[arm] = np.mean(np.vstack(preds), axis=0)       # vectors011.bag
        bag_rec = difference(bags[arm], ids, months, member['path'], not final_mode, f'{arm} bag', a.tol)
        f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = bags[arm]
        p = out / f'{arm}-b6.parquet'; f.to_parquet(p, index=False)
        recs[arm] = {'fit_receipts': fits, 'config': receipts[0][0], 'seeds': seeds, 'bag': bag_rec, 'bag_prediction': artifact(p)}
    mix = 0.5 * bags['e80'] + 0.5 * bags['drop0']           # mix012_vectors
    need('finite floored mix', np.isfinite(mix).all() and mix.min() >= 60)
    mix_diff = difference(mix, ids, months, saved_mix['path'], not final_mode, 'mix', a.tol)
    f = meta[[ID, 'month', 'utc_day', 'airport']].copy(); f['prediction'] = mix
    p = out / 'mean-ranking.parquet'; f.to_parquet(p, index=False)
    write(out / 'APPLY.json', {'fit': 'final', 'ranking_mode': mode,
        'models': 'mix12: mlp-010 e80 and drop0, final fit, seeds 701-706 each; 0.5 x e80 bag + 0.5 x drop0 bag',
        'fitted_on': {'folder': str(ref_inp), 'inputs_sha256': ref_sha, 'arms': list(ARMS),
                      'note': 'every FIT.json and seed receipt of e80 and drop0 names this manifest'},
        'saved_mix': artifact(Path(saved_mix['path'])), 'saved_mix_record': artifact(W / 'out/mlp-012/vectors/BAG-mix.json'),
        'inputs_sha256': sha(inp / 'INPUTS.json'), 'training_side_equal_to': artifact(ref_inp / 'INPUTS.json'),
        'code_sha256': sha(__file__), 'rows': len(meta), 'rows_by_month': pd.Series(months).value_counts().sort_index().to_dict(),
        'against_saved': {'arms': recs, 'mix': mix_diff}, 'prediction_ranking': artifact(p),
        'device': a.device, 'threads': a.threads, 'tol_s': a.tol, 'torch': torch.__version__, 'no_label_read': True})
    print(json.dumps({'event': 'apply-ready', 'mode': mode, 'rows': len(meta), 'mix': mix_diff}), flush=True)


if __name__ == '__main__':
    main()
