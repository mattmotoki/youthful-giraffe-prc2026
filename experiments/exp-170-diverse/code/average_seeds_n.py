"""Equal mean of n checked sequence fits or of their inference masks (n >= 2).

The n-seed form of `average_seeds.py` (three seeds, kept as run). Parents may come from more than
one input recipe (`--mixed-inputs`); the receipt then lists each parent's input manifest and does
not claim one manifest for the mean.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ID = 'MVT_ID_mvt'
KIND = 'equal n-seed prediction mean'


def read_vector(path, ids=None):
    frame = pd.read_parquet(path)
    assert frame[ID].is_unique
    if ids is not None:
        assert set(frame[ID]) == set(ids)
        frame = frame.set_index(ID).loc[ids].reset_index()
    values = frame.pred_seq170.to_numpy(float)
    assert np.isfinite(values).all() and (values >= 60.).all()
    return frame, values


def fits(a):
    parents = []
    for path in a.parents:
        rec = json.loads((path / 'FIT.json').read_text())
        assert rec['args']['mode'] == 'sequence-runway' and rec['args']['loss'] == 'huber'
        assert rec['args']['epochs'] == 12 and rec['args']['batch_size'] == 512
        assert rec['args']['final'] == (a.domain == 'ranking2026')
        assert sha(path / 'model.pt') == rec['files']['model.pt']
        assert sha(path / (a.domain + '.parquet')) == rec['files'][a.domain + '.parquet']
        parents.append({'path': str(path.resolve()), 'fit_sha256': sha(path / 'FIT.json'),
                        'model_sha256': sha(path / 'model.pt'), 'vector_sha256': sha(path / (a.domain + '.parquet')),
                        'seed': rec['args']['seed'], 'inputs': rec['args']['inputs'], 'weight': 1 / len(a.parents), 'fit': rec})
    n = len(parents)
    assert n >= 2 and len({p['fit_sha256'] for p in parents}) == n
    assert len({(p['seed'], p['fit']['inputs_manifest_sha256']) for p in parents}) == n
    base = parents[0]['fit']
    manifests = sorted({p['fit']['inputs_manifest_sha256'] for p in parents})
    assert a.mixed_inputs or len(manifests) == 1
    for p in parents:
        for key in ['train_months', 'train_rows', 'code_sha256', 'train_ids_sha256']:
            assert p['fit'][key] == base[key], key
    frame, first = read_vector(Path(parents[0]['path']) / (a.domain + '.parquet'))
    ids = frame[ID].to_numpy('int64')
    values = [first] + [read_vector(Path(p['path']) / (a.domain + '.parquet'), ids)[1] for p in parents[1:]]
    frame['pred_seq170'] = np.mean(values, axis=0)
    frame.to_parquet(a.out / (a.domain + '.parquet'), index=False)
    checked, got = read_vector(a.out / (a.domain + '.parquet'), ids)
    assert np.array_equal(got, frame.pred_seq170)
    result = {'kind': KIND, 'n_seeds': n, 'args': {**base['args'], 'seed': [p['seed'] for p in parents],
              'inputs': [p['inputs'] for p in parents], 'out': str(a.out)},
              'parents': parents, 'train_months': base['train_months'], 'train_rows': base['train_rows'],
              'train_ids_sha256': base['train_ids_sha256'],
              'inputs_manifest_sha256': manifests[0] if len(manifests) == 1 else None,
              'parent_inputs_manifest_sha256': manifests,
              'parent_code_sha256': base['code_sha256'], 'code_sha256': sha(__file__),
              'label_free_pairwise_rms_seconds': {str(i) + '-' + str(j):
                  float(np.sqrt(np.mean((values[i] - values[j]) ** 2))) for i in range(n) for j in range(i + 1, n)},
              'files': {a.domain + '.parquet': sha(a.out / (a.domain + '.parquet'))}}
    (a.out / 'FIT.json').write_text(json.dumps(result, indent=1) + '\n')
    print('mean of', n, [(p['seed'], p['inputs']) for p in parents], flush=True)


def masks(a):
    fit = json.loads((a.fit / 'FIT.json').read_text())
    assert fit['kind'] == KIND and not fit['args']['final']
    by_fit = {}
    for path in a.parents:
        rec = json.loads((path / 'PREDICT.json').read_text())
        assert rec['label_free'] and rec['unaffected_predictions_exact']
        assert sha(path / 'jjrefit.parquet') == rec['files']['jjrefit.parquet']
        assert rec['fit_sha256'] not in by_fit
        by_fit[rec['fit_sha256']] = (path, rec)
    expected = [p['fit_sha256'] for p in fit['parents']]
    assert set(by_fit) == set(expected) and len(expected) == fit['n_seeds'] == len(a.parents)
    frame, original = read_vector(a.fit / 'jjrefit.parquet')
    ids = frame[ID].to_numpy('int64')
    values, parents = [], []
    first = by_fit[expected[0]][1]
    for parent in fit['parents']:
        path, rec = by_fit[parent['fit_sha256']]
        for key in ['masked_airport', 'masked_month', 'masked_numeric_columns']:
            assert rec[key] == first[key], key
        assert rec['input_manifest_sha256'] == parent['fit']['inputs_manifest_sha256']
        assert rec['model_sha256'] == parent['model_sha256']
        values.append(read_vector(path / 'jjrefit.parquet', ids)[1])
        parents.append({'path': str(path.resolve()), 'receipt_sha256': sha(path / 'PREDICT.json'),
                        'vector_sha256': sha(path / 'jjrefit.parquet'), 'fit_sha256': rec['fit_sha256'],
                        'weight': 1 / len(a.parents)})
    frame.pred_seq170 = np.mean(values, axis=0)
    unchanged = ~((frame.ADEP_mvt == first['masked_airport']) & (frame.month == first['masked_month'])).to_numpy()
    assert np.array_equal(frame.pred_seq170.to_numpy()[unchanged], original[unchanged])
    frame.to_parquet(a.out / 'jjrefit.parquet', index=False)
    receipt = {'kind': 'equal n-seed masked prediction mean', 'n_seeds': len(a.parents), 'fit_sha256': sha(a.fit / 'FIT.json'),
               'input_manifest_sha256': fit['inputs_manifest_sha256'], 'masked_airport': first['masked_airport'],
               'masked_month': first['masked_month'], 'masked_numeric_columns': first['masked_numeric_columns'],
               'label_free': True, 'unaffected_predictions_exact': True, 'parents': parents,
               'code_sha256': sha(__file__), 'files': {'jjrefit.parquet': sha(a.out / 'jjrefit.parquet')}}
    (a.out / 'PREDICT.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print('masked mean', first['masked_airport'], len(a.parents), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['fits', 'masks'], required=True)
    ap.add_argument('--parents', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--domain', choices=['jjrefit', 'ranking2026'], default='jjrefit')
    ap.add_argument('--fit', type=Path)
    ap.add_argument('--mixed-inputs', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (fits if a.kind == 'fits' else masks)(a)


if __name__ == '__main__':
    main()
