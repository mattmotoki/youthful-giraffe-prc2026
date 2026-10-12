"""Equal mean of three checked sequence fits or their inference masks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ID = 'MVT_ID_mvt'


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
                        'seed': rec['args']['seed'], 'weight': 1 / 3, 'fit': rec})
    parents.sort(key=lambda p: p['seed'])
    assert [p['seed'] for p in parents] == [17001, 17002, 17003]
    base = parents[0]['fit']
    for p in parents:
        for key in ['train_months', 'train_rows', 'inputs_manifest_sha256', 'code_sha256', 'train_ids_sha256']:
            assert p['fit'][key] == base[key], key
    frame, first = read_vector(Path(parents[0]['path']) / (a.domain + '.parquet'))
    ids = frame[ID].to_numpy('int64')
    values = [first] + [read_vector(Path(p['path']) / (a.domain + '.parquet'), ids)[1] for p in parents[1:]]
    frame['pred_seq170'] = np.mean(values, axis=0)
    frame.to_parquet(a.out / (a.domain + '.parquet'), index=False)
    checked, got = read_vector(a.out / (a.domain + '.parquet'), ids)
    assert np.array_equal(got, frame.pred_seq170)
    result = {'kind': 'equal three-seed prediction mean', 'args': {**base['args'],
              'seed': [17001, 17002, 17003], 'out': str(a.out)},
              'parents': parents, 'train_months': base['train_months'], 'train_rows': base['train_rows'],
              'train_ids_sha256': base['train_ids_sha256'], 'inputs_manifest_sha256': base['inputs_manifest_sha256'],
              'parent_code_sha256': base['code_sha256'], 'code_sha256': sha(__file__),
              'label_free_pairwise_rms_seconds': {str(parents[i]['seed']) + '-' + str(parents[j]['seed']):
                  float(np.sqrt(np.mean((values[i] - values[j]) ** 2))) for i in range(3) for j in range(i + 1, 3)},
              'files': {a.domain + '.parquet': sha(a.out / (a.domain + '.parquet'))}}
    (a.out / 'FIT.json').write_text(json.dumps(result, indent=1) + '\n')
    print('mean', result['label_free_pairwise_rms_seconds'], flush=True)


def masks(a):
    fit = json.loads((a.fit / 'FIT.json').read_text())
    assert fit['kind'] == 'equal three-seed prediction mean' and not fit['args']['final']
    by_fit = {}
    for path in a.parents:
        rec = json.loads((path / 'PREDICT.json').read_text())
        assert rec['label_free'] and rec['unaffected_predictions_exact']
        assert sha(path / 'jjrefit.parquet') == rec['files']['jjrefit.parquet']
        assert rec['fit_sha256'] not in by_fit
        by_fit[rec['fit_sha256']] = (path, rec)
    expected = [p['fit_sha256'] for p in fit['parents']]
    assert set(by_fit) == set(expected) and len(expected) == 3
    frame, original = read_vector(a.fit / 'jjrefit.parquet')
    ids = frame[ID].to_numpy('int64')
    values, parents = [], []
    first = by_fit[expected[0]][1]
    for parent in fit['parents']:
        path, rec = by_fit[parent['fit_sha256']]
        for key in ['masked_airport', 'masked_month', 'masked_numeric_columns', 'input_manifest_sha256']:
            assert rec[key] == first[key], key
        assert rec['input_manifest_sha256'] == fit['inputs_manifest_sha256']
        assert rec['model_sha256'] == parent['model_sha256']
        values.append(read_vector(path / 'jjrefit.parquet', ids)[1])
        parents.append({'path': str(path.resolve()), 'receipt_sha256': sha(path / 'PREDICT.json'),
                        'vector_sha256': sha(path / 'jjrefit.parquet'), 'fit_sha256': rec['fit_sha256'], 'weight': 1 / 3})
    frame.pred_seq170 = np.mean(values, axis=0)
    unchanged = ~((frame.ADEP_mvt == first['masked_airport']) & (frame.month == first['masked_month'])).to_numpy()
    assert np.array_equal(frame.pred_seq170.to_numpy()[unchanged], original[unchanged])
    frame.to_parquet(a.out / 'jjrefit.parquet', index=False)
    receipt = {'kind': 'equal three-seed masked prediction mean', 'fit_sha256': sha(a.fit / 'FIT.json'),
               'input_manifest_sha256': fit['inputs_manifest_sha256'], 'masked_airport': first['masked_airport'],
               'masked_month': first['masked_month'], 'masked_numeric_columns': first['masked_numeric_columns'],
               'label_free': True, 'unaffected_predictions_exact': True, 'parents': parents,
               'code_sha256': sha(__file__), 'files': {'jjrefit.parquet': sha(a.out / 'jjrefit.parquet')}}
    (a.out / 'PREDICT.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print('masked mean', first['masked_airport'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['fits', 'masks'], required=True)
    ap.add_argument('--parents', type=Path, nargs=3, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--domain', choices=['jjrefit', 'ranking2026'], default='jjrefit')
    ap.add_argument('--fit', type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (fits if a.kind == 'fits' else masks)(a)


if __name__ == '__main__':
    main()
