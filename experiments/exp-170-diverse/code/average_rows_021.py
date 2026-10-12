"""Equal mean of n checked row-control fits (seqc unit 021).

The row form of `average_seeds_n.py` (fits mode): parents are `fit.py --mode row` fits that differ
only in the seed. The receipt has the keys that `paired.py` reads from a fit (`train_months`,
`files`) and names every parent; it does not claim to be a single model. Write-once output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ID = 'MVT_ID_mvt'
KIND = 'equal n-seed row-control prediction mean'

ap = argparse.ArgumentParser()
ap.add_argument('--parents', type=Path, nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--domain', choices=['jjrefit', 'ranking2026'], default='jjrefit')
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)


def read_vector(path, ids=None):
    frame = pd.read_parquet(path)
    assert frame[ID].is_unique
    if ids is not None:
        assert set(frame[ID]) == set(ids)
        frame = frame.set_index(ID).loc[ids].reset_index()
    values = frame.pred_seq170.to_numpy(float)
    assert np.isfinite(values).all() and (values >= 60.).all()
    return frame, values


parents = []
for path in a.parents:
    rec = json.loads((path / 'FIT.json').read_text())
    assert rec['args']['mode'] == 'row' and rec['args']['loss'] == 'huber'
    assert rec['args']['epochs'] == 12 and rec['args']['batch_size'] == 128
    assert rec['args']['final'] == (a.domain == 'ranking2026')
    assert rec['code_sha256'] == sha(Path(__file__).with_name('fit.py'))
    assert sha(path / 'model.pt') == rec['files']['model.pt']
    assert sha(path / (a.domain + '.parquet')) == rec['files'][a.domain + '.parquet']
    parents.append({'path': str(path.resolve()), 'fit_sha256': sha(path / 'FIT.json'),
                    'model_sha256': sha(path / 'model.pt'), 'vector_sha256': sha(path / (a.domain + '.parquet')),
                    'seed': rec['args']['seed'], 'inputs': rec['args']['inputs'], 'weight': 1 / len(a.parents), 'fit': rec})
n = len(parents)
assert n >= 2 and len({p['fit_sha256'] for p in parents}) == n and len({p['seed'] for p in parents}) == n
base = parents[0]['fit']
for p in parents:
    for key in ['train_months', 'train_rows', 'code_sha256', 'train_ids_sha256', 'inputs_manifest_sha256']:
        assert p['fit'][key] == base[key], key
frame, first = read_vector(Path(parents[0]['path']) / (a.domain + '.parquet'))
ids = frame[ID].to_numpy('int64')
values = [first] + [read_vector(Path(p['path']) / (a.domain + '.parquet'), ids)[1] for p in parents[1:]]
frame['pred_seq170'] = np.mean(values, axis=0)
frame.to_parquet(a.out / (a.domain + '.parquet'), index=False)
checked, got = read_vector(a.out / (a.domain + '.parquet'), ids)
assert np.array_equal(got, frame.pred_seq170)
result = {'kind': KIND, 'n_seeds': n, 'args': {**base['args'], 'seed': [p['seed'] for p in parents], 'out': str(a.out)},
          'parents': parents, 'train_months': base['train_months'], 'train_rows': base['train_rows'],
          'train_ids_sha256': base['train_ids_sha256'], 'inputs_manifest_sha256': base['inputs_manifest_sha256'],
          'parent_code_sha256': base['code_sha256'], 'code_sha256': sha(__file__),
          'label_free_pairwise_rms_seconds': {str(i) + '-' + str(j):
              float(np.sqrt(np.mean((values[i] - values[j]) ** 2))) for i in range(n) for j in range(i + 1, n)},
          'files': {a.domain + '.parquet': sha(a.out / (a.domain + '.parquet'))}}
(a.out / 'FIT.json').write_text(json.dumps(result, indent=1) + '\n')
print('row mean of', n, [p['seed'] for p in parents],
      'pairwise rms', {k: round(v, 1) for k, v in result['label_free_pairwise_rms_seconds'].items()}, flush=True)
