"""Equal mean of n checked runway-sequence fits that may differ in recipe, held or annual (seqc unit 033).

Copy of `average_fits_026.py` with `--domain` (jjrefit: ten-month fits, held vector; ranking2026:
twelve-month fits, 2026 vector) and with the parent's seed and weights form in the receipt.

The form of `average_seeds_n.py` (fits mode) without its fixed-recipe asserts: parents may come
from `fit_runway.py` and from `fit_runway_size_026.py`. Every parent has the same training rows and
the same input manifest; the receipt lists each parent's arguments and code hash and does not claim
one recipe. Write-once output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ID = 'MVT_ID_mvt'
ap = argparse.ArgumentParser()
ap.add_argument('--parents', type=Path, nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--domain', choices=['jjrefit', 'ranking2026'], default='jjrefit')
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
VECTOR = a.domain + '.parquet'


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
    assert rec['args']['mode'] == 'sequence-runway' and rec['args']['loss'] == 'huber' and rec['args']['final'] == (a.domain == 'ranking2026')
    assert sha(path / 'model.pt') == rec['files']['model.pt']
    assert sha(path / VECTOR) == rec['files'][VECTOR]
    parents.append({'path': str(path.resolve()), 'fit_sha256': sha(path / 'FIT.json'), 'model_sha256': sha(path / 'model.pt'),
                    'vector_sha256': sha(path / VECTOR), 'args': rec['args'], 'code_sha256': rec['code_sha256'],
                    'seed': rec['args']['seed'], 'weights_form': rec.get('weights'),
                    'weight': 1 / len(a.parents), 'fit': rec})
n = len(parents)
assert n >= 2 and len({p['fit_sha256'] for p in parents}) == n and len({p['model_sha256'] for p in parents}) == n
assert len({(p['seed'], p['args']['width'], p['args']['layers'], p['args']['epochs']) for p in parents}) == n
base = parents[0]['fit']
for p in parents:
    for key in ['train_months', 'train_rows', 'train_ids_sha256', 'inputs_manifest_sha256']:
        assert p['fit'][key] == base[key], key
frame, first = read_vector(Path(parents[0]['path']) / VECTOR)
ids = frame[ID].to_numpy('int64')
values = [first] + [read_vector(Path(p['path']) / VECTOR, ids)[1] for p in parents[1:]]
frame['pred_seq170'] = np.mean(values, axis=0)
frame.to_parquet(a.out / VECTOR, index=False)
got = read_vector(a.out / VECTOR, ids)[1]
assert np.array_equal(got, frame.pred_seq170)
recipes = sorted({json.dumps({k: p['args'].get(k) for k in ['epochs', 'batch_size', 'width', 'layers']}, sort_keys=True) + '|' + p['code_sha256']
                  for p in parents})
result = {'kind': 'equal n-fit prediction mean, recipes may differ', 'n_fits': n, 'domain': a.domain,
          'parents': [{k: v for k, v in p.items() if k != 'fit'} for p in parents], 'recipes': recipes,
          'train_months': base['train_months'], 'train_rows': base['train_rows'], 'train_ids_sha256': base['train_ids_sha256'],
          'inputs_manifest_sha256': base['inputs_manifest_sha256'], 'code_sha256': sha(__file__),
          'label_free_pairwise_rms_seconds': {str(i) + '-' + str(j): float(np.sqrt(np.mean((values[i] - values[j]) ** 2)))
                                              for i in range(n) for j in range(i + 1, n)},
          'files': {VECTOR: sha(a.out / VECTOR)}}
(a.out / 'FIT.json').write_text(json.dumps(result, indent=1) + '\n')
print('mean of', n, 'recipes', len(recipes), flush=True)
