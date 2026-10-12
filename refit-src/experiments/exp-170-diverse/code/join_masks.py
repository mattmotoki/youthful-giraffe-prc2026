"""Join disjoint airport input masks before any weight refit."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ID = 'MVT_ID_mvt'


def main():
    ap = argparse.ArgumentParser()
    for name in ['fit', 'lirf', 'eddf', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    fit = json.loads((a.fit / 'FIT.json').read_text())
    path = a.fit / 'jjrefit.parquet'
    assert sha(path) == fit['files']['jjrefit.parquet']
    frame = pd.read_parquet(path)
    assert frame[ID].is_unique
    ids = frame[ID].to_numpy('int64')
    original = frame.pred_seq170.to_numpy(float).copy()
    output = original.copy()
    parents = []
    for apt, root in [('LIRF', a.lirf), ('EDDF', a.eddf)]:
        rec = json.loads((root / 'PREDICT.json').read_text())
        assert rec['fit_sha256'] == sha(a.fit / 'FIT.json')
        assert rec['label_free'] and rec['unaffected_predictions_exact']
        assert rec['masked_airport'] == apt and rec['masked_month'] == 7
        assert sha(root / 'jjrefit.parquet') == rec['files']['jjrefit.parquet']
        f = pd.read_parquet(root / 'jjrefit.parquet').set_index(ID)
        assert f.index.is_unique and set(f.index) == set(ids)
        vals = f.pred_seq170.reindex(ids).to_numpy(float)
        use = frame.ADEP_mvt.eq(apt).to_numpy() & frame.month.eq(7).to_numpy()
        assert np.array_equal(vals[~use], original[~use])
        output[use] = vals[use]
        parents.append({'airport': apt, 'path': str(root.resolve()), 'receipt_sha256': sha(root / 'PREDICT.json'),
                        'vector_sha256': sha(root / 'jjrefit.parquet')})
    assert np.isfinite(output).all() and (output >= 60.).all()
    frame.pred_seq170 = output
    frame.to_parquet(a.out / 'jjrefit.parquet', index=False)
    rec = {'kind': 'joint disjoint-airport input mask', 'fit_sha256': sha(a.fit / 'FIT.json'),
           'input_manifest_sha256': fit['inputs_manifest_sha256'], 'masked_airports': ['LIRF', 'EDDF'],
           'masked_month': 7, 'label_free': True, 'unaffected_predictions_exact': True,
           'parents': parents, 'code_sha256': sha(__file__),
           'files': {'jjrefit.parquet': sha(a.out / 'jjrefit.parquet')}}
    (a.out / 'PREDICT.json').write_text(json.dumps(rec, indent=1) + '\n')
    print('joint mask ready', sha(a.out / 'jjrefit.parquet'), flush=True)


if __name__ == '__main__':
    main()
