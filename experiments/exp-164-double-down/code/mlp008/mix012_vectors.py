"""Unit 012: the mixed-recipe vectors (e80 seeds 701-706 and drop0 seeds 701-706, twelve models, equal weights); reads no label.

For each name (final, the seven jjx months, janjul, jjrank): the equal mean of the two six-seed vectors
of vectors011.py (out/mlp-011/vectors/<name>-e80b6.parquet and <name>-drop0b6.parquet, both pinned by
their BAG records), which is the equal mean of the twelve seed files. Rows and order must be equal. The
janjul vector must equal 0.5*e80x6 + 0.5*drop0x6 of the unit 012 read. Output
out/mlp-012/vectors/<name>-mixb12.parquet and BAG-mix.json. Write-once.

    python3 mix012_vectors.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from io_utils import W, ID, sha, write, artifact

NAMES = ('final', 'jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12', 'janjul', 'jjrank')
ARMS = ('e80', 'drop0')
COLS = [ID, 'month', 'utc_day', 'airport']

def need(name, ok, **facts):
    if not ok:
        raise RuntimeError(f'{name}: {facts}')

def main():
    src = W / 'out/mlp-011/vectors'
    out = W / 'out/mlp-012/vectors'; out.mkdir(parents=True, exist_ok=True)
    rec_path = out / 'BAG-mix.json'
    need('write-once record', not rec_path.exists())
    bags = {arm: json.loads((src / f'BAG-{arm}.json').read_text()) for arm in ARMS}
    by = {arm: {v['name']: v for v in bags[arm]['vectors']} for arm in ARMS}
    recs = []
    for name in NAMES:
        frames = {}
        for arm in ARMS:
            v = by[arm][name]
            need(f'{arm} {name} pin and seeds', sha(v['path']) == v['sha256'] and v['seeds'] == [701, 702, 703, 704, 705, 706])
            frames[arm] = pd.read_parquet(v['path'])
        need('same rows in the same order ' + name, frames['e80'][COLS].equals(frames['drop0'][COLS]))
        m = frames['e80'][COLS].copy()
        m['prediction'] = 0.5 * frames['e80'].prediction.to_numpy('float64') + 0.5 * frames['drop0'].prediction.to_numpy('float64')
        need('finite floored ' + name, np.isfinite(m.prediction).all() and m.prediction.min() >= 60)
        if name == 'janjul':
            e6 = pd.read_parquet(W / 'out/mlp-010/vectors/e80x6-janjul.parquet')
            d6 = pd.read_parquet(W / 'out/mlp-010/vectors/drop0x6-janjul.parquet')
            need('janjul equals the read mix', e6[COLS].equals(m[COLS]) and d6[COLS].equals(m[COLS])
                 and np.array_equal(m.prediction.to_numpy(), 0.5 * e6.prediction.to_numpy('float64') + 0.5 * d6.prediction.to_numpy('float64')))
        p = out / f'{name}-mixb12.parquet'
        need('write-once ' + name, not p.exists())
        m.to_parquet(p, index=False)
        recs.append({'name': name, 'path': str(p), 'sha256': sha(p), 'rows': int(len(m)),
                     'members': {arm: {'path': by[arm][name]['path'], 'sha256': by[arm][name]['sha256'], 'seeds': by[arm][name]['seeds'],
                                       'seed_files': by[arm][name]['seed_files'], 'checks': by[arm][name]['checks']} for arm in ARMS},
                     'weights': {arm: 0.5 for arm in ARMS}, 'note': by['e80'][name]['note']})
    write(rec_path, {'what': 'mixed-recipe MLP vectors: e80 and drop0, six seeds each (701-706), twelve models with equal weights',
                     'arms': list(ARMS), 'bags': {arm: artifact(src / f'BAG-{arm}.json') for arm in ARMS}, 'vectors': recs,
                     'code_sha256': sha(__file__), 'no_label_read': True})
    print(json.dumps({'event': 'mix-ready', 'record': str(rec_path), 'sha256': sha(rec_path), 'rows': {r['name']: r['rows'] for r in recs}}), flush=True)

if __name__ == '__main__':
    main()
