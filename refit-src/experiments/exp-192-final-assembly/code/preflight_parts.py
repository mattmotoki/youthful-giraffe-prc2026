"""Preflight of a final sources JSON (exp-192, finalplan, 2026-10-08): the checks of assemble192.read_part before the
build, plus finite rows by month. For each part: SHA-256, the ID column in template order (final_submitting.parquet),
the named column present; then the finite rows of each column by 2026 month.

Run (remote, guarded): PRC_DATA_DIR=<final raw> python -I code/preflight_parts.py <repo> <sources json> <out json>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
repo, sources_path, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
from prc import data as D  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


template = pd.read_parquet(D.submitting_file('final'))
ids = template[ID].to_numpy('int64')
rank = pd.read_parquet(D.ranking_file('final'), columns=[ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
rank = rank[rank.PHASE_mvt.eq('DEP')].set_index(ID)
month = rank.MVT_TIME_UTC_mvt.dt.month.reindex(ids).to_numpy()
assert len(ids) == D.n_ranking('final') and not np.isnan(month.astype(float)).any()
sources = json.loads(sources_path.read_text())
files = {}
for key, cols in sources.items():
    if key == 'dry_run':
        continue
    specs = {'mlp008': {**cols, 'column': 'mlp008'}, 'mix12': {**cols, 'column': 'mix12'}} if key == 'mlp_part' else cols
    for name, s in specs.items():
        files.setdefault((s['path'], s['sha256']), []).append((key, name, s['column']))
res = {'sources': str(sources_path), 'sources_sha256': sha256_file(sources_path), 'rows': len(ids), 'parts': {}}
ok = True
for (path, want), uses in files.items():
    got = sha256_file(path)
    q = pd.read_parquet(path)
    r = {'sha256_ok': got == want, 'rows': len(q), 'ids_template_order': bool(len(q) == len(ids) and np.array_equal(
        q[ID].to_numpy().astype('int64'), ids)), 'columns': {}}
    for key, name, col in uses:
        c = {'frame': key, 'present': col in q.columns}
        if c['present'] and len(q) == len(ids):
            v = q[col]
            fin = v.notna().to_numpy() if v.dtype == object else np.isfinite(v.to_numpy(float))
            c['dtype'] = str(v.dtype)
            c['finite_by_month'] = {str(m): int((fin & (month == m)).sum()) for m in (1, 2, 6, 7)}
        r['columns'][f'{key}:{name}'] = c
        ok &= c['present']
    ok &= r['sha256_ok'] and r['ids_template_order']
    res['parts'][path] = r
res['all_ok'] = bool(ok)
out.write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res, indent=1))
assert ok, 'preflight failed'
