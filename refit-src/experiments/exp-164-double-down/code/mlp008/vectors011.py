"""Unit 011: the six-seed vectors (seeds 701-706, equal weights) of one fit_sweep arm on every fit; reads no label.

bag_vectors.py (unit 009) for an arm of fit_sweep.py: final and the seven jjx months from
out/mlp-010/<arm>/<fit> (seeds 701/702) and <fit>-seeds-703-704-705-706; jjrank from <arm>/jjrank,
jjrank-seeds-703-704 and jjrank-seeds-705-706 (all on unit 007's training arrays; held part = jjrefit
rows then ranking rows). Every fit must have a passing check (results/mlp-011/CHECK-<arm>-<sub>.json,
check_sweep.py) that binds its FIT.json; every seed file is pinned through its receipt; the six seeds
are 701-706, each once. The janjul vector must equal unit 010's six-seed fold vector of the arm
(out/mlp-010/vectors/<arm>x6-janjul.parquet). Output out/mlp-011/vectors/<name>-<arm>b6.parquet and
BAG-<arm>.json. Write-once.

    python3 vectors011.py --arm e80
    python3 vectors011.py --arm drop0 --jjrank-subs jjrank jjrank-seeds-703-704-705-706   (unit 012)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from io_utils import W, ID, sha, write, artifact

JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
COLS = [ID, 'month', 'utc_day', 'airport']
JJREFIT_ROWS = 344017
SEEDS = [701, 702, 703, 704, 705, 706]

def need(name, ok, **facts):
    if not ok:
        raise RuntimeError(f'{name}: {facts}')

def sources(arm, fit, subs):
    files, checks = {}, []
    for sub in subs:
        fj = W / 'out/mlp-010' / arm / sub / 'FIT.json'
        cj = W / 'results/mlp-011' / f'CHECK-{arm}-{sub}.json'
        fit_rec, c = json.loads(fj.read_text()), json.loads(cj.read_text())
        need('fit ' + sub, fit_rec['fit'] == fit and fit_rec['arm'] == arm)
        need('check passes and binds the fit ' + sub, c['state'] == 'pass' and c['fit_receipt']['sha256'] == sha(fj) and c['sub'] == sub)
        checks.append(artifact(cj))
        for r in fit_rec['seed_records']:
            need('seed receipt pin', sha(r['path']) == r['sha256'])
            rr = json.loads(Path(r['path']).read_text())
            need('seed prediction pin', sha(rr['prediction']['path']) == rr['prediction']['sha256'])
            need('one file per seed', rr['seed'] not in files)
            files[rr['seed']] = Path(rr['prediction']['path'])
    need('seeds 701-706 ' + fit, sorted(files) == SEEDS, seeds=sorted(files))
    return [(s, files[s]) for s in SEEDS], checks

def bag(src):
    frames = [pd.read_parquet(p) for _, p in src]
    first = frames[0]
    for f in frames[1:]:
        need('same rows in the same order', f[COLS].equals(first[COLS]))
    v = first[COLS].copy()
    v['prediction'] = np.mean(np.vstack([f.prediction.to_numpy('float64') for f in frames]), axis=0)
    need('finite floored', np.isfinite(v.prediction).all() and v.prediction.min() >= 60)
    return v

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True)
    ap.add_argument('--jjrank-subs', nargs='+', default=['jjrank', 'jjrank-seeds-703-704', 'jjrank-seeds-705-706']); a = ap.parse_args()
    out = W / 'out/mlp-011/vectors'; out.mkdir(parents=True, exist_ok=True)
    rec_path = out / f'BAG-{a.arm}.json'
    need('write-once record', not rec_path.exists())
    recs = []

    def emit(name, v, src, checks, note):
        p = out / f'{name}-{a.arm}b6.parquet'
        need('write-once ' + name, not p.exists())
        v.to_parquet(p, index=False)
        recs.append({'name': name, 'path': str(p), 'sha256': sha(p), 'rows': int(len(v)), 'seeds': [s for s, _ in src],
                     'seed_files': [{'seed': s, **artifact(f)} for s, f in src], 'checks': checks, 'note': note})

    for fit in ('final',) + JJX:
        src, checks = sources(a.arm, fit, [fit, fit + '-seeds-703-704-705-706'])
        emit(fit, bag(src), src, checks, 'ranking rows' if fit == 'final' else 'held month')
    src, checks = sources(a.arm, 'jjrank', a.jjrank_subs)
    v = bag(src)
    u7 = pd.read_parquet(W / 'out/mlp-007/full/mean-held.parquet')
    jj, rk = v.iloc[:JJREFIT_ROWS].reset_index(drop=True), v.iloc[JJREFIT_ROWS:].reset_index(drop=True)
    need('janjul rows are unit 007 held rows in order', jj[COLS].equals(u7[COLS]))
    need('jjrank rows are 2026 ranking rows', len(rk) == 344453 and set(rk.month.astype(str)) == {'2026-01', '2026-07'})
    u10 = pd.read_parquet(W / 'out/mlp-010/vectors' / f'{a.arm}x6-janjul.parquet')
    need('janjul equals the unit 010 six-seed fold vector', u10[COLS].equals(jj[COLS]) and np.array_equal(u10.prediction.to_numpy(), jj.prediction.to_numpy()))
    emit('janjul', jj, src, checks, 'jjrefit part of the jjrank fit (unit 007 training arrays)')
    emit('jjrank', rk, src, checks, 'ranking part of the jjrank fit (unit 007 training arrays)')
    write(rec_path, {'what': f'six-seed MLP vectors of arm {a.arm} (701-706, equal weights; fit_sweep)', 'arm': a.arm, 'vectors': recs,
                     'unit010_fold_vector': artifact(W / 'out/mlp-010/vectors' / f'{a.arm}x6-janjul.parquet'),
                     'code_sha256': sha(__file__), 'no_label_read': True})
    print(json.dumps({'event': 'bag-ready', 'record': str(rec_path), 'sha256': sha(rec_path), 'rows': {r['name']: r['rows'] for r in recs}}), flush=True)

if __name__ == '__main__':
    main()
