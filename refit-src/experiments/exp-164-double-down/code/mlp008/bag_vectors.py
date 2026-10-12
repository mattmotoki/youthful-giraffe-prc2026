"""Unit 009: the six-seed vectors (seeds 701-706, equal weights) of each fit; reads no label.

Seeds 701/702: unit 008 fits (final, jjx-MM) and, for jjrank, unit 007's two models as unit 008's
apply.py ran them (jjrefit rows then ranking rows). Seeds 703-706: unit 009's fit_bag.py. Every seed
file is pinned through its receipt, and the checks of each fit (CHECK-FIT / APPLY, CHECK-BAG) must pass
and bind the fit receipt. Output out/mlp-009/vectors/<name>-bag6.parquet, columns as the source
vectors (ID, month, utc_day, airport, prediction), and BAG.json. Write-once.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from io_utils import W, ID, sha, write, artifact

JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
NEW = [703, 704, 705, 706]
OUT = W / 'out/mlp-009/vectors'
COLS = [ID, 'month', 'utc_day', 'airport']
UNIT007_FIT_SHA = '34c5b7334ca7b1cd9b2a7df54d85f2f588d0e5e0fd257cd43f77e5c76fa7a9e6'
JJREFIT_ROWS = 344017

def need(name, ok, **facts):
    if not ok:
        raise RuntimeError(f'{name}: {facts}')

def seeds_of(fit_json, seeds):
    fit = json.loads(Path(fit_json).read_text())
    need('seeds ' + str(fit_json), fit['seeds'] == seeds)
    out = []
    for r in fit['seed_records']:
        need('seed receipt pin', sha(r['path']) == r['sha256'])
        rr = json.loads(Path(r['path']).read_text())
        need('seed prediction pin', sha(rr['prediction']['path']) == rr['prediction']['sha256'])
        out.append((rr['seed'], Path(rr['prediction']['path'])))
    need('seed order ' + str(fit_json), [s for s, _ in out] == seeds)
    return out

def checked(check_json, fit_json):
    c = json.loads(Path(check_json).read_text())
    need('check passes ' + str(check_json), c['state'] == 'pass' and c['fit_receipt']['sha256'] == sha(fit_json))
    return artifact(check_json)

def bag(files):
    frames = [pd.read_parquet(p) for _, p in files]
    first = frames[0]
    for f in frames[1:]:
        need('same rows in the same order', f[COLS].equals(first[COLS]))
    v = first[COLS].copy()
    v['prediction'] = np.mean(np.vstack([f.prediction.to_numpy('float64') for f in frames]), axis=0)
    need('finite floored', np.isfinite(v.prediction).all() and v.prediction.min() >= 60)
    return v

def emit(name, v, sources, checks, note):
    p = OUT / f'{name}-bag6.parquet'
    need('write-once ' + name, not p.exists())
    v.to_parquet(p, index=False)
    return {'name': name, 'path': str(p), 'sha256': sha(p), 'rows': int(len(v)), 'seeds': [s for s, _ in sources],
            'seed_files': [{'seed': s, **artifact(f)} for s, f in sources], 'checks': checks, 'note': note}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    need('empty vector folder', not any(OUT.iterdir()))
    recs = []
    for fit in ('final',) + JJX:
        f8, f9 = W / 'out/mlp-008' / fit / 'FIT.json', W / 'out/mlp-009' / fit / 'FIT.json'
        checks = [checked(W / f'results/mlp-008/CHECK-FIT-{fit}.json', f8), checked(W / f'results/mlp-009/CHECK-BAG-{fit}.json', f9)]
        src = seeds_of(f8, [701, 702]) + seeds_of(f9, NEW)
        two = bag(src[:2])
        ref = pd.read_parquet(json.loads(f8.read_text())['prediction']['path'])
        need('seeds 701/702 give the unit 008 mean ' + fit, ref[COLS].equals(two[COLS]) and np.array_equal(ref.prediction.to_numpy(), two.prediction.to_numpy()))
        recs.append(emit(fit, bag(src), src, checks, 'ranking rows' if fit == 'final' else 'held month'))
    # jjrank: unit 007's models (apply.py) and the four new seeds on the same 688,470 rows
    ap_path, f9 = W / 'out/mlp-008/jjrank/APPLY.json', W / 'out/mlp-009/jjrank/FIT.json'
    ap = json.loads(ap_path.read_text())
    need('apply uses unit 007', ap['unit007_fit']['sha256'] == UNIT007_FIT_SHA == sha(W / 'out/mlp-007/full/FIT.json') and ap['mean_replay_max_abs_s'] == 0.0)
    need('apply mean pin', sha(ap['prediction_all']['path']) == ap['prediction_all']['sha256'])
    old = [(s, W / f'out/mlp-008/jjrank/seed-{s}-held.parquet') for s in (701, 702)]
    two = bag(old)
    ref = pd.read_parquet(ap['prediction_all']['path'])
    need('apply seed files give the pinned apply mean', ref[COLS].equals(two[COLS]) and np.array_equal(ref.prediction.to_numpy(), two.prediction.to_numpy()))
    for s, p in old:
        u7 = pd.read_parquet(W / f'out/mlp-007/full/seed-{s}-held.parquet')
        a7 = pd.read_parquet(p).iloc[:JJREFIT_ROWS].reset_index(drop=True)
        need(f'apply seed {s} equals unit 007 on the jjrefit rows', u7[COLS].equals(a7[COLS]) and np.array_equal(u7.prediction.to_numpy(), a7.prediction.to_numpy()))
    checks = [artifact(ap_path), checked(W / 'results/mlp-009/CHECK-BAG-jjrank.json', f9)]
    src = old + seeds_of(f9, NEW)
    v = bag(src)
    u7 = pd.read_parquet(W / 'out/mlp-007/full/mean-held.parquet')
    jj, rk = v.iloc[:JJREFIT_ROWS].reset_index(drop=True), v.iloc[JJREFIT_ROWS:].reset_index(drop=True)
    need('janjul rows are unit 007 held rows in order', jj[COLS].equals(u7[COLS]))
    need('jjrank rows are 2026 ranking rows', len(rk) == 344453 and set(rk.month.astype(str)) == {'2026-01', '2026-07'})
    recs.append(emit('janjul', jj, src, checks, 'jjrefit part of the jjrank fit (unit 007 training arrays)'))
    recs.append(emit('jjrank', rk, src, checks, 'ranking part of the jjrank fit (unit 007 training arrays)'))
    write(OUT / 'BAG.json', {'what': 'six-seed MLP vectors (701-706, equal weights), unit 007 full arm', 'vectors': recs,
                             'code_sha256': sha(__file__), 'no_label_read': True})
    print(json.dumps({'event': 'bag-ready', 'BAG_sha256': sha(OUT / 'BAG.json'), 'rows': {r['name']: r['rows'] for r in recs}}), flush=True)

if __name__ == '__main__':
    main()
