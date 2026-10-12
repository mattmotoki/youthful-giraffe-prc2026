"""exp-181 serve (role g): one annual fit of the AB bag on the annual Gmixed032 rows.

Same model, arguments and seed as the held full fit in fit_bag_033.py (imported unchanged): recipe
mixed64 or mixed128, seed 150081-150086, 12 fixed epochs, AdamW lr .001 wd .0001, batch 256,
SmoothL1 beta 2 on the 300 s correction, clip 1, floor 60, no upper cap. Only the rows change,
to those of the annual Gmixed032 recipe (inputs MANIFEST dc598af3...): the loss rows are all
2,085,047 departures of 2025 (twelve months; encoder stage `annual`) and the target rows are the
344,841 ranking rows of 2026 in the official order.

    python -u code/serve_bag.py --inputs <ROOT>/inputs --plan <ROOT>/SERVE-PLAN.json \
        --checks <ROOT>/ANNUAL-CHECKS.json --recipe mixed64 --seed 150081 --out <OUT>
    smoke (proves the path and the output form, not a fit): add --smoke-rows 2048 [--device cpu]

Writes <OUT>/model.pt, <OUT>/ranking.parquet (MVT_ID_mvt, pred_Gmixed) and <OUT>/FIT.json with the
source hashes (write-once). A real fit needs CUDA; the GPU allocator is capped (plan gpu_cap_bytes).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

import fit as F
from fit_bag_033 import RECIPES, make, seed

CODE = Path(__file__).resolve().parent
SOURCES = ['serve_bag.py', 'fit_bag_033.py', 'fit.py', 'model.py']
SPLIT = {'loss_year': 2025, 'loss_months': list(range(1, 13)), 'encoder_year': 2025, 'ranking_year': 2026}
FIT_FIELD = 'owner'  # receipt field that names the fit


def save(path, model, data, recipe, value):
    torch.save(dict(state_dict=model.state_dict(), recipe=recipe, seed=value, n_numeric=data.n_numeric,
                    vocab_sizes=data.vocab_sizes, encoder_stage='annual'), path)


def load(path, data, recipe, value, device):
    r = torch.load(path, map_location=device, weights_only=True)
    if (r['recipe'], r['seed'], r['encoder_stage'], r['n_numeric'], r['vocab_sizes']) != (
            recipe, value, 'annual', data.n_numeric, data.vocab_sizes):
        raise ValueError('Model contract differs')
    model = make(data, recipe).to(device)
    model.load_state_dict(r['state_dict'])
    return model


def verify(a, plan):
    pins = {n: F.sha(CODE / n) for n in SOURCES}
    if pins != plan['source_sha256']:
        raise ValueError('Source differs from SERVE-PLAN pins')
    root = a.inputs
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if F.sha(root / 'MANIFEST.json') != plan['annual_manifest_sha256']:
        raise ValueError('Annual input manifest differs')
    checks = json.loads(a.checks.read_text())
    if F.sha(a.checks) != plan['annual_checks_sha256'] or checks['status'] != 'PASS' \
            or checks['manifest_sha256'] != plan['annual_manifest_sha256']:
        raise ValueError('Annual input checks differ')
    for n, h in manifest['files'].items():
        if F.sha(root / n) != h:
            raise ValueError('Annual input hash differs: ' + n)
    if manifest['split'] != SPLIT:
        raise ValueError('Annual split differs')
    ix = np.load(root / 'train-indices.npy', allow_pickle=False)
    y = np.load(root / 'train-labels.npy', allow_pickle=False)
    ranking = np.load(root / 'ranking-indices.npy', allow_pickle=False)
    ids = np.load(root / 'ranking-ids.npy', allow_pickle=False)
    if len(ix) != plan['train_labels'] or len(y) != len(ix) or not np.isfinite(y).all() or len(np.unique(ix)) != len(ix):
        raise ValueError('Bad annual loss rows')
    if len(ids) != plan['ranking_rows'] or len(ranking) != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Bad ranking rows')
    if hashlib.sha256(ids.tobytes()).hexdigest() != plan['ranking_ID_bytes_sha256']:
        raise ValueError('Ranking ID order differs from Gmixed032')
    if len(np.intersect1d(ix, ranking)):
        raise ValueError('Ranking rows in the loss rows')
    return pins, manifest, ix, y, ranking, ids


def main():
    ap = argparse.ArgumentParser()
    for n in ['inputs', 'plan', 'checks', 'out']:
        ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--recipe', choices=RECIPES, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--smoke-rows', type=int, default=0)
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError('Write-once output exists')
    if not 1 <= a.threads <= 6:
        raise ValueError('Thread cap out of range')
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    started = time.monotonic()
    plan = json.loads(a.plan.read_text()); plan_sha = F.sha(a.plan)
    cfg = plan['recipes'][a.recipe]
    if (cfg['width'], cfg['layers'], cfg['feedforward']) != RECIPES[a.recipe] or cfg['heads'] != 4 or cfg['dropout'] != .1:
        raise ValueError('Recipe differs from the plan')
    if a.seed not in plan['seeds'] or plan['fixed_epochs'] != 12:
        raise ValueError('Seed/epochs differ')
    smoke = a.smoke_rows > 0
    if a.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA not available')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, plan['gpu_cap_bytes'] / total), 0)
        gpu = torch.cuda.get_device_name(0)
    elif not smoke:
        raise ValueError('A real serve fit runs on CUDA; CPU is for the smoke only')
    else:
        gpu = None
    device = torch.device(a.device)
    pins, manifest, ix, y, ranking, ids = verify(a, plan)
    epochs = 12
    if smoke:
        take = np.sort(np.random.default_rng(a.seed).choice(len(ix), a.smoke_rows, replace=False))
        ix, y, epochs = ix[take], y[take], 1
    a.out.mkdir(parents=True)
    data = F.Arrays(a.inputs, 'annual', 'sequence'); seed(a.seed)
    model = make(data, a.recipe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(a.seed); history = []
    for epoch in range(1, epochs + 1):
        row = dict(epoch=epoch, train=F.run_epoch(model, data, ix, y, device, opt, rng),
                   elapsed_seconds=time.monotonic() - started)
        if not np.isfinite(row['train']['huber']):
            raise ValueError('Non-finite train loss')
        history.append(row); print(json.dumps(row), flush=True)
    fit_seconds = time.monotonic() - started
    save(a.out / 'model.pt', model, data, a.recipe, a.seed)
    pred = F.predict(model, data, ranking, device)
    replay = F.predict(load(a.out / 'model.pt', data, a.recipe, a.seed, device), data, ranking, device)
    if not np.array_equal(pred, replay):
        raise ValueError('Full ranking reload differs')
    pd.DataFrame({'MVT_ID_mvt': ids, 'pred_Gmixed': pred}).to_parquet(a.out / 'ranking.parquet', index=False)
    if {n: F.sha(CODE / n) for n in SOURCES} != pins or F.sha(a.plan) != plan_sha:
        raise ValueError('Source or plan changed during the fit')
    rec = dict(status='smoke' if smoke else 'completed', smoke_rows=a.smoke_rows if smoke else None,
               unit='exp-181 serve role g', recipe=a.recipe, seed=a.seed, epochs=epochs,
               **{FIT_FIELD: 'final'}, encoder_stage='annual', full_history=history,
               train_labels=len(ix), train_months=manifest['summary']['train_rows_by_UTC_month'],
               train_split=manifest['split'], ranking_rows=len(ids),
               ranking_ID_bytes_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
               ranking_labels_in_payload=False, ranking_labels_used=False, floor=60, upper_cap=None,
               loss='SmoothL1 beta2 in300s; AdamW.001/.0001; batch256; clip1',
               model_reload_exact_all_ranking_rows=True, column='pred_Gmixed',
               held_counterpart_FIT_sha256=plan['held_counterparts_FIT_sha256'][a.recipe][str(a.seed)],
               manifest_sha256=F.sha(a.inputs / 'MANIFEST.json'), checks_sha256=F.sha(a.checks),
               plan_sha256=plan_sha, source_sha256=pins, device=a.device, gpu=gpu, threads=a.threads,
               host=platform.node(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               fit_seconds=fit_seconds, seconds=time.monotonic() - started,
               peak_GPU_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else None,
               torch=torch.__version__,
               files={p.name: F.sha(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    (a.out / 'FIT.json').write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: rec[k] for k in ('status', 'recipe', 'seed', 'epochs', 'fit_seconds', 'seconds', 'files')}), flush=True)


if __name__ == '__main__':
    main()
