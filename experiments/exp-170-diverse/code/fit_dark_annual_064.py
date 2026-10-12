"""Fit the frozen eighteen annual dark parents once; emit their equal LAST mean."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
from check_time_day_052 import sha

FAST = '4e86dc93cf5a698b1bc87fc875ccb998a7243eb4c91df228551d1c51b6724714'
AVERAGE = '082826acd35d2569c97fc552c56d98157e2fd76b4a14c2eee8617ab4b0edac1b'
INPUT = '8bd9349b11b477d819c77fca20a494a677495896aa2fa79ca97468af58798c76'
SPEC = '5ca546c4b7a580534c422cd750f2f952087a46c09a7ca185dcabaabdfc0effa7'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-review', type=Path, required=True)
    ap.add_argument('--input-review-sha', required=True)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    assert sha(a.input_review) == a.input_review_sha == '8f8b9ca349d13a294ee26a22d9a07189087af6dd0ebda74e70daa993bf325bdb'
    audit = json.loads(a.input_review.read_text())
    assert audit['status'] == 'PASS' and not audit['failures']
    assert len(audit['checks']) == 13 and all(c['ok'] is True for c in audit['checks'].values())
    assert audit['input']['manifest_sha256'] == INPUT
    assert sha('SPEC-064.md') == SPEC
    assert sha('code/fit_runway_fast_029.py') == FAST
    assert sha('code/average_size_027.py') == AVERAGE
    inputs = Path('inputs/coherent-dark-Z-annual-064')
    assert sha(inputs / 'MANIFEST.json') == INPUT
    manifest = json.loads((inputs / 'MANIFEST.json').read_text())
    assert manifest['darkday_training_domain'] == '2025_all'
    assert manifest['time_day_form'] == 'Z' and manifest['darkday_seed'] == 603
    assert manifest['masked_rows'] == 180629 and manifest['masked_airport_dates'] == 328
    assert manifest['all_other_columns_exact'] and manifest['nontraining_rows_exact'] and manifest['fixed57_116_exact']
    assert len(set(manifest['darkday_masked_indices'])) == 76
    assert not set(manifest['darkday_masked_indices']) & {57, 116}
    for name, digest in manifest['files'].items():
        assert sha(inputs / name) == digest, name
    meta = pd.read_parquet(inputs / 'rows.parquet', columns=['MVT_ID_mvt', 'year', 'month'])
    assert len(meta) == 2429888 and meta.MVT_ID_mvt.dtype == np.int64 and meta.MVT_ID_mvt.is_unique
    train_ids = meta.loc[meta.year.eq(2025), 'MVT_ID_mvt'].to_numpy('int64')
    rank_ids = meta.loc[meta.year.eq(2026), 'MVT_ID_mvt'].to_numpy('int64')
    assert len(train_ids) == 2085047 and len(rank_ids) == 344841
    train_sha = hashlib.sha256(train_ids.tobytes()).hexdigest()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ[key] == '2'
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19 and set(os.sched_getaffinity(0)) == {14, 15}
    assert all(not Path('out', f'darkday-Z-{seed}-annual-064').exists() for seed in range(17001, 17019))
    if a.preflight_only:
        print('PREFLIGHT_PASS: eighteen pinned annual dark r48 seeds, reviewed input, LAST equal mean', flush=True)
        return
    receipt = Path('results/DARK-ANNUAL-064.json')
    assert not receipt.exists()
    Path('logs/dark-annual-064.claim').mkdir(exist_ok=False)
    result = dict(status='RUNNING', started_utc=now(), source_row=902, source_sha256=sha(__file__),
                  spec_sha256=SPEC, input_manifest_sha256=INPUT, input_review_path=str(a.input_review),
                  input_review_sha256=a.input_review_sha, stage='LAST; retain EMA and EMA2',
                  annual=True, quality_read=False, pack_admission=False, masks_requested=False,
                  expected_end_utc='2026-10-04T00:30:00Z', processes=[], parents=[], streams={}, errors=[])
    lock = threading.Lock()

    def save():
        temp = receipt.with_suffix('.json.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(receipt)

    save()

    def run_fit(seed, gpu):
        folder = Path('out', f'darkday-Z-{seed}-annual-064')
        assert not folder.exists(), 'no duplicate or retry'
        used, total, util = map(int, subprocess.check_output([
            'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(','))
        ollama = subprocess.check_output(['ollama', 'ps'], text=True)
        load = os.getloadavg()
        check = dict(checked_utc=now(), gpu=gpu, memory_used_mib=used, memory_total_mib=total,
                     utilization_percent=util, load=list(load), ollama_ps=ollama)
        assert used <= 100 and total - used >= 10240, check
        assert len(ollama.strip().splitlines()) == 1, 'Ollama model loaded; no start'
        assert load[0] <= 32, check
        argv = [sys.executable, '-u', 'code/fit_runway_fast_029.py', '--inputs', str(inputs),
                '--out', str(folder), '--mode', 'sequence-runway', '--loss', 'huber',
                '--epochs', '48', '--batch-size', '512', '--seed', str(seed), '--width', '128',
                '--layers', '3', '--ema', '.999', '--ema2', '.9997', '--device', 'cuda', '--final']
        logfile = Path('logs', folder.name + '.log')
        with logfile.open('x') as log:
            p = subprocess.Popen(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            process = dict(seed=seed, gpu=gpu, pid=p.pid, pgid=os.getpgid(p.pid),
                           start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],
                           argv=argv, log=str(logfile), started_utc=now(), preflight=check)
            with lock:
                result['processes'].append(process)
                save()
            print('START', gpu, seed, p.pid, flush=True)
            exit_code = p.wait()
            with lock:
                process.update(exit_code=exit_code, finished_utc=now())
                save()
            assert exit_code == 0, f'failed seed{seed}; no retry'
        stages = []
        for stage, weights in [('last', 'last iterate'), ('ema2', 'exponential mean, decay 0.9997'), ('', 'exponential mean, decay 0.999')]:
            root = folder / stage
            rec = json.loads((root / 'FIT.json').read_text())
            for key, value in dict(seed=seed, width=128, epochs=48, layers=3, batch_size=512,
                                   loss='huber', mode='sequence-runway', ema=.999, ema2=.9997,
                                   final=True, device='cuda').items():
                assert rec['args'][key] == value, key
            assert rec['weights'] == weights and rec['code_sha256'] == FAST and rec['inputs_manifest_sha256'] == INPUT
            assert rec['train_rows'] == len(train_ids) and rec['train_ids_sha256'] == train_sha
            assert rec['train_months'] == list(range(1, 13)) and rec['window'] == dict(group='airport-day-runway', halo=4, core=4)
            assert [h['epoch'] for h in rec['history']] == list(range(1, 49))
            assert all(np.isfinite(h['train_loss']) for h in rec['history']) and rec['peak_gpu_bytes'] <= 10 * (1 << 30)
            for name, digest in rec['files'].items():
                assert sha(root / name) == digest, name
            frame = pd.read_parquet(root / 'ranking2026.parquet')
            assert frame.MVT_ID_mvt.dtype == np.int64 and frame.MVT_ID_mvt.is_unique
            assert np.array_equal(frame.MVT_ID_mvt.to_numpy(), rank_ids)
            assert np.isfinite(frame.pred_seq170).all() and frame.pred_seq170.min() >= 60
            stages.append(dict(stage=stage or 'ema', path=str(root.resolve()), FIT_sha256=sha(root / 'FIT.json'),
                               model_sha256=sha(root / 'model.pt'), vector_sha256=sha(root / 'ranking2026.parquet')))
        parent = dict(seed=seed, gpu=gpu, path=str((folder / 'last').resolve()),
                      FIT_sha256=stages[0]['FIT_sha256'], vector_sha256=stages[0]['vector_sha256'], stages=stages)
        with lock:
            result['parents'].append(parent)
            save()
        return parent

    def stream(gpu):
        seeds = range(17001, 17010) if gpu == 0 else range(17010, 17019)
        parents = [run_fit(seed, gpu) for seed in seeds]
        with lock:
            result['streams'][str(gpu)] = dict(status='PASS', seeds=list(seeds), finished_utc=now())
            save()
        return parents

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {gpu: pool.submit(stream, gpu) for gpu in [0, 1]}
        for gpu, job in futures.items():
            try:
                job.result()
            except Exception as exc:
                with lock:
                    result['errors'].append(f'GPU{gpu}:{type(exc).__name__}:{exc}')
                    save()
    if not result['errors']:
        parents = sorted(result['parents'], key=lambda p: p['seed'])
        assert [p['seed'] for p in parents] == list(range(17001, 17019))
        folder = Path('out/dark18-Z-annual-064')
        with Path('logs/dark18-Z-annual-mean-064.log').open('x') as log:
            subprocess.run([sys.executable, '-u', 'code/average_size_027.py', '--kind', 'fits',
                            '--domain', 'ranking2026', '--parents', *[p['path'] for p in parents],
                            '--out', str(folder)], check=True, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT)
        mean = json.loads((folder / 'FIT.json').read_text())
        assert mean['n_seeds'] == 18 and mean['args']['seed'] == list(range(17001, 17019))
        assert mean['inputs_manifest_sha256'] == INPUT and mean['args']['final'] is True
        assert all(p['weight'] == 1 / 18 for p in mean['parents'])
        assert sha(folder / 'ranking2026.parquet') == mean['files']['ranking2026.parquet']
        result['mean'] = dict(path=str(folder.resolve()), FIT_sha256=sha(folder / 'FIT.json'),
                              vector_sha256=sha(folder / 'ranking2026.parquet'), n_seeds=18)
    result.update(status='FAIL' if result['errors'] else 'PASS', finished_utc=now())
    save()
    print(json.dumps(dict(status=result['status'], mean=result.get('mean'), errors=result['errors'])), flush=True)
    assert not result['errors'], result['errors']


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        path = Path('results/DARK-ANNUAL-064.json')
        if path.exists():
            value = json.loads(path.read_text())
            value.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_utc=now())
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(value, indent=2) + '\n')
            temp.replace(path)
        raise
