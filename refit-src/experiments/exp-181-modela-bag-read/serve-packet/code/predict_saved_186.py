"""exp-186: predict-only entry for the saved g serve fits (serve_bag.py, role g); no training.

serve_bag.py always trains. This entry loads each saved fit's model.pt with serve_bag.load (the model
contract: recipe, seed, encoder stage `annual`, numeric width and vocabulary sizes of the inputs) and scores the
inputs' ranking rows with fit.predict. serve_bag.verify checks the plan, the annual inputs, their checks and the
ranking ID order, so the plan names the mode's rows: SERVE-PLAN `ranking_rows` and `ranking_ID_bytes_sha256`
(344,841 board rows, or 670,790 final rows from final annual inputs). Ranking mode: --plan is the board
SERVE-PLAN with its own inputs and checks; each vector must equal the fit's own ranking.parquet within --tol
(default 1e-3 s). Final mode: the plan of the final inputs comes from the board plan (FINAL-PLAN.json in --out):
annual_manifest_sha256 and annual_checks_sha256 of the final inputs and checks (check_annual_186.py), the
ranking rows and ID bytes of prc.data.load_submitting(); the final inputs must hold the train-only encoder and
the loss labels of the board inputs (--saved-inputs, by the board plan's manifest SHA-256), and the final checks
must come from the board's checker (--board-checks). The Jan/Jul rows against the saved vectors are a record.
--part writes the equal mean of the fits (the rule of the release g vector 7581e5f0, REPORT-SERVE.md) as column
pred_Gmixed, float64, one row per template row in template order.

    PRC_RANKING_MODE=final python code/predict_saved_186.py --inputs <final annual inputs> --plan SERVE-PLAN.json
        --checks <final checks> --saved-inputs <board annual inputs> --board-checks ANNUAL-CHECKS.json
        --fits <fit>... --out <dir> [--part <g.parquet>] [--device cuda]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fit as F
import serve_bag as S

ROOT = Path(__file__).resolve().parents[4]
SAME = ['encoder-annual.json', 'train-labels.npy']


def final_plan(a, board, D):
    """The board SERVE-PLAN with the final inputs, checks and template rows."""
    if a.saved_inputs is None or a.board_checks is None:
        raise SystemExit('final mode: give --saved-inputs and --board-checks')
    if F.sha(a.saved_inputs / 'MANIFEST.json') != board['annual_manifest_sha256']:
        raise ValueError('--saved-inputs are not the inputs of the board plan')
    if F.sha(a.board_checks) != board['annual_checks_sha256']:
        raise ValueError('--board-checks are not the checks of the board plan')
    saved = json.loads((a.saved_inputs / 'MANIFEST.json').read_text())
    new = json.loads((a.inputs / 'MANIFEST.json').read_text())
    for n in SAME:
        if new['files'][n] != saved['files'][n] or F.sha(a.saved_inputs / n) != saved['files'][n]:
            raise ValueError('final inputs differ from the board inputs in ' + n)
    checks = json.loads(a.checks.read_text())
    if checks['checker_sha256'] != json.loads(a.board_checks.read_text())['checker_sha256'] \
            or checks['manifest_sha256'] != F.sha(a.inputs / 'MANIFEST.json') or checks['ranking_rows'] != D.n_ranking():
        raise ValueError('final checks do not come from the board checker on these inputs')
    ids = D.load_submitting()[D.ID].to_numpy('int64')
    plan = dict(board)
    plan.update({'annual_manifest_sha256': F.sha(a.inputs / 'MANIFEST.json'), 'annual_checks_sha256': F.sha(a.checks),
                 'ranking_rows': int(len(ids)), 'ranking_ID_bytes_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                 'exp186': {'ranking_mode': D.ranking_mode(), 'board_plan': str(a.plan), 'board_plan_sha256': F.sha(a.plan),
                            'board_annual_manifest_sha256': board['annual_manifest_sha256'],
                            'saved_inputs': str(a.saved_inputs), 'same_files': SAME,
                            'inputs': str(a.inputs), 'checks': str(a.checks), 'submitting_file': str(D.submitting_file())}})
    return plan


def main():
    ap = argparse.ArgumentParser()
    for n in ['inputs', 'plan', 'checks', 'out']:
        ap.add_argument('--' + n, type=Path, required=True)
    ap.add_argument('--fits', type=Path, nargs='+', required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tol', type=float, default=1e-3, help='ranking mode: largest difference to the saved vector, s')
    ap.add_argument('--saved-inputs', type=Path, help='final mode: the board annual inputs of --plan')
    ap.add_argument('--board-checks', type=Path, help='final mode: the board ANNUAL-CHECKS of --plan')
    ap.add_argument('--part', type=Path, help='write the equal mean as a part (MVT_ID_mvt, pred_Gmixed)')
    a = ap.parse_args()
    if a.out.exists() or (a.part is not None and a.part.exists()):
        raise ValueError('Write-once output exists')
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    mode = os.environ.get('PRC_RANKING_MODE') or 'ranking'
    plan = json.loads(a.plan.read_text())
    sys.path.insert(1, str(ROOT))
    from prc import data as D
    if not hasattr(D, 'load_submitting'):
        raise SystemExit(f'{D.__file__} has no ranking modes')
    if D.ranking_mode() != mode:
        raise ValueError('PRC_RANKING_MODE differs from prc.data')
    if mode != 'ranking':
        plan = final_plan(a, plan, D)
    pins, manifest, ix, y, ranking, ids = S.verify(a, plan)
    trained_on = plan['exp186']['board_annual_manifest_sha256'] if mode != 'ranking' else plan['annual_manifest_sha256']
    template = D.load_submitting()[D.ID].to_numpy('int64')
    if len(ids) != D.n_ranking() or not np.array_equal(ids, template):
        raise ValueError('Ranking IDs differ from the template order of the mode')
    device = torch.device(a.device)
    data = F.Arrays(a.inputs, 'annual', 'sequence')
    a.out.mkdir(parents=True)
    if mode != 'ranking':
        (a.out / 'FINAL-PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    records, values, saved_values, bad = [], [], [], []
    for path in a.fits:
        rec = json.loads((path / 'FIT.json').read_text())
        if rec['status'] != 'completed' or F.sha(path / 'model.pt') != rec['files']['model.pt']:
            raise ValueError('Fit incomplete or model differs: ' + str(path))
        if rec['manifest_sha256'] != trained_on:
            raise ValueError('the fit was trained on other annual inputs than the board plan: ' + str(path))
        model = S.load(path / 'model.pt', data, rec['recipe'], rec['seed'], device)
        pred = F.predict(model, data, ranking, device)
        name = f"{rec['recipe']}-{rec['seed']}.parquet"
        pd.DataFrame({'MVT_ID_mvt': ids, 'pred_Gmixed': pred}).to_parquet(a.out / name, index=False)
        old = pd.read_parquet(path / 'ranking.parquet').set_index('MVT_ID_mvt').pred_Gmixed
        shared = np.isin(ids, old.index.to_numpy())
        d = pred[shared] - old.reindex(ids[shared]).to_numpy('float64')
        diff = {'shared_rows': int(shared.sum()), 'max_abs_s': float(np.abs(d).max()) if len(d) else None,
                'rms_s': float(np.sqrt(np.mean(d ** 2))) if len(d) else None}
        if mode == 'ranking' and (not shared.all() or diff['max_abs_s'] > a.tol):
            bad.append(f'{path}: prediction differs from the saved ranking vector ({diff})')
        values.append(pred.astype('float64'))
        saved_values.append(old.reindex(ids).to_numpy('float64'))
        records.append({'fit': str(path.resolve()), 'fit_sha256': F.sha(path / 'FIT.json'), 'recipe': rec['recipe'],
                        'seed': rec['seed'], 'vector': name, 'against_saved': diff})
        print(json.dumps(records[-1]), flush=True)
    if len({r['fit_sha256'] for r in records}) != len(records):
        raise ValueError('a fit is given twice')
    mean = np.mean(values, axis=0)
    saved_mean = np.mean(saved_values, axis=0)
    both = np.isfinite(saved_mean)
    d = mean[both] - saved_mean[both]
    against = {'shared_rows': int(both.sum()), 'max_abs_s': float(np.abs(d).max()) if len(d) else None,
               'rms_s': float(np.sqrt(np.mean(d ** 2))) if len(d) else None}
    if mode == 'ranking' and (not both.all() or against['max_abs_s'] > a.tol):
        bad.append(f'mean differs from the mean of the saved vectors ({against})')
    frame = pd.DataFrame({'MVT_ID_mvt': ids, 'pred_Gmixed': mean})
    if not frame.MVT_ID_mvt.is_unique or not np.isfinite(mean).all() or len(frame) != D.n_ranking():
        raise ValueError('bad mean vector')
    frame.to_parquet(a.out / 'mean.parquet', index=False)
    part = None
    if a.part is not None and not bad:
        a.part.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(a.part, index=False)
        part = {'path': str(a.part), 'sha256': F.sha(a.part), 'rows': len(frame), 'columns': list(frame.columns)}
    months = pd.read_parquet(a.inputs / 'events.parquet', columns=['MVT_TIME_UTC_mvt']) \
        .MVT_TIME_UTC_mvt.iloc[ranking].dt.strftime('%Y-%m')
    out = {'ranking_mode': mode, 'rows': len(ids), 'manifest_sha256': F.sha(a.inputs / 'MANIFEST.json'),
           'plan_sha256': F.sha(a.plan), 'checks_sha256': F.sha(a.checks), 'source_sha256': pins,
           'code_sha256': F.sha(Path(__file__)), 'device': a.device, 'torch': torch.__version__,
           'fits': records, 'label_free': True, 'tol_s': a.tol, 'threads': a.threads,
           'mean_rule': 'equal mean of the fits, float64', 'mean_against_saved_mean': against, 'part': part,
           'replay_beyond_tol': bad,
           'rows_by_month': {k: int(v) for k, v in months.value_counts().sort_index().items()},
           'files': {p.name: F.sha(p) for p in sorted(a.out.iterdir())}}
    (a.out / 'PREDICT.json').write_text(json.dumps(out, indent=2) + '\n')
    if bad:
        raise ValueError('ranking mode: ' + '; '.join(bad))


if __name__ == '__main__':
    main()
