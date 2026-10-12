"""Unit 011: the six-seed e80 MLP level-2 member pack for valid, in grind's F5 pack format (pilot `write`).

pack_level2_009.py (unit 009) with the e80 arm's six-seed vectors (out/mlp-011/vectors/BAG-e80.json;
fit_sweep.py arm e80: unit 007's recipe with 80 epochs): stage Mlp164e80b6, column pred_Mlp164e80b6;
jjx-02/03/06/08/10/11/12 (seeds 701-706 of fit_sweep on each month's unit 008 inputs), janjul and
jjrank (seeds 701-706 of fit_sweep on the jjrank inputs, unit 007's training arrays; held part =
jjrefit rows then ranking rows, so no apply step). Each record binds its split, owner, training split,
six models, inputs, checks and code; splits and IDs as the unit 008 and 009 packs (F5, Lcontrol). No
label is read.
"""
from __future__ import annotations
import os
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd
from io_utils import W, ID, sha, write, artifact

sys.path.insert(0, str(W / 'code/seasonal'))
import common as C
_spec = importlib.util.spec_from_file_location('mlp008_pilot', C.SOURCE / 'experiments/exp-147-foldserve/code/pilot.py')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

ARM = 'e80'
STAGE = 'Mlp164e80b6'
COL = 'pred_' + STAGE
JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
SEEDS = [701, 702, 703, 704, 705, 706]
F5 = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/f5-r4-023/pack'))
F5_PACK_SHA = 'ae56121090f0d10d9dd8da64ff7212c10a6f625b727175c7bd0432b6ad5694fa'
LCONTROL = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/r2-L-pack-008-repair/Lcontrol-janjul.parquet'))
OUT = W / 'out/mlp-011/pack-level2'
BAG = W / 'out/mlp-011/vectors' / f'BAG-{ARM}.json'
READS = W / 'results/mlp-011'

def need(name, ok, **facts):
    if not ok:
        raise RuntimeError(f'{name}: {facts}')

def models_of(fit_name, subs):
    models, receipts, codes, checks = [], [], {}, []
    for sub in subs:
        fj = W / 'out/mlp-010' / ARM / sub / 'FIT.json'
        fit = json.loads(fj.read_text())
        need('fit receipt ' + sub, fit['fit'] == fit_name and fit['arm'] == ARM)
        cj = READS / f'CHECK-{ARM}-{sub}.json'
        c = json.loads(cj.read_text())
        need('check passes and binds the fit ' + sub, c['state'] == 'pass' and c['fit_receipt']['sha256'] == sha(fj))
        receipts.append(artifact(fj)); checks.append(artifact(cj)); codes[sub] = fit['code_sha256']
        for r in fit['seed_records']:
            need('seed receipt pin', sha(r['path']) == r['sha256'])
            rr = json.loads(Path(r['path']).read_text())
            need('model pin', sha(rr['model']['path']) == rr['model']['sha256'])
            models.append({'seed': rr['seed'], 'model_path': rr['model']['path'], 'model_sha256': rr['model']['sha256'],
                           'receipt': artifact(r['path']), 'train_rows': rr['train_rows'], 'epochs': rr['epochs'],
                           'inputs_sha256': rr['inputs_sha256']})
    models.sort(key=lambda m: m['seed'])
    need('six models, seeds 701-706 ' + fit_name, [m['seed'] for m in models] == SEEDS)
    return models, receipts, codes, checks

def bag_vector(bag, name, models):
    v = bag[name]
    need('bag vector pin ' + name, sha(v['path']) == v['sha256'] and v['seeds'] == SEEDS)
    need('bag seeds are the pack models ' + name, [m['seed'] for m in models] == v['seeds'])
    return Path(v['path'])

def main():
    need('F5 pack pin', sha(F5 / 'PACK.json') == F5_PACK_SHA)
    bagrec = json.loads(BAG.read_text())
    need('bag record arm', bagrec['arm'] == ARM)
    bag = {v['name']: v for v in bagrec['vectors']}
    f5 = {a['fit']: a for a in json.loads((F5 / 'PACK.json').read_text())['artifacts']}
    OUT.mkdir(parents=True, exist_ok=True)
    need('empty pack folder', not any(OUT.iterdir()))
    spec = READS / 'SPEC-011.md'
    arts = []
    def emit(fit, split, owner, vec_path, ids_expected, record, native_rec):
        v = pd.read_parquet(vec_path)
        need('vector columns', list(v.columns) == [ID, 'month', 'utc_day', 'airport', 'prediction'], cols=list(v.columns))
        v[ID] = v[ID].astype('int64')
        need('IDs unique ' + fit, v[ID].is_unique)
        need('IDs equal the native contract in order ' + fit, len(v) == len(ids_expected) and (v[ID].to_numpy() == ids_expected).all(),
             ours=len(v), native=len(ids_expected))
        need('split equals the native record ' + fit, split.manifest() == native_rec['split'])
        need('train split equals the native record ' + fit, record['train_split'] == native_rec.get('train_split', native_rec['split']))
        need('finite floored ' + fit, v.prediction.notna().all() and v.prediction.min() >= 60)
        v = v.rename(columns={'prediction': COL})
        path = P.write(OUT, STAGE, split, owner, v, {**record, 'scored_rows': int(len(v)), 'fit_name': fit,
                       'stage_what': 'exp-164 embedding MLP, unit 007 full arm (210 numeric, 18 categorical) with 80 epochs (arm e80), mean of seeds 701-706',
                       'prediction_column': COL, 'source_vector': artifact(vec_path)})
        js = path.with_suffix('.json')
        arts.append({'fit': fit, 'stage': STAGE, 'column': COL, 'owner': owner, 'path': str(path), 'sha256': sha(path),
                     'record_sha256': sha(js), 'rows': int(len(v)), 'train_split': record['train_split'],
                     'models': record['models']})
    arm_cfg = None
    for fit in JJX:
        models, receipts, codes, checks = models_of(fit, [fit, fit + '-seeds-703-704-705-706'])
        inputs_path = W / 'inputs/mlp-008' / fit / 'INPUTS.json'
        inputs = json.loads(inputs_path.read_text())
        need('models use the month inputs ' + fit, all(m['inputs_sha256'] == sha(inputs_path) for m in models))
        arm_cfg = json.loads((W / 'out/mlp-010' / ARM / fit / 'FIT.json').read_text())['config']
        vec = bag_vector(bag, fit, models)
        native = F5 / Path(f5[fit]['path']).name
        need('F5 vector pin ' + fit, sha(native) == f5[fit]['sha256'] and sha(native.with_suffix('.json')) == f5[fit]['record_sha256'])
        ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
        split = P.split(fit)
        need('fit split ' + fit, inputs['split'] == split.manifest())
        emit(fit, split, fit, vec, ids,
             {'train_split': inputs['split'], 'train_rows': inputs['train_rows'], 'models': models, 'arm': ARM, 'arm_config': arm_cfg,
              'fit_receipts': receipts, 'checks': checks, 'bag': artifact(BAG), 'inputs': artifact(inputs_path),
              'code_sha256': {'fit_sweep': codes}, 'spec': artifact(spec)},
             json.loads(native.with_suffix('.json').read_text()))
    # janjul and jjrank: six fit_sweep models on the jjrank inputs (unit 007's training arrays and split)
    models, receipts, codes, checks = models_of('jjrank', ['jjrank', 'jjrank-seeds-703-704', 'jjrank-seeds-705-706'])
    jin = W / 'inputs/mlp-008/jjrank/INPUTS.json'
    inputs = json.loads(jin.read_text())
    need('models use the jjrank inputs', all(m['inputs_sha256'] == sha(jin) for m in models))
    inputs007 = json.loads((W / 'inputs/mlp-007/INPUTS.json').read_text())
    need('jjrank inputs carry unit 007 split and rows', inputs['split'] == inputs007['split'] and inputs['train_rows'] == inputs007['train_rows'])
    need('jjrank training days are janjul training days', P.janjul().excluded == tuple(inputs['split']['excluded']))
    lrec = json.loads(LCONTROL.with_suffix('.json').read_text())
    need('Lcontrol janjul pin', sha(LCONTROL) == lrec['output_sha256'])
    ids = pd.read_parquet(LCONTROL, columns=[ID])[ID].to_numpy('int64')
    common = {'train_split': P.janjul().manifest(), 'fit_split': inputs['split'], 'train_rows': inputs['train_rows'], 'models': models,
              'arm': ARM, 'arm_config': arm_cfg, 'fit_receipts': receipts, 'checks': checks, 'bag': artifact(BAG),
              'inputs': artifact(jin), 'code_sha256': {'fit_sweep': codes}, 'spec': artifact(spec),
              'held_part': 'jjrefit rows then ranking rows of one jjrank fit per seed (no apply step)'}
    emit('janjul', P.janjul(), 'janjul', bag_vector(bag, 'janjul', models), ids, common, lrec)
    native = F5 / Path(f5['jjrank']['path']).name
    need('F5 vector pin jjrank', sha(native) == f5['jjrank']['sha256'] and sha(native.with_suffix('.json')) == f5['jjrank']['record_sha256'])
    nrec = json.loads(native.with_suffix('.json').read_text())
    ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
    emit('jjrank', P.WP.Split.from_manifest(nrec['split']), 'jjrank', bag_vector(bag, 'jjrank', models), ids, common, nrec)
    reads = {'E80X6': artifact(READS / 'READ-E80X6.json')}
    pack = {'status': 'complete', 'stage': STAGE, 'prediction_column': COL, 'fits': [a['fit'] for a in arts],
            'source': 'exp-164 unit 007 full MLP with 80 epochs (arm e80), six seeds 701-706 (unit 011; reads REPORT-011.md)',
            'reads_on_B': reads, 'served_G': 'unmeasured; valid owns admission', 'source_ledger_row': 901,
            'F5_pack_sha256': F5_PACK_SHA, 'unit009_pack': 'out/mlp-009/pack-level2 (Mlp164b6, base recipe, six seeds)',
            'code_sha256': sha(__file__), 'artifacts': arts,
            'note': 'janjul and jjrank come from six models trained on unit 007 arrays (ten-month jjrank training split); no all-year refit in this pack'}
    write(OUT / 'PACK.json', pack)
    print(json.dumps({'event': 'pack-ready', 'pack_sha256': sha(OUT / 'PACK.json'), 'rows': {a['fit']: a['rows'] for a in arts}}), flush=True)

if __name__ == '__main__':
    main()
