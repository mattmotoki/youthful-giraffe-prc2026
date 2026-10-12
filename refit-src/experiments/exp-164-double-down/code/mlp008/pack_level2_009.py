"""Unit 009: the six-seed MLP level-2 member pack for valid, in grind's F5 pack format (pilot `write`).

pack_level2.py (unit 008, sha 1ec7fd71) with the six-seed vectors of out/mlp-009/vectors (BAG.json):
stage Mlp164b6, column pred_Mlp164b6; jjx-02/03/06/08/10/11/12 (seeds 701/702 of unit 008 and 703-706
of unit 009), janjul and jjrank (unit 007's two models and unit 009's four jjrank models, all trained on
unit 007's arrays). Each record binds its split, owner, training split, six models, inputs and code;
splits and IDs as unit 008's pack (F5, Lcontrol). No label is read.
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

STAGE = 'Mlp164b6'
COL = 'pred_' + STAGE
JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
F5 = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/f5-r4-023/pack'))
F5_PACK_SHA = 'ae56121090f0d10d9dd8da64ff7212c10a6f625b727175c7bd0432b6ad5694fa'
LCONTROL = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/r2-L-pack-008-repair/Lcontrol-janjul.parquet'))
REF_FIT_SHA = '34c5b7334ca7b1cd9b2a7df54d85f2f588d0e5e0fd257cd43f77e5c76fa7a9e6'
OUT = W / 'out/mlp-009/pack-level2'
BAG = W / 'out/mlp-009/vectors/BAG.json'
READS = W / 'results/mlp-009'

def need(name, ok, **facts):
    if not ok:
        raise RuntimeError(f'{name}: {facts}')

def models_of(fit_json):
    fit = json.loads(Path(fit_json).read_text())
    out = []
    for r in fit['seed_records']:
        need('seed receipt pin', sha(r['path']) == r['sha256'])
        rr = json.loads(Path(r['path']).read_text())
        need('model pin', sha(rr['model']['path']) == rr['model']['sha256'])
        out.append({'seed': rr['seed'], 'model_path': rr['model']['path'], 'model_sha256': rr['model']['sha256'],
                    'receipt': artifact(r['path']), 'train_rows': rr['train_rows'], 'epochs': rr['epochs']})
    return fit, out

def bag_vector(bag, name, seeds_models):
    v = bag[name]
    need('bag vector pin ' + name, sha(v['path']) == v['sha256'] and v['seeds'] == [701, 702, 703, 704, 705, 706])
    need('bag seeds are the pack models ' + name, [m['seed'] for m in seeds_models] == v['seeds'])
    return Path(v['path'])

def main():
    need('F5 pack pin', sha(F5 / 'PACK.json') == F5_PACK_SHA)
    bag = {v['name']: v for v in json.loads(BAG.read_text())['vectors']}
    f5 = {a['fit']: a for a in json.loads((F5 / 'PACK.json').read_text())['artifacts']}
    OUT.mkdir(parents=True, exist_ok=True)
    need('empty pack folder', not any(OUT.iterdir()))
    arts = []
    def emit(fit, split, owner, vec_path, ids_expected, record, native_rec):
        v = pd.read_parquet(vec_path)
        need('vector columns', list(v.columns) == [ID, 'month', 'utc_day', 'airport', 'prediction'], cols=list(v.columns))
        v[ID] = v[ID].astype('int64')
        need('IDs unique ' + fit, v[ID].is_unique)
        need('IDs equal the native contract in order ' + fit, len(v) == len(ids_expected) and (v[ID].to_numpy() == ids_expected).all(),
             ours=len(v), native=len(ids_expected))
        # the same split manifests as the native record (exp-170 pack_native.py compares both)
        need('split equals the native record ' + fit, split.manifest() == native_rec['split'])
        need('train split equals the native record ' + fit, record['train_split'] == native_rec.get('train_split', native_rec['split']))
        need('finite floored ' + fit, v.prediction.notna().all() and v.prediction.min() >= 60)
        v = v.rename(columns={'prediction': COL})
        path = P.write(OUT, STAGE, split, owner, v, {**record, 'scored_rows': int(len(v)), 'fit_name': fit,
                       'stage_what': 'exp-164 embedding MLP, unit 007 full arm (210 numeric, 18 categorical), mean of seeds 701 and 702',
                       'prediction_column': COL, 'source_vector': artifact(vec_path)})
        js = path.with_suffix('.json')
        arts.append({'fit': fit, 'stage': STAGE, 'column': COL, 'owner': owner, 'path': str(path), 'sha256': sha(path),
                     'record_sha256': sha(js), 'rows': int(len(v)), 'train_split': record['train_split'],
                     'models': record['models']})
    for fit in JJX:
        out = W / 'out/mlp-008' / fit
        fit_rec, models = models_of(out / 'FIT.json')
        need('fit receipt ' + fit, fit_rec['fit'] == fit and fit_rec['seeds'] == [701, 702])
        out9 = W / 'out/mlp-009' / fit
        fit9, models9 = models_of(out9 / 'FIT.json')
        need('bag fit receipt ' + fit, fit9['fit'] == fit and fit9['seeds'] == [703, 704, 705, 706] and fit9['inputs_sha256'] == fit_rec['inputs_sha256'])
        models = models + models9
        vec = bag_vector(bag, fit, models)
        inputs = json.loads((W / 'inputs/mlp-008' / fit / 'INPUTS.json').read_text())
        native = F5 / Path(f5[fit]['path']).name
        need('F5 vector pin ' + fit, sha(native) == f5[fit]['sha256'] and sha(native.with_suffix('.json')) == f5[fit]['record_sha256'])
        ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
        split = P.split(fit)
        need('fit split ' + fit, inputs['split'] == split.manifest())
        need('fit receipt binds its inputs ' + fit, fit_rec['inputs_sha256'] == sha(W / 'inputs/mlp-008' / fit / 'INPUTS.json'))
        emit(fit, split, fit, vec, ids,
             {'train_split': inputs['split'], 'train_rows': inputs['train_rows'], 'models': models,
              'fit_receipts': [artifact(out / 'FIT.json'), artifact(out9 / 'FIT.json')], 'bag': artifact(BAG),
              'inputs': artifact(W / 'inputs/mlp-008' / fit / 'INPUTS.json'),
              'code_sha256': {'fit': fit_rec['code_sha256'], 'fit_bag': fit9['code_sha256']}, 'spec': artifact(W / 'results/mlp-009/SPEC-009.md')},
             json.loads(native.with_suffix('.json').read_text()))
    # janjul: unit 007's held mean (ten-month jjrank training split)
    ref = W / 'out/mlp-007/full'
    need('unit 007 fit pin', sha(ref / 'FIT.json') == REF_FIT_SHA)
    fit_rec, models = models_of(ref / 'FIT.json')
    inputs007 = json.loads((W / 'inputs/mlp-007/INPUTS.json').read_text())
    lrec = json.loads(LCONTROL.with_suffix('.json').read_text())
    need('Lcontrol janjul pin', sha(LCONTROL) == lrec['output_sha256'])
    # unit 007 trained on the pilot's jjrank split (no held day); its excluded days are janjul's, so
    # its training rows are janjul's training rows; the records carry janjul as F5 and Lcontrol do
    need('unit 007 training days are janjul training days', P.janjul().excluded == tuple(inputs007['split']['excluded']))
    need('unit 007 fit binds its inputs', fit_rec['inputs_sha256'] == sha(W / 'inputs/mlp-007/INPUTS.json'))
    fit9, models9 = models_of(W / 'out/mlp-009/jjrank/FIT.json')
    need('bag jjrank fit uses the jjrank inputs', fit9['fit'] == 'jjrank' and fit9['inputs_sha256'] == sha(W / 'inputs/mlp-008/jjrank/INPUTS.json'))
    models = models + models9
    ids = pd.read_parquet(LCONTROL, columns=[ID])[ID].to_numpy('int64')
    common = {'train_split': P.janjul().manifest(), 'unit007_fit_split': inputs007['split'],
              'train_rows': inputs007['train_rows'], 'models': models,
              'fit_receipts': [artifact(ref / 'FIT.json'), artifact(W / 'out/mlp-009/jjrank/FIT.json')], 'bag': artifact(BAG),
              'inputs': artifact(W / 'inputs/mlp-007/INPUTS.json'), 'jjrank_inputs': artifact(W / 'inputs/mlp-008/jjrank/INPUTS.json'),
              'code_sha256': {'fit': fit_rec['code_sha256'], 'fit_bag': fit9['code_sha256']}, 'spec': artifact(W / 'results/mlp-009/SPEC-009.md')}
    emit('janjul', P.janjul(), 'janjul', bag_vector(bag, 'janjul', models), ids, common, lrec)
    # jjrank: the same models on the 2026 ranking rows, built with the same preprocessing
    ap = json.loads((W / 'out/mlp-008/jjrank/APPLY.json').read_text())
    need('jjrank apply pins', sha(ap['prediction_ranking']['path']) == ap['prediction_ranking']['sha256'])
    need('jjrank replay', ap['mean_replay_max_abs_s'] <= 1e-3)
    native = F5 / Path(f5['jjrank']['path']).name
    need('F5 vector pin jjrank', sha(native) == f5['jjrank']['sha256'] and sha(native.with_suffix('.json')) == f5['jjrank']['record_sha256'])
    nrec = json.loads(native.with_suffix('.json').read_text())
    need('jjrank apply binds its inputs', ap['inputs_sha256'] == sha(W / 'inputs/mlp-008/jjrank/INPUTS.json'))
    ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
    emit('jjrank', P.WP.Split.from_manifest(nrec['split']), 'jjrank', bag_vector(bag, 'jjrank', models), ids,
         {**common, 'apply_receipt': artifact(W / 'out/mlp-008/jjrank/APPLY.json'), 'held_replay_max_abs_s': ap['mean_replay_max_abs_s']}, nrec)
    reads = {k: artifact(READS / f'READ-{k}.json') for k in ('JOINT', 'STACK4')}
    pack = {'status': 'complete', 'stage': STAGE, 'prediction_column': COL, 'fits': [a['fit'] for a in arts],
            'source': 'exp-164 unit 007 full MLP, six seeds 701-706 (unit 009; reads REPORT-009.md)',
            'reads_on_B': reads, 'served_G': 'unmeasured; valid owns admission', 'source_ledger_row': 901,
            'F5_pack_sha256': F5_PACK_SHA, 'unit008_pack': 'out/mlp-008/pack-level2 (Mlp164, two seeds)',
            'code_sha256': sha(__file__), 'artifacts': arts,
            'note': 'janjul and jjrank come from six models trained on unit 007 arrays (ten-month jjrank training split); no all-year refit in this pack'}
    write(OUT / 'PACK.json', pack)
    print(json.dumps({'event': 'pack-ready', 'pack_sha256': sha(OUT / 'PACK.json'), 'rows': {a['fit']: a['rows'] for a in arts}}), flush=True)

if __name__ == '__main__':
    main()
