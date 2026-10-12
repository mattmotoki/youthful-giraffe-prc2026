"""Unit 008: the MLP level-2 member pack for valid, in grind's F5 pack format (pilot `write`).

Nine vectors, stage Mlp164, column pred_Mlp164: jjx-02/03/06/08/10/11/12 (unit 008 fits), janjul (unit 007
held mean) and jjrank (unit 007 models on the 2026 ranking rows). Each record binds its split, owner,
training split, models, inputs and code. The IDs must equal grind's F5 native IDs (jjx, jjrank) and
Lcontrol janjul's IDs. No label is read.
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

STAGE = 'Mlp164'
COL = 'pred_' + STAGE
JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
F5 = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/f5-r4-023/pack'))
F5_PACK_SHA = 'ae56121090f0d10d9dd8da64ff7212c10a6f625b727175c7bd0432b6ad5694fa'
LCONTROL = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/r2-L-pack-008-repair/Lcontrol-janjul.parquet'))
REF_FIT_SHA = '34c5b7334ca7b1cd9b2a7df54d85f2f588d0e5e0fd257cd43f77e5c76fa7a9e6'
OUT = W / 'out/mlp-008/pack-level2'

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

def main():
    need('F5 pack pin', sha(F5 / 'PACK.json') == F5_PACK_SHA)
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
        need('mean pin ' + fit, sha(fit_rec['prediction']['path']) == fit_rec['prediction']['sha256'])
        inputs = json.loads((W / 'inputs/mlp-008' / fit / 'INPUTS.json').read_text())
        native = F5 / Path(f5[fit]['path']).name
        need('F5 vector pin ' + fit, sha(native) == f5[fit]['sha256'] and sha(native.with_suffix('.json')) == f5[fit]['record_sha256'])
        ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
        split = P.split(fit)
        need('fit split ' + fit, inputs['split'] == split.manifest())
        need('fit receipt binds its inputs ' + fit, fit_rec['inputs_sha256'] == sha(W / 'inputs/mlp-008' / fit / 'INPUTS.json'))
        emit(fit, split, fit, Path(fit_rec['prediction']['path']), ids,
             {'train_split': inputs['split'], 'train_rows': inputs['train_rows'], 'models': models,
              'fit_receipt': artifact(out / 'FIT.json'), 'inputs': artifact(W / 'inputs/mlp-008' / fit / 'INPUTS.json'),
              'code_sha256': fit_rec['code_sha256'], 'spec': artifact(W / 'results/mlp-008/SPEC-008.md')},
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
    ids = pd.read_parquet(LCONTROL, columns=[ID])[ID].to_numpy('int64')
    common = {'train_split': P.janjul().manifest(), 'unit007_fit_split': inputs007['split'],
              'train_rows': inputs007['train_rows'], 'models': models,
              'fit_receipt': artifact(ref / 'FIT.json'), 'inputs': artifact(W / 'inputs/mlp-007/INPUTS.json'),
              'code_sha256': fit_rec['code_sha256'], 'spec': artifact(W / 'results/mlp-007/SPEC-007.md')}
    emit('janjul', P.janjul(), 'janjul', Path(fit_rec['prediction']['path']), ids, common, lrec)
    # jjrank: the same models on the 2026 ranking rows, built with the same preprocessing
    ap = json.loads((W / 'out/mlp-008/jjrank/APPLY.json').read_text())
    need('jjrank apply pins', sha(ap['prediction_ranking']['path']) == ap['prediction_ranking']['sha256'])
    need('jjrank replay', ap['mean_replay_max_abs_s'] <= 1e-3)
    native = F5 / Path(f5['jjrank']['path']).name
    need('F5 vector pin jjrank', sha(native) == f5['jjrank']['sha256'] and sha(native.with_suffix('.json')) == f5['jjrank']['record_sha256'])
    nrec = json.loads(native.with_suffix('.json').read_text())
    need('jjrank apply binds its inputs', ap['inputs_sha256'] == sha(W / 'inputs/mlp-008/jjrank/INPUTS.json'))
    ids = pd.read_parquet(native, columns=[ID])[ID].to_numpy('int64')
    emit('jjrank', P.WP.Split.from_manifest(nrec['split']), 'jjrank', Path(ap['prediction_ranking']['path']), ids,
         {**common, 'apply_receipt': artifact(W / 'out/mlp-008/jjrank/APPLY.json'),
          'jjrank_inputs': artifact(W / 'inputs/mlp-008/jjrank/INPUTS.json'), 'held_replay_max_abs_s': ap['mean_replay_max_abs_s']}, nrec)
    pack = {'status': 'complete', 'stage': STAGE, 'prediction_column': COL, 'fits': [a['fit'] for a in arts],
            'source': 'exp-164 unit 007 full MLP (fold read REPORT-007.md; joint read with the runway pair REPORT-008.md)',
            'member_G_on_B': 581.5874719786808, 'member_interval95': [374.16418, 803.40700],
            'G_beside_runway_pair_on_B': {'pair': 1287.642, 'MLP+pair': 1487.184, 'increment': 199.541, 'increment95': [105.30, 292.84]},
            'served_G': 'unmeasured; valid owns admission', 'source_ledger_row': 901,
            'F5_pack_sha256': F5_PACK_SHA, 'code_sha256': sha(__file__), 'artifacts': arts,
            'note': 'janjul and jjrank come from the unit 007 models (ten-month jjrank training split); no all-year refit in this pack'}
    write(OUT / 'PACK.json', pack)
    print(json.dumps({'event': 'pack-ready', 'pack_sha256': sha(OUT / 'PACK.json'), 'rows': {a['fit']: a['rows'] for a in arts}}), flush=True)

if __name__ == '__main__':
    main()
