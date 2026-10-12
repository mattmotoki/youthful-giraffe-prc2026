"""Replay a fixed L/Lp janjul fit, then export ranking with that same booster."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
import learner_wp as LW


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def align(frame, ids, column):
    if not frame[LW.ID].is_unique or set(frame[LW.ID]) != set(ids):
        raise RuntimeError('prediction IDs differ')
    values = frame.set_index(LW.ID).loc[ids, column].to_numpy('float64')
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite prediction')
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--references', type=Path, required=True)
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--extra-inputs', type=Path)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads((a.references/'TASKS.json').read_text())
    task = manifest['tasks'][a.task]
    learner, arm = task['learner'], task['arm']
    threads = task['threads']
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PRC_THREADS'):
        if int(os.environ.get(key, '0')) != threads:
            raise RuntimeError('thread cap differs: '+key)
    dest = a.out/a.task
    if dest.exists():
        raise RuntimeError('output exists; inspect before any retry')
    dest.mkdir(parents=True)
    started = time.monotonic()
    reference = a.references/task['reference']
    if LW.sha(reference) != task['sha256'] or LW.sha(reference.with_suffix('.json')) != task['record_sha256']:
        raise RuntimeError('reference hash differs')
    ref = json.loads(reference.with_suffix('.json').read_text())
    split = LW.P.janjul()
    if ref['train_split'] != split.manifest() or ref['split'] != split.manifest():
        raise RuntimeError('reference janjul split differs')
    params = dict(ref['fit']['params'])
    if params['num_threads'] != threads or ref['fit']['rounds'] != 1500:
        raise RuntimeError('fixed settings differ')
    # Match the exact data helper used by each original host/cloud fit.
    wanted_data = ref['code']['data.py']
    candidates = [LW.ROOT/'prc/data.py', *a.input_root.rglob('data.py')]
    data_path = next((p for p in candidates if p.exists() and LW.sha(p)==wanted_data),None)
    if data_path is None:
        raise RuntimeError('original data helper absent')
    spec = importlib.util.spec_from_file_location('prc.data',data_path)
    data_module = importlib.util.module_from_spec(spec)
    sys.modules['prc.data'] = data_module
    spec.loader.exec_module(data_module)
    import prc
    prc.data = data_module
    env = LW.setup(a.input_root)
    if env[7] != ref['code']:
        raise RuntimeError('upstream model code differs')
    raw, audit = LW.raw_months_cached(env[0], a.cache)
    base, data = env[2], env[6]
    frames = {m: base.month_frame(m) for m in LW.MONTHS}
    inputs = {f'month_{m}.parquet': LW.sha(base.locate_file(f'month_{m}.parquet')) for m in LW.MONTHS}
    if inputs != ref['inputs']:
        raise RuntimeError('training inputs differ')
    train = pd.concat([frames[m] for m in LW.MONTHS], ignore_index=True)
    train = train.loc[data.keep_reachable(train)]
    train = train.loc[split.label_mask(LW.wp_frame(train))].reset_index(drop=True)
    split.assert_training_frame(LW.wp_frame(train), target=train.y.to_numpy(), what=a.task)
    if len(train) != ref['train_rows']:
        raise RuntimeError('encoding input rows differ')
    held = [frames[m].loc[split.score_mask(LW.wp_frame(frames[m]))].drop(columns=['y']) for m in LW.SCORED_MONTHS]
    weather = None
    weather_hashes = {}
    if arm.startswith('weather'):
        if a.extra_inputs is None:
            raise RuntimeError('weather inputs absent')
        path = a.extra_inputs/'weather-core.parquet'
        wr = json.loads(path.with_suffix('.json').read_text())
        if not wr['label_free'] or LW.sha(path) != wr['output_sha256']:
            raise RuntimeError('weather receipt differs')
        weather_hashes = {path.name: LW.sha(path)}
        expected_weather = ref.get('extra_inputs') or ref['fit'].get('extra_inputs')
        if expected_weather is None:
            expected_weather = {'weather-core.parquet':ref['fit']['weather_sha256']}
        if weather_hashes != expected_weather:
            raise RuntimeError('weather input hash differs')
        weather = pd.read_parquet(path).set_index(LW.ID)
        if not weather.index.is_unique or list(weather.columns) != wr['features']:
            raise RuntimeError('weather IDs or schema differ')
    old_frames = base.feature_frames
    captured = {}
    def feature_frames(tr, sc, *args, **kwargs):
        result = old_frames(tr, sc, *args, **kwargs)
        if weather is not None:
            for role, frame in [('train',tr),('score',sc)]:
                ids = frame[LW.ID].to_numpy('int64')
                if not pd.Index(ids).isin(weather.index).all():
                    raise RuntimeError('weather coverage differs')
                captured[role] = weather.reindex(ids).reset_index(drop=True)
        return result
    base.feature_frames = feature_frames
    fitted = {}
    def fit(name, xt, xs, residual, weights, cats, anchor, model, fold):
        import lightgbm as lgb
        if weather is not None:
            xt = pd.concat([xt.reset_index(drop=True), captured['train']], axis=1)
            xs = pd.concat([xs.reset_index(drop=True), captured['score']], axis=1)
        if list(xt.columns) != list(xs.columns) or xt.columns.duplicated().any():
            raise RuntimeError('feature schema differs')
        if not fitted:
            if len(xt) != ref['fit']['train_rows'] or len(xt.columns) != ref['fit']['features']:
                raise RuntimeError('ordinary training population or features differ')
            features = list(xt.columns)
            feature_sha = hashlib.sha256(json.dumps(features).encode()).hexdigest()
            if feature_sha != ref['fit']['feature_list_sha256']:
                raise RuntimeError('fitted feature order differs')
            fitted.update(categories={c: xt[c].dtype for c in xt if isinstance(xt[c].dtype,pd.CategoricalDtype)},
                          train_hash=frame_hash(xt), label_hash=hashlib.sha256(np.asarray(residual).tobytes()).hexdigest(),
                          feature_order=features, categorical=cats, feature_sha=feature_sha)
            (dest/'before-fit.json').write_text(json.dumps({'params':params,'rounds':1500,'train_split':split.manifest(),
                'encoding_rows':len(train),'train_rows':len(xt),'features':features,'train_hash':fitted['train_hash'],
                'reference':task,'driver_sha256':LW.sha(Path(__file__))},indent=2,default=str)+'\n')
            print('fit-start',a.task,len(xt),len(xt.columns),flush=True)
            t = time.monotonic()
            booster = lgb.train(params,lgb.Dataset(xt,label=residual,weight=weights,
                                categorical_feature=cats,free_raw_data=True),num_boost_round=1500)
            fitted.update(booster=booster,fit_seconds=time.monotonic()-t,version=lgb.__version__)
            booster.save_model(str(dest/'model.txt'))
        else:
            for col, dtype in fitted['categories'].items():
                xt[col] = xt[col].astype(dtype)
                xs[col] = xs[col].astype(dtype)
            if list(xt.columns) != fitted['feature_order'] or cats != fitted['categorical']:
                raise RuntimeError('ranking feature order differs')
            if frame_hash(xt) != fitted['train_hash'] or hashlib.sha256(np.asarray(residual).tobytes()).hexdigest() != fitted['label_hash']:
                raise RuntimeError('ranking pass changed fitted training features or labels')
        pred = np.maximum(30.0,anchor+fitted['booster'].predict(xs,num_threads=threads))
        return pred, {'info':{'features':len(xt.columns),'fit_seconds':fitted['fit_seconds']}}
    base.fit_arm = fit
    held_pred, held_record = LW.fit_learner(env,raw,train.copy(),held,list(LW.SCORED_MONTHS),'jjrank-held',learner)
    old = pd.read_parquet(reference)
    column = f'pred_{learner}153-{arm}'
    ids = old[LW.ID].to_numpy()
    held_values = align(held_pred,ids,'pred_'+learner)
    rms = float(np.sqrt(np.mean((held_values-old[column].to_numpy('float64'))**2)))
    held_out = pd.DataFrame({LW.ID:ids,'y_pred':held_values})
    held_out.to_parquet(dest/'held.parquet',index=False)
    print('held-rms',a.task,rms,flush=True)
    if not np.isfinite(rms) or rms > manifest['max_held_rms_seconds']:
        (dest/'reproduction-failed.json').write_text(json.dumps({'held_rms':rms,
            'limit':manifest['max_held_rms_seconds'],'task':a.task},indent=2)+'\n')
        raise RuntimeError('held reproduction RMS exceeds the frozen limit')
    ranking = [base.month_frame(m).drop(columns=['y'],errors='ignore') for m in LW.RANK_MONTHS]
    rank_pred, _ = LW.fit_learner(env,raw,train.copy(),ranking,list(LW.RANK_MONTHS),'jjrank-ranking',learner)
    ids_path = a.references/'L-ranking-ids.parquet'
    if LW.sha(ids_path) != manifest['ranking_ids_sha256']:
        raise RuntimeError('ranking reference IDs changed')
    rank_ids = pd.read_parquet(ids_path)[LW.ID].to_numpy()
    values = align(rank_pred,rank_ids,'pred_'+learner)
    pd.DataFrame({LW.ID:rank_ids,'y_pred':values}).to_parquet(dest/'ranking.parquet',index=False)
    record = {'task':a.task,'learner':learner,'arm':arm,'domain':'ranking2026','training_split':split.manifest(),
        'train_rows':ref['train_rows'],'ordinary_train_rows':ref['fit']['train_rows'],'ranking_rows':len(values),
        'held_rows':len(ids),'held_rms':rms,'held_max_abs':float(np.max(np.abs(held_values-old[column].to_numpy('float64')))),
        'params':params,'rounds':1500,'threads':threads,'inputs':inputs,'extra_inputs':weather_hashes,
        'ranking_inputs':{m:LW.sha(base.locate_file('month_'+m+'.parquet')) for m in LW.RANK_MONTHS},
        'reference':task,'feature_list_sha256':fitted['feature_sha'],'train_feature_sha256':fitted['train_hash'],
        'model_sha256':LW.sha(dest/'model.txt'),'held_sha256':LW.sha(dest/'held.parquet'),
        'ranking_sha256':LW.sha(dest/'ranking.parquet'),'driver_sha256':LW.sha(Path(__file__)),
        'learner_code_sha256':LW.sha(Path(LW.__file__)),'upstream_code':env[7],
        'lightgbm':fitted['version'],'reference_lightgbm':ref.get('lightgbm'),'fit_seconds':fitted['fit_seconds'],
        'total_seconds':time.monotonic()-started,'same_booster_for_both_domains':True}
    (dest/'receipt.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    print('complete',a.task,flush=True)


if __name__ == '__main__':
    main()
