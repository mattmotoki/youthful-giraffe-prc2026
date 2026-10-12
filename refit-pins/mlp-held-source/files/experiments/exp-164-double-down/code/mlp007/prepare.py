"""Build the embedding MLP's inputs without a fit or a held score."""
from __future__ import annotations
import os
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from io_utils import W, ID, sha, write, artifact, lane

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
H = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/source/experiments/exp-153-grind/code'))
INPUT = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp148-work/kinput'))
RAW = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-data-challenge-2026/data/raw'))
OUT = W / 'inputs/mlp-007'
RESULT = W / 'results/mlp-007'
sys.path.insert(0, str(H))
import learner_wp as LW
sys.path.insert(0, str(W / 'code/seasonal'))
import common as C
_spec = importlib.util.spec_from_file_location('mlp007_pilot', C.SOURCE / 'experiments/exp-147-foldserve/code/pilot.py')
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

EMB = ['emb_airport', 'emb_stand', 'emb_runway', 'emb_operator', 'emb_type', 'emb_hour', 'emb_month']
NATIVE_SOURCES = {'c_apt':['ADEP_mvt'], 'c_rwy':['ADEP_mvt','RUNWAY_mvt'],
    'c_stand':['ADEP_mvt','STAND_mvt'], 'c_standpfx':['ADEP_mvt','STAND_mvt'],
    'c_actype':['AIRCRAFT_TYPE_mvt'], 'c_wtc':['WK_TBL_CAT_flt'],
    'c_seg':['MARKET_SEGMENT_flt'], 'c_ftype':['FLIGHT_TYPE_flt'],
    'c_frule':['FLIGHT_RULE_mvt'], 'c_oper':['AIRCRAFT_OPERATOR_flt'],
    'c_ades':['ADES_mvt'], 'c_adescty':['ADES_mvt'], 'c_csgn':['FLIGHT_mvt']}
CHECKS = []

def check(name, ok, **facts):
    if not bool(ok):
        raise RuntimeError(f'{name}: {facts}')
    CHECKS.append({'name': name, 'state': 'pass', **facts})

def save_array(name, x):
    p = OUT / (name + '.npy')
    if p.exists():
        raise RuntimeError('array exists')
    np.save(p, x, allow_pickle=False)
    return artifact(p, shape=list(x.shape), dtype=str(x.dtype))

def raw_embeddings(frames):
    cols = [ID, 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt',
            'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt',
            'WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','FLIGHT_RULE_mvt','ADES_mvt']
    pieces = []
    pins = {}
    for m in LW.MONTHS:
        p = RAW / f'training_{m}-01_{(pd.Period(m)+1).strftime("%Y-%m")}-01.parquet'
        pins[str(p)] = sha(p)
        a = pd.read_parquet(p, columns=cols)
        a = a[a[ID].isin(frames[m][ID])].copy()
        a['emb_airport'] = a.ADEP_mvt
        a['emb_stand'] = a.ADEP_mvt.astype('string') + '/' + a.STAND_mvt.astype('string')
        a['emb_runway'] = a.ADEP_mvt.astype('string') + '/' + a.RUNWAY_mvt.astype('string')
        a['emb_operator'] = a.AIRCRAFT_OPERATOR_flt.replace('', pd.NA)
        a['emb_type'] = a.AIRCRAFT_TYPE_mvt
        a['emb_hour'] = a.MVT_TIME_UTC_mvt.dt.hour.astype('string')
        a['emb_month'] = a.MVT_TIME_UTC_mvt.dt.month.astype('string')
        for c, sources in NATIVE_SOURCES.items():
            a['missing_'+c] = a[sources].isna().any(axis=1) | a[sources].astype('string').eq('').any(axis=1)
        for c, native in {'emb_airport':'c_apt','emb_stand':'c_stand','emb_runway':'c_rwy',
                          'emb_operator':'c_oper','emb_type':'c_actype'}.items():
            a[c] = a[c].mask(a['missing_'+native])
        pieces.append(a[[ID, *EMB, *['missing_'+c for c in NATIVE_SOURCES]]])
    a = pd.concat(pieces, ignore_index=True)
    check('raw embedding IDs unique', a[ID].is_unique)
    return a.set_index(ID), pins

def a3_inputs():
    root = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp159-work/inputs'))
    pins = {}
    for name in ['frame12', 'track12']:
        p = root / (name + '.parquet')
        rec = json.loads(p.with_suffix('.json').read_text())
        check(name + ' output pin', sha(p) == rec['output_sha256'])
        pins[name] = artifact(p, record=artifact(p.with_suffix('.json')))
    obs = [c for c in C.OBS if c not in C.RECEPTION]
    f = pd.read_parquet(root / 'frame12.parquet', columns=[ID, 'month', 'airport', 'shp_r_aobt',
                        *obs, 'clock_implied_row', 'clock_current250_row', *C.OPDI_RAW])
    t = pd.read_parquet(root / 'track12.parquet', columns=[ID, *[c for c in C.TRACK_CLOCK if not c.startswith('x_')], *C.TRACK_PUSH])
    check('A3 source IDs unique', f[ID].is_unique and t[ID].is_unique)
    check('A3 frame/track same IDs', set(f[ID]) == set(t[ID]))
    f = f.merge(t, on=ID, how='left', validate='one_to_one')
    f = pd.concat([f, C.derived(f)], axis=1)
    names = list(dict.fromkeys([*obs, *C.TABLE_NEW, *C.TRACK_CLOCK, *C.TRACK_PUSH]))
    check('no reception-density input', not set(names) & set(C.RECEPTION))
    return f.set_index(ID), names, pins

def arrays(train, held, categorical):
    nums = [c for c in train if c not in categorical]
    check('no label column in model inputs', not set(train) & {'y', 'y_true', 'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt', ID})
    check('matching input columns', list(train) == list(held))
    vocab = {}
    cats = {}
    for c in categorical:
        # Native categorical integer -1 and nullable raw categories both use zero.
        tr = train[c].astype('string').mask(train[c].isna() | train[c].astype('string').isin(['-1','-1.0']))
        sc = held[c].astype('string').mask(held[c].isna() | held[c].astype('string').isin(['-1','-1.0']))
        v = sorted(tr.dropna().unique().tolist())
        vocab[c] = v
        lookup = {x: i + 1 for i, x in enumerate(v)}
        cats[c] = (tr.map(lookup).fillna(0).to_numpy('int64'), sc.map(lookup).fillna(0).to_numpy('int64'))
    artifacts = {}
    for part, k in [('train', 0), ('held', 1)]:
        artifacts[part + '-cat'] = save_array(part + '-cat', np.stack([cats[c][k] for c in categorical], axis=1))
    del cats
    n = train[nums].to_numpy('float32')
    miss = ~np.isfinite(n)
    check('no infinite numerical train input', not np.isinf(n).any())
    med = np.nanmedian(n, axis=0)
    med = np.where(np.isfinite(med), med, 0).astype('float32')
    n = np.where(miss, med, n)
    mu = n.mean(axis=0, dtype='float64').astype('float32')
    sd = n.std(axis=0, dtype='float64').astype('float32')
    sd = np.maximum(sd, 1e-6).astype('float32')
    artifacts['train-num'] = save_array('train-num', np.clip((n-mu)/sd, -20, 20).astype('float32'))
    artifacts['train-missing'] = save_array('train-missing', miss)
    del n, miss
    n = held[nums].to_numpy('float32')
    check('no infinite numerical held input', not np.isinf(n).any())
    miss = ~np.isfinite(n)
    n = np.where(miss, med, n)
    artifacts['held-num'] = save_array('held-num', np.clip((n-mu)/sd, -20, 20).astype('float32'))
    artifacts['held-missing'] = save_array('held-missing', miss)
    pp = {'numeric': nums, 'categorical': categorical, 'vocab': vocab,
          'median': med.tolist(), 'mean': mu.tolist(), 'std': sd.tolist(),
          'category_sizes': [len(vocab[c])+1 for c in categorical],
          'unknown_index': 0, 'fit_on': 'admitted ten-month training rows only'}
    write(OUT / 'preprocessing.json', pp)
    artifacts['preprocessing'] = artifact(OUT / 'preprocessing.json')
    return pp, artifacts

def main():
    t0 = time.monotonic()
    cores = lane()
    OUT.mkdir(parents=True, exist_ok=True)
    check('no existing prepared manifest', not (OUT / 'INPUTS.json').exists())
    spec = W / 'results/mlp-007/SPEC-007.md'
    harness_record = json.loads((W / 'results/HARNESS.json').read_text())
    for n, digest in harness_record['files'].items():
        check('harness pin ' + n, sha(H / n) == digest)
    env = LW.setup(INPUT)
    print('source modules loaded', flush=True)
    PX, F134, base, D18, F, M, D, source_code = env
    split = P.split('jjrank')
    check('jjrank split token', split.token == '4abce89100ca3bad')
    check('joint hold exclusions', tuple(split.excluded) == tuple(LW.P.janjul().excluded))
    frames = {m: base.month_frame(m) for m in LW.MONTHS}
    print('twelve prepared month frames loaded', flush=True)
    input_pins = {str(base.locate_file(f'month_{m}.parquet')): sha(base.locate_file(f'month_{m}.parquet')) for m in LW.MONTHS}
    tr = pd.concat(list(frames.values()), ignore_index=True)
    tr = tr.loc[D.keep_reachable(tr)]
    tr = tr.loc[split.label_mask(LW.wp_frame(tr))].reset_index(drop=True)
    check('only ten training months', len(set(tr.month)) == 10 and not tr.month.isin(['2025-01','2025-07']).any())
    split.assert_training_frame(LW.wp_frame(tr), target=tr.y.to_numpy(), what='MLP007 inputs')
    held = [frames[m].loc[P.split('jjrefit').score_mask(LW.wp_frame(frames[m]))].drop(columns=['y']) for m in LW.SCORED_MONTHS]
    check('held y removed before encoding', all('y' not in h for h in held))
    raw, audit = LW.raw_months_cached(PX, W / 'cache')
    months = sorted(set(tr.month))
    # The cached arrival_events table drops its dates. Recover the same
    # permitted arrival fields from its pinned ARR-only source, retain dates,
    # and remove buffer days before any normalizing statistic is fitted.
    arr_path = PX.exact_file('proxy_arrivals.parquet')
    arr_hash = sha(arr_path)
    check('arrival normalizer source pin', arr_hash == audit['source_hashes']['proxy_arrivals.parquet'])
    arr = pd.read_parquet(arr_path, columns=PX.RAW_ARR_COLUMNS + ['month'])
    arr = arr.loc[arr.month.isin(months)].copy()
    arr_time = pd.to_datetime(arr.MVT_TIME_UTC_mvt, utc=True, errors='coerce')
    arr['airport'] = arr.ADES_mvt.map(PX.category)
    arr['day'] = arr_time.dt.strftime('%Y-%m-%d')
    arr['hour'] = arr_time.dt.hour
    arr['taxi_in'] = pd.to_numeric(arr.TAXITIME_SEC_mvt, errors='coerce')
    arr = arr.loc[arr_time.notna() & arr.taxi_in.notna()].copy()
    check('arrival source only ten months', set(arr.month) <= set(months))
    normalizer_support = {}
    # All normalizing statistics use ten months; exclude buffered days as well.
    denom_raw = {}
    for m in months:
        denom_raw[m] = dict(raw[m])
        denom_raw[m]['arrival_events'] = arr.loc[arr.month.eq(m), ['airport','hour','taxi_in','day']].copy()
        for field in ['arrival_events', 'hourly_counts']:
            a = denom_raw[m][field]
            if 'day' not in a:
                raise RuntimeError(f'{field}: no date column for exclusion check')
            keep = ~a.day.astype(str).isin(split.excluded)
            denom_raw[m][field] = a.loc[keep].copy()
            check('denominator admitted dates ' + m + ' ' + field, not denom_raw[m][field].day.astype(str).isin(split.excluded).any())
            normalizer_support[m+' '+field]={'rows':int(keep.sum()),'excluded_rows':int((~keep).sum()),
                 'first_day':str(a.loc[keep,'day'].astype(str).min()),'last_day':str(a.loc[keep,'day'].astype(str).max())}
    denoms = PX.fit_denominators(denom_raw, months)
    del arr
    tr = LW.with_clock_columns(tr, raw)
    held = [LW.with_clock_columns(h, raw) for h in held]
    # No pass-through row supplies a supervised table for the fitted member.
    tr = tr.loc[tr.ghost.to_numpy(float) < .5].reset_index(drop=True)
    held = [h.loc[h.ghost.to_numpy(float) < .5].reset_index(drop=True) for h in held]
    enc_tr, enc_sc, changed = LW.add_encodings_month_out(D18, tr, held)
    check('no held month in target encoding source', not enc_tr.month.isin(LW.SCORED_MONTHS).any())
    tr = enc_tr.reset_index(drop=True)
    sc = pd.concat(enc_sc, ignore_index=True)
    check('train/held IDs disjoint', not set(tr[ID]) & set(sc[ID]))
    print(f'ten-month encodings ready: train {len(tr)}, held {len(sc)}', flush=True)
    z = pd.concat([PX.transform_rows(raw[m], denoms)[0][[ID, *PX.Z_COLS]] for m in LW.MONTHS], ignore_index=True).set_index(ID)
    check('proxy IDs unique', z.index.is_unique)
    xt, xs, cols, cats = base.feature_frames(tr, sc, F, M)
    check('base Lp feature count', len(cols) == 151)
    xt = pd.concat([xt.reset_index(drop=True), F134.aligned_proxy_features(z, tr, months, 'MLP train').reset_index(drop=True)], axis=1)
    xs = pd.concat([xs.reset_index(drop=True), F134.aligned_proxy_features(z, sc, list(LW.SCORED_MONTHS), 'MLP held').reset_index(drop=True)], axis=1)
    check('Lp feature count', len(xt.columns) == 172)
    anchor_train, anchor_held = base.anchor_for(tr), base.anchor_for(sc)
    y = tr.y.to_numpy('float64')
    check('training y/anchors finite', np.isfinite(y).all() and np.isfinite(anchor_train).all() and np.isfinite(anchor_held).all())
    emb, raw_pins = raw_embeddings(frames)
    a3, a3_names, a3_pins = a3_inputs()
    for x, rows in [(xt,tr), (xs,sc)]:
        sel = a3.reindex(rows[ID])
        hit = sel.month.notna().to_numpy()
        check('A3 actual row identity ' + ('train' if x is xt else 'held'), np.array_equal(sel.airport[hit].astype(str), rows.airport[hit].astype(str)) and np.array_equal(sel.month[hit], rows.month[hit]))
        for c in cats:
            if c.startswith('c_'):
                check('known native categorical source '+c, c in NATIVE_SOURCES)
                mask = emb['missing_'+c].reindex(rows[ID])
                check('native category missing-mask IDs '+c, mask.notna().all())
                x[c] = x[c].mask(mask.to_numpy(bool), -1)
        for c in a3_names:
            if c not in x:
                x[c] = sel[c].to_numpy('float32')
        for c in EMB:
            x[c] = emb[c].reindex(rows[ID]).to_numpy()
    del frames, raw, denom_raw, z, emb, a3, enc_tr, enc_sc, held
    gc.collect()
    cats = [*cats, *EMB]
    check('all seven named embeddings', set(EMB) <= set(cats))
    nums = [c for c in xt if c not in cats]
    prefixes = ('r_', 'd_', 'q_', 'queue', 'dep_', 'arr_', 'rwy_', 'apt_', 'obs_', 'trk_', 'opdi_', 'clock_', 'x_', 'hour_', 'month_',
                'stand_', 'prev_', 'day_', 'w600_', 'w3600_', 'ms_', 'm_', 'aobt_', 'sched_', 'eobt_', 'arvt')
    # Keep explicit calendar and physical row fields that lack a family prefix.
    physical_names = {'hour','dow','dom','doy','sin_h','cos_h','sin_m','cos_m',
                      'stand_x','stand_y','stand_lat','stand_lon','stand_dist',
                      'stand_dist_km','stand_sin','stand_cos','taxitime','taxi_time',
                      'filed_taxi','slot_delay','prev_rwy_gap','prev_apt_gap',
                      'plan_flight_time','act_flight_time','demand_over_served','no_nm'}
    physical = [c for c in nums if c.startswith(prefixes) or c in PX.Z_COLS or c in physical_names]
    check('smaller physical set nonempty and smaller', 0 < len(physical) < len(nums))
    coverage={}
    for part,x,rows in [('train',xt,tr),('held',xs,sc)]:
        present=x.notna()
        for c in cats:
            present[c] &= ~x[c].astype('string').isin(['-1','-1.0'])
        coverage[part]={'overall':present.mean().to_dict(),
                       'by_month':present.groupby(rows.month.to_numpy()).mean().to_dict(),
                       'definition':'nonmissing input; categorical sentinel -1 is missing'}
    print(f'full {len(nums)} numeric columns; physical {len(physical)}; preprocessing starts', flush=True)
    pp, artifacts = arrays(xt, xs, cats)
    del xt, xs
    for part, rows, anchor in [('train',tr,anchor_train),('held',sc,anchor_held)]:
        meta = rows[[ID,'month','utc_day','airport']].copy()
        meta['anchor'] = anchor
        check(part + ' metadata has no label', 'y' not in meta)
        p = OUT / (part + '-metadata.parquet')
        meta.to_parquet(p, index=False)
        artifacts[part + '-metadata'] = artifact(p, rows=len(meta))
    artifacts['train-target'] = save_array('train-target', ((y-anchor_train)/1000).astype('float32'))
    features = {'full': {'numeric_indices':list(range(len(nums))), 'categorical_indices':list(range(len(cats))), 'numeric':nums, 'categorical':cats},
                'physical': {'numeric_indices':[nums.index(c) for c in physical], 'categorical_indices':[cats.index(c) for c in EMB], 'numeric':physical,'categorical':EMB}}
    # All fit inputs and lists exist before any model or held score.
    manifest = {'what':'MLP007 annual Lp + movement-only A3 inputs; no model fit or held score',
        'spec_sha256':sha(spec), 'split':split.manifest(), 'score_split':P.split('jjrefit').manifest(),
        'train_rows':len(tr), 'held_rows':len(sc), 'train_months':months, 'train_rows_by_month':tr.groupby('month').size().to_dict(),
        'features':features, 'coverage':coverage, 'artifacts':artifacts, 'base_source_code':source_code,
        'native_category_missing_sources':NATIVE_SOURCES,
        'source_files':input_pins, 'raw_embedding_files':raw_pins, 'A3_sources':a3_pins,
        'proxy_cache_audit':{k:v for k,v in audit.items() if k not in ['matches_by_month']},
        'proxy_denominator_months':months,'proxy_denominator_excluded_dates':list(split.excluded),
        'proxy_denominator_support':normalizer_support,'dated_arrival_normalizer_source':artifact(arr_path),
        'encoding_changed_score_values':changed, 'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['prepare.py','io_utils.py']},
        'source_module_files':{m.__name__:artifact(Path(m.__file__)) for m in [LW,P,PX,F134,base,D18,F,M,D,C]},
        'seconds':time.monotonic()-t0, 'cores':cores, 'no_model_fit':True, 'no_held_score':True}
    write(OUT / 'INPUTS.json', manifest)
    write(RESULT / 'CHECK-INPUTS.json', {'state':'pass','n_checks':len(CHECKS),'checks':CHECKS,'manifest_sha256':sha(OUT/'INPUTS.json')})
    print(json.dumps({'event':'inputs-ready','train_rows':len(tr),'held_rows':len(sc),'full_numeric':len(nums),'small_numeric':len(physical),'checks':len(CHECKS),'seconds':manifest['seconds']}), flush=True)

if __name__ == '__main__':
    main()
