"""Unit 008: build the full MLP's inputs for one fit (final, jjrank or jjx-MM); no fit, no held score.

Unit 007's builder (code/mlp007/prepare.py, sha d91d9802) with the training split and the scored rows
set by --fit. Everything a fit learns (target encodings, proxy normalizers, vocabularies, medians,
means, scales) comes from that fit's permitted training rows. Scored rows: final = the 2026 ranking
rows; jjrank = unit 007's jjrefit rows, then the 2026 ranking rows (unit 007's models score them);
jjx-MM = month MM rows of the split's score mask. Scored labels are dropped before any encoding.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import os
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
REF = W / 'inputs/mlp-007'
RESULT = W / 'results/mlp-008'
F5 = Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp153-work/out/f5-r4-023/pack'))
F5_PACK_SHA = 'ae56121090f0d10d9dd8da64ff7212c10a6f625b727175c7bd0432b6ad5694fa'
REF_INPUTS_SHA = '7899899abbbaf8411d58cc767ddfb0cf19ddf2495561a3cd9b1810913c5da79e'
JJX = ('jjx-02', 'jjx-03', 'jjx-06', 'jjx-08', 'jjx-10', 'jjx-11', 'jjx-12')
FITS = ('final', 'jjrank') + JJX
sys.path.insert(0, str(H))
import learner_wp as LW
sys.path.insert(0, str(W / 'code/seasonal'))
import common as C
_spec = importlib.util.spec_from_file_location('mlp008_pilot', C.SOURCE / 'experiments/exp-147-foldserve/code/pilot.py')
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

def save_array(out, name, x):
    p = out / (name + '.npy')
    if p.exists():
        raise RuntimeError('array exists')
    np.save(p, x, allow_pickle=False)
    return artifact(p, shape=list(x.shape), dtype=str(x.dtype))

def raw_embeddings(frames):
    """Unit 007's raw embeddings; a 2026 month reads ranking.parquet (no label column is read)."""
    cols = [ID, 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt',
            'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'MVT_TIME_UTC_mvt',
            'WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','FLIGHT_RULE_mvt','ADES_mvt']
    pieces, pins, ranking = [], {}, None
    for m in frames:
        if m.startswith('2026'):
            p = RAW / 'ranking.parquet'
            if ranking is None:
                ranking = pd.read_parquet(p, columns=cols)
            a = ranking
        else:
            p = RAW / f'training_{m}-01_{(pd.Period(m)+1).strftime("%Y-%m")}-01.parquet'
            a = pd.read_parquet(p, columns=cols)
        pins[str(p)] = sha(p)
        a = a[a[ID].isin(frames[m][ID])].copy()
        check('raw embedding rows cover the frame ' + m, set(a[ID]) == set(frames[m][ID]))
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

def arrays(out, train, held, categorical):
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
        artifacts[part + '-cat'] = save_array(out, part + '-cat', np.stack([cats[c][k] for c in categorical], axis=1))
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
    artifacts['train-num'] = save_array(out, 'train-num', np.clip((n-mu)/sd, -20, 20).astype('float32'))
    artifacts['train-missing'] = save_array(out, 'train-missing', miss)
    del n, miss
    n = held[nums].to_numpy('float32')
    check('no infinite numerical held input', not np.isinf(n).any())
    miss = ~np.isfinite(n)
    n = np.where(miss, med, n)
    artifacts['held-num'] = save_array(out, 'held-num', np.clip((n-mu)/sd, -20, 20).astype('float32'))
    artifacts['held-missing'] = save_array(out, 'held-missing', miss)
    pp = {'numeric': nums, 'categorical': categorical, 'vocab': vocab,
          'median': med.tolist(), 'mean': mu.tolist(), 'std': sd.tolist(),
          'category_sizes': [len(vocab[c])+1 for c in categorical],
          'unknown_index': 0, 'fit_on': 'admitted ten-month training rows only'}
    write(out / 'preprocessing.json', pp)
    artifacts['preprocessing'] = artifact(out / 'preprocessing.json')
    return pp, artifacts

def native_ids(fit):
    """IDs of grind's F5 pack vector for a jjx split or jjrank (native L rows), hash-checked."""
    check('F5 pack pin', sha(F5 / 'PACK.json') == F5_PACK_SHA)
    pack = json.loads((F5 / 'PACK.json').read_text())
    item = [a for a in pack['artifacts'] if a['fit'] == fit]
    check('one F5 artifact ' + fit, len(item) == 1)
    p = F5 / Path(item[0]['path']).name
    check('F5 vector pin ' + fit, sha(p) == item[0]['sha256'])
    ids = pd.read_parquet(p, columns=[ID])[ID].astype('int64')
    check('F5 IDs unique ' + fit, ids.is_unique)
    return set(ids), artifact(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', choices=FITS, required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    cores = lane()
    out = W / 'inputs/mlp-008' / a.fit
    out.mkdir(parents=True, exist_ok=True)
    check('no existing prepared manifest', not (out / 'INPUTS.json').exists())
    spec = RESULT / 'SPEC-008.md'
    check('unit 007 manifest pin', sha(REF / 'INPUTS.json') == REF_INPUTS_SHA)
    ref = json.loads((REF / 'INPUTS.json').read_text())
    harness_record = json.loads((W / 'results/HARNESS.json').read_text())
    for n, digest in harness_record['files'].items():
        check('harness pin ' + n, sha(H / n) == digest)
    env = LW.setup(INPUT)
    print('source modules loaded', flush=True)
    PX, F134, base, D18, F, M, D, source_code = env
    split = P.split(a.fit)
    ranking = a.fit in ('final', 'jjrank')
    if a.fit == 'jjrank':
        check('jjrank split is unit 007 split', split.manifest() == ref['split'])
    if a.fit == 'final':
        check('final split admits every day', len(split.excluded) == 0)
    frames = {m: base.month_frame(m) for m in LW.MONTHS}
    print('twelve prepared month frames loaded', flush=True)
    input_pins = {str(base.locate_file(f'month_{m}.parquet')): sha(base.locate_file(f'month_{m}.parquet')) for m in LW.MONTHS}
    tr = pd.concat(list(frames.values()), ignore_index=True)
    tr = tr.loc[D.keep_reachable(tr)]
    tr = tr.loc[split.label_mask(LW.wp_frame(tr))].reset_index(drop=True)
    split.assert_training_frame(LW.wp_frame(tr), target=tr.y.to_numpy(), what=f'MLP008 {a.fit} inputs')
    months = sorted(set(tr.month))
    check('training months', not tr.utc_day.astype(str).isin(split.excluded).any() if 'utc_day' in tr else True, months=months)
    # scored parts: label column dropped before any encoding
    if a.fit == 'final':
        parts = []
    elif a.fit == 'jjrank':
        jr = P.split('jjrefit')
        parts = [frames[m].loc[jr.score_mask(LW.wp_frame(frames[m]))].drop(columns=['y']) for m in LW.SCORED_MONTHS]
    else:
        sm = P.scored_months(split)
        check('one scored month', len(sm) == 1 and sm[0] not in months, scored=list(sm))
        parts = [frames[m].loc[split.score_mask(LW.wp_frame(frames[m]))].drop(columns=['y']) for m in sm]
    n_fold_parts = len(parts)
    rank_frames = {}
    if ranking:
        for m in LW.RANK_MONTHS:
            rank_frames[m] = base.month_frame(m).drop(columns=['y'], errors='ignore')
            input_pins[str(base.locate_file(f'month_{m}.parquet'))] = sha(base.locate_file(f'month_{m}.parquet'))
        parts += [rank_frames[m] for m in LW.RANK_MONTHS]
    check('scored labels removed before encoding', all('y' not in h for h in parts))
    raw, audit = LW.raw_months_cached(PX, W / 'cache')
    # Dated ARR normalizers from the pinned ARR-only source; excluded days out before any statistic.
    arr_path = PX.exact_file('proxy_arrivals.parquet')
    check('arrival normalizer source pin', sha(arr_path) == audit['source_hashes']['proxy_arrivals.parquet'])
    arr = pd.read_parquet(arr_path, columns=PX.RAW_ARR_COLUMNS + ['month'])
    arr = arr.loc[arr.month.isin(months)].copy()
    arr_time = pd.to_datetime(arr.MVT_TIME_UTC_mvt, utc=True, errors='coerce')
    arr['airport'] = arr.ADES_mvt.map(PX.category)
    arr['day'] = arr_time.dt.strftime('%Y-%m-%d')
    arr['hour'] = arr_time.dt.hour
    arr['taxi_in'] = pd.to_numeric(arr.TAXITIME_SEC_mvt, errors='coerce')
    arr = arr.loc[arr_time.notna() & arr.taxi_in.notna()].copy()
    check('arrival source only training months', set(arr.month) <= set(months))
    normalizer_support = {}
    denom_raw = {}
    for m in months:
        denom_raw[m] = dict(raw[m])
        denom_raw[m]['arrival_events'] = arr.loc[arr.month.eq(m), ['airport','hour','taxi_in','day']].copy()
        for field in ['arrival_events', 'hourly_counts']:
            x = denom_raw[m][field]
            if 'day' not in x:
                raise RuntimeError(f'{field}: no date column for exclusion check')
            keep = ~x.day.astype(str).isin(split.excluded)
            denom_raw[m][field] = x.loc[keep].copy()
            check('denominator admitted dates ' + m + ' ' + field, not denom_raw[m][field].day.astype(str).isin(split.excluded).any())
            normalizer_support[m+' '+field] = {'rows':int(keep.sum()),'excluded_rows':int((~keep).sum()),
                 'first_day':str(x.loc[keep,'day'].astype(str).min()),'last_day':str(x.loc[keep,'day'].astype(str).max())}
    denoms = PX.fit_denominators(denom_raw, months)
    del arr
    tr = LW.with_clock_columns(tr, raw)
    parts = [LW.with_clock_columns(h, raw) for h in parts]
    # No pass-through row supplies a supervised table for the fitted member (unit 007 order).
    tr = tr.loc[tr.ghost.to_numpy(float) < .5].reset_index(drop=True)
    parts = [h.loc[h.ghost.to_numpy(float) < .5].reset_index(drop=True) for h in parts]
    enc_tr, enc_sc, changed = LW.add_encodings_month_out(D18, tr, parts)
    check('no scored month in target encoding source', not enc_tr.month.isin([p.month.iloc[0] for p in parts if len(p)]).any())
    changed_rank = None
    if ranking:
        _, _, changed_rank = LW.add_encodings_month_out(D18, tr, [p.copy() for p in parts[n_fold_parts:]])
        check('ranking encodings unchanged by the month-out rule', changed_rank == 0, changed=changed_rank)
    tr = enc_tr.reset_index(drop=True)
    sc = pd.concat(enc_sc, ignore_index=True)
    check('train/held IDs disjoint', not set(tr[ID]) & set(sc[ID]))
    held_months = sorted(set(sc.month))
    print(f'{a.fit} encodings ready: train {len(tr)}, held {len(sc)} ({held_months})', flush=True)
    z = pd.concat([PX.transform_rows(raw[m], denoms)[0][[ID, *PX.Z_COLS]] for m in [*LW.MONTHS, *(LW.RANK_MONTHS if ranking else ())]],
                  ignore_index=True).set_index(ID)
    check('proxy IDs unique', z.index.is_unique)
    xt, xs, cols, cats = base.feature_frames(tr, sc, F, M)
    check('base Lp feature count', len(cols) == 151)
    xt = pd.concat([xt.reset_index(drop=True), F134.aligned_proxy_features(z, tr, months, 'MLP train').reset_index(drop=True)], axis=1)
    xs = pd.concat([xs.reset_index(drop=True), F134.aligned_proxy_features(z, sc, held_months, 'MLP held').reset_index(drop=True)], axis=1)
    check('Lp feature count', len(xt.columns) == 172)
    anchor_train, anchor_held = base.anchor_for(tr), base.anchor_for(sc)
    y = tr.y.to_numpy('float64')
    check('training y/anchors finite', np.isfinite(y).all() and np.isfinite(anchor_train).all() and np.isfinite(anchor_held).all())
    emb_frames = {m: frames[m] for m in LW.MONTHS}
    emb_frames.update(rank_frames)
    emb, raw_pins = raw_embeddings(emb_frames)
    a3, a3_names, a3_pins = a3_inputs()
    a3_hit = {}
    for x, rows, part in [(xt, tr, 'train'), (xs, sc, 'held')]:
        sel = a3.reindex(rows[ID])
        hit = sel.month.notna().to_numpy()
        a3_hit[part] = float(hit.mean())
        a3_hit[part + '_by_month'] = pd.Series(hit).groupby(rows.month.to_numpy()).mean().round(4).to_dict()
        check('A3 actual row identity ' + part, np.array_equal(sel.airport[hit].astype(str), rows.airport[hit].astype(str)) and np.array_equal(sel.month[hit], rows.month[hit]))
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
    # Track coverage differs by month (2025-01/07 about 0.47, 2026 about 0.57; frame12 counts 10-02);
    # a join failure gives a share near zero, so each scored month needs half the training share.
    check('A3 tables join every scored month', all(v >= 0.5 * a3_hit['train'] for v in a3_hit['held_by_month'].values()), **a3_hit)
    del frames, raw, denom_raw, z, emb, a3, enc_tr, enc_sc, parts, rank_frames, emb_frames
    gc.collect()
    cats = [*cats, *EMB]
    nums = [c for c in xt if c not in cats]
    check('full numeric list equals unit 007', nums == ref['features']['full']['numeric'], n=len(nums))
    check('full categorical list equals unit 007', cats == ref['features']['full']['categorical'], n=len(cats))
    # scored ID contracts
    hid = set(sc[ID].astype('int64'))
    contract = {}
    if a.fit in JJX:
        ids, src = native_ids(a.fit)
        check('scored IDs equal F5 native L IDs ' + a.fit, hid == ids, ours=len(hid), native=len(ids))
        contract[a.fit] = src
    if ranking:
        rid = set(sc.loc[sc.month.isin(LW.RANK_MONTHS), ID].astype('int64'))
        ids, src = native_ids('jjrank')
        check('ranking IDs equal F5 native jjrank IDs', rid == ids, ours=len(rid), native=len(ids))
        contract['jjrank'] = src
    if a.fit == 'jjrank':
        ref_held = pd.read_parquet(REF / 'held-metadata.parquet')
        fold = sc.loc[~sc.month.isin(LW.RANK_MONTHS)]
        check('jjrefit rows equal unit 007 held rows in order', np.array_equal(fold[ID].astype('int64').to_numpy(), ref_held[ID].to_numpy('int64')))
        check('jjrefit rows first', np.array_equal(sc[ID].to_numpy()[:len(fold)], fold[ID].to_numpy()))
    pp, artifacts = arrays(out, xt, xs, cats)
    del xt, xs
    for part, rows, anchor in [('train',tr,anchor_train),('held',sc,anchor_held)]:
        meta = rows[[ID,'month','utc_day','airport']].copy()
        meta['anchor'] = anchor
        check(part + ' metadata has no label', 'y' not in meta)
        p = out / (part + '-metadata.parquet')
        meta.to_parquet(p, index=False)
        artifacts[part + '-metadata'] = artifact(p, rows=len(meta))
    artifacts['train-target'] = save_array(out, 'train-target', ((y-anchor_train)/1000).astype('float32'))
    if a.fit == 'jjrank':
        # unit 007 parity: the same training rows give byte-equal arrays and preprocessing
        for k in ['train-num', 'train-missing', 'train-cat', 'train-target', 'train-metadata', 'preprocessing']:
            check('jjrank equals unit 007 ' + k, artifacts[k]['sha256'] == ref['artifacts'][k]['sha256'])
        nf = int((~sc.month.isin(LW.RANK_MONTHS)).sum())
        for k in ['num', 'missing', 'cat']:
            new = np.load(out / f'held-{k}.npy', mmap_mode='r')
            old = np.load(REF / f'held-{k}.npy', mmap_mode='r')
            check('jjrank jjrefit held equals unit 007 held ' + k, new.shape[0] == nf + int(sc.month.isin(LW.RANK_MONTHS).sum()) and np.array_equal(new[:nf], old))
    features = {'full': {'numeric_indices':list(range(len(nums))), 'categorical_indices':list(range(len(cats))), 'numeric':nums, 'categorical':cats}}
    manifest = {'what':f'MLP008 {a.fit} inputs: unit 007 full feature set; no model fit or held score', 'fit':a.fit,
        'spec_sha256':sha(spec), 'split':split.manifest(), 'train_rows':len(tr), 'held_rows':len(sc),
        'train_months':months, 'held_months':held_months, 'train_rows_by_month':tr.groupby('month').size().to_dict(),
        'held_rows_by_month':sc.groupby('month').size().to_dict(),
        'features':features, 'artifacts':artifacts, 'base_source_code':source_code, 'unit007_inputs_sha256':REF_INPUTS_SHA,
        'id_contracts':contract, 'A3_hit_share':a3_hit,
        'native_category_missing_sources':NATIVE_SOURCES,
        'source_files':input_pins, 'raw_embedding_files':raw_pins, 'A3_sources':a3_pins,
        'proxy_cache_audit':{k:v for k,v in audit.items() if k not in ['matches_by_month']},
        'proxy_denominator_months':months,'proxy_denominator_excluded_dates':list(split.excluded),
        'proxy_denominator_support':normalizer_support,'dated_arrival_normalizer_source':artifact(arr_path),
        'encoding_changed_score_values':changed, 'encoding_changed_ranking_values':changed_rank,
        'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['prepare.py','io_utils.py']},
        'source_module_files':{m.__name__:artifact(Path(m.__file__)) for m in [LW,P,PX,F134,base,D18,F,M,D,C]},
        'seconds':time.monotonic()-t0, 'cores':cores, 'no_model_fit':True, 'no_held_score':True}
    write(out / 'INPUTS.json', manifest)
    write(RESULT / f'CHECK-INPUTS-{a.fit}.json', {'state':'pass','n_checks':len(CHECKS),'checks':CHECKS,'manifest_sha256':sha(out/'INPUTS.json')})
    print(json.dumps({'event':'inputs-ready','fit':a.fit,'train_rows':len(tr),'held_rows':len(sc),'held_months':held_months,
                      'checks':len(CHECKS),'seconds':manifest['seconds']}), flush=True)

if __name__ == '__main__':
    main()
