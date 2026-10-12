"""Restricted proxy first; then hash-bound AOBT_3 extension build and read."""
import argparse
import copy
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from assembly import ID, COL, compose, clip
from diverse_blend import load_member
from stand_floor import apply_floor
from prepare_pair_extension import proxy_read
from read_stacks import GS

def id_digest(value):  # row IDs and callsigns stay out of the public source
    return hashlib.sha256(np.int64(value).tobytes() if not isinstance(value, str) else value.encode()).hexdigest()

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP/'inputs/stack3-aobt-011'
CANDIDATE = EXP/'exp169-stack3-runway-pair-ext-aobt-c55.parquet'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def inputs():
    old = json.loads((EXP/'results/stack3-ext-inputs-009.json').read_text())
    for key, path in old['source_paths'].items():
        assert sha(path) == old['source_sha256'][key], key
    paths = {key: Path(value) for key, value in old['source_paths'].items()}
    paths.update(raw=ROOT/'data/raw/ranking.parquet',
                 admitted_manifest=EXP/'inputs/stack3-ext-009/pair.json',
                 admitted_members=EXP/'inputs/stack3-ext-009/members-ranking.parquet',
                 candidate=EXP/'exp169-stack3-runway-pair-ext-c55.parquet',
                 off_candidate=EXP/'exp169-stack3-runway-pair-c55.parquet')
    assert sha(paths['candidate']) == 'de999d34a03cd74c44561b4595802066347d9f35253bbb7b627814a375aaceda'
    assert sha(paths['off_candidate']) == '80c672e3c78dac0a3a72dc5b6884a184c8b1467e40ccf93b3b0b0b27b8ff4c29'
    return paths


def aobt_ranking(c, path):
    raw = pd.read_parquet(path, columns=[ID, 'PHASE_mvt', 'AOBT_3_flt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')]
    assert raw[ID].is_unique and set(c[ID]).issubset(set(raw[ID]))
    return raw.set_index(ID).loc[c[ID], 'AOBT_3_flt'].notna().to_numpy()


def proxy():
    assert not OUT.exists() and not CANDIDATE.exists()
    paths = inputs()
    pins = {k: sha(v) for k, v in paths.items()}
    b = pd.read_parquet(paths['fold_bundle'])
    held = pd.read_parquet(paths['fold_members'])
    np.testing.assert_array_equal(b[ID], held[ID])
    rows = GS.load_rows().set_index(ID).loc[b[ID]]
    refs = pd.read_parquet(paths['refs']).set_index(ID).loc[b[ID]]
    v32 = refs.v32.to_numpy(float)
    head = pd.read_parquet(paths['head']).set_index(ID).pred_HB1.reindex(b[ID]).to_numpy(float)
    np.testing.assert_array_equal(head, b.HB1)
    seq = held.pred_seq170.to_numpy(float)
    y = GS.labels(b[ID].to_numpy('int64'))
    clock = b.cls.eq('clock').to_numpy()
    aobt = rows.aobt.to_numpy(bool)
    base = np.maximum(60., .5*v32+.5*np.maximum(60., head))
    on = clock & aobt & (v32 < 2400)
    pair = base.copy()
    pair[on] = np.maximum(60., base[on]+.32*(seq[on]-base[on]))
    delta = (y-base)**2-(y-pair)**2
    days = (rows.airport.astype(str)+'|'+rows.date.astype(str)).to_numpy()
    rng = np.random.default_rng(148)
    result = {'cells': {}}
    for month in [1, 7]:
        for airport in sorted(set(b.airport[clock & b.month.eq(month)])):
            mask = clock & aobt & b.month.eq(month).to_numpy() & b.airport.eq(airport).to_numpy()
            result['cells'][f'{airport}-{month:02d}'] = proxy_read(delta[mask], days[mask], rng)
    masks = {'january_six': clock & aobt & b.month.eq(1).to_numpy() & b.airport.isin(['EDDF','EDDM','EGLL','EHAM','LEBL','LSZH']).to_numpy()}
    masks.update({a+'_july': clock & aobt & b.month.eq(7).to_numpy() & b.airport.eq(a).to_numpy() for a in ['LEMD','LFPG','LIRF']})
    for name, mask in masks.items():
        result[name] = proxy_read(delta[mask], days[mask], rng)
    required = ['january_six', 'LEMD_july', 'LFPG_july']
    assert all(result[k]['gain_mse_per_row'] > 0 for k in required)
    c = pd.read_parquet(paths['ranking_bundle'])
    aobt = aobt_ranking(c, paths['raw'])
    old_scope = c.extension.to_numpy(bool) & (c.F.to_numpy() < 2400)
    scope = old_scope & aobt
    excluded = old_scope & ~aobt
    assert old_scope.sum() == 16808 and scope.sum() == 16748 and excluded.sum() == 60
    assert excluded[c[ID].map(id_digest).eq('5f4a4a3ca4990c2ceb4354b92bc9759cb2912c57cbdf1309e4d3f9f2b20fce1b')].all()
    scale = float(scope.sum()/old_scope.sum())
    parent = json.loads((EXP/'results/stack3-forecast-frozen-007.json').read_text())
    forecast = {'central': float(np.sqrt(parent['central']**2-25*scale)),
                'scenario_range': [float(np.sqrt(parent['scenario_range'][0]**2-30*scale)),
                                   float(np.sqrt(parent['scenario_range'][1]**2-20*scale))],
                'extension_ranking_MSE_gain': 25*scale, 'extension_gain_range': [20*scale,30*scale],
                'coverage_scale': scale, 'parent_forecast': parent, 'ratio':1.24,
                'ratio_basis':'ledger899 pooled v44/v45 vs v43; v47 base row901',
                'rule':'SPEC-011: unchanged forecast per admitted extension row; no gain credit for squared distance removed'}
    result.update(source_proxy_sha256=pins['proxy'], source_rule='e59b39a2', source_report='597a9776',
                  recomputed_by='stack, AOBT_3-present rows only', source_rows=[899,901],
                  required_positive_points=required, clock_rows=int((clock & rows.aobt.to_numpy(bool)).sum()),
                  rows_blended=int(on.sum()), added_scope=int(scope.sum()), excluded_rows=int(excluded.sum()),
                  by_airport=c.loc[scope].groupby('airport').size().to_dict(),
                  excluded_by_airport=c.loc[excluded].groupby('airport').size().to_dict(),
                  source_paths={k:str(v) for k,v in paths.items()}, source_sha256=pins,
                  forecast=forecast, code_sha256=sha(__file__), frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    OUT.mkdir()
    write_json(OUT/'restricted-proxy.json', result)
    write_json(EXP/'results/stack3-aobt-forecast-frozen-011.json', forecast)
    with (EXP/'SPEC-011.md').open('a') as f:
        f.write('\n## Restricted proxy and forecast, frozen before build\n\n')
        f.write('Recorded '+result['frozen_at_utc']+'. No candidate file exists at this point.\n\n')
        f.write('| proxy | rows | point | 95% airport-day interval | without top5 days |\n|---|---:|---:|---|---:|\n')
        for name in required:
            r=result[name]
            f.write(f"| {name} | {r['rows']} | {r['gain_mse_per_row']:.9f} | [{r['lo95']:.9f}, {r['hi95']:.9f}] | {r['without_top5_days']:.9f} |\n")
        f.write(f"\nAll three points are positive. New scope {int(scope.sum())}; exclusions {int(excluded.sum())}. Frozen forecast **{forecast['central']:.9f} s**, scenario **[{forecast['scenario_range'][0]:.9f}, {forecast['scenario_range'][1]:.9f}]**. Extension term **{25*scale:.9f}** rankingMSE. Restricted source JSON SHA256 `{sha(OUT/'restricted-proxy.json')}`.\n")
    print(json.dumps({k: result[k] for k in required}, indent=2))
    print('FORECAST', forecast['central'], forecast['scenario_range'])


def build():
    assert not CANDIDATE.exists()
    proof = json.loads((OUT/'restricted-proxy.json').read_text())
    assert sha(__file__) == proof['code_sha256']
    for k, path in proof['source_paths'].items():
        assert sha(path) == proof['source_sha256'][k], k
    paths = {k: Path(v) for k,v in proof['source_paths'].items()}
    c = pd.read_parquet(paths['ranking_bundle'])
    aobt = aobt_ranking(c, paths['raw'])
    previous = pd.read_parquet(paths['admitted_members'])
    np.testing.assert_array_equal(c[ID], previous[ID])
    excluded = c.extension.to_numpy(bool) & (c.F.to_numpy()<2400) & ~aobt
    members = previous.copy()
    members.loc[excluded,'member_scope'] = False
    assert (previous.member_scope != members.member_scope).sum() == 60
    for col in previous.columns:
        if col != 'member_scope': np.testing.assert_array_equal(members[col], previous[col])
    members.to_parquet(OUT/'members-ranking.parquet', index=False)
    spec = json.loads(paths['admitted_manifest'].read_text())
    spec['vectors']['fold']['path'] = '../stack3-007/members-fold.parquet'
    spec['vectors']['ranking'].update(path='members-ranking.parquet', sha256=sha(OUT/'members-ranking.parquet'))
    spec['extension_proxy'].update(path='restricted-proxy.json', sha256=sha(OUT/'restricted-proxy.json'),
        gate='clock extension, original F<2400 and AOBT_3 present; no ordinary scope added',
        recomputed_by='stack; same valid proxy_2 formula on AOBT_3-present rows; SPEC-011',
        source_proxy_sha256=proof['source_proxy_sha256'])
    write_json(OUT/'pair.json', spec)
    command = json.loads((EXP/'results/stack3-ext-build-command-009.json').read_text())
    argv = command['argv']
    argv[argv.index('--blend-manifest')+1] = str((OUT/'pair.json').relative_to(ROOT))
    argv[argv.index('--blend-manifest-sha256')+1] = sha(OUT/'pair.json')
    argv[argv.index('--out')+1] = str(CANDIDATE.relative_to(ROOT))
    code = {name:sha(EXP/'code'/name) for name in ['assembly.py','pair_blend.py','diverse_blend.py','stand_floor.py','stack3_aobt.py']}
    write_json(EXP/'results/stack3-aobt-build-command-011.json', {'argv':argv,'code_sha256':code,'spec_sha256':sha(EXP/'SPEC-011.md')})
    subprocess.run(argv, cwd=ROOT, check=True)
    config = dict(route='R4mk',noplan='v3cal',practice='selected',c55=True,pair_extension=True)
    checks = {}
    for split in ['fold','ranking']:
        b = pd.read_parquet(paths[split+'_bundle'])
        old = load_member(paths['admitted_manifest'], proof['source_sha256']['admitted_manifest'], b, split)
        new = load_member(OUT/'pair.json', sha(OUT/'pair.json'), b, split)
        before, after = compose(b,blend=old,**config), compose(b,blend=new,**config)
        off = compose(b,blend=new,**{**config,'pair_extension':False})
        removed = old.scope & ~new.scope
        assert not (new.scope & ~old.scope).any()
        np.testing.assert_array_equal(before[~removed],after[~removed])
        np.testing.assert_array_equal(after[removed],off[removed])
        pre = compose(b,blend=new,**{**config,'c55':False,'c33':False})
        clipped = clip(pre,b.lo.to_numpy(),b.hi.to_numpy())
        manual = apply_floor(clipped,b.stand_floor.to_numpy(),b.airport.to_numpy(),b.lo.to_numpy(),b.hi.to_numpy())
        np.testing.assert_array_equal(manual,after)
        checks[split] = {'removed':int(removed.sum()), 'changed':int((before!=after).sum()),
                         'C33_clips':int((pre!=clipped).sum()),'outside_exact':True,'parent_restored':True}
        if split == 'fold':
            np.testing.assert_array_equal(before,after)
            continue
        np.testing.assert_array_equal(removed,excluded)
        saved = pd.read_parquet(CANDIDATE)
        template = pd.read_parquet(ROOT/'data/raw/submitting.parquet',columns=[ID])
        assert list(saved.columns)==[ID,COL] and len(saved)==344841 and saved[ID].is_unique
        assert saved[ID].dtype==saved[COL].dtype==np.dtype('float64')
        np.testing.assert_array_equal(saved[ID],template[ID])
        np.testing.assert_array_equal(saved[ID],b[ID])
        np.testing.assert_array_equal(saved[COL],after)
        np.testing.assert_array_equal(before,pd.read_parquet(paths['candidate'])[COL])
        np.testing.assert_array_equal(off,pd.read_parquet(paths['off_candidate'])[COL])
        assert np.isfinite(after).all() and (after>=60).all()
        np.testing.assert_array_equal(clip(after,b.lo.to_numpy(),b.hi.to_numpy()),after)
        np.testing.assert_array_equal(apply_floor(after,b.stand_floor.to_numpy(),b.airport.to_numpy(),b.lo.to_numpy(),b.hi.to_numpy()),after)
        audit = b.loc[removed,[ID,'airport','month','F','lo','hi']].copy()
        audit['old'] = before[removed];audit['new'] = after[removed];audit['parent'] = off[removed]
        audit.to_parquet(EXP/'results/stack3-aobt-restored-rows-011.parquet',index=False)
        checks[split].update(distance_vs_off=float(np.mean((after-off)**2)),distance_removed=float(np.mean((before-after)**2)),
                            audited_row=audit[audit[ID].map(id_digest).eq('5f4a4a3ca4990c2ceb4354b92bc9759cb2912c57cbdf1309e4d3f9f2b20fce1b')].iloc[0].to_dict())
    for k,path in proof['source_paths'].items(): assert sha(path)==proof['source_sha256'][k],k
    read = {'source_rows':[899,901], 'candidate':str(CANDIDATE.relative_to(ROOT)), 'sha256':sha(CANDIDATE),
            'manifest_sha256':sha(OUT/'pair.json'),'restricted_proxy_sha256':sha(OUT/'restricted-proxy.json'),
            'forecast':proof['forecast'],'checks':checks,'code_sha256':code,
            'held_scores_unchanged':json.loads((EXP/'results/stack3-ext-009.json').read_text())['held_scores_unchanged'],
            'review':'pending'}
    write_json(EXP/'results/stack3-aobt-011.json',read)
    print(json.dumps({'sha256':read['sha256'],'checks':checks},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['proxy','build'])
    {'proxy':proxy,'build':build}[parser.parse_args().stage]()
