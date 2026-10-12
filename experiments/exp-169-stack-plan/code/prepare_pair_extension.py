"""Reproduce valid's fixed proxy and bind the admitted annual clock extension."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assembly import ID, compose
from read_stacks import GS

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
VALID = ROOT.parent/'prc-data-challenge-2026-exp147/experiments/exp-148-short-route'
OUT = EXP/'inputs/stack3-ext-009'
FIT_FIELD = 'owner'  # record field that names the reader
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def proxy_read(delta, days, rng):
    grouped = pd.DataFrame({'day': days, 'gain': delta}).groupby('day').gain.agg(['sum','size'])
    sums, counts = grouped['sum'].to_numpy(), grouped['size'].to_numpy()
    draw = rng.integers(0, len(grouped), (2000, len(grouped)))
    boot = sums[draw].sum(1)/counts[draw].sum(1)
    return {'rows': len(delta), 'airport_days': len(grouped), 'gain_mse_per_row': float(delta.mean()),
            'lo95': float(np.quantile(boot,.025)), 'hi95': float(np.quantile(boot,.975)),
            'without_top5_days': float((delta.sum()-np.sort(sums)[-5:].sum())/len(delta))}


def main():
    assert not OUT.exists(), 'Extension inputs are write-once.'
    parent = json.loads((EXP/'results/stack3-inputs-007.json').read_text())
    paths = {'proxy': VALID/'results/ext-pair/ext-pair.json', 'proxy_code': VALID/'code/ext_pair.py',
             'refs': ROOT/'data/exp167/refs.parquet', 'head': EXP/'inputs/HB1-jjrefit.parquet',
             'manifest': ROOT/parent['manifest'],
             'fold_bundle': EXP/'inputs/stack3-007/bundle-fold.parquet',
             'ranking_bundle': EXP/'inputs/stack3-007/bundle-ranking.parquet',
             'fold_members': EXP/'inputs/stack3-007/members-fold.parquet',
             'ranking_members': EXP/'inputs/stack3-007/members-ranking.parquet'}
    pins = {name: sha(path) for name,path in paths.items()}
    proxy = json.loads(paths['proxy'].read_text())
    assert pins['proxy'] == '2dbe818851f754129d2aeb7e35fb91dc0f68ea6a8a0b95d354e1a2a63d67c741'
    assert pins['proxy_code'] == proxy['code_sha256']['ext_pair.py']
    assert pins['head'] == proxy['inputs_sha256']['HB1-jjrefit']
    assert parent['source_sha256']['sequence'] == proxy['inputs_sha256']['sequence jjrefit']
    assert pins['manifest'] == parent['manifest_sha256']
    spec = json.loads(paths['manifest'].read_text())
    for split in ['fold','ranking']:
        assert pins[split+'_bundle'] == parent['bundle_sha256'][split]
        assert pins[split+'_members'] == spec['vectors'][split]['sha256']
    assert proxy['w_clock'] == spec['ranking_weights']['clock']['sequence'] == .32
    assert proxy['gate_v32_under'] == 2400
    assert proxy['rule'] == {'LEMD-01': True, 'LFPG-01': True}
    required = ['january_six','LEMD_july','LFPG_july']
    assert all(proxy[k]['gain_mse_per_row'] > 0 for k in required)

    b = pd.read_parquet(paths['fold_bundle'])
    held = pd.read_parquet(paths['fold_members'])
    np.testing.assert_array_equal(b[ID], held[ID])
    rows = GS.load_rows().set_index(ID).loc[b[ID]]
    refs = pd.read_parquet(paths['refs'], columns=[ID,'v32']).set_index(ID)
    assert refs.index.is_unique
    v32 = refs.v32.reindex(b[ID]).to_numpy(float)
    head = pd.read_parquet(paths['head']).set_index(ID).pred_HB1.reindex(b[ID]).to_numpy(float)
    np.testing.assert_array_equal(head, b.HB1)
    seq, y = held.pred_seq170.to_numpy(float), GS.labels(b[ID].to_numpy('int64'))
    clock = b.cls.eq('clock').to_numpy()
    assert np.isfinite(head[clock]).all() and np.isfinite(v32[clock]).all() and np.isfinite(seq[clock]).all()
    p = np.maximum(60., .5*v32 + .5*np.maximum(60.,head))
    on = clock & (v32 < 2400)
    pair = p.copy(); pair[on] = np.maximum(60.,p[on]+.32*(seq[on]-p[on]))
    delta = (y-p)**2-(y-pair)**2
    days = (rows.airport.astype(str)+'|'+rows.date.astype(str)).to_numpy()
    rng = np.random.default_rng(148)
    reads = {'cells': {}}
    for month in [1,7]:
        for airport in sorted(set(b.airport[clock & b.month.eq(month)])):
            mask = clock & b.month.eq(month).to_numpy() & b.airport.eq(airport).to_numpy()
            name = f'{airport}-{month:02d}'
            reads['cells'][name] = proxy_read(delta[mask],days[mask],rng)
            for key,value in reads['cells'][name].items():
                assert np.isclose(value,proxy['cells'][name][key],rtol=0,atol=1e-8),(name,key)
    masks = {'january_six': clock & b.month.eq(1).to_numpy() & b.airport.isin(['EDDF','EDDM','EGLL','EHAM','LEBL','LSZH']).to_numpy()}
    masks.update({a+'_july': clock & b.month.eq(7).to_numpy() & b.airport.eq(a).to_numpy() for a in ['LEMD','LFPG','LIRF']})
    for name,mask in masks.items():
        reads[name] = proxy_read(delta[mask],days[mask],rng)
        for key,value in reads[name].items():
            assert np.isclose(value,proxy[name][key],rtol=0,atol=1e-8),(name,key)
    assert int(clock.sum()) == proxy['clock_rows'] and int(on.sum()) == proxy['rows_blended']
    print('PROXY PARITY', {k: reads[k]['gain_mse_per_row'] for k in required},flush=True)

    c = pd.read_parquet(paths['ranking_bundle'])
    annual = pd.read_parquet(paths['ranking_members'])
    np.testing.assert_array_equal(c[ID],annual[ID])
    extension = c.extension.to_numpy(bool)
    added = extension & (c.F.to_numpy()<2400)
    assert not (annual.member_scope.to_numpy() & extension).any()
    assert int(added.sum()) == 16808 and c.cls[added].eq('clock').all()
    assert c.month[added].eq(1).all() and c.airport[added].isin(['LEMD','LFPG']).all()
    assert np.isfinite(annual.pred_seq170[added]).all() and (annual.pred_seq170[added]>=60).all()
    np.testing.assert_array_equal(c.R4mk[extension],c.F[extension])
    pre = compose(c,route='R4mk',noplan='none',practice='none',c33=False)
    np.testing.assert_array_equal(pre[extension],np.maximum(60.,.5*c.F[extension]+.5*np.maximum(60.,c.HB1[extension])))
    new_annual = annual.copy()
    new_annual.loc[added,'member_scope'] = True
    for column in annual.columns:
        if column != 'member_scope':np.testing.assert_array_equal(annual[column],new_annual[column])
    assert np.array_equal(new_annual.control_scope,new_annual.member_scope & c.cls.eq('ordinary'))
    assert not (added & (c.noplan | c.practice)).any()
    OUT.mkdir()
    (OUT/'valid-ext-pair.json').write_bytes(paths['proxy'].read_bytes())
    (OUT/'valid-ext-pair.py').write_bytes(paths['proxy_code'].read_bytes())
    (OUT/'VALID-REPORT.md').write_bytes((VALID/'REPORT.md').read_bytes())
    new_annual.to_parquet(OUT/'members-ranking.parquet',index=False)
    new_spec = copy.deepcopy(spec)
    new_spec['vectors']['fold']['path'] = '../stack3-007/members-fold.parquet'
    new_spec['vectors']['ranking'].update(path='members-ranking.parquet',sha256=sha(OUT/'members-ranking.parquet'))
    new_spec['extension_proxy'] = {FIT_FIELD:'valid','read':'proxy_2','path':'valid-ext-pair.json',
        'sha256':pins['proxy'],'gain_path':['LEMD_july','gain_mse_per_row'],
        'required_positive_paths':[[key,'gain_mse_per_row'] for key in required],
        'admitted_cells':['LEMD-01','LFPG-01'],'rule_commit':'e59b39a2','report_commit':'597a9776',
        'gate':'clock extension and original F(=v32)<2400; no ordinary scope added'}
    (OUT/'pair.json').write_text(json.dumps(new_spec,indent=2)+'\n')
    for key,path in paths.items():assert sha(path) == pins[key],key
    record = {'source_paths':{key:str(path) for key,path in paths.items()},'source_sha256':pins,
              'source_commits':{'rule':'e59b39a2','report':'597a9776'},'proxy_reproduction':reads,
              'positive_rule':{key:proxy[key]['gain_mse_per_row'] for key in required},
              'added_scope':int(added.sum()),'by_airport':c.loc[added].groupby('airport').size().to_dict(),
              'manifest':str((OUT/'pair.json').relative_to(ROOT)),'manifest_sha256':sha(OUT/'pair.json'),
              'ranking_vector_sha256':sha(OUT/'members-ranking.parquet'),
              'bundle_sha256':parent['bundle_sha256'],
              'code_sha256':{q.name:sha(q) for q in [Path(__file__),EXP/'code/assembly.py',EXP/'code/pair_blend.py']}}
    (EXP/'results/stack3-ext-inputs-009.json').write_text(json.dumps(record,indent=2)+'\n')
    print('EXTENSION INPUTS',record['added_scope'],record['by_airport'],record['manifest_sha256'],flush=True)


if __name__ == '__main__':
    main()
