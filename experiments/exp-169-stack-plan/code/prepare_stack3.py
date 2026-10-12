"""Bind the admitted annual pair and select the predeclared pooled 2025 weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assembly import compose, clip, ID
from diverse_blend import load_member, vector_digest
from pair_blend import PairedBlend
from read_stacks import GS

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
LANE = ROOT/'experiments/exp-170-diverse'
OUT = EXP/'inputs/stack3-007'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Stack3 inputs are write-once.'
    source = {
        'pack': LANE/'results/pack-runway-pair-011/PACK.json',
        'annual': LANE/'results/pack-runway-pair-011/members-final.parquet',
        'read': LANE/'results/paired-runway-008-B/PAIRED.json',
        'source_served': LANE/'results/paired-runway-008-B/paired_ls.parquet',
        'sequence': LANE/'results/sequence-runway-006-fit/jjrefit.parquet',
        'control': LANE/'results/vectors/row-001/jjrefit.parquet',
        'sequence_fit': LANE/'results/sequence-runway-006-fit/FIT.json',
        'control_fit': LANE/'results/fit-records/row-001/FIT.json',
        'sequence_annual_fit': LANE/'results/runway-final-009-FIT.json',
        'control_annual_fit': LANE/'results/row-final-005-FIT.json',
        'fold_bundle': EXP/'inputs/c55-006/bundle-fold.parquet',
        'ranking_bundle': EXP/'inputs/c55-006/bundle-ranking.parquet',
        'ranking_raw': ROOT/'data/raw/ranking.parquet',
    }
    pins = {k:sha(p) for k,p in source.items()}
    pack = json.loads(source['pack'].read_text())
    read = json.loads(source['read'].read_text())
    assert pins['pack'] == '96ce41d5fa425dc73299fbc4a77dae26e87224c9ead30a36a85d9e678ac4060e'
    assert pins['annual'] == pack['vector_sha256'] == 'eed0835c30d66be7c687feb0a44d76a9c3b2cbe6a8084b824710602d00ef16d7'
    assert pins['read'] == pack['read_sha256']
    assert pack['weights'] == read['forms']['paired_ls']['weights']
    for name in ['sequence','control']:
        r = pack['records'][name]
        assert pins[name] == r['held_vector_sha256'] == read['fits'][name]['vector_sha256']
        assert pins[name+'_fit'] == r['held_fit_sha256'] == read['fits'][name]['fit_sha256']
        assert pins[name+'_annual_fit'] == r['annual_fit_sha256']
        fit = json.loads(source[name+'_fit'].read_text())
        annual = json.loads(source[name+'_annual_fit'].read_text())
        assert fit['train_months'] == [2,3,4,5,6,8,9,10,11,12]
        assert annual['train_months'] == list(range(1,13))
        assert fit['inputs_manifest_sha256'] == annual['inputs_manifest_sha256'] == r['input_manifest_sha256']
        assert fit['code_sha256'] == annual['code_sha256']
        for k in ['mode','loss','epochs','seed','batch_size']:
            assert fit['args'][k] == annual['args'][k]
    b,c = pd.read_parquet(source['fold_bundle']),pd.read_parquet(source['ranking_bundle'])
    ids = b[ID].to_numpy('int64')
    rows = GS.load_rows().set_index(ID).loc[ids]
    assert np.array_equal(rows.cls,b.cls) and np.array_equal(rows.month,b.month)
    count = rows.groupby(['month','airport','cls']).size()
    cells = GS.load_cells().set_index(['month','airport','cls']).N
    w = np.array([cells.get((m,a,z),0)/GS.N_RANKING/count[m,a,z] if count[m,a,z]>=GS.MIN_SUPPORT else 0.
                  for m,a,z in zip(rows.month,rows.airport,rows.cls)])
    scope = (b.F.to_numpy() < 2400) & rows.aobt.to_numpy(bool) & b.cls.ne('noac').to_numpy() & (w > 0) & ~b.noplan.to_numpy(bool)
    held = b[[ID,'month','cls']].copy()
    held['member_scope'] = scope
    held['control_scope'] = scope & b.cls.eq('ordinary').to_numpy()
    for name,col in [('sequence','pred_seq170'),('control','pred_row170')]:
        f = pd.read_parquet(source[name]).set_index(ID)
        assert f.index.is_unique and set(f.index) == set(ids)
        held[col] = f.pred_seq170.reindex(ids).to_numpy(float)
    ranking = pd.read_parquet(source['annual'])
    assert np.array_equal(ranking[ID].to_numpy(),c[ID].to_numpy())
    raw = pd.read_parquet(source['ranking_raw'],columns=[ID,'PHASE_mvt','AIRCRAFT_TYPE_mvt','AOBT_3_flt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')].set_index(ID).loc[c[ID]]
    clock = np.isfinite(c.HB1.to_numpy())
    assert np.array_equal(ranking.clock,clock)
    c['cls'] = np.where(raw.AIRCRAFT_TYPE_mvt.isna(),'noac',np.where(clock,'clock','ordinary'))
    ranking['cls'] = c.cls.to_numpy()
    support = np.array([count.get((m,a,z),0)>=GS.MIN_SUPPORT for m,a,z in zip(c.month,c.airport,c.cls)])
    expected_scope = (c.F.to_numpy()<2400) & raw.AOBT_3_flt.notna().to_numpy() & raw.AIRCRAFT_TYPE_mvt.notna().to_numpy() & support
    assert np.array_equal(ranking.member_scope,expected_scope)
    assert np.array_equal(ranking.control_scope,expected_scope & c.cls.eq('ordinary').to_numpy())
    assert int(expected_scope.sum()) == 312416 and not (expected_scope & c.noplan).any()
    assert np.array_equal(ranking.month,c.month)

    base = compose(b,route='R4mk',noplan='v3cal',practice='selected')
    # On scope, this is the source pred_pre. No-plan cache changes are outside scope.
    pre = compose(b,route='R4mk',noplan='none',practice='none',c33=False)
    pre26 = compose(c,route='R4mk',noplan='none',practice='none',c33=False)
    folds = {}
    ws,wc = np.zeros(len(b)),np.zeros(len(b))
    for cls in ['ordinary','clock']:
        folds[cls] = {}
        for m in [1,7]:
            r = pack['weights'][cls][str(m)]
            pair = r.get('weights',{'sequence':r.get('weight'),'control':0.})
            folds[cls][str(m)] = dict(pair,trained_on_months=[r['fit_month']])
            on = scope & b.cls.eq(cls).to_numpy() & b.month.eq(m).to_numpy()
            ws[on],wc[on] = pair['sequence'],pair['control']
    fixture = PairedBlend(ids,held.pred_seq170.to_numpy(),held.pred_row170.to_numpy(),scope,
                           held.control_scope.to_numpy(),ws,wc,vector_digest(ids,pre))
    pred = compose(b,route='R4mk',noplan='v3cal',practice='selected',blend=fixture)
    saved = pd.read_parquet(source['source_served']).set_index(ID).pred_seq170.reindex(ids).to_numpy()
    np.testing.assert_allclose(pred,saved,rtol=0,atol=1e-9)
    y = GS.labels(ids)
    source_score = GS.score(ids,base,pred,y=y,require_complete=True)
    want = read['forms']['paired_ls']['score']
    assert abs(source_score['G']-want['G']) < 1e-7
    for k in ['lo95','hi95']:
        assert abs(source_score['interval'][k]-want['interval'][k]) < 1e-7
    print('SOURCE PARITY',source_score['G'],source_score['interval'],flush=True)

    # The desk's rule is pooled across both months; no new model or gate choice.
    practice = b.practice.to_numpy(bool)
    scale = np.ones(len(b));scale[practice] = 1-b.practice_weight.to_numpy()[practice]
    X = np.column_stack([(held[col].to_numpy()-pre)*scale for col in ['pred_seq170','pred_row170']])
    ordinary = scope & b.cls.eq('ordinary').to_numpy()
    sw = np.sqrt(w[ordinary])
    design,residual = X[ordinary]*sw[:,None],(y-base)[ordinary]*sw
    beta = np.linalg.lstsq(design,residual,rcond=None)[0]
    xtwx,xtwr = design.T@design,design.T@residual
    np.testing.assert_allclose(np.linalg.solve(xtwx,xtwr),beta,rtol=0,atol=1e-10)
    clock_on = scope & b.cls.eq('clock').to_numpy()
    grid = np.arange(51)*.02
    losses = []
    for v in grid:
        delta = v*X[:,0]
        raw_delta = np.divide(delta,scale,out=np.zeros_like(delta),where=scale>0)
        z = np.maximum(60,pre+raw_delta)
        z[practice] = np.maximum(60,z[practice]+b.practice_weight.to_numpy()[practice]*(b.practice_point.to_numpy()[practice]-z[practice]))
        z = clip(z,b.lo.to_numpy(),b.hi.to_numpy())
        losses.append(float(w[clock_on]@((y[clock_on]-z[clock_on])**2)))
    pooled = {'trained_on_months':[1,7],
              'ordinary':{'method':'pooled_practice_scaled_wls','sequence':float(beta[0]),'control':float(beta[1]),
                           'xtwx':xtwx.tolist(),'xtwr':xtwr.tolist(),'fit_rows':int(ordinary.sum())},
              'clock':{'method':'pooled_served_grid','sequence':float(grid[int(np.argmin(losses))]),'control':0.,
                       'grid':grid.tolist(),'losses':losses,'fit_rows':int(clock_on.sum())}}
    print('POOLED',beta,pooled['clock']['sequence'],flush=True)
    OUT.mkdir()
    vectors = {}
    for split,frame,bundle,pp in [('fold',held,b,pre),('ranking',ranking,c,pre26)]:
        path = OUT/('members-'+split+'.parquet');frame.to_parquet(path,index=False)
        bundle.to_parquet(OUT/('bundle-'+split+'.parquet'),index=False)
        vectors[split] = {'path':path.name,'sha256':sha(path),'base_digest':vector_digest(bundle[ID],pp)}
    spec = {'schema':3,'lane':'C','member':'runway-pair-011','weight_basis':'two_fold_2025','serving_point':'after_head',
            'member_training_months':{'fold':[2,3,4,5,6,8,9,10,11,12],'ranking':list(range(1,13))},
            'fold_weights':folds,'ranking_weights':pooled,'vectors':vectors,'source_sha256':pins,
            'weight_rule':'SPEC-007: other-month held, pooled two-month annual; negative control retained'}
    path = OUT/'pair.json';path.write_text(json.dumps(spec,indent=2)+'\n')
    loaded = load_member(path,sha(path),b,'fold')
    np.testing.assert_array_equal(compose(b,route='R4mk',noplan='v3cal',practice='selected',blend=loaded),pred)
    # Load-time validation of the annual contract; candidate is built by assembly CLI later.
    load_member(path,sha(path),c,'ranking')
    for k,p in source.items():assert sha(p) == pins[k],k
    record = {'sources':{k:str(p.relative_to(ROOT)) for k,p in source.items()},'source_sha256':pins,
              'source_score':source_score,'source_vector_max_abs_difference':float(np.max(np.abs(pred-saved))),
              'manifest':str(path.relative_to(ROOT)),'manifest_sha256':sha(path),
              'bundle_sha256':{s:sha(OUT/f'bundle-{s}.parquet') for s in ['fold','ranking']},
              'ranking_weights':pooled,'fold_scope':int(scope.sum()),'ranking_scope':int(expected_scope.sum()),
              'code_sha256':{p.name:sha(p) for p in [Path(__file__),EXP/'code/assembly.py',EXP/'code/diverse_blend.py',EXP/'code/pair_blend.py']}}
    (EXP/'results/stack3-inputs-007.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
