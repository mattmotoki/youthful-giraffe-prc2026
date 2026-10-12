"""Build C55 input bounds, audit 2025 holds and score B; write no upload file."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assembly import compose, ID, COL
from read_stacks import GS
from stand_floor import bounds, INPUT_COLUMNS, AIRPORTS

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
OUT = EXP/'inputs/c55-006'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def hold_table(frame):
    result = {}
    for name,on in [('all',np.ones(len(frame),bool)),('joined',frame.stand_joined.to_numpy()),
                    ('no_plan',~frame.stand_joined.to_numpy())]:
        s = frame[on]
        result[name] = {'n':len(s),'strict_holds':int((s.y >= s.stand_floor).sum()),
                        'tolerant_holds':int((s.y >= s.stand_floor-60).sum()),
                        'strict_rate':float((s.y >= s.stand_floor).mean()),
                        'tolerant_rate':float((s.y >= s.stand_floor-60).mean())}
    return result


def save_frame(frame,path):
    if path.exists():
        # Parquet normalizes all-bool object audit columns to bool on read-back.
        back = pd.read_parquet(path)
        pd.testing.assert_frame_equal(back,frame,check_dtype=False)
        assert back[ID].dtype == frame[ID].dtype
        if 'stand_floor' in frame:
            assert back.stand_floor.dtype == frame.stand_floor.dtype
    else:
        frame.to_parquet(path,index=False)


def main():
    code = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),EXP/'code/assembly.py',
            EXP/'code/stand_floor.py',EXP/'code/diverse_blend.py',ROOT/'prc/gscore.py']}
    files = sorted((ROOT/'data/raw').glob('training_2025-*.parquet'))
    assert len(files) == 12
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in files}
    for p in [ROOT/'data/raw/ranking.parquet',EXP/'inputs/unit2-ready/bundle-fold.parquet',
              EXP/'inputs/unit2-ready/bundle-ranking.parquet',
              ROOT/'experiments/exp-168-fresh-look/code/u2_reuse_joined_lib.py',
              ROOT/'experiments/exp-168-fresh-look/code/u9_extra.py']:
        inputs[str(p.relative_to(ROOT))] = sha(p)
    parts = []
    for p in files:
        d = pd.read_parquet(p,columns=INPUT_COLUMNS+['TAXITIME_SEC_mvt'])
        ap = d.ADEP_mvt.where(d.PHASE_mvt.eq('DEP'),d.ADES_mvt)
        parts.append(d[ap.isin(AIRPORTS)])
    raw = pd.concat(parts,ignore_index=True)
    del parts
    annual = bounds(raw).set_index(ID)
    dep = raw[raw.PHASE_mvt.eq('DEP')].set_index(ID)
    annual['month'] = annual.MVT_TIME_UTC_mvt.dt.month
    annual['y'] = dep.TAXITIME_SEC_mvt.reindex(annual.index).to_numpy(float)
    annual['placeholder'] = (dep.BLOCK_TIME_UTC_mvt-dep.SCHED_TIME_UTC_mvt).dt.total_seconds().abs().le(6).reindex(annual.index)
    eligible = annual[annual.stand_floor.notna()].copy()
    holds = {'definition':'strict y >= g1; tolerant y >= g1 - 60; no label enters the serving mask',
             'year_all':hold_table(eligible),'year_nonplaceholder':hold_table(eligible[~eligible.placeholder]),
             'other_eleven':{},'by_airport':{},'by_month':{}}
    for m in [1,7]:
        s = eligible[eligible.month.ne(m)]
        holds['other_eleven'][str(m)] = {'all':hold_table(s),'nonplaceholder':hold_table(s[~s.placeholder])}
    holds['other_ten'] = hold_table(eligible[~eligible.month.isin([1,7])])
    for a in AIRPORTS:
        holds['by_airport'][a] = hold_table(eligible[eligible.airport.eq(a)])
    for m in range(1,13):
        holds['by_month'][str(m)] = hold_table(eligible[eligible.month.eq(m)])
    holds['source_gate_pass'] = all(holds['other_eleven'][str(m)]['nonplaceholder'][k]['tolerant_rate'] >= .99
                                  for m in [1,7] for k in ['joined','no_plan'])
    holds['all_row_strict_gate_pass'] = all(holds['other_eleven'][str(m)]['all'][k]['strict_rate'] >= .99
                                         for m in [1,7] for k in ['joined','no_plan'])
    (EXP/'results/c55-holds-006.json').write_text(json.dumps(holds,indent=2)+'\n')
    print('HOLD',json.dumps(holds['other_eleven']),flush=True)
    assert holds['source_gate_pass'], 'Source hold gate failed; report to desk before admission.'
    assert holds['all_row_strict_gate_pass'], 'Strict hold gate failed; report to desk before admission.'

    # Exact unit-9 parity uses the January/July movement files together.
    held_raw = raw[raw.MVT_TIME_UTC_mvt.dt.month.isin([1,7])].copy()
    held = bounds(held_raw).set_index(ID)
    full = annual.reindex(held.index)
    a,z = held.stand_floor.to_numpy(),full.stand_floor.to_numpy()
    context_different = ~((a == z) | (np.isnan(a) & np.isnan(z)))
    assert not context_different.any(), 'Full-year context changes an eligible held floor; review source scope.'
    # Poison all departure target fields; legal arrival block times stay unchanged.
    poison = held_raw.copy()
    isdep = poison.PHASE_mvt.eq('DEP')
    poison.loc[isdep,'BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2040-01-01',tz='UTC')
    poison.loc[isdep,'TAXITIME_SEC_mvt'] = -123456.
    pd.testing.assert_frame_equal(bounds(poison).set_index(ID),held)
    b = pd.read_parquet(EXP/'inputs/unit2-ready/bundle-fold.parquet')
    c = pd.read_parquet(EXP/'inputs/unit2-ready/bundle-ranking.parquet')
    ranking_raw = pd.read_parquet(ROOT/'data/raw/ranking.parquet',columns=INPUT_COLUMNS)
    ranking = bounds(ranking_raw).set_index(ID)
    assert set(ranking.index) == set(c[ID]) and len(ranking) == len(c)
    for table,cache in [(b,held),(c,ranking)]:
        table['stand_floor'] = cache.stand_floor.reindex(table[ID]).to_numpy(float)
    base = compose(b,route='R4mk',noplan='v3cal',practice='selected')
    pred = compose(b,route='R4mk',noplan='v3cal',practice='selected',c55=True)
    base26 = compose(c,route='R4mk',noplan='v3cal',practice='selected')
    pred26 = compose(c,route='R4mk',noplan='v3cal',practice='selected',c55=True)
    approved = pd.read_parquet(EXP/'exp169-HB1x-noplan-v3cal-R4mk-practice-c33.parquet')
    assert np.array_equal(approved[ID],c[ID]) and np.array_equal(approved[COL],base26)
    np.testing.assert_array_equal(pred,np.fmax(base,b.stand_floor.to_numpy()))
    np.testing.assert_array_equal(pred26,np.fmax(base26,c.stand_floor.to_numpy()))
    y = GS.labels(b[ID].to_numpy('int64'))
    scores = {}
    for name,on in [('all',np.ones(len(b),bool)),('joined',b.joined.to_numpy(bool)),('no_plan',~b.joined.to_numpy(bool))]:
        p = np.where(on,pred,base)
        scores[name] = GS.score(b[ID].to_numpy(),base,p,y=y,require_complete=True)
        print('SCORE',name,scores[name]['G'],scores[name]['interval']['lo95'],scores[name]['interval']['hi95'],flush=True)
    v47 = compose(b,route='R4',form='HA3',extension=False,noplan='none',c33=False)
    scores['B_plus_C55_vs_v47'] = GS.score(b[ID].to_numpy(),v47,pred,y=y,require_complete=True)
    rows = GS.load_rows().set_index(ID).loc[b[ID].astype('int64')]
    cells = GS.load_cells().set_index(['month','airport','cls']).N
    count = rows.groupby(['month','airport','cls']).size()
    weight = np.array([cells.get((m,a,z),0)/GS.N_RANKING/count[m,a,z] if count[m,a,z]>=GS.MIN_SUPPORT else 0
                       for m,a,z in zip(rows.month,rows.airport,rows.cls)])
    gain = weight*((base-y)**2-(pred-y)**2)
    assert abs(gain.sum()-scores['all']['G']) < 1e-7
    top = int(np.argmax(gain))
    assert b.airport.iloc[top] == 'EGLL' and b.month.iloc[top] == 7
    without_top = pred.copy();without_top[top] = base[top]
    scores['without_top_EGLL_move'] = GS.score(b[ID].to_numpy(),base,without_top,y=y,require_complete=True)
    assert scores['without_top_EGLL_move']['G'] > 100
    assert round(scores['joined']['G']) == 63 and round(scores['no_plan']['G']) == 781
    expected = [(b,557,208,28,18,base,pred),(c,None,None,14,7,base26,pred26)]
    scope = {}
    for label,(table,nj,nn,mj,mn,p0,p1) in zip(['fold','ranking'],expected):
        on = table.stand_floor.notna().to_numpy();joined = table.joined.to_numpy(bool);moved = p1 != p0
        scope[label] = {'window_joined':int((on & joined).sum()),'window_no_plan':int((on & ~joined).sum()),
                        'moved_joined':int((moved & joined).sum()),'moved_no_plan':int((moved & ~joined).sum()),
                        'squared_distance':float(np.mean((p1-p0)**2))}
        assert scope[label]['moved_joined'] == mj and scope[label]['moved_no_plan'] == mn
        if label == 'fold':
            assert scope[label]['window_joined'] == nj and scope[label]['window_no_plan'] == nn
    print('SCOPE',json.dumps(scope),flush=True)
    OUT.mkdir(exist_ok=True)
    outputs = {}
    for label,table,cache,p0,p1,yy in [('fold',b,held,base,pred,y),('ranking',c,ranking,base26,pred26,None)]:
        path = OUT/('bundle-'+label+'.parquet');save_frame(table,path);outputs[str(path.relative_to(ROOT))] = sha(path)
        save_frame(cache.reset_index(),OUT/('bounds-'+label+'.parquet'))
        on = table.stand_floor.notna().to_numpy()
        audit = table.loc[on,[ID,'airport','month','joined','stand_floor']].copy()
        audit['base'] = p0[on];audit['after_floor'] = p1[on];audit['moved'] = audit.base != audit.after_floor
        for col in ['arrival_id','arrival_block','arrival_type','STAND_mvt','AIRCRAFT_TYPE_mvt']:
            audit[col] = cache[col].reindex(audit[ID]).to_numpy()
        if yy is not None:
            audit['label'] = yy[on];audit['gain'] = gain[on]
        path = OUT/('audit-'+label+'.parquet');save_frame(audit.reset_index(drop=True),path);outputs[str(path.relative_to(ROOT))] = sha(path)
    failures = eligible[eligible.y < eligible.stand_floor].reset_index()
    save_frame(failures,OUT/'hold-failures-2025.parquet')
    forecast = {'prior_frozen_B':230.18788755064497,'local_formula':float(np.sqrt(237.7071**2-1.24*scores['B_plus_C55_vs_v47']['G'])),
                'source_2026_G_scenario':[30.,80.],
                'scenario_formula_range':[float(np.sqrt(230.18788755064497**2-1.24*g)) for g in [80.,30.]],
                'conditional_floor':scope['ranking']['squared_distance'],
                'note':'Squared-distance floor is conditional on all moved rows satisfying the strict bound; observed exceptions forbid an unconditional guarantee.'}
    for p,h in inputs.items():assert sha(ROOT/p) == h,p
    for p,h in code.items():assert sha(ROOT/p) == h,p
    rec = {'source_rows':[899,901],'peer_source':'fresh c266bed0, C55 f4de1ce8','inputs_sha256':inputs,'code_sha256':code,
           'scores':scores,'scope':scope,'holds_path':'results/c55-holds-006.json','holds_sha256':sha(EXP/'results/c55-holds-006.json'),
           'top_EGLL':{'id':int(b[ID].iloc[top]),'gain':float(gain[top]),'base':float(base[top]),'bound':float(pred[top]),'label':float(y[top])},
           'full_year_context_changes':int(context_different.sum()),'departure_label_poison_exact':True,
           'forecast':forecast,'outputs_sha256':outputs,'new_upload_file_written':False,'review':'pending'}
    (EXP/'results/c55-006.json').write_text(json.dumps(rec,indent=2)+'\n')
    print('FORECAST',json.dumps(forecast),flush=True)


if __name__ == '__main__':
    main()
