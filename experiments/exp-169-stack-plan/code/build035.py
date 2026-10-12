"""Final035 build, gated by the held read, frozen forecast and actual064 review.

The build manifest is written after intake064. It binds parent_manifest,
extra (fold/ranking), annual (delivery/audit/review/mean_fit/parent_fits),
held_read, forecast, spec and code records. No source is chosen here.
"""
import argparse,copy,io,re,os
from pathlib import Path
from datetime import datetime
from math import sqrt
import numpy as np
import pandas as pd
import assemble_final_032 as A
import dark_twin035 as T
from verdict026 import parse,require_pass
E=A.E;R=A.ROOT;P=R/'experiments/exp-181-modela-bag-read'

def annual_gate(m,rd):
    annual=m['annual'];delivery=A.js(rd.path(annual['delivery']));audit=A.js(rd.path(annual['audit']))
    assert delivery['status']==audit['status']=='PASS'
    assert not audit.get('failures',[]) and audit.get('n_failures',0)==0
    assert audit['checks'] and all(v['ok'] for v in audit['checks'].values())
    assert len(audit['checks'])==27 and audit['receipt']['sha256']==annual['delivery']['sha256']
    assert not delivery['errors'] and delivery['annual'] and not delivery['masks_requested']
    assert not delivery['quality_read'] and not delivery['pack_admission']
    for rec in annual['evidence']:rd.path(rec)
    text=rd.path(annual['review']).read_text()
    text=re.sub(r'(?m)^## Output checks:', 'Verdict:',text)
    text=re.sub(r'(?m)^#{0,6}\s*Part 064:', 'Verdict:',text)
    text=re.sub(r'(?m)^## Part [12]:.*?,\s*(PASS|FAIL|FAILED)\b.*$',r'Verdict: \1',text)
    verdicts=parse(text);assert verdicts and all(v['verdict']=='PASS' for v in verdicts)
    fit=A.js(rd.path(annual['mean_fit']));vector=m['extra']['ranking']
    assert fit['n_seeds']==18 and fit['train_months']==list(range(1,13))
    assert fit['files']['ranking2026.parquet']==vector['sha256']
    assert audit['mean']['FIT_sha256']==annual['mean_fit']['sha256']==delivery['mean']['FIT_sha256']
    assert audit['mean']['vector_sha256']==vector['sha256']==delivery['mean']['vector_sha256']
    assert vector['sha256'] in __import__('json').dumps(audit), 'Annual vector not bound by audit.'
    assert len(fit['parents'])==18 and len(annual['parent_fits'])==18
    assert fit['parent_inputs_manifest_sha256']==[delivery['input_manifest_sha256']]
    assert fit['inputs_manifest_sha256']==delivery['input_manifest_sha256']
    for parent,rec,seed in zip(fit['parents'],annual['parent_fits'],range(17001,17019)):
        q=A.js(rd.path(rec));assert rec['sha256']==parent['fit_sha256']
        args=q['args'];assert q['weights']=='last iterate' and q['train_months']==list(range(1,13))
        assert parent['seed']==args['seed']==seed and parent['weight']==1/18
        assert args['final'] is True and (args['width'],args['layers'],args['epochs'])==(128,3,48)
        assert q['inputs_manifest_sha256']==delivery['input_manifest_sha256']
        assert q==parent['fit']
        assert q['files']['ranking2026.parquet']==parent['vector_sha256']
        assert q['files']['model.pt']==parent['model_sha256']
    return {'review':annual['review'],'audit':annual['audit'],'checks':len(audit['checks']),'verdicts':verdicts}

def run(path,out):
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:assert int(os.environ.get(key,'1'))<=2
    out=Path(out).resolve();assert out.is_relative_to(E) and not out.exists()
    rd=A.Reader();m=A.js(rd.path(A.record(path)));started=A.now()
    assert m['schema']==1 and m['form']=='dark18_further_member_035' and m['primary_columns']==T.NAMES
    read=A.js(rd.path(m['held_read']));forecast=A.js(rd.path(m['forecast']));rd.path(m['spec'])
    assert read['gate']['pass'] and read['reads']['dark18_vs_parent']['G']>0
    assert forecast['read']==m['held_read'] and not forecast['candidate_exists']
    assert forecast['spec']==m['spec']==read['spec']
    assert forecast['weights']==read['weights'] and forecast['statistics']==read['statistics']
    for key in ['parent_forecast','parent_spec','code']:rd.path(forecast[key])
    parent_forecast=A.js(rd.path(forecast['parent_forecast']))['forecast']
    step=read['reads']['dark18_vs_parent']
    for name,ratio in [('frozen124',1.24),('ratio083',0.83)]:
        q=forecast['forecast'][name];p=parent_forecast[name]['central']
        assert q['parent']==p and q['ratio']==ratio and q['step_G']==step['G']
        assert q['central']==sqrt(p*p-ratio*step['G'])
        assert q['range_at_G_95']==[sqrt(p*p-ratio*step['interval'][key]) for key in ['hi95','lo95']]
    assert datetime.fromisoformat(forecast['frozen_at_utc'])<datetime.fromisoformat(started)
    for rec in m['code'].values():rd.path(rec)
    assert m['code']['dark_twin035.py']==A.record(T.__file__)
    assert m['code']['build035.py']==A.record(__file__)
    assert m['code']['assemble_final_032.py']==A.record(A.__file__)
    frozen_manifest=A.js(rd.path(read['manifest']))
    assert m['parent_manifest']==frozen_manifest['parent_final'] and m['spec']==read['spec']
    intake=A.js(rd.path(frozen_manifest['intake']));assert m['extra']['fold']==intake['vectors']['dark18']
    for rec in intake['evidence']:rd.path(rec)
    gate=annual_gate(m,rd)
    require_pass(rd.path(A.record(R/'docs/status/2026-10-03-exp181-ab-final-review/review-ab-final.log')))
    parent,ref=A.validate_manifest(rd.path(m['parent_manifest']),rd)
    assert parent['vectors']['fold']==frozen_manifest['parent_held_vectors']
    contexts=A.load(parent,rd);pw=A.fit_weights(parent,contexts,rd,{'refit':True})
    assert pw==A.js(rd.path(A.record(P/'results/ab-final-build/weights.json')))
    old={}
    for split,c in contexts.items():old[split],_,_=A.pre_stack(c,pw,split)
    A.rules(parent,contexts,old,rd);A.tail(parent,contexts,old,rd)
    _,oldbody=A.serialize(contexts['ranking'],old['ranking']['final'],rd,parent['common']['template'])
    assert __import__('hashlib').sha256(oldbody).hexdigest()=='9436b85b0c5b2c83dd4b6b1be16076166337e7c274a3d4e6a59481843fd9cf9c'
    assert A.vector_digest(contexts['fold']['b'][A.ID],old['fold']['final'])==read['parent_digest']
    extras={split:rd.aligned(m['extra'][split],c['b'],c['scope'],year=2025 if split=='fold' else 2026) for split,c in contexts.items()}
    frame=rd.frame(parent['common']['rule_frame']['fold']);exposure=rd.frame(parent['common']['exposure'])
    fw,pp=T.fit(contexts['fold'],extras['fold'],frame.y.to_numpy(),exposure.weight.to_numpy())
    frozen_weights=A.js(rd.path(read['weights']))['dark18']
    assert fw==frozen_weights['primary_fold'] and pp==frozen_weights['primary_ranking']
    assert pw['pair_fold']==frozen_weights['pair_fold'] and pw['pair_ranking']==frozen_weights['pair_ranking']
    stages={};coef={}
    for split,c in contexts.items():
        assert T.zero_identity(c,extras[split],pw,split)
        stages[split],coef[split]=T.pre(c,extras[split],pw,fw,pp,split)
    stats,_=A.rules(parent,contexts,stages,rd);A.tail(parent,contexts,stages,rd)
    assert stats['fold']==A.js(rd.path(read['statistics']))['dark18']['fold']
    assert A.vector_digest(contexts['fold']['b'][A.ID],stages['fold']['final'])==read['checks']['dark18']['final_digest']
    q,body=A.serialize(contexts['ranking'],stages['ranking']['final'],rd,parent['common']['template'])
    comparisons={'9436b85b':A.distance(contexts['ranking']['b'],old['ranking']['final'],stages['ranking']['final'])}
    for name,rec in parent['common']['comparisons'].items():
        comparison=rd.frame(rec);np.testing.assert_array_equal(comparison[A.ID],q[A.ID])
        comparisons[name]=A.distance(contexts['ranking']['b'],comparison[A.COL],q[A.COL])
    rd.unchanged();out.mkdir(parents=True)
    (out/'candidate.parquet').write_bytes(body)
    A.write(out/'weights.json',frozen_weights);A.write(out/'statistics.json',stats)
    A.table(out/'weights-held.parquet',pd.DataFrame({A.ID:contexts['fold']['b'][A.ID],**{name:coef['fold'][:,j] for j,name in enumerate(T.NAMES)}}))
    receipt={'candidate':A.record(out/'candidate.parquet'),'manifest':A.record(path),'forecast':m['forecast'],
        'held_read':m['held_read'],'held_prediction_digest':A.vector_digest(contexts['fold']['b'][A.ID],stages['fold']['final']),
        'parent_byte_parity9436b85b':True,'parent_held_exact':True,'zero_member_identity_exact':True,
        'weights_exact_to_read':True,'held_statistics_exact_to_read':True,'source_annual_gate':gate,
        'comparisons':comparisons,'input_sha256':rd.inputs,'annual_label_used':False,
        'started_at_utc':started,'ended_at_utc':A.now(),'upload_allowed':False,'D123':'desk pending'}
    A.write(out/'RECEIPT.json',receipt);print('READY',receipt['candidate']['sha256'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.manifest,a.out)
