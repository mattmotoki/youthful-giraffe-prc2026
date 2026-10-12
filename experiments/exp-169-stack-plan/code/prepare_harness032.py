"""Create the three parity manifests from existing reviewed local sources."""
from copy import deepcopy
from pathlib import Path
import sys
import pandas as pd
import assemble_final_032 as H
import stack10 as P

E=H.E;ROOT=H.ROOT;D=H.INPUTS

def vector(path,column): return {**H.record(path),'column':column,'id_column':H.ID}

def main():
    assert not H.REFERENCE.exists();D.mkdir(parents=True,exist_ok=True)
    pair=H.js(E/'inputs/stack3-aobt-011/pair.json')
    pairframes={s:(E/'inputs/stack3-aobt-011'/pair['vectors'][s]['path']).resolve() for s in ['fold','ranking']}
    contexts={s:P.context(s) for s in ['fold','ranking']}
    for s,(b,m,h,t,z,f) in contexts.items():
        q=pd.read_parquet(pairframes[s]);H.np.testing.assert_array_equal(q.member_scope,m.fallback.scope)
        H.np.testing.assert_array_equal(q.control_scope,m.fallback.control_scope)
        H.np.testing.assert_array_equal(q.pred_seq170,m.fallback.sequence)
        H.np.testing.assert_array_equal(q.pred_row170,m.fallback.control)
    exposure=D/'fold-exposure.parquet';b=contexts['fold'][0]
    H.table(exposure,pd.DataFrame({H.ID:b[H.ID],'weight':P.weights(b)}))
    ref={'schema':1,'recipe':'stack12_other_month_then_pooled_029_no_bets','common':{},'vectors':{},
         'baseline_inputs':{},'evidence':[],'protected':{}}
    c=ref['common'];c['bundle']={s:H.record(E/f'inputs/stack10-025/bundle-{s}.parquet') for s in contexts}
    c['primary_frame']={s:H.record(E/f'inputs/stack12-028/matrix-{s}.parquet') for s in contexts}
    c['pair_frame']={s:H.record(p) for s,p in pairframes.items()}
    c['rule_frame']={'fold':H.record(E/'inputs/stack9-024/frame.parquet'),
                     'ranking':H.record(E/'inputs/stack11-026/frame-ranking.parquet')}
    c['exploit_frame']={s:H.record(E/f'inputs/exploit-029/rules-{s}.parquet') for s in contexts}
    c['pair_weights']=H.record(E/'inputs/stack10-025/route-swap.json')
    c['primary_weights']=H.record(E/'inputs/stack12-028/MANIFEST.json')
    c['exposure']=H.record(exposure);c['template']=H.record(ROOT/'data/raw/submitting.parquet')
    proxy=pair['extension_proxy'];c['extension_proxy']=H.record(E/'inputs/stack3-aobt-011'/proxy['path'])
    c['extension_gain_paths']=[proxy['gain_path'],*proxy['required_positive_paths']]
    c['head_policy']=H.record(E/'inputs/stack6-021/head-swap.json')
    c['exploit_audit']=H.record(E/'inputs/exploit-029/AUDIT.json')
    c['C71_hold']=H.record(E/'results/stack9-c71-hold-024.json')
    names=['assembly.py','stand_floor.py','diverse_blend.py','pair_blend.py','joint_blend.py',
           'route_swap.py','lane_c_swap.py','stack7.py','exploit_rules.py','exploit029_rules.py']
    c['recipe_code']={name:H.record(E/'code'/name) for name in names}
    c['comparisons']={name:H.record(E/file) for name,file in {
        '22b71023':'exp169-stack12-exploit-bounds-029.parquet',
        'e6dfedd5':'exp169-stack12-exploit-final-R4mks-030.parquet',
        '8703a7d5':'exp169-stack12-exploit-time57-minus365-031.parquet'}.items()}
    for name,value in zip(['fold','final','shift'],c['comparisons'].values()): assert value['sha256']==H.EXPECTED[name]
    for s in contexts:
        frame=E/f'inputs/stack12-028/matrix-{s}.parquet'
        v={role:vector(frame,col) for role,col in H.ROLES.items()}
        v['sequence011']=vector(pairframes[s],'pred_seq170');v['row011']=vector(pairframes[s],'pred_row170')
        v['route']=vector(E/f'inputs/stack10-025/bundle-{s}.parquet','R4mk')
        v['head']=vector(E/f'inputs/stack6-021/HB3u-{"jjrefit" if s=="fold" else "jjrank"}.parquet','pred_HB3u')
        ref['vectors'][s]=v
    ref['baseline_inputs']=deepcopy(ref['vectors'])
    ref['protected']=H.js(E/'results/time-shift-before-031.json')['files']
    for folder in [E/'inputs/time-shift-031']:
        for p in folder.rglob('*'):
            if p.is_file():ref['protected'][str(p.relative_to(ROOT))]=H.sha(p)
    for p in [E/'SPEC-031.md',E/'HANDOFF-019.md',E/'REPORT-031.md',E/'REVIEW-PROMPT-time-shift-031.md',
              *[E/'code'/n for n in ['time_shift031.py','time_shift_sources031.py','check_time_shift031.py']],
              *list((E/'results').glob('time-shift-*-031.json'))]:
        ref['protected'][str(p.relative_to(ROOT))]=H.sha(p)
    for rec in c['comparisons'].values():ref['protected'][rec['path']]=rec['sha256']
    H.write(H.REFERENCE,ref)
    for name in ['fold','final','shift']:
        m=deepcopy(ref);m.pop('protected');m['reference']=H.record(H.REFERENCE)
        m['description']='Existing '+name+' parity; development only'
        if name=='final':m['vectors']['ranking']['route']=vector(E/'inputs/final-route-030/ranking-final-R4mks-B50.parquet',H.COL)
        if name=='shift':
            source=H.js(E/'inputs/time-shift-031/SOURCE.json')['proof']
            mapping={'sequence039':'sequence','row6':'control','sequence021':'sequence021','sequence011':'sequence011','row011':'control011'}
            for role,key in mapping.items():
                r=source[key]['vector'];m['vectors']['ranking'][role]={**r,'column':'pred_seq170','id_column':H.ID}
            m['evidence']=[H.record(E/'inputs/time-shift-031/SOURCE.json')]
        H.write(D/f'parity-{name}.json',m)
    print('Created reference and fold/final/shift parity manifests',flush=True)

if __name__=='__main__':main()
