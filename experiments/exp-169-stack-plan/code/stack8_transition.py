"""Checked stack7-to-stack8 primary member substitution, no route or gate change."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from head_swap import checked,fingerprint
from diverse_blend import vector_digest
from lane_c_swap import make_transition,load_lane_c_swap,NAMES
from review_verdict import require_pass
ID='MVT_ID_mvt'
PARENT='89f3c83cd1d689f1cf3a193e3c17a0e7e64119ddc1c76b48e7631fba33d1f381'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())


def sequence_receipt(pack,key,split,vector,receipt):
    fit=js(receipt);want=list(range(1,13)) if split=='ranking' else [2,3,4,5,6,8,9,10,11,12]
    assert fit['train_months']==want
    typ='annual' if split=='ranking' else 'held';seq_record=pack['records']['sequence']
    assert sha(receipt)==seq_record[typ+'_fit_sha256']
    assert sha(vector)==seq_record['annual_raw_vector_sha256' if split=='ranking' else 'held_vector_sha256']
    expected=[(128,24,s) for s in range(17001,17013)]
    expected += [(256,24,s) for s in range(17001,17013 if key=='037' else 17007)]
    if key=='037':expected += [(128,48,s) for s in range(17001,17007)]
    assert key in ('033b','037') and fit['n_fits']==len(expected)==len(fit['parents'])
    assert pack['sequence_weights']=='last iterate'
    for parent,(width,epochs,seed) in zip(fit['parents'],expected):
        a=parent['args'];assert (a['width'],a['epochs'],a['seed'],a['layers'])==(width,epochs,seed,3)
        assert bool(a['final'])==(split=='ranking')
        assert parent['weights_form']=='last iterate' and parent['weight']==1/len(expected)
    assert [(x['width'],x['epochs'],x['seed']) for x in pack['bag_fits']]==expected
    return {'fits':len(expected),'months':want,'receipt_sha256':sha(receipt),'vector_sha256':sha(vector)}


def load_stack8(path,expected_sha256,bundle,member,head,split):
    path=Path(path);assert sha(path)==expected_sha256
    r=js(path);assert r['schema']==3 and r['form']=='stack8'
    assert r['parent']['sha256']==PARENT;checked(r['parent']);require_pass(checked(r['parent_review']))
    spec=checked(r['spec']);approval=js(checked(r['desk_approval']))
    assert approval['approved'] and approval['spec_sha256']==sha(spec) and approval['parent_sha256']==PARENT
    old=load_lane_c_swap(checked(r['parent_transition']),r['parent_transition']['sha256'],bundle,member,head,split)
    q=r['splits'][split];assert fingerprint(old)==q['parent_transition_fingerprint']
    pd.testing.assert_frame_equal(bundle,pd.read_parquet(checked(q['bundle'])),check_exact=True)
    assert vector_digest(member.ids,head.new_base)==q['base_digest']
    pack=js(checked(r['pack']));review=checked(r['source_review']);require_pass(review)
    assert r['pack']['sha256'] in review.read_text() and pack['vector_sha256'] in review.read_text()
    require_pass(checked(r['mlp_review']));mlp=js(checked(r['mlp_pack']))
    assert r['mlp_pack']['sha256']=='5ffc057af6fd5ccde5ba479420817b494e4fbbc09ee394641b8d77830b557de0'
    assert all(mlp['vectors'][k]['sha256'][:8] in checked(r['mlp_review']).read_text() for k in ['fold','ranking'])
    assert mlp['member_training_months']['fold']==[2,3,4,5,6,8,9,10,11,12]
    assert mlp['member_training_months']['ranking_final']==list(range(1,13))
    f=pd.read_parquet(checked(q['matrix']))
    for name in [ID,'month','cls']:np.testing.assert_array_equal(f[name],bundle[name])
    np.testing.assert_array_equal(f.member_scope,member.scope)
    for name,j in [('g',0),('sequence021',2),('control',4)]:np.testing.assert_array_equal(f[name],old.values[:,j])
    mv=pd.read_parquet(checked(q['mlp_vector']));assert q['mlp_vector']['sha256']==mlp['vectors'][split]['sha256']
    for name in [ID,'month','cls']:np.testing.assert_array_equal(mv[name],bundle[name])
    use=member.scope & ~member.ordinary;assert mv.mlp_scope.to_numpy(bool)[use].all()
    expected=old.values[:,1].copy();expected[use]=mv.pred_mlp164.to_numpy(float)[use]
    np.testing.assert_array_equal(f.mlp,expected)
    sq=pd.read_parquet(checked(q['sequence_new'])).set_index(ID);assert sq.index.is_unique
    np.testing.assert_array_equal(f.sequence_new,sq.pred_seq170.reindex(bundle[ID]).to_numpy(float))
    sequence_receipt(pack,r['source_key'],split,checked(q['sequence_new']),checked(q['sequence_receipt']))
    t=make_transition(old,member,f,r['fold_weights'],r['ranking_weights'],split)
    assert t.matrix_digest==q['matrix_digest']
    return t
