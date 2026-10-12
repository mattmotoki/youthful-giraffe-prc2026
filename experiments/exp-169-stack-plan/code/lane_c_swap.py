"""SPEC-022 primary lane C and optional clock MLP transition; stack6 stays fixed."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from diverse_blend import load_member,vector_digest
from head_swap import load_head_swap,fingerprint,checked
from joint_blend import _weights
from review_verdict import require_pass
from final_months169 import serve_months, assert_served

ROOT=Path(__file__).resolve().parents[3];EXP=Path(__file__).resolve().parents[1]
ID='MVT_ID_mvt'
PACK='733c00ab8fc7f03ba4156b6f724bcb6ed0e3ee6d38a9c532a073e8bfaee39c7d'
PARENT='46ee2f909cf8db81eee1ff7af06d58fda2fdf9df32128ccfb819f0491b3e9a70'
NAMES=['g','mlp','sequence021','sequence_new','control']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())


def array_digest(names,values,weights,scope):
    h=hashlib.sha256(json.dumps(names).encode())
    for a in [values,weights,scope]:h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def matrix_apply(base,prior,values,weights,scope,ordinary):
    assert values.shape==weights.shape==(len(base),values.shape[1]) and values.shape[1] in (4,5)
    assert np.isfinite(weights).all() and (weights[~scope]==0).all()
    assert (weights[~ordinary,-1]==0).all(),'Control serves clock rows.'
    required=np.repeat(scope[:,None],values.shape[1],axis=1);required[:,-1]&=ordinary
    assert np.isfinite(values[required]).all() and (values[required]>=60).all()
    delta=np.zeros(len(base))
    for k in range(values.shape[1]):
        on=scope&(weights[:,k]!=0)
        delta[on]+=weights[on,k]*(values[on,k]-base[on])
    out=prior.copy();out[scope]=np.maximum(60.,base[scope]+delta[scope])
    np.testing.assert_array_equal(out[~scope],prior[~scope])
    return out


@dataclass(frozen=True)
class LaneCTransition:
    head_swap: object
    names: list
    values: np.ndarray
    weights: np.ndarray
    scope: np.ndarray
    ordinary: np.ndarray
    matrix_digest: str

    def bases(self,bundle,route):return self.head_swap.bases(bundle,route)

    def blend(self,parent,ids,base,enable_extension):
        # Runs the unchanged original HB1-fit checks and the authorized HB3u rebase.
        prior=self.head_swap.blend(parent,ids,base,enable_extension)
        assert np.array_equal(self.scope,parent.scope)
        assert np.array_equal(self.ordinary,parent.ordinary)
        assert array_digest(self.names,self.values,self.weights,self.scope)==self.matrix_digest,'Lane C arrays changed.'
        return matrix_apply(base,prior,self.values,self.weights,self.scope,self.ordinary)


def make_transition(head_swap,parent,frame,fold_weights,ranking_weights,split):
    names=NAMES;weights=np.zeros((len(frame),len(names)));on=frame.member_scope.to_numpy(bool)
    np.testing.assert_array_equal(frame[ID].to_numpy('int64'),parent.ids)
    np.testing.assert_array_equal(on,parent.scope)
    for cls in ['ordinary','clock']:
        for m,other in [(1,7),(7,1)]:
            held=_weights(fold_weights[cls][str(m)],cls,[other],names)
            annual=_weights(ranking_weights[cls],cls,[1,7],names)
            use=on&frame.cls.eq(cls).to_numpy()&frame.month.eq(m).to_numpy()
            weights[use]=held if split=='fold' else annual
        for m in [m for m in serve_months(split) if m not in (1,7)]:
            use=on&frame.cls.eq(cls).to_numpy()&frame.month.eq(m).to_numpy()
            weights[use]=_weights(ranking_weights[cls],cls,[1,7],names)
    assert_served(on,frame.cls,frame.month,serve_months(split))
    values=frame[names].to_numpy(float)
    return LaneCTransition(head_swap,names,values,weights,on,parent.ordinary,array_digest(names,values,weights,on))


def check_sequence(pack,split,vector,receipt):
    fit=js(receipt);want=list(range(1,13)) if split=='ranking' else [2,3,4,5,6,8,9,10,11,12]
    assert fit['train_months']==want
    seq_record=pack['records']['sequence'];key='annual' if split=='ranking' else 'held'
    assert sha(receipt)==seq_record[key+'_fit_sha256']
    assert sha(vector)==seq_record['annual_raw_vector_sha256' if split=='ranking' else 'held_vector_sha256']
    assert pack['sequence_weights']=='last iterate'
    mixed=fit.get('n_fits')==18
    expected=[(128,s) for s in range(17001,17013)]+([(256,s) for s in range(17001,17007)] if mixed else [])
    assert len(fit['parents'])==len(expected)
    for parent,(width,seed) in zip(fit['parents'],expected):
        r=parent if mixed else parent['fit'];a=r['args']
        assert (a['width'],a['layers'],a['epochs'],a['seed'])==(width,3,24,seed)
        assert parent['weight']==1/len(expected)
        assert (r['weights_form'] if mixed else r['weights'])=='last iterate'
        if not mixed:assert r['train_months']==want
    return {'fits':len(expected),'train_months':want,'receipt_sha256':sha(receipt),'vector_sha256':sha(vector)}


def load_lane_c_swap(path,expected_sha256,bundle,parent,head_swap,split):
    path=Path(path);assert sha(path)==expected_sha256,'Lane C manifest changed.'
    rec=js(path);assert rec['schema']==2 and rec['form']=='J'
    assert rec['weight_policy']=='other_month_practice_scaled_wls_then_pooled_2026'
    assert rec['parent']['sha256']==PARENT;checked(rec['parent'])
    pack=js(checked(rec['pack']));review=checked(rec['source_review']).read_text()
    require_pass(checked(rec['source_review']))
    assert rec['pack']['sha256'][:8] in review and pack['vector_sha256'][:8] in review
    require_pass(checked(rec['parent_review']))
    spec=checked(rec['spec']);approval=js(checked(rec['desk_approval']))
    assert approval['approved'] and approval['spec_sha256']==sha(spec)
    assert approval['pack_sha256']==rec['pack']['sha256'] and approval['parent_sha256']==PARENT
    checked(rec['head_swap']);assert rec['head_swap']['sha256']==sha(EXP/'inputs/stack6-021/head-swap.json')
    r=rec['splits'][split];assert fingerprint(parent)==r['parent_member_fingerprint']
    pd.testing.assert_frame_equal(bundle,pd.read_parquet(checked(r['bundle'])),check_exact=True)
    assert vector_digest(parent.ids,head_swap.new_base)==r['base_digest']
    f=pd.read_parquet(checked(r['matrix']));assert f[ID].is_unique and len(f)==len(bundle)
    for name in [ID,'month','cls']:np.testing.assert_array_equal(f[name],bundle[name])
    np.testing.assert_array_equal(f.member_scope,parent.scope)
    for name,j in [('g',0),('sequence021',2),('control',3)]:np.testing.assert_array_equal(f[name],parent.values[:,j])
    mlp=parent.values[:,1].copy()
    if rec['clock_mlp_e80']:
        mlp_member=js(checked(rec['mlp_pack']));mr=checked(rec['mlp_review']).read_text()
        require_pass(checked(rec['mlp_review']))
        assert rec['mlp_pack']['sha256'][:8] in mr
        v=pd.read_parquet(checked(r['mlp_vector']));assert r['mlp_vector']['sha256']==mlp_member['vectors'][split]['sha256']
        for key in [ID,'month','cls']:np.testing.assert_array_equal(v[key],bundle[key])
        use=parent.scope & ~parent.ordinary;assert v.mlp_scope.to_numpy(bool)[use].all()
        mlp[use]=v.pred_mlp164.to_numpy(float)[use]
    np.testing.assert_array_equal(f.mlp,mlp)
    seq=pd.read_parquet(checked(r['sequence_new'])).set_index(ID);assert seq.index.is_unique
    np.testing.assert_array_equal(f.sequence_new,seq.pred_seq170.reindex(bundle[ID]).to_numpy(float))
    check_sequence(pack,split,checked(r['sequence_new']),checked(r['sequence_receipt']))
    transition=make_transition(head_swap,parent,f,rec['fold_weights'],rec['ranking_weights'],split)
    assert transition.matrix_digest==r['matrix_digest']
    return transition
