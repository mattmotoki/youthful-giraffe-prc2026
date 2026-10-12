"""M31 joint MLP/sequence/control blend; preserve the reviewed pair extension."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diverse_blend import vector_digest
from pair_blend import PairedBlend

ID='MVT_ID_mvt'
NAMES=['mlp','sequence','control']
G_NAMES=['g',*NAMES]


@dataclass(frozen=True)
class JointBlend:
    ids: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    scope: np.ndarray
    ordinary: np.ndarray
    base_digest: str
    fallback: PairedBlend

    def apply(self,ids,base,*,enable_extension=False):
        assert isinstance(enable_extension,bool)
        assert np.array_equal(ids,self.ids),'Joint arrays belong to another ID order.'
        assert np.isfinite(base).all() and (base>=60).all()
        assert vector_digest(ids,base)==self.base_digest,'Joint weights belong to another base.'
        n=len(base)
        assert self.values.shape==self.weights.shape
        assert self.values.ndim==2 and self.values.shape[0]==n
        width=self.values.shape[1]
        assert width in (3,4)
        assert self.scope.shape==self.ordinary.shape==(n,)
        assert self.scope.dtype==self.ordinary.dtype==np.dtype(bool)
        assert np.isfinite(self.weights).all()
        assert (self.weights[~self.scope]==0).all()
        assert (self.weights[~self.ordinary,-1]==0).all(),'Clock rows cannot use the control.'
        ext=self.fallback.extension
        assert ext is not None
        assert np.array_equal(self.scope,self.fallback.scope & ~ext),'Joint scope differs from admitted primary scope.'
        required=np.repeat(self.scope[:,None],width,axis=1)
        required[:,-1]&=self.ordinary
        assert np.isfinite(self.values[required]).all() and (self.values[required]>=60).all()
        # Both transforms refer to the same pre-member base, never to each other.
        out=self.fallback.apply(ids,base,enable_extension=enable_extension)
        before=out.copy()
        delta=np.zeros(n)
        for col in range(width):
            active=self.scope & (self.weights[:,col]!=0)
            delta[active]+=self.weights[active,col]*(self.values[active,col]-base[active])
        out[self.scope]=np.maximum(60.,base[self.scope]+delta[self.scope])
        np.testing.assert_array_equal(out[~self.scope],before[~self.scope])
        assert np.isfinite(out).all() and (out>=60).all()
        return out


def _path(parent,record):
    path=Path(record['path'])
    if not path.is_absolute():path=parent/path
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],'Joint input changed.'
    return path


def _weights(record,cls,months,names_all=NAMES):
    assert record['trained_on_months']==months
    assert record['method']=='practice_scaled_wls'
    names=names_all if cls=='ordinary' else names_all[:-1]
    assert record['columns']==names and record['fit_rows']>0
    matrix=np.asarray(record['xtwx'],float);rhs=np.asarray(record['xtwr'],float)
    beta=np.asarray(record['coefficients'],float);n=len(names)
    assert matrix.shape==(n,n) and rhs.shape==beta.shape==(n,)
    assert np.isfinite(matrix).all() and np.isfinite(rhs).all() and np.isfinite(beta).all()
    assert np.allclose(matrix,matrix.T,rtol=0,atol=1e-10)
    assert np.linalg.eigvalsh(matrix).min()>0
    np.testing.assert_allclose(np.linalg.solve(matrix,rhs),beta,rtol=0,atol=1e-10)
    return np.r_[beta,0.] if cls=='clock' else beta


def load_joint(path,spec,bundle,split):
    from diverse_blend import load_member
    assert (spec['schema'],spec['lane']) in [(4,'B+C'),(5,'B+C+G')]
    names=NAMES if spec['schema']==4 else G_NAMES
    assert split in ['fold','ranking']
    assert spec['weight_basis']=='two_fold_2025' and spec['serving_point']=='after_head'
    assert spec['member_training_months']=={'fold':[2,3,4,5,6,8,9,10,11,12],'ranking':list(range(1,13))}
    parent=Path(path).parent
    fallback_path=_path(parent,spec['fallback'])
    assert json.loads(fallback_path.read_text())['schema']==3,'Joint fallback must be the reviewed pair.'
    fallback=load_member(fallback_path,spec['fallback']['sha256'],bundle,split)
    assert isinstance(fallback,PairedBlend)
    rec=spec['vectors'][split]
    frame=pd.read_parquet(_path(parent,rec))
    assert frame[ID].is_unique and len(frame)==len(bundle) and set(frame[ID])==set(bundle[ID])
    assert np.isfinite(frame[ID]).all() and np.equal(frame[ID],frame[ID].astype('int64')).all()
    frame=frame.set_index(ID).loc[bundle[ID]]
    for name in ['month','cls']:np.testing.assert_array_equal(frame[name],bundle[name])
    assert frame.member_scope.dtype==np.dtype(bool)
    on=frame.member_scope.to_numpy(bool)
    assert np.array_equal(on,fallback.scope & ~fallback.extension)
    assert not (on & bundle.cls.eq('noac').to_numpy()).any()
    ordinary=bundle.cls.eq('ordinary').to_numpy()
    weights=np.zeros((len(bundle),len(names)))
    assert set(spec['fold_weights'])==set(spec['ranking_weights'])=={'ordinary','clock'}
    for cls in ['ordinary','clock']:
        assert set(spec['fold_weights'][cls])=={'1','7'}
        for m,other in [(1,7),(7,1)]:
            held=_weights(spec['fold_weights'][cls][str(m)],cls,[other],names)
            annual=_weights(spec['ranking_weights'][cls],cls,[1,7],names)
            use=on & bundle.cls.eq(cls).to_numpy() & bundle.month.eq(m).to_numpy()
            weights[use]=held if split=='fold' else annual
    assert rec['base_digest']==fallback.base_digest
    return JointBlend(bundle[ID].to_numpy('int64').copy(),frame[names].to_numpy(float),weights,on,
                      ordinary,rec['base_digest'],fallback)
