"""SPEC-025: checked R4mks substitution; all earlier assembly code stays fixed."""
from dataclasses import dataclass,replace
from pathlib import Path
import numpy as np
import pandas as pd
from assembly import ID,head_blend,clip
from head_swap import fingerprint
from diverse_blend import vector_digest
from verdict026 import parse as parse_verdicts, require_pass as require_review_pass
from final_months169 import serve_months, assert_served


def verdicts(text):
    return [row['verdict'] for row in parse_verdicts(text)]


def require_pass(path):
    return [row['verdict'] for row in require_review_pass(path)]


def fit_pair(b,parent,base,B,y,w):
    scale=np.where(b.practice,1-b.practice_weight,1.)
    X=np.column_stack([parent.sequence-base,parent.control-base])*scale[:,None]
    folds={};pool={}
    for cls in ['ordinary','clock']:
        folds[cls]={}
        for mon in [1,7,None]:
            q=parent.scope & b.cls.eq(cls).to_numpy()
            if mon:q &= b.month.eq(mon).to_numpy()
            months=[mon] if mon else [1,7]
            if cls=='ordinary':
                A=X[q]*np.sqrt(w[q,None]);r=(y-B)[q]*np.sqrt(w[q])
                beta,_,rank,_=np.linalg.lstsq(A,r,rcond=None);assert rank==2
                normal=A.T@A;rhs=A.T@r
                np.testing.assert_allclose(np.linalg.solve(normal,rhs),beta,rtol=0,atol=1e-10)
                rec=dict(method='practice_scaled_wls',sequence=float(beta[0]),control=float(beta[1]),xtwx=normal.tolist(),xtwr=rhs.tolist(),condition_number=float(np.linalg.cond(normal)))
            else:
                grid=np.arange(51)*.02;losses=[];practice=b.practice.to_numpy(bool)
                for v in grid:
                    delta=v*X[:,0];raw=np.divide(delta,scale,out=np.zeros_like(delta),where=scale>0)
                    z=np.maximum(60,base+raw)
                    z[practice]=np.maximum(60,z[practice]+b.practice_weight.to_numpy()[practice]*(b.practice_point.to_numpy()[practice]-z[practice]))
                    z=clip(z,b.lo.to_numpy(),b.hi.to_numpy())
                    losses.append(float(w[q]@((y[q]-z[q])**2)))
                rec=dict(method='served_grid',sequence=float(grid[np.argmin(losses)]),control=0.,grid=grid.tolist(),losses=losses)
            rec.update(trained_on_months=months,fit_rows=int(q.sum()))
            if mon:folds[cls][str(8-mon)]=rec
            else:pool[cls]=rec
    return folds,pool


def pair_weights(rec,cls,months):
    assert rec['trained_on_months']==months and rec['fit_rows']>0
    values=np.array([rec['sequence'],rec['control']],float);assert np.isfinite(values).all()
    if cls=='ordinary':
        assert rec['method']=='practice_scaled_wls'
        A=np.asarray(rec['xtwx']);r=np.asarray(rec['xtwr'])
        assert A.shape==(2,2) and np.linalg.eigvalsh(A).min()>0
        np.testing.assert_allclose(A,A.T,rtol=0,atol=1e-10)
        np.testing.assert_allclose(np.linalg.solve(A,r),values,rtol=0,atol=1e-10)
    else:
        assert rec['method']=='served_grid' and values[1]==0
        grid=np.asarray(rec['grid']);losses=np.asarray(rec['losses'])
        np.testing.assert_array_equal(grid,np.arange(51)*.02)
        assert losses.shape==(51,) and np.isfinite(losses).all() and (losses>=0).all()
        assert values[0]==grid[np.argmin(losses)]
    return values


def rebase_pair(pair,b,base,folds=None,pool=None,split='fold'):
    ws=pair.sequence_weight.copy();wc=pair.control_weight.copy()
    if folds is not None:
        ws[:]=0;wc[:]=0
        for cls in ['ordinary','clock']:
            for mon in [1,7]:
                held=pair_weights(folds[cls][str(mon)],cls,[8-mon]);annual=pair_weights(pool[cls],cls,[1,7])
                q=pair.scope & b.cls.eq(cls).to_numpy() & b.month.eq(mon).to_numpy()
                ws[q],wc[q]=held if split=='fold' else annual
            for mon in [m for m in serve_months(split) if m not in (1,7)]:
                q=pair.scope & b.cls.eq(cls).to_numpy() & b.month.eq(mon).to_numpy()
                ws[q],wc[q]=pair_weights(pool[cls],cls,[1,7])
        assert_served(pair.scope,b.cls,b.month,serve_months(split))
    return replace(pair,base_digest=vector_digest(pair.ids,base),sequence_weight=ws,control_weight=wc)


@dataclass(frozen=True)
class RouteSwap:
    old_bundle: pd.DataFrame
    new_bundle: pd.DataFrame
    old_transition: object
    head: object
    pair: object
    parent_fingerprint: str
    old_base: np.ndarray
    new_base: np.ndarray

    def bases(self,bundle,route):
        pd.testing.assert_frame_equal(bundle,self.new_bundle,check_exact=True)
        pd.testing.assert_frame_equal(bundle.drop(columns='R4mk'),self.old_bundle.drop(columns='R4mk'),check_exact=True)
        np.testing.assert_array_equal(route,bundle.R4mk)
        np.testing.assert_array_equal(self.old_transition.bases(self.old_bundle,self.old_bundle.R4mk.to_numpy()),self.old_base)
        expected=head_blend(route,self.head.head,self.head.scope,.5)
        np.testing.assert_array_equal(expected,self.new_base)
        return expected

    def blend(self,parent,ids,base,enable_extension):
        assert fingerprint(parent)==self.parent_fingerprint,'Original members changed.'
        np.testing.assert_array_equal(ids,parent.ids)
        np.testing.assert_array_equal(base,self.new_base)
        # Replay every original parent contract before the authorized route rebase.
        self.old_transition.blend(parent,ids,self.old_base,enable_extension)
        for name in ['ids','sequence','control','scope','control_scope','extension']:
            np.testing.assert_array_equal(getattr(self.pair,name),getattr(parent.fallback,name))
        assert self.pair.extension_proxy_gain==parent.fallback.extension_proxy_gain
        return self.pair.apply(ids,base,enable_extension=enable_extension)


def make_route(b,route,t,h,m,split,folds=None,pool=None):
    assert route.shape==(len(b),) and np.isfinite(route).all() and (route>=60).all()
    new=b.copy();new['R4mk']=route
    base=head_blend(route,h.head,h.scope,.5)
    pair=rebase_pair(m.fallback,new,base,folds,pool,split)
    return new,RouteSwap(b,new,t,h,pair,fingerprint(m),h.new_base,base)
