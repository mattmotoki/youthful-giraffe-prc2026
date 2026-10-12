"""Frozen035 six-column primary contract; existing assembly functions are unchanged."""
import copy
import numpy as np
import assemble_final_032 as A
from stack7 import independent_tail
from final_months169 import serve_months, assert_served
NAMES=['g','mlp','sequence021','sequence_new','dark','control']

def values(c,extra):
    assert np.asarray(extra).shape==(len(c['b']),)
    f=c['frame'];return np.column_stack([f.g,f.mlp,f.sequence021,f.sequence_new,extra,f.control])

def fit(c,extra,y,w):
    b=c['b'];on=c['scope'];scale=np.where(b.practice,1-b.practice_weight,1.)
    X=(values(c,extra)-c['base'][:,None])*scale[:,None];B=A.base_tail(c);folds={};pooled={}
    for cls in ['ordinary','clock']:
        names=NAMES if cls=='ordinary' else NAMES[:-1];folds[cls]={}
        for month in [1,7,None]:
            mask=on & b.cls.eq(cls).to_numpy()
            if month:mask &= b.month.eq(month).to_numpy()
            design=X[mask,:len(names)]*np.sqrt(w[mask,None]);target=(y-B)[mask]*np.sqrt(w[mask])
            assert np.isfinite(design).all() and np.isfinite(target).all()
            beta,_,rank,_=np.linalg.lstsq(design,target,rcond=None);assert rank==len(names)
            normal=design.T@design;rhs=design.T@target
            np.testing.assert_allclose(np.linalg.solve(normal,rhs),beta,rtol=0,atol=1e-10)
            rec={'method':'practice_scaled_wls','columns':names,'coefficients':beta.tolist(),
                 'trained_on_months':[month] if month else [1,7],'fit_rows':int(mask.sum()),
                 'xtwx':normal.tolist(),'xtwr':rhs.tolist(),'condition_number':float(np.linalg.cond(normal))}
            if month:folds[cls][str(8-month)]=rec
            else:pooled[cls]=rec
    return folds,pooled

def weights(c,folds,pooled,split):
    assert split in ['fold','ranking'];b=c['b'];out=np.zeros((len(b),6))
    for cls in ['ordinary','clock']:
        names=NAMES if cls=='ordinary' else NAMES[:-1]
        for month in serve_months(split):
            rec=folds[cls][str(month)] if split=='fold' else pooled[cls]
            assert rec['method']=='practice_scaled_wls' and rec['columns']==names
            assert rec['trained_on_months']==([8-month] if split=='fold' else [1,7]) and rec['fit_rows']>0
            coef=np.asarray(rec['coefficients'],float);normal=np.asarray(rec['xtwx']);rhs=np.asarray(rec['xtwr'])
            assert coef.shape==(len(names),) and np.isfinite(coef).all()
            assert normal.shape==(len(names),len(names)) and np.linalg.eigvalsh(normal).min()>0
            np.testing.assert_allclose(np.linalg.solve(normal,rhs),coef,rtol=0,atol=1e-10)
            on=c['scope'] & b.cls.eq(cls).to_numpy() & b.month.eq(month).to_numpy()
            out[on,:len(names)]=coef
    assert_served(c['scope'],b.cls,b.month,serve_months(split))
    return out

def apply_matrix(base,prior,v,w,scope,ordinary):
    assert v.shape==w.shape==(len(base),6) and np.isfinite(w).all()
    assert (w[~scope]==0).all() and (w[~ordinary,-1]==0).all()
    required=np.repeat(scope[:,None],6,axis=1);required[:,-1] &= ordinary
    assert np.isfinite(v[required]).all() and (v[required]>=60).all()
    delta=np.zeros(len(base))
    for j in range(6):
        on=scope & (w[:,j]!=0);delta[on]+=w[on,j]*(v[on,j]-base[on])
    result=prior.copy();result[scope]=np.maximum(60,base[scope]+delta[scope])
    np.testing.assert_array_equal(result[~scope],prior[~scope]);return result

def pre(c,extra,parent_weights,folds,pooled,split):
    baseline,_,_=A.pre_stack(c,parent_weights,split)
    matrix=values(c,extra);coef=weights(c,folds,pooled,split)
    primary=apply_matrix(c['base'],baseline['pair'],matrix,coef,c['scope'],c['ordinary'])
    out=copy.copy(baseline);out['primary']=primary;out['pre_exploit']=independent_tail(c['b'],primary)
    return out,coef

def zero_identity(c,extra,parent_weights,split):
    baseline,_,_=A.pre_stack(c,parent_weights,split);zero=np.zeros((len(c['b']),6))
    for cls in ['ordinary','clock']:
        for month in serve_months(split):
            rec=parent_weights['primary_fold'][cls][str(month)] if split=='fold' else parent_weights['primary_ranking'][cls]
            on=c['scope'] & c['b'].cls.eq(cls).to_numpy() & c['b'].month.eq(month).to_numpy()
            for name,coefficient in zip(rec['columns'],rec['coefficients']):zero[on,NAMES.index(name)]=coefficient
    assert_served(c['scope'],c['b'].cls,c['b'].month,serve_months(split))
    z=apply_matrix(c['base'],baseline['pair'],values(c,extra),zero,c['scope'],c['ordinary'])
    np.testing.assert_array_equal(z,baseline['primary'])
    np.testing.assert_array_equal(independent_tail(c['b'],z),baseline['pre_exploit'])
    return True
