import numpy as np, pandas as pd, lightgbm as lgb, prc.data as D
from noac_model import FEATS, CATS, PH_TOL, _X
ART=D.ROOT/"artifacts"/"exp-002-tail"
df=pd.concat([pd.read_parquet(ART/f"signals_{m}.parquet") for m in D.MONTHS],ignore_index=True)
na=df[(df.noac==1)&df.y.notna()].reset_index(drop=True)
NR,NN=344841,388
def lomo(make):
    ys,ps=[],[]
    for m in D.MONTHS:
        te=na[na.month==m]; tr=na[na.month!=m]
        if not len(te): continue
        ps.append(make(tr,te)); ys.append(te.y.to_numpy(float))
    y=np.concatenate(ys); p=np.concatenate(ps); se=(y-p)**2
    return np.sqrt(se.mean()), np.sqrt(NN*se.mean()/NR)
def mk(nl,ne,lr,mcs,ff,seed=11,extra_p=False):
    def f(tr,te):
        cats={c:pd.Index(sorted(set(tr[c].astype(str))|set(te[c].astype(str)))) for c in CATS}
        Xtr,Xte=_X(tr,cats),_X(te,cats)
        o=tr.mvt_sched.to_numpy(float)-tr.y.to_numpy(float)
        if extra_p:
            ph=(np.abs(o)<PH_TOL).astype(int)
            c=lgb.LGBMClassifier(objective='binary',learning_rate=lr,num_leaves=nl,n_estimators=ne,
                min_child_samples=mcs,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,n_jobs=4,verbose=-1,seed=seed).fit(Xtr,ph,categorical_feature=CATS)
            Xtr=Xtr.assign(_p=c.predict_proba(Xtr)[:,1]); Xte=Xte.assign(_p=c.predict_proba(Xte)[:,1])
        r=lgb.LGBMRegressor(objective='regression',learning_rate=lr,num_leaves=nl,n_estimators=ne,
            min_child_samples=mcs,feature_fraction=ff,bagging_fraction=0.8,bagging_freq=1,n_jobs=4,verbose=-1,seed=seed).fit(Xtr,o,categorical_feature=CATS)
        return te.mvt_sched.to_numpy(float)-r.predict(Xte)
    return f
def mkseedavg(g,seeds=(11,23,37,51,77)):
    def f(tr,te): return np.mean([mk(*g,seed=s)(tr,te) for s in seeds],axis=0)
    return f
grid=[(8,300,0.04,20,0.7),(8,600,0.03,20,0.7),(15,400,0.04,15,0.7),(31,400,0.04,10,0.8),
      (8,300,0.04,30,0.6),(4,600,0.05,20,0.8),(15,800,0.02,15,0.7)]
best=None
for g in grid:
    r,c=lomo(mk(*g)); print(f'nl={g[0]} ne={g[1]} lr={g[2]} mcs={g[3]} ff={g[4]}: LOMO rmse={r:.0f} contrib={c:.0f}',flush=True)
    if best is None or r<best[0]: best=(r,g)
print('\nbest', best)
r,c=lomo(mkseedavg(best[1])); print(f'5-seed average of best: rmse={r:.0f} contrib={c:.0f}')
r,c=lomo(mk(*best[1],extra_p=True)); print(f'best + placeholder-prob feature: rmse={r:.0f} contrib={c:.0f}')
