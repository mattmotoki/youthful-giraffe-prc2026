import numpy as np, pandas as pd, lightgbm as lgb, prc.data as D
from noac_model import FEATS, CATS, PH_TOL, _X
ART=D.ROOT/"artifacts"/"exp-002-tail"
df=pd.concat([pd.read_parquet(ART/f"signals_{m}.parquet") for m in D.MONTHS],ignore_index=True)
na=df[(df.noac==1)&df.y.notna()].reset_index(drop=True)
o_all=na.mvt_sched.to_numpy(float)-na.y.to_numpy(float); ph_all=np.abs(o_all)<PH_TOL
NR,NN=344841,388
P=dict(objective='regression',learning_rate=0.04,num_leaves=8,n_estimators=300,min_child_samples=30,
       feature_fraction=0.6,bagging_fraction=0.8,bagging_freq=1,n_jobs=4,verbose=-1,seed=11)
C=dict(P); C['objective']='binary'
ys,pmix,poracle,pclf=[],[],[],[]
for m in D.MONTHS:
    te=na[na.month==m]; tr=na[na.month!=m]
    if not len(te): continue
    cats={c:pd.Index(sorted(set(tr[c].astype(str))|set(te[c].astype(str)))) for c in CATS}
    Xtr,Xte=_X(tr,cats),_X(te,cats)
    otr=tr.mvt_sched.to_numpy(float)-tr.y.to_numpy(float); phtr=np.abs(otr)<PH_TOL
    r=lgb.LGBMRegressor(**P).fit(Xtr,otr,categorical_feature=CATS)
    pmix.append(te.mvt_sched.to_numpy(float)-r.predict(Xte))
    rnp=lgb.LGBMRegressor(**P).fit(Xtr[~phtr],otr[~phtr],categorical_feature=CATS)
    ohat=rnp.predict(Xte)
    phte=np.abs(te.mvt_sched.to_numpy(float)-te.y.to_numpy(float))<PH_TOL
    poracle.append(np.where(phte, te.mvt_sched.to_numpy(float), te.mvt_sched.to_numpy(float)-ohat))
    c=lgb.LGBMClassifier(**C).fit(Xtr,phtr.astype(int),categorical_feature=CATS)
    pclf.append(c.predict_proba(Xte)[:,1])
    ys.append(te.y.to_numpy(float))
y=np.concatenate(ys)
for name,p in [('lgb-direct-offset',np.concatenate(pmix)),('ORACLE placeholder flag',np.concatenate(poracle))]:
    se=(y-p)**2; print(f'{name:26s} rmse={np.sqrt(se.mean()):7.0f}  ranking contrib={np.sqrt(NN*se.mean()/NR):5.0f} s')
p=np.concatenate(pclf); ph=np.concatenate([np.abs(na[na.month==m].mvt_sched.to_numpy(float)-na[na.month==m].y.to_numpy(float))<PH_TOL for m in D.MONTHS if (na.month==m).any()])
from sklearn.metrics import roc_auc_score, brier_score_loss
print(f'\nplaceholder classifier: AUC={roc_auc_score(ph,p):.3f} Brier={brier_score_loss(ph,p):.3f} base={ph.mean():.3f}')
# feature importance of the offset regressor fitted on all rows
cats={c:pd.Index(sorted(set(na[c].astype(str)))) for c in CATS}
X=_X(na,cats); r=lgb.LGBMRegressor(**P).fit(X,o_all,categorical_feature=CATS)
imp=pd.Series(r.feature_importances_,index=X.columns).sort_values(ascending=False)
print('\ntop features (offset regressor):'); print(imp.head(12).to_string())
