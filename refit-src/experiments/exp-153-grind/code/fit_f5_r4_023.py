"""Unchanged F5 learner on one R4 split, or one held-model jjrank export."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PRC_THREADS'):os.environ[k]='4'
import numpy as np
import pandas as pd
import learner_wp as LW
import r4_pilot as P
from fit_jjrank import frame_hash

FITS=('jjx-12','jjx-02','jjx-06','jjx-08','jjx-03','jjx-10','jjx-11','jjrank')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fit',choices=FITS,required=True)
 for k in ('inputs','input-root','cache','out'):ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();root=a.inputs;r=json.loads((root/'F5-RECIPE.json').read_text())
 for name,key in [('source-f5-janjul.json','source_record_sha256'),('f5-state.parquet','raw_sha256'),('f5-state-inputs.json','input_receipt_sha256'),('source-f5-held.parquet','held_sha256'),('reference-ids.parquet','ids_sha256')]:
  if LW.sha(root/name)!=r[key]:raise ValueError('F5 bound input differs: '+name)
 ref=json.loads((root/'source-f5-janjul.json').read_text());params=ref['fit']['params'];columns=ref['fit']['f5_columns'];inputrec=json.loads((root/'f5-state-inputs.json').read_text())
 if params['num_threads']!=4 or ref['fit']['rounds']!=1500 or inputrec['columns']!=columns:raise ValueError('F5 settings differ')
 if a.out.exists():raise ValueError('F5 output exists; no overwrite')
 a.out.mkdir(parents=True)
 # Pin the same data helper that built the measured F5 member.
 wanted=ref['code']['data.py'];candidates=[LW.ROOT/'prc/data.py',*a.input_root.rglob('data.py')]
 dp=next((p for p in candidates if p.exists() and LW.sha(p)==wanted),None)
 if dp is None:raise ValueError('measured F5 data helper absent')
 spec=importlib.util.spec_from_file_location('prc.data',dp);mod=importlib.util.module_from_spec(spec);sys.modules['prc.data']=mod;spec.loader.exec_module(mod)
 import prc
 prc.data=mod
 env=LW.setup(a.input_root);base,D=env[2],env[6]
 if env[7]!=ref['code']:raise ValueError('F5 upstream learner code changed')
 raw,audit=LW.raw_months_cached(env[0],a.cache)
 for key in ('source_hashes','cache_key'):
  if audit.get(key)!=ref['proxy_audit'].get(key):raise ValueError('F5 proxy source differs: '+key)
 frames={m:base.month_frame(m) for m in LW.MONTHS}
 inputs={f'month_{m}.parquet':LW.sha(base.locate_file(f'month_{m}.parquet')) for m in LW.MONTHS}
 if inputs!=ref['inputs']:raise ValueError('F5 native inputs changed')
 split=P.janjul() if a.fit=='jjrank' else P.split(a.fit)
 if a.fit!='jjrank':
  contract=next(x for x in r['records'] if x['fit']==a.fit)
  if split.manifest()!=contract['split']:raise ValueError('F5 split differs from L control')
 train=pd.concat(list(frames.values()),ignore_index=True);train=train.loc[D.keep_reachable(train)];train=train.loc[split.label_mask(LW.wp_frame(train))].reset_index(drop=True)
 split.assert_training_frame(LW.wp_frame(train),target=train.y.to_numpy(),what='F5 R4')
 months=list(LW.SCORED_MONTHS) if a.fit=='jjrank' else sorted({m for m in P.scored_months(split)})
 scoring=[frames[m].loc[split.score_mask(LW.wp_frame(frames[m]))].drop(columns=['y']) for m in months]
 if a.fit=='jjrank':
  scoring.extend([base.month_frame(m).drop(columns=['y'],errors='ignore') for m in LW.RANK_MONTHS]);months+=list(LW.RANK_MONTHS)
  inputs.update({f'month_{m}.parquet':LW.sha(base.locate_file(f'month_{m}.parquet')) for m in LW.RANK_MONTHS})
 tr=train.loc[train.ghost.to_numpy(float)<.5].reset_index(drop=True);sc=pd.concat([f.loc[f.ghost.to_numpy(float)<.5] for f in scoring],ignore_index=True)
 if np.isin(sc[LW.ID],train[LW.ID]).any():raise ValueError('F5 train/scored ID overlap')
 state=pd.read_parquet(root/'f5-state.parquet').set_index(LW.ID,drop=False)
 if not state.index.is_unique or not np.r_[tr[LW.ID],sc[LW.ID]].size:raise ValueError('F5 state IDs invalid')
 aligned=[]
 for part in (tr,sc):
  if not part[LW.ID].isin(state.index).all():raise ValueError('F5 state coverage missing')
  v=state.reindex(part[LW.ID]).reset_index(drop=True)
  if not np.array_equal(v.month,part.month) or not np.array_equal(v.airport,part.airport):raise ValueError('F5 state context differs')
  aligned.append(v[columns].reset_index(drop=True))
 captured={};original=base.fit_arm
 def fit(name,xt,xs,residual,weights,cats,anchor,model,fold):
  import lightgbm as lgb
  if lgb.__version__!=ref['fit']['lightgbm_version']:raise ValueError('measured F5 LightGBM version differs')
  if len(xt)!=len(aligned[0]) or len(xs)!=len(aligned[1]):raise ValueError('F5 feature rows differ')
  xt=pd.concat([xt.reset_index(drop=True),aligned[0]],axis=1);xs=pd.concat([xs.reset_index(drop=True),aligned[1]],axis=1)
  if xt.columns.duplicated().any() or list(xt)!=list(xs):raise ValueError('F5 feature schema differs')
  featurehash=hashlib.sha256(json.dumps(list(xt)).encode()).hexdigest()
  if featurehash!=ref['fit']['feature_list_sha256']:raise ValueError('F5 fixed feature list differs')
  rec={'params':params,'rounds':1500,'train_rows':len(xt),'features':list(xt),'categorical':cats,'train_feature_sha256':frame_hash(xt),'train_target_sha256':hashlib.sha256(np.asarray(residual).tobytes()).hexdigest(),'train_split':split.manifest(),'fit_code_sha256':LW.sha(Path(__file__))}
  (a.out/'before-fit.json').write_text(json.dumps(rec,indent=2)+'\n');print('F5 fit start',a.fit,len(xt),len(xs),flush=True)
  t=time.monotonic();booster=lgb.train(params,lgb.Dataset(xt,label=residual,weight=weights,categorical_feature=cats),num_boost_round=1500);booster.save_model(str(a.out/'model.txt'))
  pred=np.maximum(30.,anchor+booster.predict(xs,num_threads=4))
  if not np.isfinite(pred).all():raise ValueError('F5 prediction nonfinite')
  captured.update(rec,fit_seconds=time.monotonic()-t,model_sha256=LW.sha(a.out/'model.txt'),lightgbm_version=lgb.__version__)
  return pred,{'info':captured}
 base.fit_arm=fit
 try:pred,info=LW.fit_learner(env,raw,train,scoring,months,'F5-'+a.fit,'L')
 finally:base.fit_arm=original
 pred=pred.rename(columns={'pred_L':'pred_Lf5'});target_ids=pd.read_parquet(root/'reference-ids.parquet');ids=target_ids.loc[target_ids.fit.eq(a.fit),LW.ID].to_numpy('int64')
 part=pred.loc[pred.month.str.startswith('2026')] if a.fit=='jjrank' else pred
 if not part[LW.ID].is_unique or set(part[LW.ID])!=set(ids):raise ValueError('F5 exact native L ID set differs')
 part=part.set_index(LW.ID).loc[ids].reset_index()
 record={'stage_what':'features2 frozen F5 on R4 L','fit_name':a.fit,'train_split':split.manifest(),'train_rows':len(train),'fit':captured,'inputs':inputs,'code':env[7],'code_learner_wp':LW.sha(Path(LW.__file__)),'code_pilot':LW.sha(Path(P.__file__)),'weekpair_sha256':LW.sha(Path(LW.WP.__file__)),'f5_recipe_sha256':LW.sha(root/'F5-RECIPE.json'),'f5_input_sha256':r['raw_sha256'],'source_f5_record_sha256':r['source_record_sha256'],'scored_rows':len(part),'proxy_source_audit':{k:v for k,v in audit.items() if k not in ('build_seconds','cache','matches_by_month')}}
 if a.fit=='jjrank':
  held=pred.loc[~pred.month.str.startswith('2026')].copy();old=pd.read_parquet(root/'source-f5-held.parquet').set_index(LW.ID)
  if set(held[LW.ID])!=set(old.index):raise ValueError('F5 held reproduction IDs differ')
  diff=held.pred_Lf5.to_numpy()-old.loc[held[LW.ID],'pred_L153-f5'].to_numpy();rms=float(np.sqrt(np.mean(diff**2)))
  held.to_parquet(a.out/'held-replay.parquet',index=False);record.update(held_rms=rms,held_max_abs=float(np.max(np.abs(diff))),held_replay_sha256=LW.sha(a.out/'held-replay.parquet'),same_fit_both_domains=True)
  if not np.isfinite(rms) or rms>r['max_held_rms']:
   (a.out/'failure.json').write_text(json.dumps(record,indent=2)+'\n');raise ValueError('F5 held reproduction failed')
  days=tuple(sorted(part.utc_day.astype(str).unique()))
  artifact_split=LW.WP.Split('jjrank',days,tuple(sorted(set(split.excluded)|set(days))));owner=('jjrank')
 else:artifact_split=split;owner=(split.name)
 path=P.write(a.out,'Lf5',artifact_split,owner,part,record);print('F5 ready',path,flush=True)
if __name__=='__main__':main()
