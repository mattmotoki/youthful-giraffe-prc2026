"""Freeze coverage shrinkage, score development vectors, and make an inspection file.

The published held-month aggregate carries held-label information. All its 2025
scores are explicitly development diagnostics, even with leave-one-month-out coefficients.
"""
import argparse
import os
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[1]
SRC=Path(os.environ.get("PRC_SOURCE_ROOT", str(ROOT)))
sys.path.insert(0,str(ROOT))
from prc import gscore as GS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_offset(hist, airport, omit=None):
    h=hist[hist.airport.eq(airport)]
    if omit is not None:
        h=h[h.month.ne(omit)]
    X=np.column_stack([np.ones(len(h)),h.excluded_share.to_numpy()])
    y=h.offset_s.to_numpy()
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    variance=float(np.sum((y-X@b)**2)/(len(h)-2))
    return dict(coef=b.tolist(),residual_variance_s2=variance,xtx_inverse=np.linalg.inv(X.T@X).tolist(),
                n=len(h),months=h.month.astype(int).tolist())


def target(fit, x):
    X=np.array([1,x],float)
    return float(X@fit['coef']), float(fit['residual_variance_s2']*(1+X@np.asarray(fit['xtx_inverse'])@X))


def freeze():
    out=EXP/'results/step-d'
    out.mkdir(exist_ok=True)
    path=out/'frozen-parameters.json'
    assert not path.exists(), 'parameters already frozen'
    hist=pd.read_csv(EXP/'results/definition-2025.csv')
    future=pd.read_csv(EXP/'results/table-2026.csv')
    z=np.load(EXP/'results/held-development.npz',allow_pickle=False)
    rows=GS.load_rows().set_index('MVT_ID_mvt').loc[z['ids']].copy()
    rows['y'],rows['pred']=z['y'],z['v24']
    bias=rows.groupby(['airport','month']).agg(y=('y','mean'),pred=('pred','mean'))
    bias['bias']=bias.pred-bias.y
    params=[]
    for mode,cells in [('month_development',hist[hist.month.isin([1,7])]),('ranking_inspection',future)]:
        for cell in cells.itertuples():
            omit=cell.month if mode=='month_development' else None
            fit=fit_offset(hist,cell.airport,omit)
            # Zero-centred random cell bias prior; its variance is E[bias^2].
            # For a held month, no cell bias from that month enters this estimate.
            admitted=bias[bias.index.get_level_values('month')!=omit] if omit is not None else bias
            tau2=float(np.mean(np.square(admitted.bias)))
            x=1-cell.coverage
            predicted_offset, predictive_variance=target(fit,x)
            lam=tau2/(tau2+predictive_variance) if tau2+predictive_variance>0 else 0
            params.append(dict(mode=mode,airport=cell.airport,month=int(cell.month),coverage=cell.coverage,
                published_mean_s=cell.published_mean_s,predicted_offset_s=predicted_offset,
                corrected_mean_s=cell.published_mean_s-predicted_offset,predictive_variance_s2=predictive_variance,
                cell_bias_variance_s2=tau2,cell_bias_source_months=sorted(set(admitted.index.get_level_values('month'))),
                cell_bias_source_cells=len(admitted),lambda_value=lam,coverage_fit=fit))
    record=dict(status='frozen before step-d scores; inspection and development only',
        formula='lambda=tau2/(tau2+sigma2*(1+xvec.T@inverse(X.T@X)@xvec)); zero-centred cell-bias prior',
        tau2='unweighted mean squared v24 airport-month mean error; exclude the held month for development',
        target='published mean minus OLS predicted all-label offset at this coverage',
        validation_limit='Held-month aggregate contains held labels; LOO coefficients do not remove that path; historical v24 also has known held-label paths.',
        no_board_inputs=True,parameters=params,
        inputs={name:sha(EXP/'results'/name) for name in ['definition-2025.csv','table-2026.csv','held-development.npz']})
    path.write_text(json.dumps(record,indent=2)+'\n')
    (out/'frozen-parameters.sha256').write_text(sha(path)+'\n')
    pd.DataFrame([{k:v for k,v in p.items() if k!='coverage_fit'} for p in params]).to_csv(out/'frozen-cells.csv',index=False)
    print('frozen',sha(path),'lambda range',min(p['lambda_value'] for p in params),max(p['lambda_value'] for p in params))


def load_parameters():
    path=Path(os.environ.get('PRC_SHIFT_PARAMETERS', EXP/'frozen-parameters.json'))
    want='20a3ef9e2854fd81d325db6ffee8470c8fd132a197a673a2c4c2faa9d82297ad'
    assert sha(path)==want
    return json.loads(path.read_text()),want


def apply(rows,base,mode,parameters):
    out=base.copy()
    frozen=GS.load_rows()
    support=frozen[frozen.month.isin([1,7])].groupby(['month','airport','cls']).size()
    eligible=np.array([support.get((int(m),a,c),0)>=2 for m,a,c in zip(rows.month,rows.airport,rows.cls)])
    cells=[]
    for p in parameters:
        if p['mode']!=mode:
            continue
        keep=(rows.airport.to_numpy()==p['airport'])&(rows.month.to_numpy()==p['month'])
        moving=keep&eligible
        n=int(keep.sum());nm=int(moving.sum())
        assert n>0 and nm>0
        mean=float(base[keep].mean())
        wanted=mean+p['lambda_value']*(p['corrected_mean_s']-mean)
        # Keep the same mean change on admitted rows. Do not move the held
        # classes' share onto other classes (especially LEMD January).
        target_moving=float(base[moving].mean())+p['lambda_value']*(p['corrected_mean_s']-mean)
        target_moving=max(60,target_moving)
        lo=60-float(base[moving].max())-1;hi=target_moving+float(base[moving].max())+1
        for _ in range(80):
            mid=(lo+hi)/2
            if np.maximum(60,base[moving]+mid).mean()<target_moving:lo=mid
            else:hi=mid
        shift=(lo+hi)/2
        out[moving]=np.maximum(60,base[moving]+shift)
        cells.append(dict(airport=p['airport'],month=p['month'],rows=n,moving_rows=nm,held_rows=n-nm,
            base_mean_s=mean,proxy_mean_s=p['corrected_mean_s'],lambda_value=p['lambda_value'],
            wanted_mean_s=wanted,served_mean_s=float(out[keep].mean()),shift_s=shift,
            floor_rows=int(np.sum(out[moving]==60))))
    assert np.isfinite(out).all() and out.min()>=60
    assert np.array_equal(out[~eligible],base[~eligible])
    return out,cells


def score(base_path=None,pred_column=None,tag='v24-months'):
    record,pin=load_parameters()
    z=np.load(EXP/'results/held-development.npz',allow_pickle=False)
    ids,y,base=z['ids'],z['y'],z['v24']
    input_sha=sha(EXP/'results/held-development.npz')
    if base_path is not None:
        f=pd.read_parquet(base_path)
        assert pred_column and not f.MVT_ID_mvt.duplicated().any()
        base=f.set_index(f.MVT_ID_mvt.astype('int64'))[pred_column].reindex(ids).to_numpy(float)
        assert np.isfinite(base).all()
        input_sha=sha(base_path)
    rows=GS.load_rows().set_index('MVT_ID_mvt').loc[ids]
    cand,cells=apply(rows,base,'month_development',record['parameters'])
    result=dict(status='development only; published held-month aggregate contains held-label information',
        frozen_parameters_sha256=pin,baseline_sha256=input_sha,
        all=GS.score(ids,base,cand,y=y,groups={'pair':rows.pair.to_numpy()}),
        clean=GS.score(ids,base,cand,y=y,view='clean'),months={})
    for month in [1,7]:
        keep=rows.month.to_numpy()==month
        result['months'][str(month)]=GS.score(ids[keep],base[keep],cand[keep],y=y[keep],require_complete=False)
    out=EXP/'results/step-d'
    assert not (out/f'{tag}-score.json').exists()
    (out/f'{tag}-score.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(cells).to_csv(out/f'{tag}-cells.csv',index=False)
    pd.DataFrame({'MVT_ID_mvt':ids,'base':base,'shifted':cand}).to_parquet(out/f'{tag}-development.parquet',index=False)
    print(tag,'G',result['all']['G'],'interval',result['all']['interval'])


def serve():
    record,pin=load_parameters()
    path=SRC/'submissions/uploaded/youthful-giraffe_v32.parquet'
    assert sha(path)=='f09f609429c81fd888e19e75e16f51b57978d55cc2639d6161c894da0baaa0de'
    base=pd.read_parquet(path)
    meta=pd.read_parquet(EXP/'results/ranking-row-map.parquet').set_index('MVT_ID_mvt').loc[base.MVT_ID_mvt.astype('int64')]
    want=GS.load_cells().set_index(['month','airport','cls']).N
    counts=meta.groupby(['month','airport','cls']).size().reindex(want.index,fill_value=0)
    assert np.array_equal(counts.to_numpy(),want.to_numpy())
    pred,cells=apply(meta,base.TAXITIME_SEC_mvt.to_numpy(float),'ranking_inspection',record['parameters'])
    artifact=base.copy();artifact.TAXITIME_SEC_mvt=pred
    out=EXP/'results/step-d';dest=out/'v32-published-shift-INSPECTION-ONLY.parquet'
    assert not dest.exists()
    artifact.to_parquet(dest,index=False)
    pd.DataFrame(cells).to_csv(out/'v32-inspection-cells.csv',index=False)
    data=dict(status='INSPECTION ONLY; not a candidate; no upload allowed',rows=len(artifact),
        frozen_parameters_sha256=pin,comparator_sha256=sha(path),sha256=sha(dest),
        artifact=str(dest.relative_to(ROOT)),changed_rows=int(np.sum(pred!=base.TAXITIME_SEC_mvt.to_numpy())),
        rms_change_s=float(np.sqrt(np.mean((pred-base.TAXITIME_SEC_mvt.to_numpy())**2))),
        unsupported_ranking_rows_held=int(sum(c['held_rows'] for c in cells)),
        min_prediction_s=float(pred.min()),max_abs_change_s=float(np.max(np.abs(pred-base.TAXITIME_SEC_mvt.to_numpy()))))
    (out/'v32-inspection-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','score','serve'])
    parser.add_argument('--base-file',type=Path)
    parser.add_argument('--pred-column')
    parser.add_argument('--tag',default='v24-months')
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='score':score(args.base_file,args.pred_column,args.tag)
    else:serve()
