"""Build-only correction: commit input hash records, not large Parquet blobs.

The fixed read code, forecast, source hashes and prediction digest stay exact.
This wrapper must itself be committed before it can write the candidate.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import stack10 as S


def main():
    S.sources();assert not S.CANDIDATE.exists()
    r=S.js(S.READ);f=S.js(S.FORECAST)
    assert r['gate']['pass'] and r['reads']['full_vs_stack8']['G']>0 and r['reads']['full_vs_stack8']['interval']['lo95']>-50
    assert f['read_sha256']==S.sha(S.READ) and f['spec_sha256']==S.SPEC_SHA and not f['candidate_exists']
    assert r['code_sha256']==S.code_pins()
    # These committed records bind every input's exact local bytes, including data
    # files that repository policy intentionally leaves outside Git.
    commit=S.S8.s7.committed(Path(__file__).resolve())
    for p in [S.SPEC,S.READ,S.FORECAST,S.MANIFEST,*[S.E/'code'/n for n in S.CODE]]:S.S8.s7.committed(p)
    for path,h in r['source_sha256'].items():
        p=S.R/path;assert S.sha(p)==h,path
        if 'stack10-025' in path and p.suffix!='.parquet':S.S8.s7.committed(p)
    b,n,m,a,t,old=S.load_route('ranking');pred=S.compose(n,blend=m,head_swap=t,**S.S8.CFG)
    S.scope_check(b,n,m,a,t,old,pred)
    assert S.vector_digest(b[S.ID],pred)==r['prediction_digests']['ranking']
    np.testing.assert_array_equal(b[S.ID],pd.read_parquet(S.R/'data/raw/submitting.parquet')[S.ID])
    q=pd.DataFrame({S.ID:b[S.ID].astype('float64'),S.COL:pred.astype('float64')})
    assert len(q)==344841 and q[S.ID].is_unique and np.isfinite(q[S.COL]).all() and q[S.COL].min()>=60
    q.to_parquet(S.CANDIDATE,index=False);pd.testing.assert_frame_equal(pd.read_parquet(S.CANDIDATE),q,check_exact=True)
    S.write(S.E/'results/stack10-build-025.json',{'candidate':S.record(S.CANDIDATE),'read':S.record(S.READ),'forecast':S.record(S.FORECAST),'manifest':S.record(S.MANIFEST),'commit_before_build':commit,'build_wrapper':S.record(Path(__file__).resolve()),'built_at_utc':S.now(),'review':'desk D123 pending'})
    print('BUILT',S.sha(S.CANDIDATE),flush=True)

if __name__=='__main__':main()
