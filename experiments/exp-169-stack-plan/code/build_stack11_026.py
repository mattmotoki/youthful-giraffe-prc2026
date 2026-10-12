"""Build the frozen candidate only after an explicit parent candidate-review PASS."""
from pathlib import Path
import subprocess
import numpy as np,pandas as pd
import stack11 as S
from verdict026 import require_pass

def main():
    assert not S.CANDIDATE.exists()
    review=S.PARENT_REVIEW
    verdicts=require_pass(review)
    parent_status='PASS'
    assert S.PARENT_SHA in review.read_text()
    r=S.js(S.READ);f=S.js(S.FORECAST);m=S.js(S.MANIFEST)
    assert r['gates']['pass'] and S.sha(S.PARENT)==S.PARENT_SHA
    assert S.sha(S.SPEC)==r['spec_sha256']==f['spec_sha256']==m['spec_sha256']
    assert S.sha(S.READ)==f['read_sha256'] and S.sha(S.MANIFEST)==r['manifest_sha256']==f['manifest_sha256']
    for p,h in m['source_sha256'].items():assert S.sha(S.ROOT/p)==h,p
    for n,h in r['code_sha256'].items():assert S.sha(S.E/'code'/n)==h,n
    for p in [S.SPEC,S.READ,S.FORECAST,S.MANIFEST,*[S.E/'code'/n for n in S.CODE]]:S.P.S8.s7.committed(p)
    q,parent,kw=S.context('ranking');rule=S.load_rule('ranking',parent)
    out=S.compose11(q,rule,**kw)['stack11'];assert S.vector_digest(q[S.ID],out)==r['prediction_digests']['ranking']
    ids=pd.read_parquet(S.ROOT/'data/raw/submitting.parquet')[S.ID]
    np.testing.assert_array_equal(ids,q[S.ID]);assert len(ids)==344841 and ids.is_unique
    result=pd.DataFrame({S.ID:ids.to_numpy('float64'),S.COL:out.astype('float64')})
    assert np.isfinite(result.to_numpy()).all() and result[S.COL].min()>=60
    S.old_unchanged();S.parquet(result,S.CANDIDATE)
    pd.testing.assert_frame_equal(pd.read_parquet(S.CANDIDATE),result,check_exact=True)
    S.write(S.O/'stack11-build-026.json',{'candidate':S.record(S.CANDIDATE),'parent_review':{'path':str(review.relative_to(S.ROOT)),'sha256_at_build':S.sha(review),'status_at_build':parent_status,'verdict_lines_at_build':verdicts},'authority':S.record(S.E/'BUILD-AUTHORITY-026.md'),'authority_order':'desk2026-10-03 00:19UTC: build now; desk D123 after stack10 review ends','build_wrapper':S.record(Path(__file__).resolve()),'spec':S.record(S.SPEC),'read':S.record(S.READ),'forecast':S.record(S.FORECAST),'manifest':S.record(S.MANIFEST),'built_at_utc':S.now(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.ROOT,text=True).strip(),'review':'desk D123 pending','upload':False})
    print('BUILT',S.sha(S.CANDIDATE),flush=True)

if __name__=='__main__':main()
