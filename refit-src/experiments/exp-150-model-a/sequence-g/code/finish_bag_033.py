"""Wait for both frozen bags, audit all models, then one bounded registered read."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from stage_bag_stack6_033 import sha

W=Path(os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp150-work'))
E=W/'source/experiments/exp-150-model-a/sequence-g'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    lock=(E/'bag-finish-033.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=E/'results/bag-finish-033.json'
    if path.exists():raise ValueError('Finish receipt exists')
    rev=json.loads(a.review.read_text())
    if rev['verdict']!='PASS':raise ValueError('Finish source not reviewed')
    pins=rev['finish_source_sha256'];review_sha=sha(a.review);read_review=E/'REVIEW-BAG-READ-SOURCE-033.json'
    audit=E/'results/BAG-MODEL-AUDIT-033.json';out=W/'out/sequence-g-bag-read-033'
    if audit.exists() or out.exists():raise ValueError('Audit/read already exists')
    r=dict(status='waiting',PID=os.getpid(),started_UTC=datetime.now(timezone.utc).isoformat(),
           review_sha256=review_sha,source_sha256=pins,audit=str(audit),out=str(out))
    def write():path.write_text(json.dumps(r,indent=2)+'\n')
    def check(active=False):
        if sha(a.review)!=review_sha:raise ValueError('Finish review changed')
        for n,h in pins.items():
            if sha(E/'code'/n)!=h:raise ValueError('Reviewed finish source changed: '+n)
        if sha(read_review)!=rev['read_review_sha256']:raise ValueError('Read review changed')
        # Scheduling pause preserves a running remote job. The read-specific
        # cancel flag also stops the current owned child, through cleanup below.
        if (E/'STOP-BAG-READ-033').exists() or (not active and (E/'STOP-BAG-033').exists()):
            raise InterruptedError('Desk stop file present')
    write();child=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
             PRC_DATA_DIR=os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp148-src/data/raw'),
             PRC_VALIDATION_DIR=str(W/'source/experiments/exp-146-validation/frozen'))
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise InterruptedError('Finish supervisor stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        deadline=time.monotonic()+72*3600
        while True:
            check();lanes={k:json.loads((E/f'results/bag-lane-{k}-033.json').read_text()) for k in ['a','b']}
            if any(v['status'] not in ['running','completed'] for v in lanes.values()):raise RuntimeError('A fit lane did not complete')
            r['fit_counts']={k:len(v['finished']) for k,v in lanes.items()};r['checked_UTC']=datetime.now(timezone.utc).isoformat();write()
            if all(v['status']=='completed' for v in lanes.values()):break
            if time.monotonic()>deadline:raise TimeoutError('Fit wait exceeded72h')
            time.sleep(30)
        prefix=['taskset','-c','4-7,68-71','nice','-n','19']
        commands=[('auditing',prefix+['/usr/bin/python3','-u',str(E/'code/audit_bag_models_033.py'),
            '--work',str(W),'--out',str(audit),'--review',str(a.review)],1200),
            ('reading',prefix+[str(W/'venv/bin/python'),'-u',str(E/'code/read_bag_033.py'),
            '--fit-root',str(W/'out'),'--stage',str(W/'out/sequence-g-bag-stack6-stage-033'),
            '--out',str(out),'--review',str(read_review)],3600)]
        for status,cmd,limit in commands:
            check()
            if status=='reading' and (not audit.exists() or json.loads(audit.read_text())['status']!='PASS'):
                raise ValueError('Full model audit did not pass')
            log=E/f'logs/bag-finish-{status}-033.log'
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                r.update(status=status,child_PID=child.pid,command=cmd,log=str(log));write()
                end=time.monotonic()+limit
                while True:
                    check(active=True)
                    left=end-time.monotonic()
                    if left<=0:raise TimeoutError(status+' timed out')
                    try:rc=child.wait(timeout=min(5,left));break
                    except subprocess.TimeoutExpired:pass
            if rc:raise RuntimeError(status+f' failed exit{rc}: {log}')
            r[status+'_log_sha256']=sha(log);child=None
        read=out/'READ.json';result=json.loads(read.read_text())
        if result['status']!='completed':raise ValueError('Read did not complete')
        r.update(status='completed',audit_sha256=sha(audit),read_sha256=sha(read),choice='Desk decides after result review')
    except BaseException as ex:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        r.update(status='paused' if isinstance(ex,InterruptedError) else 'failed',error=repr(ex))
    r['ended_UTC']=datetime.now(timezone.utc).isoformat();write();print(json.dumps(r),flush=True)
    if r['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
