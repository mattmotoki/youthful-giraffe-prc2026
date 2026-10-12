"""Bounded CPU preparation, independent checks and smoke; no GPU fit."""
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import hashlib

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--stage',choices=['prepare','check','smoke'],required=True);ap.add_argument('--attempt',default='001');a=ap.parse_args()
    if os.sched_getaffinity(0)!={4,5,6,7,68,69,70,71} or os.getpriority(os.PRIO_PROCESS,0)!=19:raise ValueError('Annual CPU lane/nice differs')
    resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
    os.environ.update(OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    code=Path(__file__).resolve().parent;exp=code.parent;root=a.work/'out/sequence-g-annual-inputs-001'
    checks=exp/'results/ANNUAL-CHECKS-001.json';smoke=a.work/'out/sequence-g-annual-smoke-001'
    plan=exp/'ANNUAL-PLAN.json';receipt=exp/f'results/annual-{a.stage}-receipt-{a.attempt}.json';log=exp/f'logs/annual-{a.stage}-{a.attempt}.log'
    if receipt.exists() or log.exists():raise ValueError('Annual CPU receipt/log exists')
    receipt.parent.mkdir(parents=True,exist_ok=True);log.parent.mkdir(parents=True,exist_ok=True)
    script={'prepare':'prepare_annual.py','check':'check_annual.py','smoke':'smoke_annual.py'}[a.stage]
    args={'prepare':['--plan',str(plan),'--out',str(root)],'check':['--inputs',str(root),'--out',str(checks)],
          'smoke':['--inputs',str(root),'--checks',str(checks),'--out',str(smoke),'--threads','6']}[a.stage]
    interpreter='/usr/bin/python3' if a.stage=='smoke' else sys.executable
    names=[script,'prepare.py',Path(__file__).name]
    if a.stage=='check':names+=['prepare_annual.py']
    if a.stage=='smoke':names+=['fit.py','model.py','annual_job.py']
    pins={n:sha(code/n) for n in set(names)};pins['ANNUAL-PLAN.json']=sha(plan)
    started=time.monotonic();rec={'status':'running','stage':a.stage,'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
        'affinity':sorted(os.sched_getaffinity(0)),'nice':19,'threads':6,'memory_cap_bytes':24*1024**3,'source_sha256':pins}
    try:
        with log.open('x') as stream:
            run=subprocess.run([interpreter,'-u',str(code/script),*args],stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
        rec['returncode']=run.returncode
        if run.returncode:raise RuntimeError(script+' failed; see log')
        for n,want in pins.items():
            if sha(plan if n=='ANNUAL-PLAN.json' else code/n)!=want:raise ValueError('CPU source changed: '+n)
        result={'prepare':root/'MANIFEST.json','check':checks,'smoke':smoke/'SMOKE.json'}[a.stage]
        rec.update(status='completed',result_path=str(result),result_sha256=sha(result))
    except Exception as ex:rec.update(status='failed',error=repr(ex))
    rec.update(seconds=time.monotonic()-started,log_sha256=sha(log));receipt.write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
    if rec['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
