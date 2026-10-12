from __future__ import annotations
import hashlib, json, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
BINDING=EXP/'exact-binding.json'; STARTED=EXP/'started.json'; EXIT=EXP/'exit.json'; LOCK=EXP/'.launch-lock'; LOG=EXP/'run.log'; PROGRAM=EXP/'score.py'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def atomic_json(path,obj):
 data=(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n').encode(); fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path); d=os.open(path.parent,os.O_RDONLY); os.fsync(d); os.close(d)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def validate_binding():
 b=json.loads(BINDING.read_text()); expected={n:EXP/n for n in ['SPEC.md','source-manifest.json','score.py','test_score.py','controller.py','test_controller.py']}
 if set(b)!=set(expected): raise RuntimeError('binding file list differs')
 for n,p in expected.items():
  if sha(p)!=b[n]: raise RuntimeError('registered hash differs: '+n)
 return b
def claim_launch(lock=LOCK,started=STARTED,exitp=EXIT):
 try: lock.mkdir()
 except FileExistsError: raise RuntimeError('launch already claimed')
 if started.exists() or exitp.exists(): raise RuntimeError('receipt already exists')
def main():
 b=validate_binding(); claim_launch(); load=os.getloadavg()[0]
 if load>10: LOCK.rmdir(); raise RuntimeError(f'one-minute load exceeds launch limit: {load:.2f}')
 env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
 with LOG.open('ab',buffering=0) as log:
  p=subprocess.Popen([sys.executable,str(PROGRAM)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
  atomic_json(STARTED,{'launch_time_utc':datetime.now(timezone.utc).isoformat(),'one_minute_load':load,'pid':p.pid,'binding':b})
  code=p.wait()
 atomic_json(EXIT,{'exit_time_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code}); return code
if __name__=='__main__': raise SystemExit(main())
