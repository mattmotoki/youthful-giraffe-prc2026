#!/usr/bin/env python3
from __future__ import annotations
import argparse,contextlib,fcntl,gzip,hashlib,io,json,math,os,pathlib,random,subprocess,sys,tarfile,tempfile
from datetime import datetime,timezone
REPO=pathlib.Path(__file__).resolve().parents[3]; EXP=REPO/'experiments/exp-089-movement-set'
STATE=EXP/'state'; JOURNAL=STATE/'journal.json'; FAILURE=STATE/'first-failure.json'; LOCK=STATE/'lock'; BINDING=EXP/'binding.json'; PACKAGE=EXP/'package/exp089-source.tar.gz'; ACCEPTANCE=EXP/'review/acceptance.json'
THREAD_VARS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
FORBIDDEN=('blocktime','taxitime','departurelabel','target','residual','ytrue','ypred','prediction','score','leaderboard')
TRANSITIONS={'registered':'accepted','accepted':'package_built','package_built':'dataset_published','dataset_published':'dataset_verified','dataset_verified':'kernel_launched','kernel_launched':'kernel_verified','kernel_verified':'landed_verified','landed_verified':'real_data_boundary','real_data_boundary':'complete'}
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha256_path(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_write(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(v if isinstance(v,bytes) else canonical(v));f.flush();os.fsync(f.fileno())
  os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
 except BaseException:
  with contextlib.suppress(FileNotFoundError):os.unlink(t)
  raise
def exact_json(p,keys):
 v=json.loads(pathlib.Path(p).read_text())
 if set(v)!=set(keys):raise ValueError('unexpected keys')
 return v
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
def reject_forbidden(cols):
 bad=[c for c in cols if any(x in ''.join(ch.lower() for ch in c if ch.isalnum()) for x in FORBIDDEN)]
 if bad:raise ValueError('forbidden label or derivative column')
def utc(v):
 d=datetime.fromisoformat(v.replace('Z','+00:00'))
 if d.tzinfo is None or d.utcoffset()!=timezone.utc.utcoffset(d):raise ValueError('UTC required')
 return d
def validate_rows(rows,fields):
 if not rows:raise ValueError('empty rows')
 reject_forbidden(fields);prev=None
 for r in rows:
  if set(r)!=set(fields):raise ValueError('schema mismatch')
  if type(r['MVT_ID_mvt']) is not float:raise TypeError('native ID must remain float64')
  if not math.isfinite(r['MVT_ID_mvt']) or any(isinstance(v,float) and not math.isfinite(v) for v in r.values()):raise ValueError('nonfinite')
  if prev is not None and r['MVT_ID_mvt']<=prev:raise ValueError('duplicate or unordered ID')
  prev=r['MVT_ID_mvt'];utc(r['MVT_TIME_UTC_mvt'])
 return rows
def density_stratum(n):return '0' if n==0 else '1-4' if n<=4 else '5-16' if n<=16 else '17-32' if n<=32 else 'over-32-before-cap'
def build_relations(rows,max_degree=32):
 fields=('MVT_ID_mvt','MVT_TIME_UTC_mvt','ADEP_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','x');validate_rows(rows,fields);out=[]
 rels=(('same_airport_runway','RUNWAY_mvt',-1800,600),('same_airport_stand','STAND_mvt',-7200,1800),('same_airport_operator','AIRCRAFT_OPERATOR_flt',-1800,600))
 for q in rows:
  cand=[];qt=utc(q['MVT_TIME_UTC_mvt'])
  for order,(name,field,lo,hi) in enumerate(rels):
   if q[field] is None:continue
   for n in rows:
    if n['MVT_ID_mvt']==q['MVT_ID_mvt']:continue
    dt=(utc(n['MVT_TIME_UTC_mvt'])-qt).total_seconds()
    if n['ADEP_mvt']==q['ADEP_mvt'] and n[field]==q[field] and lo<=dt<=hi:cand.append((abs(dt),order,n['MVT_ID_mvt'],name,dt,n['x']))
  cand.sort(key=lambda z:(z[0],z[1],z[2]));kept=cand[:max_degree]
  out.append({'query_id':q['MVT_ID_mvt'],'degree_before_cap':len(cand),'degree_after_cap':len(kept),'density':density_stratum(len(cand)),'neighbour_ids':[x[2] for x in kept],'relations':[x[3] for x in kept],'deltas':[x[4] for x in kept],'features':[x[5] for x in kept],'mask':[True]*len(kept)+[False]*(max_degree-len(kept))})
 return out
def relation_controls(graph,seed=89013):
 rng=random.Random(seed);a=[];b=[]
 for x in graph:
  y=dict(x);z=dict(x);y['relations']=list(x['relations']);z['features']=list(x['features']);rng.shuffle(y['relations']);rng.shuffle(z['features']);a.append(y);b.append(z)
 return a,b
def parameter_counts(w=16):return {'row_only':(w*48+48)+(48*48+48)+49,'pooled_context':(w*32+32)+(32*24+24)+(24*24+24)+25,'set_encoder':(w*24+24)+4*(24*24+24)+(24*24+24)+25}
def verify_threads():
 for n in THREAD_VARS:os.environ[n]='3'
 v=json.loads(subprocess.check_output([sys.executable,'-c','import os,json;print(json.dumps({k:os.environ.get(k) for k in '+repr(THREAD_VARS)+'}))'],text=True))
 if set(v.values())!={'3'}:raise RuntimeError('thread limit')
 return v
def synthetic_fixture(p):
 raw=pathlib.Path(p).read_bytes();v=json.loads(raw)
 if set(v)!={'schema','rows'} or v['schema']!=1:raise ValueError('fixture envelope')
 build_relations(v['rows']);return v,hashlib.sha256(raw).hexdigest()
def build_package(commit):
 reg=json.loads(subprocess.check_output(['git','show',f'{commit}:experiments/exp-089-movement-set/registration.json'],cwd=REPO));members={p:subprocess.check_output(['git','show',f'{commit}:{p}'],cwd=REPO) for p in reg['package_sources']};pm={'schema':1,'registration_commit':commit,'files':{p:hashlib.sha256(v).hexdigest() for p,v in members.items()}};members['PACKAGE-MANIFEST.json']=canonical(pm);raw=io.BytesIO()
 with tarfile.open(fileobj=raw,mode='w',format=tarfile.PAX_FORMAT) as t:
  for n,v in sorted(members.items()):i=tarfile.TarInfo('repo/'+n);i.size=len(v);i.mode=0o644;i.mtime=i.uid=i.gid=0;i.uname=i.gname='';t.addfile(i,io.BytesIO(v))
 out=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as z:z.write(raw.getvalue())
 return out.getvalue(),pm
def verify_registration():
 b=exact_json(BINDING,('schema','registration_commit','registration_tree','package','sources','inputs','permitted_later_files'));r=json.loads(subprocess.check_output(['git','show',f'{b["registration_commit"]}:experiments/exp-089-movement-set/registration.json'],cwd=REPO))
 if git('rev-parse',f'{b["registration_commit"]}^{{tree}}')!=b['registration_tree'] or subprocess.run(['git','merge-base','--is-ancestor',b['registration_commit'],'HEAD'],cwd=REPO).returncode:raise ValueError('registration identity or ancestry')
 for p,d in b['sources'].items():
  raw=subprocess.check_output(['git','show',f'{b["registration_commit"]}:{p}'],cwd=REPO)
  if hashlib.sha256(raw).hexdigest()!=d or sha256_path(REPO/p)!=d:raise ValueError('committed source bytes')
 for p,d in b['inputs'].items():
  if sha256_path(REPO/p)!=d:raise ValueError('input identity')
 pkg,_=build_package(b['registration_commit'])
 if (len(pkg),hashlib.sha256(pkg).hexdigest())!=(b['package']['bytes'],b['package']['sha256']):raise ValueError('package identity')
 changed=set(git('diff','--name-only',b['registration_commit'],'HEAD').splitlines())
 if changed-set(b['permitted_later_files']):raise ValueError('unpermitted later file')
 return b,r
class Controller:
 @contextlib.contextmanager
 def locked(self):
  STATE.mkdir(parents=True,exist_ok=True)
  with LOCK.open('a+b') as f:fcntl.flock(f,fcntl.LOCK_EX);yield
 def state(self):return json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'state':'registered','sequence':0,'pending':None}
 def fail(self,e):
  if not FAILURE.exists():atomic_write(FAILURE,{'type':type(e).__name__,'message':str(e)})
  s=self.state();atomic_write(JOURNAL,{'state':'failed','sequence':s['sequence']+1,'pending':None,'failure_sha256':sha256_path(FAILURE)})
 def transition(self,target,crash=False):
  s=self.state()
  if TRANSITIONS.get(s['state'])!=target:raise RuntimeError('invalid transition')
  atomic_write(JOURNAL,{'state':s['state'],'sequence':s['sequence'],'pending':target})
  if crash:raise SystemExit(89)
  atomic_write(JOURNAL,{'state':target,'sequence':s['sequence']+1,'pending':None})
 def recover(self):
  if self.state().get('pending'):self.fail(RuntimeError('interrupted transition'));raise RuntimeError('fail-closed crash recovery')
 def run(self):
  if os.getloadavg()[0]>10:return {'status':'hold'}
  with self.locked():
   self.recover();b,_=verify_registration()
   if not ACCEPTANCE.exists():raise RuntimeError('DISABLED: fresh Sol-xhigh acceptance is absent')
   a=exact_json(ACCEPTANCE,('schema','registration_commit','verdict','review_sha256'))
   if a['registration_commit']!=b['registration_commit'] or a['verdict']!='accepted':raise RuntimeError('acceptance mismatch')
   raise RuntimeError('DISABLED: real-data execution requires later desk launch commit')
def remote_probe():
 p=list(pathlib.Path('/kaggle/input').glob('**/exp089-source.tar.gz'))
 if len(p)!=1:raise RuntimeError('one package required')
 with tarfile.open(p[0],'r:gz') as t:
  if len(t.getnames())!=len(set(t.getnames())) or any(not m.isfile() or '..' in pathlib.PurePosixPath(m.name).parts for m in t.getmembers()):raise RuntimeError('unsafe package')
  pm=json.loads(t.extractfile('repo/PACKAGE-MANIFEST.json').read())
 atomic_write('/kaggle/working/exp089-transport.json',{'schema':1,'package_sha256':sha256_path(p[0]),'package_bytes':p[0].stat().st_size,'registration_commit':pm['registration_commit'],'member_hashes':pm['files']})
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','build','run','remote-probe'));a=p.parse_args(argv).action
 if a=='remote-probe':remote_probe()
 elif a=='verify':verify_registration()
 elif a=='build':b,_=verify_registration();raw,_=build_package(b['registration_commit']);atomic_write(PACKAGE,raw)
 else:Controller().run()
if __name__=='__main__':main()
