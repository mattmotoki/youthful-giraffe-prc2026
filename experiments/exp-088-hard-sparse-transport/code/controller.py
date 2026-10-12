"""Only public exp-088 route: fixed Phase-A controller and remote probe."""
import os
import argparse,contextlib,fcntl,gzip,hashlib,importlib,io,json,math,os,pathlib,shutil,subprocess,sys,tarfile,tempfile
from datetime import date,timedelta

def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

EXP=pathlib.Path(__file__).resolve().parents[1];REPO=EXP.parents[1];STATE=EXP/'repair-state';LOCK=STATE/'lock';JOURNAL=STATE/'journal.json';FAILURE=STATE/'first-failure.json';MANIFEST=EXP/'source-manifest.json';BINDING=EXP/'binding-receipt.json';PACKAGE=EXP/'package-v2/exp088-repair.bin';
def dataset_ref():
    return f'{kaggle_user()}/prc-exp088-hard-sparse-repair-transport'

def kernel_ref():
    return f'{kaggle_user()}/prc-exp-088-hard-sparse-repair-probe'

SOURCES=('experiments/exp-088-hard-sparse-transport/code/controller.py','experiments/exp-088-hard-sparse-transport/private-probe-v2/kernel.py','experiments/exp-088-hard-sparse-transport/fixtures/synthetic-v2.json','experiments/exp-088-hard-sparse-transport/scientific-contract.json')
ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def blob(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(v if isinstance(v,bytes) else blob(v));f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 except BaseException:
  with contextlib.suppress(FileNotFoundError):os.unlink(t)
  raise
def exact(p,keys):
 v=json.loads(pathlib.Path(p).read_text())
 if set(v)!=set(keys):raise ValueError('exact keys')
 return v
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO,text=True).strip()
def contract():
 c=exact(EXP/'scientific-contract.json',('schema','rotation','members','families','airports','thresholds','supports','lambda_grid','beta_grid','singleton_rule','controls','orders','ties','draws','gate','bound','intervals','branches','early_stops','endpoint'))
 if c['rotation']!={'V':'2025-05','S':'2025-06','I':'2025-04'} or c['thresholds']!=[25,50,100,200] or c['supports']!=[10,30,100,300,1000,'inf'] or c['lambda_grid']!=[3,10,30,100,300,1000,3000,10000,'inf'] or c['endpoint']!={'month':'2025-05','board_recombination':False}:raise ValueError('D47 contract')
 return c
def fixture(p):
 raw=p.read_bytes();d=json.loads(raw)
 if set(d)!={'schema','a_rows','feature_rows','null_rows'} or d['schema']!=1 or d['null_rows']:raise ValueError('fixture schema')
 for name,fields in [('a_rows',{'MVT_ID_mvt','x','y_synthetic'}),('feature_rows',{'MVT_ID_mvt','utc_date','family','airport','key','n_A','value','S','I','V'})]:
  rows=d[name]
  if not rows:raise ValueError('zero rows')
  ids=[];keys=[]
  for r in rows:
   if set(r)!=fields or type(r['MVT_ID_mvt']) is not int:raise ValueError('schema/dtype')
   ids.append(r['MVT_ID_mvt'])
   if any(isinstance(v,float) and not math.isfinite(v) for v in r.values()):raise ValueError('nonfinite')
   if 'n_A' in r and (type(r['n_A']) is not int or r['n_A']<0):raise ValueError('count')
   if 'utc_date' in r:date.fromisoformat(r['utc_date'])
   if 'key' in r:keys.append((r['family'],r['airport'],r['key']))
  if len(ids)!=len(set(ids)) or keys and len(keys)!=len(set(keys)):raise ValueError('duplicate')
 return d,hashlib.sha256(raw).hexdigest()
def blocks(rows,n):
 ds=[date.fromisoformat(r['utc_date']) for r in rows];out=[]
 if len(ds)!=len(set(ds)):raise ValueError('dates')
 for i in range(len(ds)-n+1):
  if all(ds[i+j]==ds[i]+timedelta(days=j) for j in range(n)):out.append(list(range(i,i+n)))
 if not out:raise ValueError('block')
 return out
def null_record(rows,c):
 req={(f,a) for f in c['families'] for a in c['airports']}
 if {(r['family'],r['airport']) for r in rows}!=req:raise ValueError('groups')
 attempts=[];selected=[]
 for sham in range(1,11):
  pick=None
  for seed in range(1,100001):
   cells=[];allok=True
   for f,a in sorted(req):
    base=sorted([r for r in rows if r['family']==f and r['airport']==a],key=lambda r:r['key'])
    for n in c['thresholds']:
     e=[r for r in base if r['n_A']<n];single=len(e)<2;order=list(range(len(e)))
     if not single:
      shift=1+(int(hashlib.sha256(f'{seed}:{sham}:{f}:{a}:{n}'.encode()).hexdigest(),16)%(len(e)-1));perm=order[shift:]+order[:shift]
     else:perm=order
     for support in c['supports']:
      vals=[0. if single else r['value']*(1. if support=='inf' else support/(support+r['n_A'])) for r in e];sv=[vals[j] for j in perm];mi=[i for i,j in enumerate(perm) if i!=j and vals[i]!=vals[j]];moved=[e[i]['key'] for i in mi];affected={m:sum(int(e[i][m]) for i in mi) for m in ('S','I','V')};ok=single or bool(moved) and sorted(vals)==sorted(sv) and all(affected.values());allok &= ok
      cells.append({'family':f,'airport':a,'threshold':n,'support':str(support),'permutation':perm,'moved_keys':moved,'supported_strata':[e[i]['key'] for i,v in enumerate(vals) if v!=0],'true_sha256':hashlib.sha256(blob(vals)).hexdigest(),'sham_sha256':hashlib.sha256(blob(sv)).hexdigest(),'affected_rows':affected,'singleton':single,'accepted':ok})
   x={'sham':sham,'seed':seed,'cells':cells,'accepted':allok};attempts.append(x)
   if allok:pick=x;break
  if not pick:raise RuntimeError('matched-null unresolved; open')
  selected.append({'sham':sham,'seed':pick['seed']})
 return {'schema':1,'required_groups':[list(x) for x in sorted(req)],'attempts':attempts,'selected':selected}
def threads():
 for n in ENV:os.environ[n]='3'
 q=subprocess.check_output([sys.executable,'-c','import os,json;print(json.dumps({k:os.environ[k] for k in '+repr(ENV)+'}))'],text=True).strip();facts={'environment_subprocess':json.loads(q),'tail_adapter':{'n_jobs':3,'num_threads':3}}
 try:
  import numpy as np;x=np.array([[0.],[1.],[2.],[3.]]);y=np.arange(4.)
  facts['blas_dot']=float((x.T@x)[0,0])
  try:
   from lightgbm import LGBMRegressor;m=LGBMRegressor(n_estimators=2,num_threads=3,n_jobs=3,verbosity=-1).fit(x,y);facts['lightgbm']={k:m.get_params()[k] for k in ('num_threads','n_jobs')}
  except ImportError:facts['lightgbm']={'unavailable':True,'command_args':['num_threads=3','n_jobs=3']}
  try:
   from catboost import CatBoostRegressor;m=CatBoostRegressor(iterations=2,thread_count=3,task_type='CPU',verbose=False,allow_writing_files=False).fit(x,y);facts['catboost']={'thread_count':m.get_params()['thread_count']}
  except ImportError:facts['catboost']={'unavailable':True,'command_args':['thread_count=3','task_type=CPU']}
 except ImportError:facts['blas_dot']='unavailable'
 return facts
def build(commit):
 files={};members={}
 for rel in SOURCES:
  b=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=REPO);mode=git('ls-tree',commit,rel).split()[0]
  if mode!='100644':raise ValueError('mode')
  files[rel]={'sha256':hashlib.sha256(b).hexdigest(),'mode':mode};members[rel]=b
 pm={'schema':1,'source_commit':commit,'files':files};members['GIT_COMMIT.txt']=(commit+'\n').encode();members['PACKAGE-MANIFEST.json']=blob(pm);raw=io.BytesIO()
 with tarfile.open(fileobj=raw,mode='w',format=tarfile.PAX_FORMAT) as tf:
  for name,b in sorted(members.items()):
   ti=tarfile.TarInfo('repo/'+name);ti.size=len(b);ti.mode=0o644;ti.mtime=ti.uid=ti.gid=0;ti.uname=ti.gname='';tf.addfile(ti,io.BytesIO(b))
 out=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as z:z.write(raw.getvalue())
 return out.getvalue(),pm
class Controller:
 @contextlib.contextmanager
 def lock(self):
  STATE.mkdir(parents=True,exist_ok=True)
  with LOCK.open('a+b') as f:fcntl.flock(f,fcntl.LOCK_EX);yield
 def state(self):return json.loads(JOURNAL.read_text()) if JOURNAL.exists() else {'state':'registered','sequence':0,'pending':None}
 def fail(self,e):
  if not FAILURE.exists():atomic(FAILURE,{'type':type(e).__name__,'message':str(e)})
  s=self.state();atomic(JOURNAL,{'state':'failed','sequence':s['sequence']+1,'pending':None,'failure_sha256':sha(FAILURE)})
 def transition(self,old,new,facts=None,crash=False):
  s=self.state()
  if s['state']!=old:raise RuntimeError('state order')
  atomic(JOURNAL,{'state':old,'sequence':s['sequence'],'pending':new})
  if crash:raise SystemExit(88)
  atomic(JOURNAL,{'state':new,'sequence':s['sequence']+1,'pending':None,'facts':facts or {}})
 def verify(self):
  m=exact(MANIFEST,('schema','registration_commit','package_source_commit','sources','package','inputs','scientific_sha256','kernel','outputs','permitted_post_registration'));b=exact(BINDING,('schema','registration_commit','registration_tree','binding_commit','package_source_commit','package','sources','inputs','kernel','outputs'))
  if m['schema']!=2 or b['registration_commit']!=m['registration_commit'] or git('rev-parse',f"{m['registration_commit']}^{{tree}}")!=b['registration_tree']:raise ValueError('registration')
  if subprocess.run(['git','merge-base','--is-ancestor',m['registration_commit'],b['binding_commit']],cwd=REPO).returncode:raise ValueError('ancestry')
  for rel,f in m['sources'].items():
   if sha(REPO/rel)!=f['sha256']:raise ValueError('source mutation')
  if sha(EXP/'scientific-contract.json')!=m['scientific_sha256']:raise ValueError('contract')
  contract();return m,b
 def public(self,action,crash=False):
  if os.getloadavg()[0]>10:return {'status':'hold','state':self.state()['state']}
  with self.lock():
   s=self.state()
   if s.get('pending'):self.fail(RuntimeError('crash recovery'));raise RuntimeError('terminal crash recovery')
   try:
    if action=='verify':self.verify();return {'status':'verified'}
    if action=='build':
     m,_=self.verify();a,pm=build(m['package_source_commit']);b,_=build(m['package_source_commit'])
     if a!=b or len(a)>=200000000:raise ValueError('build')
     atomic(PACKAGE,a);self.transition('registered','built',{'bytes':len(a),'sha256':hashlib.sha256(a).hexdigest()},crash);return self.state()
    if action=='label-boundary':self.verify();self.transition('landed_verified','label_boundary');raise RuntimeError('DISABLED REAL LABEL BOUNDARY')
    if action=='phase-a':
     m,_=self.verify()
     if self.state()['state']!='built':raise RuntimeError('state order')
     x=subprocess.run(['kaggle','datasets','version','-p',str(EXP/'package-v2'),'-m','Fresh repaired controller package','--dir-mode','zip'],text=True,capture_output=True)
     if x.returncode:raise RuntimeError('dataset publication:'+x.stderr)
     self.transition('built','dataset_published',{'stdout':x.stdout})
     status=subprocess.check_output(['kaggle','datasets','status',dataset_ref()],text=True).strip();files=subprocess.check_output(['kaggle','datasets','files',dataset_ref(),'-v'],text=True)
     if status!='ready' or f"exp088-repair.bin,{m['package']['bytes']}," not in files:raise RuntimeError('dataset verification')
     atomic(EXP/'private-probe-v2/dataset-api.txt',(status+'\n'+files).encode());self.transition('dataset_published','dataset_verified',{'sha256':sha(EXP/'private-probe-v2/dataset-api.txt')})
     pushed=subprocess.check_output(['kaggle','kernels','push','-p',str(EXP/'private-probe-v2')],text=True);self.transition('dataset_verified','kernel_launched',{'stdout':pushed})
     listing=subprocess.check_output(['kaggle','kernels','list','--mine','-s','hard-sparse-repair','-v'],text=True);atomic(EXP/'private-probe-v2/kernel-start-api.txt',listing.encode());self.transition('kernel_launched','start_verified',{'sha256':sha(EXP/'private-probe-v2/kernel-start-api.txt')})
     terminal=''
     for _ in range(60):
      terminal=subprocess.check_output(['kaggle','kernels','status',kernel_ref()],text=True).strip()
      if 'COMPLETE' in terminal:break
      if 'ERROR' in terminal or 'CANCEL' in terminal:raise RuntimeError(terminal)
      __import__('time').sleep(5)
     if 'COMPLETE' not in terminal:raise RuntimeError('terminal timeout')
     atomic(EXP/'private-probe-v2/kernel-terminal-api.txt',(terminal+'\n').encode());self.transition('start_verified','terminal_verified',{'sha256':sha(EXP/'private-probe-v2/kernel-terminal-api.txt')})
     out=EXP/'private-probe-v2/output';out.mkdir(parents=True,exist_ok=True);subprocess.check_call(['kaggle','kernels','output',kernel_ref(),'-p',str(out),'-q']);result=json.loads((out/'transport-result-v2.json').read_text())
     if not result['pass'] or result['package_sha256']!=m['package']['sha256']:raise RuntimeError('landed output')
     facts={p.name:sha(p) for p in out.iterdir() if p.is_file()};atomic(EXP/'private-probe-v2/landed.json',facts);self.transition('terminal_verified','landed_verified',facts);return self.state()
    raise ValueError('public action')
   except BaseException as e:self.fail(e);raise
def remote_probe():
 ps=list(pathlib.Path('/kaggle/input').glob('**/exp088-repair.bin'))
 if len(ps)!=1:raise RuntimeError('package')
 p=ps[0]
 with tarfile.open(p,'r:gz') as tf:
  ns=tf.getnames()
  if len(ns)!=len(set(ns)) or any('..' in pathlib.PurePosixPath(n).parts or not m.isfile() for n,m in zip(ns,tf.getmembers())):raise RuntimeError('members')
  pm=json.loads(tf.extractfile('repo/PACKAGE-MANIFEST.json').read());td=pathlib.Path(tempfile.mkdtemp());tf.extractall(td,filter='data')
  try:
   sys.path.insert(0,str(td/'repo/experiments/exp-088-hard-sparse-transport/code'));c=importlib.import_module('controller');d,fh=c.fixture(td/'repo/experiments/exp-088-hard-sparse-transport/fixtures/synthetic-v2.json');nr=c.null_record(d['feature_rows'],c.contract());result={'pass':True,'package_bytes':p.stat().st_size,'package_sha256':sha(p),'internal_manifest_sha256':hashlib.sha256(blob(pm)).hexdigest(),'source_commit':pm['source_commit'],'fixture_sha256':fh,'null_sha256':hashlib.sha256(blob(nr)).hexdigest(),'threads':c.threads(),'blocks':{str(n):c.blocks(d['feature_rows'],n) for n in (1,2,3)},'members':ns};atomic('/kaggle/working/transport-result-v2.json',result);atomic('/kaggle/working/probe-v2.log',b'PASS\n')
  finally:shutil.rmtree(td)
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('verify','build','phase-a','label-boundary','remote-probe'));a=p.parse_args().action
 remote_probe() if a=='remote-probe' else Controller().public(a)
if __name__=='__main__':main()
