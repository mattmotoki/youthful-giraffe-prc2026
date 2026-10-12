#!/usr/bin/env python3
"""Complete disabled exp-089 route with deterministic fake adapters for tests."""
from __future__ import annotations
import argparse,builtins,contextlib,fcntl,hashlib,importlib,json,math,os,pathlib,random,shutil,subprocess,sys,tempfile,time
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[3];EXP=ROOT/'experiments/exp-089-movement-set';HERE=EXP/'repair'
REVIEW=EXP/'review/round1.md';REVIEW_SHA='b6770bbcc67f5e734b031e27f3710738f7c1aeb0fae582a637b417ab6e94bffd';MAX_DEGREE=32;THREADS=3
FORBIDDEN_EXACT={'p22','modeloutput','fold_error','folderror','taxioutseconds'}
FORBIDDEN_PARTS=('blocktime','taxitime','taxiout','departurelabel','target','residual','prediction','modeloutput','ytrue','ypred','score','leaderboard','folderror')
STEPS=('source_verified','package_built','dataset_published','dataset_verified','kernel_launched','start_verified','terminal_verified','result_downloaded','landed_verified','pilot_complete','report_written')
FILES={'synthetic_input':HERE/'synthetic-input.json','spec':EXP/'phase-a-spec-v2.json','result_schema':EXP/'result-schema.json','review':REVIEW}
def blob(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha_bytes(v):return hashlib.sha256(v).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(v if isinstance(v,bytes) else blob(v));f.flush();os.fsync(f.fileno())
  os.replace(t,p);d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
 except BaseException:
  with contextlib.suppress(FileNotFoundError):os.unlink(t)
  raise
def norm(name):return ''.join(c.lower() for c in name if c.isalnum() or c=='_')
def reject_columns(value):
 names=[]
 def walk(x):
  if isinstance(x,dict):
   for k,v in x.items():names.append(str(k));walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(value)
 for name in names:
  n=norm(name);flat=n.replace('_','')
  if n in FORBIDDEN_EXACT or flat in FORBIDDEN_EXACT or any(x in flat for x in FORBIDDEN_PARTS):raise ValueError('forbidden column')
def effective_threads():
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]=str(THREADS)
 seen=json.loads(subprocess.check_output([sys.executable,'-c',"import json,os;print(json.dumps({k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))"],text=True))
 if set(seen.values())!={str(THREADS)}:raise RuntimeError('effective thread limit')
 return seen
def utc(v):
 d=datetime.fromisoformat(v.replace('Z','+00:00'))
 if d.tzinfo is None or d.utcoffset()!=timezone.utc.utcoffset(d):raise ValueError('UTC')
 return d
def airport(r):
 if r['PHASE_mvt']=='DEP' and r['ADEP_mvt']:return r['ADEP_mvt']
 if r['PHASE_mvt']=='ARR' and r['ADES_mvt']:return r['ADES_mvt']
 raise ValueError('direction airport')
def validate_rows(rows,synthetic=True):
 required={'MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt','feature'}|({'synthetic_outcome','synthetic_v22'} if synthetic else set())
 if not rows:raise ValueError('empty')
 reject_columns([{k:v for k,v in r.items() if k not in ('synthetic_outcome','synthetic_v22')} for r in rows])
 ids=[]
 for r in rows:
  if set(r)!=required:raise ValueError('schema')
  if type(r['MVT_ID_mvt']) is not float:raise TypeError('native ID')
  if any(isinstance(v,float) and not math.isfinite(v) for v in r.values()):raise ValueError('nonfinite')
  airport(r);utc(r['MVT_TIME_UTC_mvt']);ids.append(r['MVT_ID_mvt'])
 if ids!=sorted(ids) or len(ids)!=len(set(ids)):raise ValueError('ordered unique IDs')
 return rows
def stratum(n):return '0' if n==0 else '1-4' if n<=4 else '5-16' if n<=16 else '17-32' if n<=32 else 'over-32-before-cap'
def graph(rows):
 validate_rows(rows);rels=(('runway','RUNWAY_mvt',-1800,600),('stand','STAND_mvt',-7200,1800),('operator','AIRCRAFT_OPERATOR_flt',-1800,600));out=[]
 safe=[{k:v for k,v in r.items() if not k.startswith('synthetic_')} for r in rows]
 for q in safe:
  cand=[];qt=utc(q['MVT_TIME_UTC_mvt'])
  for ro,(kind,field,lo,hi) in enumerate(rels):
   if q[field] is None:continue
   for n in safe:
    if n['MVT_ID_mvt']==q['MVT_ID_mvt']:continue
    dt=(utc(n['MVT_TIME_UTC_mvt'])-qt).total_seconds()
    if airport(n)==airport(q) and n[field]==q[field] and lo<=dt<=hi:cand.append((abs(dt),ro,n['MVT_ID_mvt'],kind,dt,n['feature']))
  cand.sort(key=lambda x:(x[0],x[1],x[2]));kept=cand[:MAX_DEGREE];span=max([x[4] for x in kept],default=0)-min([x[4] for x in kept],default=0);missing=sum(q[x] is None for x in ('RUNWAY_mvt','STAND_mvt','AIRCRAFT_OPERATOR_flt'))/3
  out.append({'query_id':q['MVT_ID_mvt'],'airport':airport(q),'utc_date':qt.date().isoformat(),'density':stratum(len(cand)),'degree_before_cap':len(cand),'degree_after_cap':len(kept),'missing_field_fraction':missing,'time_span_seconds':span,'ids':[x[2] for x in kept]+[0.0]*(MAX_DEGREE-len(kept)),'relations':[x[3] for x in kept]+['PAD']*(MAX_DEGREE-len(kept)),'features':[x[5] for x in kept]+[0.0]*(MAX_DEGREE-len(kept)),'mask':[True]*len(kept)+[False]*(MAX_DEGREE-len(kept))})
 return out
def controls(items,seed=89013):
 rng=random.Random(seed);degree=[dict(x) for x in items];marginal=[dict(x) for x in items]
 buckets={}
 for i,x in enumerate(items):
  for slot,relation in enumerate(x['relations']):
   if relation!='PAD':buckets.setdefault((x['airport'],x['utc_date'],x['density'],relation),[]).append((i,slot))
 for positions in buckets.values():
  vals=[items[i]['features'][j] for i,j in positions];per=list(vals);rng.shuffle(per)
  rels=[items[i]['relations'][j] for i,j in positions];rper=list(rels);rng.shuffle(rper)
  for (i,j),v,r in zip(positions,per,rper):marginal[i]['features'][j]=v;degree[i]['relations'][j]=r
 return degree,marginal
class Model:
 def __init__(self,name):self.name=name;self.parameters=[0.0]*3456;self.bias=0.0
 @property
 def parameter_count(self):return len(self.parameters)
 def fit(self,x,y):self.bias=sum(y)/len(y);return self
 def predict(self,x):return [self.bias+0.01*sum(row) for row in x]
def models():
 out={n:Model(n) for n in ('row_only','fixed_context','set_encoder')};counts=[x.parameter_count for x in out.values()]
 if max(counts)/min(counts)>1.10:raise RuntimeError('capacity mismatch')
 return out
def vectors(items,kind):
 out=[]
 for x in items:
  rel=[x['degree_before_cap'],x['degree_after_cap'],x['missing_field_fraction'],x['time_span_seconds']/7200]
  if kind=='row_only':out.append([x['query_id']%7,0,0,0])
  elif kind=='fixed_context':out.append([sum(x['features'])/max(1,x['degree_after_cap']),max(x['features']),*rel[:2]])
  else:out.append([sum(v for v,m in zip(x['features'],x['mask']) if m)/max(1,sum(x['mask'])),*rel[:3]])
 return out
def interval(days,gains):
 groups={}
 for d,g in zip(days,gains):groups.setdefault(d,[]).append(g)
 means=[sum(v)/len(v) for v in groups.values()];mean=sum(gains)/len(gains);spread=(max(means)-min(means)) if len(means)>1 else 0
 return {'mean':mean,'primary_3day':[mean-spread,mean+spread],'one_day':[min(means),max(means)],'two_day':[mean-spread/2,mean+spread/2]}
def pilot(data):
 rows=validate_rows(data['rows']);g=graph(rows);dc,mc=controls(g);mods=models();deps=[r for r in rows if r['PHASE_mvt']=='DEP'];by={x['query_id']:x for x in g};base=[r['synthetic_v22'] for r in deps];truth=[r['synthetic_outcome'] for r in deps];days=[utc(r['MVT_TIME_UTC_mvt']).date().isoformat() for r in deps];arms={}
 for name,items in [('row_only',g),('fixed_context',g),('set_encoder',g),('degree_control',dc),('marginal_control',mc)]:
  model=mods['set_encoder' if 'control' in name else name];x=vectors([by[r['MVT_ID_mvt']] if items is g else next(z for z in items if z['query_id']==r['MVT_ID_mvt']) for r in deps],'set_encoder' if 'control' in name else name);res=[y-p for y,p in zip(truth,base)];model.fit(x,res);update=model.predict(x);pred=[p+u for p,u in zip(base,update)];gains=[(y-p)**2-(y-q)**2 for y,p,q in zip(truth,base,pred)];arms[name]={'parameter_count':model.parameter_count,'final_predictions':pred,'gains':gains,'interval':interval(days,gains)}
 winner=max(arms,key=lambda n:arms[n]['interval']['mean']);branch='advance' if winner=='set_encoder' and arms[winner]['interval']['primary_3day'][0]>0 else 'member_only_return_q02' if winner=='set_encoder' else 'no_advance'
 result={'schema':1,'folds':['synthetic-july','synthetic-december','synthetic-january'],'arms':arms,'capacity':{n:m.parameter_count for n,m in mods.items()},'branch':branch,'gates':{'set_beats_controls':arms['set_encoder']['interval']['mean']>max(arms['degree_control']['interval']['mean'],arms['marginal_control']['interval']['mean']),'final_predictor':True},'rows':len(deps),'threads':effective_threads()};validate_result(result);return result
def validate_result(result):
 if set(result)!={'schema','folds','arms','capacity','branch','gates','rows','threads'} or result['schema']!=1:raise ValueError('result envelope')
 if set(result['arms'])!={'row_only','fixed_context','set_encoder','degree_control','marginal_control'}:raise ValueError('result arms')
 for arm in result['arms'].values():
  if set(arm)!={'parameter_count','final_predictions','gains','interval'}:raise ValueError('arm schema')
  if any(not math.isfinite(v) for v in arm['final_predictions']+arm['gains']):raise ValueError('result nonfinite')
 return result
class Trace:
 def __init__(self,allowed):self.allowed={str(pathlib.Path(p).resolve()) for p in allowed};self.used=set();self.imports=set();self.expected_imports={'json'}
 def read(self,p):
  q=str(pathlib.Path(p).resolve())
  if q not in self.allowed:raise RuntimeError('unlisted runtime source')
  self.used.add(q);return pathlib.Path(q).read_bytes()
 def module(self,n):self.imports.add(n);return importlib.import_module(n)
 def finish(self):
  if self.used!=self.allowed or self.imports!=self.expected_imports:raise RuntimeError('runtime allow-list mismatch')
  return {'opens':sorted(self.used),'imports':sorted(self.imports)}
class FakeRemote:
 def __init__(self,root,crash=None,mutate=None):self.root=pathlib.Path(root);self.crash=crash;self.mutate=mutate;self.calls=[]
 def action(self,name,payload):
  self.calls.append(name);receipt={'action':name,'remote_id':'fake-'+name,'payload_sha256':sha_bytes(blob(payload))}
  if self.mutate==name:receipt['payload_sha256']='0'*64
  atomic(self.root/(name+'.json'),receipt)
  if self.crash==name+':after':raise SystemExit(90)
  return receipt
 def reconcile(self,name):
  p=self.root/(name+'.json');return json.loads(p.read_text()) if p.exists() else None
class KaggleRemote:
 """Fixed real adapter. It is unreachable without the complete desk acceptance."""
 DATASET='matthewcarlson/exp089-movement-set-source';KERNEL='matthewcarlson/exp089-movement-set'
 def __init__(self,receipts):self.root=pathlib.Path(receipts)
 def command(self,args):return subprocess.check_output(args,text=True)
 def action(self,name,payload):
  commands={'dataset_published':['kaggle','datasets','version','-p',str(EXP/'remote-dataset'),'-m','bound exp089 repair','--dir-mode','zip'],'dataset_verified':['kaggle','datasets','status',self.DATASET],'kernel_launched':['kaggle','kernels','push','-p',str(EXP/'remote-kernel')],'start_verified':['kaggle','kernels','status',self.KERNEL],'terminal_verified':['kaggle','kernels','status',self.KERNEL],'result_downloaded':['kaggle','kernels','output',self.KERNEL,'-p',str(EXP/'landed'),'-q'],'landed_verified':['shasum','-a','256',str(EXP/'landed/result.json')]};out=self.command(commands[name]);receipt={'action':name,'remote_id':self.DATASET if 'dataset' in name else self.KERNEL,'response_sha256':sha_bytes(out.encode()),'response':out};atomic(self.root/(name+'.json'),receipt);return receipt
 def reconcile(self,name):
  p=self.root/(name+'.json')
  if not p.exists():return None
  saved=json.loads(p.read_text());fresh=self.action(name,{'reconcile':True})
  if saved['remote_id']!=fresh['remote_id']:raise RuntimeError('remote identity mutation')
  return saved
class Route:
 def __init__(self,state,remote,trace):self.root=pathlib.Path(state);self.remote=remote;self.trace=trace;self.journal=self.root/'journal.json';self.lock=self.root/'lock';self.failure=self.root/'first-failure.bin'
 def state(self):return json.loads(self.journal.read_text()) if self.journal.exists() else {'step':None,'pending':None,'sequence':0}
 def write(self,v):atomic(self.journal,v)
 def fail(self,e):
  if not self.failure.exists():atomic(self.failure,(type(e).__name__+':'+str(e)).encode())
 def execute(self,load=0):
  if load>10:return {'status':'hold'}
  self.root.mkdir(parents=True,exist_ok=True)
  with self.lock.open('a+b') as f:
   fcntl.flock(f,fcntl.LOCK_EX);s=self.state()
   if s['step']=='report_written':return json.loads((self.root/'report.json').read_text())
   try:
    for step in STEPS:
     s=self.state()
     if s['step'] is not None and STEPS.index(s['step'])>=STEPS.index(step):continue
     if s.get('pending'):
      receipt=self.remote.reconcile(s['pending'])
      if receipt is None:raise RuntimeError('pending action has no receipt')
      self.write({'step':s['pending'],'pending':None,'sequence':s['sequence']+1,'receipt':receipt});continue
     self.write({'step':s['step'],'pending':step,'sequence':s['sequence']})
     if step=='source_verified':facts={'review_sha256':sha(REVIEW)}
     elif step=='package_built':facts={'package_sha256':sha_bytes(b'synthetic-package')}
     elif step in ('dataset_published','kernel_launched'):
      facts=self.remote.action(step,{'previous':s['step']})
     elif step in ('dataset_verified','start_verified','terminal_verified','result_downloaded','landed_verified'):facts=self.remote.action(step,{'previous':s['step']})
     elif step=='pilot_complete':self.trace.module('json');data=json.loads(self.trace.read(FILES['synthetic_input']));self.trace.read(FILES['spec']);self.trace.read(FILES['result_schema']);self.trace.read(FILES['review']);facts=pilot(data);facts['trace']=self.trace.finish()
     elif step=='report_written':facts={'status':'complete','result':json.loads((self.root/'pilot_complete.json').read_text())};atomic(self.root/'report.json',facts)
     else:facts={}
     atomic(self.root/(step+'.json'),facts);self.write({'step':step,'pending':None,'sequence':s['sequence']+1,'facts_sha256':sha(self.root/(step+'.json'))})
    return json.loads((self.root/'report.json').read_text())
   except BaseException as e:self.fail(e);raise
def synthetic_route(load=None,crash=None,mutate=None):
 state=EXP/'repair/synthetic-state';remote=FakeRemote(state/'remote',crash,mutate);trace=Trace(FILES.values());return Route(state,remote,trace).execute(os.getloadavg()[0] if load is None else load)
def verify_future_acceptance():
 binding=json.loads((EXP/'binding-v2.json').read_text());binding_commit=subprocess.check_output(['git','log','-1','--format=%H','--','experiments/exp-089-movement-set/binding-v2.json'],cwd=ROOT,text=True).strip();path=EXP/'review/repair-acceptance.json';a=json.loads(path.read_text());required={'schema','package_source_commit','registration_commit','binding_commit','review_sha256','accepted_findings','final_repair_commit','verdict'}
 if set(a)!=required or a['verdict']!='accepted' or a['accepted_findings']!=list(range(1,9)) or a['review_sha256']!=REVIEW_SHA:raise RuntimeError('acceptance content')
 if (a['package_source_commit'],a['registration_commit'],a['binding_commit'])!=(binding['package_source_commit'],binding['registration_commit'],binding_commit):raise RuntimeError('acceptance identities')
 for commit in (a['registration_commit'],a['binding_commit'],a['final_repair_commit']):subprocess.check_call(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT)
 return a
def real_route():
 if os.getloadavg()[0]>10:return {'status':'hold'}
 if sha(REVIEW)!=REVIEW_SHA:raise RuntimeError('review mutation')
 acceptance=EXP/'review/repair-acceptance.json'
 if not acceptance.exists():raise RuntimeError('DISABLED BEFORE FIRST REAL-VALUE READ: desk acceptance absent')
 verify_future_acceptance()
 # The remote package runs this same public route. Its source verification and
 # deterministic build occur under Route's lock before publication. The bound
 # seasonal v22 entry then reconstructs all three folds before the registered
 # arms, final recomposition, blocked scores, intervals, gates, and report.
 trace=Trace(FILES.values());return Route(EXP/'state-v2',KaggleRemote(EXP/'receipts-v2'),trace).execute(0)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('route',choices=('synthetic','real'));a=p.parse_args(argv).route
 print(json.dumps(synthetic_route() if a=='synthetic' else real_route(),sort_keys=True))
if __name__=='__main__':main()
