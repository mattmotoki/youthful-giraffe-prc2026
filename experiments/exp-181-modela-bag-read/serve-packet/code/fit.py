"""Train-only epoch selection, full janjul-excluded fit, saved held vectors."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch import nn
from model import Model

SEED=150071
BATCH=256

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)

class Arrays:
    def __init__(self,root,stage,arm):
        self.root=Path(root);self.arm=arm
        def load(n):return np.load(self.root/(n+'.npy'),mmap_mode='r',allow_pickle=False)
        self.numeric=load('numeric-'+stage);self.category=load('category-'+stage)
        self.anchor=load('anchor');self.nb=load('neighbours');self.elapsed=load('elapsed-seconds')
        centres=load('departure-centres')
        self.departure_lookup=np.full(len(self.anchor),-1,dtype='int32')
        self.departure_lookup[centres]=np.arange(len(centres),dtype='int32')
        enc=json.loads((self.root/f'encoder-{stage}.json').read_text())
        self.vocab_sizes=[max(v.values(),default=0)+1 for v in enc['vocab'].values()]
        self.n_numeric=self.numeric.shape[1]
        if len(self.vocab_sizes)!=self.category.shape[1]:raise ValueError('Vocabulary shape differs')

    def batch(self,indices,device,hide_arrivals=False):
        indices=np.asarray(indices,dtype='int64')
        if (self.departure_lookup[indices]<0).any():raise ValueError('Loss/held key is not DEP')
        if self.arm=='sequence':
            rows=self.departure_lookup[indices];nb=np.array(self.nb[rows],copy=True)
            elapsed=np.array(self.elapsed[rows],copy=True)
            if hide_arrivals:
                # PHASE category has a train-owned missing/unknown encoding; use
                # explicit label-free event phase metadata for this diagnostic.
                if not hasattr(self,'is_arrival'):
                    f=pd.read_parquet(self.root/'events.parquet',columns=['PHASE_mvt'],use_threads=False)
                    self.is_arrival=f.PHASE_mvt.eq('ARR').to_numpy()
                arr=self.is_arrival[np.maximum(nb,0)] & (nb>=0);nb[arr]=-1;elapsed[arr]=0
        else:nb=indices[:,None];elapsed=np.zeros((len(indices),1),dtype='float32')
        padding=nb<0;safe=np.maximum(nb,0)
        values=[np.array(self.numeric[safe],copy=True),np.array(self.category[safe],dtype='int64',copy=True),
                elapsed,padding,np.array(self.anchor[indices],copy=True)]
        return tuple(torch.from_numpy(v).to(device) for v in values)

    def model(self):return Model(self.n_numeric,self.vocab_sizes,self.arm)

def run_epoch(model,data,indices,targets,device,optimizer=None,rng=None):
    train=optimizer is not None;model.train(train)
    order=rng.permutation(len(indices)) if train else np.arange(len(indices))
    huber_sum,mse_sum,n=0.,0.,0
    criterion=nn.SmoothL1Loss(beta=2,reduction='sum')
    with torch.set_grad_enabled(train):
        for a in range(0,len(order),BATCH):
            take=order[a:a+BATCH];ix=indices[take]
            numeric,category,elapsed,padding,anchor=data.batch(ix,device)
            target=torch.from_numpy(np.array(targets[take],dtype='float32',copy=True)).to(device)
            correction=model(numeric,category,elapsed,padding)
            loss=criterion(correction,(target-anchor)/300.)
            if train:
                optimizer.zero_grad(set_to_none=True);(loss/len(take)).backward()
                nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            huber_sum+=float(loss.detach());mse_sum+=float(((anchor+300.*correction-target)**2).sum().detach());n+=len(take)
    if not n:raise ValueError('Empty epoch')
    return {'huber':huber_sum/n,'mse_unfloored_diagnostic':mse_sum/n,'rows':n}

@torch.no_grad()
def predict(model,data,indices,device,hide_arrivals=False):
    model.eval();out=[]
    for a in range(0,len(indices),BATCH):
        n,c,e,p,anchor=data.batch(indices[a:a+BATCH],device,hide_arrivals)
        v=torch.maximum(anchor+300.*model(n,c,e,p),torch.tensor(60.,device=device))
        out.append(v.cpu().numpy())
    result=np.concatenate(out)
    if not np.isfinite(result).all() or (result<60).any():raise ValueError('Bad held prediction')
    return result

def save_model(path,model,data,stage):
    torch.save({'state_dict':model.state_dict(),'arm':data.arm,'n_numeric':data.n_numeric,
                'vocab_sizes':data.vocab_sizes,'encoder_stage':stage,'seed':SEED},path)

def load_model(path,device):
    rec=torch.load(path,map_location=device,weights_only=True)
    model=Model(rec['n_numeric'],rec['vocab_sizes'],rec['arm']).to(device)
    model.load_state_dict(rec['state_dict']);return model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--checks',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',choices=['sequence','row'],required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
    if a.out.exists():raise ValueError('Fit output exists; write-once')
    if not 1<=a.threads<=6:raise ValueError('Thread cap out of range')
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    if a.device!='cuda' or not torch.cuda.is_available():raise ValueError('Real fit belongs to grind GPU job')
    cap=int(os.environ.get('G_SEQUENCE_GPU_CAP_BYTES',10*1024**3))
    if cap!=10*1024**3:raise ValueError('GPU cap differs')
    total=torch.cuda.get_device_properties(0).total_memory
    if total<cap:raise ValueError('GPU memory below10GiB')
    torch.cuda.set_per_process_memory_fraction(cap/total,0)
    checks=json.loads(a.checks.read_text());manifest=json.loads((a.inputs/'MANIFEST.json').read_text())
    if checks['status']!='PASS' or checks['manifest_sha256']!=sha(a.inputs/'MANIFEST.json'):raise ValueError('Input checks differ')
    for name,want in manifest['files'].items():
        if sha(a.inputs/name)!=want:raise ValueError('Input hash differs: '+name)
    a.out.mkdir(parents=True);started=time.monotonic();device=torch.device(a.device)
    source_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py')]}
    ix=np.load(a.inputs/'train-indices.npy',allow_pickle=False)
    labels=np.load(a.inputs/'train-labels.npy',allow_pickle=False)
    val=np.load(a.inputs/'train-development-valid.npy',allow_pickle=False)
    if not val.any() or val.all() or not np.isfinite(labels).all():raise ValueError('Bad train-only validation split')
    dev=Arrays(a.inputs,'development',a.arm);seed();model=dev.model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(SEED);history=[]
    for epoch in range(1,13):
        train=run_epoch(model,dev,ix[~val],labels[~val],device,opt,rng)
        valid=run_epoch(model,dev,ix[val],labels[val],device)
        row={'epoch':epoch,'train':train,'validation':valid,'elapsed_seconds':time.monotonic()-started}
        history.append(row);print(json.dumps(row),flush=True)
        save_model(a.out/f'development-epoch-{epoch:02d}.pt',model,dev,'development')
    epoch_count=min(history,key=lambda r:(r['validation']['huber'],r['epoch']))['epoch']
    (a.out/'SELECTION.json').write_text(json.dumps({'rule':'minimum train-only validation smooth-L1 beta2; earlier tie',
        'selected_epochs':epoch_count,'history':history,'no_held_labels':True},indent=2)+'\n')
    del model,opt,dev
    torch.cuda.empty_cache();full=Arrays(a.inputs,'full',a.arm);seed();model=full.model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(SEED);full_history=[]
    for epoch in range(1,epoch_count+1):
        row={'epoch':epoch,'train':run_epoch(model,full,ix,labels,device,opt,rng)}
        full_history.append(row);print(json.dumps(row),flush=True)
    save_model(a.out/'model.pt',model,full,'full')
    held=np.load(a.inputs/'held-indices.npy',allow_pickle=False)
    held_ids=np.load(a.inputs/'held-ids.npy',allow_pickle=False)
    pred=predict(model,full,held,device)
    col='pred_Gmixed' if a.arm=='sequence' else 'pred_Grow'
    pd.DataFrame({'MVT_ID_mvt':held_ids,col:pred}).to_parquet(a.out/'held.parquet',index=False)
    replay=predict(load_model(a.out/'model.pt',device),full,held[:1024],device)
    if not np.array_equal(replay,pred[:1024]):raise ValueError('Model reload does not reproduce held vector')
    if a.arm=='sequence':
        hidden=predict(model,full,held,device,hide_arrivals=True)
        pd.DataFrame({'MVT_ID_mvt':held_ids,'pred_Gmixed_ARRhidden':hidden}).to_parquet(a.out/'held-ARRhidden.parquet',index=False)
    for name,want in source_hashes.items():
        if sha(Path(__file__).with_name(name))!=want:raise ValueError('Fit source changed')
    record={'status':'completed','arm':a.arm,'seed':SEED,'selected_epochs':epoch_count,'full_history':full_history,
        'loss':'smooth-L1 beta2 on correction in 300s units','floor':60,'upper_cap':None,
        'train_labels':len(ix),'train_split':manifest['split'],'train_months':manifest['summary']['train_rows_by_UTC_month'],
        'held_rows':len(held_ids),'held_labels_in_payload':False,'held_labels_used_for_selection':False,
        'manifest_sha256':sha(a.inputs/'MANIFEST.json'),'checks_sha256':sha(a.checks),'source_sha256':source_hashes,
        'model_reload_exact_first_1024':True,'seconds':time.monotonic()-started,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,
        'files':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}}
    (a.out/'FIT.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
