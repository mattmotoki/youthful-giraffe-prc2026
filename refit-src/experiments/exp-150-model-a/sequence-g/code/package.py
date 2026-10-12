"""Write-once reviewed input/source archives for desk and grind; no upload."""
import argparse,gzip,hashlib,json,os,resource,tarfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def archive(path,items):
    with path.open('xb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=1) as z,tarfile.open(fileobj=z,mode='w|') as tar:
        for name,p in sorted(items.items()):
            info=tar.gettarinfo(str(p),arcname=name)
            if not info.isfile():raise ValueError('Archive item is not a regular file')
            info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
            with p.open('rb') as f:tar.addfile(info,f)
    with tarfile.open(path,'r|gz') as tar:
        seen=set()
        for item in tar:
            if item.name not in items or not item.isfile():raise ValueError('Unexpected archive member')
            h=hashlib.sha256();f=tar.extractfile(item)
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            if h.hexdigest()!=sha(items[item.name]):raise ValueError('Archive content hash differs')
            seen.add(item.name)
        if seen!=set(items):raise ValueError('Archive is incomplete')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--experiment',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();code=a.experiment/'code'
    if a.out.exists():raise ValueError('Package root exists')
    lane={4,5,6,7,68,69,70,71}
    if os.sched_getaffinity(0)!=lane or os.getpriority(os.PRIO_PROCESS,0)!=19:raise ValueError('Package lane/nice differs')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    review=json.loads((a.experiment/'REVIEW-MODEL-SOURCE.json').read_text())
    if review['status']!='PASS':raise ValueError('Model source review missing')
    for name,want in review['source_sha256'].items():
        if sha(code/name)!=want:raise ValueError('Reviewed model source changed')
    manifest=json.loads((a.inputs/'MANIFEST.json').read_text())
    for name,want in manifest['files'].items():
        if sha(a.inputs/name)!=want:raise ValueError('Input changed')
    smoke=a.experiment/'results/SMOKE-001.json';checks=a.experiment/'results/input-checks-001.json'
    for p in (smoke,checks):
        r=json.loads(p.read_text())
        if r['status']!='PASS' or r['manifest_sha256']!=sha(a.inputs/'MANIFEST.json'):raise ValueError('Check record differs')
    inputs={'inputs/'+n:a.inputs/n for n in [*manifest['files'],'MANIFEST.json']}
    sources={'code/'+n:code/n for n in ['model.py','fit.py','run_job.py','smoke.py']}
    sources.update({'SMOKE.json':smoke,'input-checks.json':checks,
        **{n:a.experiment/n for n in ['SPEC.md','SOURCE-PINS.json','GRIND-HANDOFF.md','REVIEW-MODEL-SOURCE.json']}})
    origin_hashes={n:sha(p) for n,p in {**inputs,**sources}.items()}
    a.out.mkdir(parents=True);archive(a.out/'inputs.tgz',inputs);archive(a.out/'source.tgz',sources)
    if {n:sha(p) for n,p in {**inputs,**sources}.items()}!=origin_hashes:raise ValueError('Source changed during package')
    pack={'status':'ready; desk allocation to grind required','root':str(a.out),'input_manifest_sha256':sha(a.inputs/'MANIFEST.json'),
        'source_and_input_files_sha256':origin_hashes,'archive_bytes_reproduced':True,
        'held_rows':344419,'held_labels_in_payload':False,'GPU_hours_cap':4,'GPU_allocator_cap_bytes':10*1024**3,
        'archives':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(a.out.iterdir())}}
    (a.out/'PACK.json').write_text(json.dumps(pack,indent=2)+'\n');print(json.dumps(pack),flush=True)

if __name__=='__main__':main()
