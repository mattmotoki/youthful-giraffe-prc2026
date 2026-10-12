"""Deterministic annual source/input packet; no account or kernel action."""
import argparse,json,os,resource
from pathlib import Path
from package import archive,sha

def main():
    ap=argparse.ArgumentParser()
    for n in ['inputs','experiment','out','smoke']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();exp=a.experiment;code=exp/'code'
    if a.out.exists():raise ValueError('Annual package exists')
    if os.sched_getaffinity(0)!={4,5,6,7,68,69,70,71} or os.getpriority(os.PRIO_PROCESS,0)!=19:raise ValueError('Annual package lane/nice differs')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    review=exp/'REVIEW-ANNUAL-SOURCE.json';r=json.loads(review.read_text())
    if r['status']!='PASS':raise ValueError('Missing annual source review')
    for n,want in r['source_sha256'].items():
        if sha(code/n)!=want:raise ValueError('Reviewed source changed: '+n)
    manifest=json.loads((a.inputs/'MANIFEST.json').read_text());checks=exp/'results/ANNUAL-CHECKS-001.json'
    for p in [checks,a.smoke]:
        c=json.loads(p.read_text())
        if c['status']!='PASS' or c['manifest_sha256']!=sha(a.inputs/'MANIFEST.json'):raise ValueError('Annual check/smoke differs')
    for n,want in manifest['files'].items():
        if sha(a.inputs/n)!=want:raise ValueError('Annual input changed: '+n)
    if json.loads(checks.read_text())['checker_sha256']!=sha(code/'check_annual.py'):raise ValueError('Annual checker changed')
    plan=exp/'ANNUAL-PLAN.json'
    if manifest['plan_sha256']!=sha(plan):raise ValueError('Annual plan changed')
    sources={'code/'+n:code/n for n in r['source_sha256']}
    sources.update({'ANNUAL-CHECKS.json':checks,'ANNUAL-SMOKE.json':a.smoke,'REVIEW-ANNUAL-SOURCE.json':review,
        **{n:exp/n for n in ['SPEC.md','ANNUAL-PLAN.json','ANNUAL-NEXT.md','ANNUAL-STATE.json','ANNUAL-GRIND-HANDOFF.md']}})
    inputs={'inputs/'+n:a.inputs/n for n in [*manifest['files'],'MANIFEST.json']}
    originals={**sources,**inputs};hashes={n:sha(p) for n,p in originals.items()}
    a.out.mkdir(parents=True);archive(a.out/'inputs.tgz',inputs);archive(a.out/'source.tgz',sources)
    if hashes!={n:sha(p) for n,p in originals.items()}:raise ValueError('Annual source/input changed while packaging')
    command='timeout --signal=KILL 5400s python -u <ROOT>/code/annual_job.py --inputs <ROOT>/inputs --checks <ROOT>/ANNUAL-CHECKS.json --smoke <ROOT>/ANNUAL-SMOKE.json --review <ROOT>/REVIEW-ANNUAL-SOURCE.json --out /kaggle/working/sequence-g-annual-001'
    pack={'status':'prepared; fresh packet review required before the one approved grind push',
        'root':str(a.out),'GPU_budget_seconds':5400,'GPU_allocator_cap_bytes':10*1024**3,'fit_count':1,
        'fixed_epochs':12,'seed':150071,'ranking_rows':344841,'ranking_labels_in_payload':False,
        'input_manifest_sha256':sha(a.inputs/'MANIFEST.json'),'review_sha256':sha(review),
        'exact_command':command,'outer_timeout':'GNU timeout sends KILL to whole process group at5400s including preflight and both child processes',
        'source_and_input_files_sha256':hashes,'archive_bytes_reproduced':True,
        'archives':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(a.out.iterdir())}}
    (a.out/'PACK.json').write_text(json.dumps(pack,indent=2)+'\n');print(json.dumps(pack),flush=True)

if __name__=='__main__':main()
