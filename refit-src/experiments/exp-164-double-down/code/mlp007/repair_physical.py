"""Repair the fixed smaller list before fits; retain all prepared array bytes."""
import json
from pathlib import Path
from io_utils import W, sha, artifact, write, lane

def main():
    lane();inp=W/'inputs/mlp-007';result=W/'results/mlp-007'
    if any(p.is_file() for p in (W/'out/mlp-007').rglob('*')):raise RuntimeError('fit output exists; stop before changing a fixed input list')
    p=inp/'INPUTS.json';old=p.read_bytes();m=json.loads(old)
    if (inp/'INPUTS-prepare03.json').exists():raise RuntimeError('original manifest archive exists')
    nums=m['features']['full']['numeric'];small=m['features']['physical']
    admin={'lirf_flight_len','lirf_ades_is_adep','lirf_stand_num','lirf_stand_retired'}
    names=[c for c in nums if c in small['numeric'] or (c.startswith('lirf_') and c not in admin)]
    old_names=small['numeric'];small['numeric']=names;small['numeric_indices']=[nums.index(c) for c in names]
    if not len(old_names)<len(names)<len(nums):raise RuntimeError('physical list repair changed no width or adds full list')
    (inp/'INPUTS-prepare03.json').write_bytes(old)
    m['physical_list_repair']={'source_manifest':artifact(inp/'INPUTS-prepare03.json'),
        'code':artifact(Path(__file__)),'reason':'fresh review found omitted LIRF physical family before any fit or score',
        'added':[c for c in names if c not in old_names],'excluded_admin':sorted(admin),
        'arrays_changed':False,'no_model_fit':True,'no_held_score':True}
    write(p,m,overwrite=True)
    write(result/'CHECK-INPUTS-REPAIRED.json',{'state':'pass','manifest_sha256':sha(p),
        'original_source_checks':artifact(result/'CHECK-INPUTS.json'),'repair':m['physical_list_repair'],
        'full_numeric':len(nums),'physical_numeric':len(names)})
    print(json.dumps({'physical_numeric':len(names),'added':m['physical_list_repair']['added']}))

if __name__=='__main__':main()
