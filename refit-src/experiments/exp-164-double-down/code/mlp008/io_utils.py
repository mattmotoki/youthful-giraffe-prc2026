"""Own unit-007 records; peer paths are read only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

W = Path(os.environ.get('EXP164_WORK', os.path.join(os.environ.get('PRC_COMP_ROOT', '/path/to/competitions'), 'prc-exp164-work')))
ID = 'MVT_ID_mvt'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value, overwrite=False):
    p = Path(p)
    if p.exists() and not overwrite:
        raise RuntimeError(f'{p}: write-once')
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + f'.tmp{os.getpid()}')
    t.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    t.replace(p)

def artifact(p, **facts):
    return {'path': str(p), 'sha256': sha(p), **facts}

def cpu_list(text):
    """CPU list syntax of taskset -c: '8-11,72-75' gives {8, 9, 10, 11, 72, 73, 74, 75}."""
    cpus = set()
    for part in text.split(','):
        lo, dash, hi = part.strip().partition('-')
        if not lo.isdigit() or (dash and not hi.isdigit()) or int(hi or lo) < int(lo):
            raise RuntimeError(f'EXP164_CPUS: bad CPU list {text!r}')
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus

def lane(threads=4):
    affinity = set(os.sched_getaffinity(0))
    # exp-186: EXP164_CPUS names a later grant ('8-11,72-75'); unset, the unit-007 lane
    allowed = cpu_list(os.environ['EXP164_CPUS']) if 'EXP164_CPUS' in os.environ else {24, 25, 26, 27, 88, 89, 90, 91}
    if not affinity or not affinity <= allowed or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError('unit 007 CPU lane or nice differs')
    for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if not 1 <= int(os.environ.get(k, '0')) <= threads:
            raise RuntimeError(f'{k}: thread cap differs')
    return sorted(affinity)
