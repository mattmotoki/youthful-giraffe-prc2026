"""Make a final export copy of the zabd release export (exp-192, finalplan, 2026-10-08).

The copy is the source export tree with four overlays (PROVENANCE.md):
1. experiments/exp-169-stack-plan/code/: the five files of finalcore's E-final-months.patch (this folder's export/);
2. release/assembly_replay.py: the final-mode file names under PRC_RANKING_MODE=final (this folder's export/);
3. prc/data.py: main's file with the ranking and final modes (final_months169 reads it from the copy's root);
4. release/ZABD-PACK.json is moved to release/ZABD-PACK-ranking.json; assemble192.py --pack-out writes the final
   pack as release/ZABD-PACK.json, so that iso/chain.py and t11b_stage.py read it under their released name.
Every other file is byte-equal to the source (checked). The record is <dest>/FINAL-COPY.json.

Run: .venv/bin/python -I code/export_copy.py <source export> <dest (new)> <repo root>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

src, dest, repo = (Path(a).resolve() for a in sys.argv[1:4])
HERE = Path(__file__).resolve().parents[1]
assert (src / 'release/ZABD-PACK.json').is_file() and not dest.exists()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


shutil.copytree(src, dest, symlinks=True, ignore=shutil.ignore_patterns('__pycache__'))
overlays = {}
code = 'experiments/exp-169-stack-plan/code'
for p in sorted((HERE / 'export' / code).glob('*.py')):
    target = dest / code / p.name
    before = sha(target) if target.exists() else None
    if before != sha(p):
        overlays[f'{code}/{p.name}'] = {'from': str(p.relative_to(repo)), 'before': before, 'after': sha(p)}
    shutil.copy2(p, target)
for rel, source in [('release/assembly_replay.py', HERE / 'export/release/assembly_replay.py'),
                    ('prc/data.py', repo / 'prc/data.py')]:
    overlays[rel] = {'from': str(source.relative_to(repo)), 'before': sha(dest / rel), 'after': sha(source)}
    shutil.copy2(source, dest / rel)
(dest / 'release/ZABD-PACK.json').rename(dest / 'release/ZABD-PACK-ranking.json')
moved = {'release/ZABD-PACK.json': 'release/ZABD-PACK-ranking.json'}
# Every other file is byte-equal to the source.
same = 0
for p in src.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        rel = str(p.relative_to(src))
        if rel in overlays or rel in moved:
            continue
        assert sha(dest / rel) == sha(p), rel
        same += 1
assert set(o for o in overlays if o.startswith(code)) == {f'{code}/{n}' for n in (
    'assemble_final_032.py', 'dark_twin035.py', 'lane_c_swap.py', 'route_swap.py', 'final_months169.py')}, overlays
record = {'source': str(src), 'overlays': overlays, 'moved': moved, 'unchanged_files': same,
          'note': 'release/ZABD-PACK.json is written by assemble192.py --pack-out'}
(dest / 'FINAL-COPY.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record, indent=1))
