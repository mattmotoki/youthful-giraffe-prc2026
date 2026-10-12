"""Replay a final export copy as its consumers do (exp-192, 2026-10-08): `release/assembly_replay.assemble` of the
copy, with the copy's patched exp-169 code, on the copy's `release/ZABD-PACK.json`, in the final mode. The parent,
C65 and candidate bytes must equal the pack's recorded hashes. This is the check that finalv's iso/chain.py can
load the pack (it calls the same rebuild_frame, make_reader and load).

Run: PRC_RANKING_MODE=final OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -I code/pack_check.py <copy root>
       <input root with data/raw/final_ranking.parquet, final_submitting.parquet and the 2025 Jan/Jul files> <out json>
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

copy, data_root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
assert os.environ.get('PRC_RANKING_MODE') == 'final'
sys.path.insert(0, str(copy / 'release'))
import assembly_replay as R  # noqa: E402

assert Path(R.__file__).resolve().parent == copy / 'release' and R.RANKING_MODE == 'final'
pack = json.loads((copy / 'release/ZABD-PACK.json').read_text())
t0 = time.time()
res = R.assemble(pack, data_root)
A, _ = R._assembly_modules()
assert Path(A.__file__).resolve().parent == copy / 'experiments/exp-169-stack-plan/code'
h = {k: hashlib.sha256(res[b]).hexdigest() for k, b in
     (('parent', 'parent_body'), ('with_c65', 'with_c65_body'), ('candidate', 'body'))}
want = {'parent': pack['parent_sha256'], 'with_c65': pack['with_c65_sha256'], 'candidate': pack['candidate_sha256']}
rec = {'copy': str(copy), 'pack_sha256': R.sha256_file(copy / 'release/ZABD-PACK.json'), 'target': pack['target'],
       'rows': pack['rows'], 'hashes': h, 'equal_pack': {k: h[k] == want[k] for k in h},
       'seconds': round(time.time() - t0, 1)}
out.write_text(json.dumps(rec, indent=1) + '\n')
print(json.dumps(rec, indent=1))
assert all(rec['equal_pack'].values()), rec['equal_pack']
