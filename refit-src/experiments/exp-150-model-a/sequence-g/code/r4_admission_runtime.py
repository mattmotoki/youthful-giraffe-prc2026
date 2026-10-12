"""Load verified upstream byte copies under the names their imports require."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import hashlib

EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parents[1]
UP = EXP/'r4/admission/upstream'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def load():
    manifest = json.loads((UP/'MANIFEST.json').read_text())
    for item in manifest['files']:
        if sha(ROOT/item['copy_path']) != item['sha256']:
            raise RuntimeError(f"Upstream changed: {item['copy_path']}")
    if sha(ROOT/'prc/weekpair.py') != manifest['weekpair_sha256']:
        raise RuntimeError('Split source differs from the pin')
    os.environ.setdefault('PRC_SOURCE_ROOT', str(ROOT))
    sys.path.insert(0, str(ROOT))
    pilot = module('pilot', UP/'pilot.py')
    st = module('stacker', UP/'exp148/code/stacker.py')
    st.ROOT = ROOT
    module('clockloss', UP/'exp148/code/clockloss.py')
    module('report_tables', UP/'exp148/code/report_tables.py')
    score = module('score_r2', UP/'exp148/code/score_r2.py')
    serve = module('exp150_pinned_serve_r2', UP/'exp148/code/serve_r2.py')
    assert st.P is pilot and score.P is pilot and serve.P is pilot
    score4 = module('exp150_pinned_score_r4', UP/'exp148/code/score_r4.py')
    score4.OWNERS['R4Am'] = ('q01', 'q07')
    for stage in ('R4', 'R4Am'):
        score.MONTH_OWNERS[stage] = (('q01', 'month-2025-01'), ('q07', 'month-2025-07'))
    return st, score, score4, serve, manifest
