"""Final sources JSON from the producers' MANIFEST rows (exp-192, finalplan, 2026-10-09).

The build (build_final.sh) reads each part by path, SHA-256 and column from a sources JSON. This script writes that
JSON from the MANIFEST.tsv rows of finalcore (exp-186), finallirf (exp-187), finalroute (exp-189) and finalclock
(exp-190) at one git ref, so no path or hash is copied by hand. The columns that need a part are the pack's saved
ranking columns (ZABD-PACK.json) less the rule columns that the driver builds; mlp comes from finalcore's mlp part
(columns mlp008 and mix12). A MANIFEST is append-only, so the last row of a part name is the current one.

Writes <out> (the sources JSON) and <out stem>-from.json (ref, commit, the MANIFEST row of each source, the columns
still without a part). Without --allow-missing, every needed column must have a part (--r-only: every column of the R
chain, so the members and mlp may be missing).

Run (Mac): .venv/bin/python -I code/final_sources.py <repo> <ZABD-PACK.json> <out json> [--ref origin/main]
           [--allow-missing] [--r-only] [--lirf-tag TAG | --lirf-tag bundle=T1,rule_frame=T2,exploit_frame=T3]
"""
import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble192 import ID, MEMBER_COLUMNS, RULE_COLUMNS  # noqa: E402

MEMBERS = ('g', 'row6', 'row011', 'sequence011', 'sequence021', 'dark18', 'sequence039')
FRAMES = ('bundle', 'rule_frame', 'exploit_frame')
#: Columns whose MANIFEST field is a count, not names: the column is the one the producer's part row describes.
NAMED = {('exp-190-final-clock', 'head'): ['pred_HB3u']}


def manifest(repo, ref, exp):
    """Rows of one MANIFEST.tsv at the ref: {part name: (line number, row dict)}, the last row of a name wins."""
    text = subprocess.run(['git', '-C', str(repo), 'show', f'{ref}:experiments/{exp}/MANIFEST.tsv'],
                          capture_output=True, text=True, check=True).stdout
    lines = text.rstrip('\n').split('\n')
    head = lines[0].split('\t')
    assert head[:3] == ['part', 'remote_path', 'sha256'] and 'columns' in head, (exp, head)
    rows = {}
    for n, line in enumerate(lines[1:], start=2):
        r = dict(zip(head, line.split('\t')))
        rows[r['part'].split(' (')[0].strip()] = (n, r)
    return rows


def columns_of(exp, name, row):
    """The value columns of a part row: from the names in the columns field, else from NAMED."""
    f = row['columns']
    if ',' in f or not f.isdigit():
        return [c for c in f.split(',') if c != ID]
    assert (exp, name) in NAMED, f'{exp} {name}: columns field is a count ({f}); name the column in NAMED'
    return NAMED[(exp, name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo', type=Path)
    ap.add_argument('pack', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--ref', default='origin/main')
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--r-only', action='store_true')
    ap.add_argument('--lirf-tag', default=None, help='tag of frames_<key>-<tag> when finallirf has more than one: '
                    'one tag for all three frames, or key=tag pairs (bundle=npee84b9f3-a444,rule_frame=npee84b9f3,...)')
    a = ap.parse_args()
    commit = subprocess.run(['git', '-C', str(a.repo), 'rev-parse', a.ref], capture_output=True, text=True,
                            check=True).stdout.strip()
    pack = json.loads(a.pack.read_text())
    need = {(k, c['name']) for k, s in pack['frames'].items() if s['split'] == 'ranking' for c in s['columns']
            if c['kind'] == 'saved' and c['name'] not in RULE_COLUMNS.get(k, [])}
    assert ('ranking/primary_frame', 'mlp') in need

    # (sources key, MANIFEST experiment, part name)
    plan = [(f'ranking/vector/{m}', 'exp-186-final-core', m) for m in MEMBERS]
    plan += [('ranking/vector/head', 'exp-190-final-clock', 'head'), ('ranking/vector/route', 'exp-189-final-route',
             'route'), ('ranking/rule_frame', 'exp-189-final-route', 'F'), ('mlp_part', 'exp-186-final-core', 'mlp')]
    rows = {e: manifest(a.repo, commit, e) for e in sorted({e for _, e, _ in plan} | {'exp-187-final-lirf'})}
    lirf = rows['exp-187-final-lirf']
    tags = dict(t.split('=') for t in a.lirf_tag.split(',')) if a.lirf_tag and '=' in a.lirf_tag else \
        {f: a.lirf_tag for f in FRAMES}
    assert set(tags) == set(FRAMES), tags
    for f in FRAMES:
        names = sorted(n for n in lirf if n.startswith(f'frames_{f}-') and (tags[f] is None or
                                                                             n == f'frames_{f}-{tags[f]}'))
        assert len(names) <= 1, f'finallirf has {names}; pass --lirf-tag'
        plan += [(f'ranking/{f}', 'exp-187-final-lirf', names[0])] if names else []

    sources, origin, ignored = {}, {}, {}
    for key, exp, name in plan:
        if name not in rows[exp]:
            continue
        n, r = rows[exp][name]
        cols = columns_of(exp, name, r)
        if key == 'mlp_part':
            want = ['mlp008', 'mix12']
            assert set(want) <= set(cols), (exp, name, cols)
            sources[key] = {'path': r['remote_path'], 'sha256': r['sha256']}
            got = [('ranking/primary_frame', 'mlp')]
        else:
            # The part gives every needed column of its key; other columns of the part (route_source) are not read.
            # F is finalroute's part; finallirf's rule frame gives the other six (finallirf REPORT-V section 1).
            want = sorted(c for k, c in need if k == key and (c == 'F') == (name == 'F'))
            assert want and set(want) <= set(cols), f'{exp} {name}: needs {want}, has {cols}'
            for c in want:
                assert c not in sources.get(key, {}), f'two parts for {key}:{c}'
                sources.setdefault(key, {})[c] = {'path': r['remote_path'], 'sha256': r['sha256'], 'column': c}
            got = [(key, c) for c in want]
        if set(cols) - set(want):
            ignored[f'{exp}:{name}'] = sorted(set(cols) - set(want))
        assert len(r['sha256']) == 64 and r['remote_path'].startswith(os.environ.get('PRC_FINAL_ROOT', '/path/to/final') + '/'), r
        for k, c in got:
            origin[f'{k}:{c}'] = {'manifest': f'experiments/{exp}/MANIFEST.tsv', 'line': n, 'part': name,
                                  'path': r['remote_path'], 'sha256': r['sha256']}

    missing = sorted(f'{k}:{c}' for k, c in need if f'{k}:{c}' not in origin)
    optional = {f'{k}:{c}' for k, c in MEMBER_COLUMNS} if a.r_only else set()
    blocking = [m for m in missing if m not in optional]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    assert not a.out.exists(), f'{a.out} exists (write-once)'
    a.out.write_text(json.dumps(sources, indent=1) + '\n')
    rec = {'ref': a.ref, 'commit': commit, 'pack': str(a.pack), 'r_only': a.r_only, 'sources': str(a.out),
           'needed': len(need), 'with_part': len(origin), 'missing': missing, 'ignored': ignored, 'origin': origin}
    a.out.with_name(a.out.stem + '-from.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'origin'}, indent=1))
    assert a.allow_missing or not blocking, f'no part yet for {blocking}'


if __name__ == '__main__':
    main()
