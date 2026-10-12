"""exp-186: the harness record of the final mode (EXP164_HARNESS) for prepare.py, from a git archive of main.

prepare.py pins each file of the record's `files` against the EXP153_CODE tree. The saved record
(results/HARNESS.json, b79ba281) pins three files of the board tree. This script reads the exp-153 code tree
of a named commit as `git archive` writes it (the form of the remote snapshot) and writes:
  files              the saved file names with their SHA-256 in that tree (what prepare.py checks);
  tree_sha256        the SHA-256 of every file in that tree (a record);
  changed            each pinned file whose digest differs from the saved record, with the commits that changed
                     it after the saved version (git log of the file, newest back to the saved blob).
No file outside --out is written.

    python code/mlp008/harness_final186.py --saved <results/HARNESS.json copy> --commit <c> \
        --tree-path $PRC_FINAL_ROOT/finalcore/src/<c>/experiments/exp-153-grind/code --out <json>
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

TREE = 'experiments/exp-153-grind/code'
REPO = Path(__file__).resolve().parents[4]


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True).stdout


def blob_sha(commit, path):
    return hashlib.sha256(git('show', f'{commit}:{path}')).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved', type=Path, required=True)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--tree-path', required=True, help='EXP153_CODE of the final run (the snapshot tree)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out}: write-once')
    commit = git('rev-parse', '--verify', a.commit + '^{commit}').decode().strip()
    saved = json.loads(a.saved.read_text())
    tree = {}
    with tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', commit, TREE))) as t:
        for m in t.getmembers():
            if m.isfile():
                tree[m.name[len(TREE) + 1:]] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    missing = [n for n in saved['files'] if n not in tree]
    if missing:
        raise SystemExit(f'pinned files absent from {TREE} at {commit[:9]}: {missing}')
    changed = []
    for n, old in sorted(saved['files'].items()):
        if tree[n] == old:
            continue
        path = f'{TREE}/{n}'; by = []
        for c in git('log', '--format=%H', commit, '--', path).decode().split():
            if blob_sha(c, path) == old:
                break
            by.append({'commit': c, 'subject': git('log', '-1', '--format=%s', c).decode().strip()[:200]})
        else:
            raise SystemExit(f'{n}: no commit holds the saved digest {old[:12]}')
        changed.append({'file': n, 'saved_sha256': old, 'final_sha256': tree[n], 'changed_by': by[::-1]})
    rec = {'files': {n: tree[n] for n in sorted(saved['files'])}, 'read_only_harness': a.tree_path,
           'source_spec': saved.get('source_spec'),
           'what': 'exp-186 final-mode harness record: the saved file list with the digests of the exp-153 tree at commit',
           'commit': commit, 'tree': TREE, 'tree_files': len(tree), 'tree_sha256': dict(sorted(tree.items())),
           'saved_record': {'path': str(a.saved), 'sha256': hashlib.sha256(a.saved.read_bytes()).hexdigest(),
                            'files': saved['files'], 'read_only_harness': saved.get('read_only_harness')},
           'changed': changed, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'harness-record', 'out': str(a.out), 'commit': commit[:9], 'files': rec['files'],
                      'changed': [(c['file'], [b['commit'][:9] for b in c['changed_by']]) for c in changed]}))


if __name__ == '__main__':
    main()
