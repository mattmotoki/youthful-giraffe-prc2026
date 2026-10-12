"""exp-186: mode caller of the reviewed prepare_annual.py (g annual inputs); the reviewed file stays unchanged.

annual_job.py and REVIEW-ANNUAL-SOURCE.json pin prepare_annual.py and check_annual.py by SHA-256, so this file
reads the reviewed source, checks its SHA-256, sets the one ranking-count literal (`len(ids)!=344841`) to
prc.data.n_ranking() and runs its main(). In the ranking mode the count is 344,841 and the code that runs is
byte-equal to the reviewed file. The plan of the mode comes from the board plan (ANNUAL-PLAN.json):
the training sources by file name in prc.data.data_dir() with the board SHA-256 values, ranking_source =
prc.data.ranking_file(), template_source = official_source = prc.data.submitting_file() (prepare_annual.py
asserts that the template and official ID orders are equal), ranking_rows and ranking_ID_bytes_sha256 of the
mode's template. Ranking mode: the ranking and submitting files and the template ID bytes must be the board's.

    PRC_RANKING_MODE=final python code/prepare_annual_186.py --board-plan ANNUAL-PLAN.json \
        --plan-out <plan.json> --out <dir> [--compare <saved annual inputs>]

--compare names saved annual inputs (MANIFEST.json and arrays). Ranking mode: every file hash of the new
MANIFEST must equal the saved one. Final mode: the equal and the changed files are a record (the 2026 event
rows change; the train-only encoder and the loss labels must stay equal). The caller record CALLER-186.json
goes next to the inputs; MANIFEST.json does not list it.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT)]
from prc import data as D  # noqa: E402

REVIEWED = {'prepare_annual.py': '5e6b21c7ce371df5ddbeeb8789b459adea9f139e181e16363b70b598cdf9659a',
            'check_annual.py': 'eb8fc05cbfdbd4542b67296e5e578daf6cc394f8a980b4e5a0f967d836f4c58d'}
LITERAL = 'len(ids)!=344841'
SAME_IN_FINAL = ['encoder-annual.json', 'train-labels.npy']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reviewed(name):
    """The reviewed source `name` as a module, with the ranking count of the mode; and its record."""
    path = HERE / name
    text = path.read_bytes()
    if hashlib.sha256(text).hexdigest() != REVIEWED[name]:
        raise ValueError(f'{name} differs from the reviewed source (REVIEW-ANNUAL-SOURCE.json)')
    review = json.loads((HERE.parent / 'REVIEW-ANNUAL-SOURCE.json').read_text())
    if review['source_sha256'][name] != REVIEWED[name]:
        raise ValueError(f'REVIEW-ANNUAL-SOURCE.json names another {name}')
    source = text.decode()
    if source.count(LITERAL) != 1:
        raise ValueError(f'{name}: the ranking-count literal is not unique')
    n = D.n_ranking()
    source = source.replace(LITERAL, f'len(ids)!={n}')
    namespace = {'__name__': name[:-3] + '_186', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    record = {'source': str(path), 'reviewed_sha256': REVIEWED[name], 'literal': LITERAL,
              'ranking_count': n, 'code_equals_reviewed': source.encode() == text}
    return namespace, record


def ids_sha(ids):
    return hashlib.sha256(ids.tobytes()).hexdigest()


def mode_plan(board, board_path):
    mode = D.ranking_mode()
    spec = D.ranking_spec()
    raw = D.data_dir()
    training = {str(raw / Path(p).name): h for p, h in board['training_sources_sha256'].items()}
    ranking, official = D.ranking_file(), D.submitting_file()
    rank_sha, official_sha = sha(ranking), sha(official)
    if rank_sha != spec.ranking_sha256 or official_sha != spec.submitting_sha256:
        raise ValueError('ranking or submitting file differs from prc.data.ranking_spec()')
    ids = D.load_submitting()[D.ID].to_numpy('int64')
    if len(ids) != D.n_ranking():
        raise ValueError('template rows differ from prc.data.n_ranking()')
    if mode == 'ranking':
        if rank_sha != board['sources_sha256'][board['ranking_source']] \
                or official_sha != board['sources_sha256'][board['official_source']]:
            raise ValueError('ranking mode: ranking or submitting file differs from the board plan')
        if len(ids) != board['ranking_rows'] or ids_sha(ids) != board['ranking_ID_bytes_sha256']:
            raise ValueError('ranking mode: template ID order differs from the board plan')
    plan = dict(board)
    plan.update({'training_sources_sha256': training,
                 'sources_sha256': {**training, str(ranking): rank_sha, str(official): official_sha},
                 'ranking_source': str(ranking), 'template_source': str(official), 'official_source': str(official),
                 'ranking_rows': int(len(ids)), 'ranking_ID_bytes_sha256': ids_sha(ids),
                 'exp186': {'ranking_mode': mode, 'board_plan': str(board_path), 'board_plan_sha256': sha(board_path),
                            'board_template_source': board['template_source'],
                            'caller': str(Path(__file__).resolve()), 'caller_sha256': sha(__file__)}})
    return plan


def compare(new, saved, strict):
    old = json.loads((saved / 'MANIFEST.json').read_text())
    for name, want in old['files'].items():
        if sha(saved / name) != want:
            raise ValueError('saved annual input differs from its MANIFEST: ' + name)
    mine = json.loads((new / 'MANIFEST.json').read_text())['files']
    equal = sorted(n for n in old['files'] if mine.get(n) == old['files'][n])
    changed = sorted(n for n in set(old['files']) | set(mine) if n not in equal)
    rec = {'saved': str(saved), 'saved_manifest_sha256': sha(saved / 'MANIFEST.json'),
           'equal_files': equal, 'changed_files': changed}
    if strict and changed:
        raise ValueError(f'ranking mode: annual inputs differ from the saved inputs: {changed}')
    if not strict:
        if not set(SAME_IN_FINAL) <= set(equal):
            raise ValueError(f'final mode: {SAME_IN_FINAL} must equal the saved inputs (train-only encoder, 2025 labels)')
        rec['event_rows'] = event_rows(new, saved)
    return rec


def event_rows(new, saved):
    """Encoded rows of the events that both inputs hold, by ID: 2025 rows must be equal; 2026 rows are a record."""
    import numpy as np
    import pandas as pd
    cols = ['MVT_ID_mvt', 'PHASE_mvt', 'encoder_train']
    a, b = (pd.read_parquet(p / 'events.parquet', columns=cols) for p in [new, saved])
    pos = pd.Index(b.MVT_ID_mvt.to_numpy('int64')).get_indexer(a.MVT_ID_mvt.to_numpy('int64'))
    hit = pos >= 0
    equal = np.ones(len(a), dtype=bool)
    for name in ['numeric-annual', 'category-annual', 'anchor']:
        x = np.load(new / (name + '.npy'), mmap_mode='r')
        y = np.load(saved / (name + '.npy'), mmap_mode='r')
        for lo in range(0, len(a), 500000):
            sl = slice(lo, lo + 500000)
            h = hit[sl]
            e = np.zeros(h.shape, dtype=bool)
            e[h] = (np.asarray(x[sl])[h] == np.asarray(y)[pos[sl][h]]).reshape(int(h.sum()), -1).all(axis=1)
            equal[sl] &= e
    train = a.encoder_train.to_numpy()
    rec = {'events': int(len(a)), 'saved_events': int(len(b)), 'shared_events': int(hit.sum()),
           '2025_events': int(train.sum()), '2025_shared': int((train & hit).sum()),
           '2025_equal': int((train & hit & equal).sum()),
           '2026_events': int((~train).sum()), '2026_shared': int((~train & hit).sum()),
           '2026_shared_equal': int((~train & hit & equal).sum())}
    if rec['2025_shared'] != rec['2025_events'] or rec['2025_equal'] != rec['2025_events'] \
            or int(b.encoder_train.sum()) != rec['2025_events']:
        raise ValueError(f'final mode: 2025 event rows differ from the saved inputs: {rec}')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board-plan', type=Path, required=True)
    ap.add_argument('--plan-out', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compare', type=Path, help='saved annual inputs of the board plan')
    a = ap.parse_args()
    if a.plan_out.exists() or a.out.exists():
        raise ValueError('Write-once output exists')
    mode = D.ranking_mode()
    board = json.loads(a.board_plan.read_text())
    plan = mode_plan(board, a.board_plan)
    a.plan_out.parent.mkdir(parents=True, exist_ok=True)
    a.plan_out.write_text(json.dumps(plan, indent=2) + '\n')
    module, record = reviewed('prepare_annual.py')
    sys.argv = [record['source'], '--plan', str(a.plan_out), '--out', str(a.out)]
    module['main']()
    out = {'what': 'g annual inputs of the mode through the reviewed prepare_annual.py (exp-186)',
           'ranking_mode': mode, 'reviewed_source': record, 'plan': str(a.plan_out), 'plan_sha256': sha(a.plan_out),
           'manifest_sha256': sha(a.out / 'MANIFEST.json'), 'caller_sha256': sha(__file__),
           'parity': compare(a.out, a.compare, mode == 'ranking') if a.compare else None}
    (a.out / 'CALLER-186.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ['ranking_mode', 'manifest_sha256', 'parity']}), flush=True)


if __name__ == '__main__':
    main()
