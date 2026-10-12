"""Find accidental repeated words and phrases on rewritten source lines."""
import re

DOUBLED = re.compile(
    r"\b(?P<phrase>[A-Za-z][A-Za-z0-9_'-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'-]*){0,5})"
    r"(?:[ \t]+|[ \t]*/[ \t]*)(?P=phrase)\b", re.I)


def doubled_phrases(line):
    return [match.group('phrase') for match in DOUBLED.finditer(line)]


def inspect_rewritten(root, inventory):
    issues = []
    for record in inventory['files']:
        path = root / record['path']
        if not path.is_file():
            issues.append((record['path'], 0, 'rewrite source is missing'))
            continue
        lines = path.read_text().splitlines()
        for number in record['lines']:
            if number < 1 or number > len(lines):
                issues.append((record['path'], number, 'rewrite scope is stale'))
                continue
            expected = record.get('expected_new_text', {}).get(str(number))
            if expected is None:
                issues.append((record['path'], number, 'expected rewrite text is missing'))
            elif lines[number - 1] != expected:
                issues.append((record['path'], number, 'rewritten text differs from the reviewed text'))
            issues.extend((record['path'], number, phrase)
                          for phrase in doubled_phrases(lines[number - 1]))
    return issues


def line_hashes(text):
    """Keep full text, including comments, docstrings and line endings, without publishing it."""
    import hashlib
    return [hashlib.sha256(line.encode('utf-8')).hexdigest()
            for line in text.splitlines(keepends=True)]


def changed_import_lines(original_hashes, current_hashes):
    """Every added/replaced line; a deletion also marks its surviving boundary."""
    import difflib
    numbers = set()
    for kind, a, b, c, d in difflib.SequenceMatcher(
            None, original_hashes, current_hashes, autojunk=False).get_opcodes():
        if kind == 'equal':
            continue
        if c < d:
            numbers.update(range(c + 1, d + 1))
        else:
            numbers.add(min(c + 1, len(current_hashes)))
    return numbers


def inspect_imported(root, contract, inventory):
    """Fail if any imported text change lacks damage-test coverage."""
    from pathlib import Path
    root = Path(root)
    actual = {'prc/foldserve.py'} | {p for p in ('chain_ids.py',) if (root / p).is_file()}
    for folder in ('experiments/exp-147-foldserve/code', 'experiments/exp-148-short-route/code',
                   'experiments/exp-151-published-means/code',
                   # zabd lineage: assembly, frame builders, member fits and rule sources
                   'experiments/exp-159-clock-head/code', 'experiments/exp-160-feature-worker/code',
                   'experiments/exp-164-double-down/code', 'experiments/exp-165-levers/checks',
                   'experiments/exp-167-target/code', 'experiments/exp-168-fresh-look/code',
                   'experiments/exp-169-stack-plan/code', 'experiments/exp-170-diverse/code',
                   'experiments/exp-171-rules/code', 'experiments/exp-181-modela-bag-read/serve-packet/code',
                   # v73 steps: the c5 steps and the L2C and ORD-S steps, from the closure commit
                   'iso', 'w2'):
        actual.update(p.relative_to(root).as_posix() for p in (root / folder).rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.sh'))
    records = {item['path']: item for item in contract['files']}
    scope = {item['path']: set(item['lines']) for item in inventory['files']}
    issues = []
    if set(records) != actual or len(records) != len(contract['files']):
        issues.append(('release/IMPORTED-SOURCES.json', 0, 'imported file set is incomplete or duplicated'))
    for rel, item in records.items():
        path = root / rel
        if not path.is_file():
            issues.append((rel, 0, 'imported file is missing'))
            continue
        text = path.read_bytes().decode('utf-8')
        hashes = line_hashes(text)
        if not hashes:
            issues.append((rel, 0, 'imported source is empty'))
        if rel not in scope:
            issues.append((rel, 0, 'imported file has no rewrite scope'))
        for number in sorted(changed_import_lines(item['original_line_sha256'], hashes) - scope.get(rel, set())):
            issues.append((rel, number, 'imported text changed outside the damage-test scope'))
    return issues
