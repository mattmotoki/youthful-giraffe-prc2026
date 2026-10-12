"""Whole-input oracle and chunk equivalence controls, using synthetic text only."""
from __future__ import annotations

import io
import itertools
import sys
import unittest

from check_release import scan_readable
from content_policy import (
    DETAIL_COLUMNS, ID_COLUMNS, TARGET_COLUMNS, KEYED_MOVEMENT, KEYED_GAP_LIMIT,
    MOVEMENT_DETAIL, MOVEMENT_DETAIL_NAMES, MOVEMENT_FIELD, MOVEMENT_FIELD_NAMES,
    MOVEMENT_FREE_TEXT_ALLOWED, MOVEMENT_ID, MOVEMENT_ID_MIN, MOVEMENT_KEY,
    MOVEMENT_RELATION_WINDOW, MovementScanner, canonical_movement_id,
)


def reference(text, rel='fixture.txt', suffix=''):
    """Trusted small oracle: whole-input spans, no scanner or chunk operations."""
    matches = list(MOVEMENT_ID.finditer(text))
    ids = []
    for match in matches:
        value = canonical_movement_id(match.group())
        if value not in ids and len(ids) < MOVEMENT_ID_MIN:
            ids.append(value)
    keys = list(MOVEMENT_KEY.finditer(text))
    details = list(MOVEMENT_DETAIL.finditer(text))
    structured = bool(KEYED_MOVEMENT.search(text))
    for match in matches:
        lo = match.start() - MOVEMENT_RELATION_WINDOW
        hi = match.end() + MOVEMENT_RELATION_WINDOW
        structured |= (any(lo <= key.start() and key.end() <= hi for key in keys)
                       and any(lo <= detail.start() and detail.end() <= hi for detail in details))
    columns = set()
    if suffix in ('.csv', '.tsv'):
        delimiter = '\t' if suffix == '.tsv' else ','
        columns = {cell.strip().strip('"') for cell in text.split('\n', 1)[0].split(delimiter)}
    header_ids = columns & ID_COLUMNS
    table = sorted(header_ids | (columns & TARGET_COLUMNS) |
                   (columns & DETAIL_COLUMNS if header_ids else set()))
    field = bool(MOVEMENT_FIELD.search(text))
    return dict(ids=sorted(ids), field=field, structured=structured,
                free=rel not in MOVEMENT_FREE_TEXT_ALLOWED and field and len(ids) >= MOVEMENT_ID_MIN,
                columns=table)


def reference_findings(result, rel):
    findings = []
    if result['columns']:
        findings.append(dict(path=rel, columns=result['columns'], shape='table'))
    if result['free']:
        findings.append(dict(path=rel, ids=len(result['ids']), shape='free'))
    if result['structured']:
        findings.append(dict(path=rel, shape='structured'))
    return findings


def corpus():
    row = '200' + '000001'
    cases = []
    def add(name, text, suffix='', rel='fixture.txt'):
        cases.append(dict(name=name, text=text, suffix=suffix, rel=rel))
    valid = ('', '.0', '.00', '.000', '.\u0030' + '0' * 2048)
    invalid = ('.1', '.01', '0', '#fragment', 'suffix', '.', '.0.', '.0#x',
               '.0x', '.0.0', '.0' + '0' * 2048 + '1')
    for suffix in valid + invalid:
        add('id-suffix-' + str(len(cases)), 'MVT_ID_mvt: ' + row + suffix)
    for value in ('prefix' + row, '_' + row, '#' + row, '.' + row,
                  'é' + row, row + 'é', row + '_', '17' + '9999999',
                  '22' + '0000000', '18' + '0000000', '21' + '9999999',
                  row[:-1], '２００' + '０００００１'):
        add('integer-boundary-' + str(len(cases)), 'id: ' + value + ' target')
    for size in (KEYED_GAP_LIMIT - 1, KEYED_GAP_LIMIT, KEYED_GAP_LIMIT + 1):
        for whitespace, label in ((' ', 'space'), ('\n', 'newline')):
            add(f'key-gap-before-{label}-{size}', 'MVT_ID_mvt' + whitespace * size + ': ' + row)
            add(f'key-gap-after-{label}-{size}', 'MVT_ID_mvt:' + whitespace * size + row)
    for name in MOVEMENT_FIELD_NAMES:
        for affix in ('', 'X', '_x'):
            add('field-' + name + affix, name + affix + ': ' + row)
        add('prefix-field-' + name, 'prefix' + name + ': ' + row)
        add('bulk-field-' + name, name + '\n' + ' '.join('200' + f'{i:06d}' for i in range(1, MOVEMENT_ID_MIN + 1)))
    for name in MOVEMENT_DETAIL_NAMES:
        add('detail-right-' + name, 'id: ' + row + ' ' + name)
        add('detail-left-' + name, name + ' id: ' + row)
        add('detail-suffix-' + name, 'id: ' + row + ' ' + name + 'X')
        add('detail-prefix-' + name, 'prefix' + name + ' id: ' + row)
    for distance in (MOVEMENT_RELATION_WINDOW - 1, MOVEMENT_RELATION_WINDOW,
                     MOVEMENT_RELATION_WINDOW + 1):
        for name in MOVEMENT_DETAIL_NAMES:
            add(f'distance-right-{distance}-{name}', 'id: ' + row + ' ' * (distance - len(name)) + name)
            add(f'distance-left-{distance}-{name}', name + ' ' * (distance - len(name) - len('id: ')) + 'id: ' + row)
        # Key itself must also fit the complete interval.
        add(f'key-distance-right-{distance}', 'target ' + row + ' ' * (distance - len('id')) + 'id')
        add(f'key-distance-left-{distance}', 'id' + ' ' * (distance - len('id') - len('target ')) + 'target ' + row)
    for count in range(MOVEMENT_ID_MIN + 2):
        add('bulk-count-' + str(count), 'MVT_ID_mvt\n' + ' '.join('200' + f'{i:06d}' for i in range(1, count + 1)))
    add('canonical-repeats', 'MVT_ID_mvt\n' + ' '.join(row + suffix for suffix in valid))
    add('repeated-invalid', ('MVT_ID_mvt: ' + row + '.1\n') * 4)
    add('late-valid', ('MVT_ID_mvt: ' + row + '.1\n') * 4 + 'MVT_ID_mvt: ' + row)
    add('long-left', 'target id: ' + row + '.' + '0' * 4097)
    add('long-right', 'id: ' + row + '.' + '0' * 4097 + ' target')
    add('long-invalid-right', 'id: ' + row + '.' + '0' * 4097 + '1 target')
    add('long-field', 'MVT_ID_mvt' + 'x' * 2049 + ': ' + row)
    add('bare-id-suffix', 'idX: ' + row + ' target')
    add('bare-id-prefix', 'xid: ' + row + ' target')
    add('unrelated-ids', 'id ' + row + ' ' * 513 + row + ' target')
    add('adjacent-ids', 'id ' + row + ' ' + row + ' target')
    add('empty', '')
    add('only-field-eof', 'MVT_ID_mvt')
    add('only-detail-eof', 'target')
    add('only-id-eof', row)
    add('unicode-case', 'ıd: ' + row + ' target')
    for rel in sorted(MOVEMENT_FREE_TEXT_ALLOWED):
        add('exception-bulk-' + rel, 'MVT_ID_mvt\n' + ' '.join('200' + f'{i:06d}' for i in range(1, MOVEMENT_ID_MIN + 1)), rel=rel)
        add('exception-keyed-' + rel, 'MVT_ID_mvt: ' + row, rel=rel)
    for suffix in ('.csv', '.tsv'):
        sep = ',' if suffix == '.csv' else '\t'
        for name in sorted(ID_COLUMNS | TARGET_COLUMNS | DETAIL_COLUMNS):
            add('header-' + suffix + '-' + name, '"' + name + '"' + sep + 'FAKE\n', suffix)
            add('header-false-' + suffix + '-' + name, name + 'X' + sep + 'FAKE', suffix)
    return cases


SCHEDULES = ((1,), (7,), (511,), (512,), (513,), (1, 7, 2, 509, 17, 1021))


def chunks(text, schedule):
    offset = 0
    for size in itertools.cycle(schedule):
        if offset >= len(text):
            return
        yield text[offset:offset + size]
        offset += size


class ChunkPath:
    """Supply explicit decoded read chunks to the real scan_readable path."""
    def __init__(self, parts, suffix=''):
        self.parts = list(parts)
        self.suffix = suffix

    def open(self, mode, **kwargs):
        if mode == 'rb':
            return io.BytesIO(b'fake')
        parts = iter(part for part in self.parts if part)
        class Reader:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size):
                return next(parts, '')
        return Reader()


def retained_bytes(scanner):
    """Deep Python instance state, excluding shared class/module policy."""
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple, set, frozenset)):
            total += sum(map(size, value))
        return total
    return sys.getsizeof(scanner) + size(vars(scanner))


def compare(case, parts):
    expected = reference(case['text'], case['rel'], case['suffix'])
    scanner = MovementScanner(case['rel'], case['suffix'])
    maximum = retained_bytes(scanner)
    for part in parts:
        scanner.feed('')
        scanner.feed(part)
        maximum = max(maximum, retained_bytes(scanner))
    actual = scanner.finalize()
    assert actual == expected, (case['name'], actual, expected)
    before = vars(scanner).copy()
    assert scanner.finalize() == expected
    assert vars(scanner) == before
    gate = scan_readable(ChunkPath(parts, case['suffix']), case['rel'])
    assert gate == (reference_findings(expected, case['rel']), [], []), (case['name'], gate, expected)
    return maximum


class StatefulMovementTests(unittest.TestCase):
    def test_every_split_and_read_schedule(self):
        for case in corpus():
            for split in range(len(case['text']) + 1):
                compare(case, [case['text'][:split], case['text'][split:]])
            for schedule in SCHEDULES:
                compare(case, list(chunks(case['text'], schedule)))

    def test_declared_positive_negative_contract(self):
        for case in corpus():
            result = reference(case['text'], case['rel'], case['suffix'])
            name = case['name']
            if name.startswith(('distance-', 'key-distance-')):
                distance = int(name.split('-')[3]) if name.startswith('key-') else int(name.split('-')[2])
                self.assertEqual(result['structured'], distance <= MOVEMENT_RELATION_WINDOW, name)
            if name.startswith(('detail-suffix-', 'detail-prefix-', 'prefix-field-')):
                self.assertFalse(result['structured'], name)
            if name.startswith('key-gap-'):
                self.assertEqual(result['structured'], int(name.split('-')[-1]) <= KEYED_GAP_LIMIT, name)
            if name.startswith('bulk-count-'):
                self.assertEqual(result['free'], int(name.split('-')[-1]) >= MOVEMENT_ID_MIN, name)

    def test_finalize_is_idempotent_and_feed_after_close_fails(self):
        scanner = MovementScanner()
        scanner.feed('MVT_ID_mvt: ' + '200' + '000001')
        self.assertTrue(scanner.finalize()['structured'])
        with self.assertRaises(ValueError):
            scanner.feed('')

    def test_incomplete_tokens_have_no_early_finding(self):
        scanner = MovementScanner()
        scanner.feed('MVT_ID_mvt')
        self.assertFalse(scanner.field)
        scanner.feed('X: ' + '200' + '000001')
        self.assertFalse(scanner.ids)
        scanner.feed('.' + '0' * 8193)
        self.assertFalse(scanner.ids)
        scanner.feed('1 ')
        self.assertFalse(scanner.finalize()['ids'])

    def test_long_suffix_and_state_bound(self):
        scanner = MovementScanner()
        scanner.feed('target id: ' + '200' + '000001' + '.')
        for _ in range(100):
            scanner.feed('0' * 10000)
            self.assertLess(retained_bytes(scanner), 12000)
            self.assertLessEqual(len(scanner.pending), 3)
        self.assertTrue(scanner.finalize()['structured'])


if __name__ == '__main__':
    unittest.main()
