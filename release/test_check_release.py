"""Synthetic adversarial content tests; strings are assembled to test the scanner itself."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_release import MOVEMENT_FREE_TEXT_ALLOWED, scan_readable
from content_policy import (MOVEMENT_DETAIL_NAMES, MOVEMENT_RELATION_WINDOW,
                            canonical_movement_id, movement_ids)


def private_samples():
    """Clearly fake identities; no real account or object is used."""
    fake = 'fake-release-user'
    route = fake + '/' + 'prc-fake-object'
    return {
        'kaggle_slug': 'Download from `' + route + '`.',
        'private_dataset_slug': 'private slug: ' + 'prc-' + '2099-' + 'data',
        'kaggle_account_name': 'Kaggle ' + 'account: ' + fake,
        'kaggle_owner': 'owner' + ': ' + fake,
        'kaggle_reference': 'dataset_ref' + ' = ' + fake + '/fake-data',
        'private_object_id': 'kernel_id' + ': ' + str(654321),
        'transport' + '_receipt': 'transport_receipt' + ': fake-complete',
        'remote' + '_custody': 'remote:' + '/home/fake-release-user/repo',
        'kaggle_private_metadata': ('is_' + 'private: true, id: ' + fake + '/fake-object'),
        'private_storage_identifier': 'private bucket: ' + 'prc-' + '2099-' + 'fake-team',
        'aws_profile_identifier': 'AWS_' + 'PROFILE=' + 'opensky-' + 'prc-' + '2099',
        'account_bound_url': 'https://' + 'chatgpt.com/c/' + 'fake-conversation',
    }


class ContentGateTests(unittest.TestCase):
    def scan(self, text, suffix='.txt', rel='fixture.txt', read_size=1 << 20):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ('sample' + suffix)
            path.write_text(text)
            return scan_readable(path, rel, read_size=read_size)

    def test_movement_id_right_boundary_is_final_before_acceptance(self):
        row = '200' + '000001'
        valid_suffixes = ('', ' ', '\n', ',', ']', '}')
        invalid_suffixes = ('.1', '0', '#fragment', 'suffix')
        prefix = 'MVT_ID_mvt: '
        for suffix in valid_suffixes + invalid_suffixes:
            text = prefix + row + suffix
            want = suffix in valid_suffixes
            # Put each read boundary before, inside, and after the token, and
            # directly after the candidate where an unseen suffix matters.
            for split in range(len(prefix) - 1, len(prefix) + len(row) + len(suffix) + 1):
                with self.subTest(suffix=suffix, split=split):
                    found = bool(self.scan(text, read_size=max(1, split))[0])
                    self.assertEqual(found, want)

    def test_every_id_character_around_internal_stable_cut(self):
        """Align each candidate character around the real len(text)-512 cut."""
        integer = '200' + '000001'
        cases = (
            (integer, True), (integer + '.0', True), (integer + '.00', True),
            (integer + '.000', True), (integer + '.1', False),
            (integer + '.01', False), (integer + '0', False),
            (integer + '#x', False), (integer + 'x', False),
            ('x' + integer, False),
        )
        read_size = 700
        cut = read_size - MOVEMENT_RELATION_WINDOW
        for value, want in cases:
            row = 'MVT_ID_mvt: ' + value
            for char_index in range(len(row)):
                for side in (-1, 0, 1):
                    prefix_size = cut + side - char_index
                    if prefix_size < 0:
                        continue
                    text = ' ' * prefix_size + row + ' ' * (MOVEMENT_RELATION_WINDOW + 3)
                    with self.subTest(value=value, char=char_index, side=side):
                        self.assertEqual(bool(self.scan(text, read_size=read_size)[0]), want)

    def test_structured_relation_uses_complete_512_character_window(self):
        row = '200' + '000001'
        detail = MOVEMENT_DETAIL_NAMES[0]
        right = 'id: ' + row + ' ' * (MOVEMENT_RELATION_WINDOW - len(detail)) + detail
        left = (detail + ' ' * (MOVEMENT_RELATION_WINDOW - len(detail) - len('id: '))
                + 'id: ' + row)
        for text in (right, left):
            for read_size in (1, 7, 511, 512, 513):
                with self.subTest(order=text.startswith('id'), read_size=read_size):
                    self.assertTrue(self.scan(text, read_size=read_size)[0])

    def test_streaming_finalization_repeated_overlap_and_far_offset(self):
        row = '200' + '000001'
        prefix = 'x' * 16_500_000 + '\nMVT_ID_mvt: '
        for suffix, want in (('', True), ('.00', True), ('.1', False),
                             ('0', False), ('#x', False), ('x', False)):
            with self.subTest(suffix=suffix):
                self.assertEqual(bool(self.scan(prefix + row + suffix,
                                                read_size=4093)[0]), want)
        repeated = ('x' * 510 + '\nMVT_ID_mvt: ' + row + '.1\n') * 4
        repeated += 'MVT_ID_mvt: ' + row + '\n'
        self.assertTrue(self.scan(repeated, read_size=511)[0])

    def test_private_classes_in_all_text_extensions(self):
        for suffix in ('.md', '.py', '.sh', '.yaml', '.json', '.csv', '.tsv', '.custom', ''):
            for name, text in private_samples().items():
                with self.subTest(suffix=suffix, name=name):
                    self.assertIn(name, self.scan(text, suffix)[2])

    def test_private_unquoted_json_source_table_and_prose(self):
        key = 'dataset_id'
        value = str(654321)
        for text in (key + ': ' + value, '"' + key + '": ' + value,
                     key + ' = ' + value, '| ' + key + ' | ' + value + ' |',
                     key.replace('_', ' ') + ' is ' + value):
            self.assertIn('private_object_id', self.scan(text)[2])

    def test_routes_after_16_mb_and_across_chunk_boundary(self):
        samples = private_samples()
        for name in ('kaggle_slug', 'private_storage_identifier', 'aws_profile_identifier'):
            for offset in (16_500_000, (1 << 20) - 15):
                with self.subTest(name=name, offset=offset), tempfile.TemporaryDirectory() as td:
                    path = Path(td) / 'large.custom'
                    with path.open('w') as stream:
                        for _ in range(offset // 1000):
                            stream.write('x' * 999 + '\n')
                        stream.write(' ' * (offset % 1000))
                        stream.write(samples[name])
                    self.assertIn(name, scan_readable(path, 'large.custom')[2])

    def test_storage_and_profile_defaults_in_shell_and_python(self):
        storage = 'prc-' + '2099-' + 'fake-team'
        profile = 'opensky-' + 'prc-' + '2099'
        samples = (
            ('private_storage_identifier', 'bucket="${DATA_BUCKET:-' + storage + '}"'),
            ('private_storage_identifier', 'DATA_BUCKET = "' + storage + '"'),
            ('private_storage_identifier', 'download_bucket: ' + storage),
            ('private_storage_identifier', 'upload_bucket=' + storage),
            ('aws_profile_identifier', 'profile="${AWS_PROFILE:-' + profile + '}"'),
            ('aws_profile_identifier', '--profile ' + profile),
        )
        for name, sample in samples:
            with self.subTest(name=name, sample=sample):
                self.assertIn(name, self.scan(sample)[2])

    def test_both_fixtures_reject_json_lists_and_tables(self):
        row = '200' + '000001'
        for rel in MOVEMENT_FREE_TEXT_ALLOWED:
            for suffix, text in (
                ('.py', '{"MVT_ID_' + 'mvt": ' + row + '}'),
                ('.json', '{"MVT_ID_' + 'mvt": [' + row + ']}'),
                ('.md', '| `MVT_ID_mvt` | ' + row + ' |'),
                ('.txt', 'MVT_ID_mvt = ' + row),
                ('.csv', 'MVT_ID_mvt,y\n' + row + ',4\n'),
                ('.tsv', 'MVT_ID_mvt\ty\n' + row + '\t4\n'),
            ):
                self.assertTrue(self.scan(text, suffix, rel)[0], (rel, suffix))
            for name, text in private_samples().items():
                self.assertIn(name, self.scan(text, rel=rel)[2])

    def test_free_text_and_markdown_rows(self):
        text = 'MVT_ID_mvt\n' + '\n'.join('200' + f'{i:06d}' for i in range(1, 4))
        self.assertTrue(self.scan(text)[0])
        self.assertTrue(self.scan('| ' + text.replace('\n', ' |\n| ') + ' |')[0])
        for rel in MOVEMENT_FREE_TEXT_ALLOWED:
            self.assertFalse(self.scan(text, rel=rel)[0])

    def test_integer_and_decimal_rows_in_structured_forms(self):
        row = '200' + '000001'
        samples = (
            ('.json', '{"MVT_ID_mvt": ' + row + '.0, "target": 4}'),
            ('.py', "{'mvt_id': " + row + ".0, 'airport': 'FAKE'}"),
            ('.md', '| MVT_ID_mvt | callsign |\n|---|---|\n| ' + row + '.0 | FAKE1 |'),
            ('.csv', 'MVT_ID_mvt,AIRCRAFT_TYPE_mvt\n' + row + '.0,FAKE'),
            ('.txt', 'row_ids = [' + row + '.0]'),
        )
        for suffix, text in samples:
            with self.subTest(suffix=suffix):
                self.assertTrue(self.scan(text, suffix)[0])

    def test_decimal_rows_after_16_mb_and_across_chunk_boundary(self):
        marker = 'MVT_ID_' + 'mvt = 200' + '000001.0'
        for offset in (16_500_000, (1 << 20) - 12):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'large.custom'
                with path.open('w') as stream:
                    stream.write('x' * offset)
                    stream.write('\n' + marker)
                self.assertTrue(scan_readable(path, 'large.custom')[0])

    def test_bare_id_with_each_detail_class_and_both_orders(self):
        row = '200' + '000001'
        details = ('target', 'y', 'y_true', 'callsign', 'flight', 'airport',
                   'stand', 'runway', 'aircraft_type')
        forms = (
            lambda detail: 'id: ' + row + '.00\n' + detail + ': FAKE',
            lambda detail: detail + ': FAKE\n' + 'id: ' + row,
            lambda detail: '{"id": ' + row + ', "' + detail + '": "FAKE"}',
            lambda detail: "{'" + detail + "': 'FAKE', 'id': " + row + "}",
            lambda detail: '| id | ' + detail + ' |\n|---|---|\n| ' + row + ' | FAKE |',
            lambda detail: 'id,' + detail + '\n' + row + ',FAKE',
        )
        for detail in details:
            for index, form in enumerate(forms):
                with self.subTest(detail=detail, form=index):
                    self.assertTrue(self.scan(form(detail))[0])

    def test_bare_id_bounded_split_and_after_16_mb(self):
        row = '200' + '000001'
        for prefix in ('x' * ((1 << 20) - 8), 'x' * 16_500_000):
            self.assertTrue(self.scan(prefix + '\ntarget: FAKE\nid: ' + row)[0])

    def test_canonical_spellings_share_one_distinct_id(self):
        row = '200' + '000001'
        text = 'MVT_ID_mvt\n' + row + ' ' + row + '.0 ' + row + '.00'
        self.assertEqual(movement_ids(text), {row})
        self.assertFalse(self.scan(text)[0])
        for spelling in (row, row + '.0', row + '.00'):
            self.assertEqual(canonical_movement_id(spelling), row)

    def test_invalid_id_spellings_are_not_ids(self):
        row = '200' + '000001'
        for value in (row + '.1', row + '0', row + '#fragment',
                      'prefix' + row, row + 'suffix'):
            with self.subTest(value=value):
                self.assertIsNone(canonical_movement_id(value))
                self.assertEqual(movement_ids('MVT_ID_mvt: ' + value), set())

    def test_credentials_still_run_on_fixtures(self):
        text = 'AKIA' + 'Z' * 16
        for rel in MOVEMENT_FREE_TEXT_ALLOWED:
            self.assertIn('aws_access_key_id', self.scan(text, rel=rel)[1])

    def test_missing_file_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'cannot scan'):
            scan_readable(Path('/nonexistent-exp070-test-file'), 'missing')

def png(*chunks, signature=b'\x89PNG\r\n\x1a\n', tail=b''):
    """A small PNG built from (type, data) pairs; a third item replaces the CRC."""
    import zlib
    out = signature
    for chunk in chunks:
        kind, data = chunk[0].encode('ascii'), chunk[1]
        crc = chunk[2] if len(chunk) > 2 else zlib.crc32(kind + data).to_bytes(4, 'big')
        out += len(data).to_bytes(4, 'big') + kind + data + crc
    return out + tail


def one_pixel(*middle, header=None, idat=None):
    """A one-pixel grey image: IHDR, the chunks given, one IDAT, IEND."""
    import zlib
    header = header or (1).to_bytes(4, 'big') * 2 + bytes([8, 0, 0, 0, 0])
    idat = idat or (('IDAT', zlib.compress(b'\x00\x00')),)
    return (('IHDR', header),) + middle + tuple(idat) + (('IEND', b''),)


class FigureGateTests(unittest.TestCase):
    """PNG figures in docs/writeup/figures/ are parsed; every other binary fails closed."""
    REL = 'docs/writeup/figures/01-sample.png'

    def scan(self, data, rel=REL):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sample.png'
            path.write_bytes(data)
            return scan_readable(path, rel)

    def text(self, data, rel=REL):
        from check_release import figure_text
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sample.png'
            path.write_bytes(data)
            return figure_text(path, rel)

    def test_clean_figure_passes(self):
        self.assertEqual(self.scan(png(*one_pixel())), ([], [], []))
        software = ('tEXt', b'Software\x00Matplotlib version3.11.1, https://matplotlib.org/')
        self.assertEqual(self.scan(png(*one_pixel(software, ('pHYs', bytes(9))))), ([], [], []))

    def test_text_chunks_reach_every_scanner(self):
        secret = ('tEXt', b'Comment\x00' + ('AKIA' + 'Z' * 16).encode())
        self.assertIn('aws_access_key_id', self.scan(png(*one_pixel(secret)))[1])
        for name, sample in private_samples().items():
            with self.subTest(name=name):
                chunk = ('tEXt', b'Comment\x00' + sample.encode('latin-1'))
                self.assertTrue(self.scan(png(*one_pixel(chunk)))[2])
        movement = ('tEXt', b'Comment\x00MVT_ID_mvt: ' + ('200' + '000001').encode())
        self.assertTrue(self.scan(png(*one_pixel(movement)))[0])

    def test_text_in_image_samples_reaches_the_scanners(self):
        """Review 15 N1: text stored as image samples behind a row filter is decoded and scanned."""
        import zlib

        def grey_row(sample):
            header = len(sample).to_bytes(4, 'big') + (1).to_bytes(4, 'big') + bytes([8, 0, 0, 0, 0])
            sub = bytes([sample[0]] + [(sample[i] - sample[i - 1]) % 256 for i in range(1, len(sample))])
            self.assertNotIn(sample[:4], sub)  # the Sub filter hides the text in the stream
            return png(*one_pixel(header=header, idat=(('IDAT', zlib.compress(b'\x01' + sub)),)))

        from reference_policy import scan_references
        self.assertEqual(self.scan(grey_row(b'plain grey samples')), ([], [], []))
        link = grey_row(b'see [notes](../../status/2026-10-04-notes.md)')
        self.assertTrue(scan_references(self.text(link), self.REL)['excluded_links'])
        self.assertIn('aws_access_key_id', self.scan(grey_row(('AKIA' + 'Z' * 16).encode()))[1])
        self.assertTrue(self.scan(grey_row(b'MVT_ID_mvt: ' + ('200' + '000001').encode()))[0])
        for name, sample in private_samples().items():
            with self.subTest(name=name):
                self.assertTrue(self.scan(grey_row(sample.encode('latin-1')))[2])

    def test_image_over_the_sample_limit_fails_closed(self):
        side = (1 << 13).to_bytes(4, 'big')
        with self.assertRaisesRegex(ValueError, 'cannot scan'):
            self.scan(png(*one_pixel(header=side + side + bytes([8, 6, 0, 0, 0]))))

    def test_gate_k_reads_the_same_text(self):
        from reference_policy import scan_references
        link = ('tEXt', b'Comment\x00see [notes](../../status/2026-10-04-notes.md)')
        text = self.text(png(*one_pixel(link)))
        self.assertTrue(scan_references(text, self.REL)['excluded_links'])

    def test_malformed_or_unlisted_content_fails_closed(self):
        import zlib
        good = one_pixel()
        cases = {
            'bad signature': png(*good, signature=b'\x89PNG\r\n\x1a\x00'),
            'bad crc': png(good[0], (*good[1], b'\x00' * 4), *good[2:]),
            'compressed text': png(*one_pixel(('zTXt', b'Comment\x00\x00' + zlib.compress(b'x')))),
            'international text': png(*one_pixel(('iTXt', b'Comment\x00\x00\x00\x00\x00x'))),
            'exif': png(*one_pixel(('eXIf', b'MM\x00*'))),
            'private chunk': png(*one_pixel(('prVt', b'x'))),
            'bytes after IEND': png(*good, tail=b'note'),
            'no IEND': png(*good[:-1]),
            'IHDR not first': png(good[1], good[0], good[2]),
            'second IHDR': png(good[0], good[0], *good[1:]),
            'cut short': png(*good)[:-3],
            'text over the limit': png(*one_pixel(('tEXt', b'Comment\x00' + b'x' * (1 << 16)))),
            'text file named .png': b'MVT_ID_mvt: ' + ('200' + '000001').encode() + b'\n',
        }
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, 'cannot scan'):
                self.scan(data)

    def test_review_14_hiding_places_fail_closed(self):
        """Review 14 item 3: four places a crafted figure could carry unscanned text."""
        import zlib
        header = (1).to_bytes(4, 'big') * 2 + bytes([8, 0, 0, 0, 0])
        note = b'hidden ' + b'note text'
        stream = zlib.compress(b'\x00\x00')
        cases = {
            'IHDR longer than 13': png(*one_pixel(header=header + note)),
            'pHYs longer than 9': png(*one_pixel(('pHYs', bytes(9) + note))),
            'pHYs shorter than 9': png(*one_pixel(('pHYs', bytes(8)))),
            'bKGD of 3 bytes': png(*one_pixel(('bKGD', bytes(3)))),
            'bKGD of 6 bytes on a grey image': png(*one_pixel(('bKGD', bytes(6)))),
            'IDAT chunk after the stream ends': png(*one_pixel(idat=(('IDAT', stream), ('IDAT', note)))),
            'bytes after the stream in its IDAT': png(*one_pixel(idat=(('IDAT', stream + note),))),
        }
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, 'cannot scan'):
                self.scan(data)

    def test_same_class_hiding_places_fail_closed(self):
        import zlib
        stream = zlib.compress(b'\x00\x00')
        grey = (1).to_bytes(4, 'big') * 2 + bytes([8, 0, 0, 0, 0])
        cases = {
            'rows past the declared size': png(*one_pixel(idat=(('IDAT', zlib.compress(b'\x00\x00' + b'note')),))),
            'stream shorter than the image': png(*one_pixel(idat=(('IDAT', zlib.compress(b'\x00')),))),
            'stream not finished': png(*one_pixel(idat=(('IDAT', stream[:-4]),))),
            'second pHYs': png(*one_pixel(('pHYs', bytes(9)), ('pHYs', bytes(9)))),
            'cHRM not 32 bytes': png(*one_pixel(('cHRM', bytes(31)))),
            'pHYs after the image data': png(('IHDR', grey), ('IDAT', stream), ('pHYs', bytes(9)), ('IEND', b'')),
            'IDAT runs split by text': png(('IHDR', grey), ('IDAT', stream[:4]), ('tEXt', b'a\x00b'),
                                          ('IDAT', stream[4:]), ('IEND', b'')),
            'palette in a grey image': png(*one_pixel(('PLTE', bytes(3)))),
            'interlaced': png(*one_pixel(header=grey[:12] + b'\x01')),
            'bit depth 3': png(*one_pixel(header=grey[:8] + bytes([3, 0, 0, 0, 0]))),
        }
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, 'cannot scan'):
                self.scan(data)

    def test_valid_variants_pass_and_every_chunk_is_scanned(self):
        import zlib
        stream = zlib.compress(b'\x00\x00')
        split = (('IDAT', stream[:3]), ('IDAT', stream[3:]))
        self.assertEqual(self.scan(png(*one_pixel(('bKGD', bytes(2)), idat=split))), ([], [], []))
        secret = ('cHRM', ('AKIA' + 'Z' * 16).encode() + bytes(12))
        self.assertIn('aws_access_key_id', self.scan(png(*one_pixel(secret)))[1])

    def test_png_outside_the_figure_folder_fails_closed(self):
        data = png(*one_pixel())
        for rel in ('docs/sample.png', 'docs/writeup/figures/src/sample.png',
                    'experiments/exp-001-x/sample.png', 'docs/writeup/figures/sample.PNG'):
            with self.subTest(rel=rel), self.assertRaisesRegex(ValueError, 'cannot scan'):
                self.scan(data, rel=rel)


class FullContentGateTests(unittest.TestCase):
    def test_private_classes_fail_gate_i_in_a_physical_tree(self):
        import hashlib
        import json
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        for name, text in private_samples().items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                tree = Path(td) / 'tree'
                (tree / 'release').mkdir(parents=True)
                (tree / 'LICENSE').write_bytes((root / 'LICENSE').read_bytes())
                (tree / 'sample.custom').write_text(text)
                rows = ['path\tsha256\tbytes\trole']
                for rel in ('LICENSE', 'sample.custom'):
                    blob = (tree / rel).read_bytes()
                    rows.append(f'{rel}\t{hashlib.sha256(blob).hexdigest()}\t{len(blob)}\ttest')
                manifest = '\n'.join(rows) + '\n'
                (tree / 'release/MANIFEST.tsv').write_text(manifest)
                contract = Path(td) / 'trusted.tsv'
                contract.write_text(manifest)
                result = subprocess.run([sys.executable, str(root / 'release/check_release.py'),
                                         '--root', str(tree), '--contract', str(contract), '--json'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(len(report['failures']), 1, report)
                self.assertTrue(report['failures'][0].startswith('I:'), report)
                self.assertIn(name, report['private_routes'][0]['patterns'])

    def test_private_metadata_order_and_legacy_kaggle_url(self):
        for text in (
            'id: fake-release-user/fake-object, is_' + 'private: true',
            'https://' + 'www.kaggle.com/' + 'fake-release-user/fake-object',
        ):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'sample.custom'
                path.write_text(text)
                self.assertTrue(scan_readable(path, 'sample.custom')[2])

    def test_bom_text_is_scanned(self):
        for encoding in ('utf-16', 'utf-32'):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'text.custom'
                path.write_text(private_samples()['kaggle_slug'], encoding=encoding)
                self.assertIn('kaggle_slug', scan_readable(path, 'text.custom')[2])


class ProtocolOwnerTests(unittest.TestCase):
    def test_python_split_owner_plumbing_is_not_an_account_identity(self):
        samples = ('def consume(owner: int | str): pass\n', 'record = {"owner": owner}\n')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'protocol.py'
            for text in samples:
                p.write_text(text)
                self.assertNotIn('kaggle_owner', scan_readable(p, 'protocol.py')[2])
            for value in ('synthetic_account', 'int', 'owner'):
                p.write_text('record = {"owner": "' + value + '"}\n')
                self.assertIn('kaggle_owner', scan_readable(p, 'protocol.py')[2])
                p.write_text('owner: `' + value + '`\n')
                self.assertIn('kaggle_owner', scan_readable(p, 'protocol.py')[2])
            for value in ('owner-account', 'int-account'):
                p.write_text('owner: ' + value + '\n')
                self.assertIn('kaggle_owner', scan_readable(p, 'protocol.py')[2])


if __name__ == '__main__':
    unittest.main()
