"""Target selection, exact bytes and changed inputs fail as specified."""
import base64
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_entry', ROOT / 'scripts/build_entry.py')
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryReplayTests(unittest.TestCase):
    def test_target_alias_default_and_unknown_target(self):
        self.assertEqual(entry.target_record('36')[0], 'v36')
        self.assertEqual(entry.target_record()[0], entry.registry()['default_target'])
        with self.assertRaises(ValueError):
            entry.target_record('unregistered')

    def test_new_saved_target_replays_without_private_source_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);raw = root / 'input/data/raw';raw.mkdir(parents=True)
            template = pd.DataFrame({entry.legacy.ID: np.array([101., 102.])})
            template.to_parquet(raw / 'submitting.parquet', index=False)
            template.to_parquet(raw / 'ranking.parquet', index=False)
            values = np.array([60., 150.], dtype='<f8')
            encoded = {'encoding': 'gzip-base64', 'dtype': '<f8', 'rows': 2,
                       'array_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                       'data': base64.b64encode(gzip.compress(values.tobytes(), mtime=0)).decode()}
            vector = root / 'vector.json';vector.write_text(json.dumps(encoded))
            expected = template.assign(**{entry.legacy.Y: values})
            expected.to_parquet(root / 'expected.parquet', index=False)
            record = {'recipe': 'saved-values', 'public_vector': 'vector.json',
                      'public_vector_sha256': entry.legacy.sha256(vector),
                      'sha256': entry.legacy.sha256(root / 'expected.parquet'),
                      'value_sha256': entry.legacy.value_sha256(expected), 'rows': 2,
                      'bytes': (root / 'expected.parquet').stat().st_size, 'id_dtype': 'float64'}
            book = {'default_target': 'v99', 'entries': {'v99': record},
                    'input_template_sha256': entry.legacy.sha256(raw / 'submitting.parquet'),
                    'ranking_sha256': entry.legacy.sha256(raw / 'ranking.parquet')}
            with patch.object(entry, 'registry', return_value=book), patch.object(entry.legacy, 'ROOT', root), patch.object(entry.legacy, 'N_ROWS', 2):
                output = entry.replay(None, raw.parents[1], root / 'out', root / 'receipt.json')
                self.assertEqual(entry.legacy.sha256(output), record['sha256'])
                self.assertFalse(json.loads((root / 'receipt.json').read_text())['fit_ran'])
                with self.assertRaises(SystemExit):
                    entry.replay('v99', raw.parents[1], root / 'out')
                # A new composition target uses the two component routes, with no new saved array.
                vb = np.array([100., 300.], dtype='<f8')
                encoded_b = dict(encoded, array_sha256=hashlib.sha256(vb.tobytes()).hexdigest(),
                                 data=base64.b64encode(gzip.compress(vb.tobytes(), mtime=0)).decode())
                (root / 'vector-b.json').write_text(json.dumps(encoded_b))
                fb = template.assign(**{entry.legacy.Y: vb})
                fb.to_parquet(root / 'expected-b.parquet', index=False)
                book['entries']['v100'] = dict(record, public_vector='vector-b.json',
                    public_vector_sha256=entry.legacy.sha256(root / 'vector-b.json'),
                    sha256=entry.legacy.sha256(root / 'expected-b.parquet'),
                    value_sha256=entry.legacy.value_sha256(fb))
                mean = template.assign(**{entry.legacy.Y: .5 * values + .5 * vb})
                mean.to_parquet(root / 'expected-blend.parquet', index=False)
                book['entries']['v101'] = dict(record, recipe='fixed-blend', a='v99', b='v100',
                    weight_on_a=.5, sha256=entry.legacy.sha256(root / 'expected-blend.parquet'),
                    value_sha256=entry.legacy.value_sha256(mean))
                blend = entry.replay('v101', raw.parents[1], root / 'blend', root / 'blend.json')
                np.testing.assert_array_equal(pd.read_parquet(blend)[entry.legacy.Y], [80., 225.])
                self.assertEqual(entry.legacy.sha256(blend), book['entries']['v101']['sha256'])
                components = json.loads((root / 'blend.json').read_text())['components']
                self.assertEqual(components['a']['sha256'], record['sha256'])
                self.assertEqual(components['b']['sha256'], book['entries']['v100']['sha256'])
                book['entries']['v101']['a'] = 'v101'
                with self.assertRaisesRegex(ValueError, 'Invalid fixed-blend'):
                    entry.replay('v101', raw.parents[1], root / 'cycle')
                vector.write_text(vector.read_text() + '\n')
                with self.assertRaisesRegex(SystemExit, 'saved array hash mismatch'):
                    entry.replay('v99', raw.parents[1], root / 'bad-vector')
                (raw / 'submitting.parquet').write_bytes(b'wrong official file')
                with self.assertRaisesRegex(SystemExit, 'wrong submitting.parquet'):
                    entry.replay('v99', raw.parents[1], root / 'bad-input')


if __name__ == '__main__':
    unittest.main()
