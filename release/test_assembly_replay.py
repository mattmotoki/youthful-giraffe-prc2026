"""Assembly replay (target zabd): exact frame rebuilds, tamper checks and pack hygiene."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'release'))
import assembly_replay as R  # noqa: E402

ID, Y = R.ID, R.Y


def synthetic_official():
    times = pd.to_datetime(['2025-01-03 10:00:00', '2025-01-31 23:59:57', '2025-07-02 08:30:00'], utc=True)
    return pd.DataFrame({ID: np.array([3e8 + 1, 3e8 + 2, 3e8 + 3]), 'ADEP_mvt': ['AAAA', 'BBBB', None],
                         'MVT_TIME_UTC_mvt': times.astype('datetime64[us, UTC]'), Y: np.array([600., 720., 480.])})


class AssemblyReplayTests(unittest.TestCase):
    def test_saved_arrays_round_trip_exactly(self):
        weird_nan = np.frombuffer(np.array([0xFFF8000000000001], dtype='<u8').tobytes(), dtype='<f8')[0]
        frame = pd.DataFrame({'f': np.array([1.5, np.nan, weird_nan]), 'i': np.array([1, 2, 3], dtype='int8'),
                              'b': np.array([True, False, True]), 'o': np.array(['x', None, np.nan], dtype=object)})
        with tempfile.TemporaryDirectory() as td:
            for name in frame.columns:
                path = Path(td) / f'{name}.json'
                path.write_text(json.dumps(R.encode_array(frame[name])))
                values = R.decode_array(path, R.sha256_file(path), 3)
                rebuilt = pd.Series(values, name=name).astype(frame[name].dtype)
                self.assertEqual(R.column_bytes(rebuilt), R.column_bytes(frame[name]))
            self.assertIsNone(R.decode_array(Path(td) / 'o.json', R.sha256_file(Path(td) / 'o.json'), 3)[1])

    def test_nan_payload_does_not_change_the_digest_but_values_do(self):
        a = pd.DataFrame({'x': [1.0, np.nan]})
        b = pd.DataFrame({'x': [1.0, -np.nan]})
        c = pd.DataFrame({'x': [1.0, 2.0]})
        self.assertEqual(R.frame_digest(a), R.frame_digest(b))
        self.assertNotEqual(R.frame_digest(a), R.frame_digest(c))
        self.assertNotEqual(R.frame_digest(a), R.frame_digest(a.astype('float32')))

    def test_rebuild_uses_official_label_and_saved_columns_and_rejects_tampering(self):
        official = synthetic_official()
        saved = pd.Series([1.25, 2.5, 3.75], name='member')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'member.json'
            path.write_text(json.dumps(R.encode_array(saved)))
            spec = {'columns': [
                {'name': ID, 'kind': 'id', 'dtype': 'int64'},
                {'name': 'airport', 'kind': 'official', 'dtype': 'object', 'source': 'ADEP_mvt'},
                {'name': 'month', 'kind': 'official', 'dtype': 'int8', 'source': 'MVT_TIME_UTC_mvt', 'transform': 'month'},
                {'name': 'y', 'kind': 'label', 'dtype': 'float64'},
                {'name': 'member', 'kind': 'saved', 'dtype': 'float64', 'file': str(path), 'sha256': R.sha256_file(path)}]}
            expected = pd.DataFrame({ID: official[ID].astype('int64'), 'airport': official.ADEP_mvt,
                                     'month': np.array([1, 1, 7], dtype='int8'), 'y': official[Y], 'member': saved})
            rebuilt = R.rebuild_frame(spec, official, 3)
            self.assertEqual(R.frame_digest(rebuilt), R.frame_digest(expected))
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(SystemExit):
                R.rebuild_frame(spec, official, 3)

    def test_official_inputs_are_checked_by_hash(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / 'data/raw'
            raw.mkdir(parents=True)
            for key, rel in R.OFFICIAL_FILES.items():
                rows = synthetic_official().assign(PHASE_mvt='DEP')
                if key == 'training_2025_07':
                    rows[ID] += 10
                rows.to_parquet(Path(td) / rel, index=False)
            good = {key: R.sha256_file(Path(td) / rel) for key, rel in R.OFFICIAL_FILES.items()}
            tables = R.official_tables(Path(td), good)
            self.assertEqual(len(tables['fold']), 6)
            self.assertEqual(tables['fold'][ID].iloc[3], 3e8 + 11)
            rel = R.OFFICIAL_FILES['training_2025_07']
            synthetic_official().assign(PHASE_mvt='DEP').to_parquet(Path(td) / rel, index=False)
            with self.assertRaises(SystemExit):
                R.official_tables(Path(td), dict(good, training_2025_07=R.sha256_file(Path(td) / rel)))
            with self.assertRaises(SystemExit):
                R.official_tables(Path(td), dict(good, ranking='0' * 64))

    def test_zabd_pack_is_registered_and_saves_no_id_or_label(self):
        book = json.loads((ROOT / 'release/ENTRIES.json').read_text())
        spec = book['entries']['zabd']
        self.assertEqual(spec['recipe'], 'assembly')
        self.assertEqual(R.sha256_file(ROOT / spec['pack']), spec['pack_sha256'])
        pack = json.loads((ROOT / spec['pack']).read_text())
        self.assertEqual(pack['candidate_sha256'], spec['sha256'])
        self.assertEqual(pack['with_c65_sha256'], spec['with_c65_sha256'])
        self.assertEqual(pack['parent_sha256'], spec['parent_sha256'])
        self.assertEqual(pack['official_sha256']['submitting'], book['input_template_sha256'])
        self.assertEqual(pack['official_sha256']['ranking'], book['ranking_sha256'])
        listed = set()
        for key, frame in pack['frames'].items():
            kinds = {c['name']: c['kind'] for c in frame['columns']}
            for name in (ID, 'MVT_ID', 'FLIGHT_ID_mvt'):
                self.assertNotEqual(kinds.get(name), 'saved', f'{key}:{name}')
            # 2026 has no labels; there a TAXITIME_SEC_mvt column is a prediction in submission form.
            if frame['split'] == 'fold':
                for name in ('y', 'y_true', 'target', Y):
                    self.assertNotEqual(kinds.get(name), 'saved', f'{key}:{name}')
            for col in frame['columns']:
                if col['kind'] == 'saved':
                    self.assertEqual(R.sha256_file(ROOT / col['file']), col['sha256'])
                    listed.add(col['file'])
        for rel in pack['records'].values():
            listed.add(rel)
        on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'release/saved-vectors/zabd').rglob('*.json')}
        self.assertEqual(on_disk, listed)
        self.assertEqual(R.sha256_file(ROOT / spec['comparison_vector']['public_vector']),
                         spec['comparison_vector']['public_vector_sha256'])


if __name__ == '__main__':
    unittest.main()
