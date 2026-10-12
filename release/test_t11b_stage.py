"""T11b-L stage (target v64): the long-taxi day rule, the move, the P2 rows and the registration."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'release'))
sys.path.insert(0, str(ROOT / 'scripts'))
import t11b_stage as S  # noqa: E402

ID = S.ID
spec = importlib.util.spec_from_file_location('build_entry', ROOT / 'scripts/build_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def departures(ap, day, n, n_long, ades='ZZZZ', aobt=True, phase='DEP', start=0):
    """n rows at airport ap on 2026-01-<day>; the first n_long have an NM taxi of one hour."""
    mvt = pd.Timestamp(f'2026-01-{day:02d} 12:00:00', tz='UTC')
    taxi = np.where(np.arange(n) < n_long, 3600, 600)
    aobt3 = mvt - pd.to_timedelta(taxi, unit='s')
    return pd.DataFrame({ID: np.arange(start, start + n, dtype='float64'), 'PHASE_mvt': phase, 'ADEP_mvt': ap,
                         'ADES_mvt': ades, 'MVT_TIME_UTC_mvt': pd.DatetimeIndex([mvt] * n).astype('datetime64[us, UTC]'),
                         'AOBT_3_flt': (aobt3 if aobt else pd.DatetimeIndex([pd.NaT] * n, tz='UTC')).astype('datetime64[us, UTC]')})


def raw_ranking():
    parts, start = [], 0
    for args in (('AAAA', 5, 400, 8), ('AAAA', 6, 401, 8), ('BBBB', 5, 100, 7)):
        parts.append(departures(*args, start=start)); start += args[2]
    # Rows with ADES = ADEP and arrivals do not count, even with a long NM taxi.
    parts.append(departures('CCCC', 5, 6, 6, start=start)); start += 6
    parts.append(departures('CCCC', 5, 2, 2, ades='CCCC', start=start)); start += 2
    parts.append(departures('CCCC', 5, 5, 5, phase='ARR', start=start)); start += 5
    # The share counts only rows with AOBT_3.
    parts.append(departures('DDDD', 5, 400, 8, start=start)); start += 400
    parts.append(departures('DDDD', 5, 1000, 0, aobt=False, start=start)); start += 1000
    return pd.concat(parts, ignore_index=True)


class LongTaxiDayTests(unittest.TestCase):
    def test_rule_needs_eight_long_rows_and_two_percent_of_rows_with_aobt3(self):
        _, g = S.flags(raw_ranking())
        long_days = set(g[g.snow].ap + '|' + g[g.snow].dstr)
        self.assertEqual(long_days, {'AAAA|01-05', 'DDDD|01-05'})
        cccc = g[g.ap == 'CCCC'].iloc[0]
        self.assertEqual((int(cccc.rows), int(cccc.n_long)), (6, 6))

    def test_rows_of_a_long_day_and_the_registered_day_count(self):
        raw = raw_ranking()
        ids = raw[raw.PHASE_mvt == 'DEP'][ID].to_numpy('int64')
        rows, keys = S.long_day_rows(ids, raw, {'AAAA': 1, 'DDDD': 1})
        self.assertEqual(keys, ['AAAA|01-05', 'DDDD|01-05'])
        self.assertEqual(int(rows.sum()), 400 + 1400)
        with self.assertRaises(SystemExit):
            S.long_day_rows(ids, raw, {'AAAA': 1})
        with self.assertRaises(AssertionError):
            S.long_day_rows(np.append(ids, 10 ** 9), raw, {'AAAA': 1, 'DDDD': 1})


class MoveTests(unittest.TestCase):
    def test_quarter_move_inside_bounds_with_floor(self):
        base = np.array([100., 100., 200., 1000., 61.])
        member = np.array([300., -50., 200., np.nan, 0.])
        lo, hi = np.zeros(5), np.array([1000., 1000., 150., 2000., 1000.])
        m = np.array([True, True, True, False, True])
        np.testing.assert_array_equal(S.move(base, member, lo, hi, m), [150., 90., 150., 1000., 60.75])
        with self.assertRaises(AssertionError):
            S.move(base, member, lo, hi, np.ones(5, bool))

    def test_long_day_rows_keep_the_base(self):
        base, member = np.full(3, 400.), np.full(3, 800.)
        out = S.apply(base, member, np.zeros(3), np.full(3, 1e4), np.array([True, True, False]),
                      np.array([False, True, True]))
        np.testing.assert_array_equal(out, [500., 400., 400.])


class P2RowTests(unittest.TestCase):
    def test_typed_rows_without_a_join_and_the_bracket_check(self):
        t = pd.Timestamp('2026-01-05 10:00', tz='UTC')
        ranking = pd.DataFrame({'LOBT_flt': pd.DatetimeIndex([t, pd.NaT, pd.NaT, pd.NaT]),
                                'AIRCRAFT_TYPE_mvt': ['A320', 'A320', None, 'B738']})
        cls = np.array(['a', 'a', 'a', 'noac'], dtype=object)
        lo, hi = np.array([0., np.nan, np.nan, np.nan]), np.array([900., np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(S.p2_rows(ranking, cls, lo, hi), [False, True, False, False])
        with self.assertRaises(SystemExit):
            S.p2_rows(ranking, cls, np.array([0., 0., np.nan, np.nan]), np.array([900., 900., np.nan, np.nan]))


class RegistrationTests(unittest.TestCase):
    def test_the_default_target_and_v73_compose_their_registered_chains(self):
        book = json.loads((ROOT / 'release/ENTRIES.json').read_text())
        self.assertEqual(book['default_target'], 'final-v-dh-r9-ld')
        # the DH step on each final V entry, with the same DH part; final-v-dh-r9-ld is the default target
        for dh_name, v_name in (('final-dh', 'final'), ('final-ii-dh', 'final-ii'), ('final-iii-dh', 'final-iii'),
                                ('final-iii-dh-r9', 'final-iii-r9'), ('final-iii-dh-r9-ld', 'final-iii-r9-ld'),
                                ('final-v-dh', 'final-v'), ('final-v-dh-r9', 'final-v-r9'),
                                ('final-v-dh-r9-ld', 'final-v-r9-ld')):
            dh = book['entries'][dh_name]
            self.assertEqual((dh['recipe'], dh['stage'], dh['mode'], dh['base'], dh['base_sha256']),
                             ('steps', 'dh-step', 'final', v_name, book['entries'][v_name]['sha256']))
            self.assertEqual((dh['code'], dh['member']), (book['entries']['final-dh']['code'],
                                                          book['entries']['final-dh']['member']))
        # final-ii and final-iii: the steps of final with the 9-seed s7 member; final-iii also with candidate A's L2C;
        # final-iii-r9: final-iii's steps on the R9 chain; final-iii-r9-ld: also LD's ORD-S with its flag;
        # final-v, final-v-r9, final-v-r9-ld: the same three with the 15-seed s7 member
        final = book['entries']['final']
        for name, l2c_npz, ld, seeds in (('final-ii', None, False, 9), ('final-iii', 'work/w2/s7c-l2c.npz', False, 9),
                                         ('final-iii-r9', 'work/w2/s7c-l2c.npz', False, 9),
                                         ('final-iii-r9-ld', 'work/w2/s7c-l2c.npz', True, 9),
                                         ('final-v', 'work/w2/s7c-l2c.npz', False, 15),
                                         ('final-v-r9', 'work/w2/s7c-l2c.npz', False, 15),
                                         ('final-v-r9-ld', 'work/w2/s7c-l2c.npz', True, 15)):
            e = book['entries'][name]
            self.assertEqual((e['stage'], e['member']['seeds_per_mode'], e['member'].get('l2c_npz'),
                              e['member'].get('ords_flags')),
                             (final['stage'], seeds, l2c_npz, ['--long-days'] if ld else None))
            changed = {k for k in final['code'] if e['code'][k] != final['code'][k]}
            sources = {'w2/l2c.py': 'w2/l2c_candidate_a.py'} if l2c_npz else {}
            if ld:
                sources['w2/ords.py'] = 'w2/ords_ld.py'
            self.assertEqual((set(e['code']), changed, e.get('code_sources', {})),
                             (set(final['code']), set(sources), sources))
        # final, final-ii, final-iii -> v64-final -> zabd-final (final files), v73 -> v64 -> zabd (ranking files),
        # final-iii-r9 -> v64-final-r9 -> zabd-final-r9 (final files, the R9 pack); the 15-seed entries as their
        # 9-seed siblings
        for top, v64_name, zabd_name, days in (('final', 'v64-final', 'zabd-final', 21),
                                               ('final-ii', 'v64-final', 'zabd-final', 21),
                                               ('final-iii', 'v64-final', 'zabd-final', 21), ('v73', 'v64', 'zabd', 15),
                                               ('final-iii-r9', 'v64-final-r9', 'zabd-final-r9', 21),
                                               ('final-iii-r9-ld', 'v64-final-r9', 'zabd-final-r9', 21),
                                               ('final-v', 'v64-final', 'zabd-final', 21),
                                               ('final-v-r9', 'v64-final-r9', 'zabd-final-r9', 21),
                                               ('final-v-r9-ld', 'v64-final-r9', 'zabd-final-r9', 21)):
            head, v64, zabd = (book['entries'][n] for n in (top, v64_name, zabd_name))
            self.assertEqual((head['base'], head['base_sha256']), (v64_name, v64['sha256']))
            self.assertEqual((v64['recipe'], v64['base'], v64['stage']), ('composition', zabd_name, 'T11b-L'))
            self.assertEqual(v64['base_sha256'], zabd['sha256'])
            self.assertEqual((v64['pack'], v64['pack_sha256']), (zabd['pack'], zabd['pack_sha256']))
            self.assertEqual({e.get('mode', 'ranking') for e in (head, v64, zabd)}, {'final' if top.startswith('final') else 'ranking'})
            pack = json.loads((ROOT / v64['pack']).read_text())
            for key in S.FRAMES.values():
                self.assertEqual(pack['frames'][key]['split'], 'ranking')
            self.assertEqual(sum(v64['long_taxi_days'].values()), days)
            self.assertLess(v64['changed_rows'], v64['p2_rows'])

    def test_wrong_stage_base_or_base_bytes_are_rejected(self):
        book = json.loads((ROOT / 'release/ENTRIES.json').read_text())
        v64 = book['entries']['v64']
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            with self.assertRaisesRegex(ValueError, 'Unsupported composition stage'):
                entry.composition_replay('v64', dict(v64, stage='other'), out, out, out / 'x.parquet', None, False, ())
            with self.assertRaisesRegex(ValueError, 'not the registered base'):
                entry.composition_replay('v64', dict(v64, base_sha256='0' * 64), out, out, out / 'x.parquet', None, False, ())
            wrong = out / 'base.parquet'
            pd.DataFrame({ID: [1.], S.Y: [600.]}).to_parquet(wrong, index=False)
            with self.assertRaisesRegex(SystemExit, 'base entry differs'):
                S.compose(v64, wrong, out)


if __name__ == '__main__':
    unittest.main()
