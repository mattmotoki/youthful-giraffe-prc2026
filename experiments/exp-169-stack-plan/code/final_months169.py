"""Final-mode months and row count for the exp-169 assembly (finalcore, 2026-10-08).

The ranking mode returns the old literals and imports nothing. The final mode ($PRC_RANKING_MODE=final) reads
prc.data from the repo root and stops when that prc.data has no modes (a frozen source tree).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def final():
    """True unless $PRC_RANKING_MODE is unset or 'ranking'; prc.data checks every other name."""
    return (os.environ.get('PRC_RANKING_MODE') or 'ranking') != 'ranking'


def _data():
    if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
    from prc import data as D
    if not hasattr(D, 'rank_month_numbers'):
        raise SystemExit(f'Final mode: {D.__file__} has no rank_month_numbers (a prc without modes).')
    D.ranking_mode()
    return D


def serve_months(split):
    """Months whose rows get weights: [1, 7] for the held split and in the ranking mode; in the final mode the
    ranking split gets prc.data.rank_month_numbers()."""
    assert split in ('fold', 'ranking')
    if split == 'fold' or not final(): return [1, 7]
    months = [int(m) for m in _data().rank_month_numbers()]
    assert 1 in months and 7 in months, months
    return months


def n_rows():
    """Template rows: 344841 in the ranking mode, prc.data.n_ranking() in the final mode."""
    return _data().n_ranking() if final() else 344841


def assert_served(scope, cls, month, months):
    """PLAN.md (b) silent zero: every in-scope ordinary or clock row must sit in a month that got weights."""
    left = scope & cls.isin(['ordinary', 'clock']).to_numpy() & ~month.isin(months).to_numpy()
    assert not left.any(), f'{int(left.sum())} in-scope rows of months {sorted(set(month[left].tolist()))} get no weights.'
