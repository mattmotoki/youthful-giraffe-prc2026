"""Member and blend frames of the zabd pack (exp-192, finalplan, 2026-10-08): the rules that make them.

The rules are those of the exp-169 builders (paths are in the release export EXP of SPEC.md):
- held exposure weight: `prepare_stack3.py:65-67`, w = N(cell) / N_rows / count_held(cell) when count_held >= 2;
- held member scope: `prepare_stack3.py:69-72`, (F < 2400) & aobt & cls != noac & (w > 0) & ~noplan;
- 2026 member scope: `prepare_stack3.py:85-87`, (F < 2400) & AOBT_3 & type & support(cell);
- 2026 pair scope: the primary scope plus the extension rows with F < 2400 and AOBT_3
  (`prepare_pair_extension.py:104-106`, `stack3_aobt.py:139-141`); control scope = pair scope & ordinary.

Season map (COMMON design row; SPEC.md design rows 2 and 3): a 2026 row of month m takes the held cells of SEASON[m].
In the ranking mode the map is the identity on months 1 and 7.
"""
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
SEASON = {1: 1, 2: 1, 6: 7, 7: 7}
MIN_SUPPORT = 2
F_MAX = 2400.


def held_counts(rows):
    """Held rows per (month, airport, cls); `rows` is the frozen exp-146 row table on the fold IDs."""
    return rows.groupby(['month', 'airport', 'cls']).size()


def cell_rows(month, airport, cls):
    """2026 rows per held cell (season month, airport, cls)."""
    q = pd.DataFrame({'month': pd.Series(month).map(SEASON).to_numpy(), 'airport': np.asarray(airport),
                      'cls': np.asarray(cls)})
    assert q.month.notna().all(), 'A 2026 month outside the season map.'
    q['month'] = q.month.astype(int)
    return q.groupby(['month', 'airport', 'cls']).size()


def exposure(rows, cells, n_rows):
    """Held fit weight of each fold row: the 2026 row share of its cell over the held rows of the cell."""
    count = held_counts(rows)
    key = zip(rows.month, rows.airport, rows.cls)
    return np.array([cells.get((m, a, z), 0) / n_rows / count[m, a, z] if count[m, a, z] >= MIN_SUPPORT else 0.
                     for m, a, z in key])


def fold_scope(F, aobt, cls, w, noplan):
    return (np.asarray(F) < F_MAX) & np.asarray(aobt, bool) & (np.asarray(cls) != 'noac') & (np.asarray(w) > 0) \
        & ~np.asarray(noplan, bool)


def support(count, month, airport, cls):
    """True where the held cell of the row's season month has at least MIN_SUPPORT held rows."""
    return np.array([count.get((SEASON[int(m)], a, z), 0) >= MIN_SUPPORT for m, a, z in zip(month, airport, cls)])


def ranking_scope(F, aobt3, typed, supported):
    return (np.asarray(F) < F_MAX) & np.asarray(aobt3, bool) & np.asarray(typed, bool) & np.asarray(supported, bool)


def pair_scope(scope, extension, F, aobt3):
    return np.asarray(scope, bool) | (np.asarray(extension, bool) & (np.asarray(F) < F_MAX) & np.asarray(aobt3, bool))


def control_scope(pair, cls):
    return np.asarray(pair, bool) & (np.asarray(cls) == 'ordinary')


def mlp_choice(mlp008, mix12, scope, cls):
    """The released primary-frame mlp: mix12 on member-scope clock rows, mlp008 on every other row."""
    return np.where(np.asarray(scope, bool) & (np.asarray(cls) == 'clock'), mix12, mlp008)
