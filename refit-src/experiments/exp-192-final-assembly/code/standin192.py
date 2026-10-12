"""DRY RUN ONLY: stand-in values for every saved column of the 2026 frames on the final file (exp-192, 2026-10-08).

Purpose: run the final-mode code path of assemble192.py (four months, the patched loops, the season map, the
exposure mix, the 670,790-row serialization, the stages record) before the real parts exist. No output of a
stand-in run is a part or a candidate.

- January/July rows: the pack's saved value of the same movement ID (a ranking-file vector; allowed here only
  because this is a code-path test, COMMON "parity reference only").
- February/June rows:
  - route and F values: the official-columns GBDT of finallirf (`exp-187-final-lirf/out/gbdt-final.parquet`);
    each member: the GBDT value plus a fixed offset per member (OFFSET), so that a blend that reaches a
    February/June row changes it (equal members would hide the month-loop trap of PLAN.md (b));
  - head: NaN; cls: noac without a type, else ordinary (no clock rows, so no head rule);
  - lo, hi: the C33 bracket of `read_stacks.add_bracket` on the final rows; noplan: LIRF without a type;
  - np_v3cal: the GBDT value on the noplan rows; rule frame t = MVT - 2025-01-01 UTC, scope = plan and
    ADES != ADEP; lower = max(60, lo), upper = hi (inf without a bracket);
  - every other mask False, weights 0, points and floors NaN, order_delta 0.
"""
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
T0 = pd.Timestamp('2025-01-01', tz='UTC')
VALUE_COLUMNS = {('ranking/rule_frame', 'F'), ('ranking/primary_frame', 'mlp')}
OFFSET = {'ranking/primary_frame': 40., 'ranking/vector/g': 20., 'ranking/vector/sequence021': 60.,
          'ranking/vector/sequence039': 80., 'ranking/vector/row6': 100., 'ranking/vector/sequence011': 30.,
          'ranking/vector/row011': 50., 'ranking/vector/dark18': 70.}


def standins(R, export, pack, off, gbdt_path):
    rk = off['ranking']; ids = rk[ID].to_numpy('int64'); n = len(ids)
    month = pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    new = ~np.isin(month, [1, 7])
    g = pd.read_parquet(gbdt_path).set_index(ID).TAXITIME_SEC_mvt.reindex(ids).to_numpy(float)
    assert np.isfinite(g).all()
    typed = rk.AIRCRAFT_TYPE_mvt.notna().to_numpy(); lirf = rk.ADEP_mvt.eq('LIRF').to_numpy()
    mvt = pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True); lobt = pd.to_datetime(rk.LOBT_flt, utc=True)
    m = (mvt - lobt).dt.total_seconds().to_numpy(float)
    lo, hi = m - 3606, m + 3606
    noplan = lirf & ~typed
    cls = np.where(typed, 'ordinary', 'noac').astype(object)
    feb_jun = {
        'lo': lo, 'hi': hi, 'rule': np.zeros(n), 'extension': np.zeros(n), 'noplan': noplan.astype(float),
        'np_v3cal': np.where(noplan, g, np.nan), 'practice': np.zeros(n), 'practice_weight': np.zeros(n),
        'practice_point': np.full(n, np.nan), 'stand_floor': np.full(n, np.nan), 'cls': cls,
        'c73': np.zeros(n), 'c73_weight': np.zeros(n), 'c73_point': np.full(n, np.nan),
        't': (mvt - T0).dt.total_seconds().to_numpy(float),
        'scope': (lobt.notna() & rk.ADES_mvt.ne(rk.ADEP_mvt)).to_numpy().astype(float),
        'g': np.full(n, np.nan), 'g71': np.full(n, np.nan), 'c71': np.zeros(n),
        'C40': np.zeros(n), 'C58': np.zeros(n), 'C63': np.zeros(n), 'C64': np.zeros(n), 'C66': np.zeros(n),
        'C68': np.zeros(n), 'C71': np.zeros(n), 'order_delta': np.zeros(n), 'point58': np.full(n, np.nan),
        'point66': np.full(n, np.nan), 'lower': np.maximum(60., np.nan_to_num(lo, nan=60.)),
        'upper': np.where(np.isfinite(hi), hi, np.inf), 'pred_HB3u': np.full(n, np.nan),
    }
    pack_ids = off['pack_ranking_ids']
    pos = pd.Series(np.arange(len(pack_ids)), index=pack_ids).reindex(ids).to_numpy()
    assert np.isnan(pos[new]).all() and not np.isnan(pos[~new]).any(), 'Jan/Jul IDs must be the pack IDs.'
    old = pos[~new].astype(int)
    injected = {}
    for key, spec in pack['frames'].items():
        if spec['split'] != 'ranking':
            continue
        for col in spec['columns']:
            if col['kind'] != 'saved':
                continue
            saved = R.decode_array(export / col['file'], col['sha256'], len(pack_ids))
            name = col['name']
            if name in feb_jun and (key, name) not in VALUE_COLUMNS:
                fill = feb_jun[name]
            else:
                fill = g + OFFSET.get(key, 0.)
            v = np.empty(n, dtype=object if saved.dtype == object else float)
            v[new] = fill[new]
            v[~new] = saved[old]
            injected.setdefault(key, {})[name] = v
    return injected
