"""C55: stand re-use floor from supplied movement inputs, never departure labels."""
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
AIRPORTS = ('EDDF','EGLL','LFPG','LTFM')
DEP_COLUMNS = [ID,'ADEP_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt',
               'STAND_mvt','AIRCRAFT_TYPE_mvt','LOBT_flt','EOBT_1_flt','AOBT_3_flt']
ARR_COLUMNS = [ID,'ADES_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','BLOCK_TIME_UTC_mvt']
INPUT_COLUMNS = list(dict.fromkeys(['PHASE_mvt']+DEP_COLUMNS+ARR_COLUMNS))


def bounds(movements):
    """One result per departure, with NaN floor outside the fixed source window.

    Select the last arrival first, then check its type. A same-type arrival
    blocks use of any earlier arrival. Exact take-off-time matches are excluded.
    """
    dep = movements.loc[movements.PHASE_mvt.eq('DEP'),DEP_COLUMNS].copy()
    arr = movements.loc[movements.PHASE_mvt.eq('ARR'),ARR_COLUMNS].copy()
    assert dep[ID].is_unique and arr[ID].is_unique
    dep = dep.rename(columns={'ADEP_mvt':'airport'})
    arr = arr.rename(columns={ID:'arrival_id','ADES_mvt':'airport',
                              'AIRCRAFT_TYPE_mvt':'arrival_type','BLOCK_TIME_UTC_mvt':'arrival_block'})
    arr = arr[arr.airport.isin(AIRPORTS) & arr.STAND_mvt.notna() & arr.arrival_block.notna()]
    # The input order cannot decide which aircraft occupies a tied stand/time.
    key = ['airport','STAND_mvt','arrival_block']
    arr['ambiguous_type'] = arr.groupby(key,dropna=False).arrival_type.transform('nunique',dropna=False).gt(1)
    arr = arr.sort_values(['arrival_block','arrival_id'],kind='stable')
    on = dep.airport.isin(AIRPORTS) & dep.STAND_mvt.notna() & dep.MVT_TIME_UTC_mvt.notna()
    left = dep[on].sort_values(['MVT_TIME_UTC_mvt',ID],kind='stable')
    m = pd.merge_asof(left,arr,left_on='MVT_TIME_UTC_mvt',right_on='arrival_block',
                      by=['airport','STAND_mvt'],direction='backward',allow_exact_matches=False)
    g1 = (m.MVT_TIME_UTC_mvt-m.arrival_block).dt.total_seconds()
    typed = m.AIRCRAFT_TYPE_mvt.notna() & m.arrival_type.notna() & m.AIRCRAFT_TYPE_mvt.ne(m.arrival_type)
    joined = m.LOBT_flt.notna()
    taxi_nm = (m.MVT_TIME_UTC_mvt-m.AOBT_3_flt).dt.total_seconds()
    filed = (m.MVT_TIME_UTC_mvt-m.EOBT_1_flt).dt.total_seconds()
    sched = (m.MVT_TIME_UTC_mvt-m.SCHED_TIME_UTC_mvt).dt.total_seconds()
    win = typed & g1.gt(0) & g1.le(7200) & (
        (joined & g1.gt(taxi_nm+300) & g1.lt(filed+900)) |
        (~joined & g1.gt(300) & g1.lt(sched+900)))
    assert not (win & m.ambiguous_type.eq(True)).any(), 'Ambiguous arrival type at the same stand/time.'
    m['stand_g1'] = g1
    m['stand_floor'] = g1.where(win)
    m['other_type'] = typed
    m['stand_joined'] = joined
    audit = m.set_index(ID)[['stand_floor','stand_g1','arrival_id','arrival_block','arrival_type',
                             'other_type','stand_joined']]
    out = dep[[ID,'airport','MVT_TIME_UTC_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','LOBT_flt']].set_index(ID).join(audit)
    out['stand_joined'] = out.LOBT_flt.notna()
    assert len(out) == len(dep) and out.index.is_unique
    active = out.stand_floor.notna()
    assert out.airport[active].isin(AIRPORTS).all()
    assert out.stand_floor[active].between(0,7200,inclusive='right').all()
    return out.reset_index()


def apply_floor(base, floor, airport, lo, hi):
    """Apply max(base, g1) after C33 and retain its joined-row bracket."""
    base, floor = np.asarray(base,dtype=float), np.asarray(floor,dtype=float)
    lo, hi = np.asarray(lo,dtype=float), np.asarray(hi,dtype=float)
    assert base.shape == floor.shape == np.asarray(airport).shape == lo.shape == hi.shape
    assert np.isfinite(base).all() and (base >= 60).all()
    assert not np.isinf(floor).any(), 'Infinite stand bound.'
    on = np.isfinite(floor)
    assert np.isin(np.asarray(airport)[on],AIRPORTS).all()
    assert ((floor[on] > 0) & (floor[on] <= 7200)).all()
    joined = np.isfinite(lo)
    assert np.array_equal(joined,np.isfinite(hi))
    assert ((base[joined] >= lo[joined]) & (base[joined] <= hi[joined])).all(), 'C33 must precede C55.'
    assert (floor[on & joined] <= hi[on & joined]).all(), 'C55 floor conflicts with the C33 upper bound.'
    out = base.copy()
    out[on] = np.maximum(out[on],floor[on])
    assert np.array_equal(out[~on],base[~on])
    assert (out >= base).all() and np.isfinite(out).all()
    assert ((out[joined] >= lo[joined]) & (out[joined] <= hi[joined])).all()
    return out
