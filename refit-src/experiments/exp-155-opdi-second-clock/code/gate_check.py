"""Replay the pinned exp-125 matcher; change only its identity gate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
CLOCKS = {'exit-parking_position': 'exit_parking_position_utc',
          'exit-apron': 'exit_apron_utc', 'entry-taxiway': 'entry_taxiway_utc',
          'first_seen': 'first_seen_utc'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matcher(wide=False):
    path = EXP/'source/exp125_analyze.py'
    source = path.read_text()
    old = 'pairs = pairs[np.isfinite(pairs["_first_seen_abs_delta_s"]) & (pairs["_first_seen_abs_delta_s"] <= 600)].copy()'
    new = '''dx = event_delta(pairs["first_climb_utc"], pairs["_match_time"])
    pairs = pairs[pairs["_first_seen_delta_s"].between(-5400, 300) & (dx.isna() | dx.between(0, 720))].copy()'''
    assert source.count(old) == 1
    if wide:
        source = source.replace(old, new)
    tree = ast.parse(source)
    names = {'norm_call', 'event_delta', 'empty_matches', 'choose_matches', 'first_seen_ground_ok'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(pd=pd, np=np, CANDIDATES=CLOCKS, FIRST_SEEN_EVENT_TOLERANCE_S=60)
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def compact(fl, ev, airports):
    # Preserve exp-125 departure-side, first-event and altitude rules.
    fl = fl.rename(columns={'id': 'flight_id', 'dof': 'dof_utc',
                            'first_seen': 'first_seen_utc', 'last_seen': 'last_seen_utc'}).copy()
    fl.flight_id = fl.flight_id.astype('uint64').to_numpy().view('int64')
    ev = ev.copy()
    ev.flight_id = ev.flight_id.astype('uint64').to_numpy().view('int64')
    ev.event_time = pd.to_datetime(ev.event_time, utc=True)
    if 'adep' not in ev:
        ev = ev.join(fl.set_index('flight_id').adep, on='flight_id')
    cols = {'exit-parking_position': 'exit_parking_position_utc', 'exit-apron': 'exit_apron_utc',
            'entry-taxiway': 'entry_taxiway_utc', 'entry-runway': 'entry_runway_utc',
            'take-off': 'takeoff_utc', 'first-xing-fl50': 'first_climb_utc',
            'first_seen': 'first_seen_event_utc', 'last_seen': 'last_seen_event_utc'}
    extract_tree = ast.parse((EXP/'source/exp125_extract.py').read_text())
    select = {'haversine_km', 'is_named_airport'}
    extract_tree.body = [n for n in extract_tree.body if
                        (isinstance(n, ast.FunctionDef) and n.name in select) or
                        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'AIRPORT_NAME_TOKENS' for t in n.targets))]
    scope = dict(np=np)
    exec(compile(extract_tree, 'exp125_extract.py', 'exec'), scope)
    ev['_distance_km'] = np.nan
    for a, row in airports.loc[sorted(ev.adep.dropna().unique())].iterrows():
        m = ev.adep.eq(a)
        ev.loc[m, '_distance_km'] = scope['haversine_km'](row.longitude_deg, row.latitude_deg,
                                                       ev.loc[m, 'longitude'].to_numpy(), ev.loc[m, 'latitude'].to_numpy())
    named = np.array([scope['is_named_airport'](v, a) for v, a in zip(ev['info'], ev.adep)])
    surface = ev.type.isin(['exit-parking_position', 'exit-apron', 'entry-taxiway', 'entry-runway', 'take-off', 'first_seen'])
    ev = ev[~surface | ev._distance_km.le(5) | named].copy()
    elev = ev.adep.map(airports.elevation_ft)
    ev['_ground_proxy'] = ev._distance_km.le(5) & ev.altitude.notna() & (ev.altitude-elev).abs().le(500)
    for typ, col in cols.items():
        g = ev[ev.type.eq(typ)].sort_values('event_time').drop_duplicates('flight_id')
        if typ == 'first_seen':
            g = g[g._ground_proxy]
            fl['first_seen_ground_altitude_ft'] = fl.flight_id.map(g.set_index('flight_id').altitude)
            fl['first_seen_ground_distance_km'] = fl.flight_id.map(g.set_index('flight_id')._distance_km)
        fl[col] = fl.flight_id.map(g.set_index('flight_id').event_time)
    for c in [c for c in fl if c.endswith('_utc')]:
        fl[c] = pd.to_datetime(fl[c], utc=True, errors='coerce')
    return fl


def replay(prc, od, wide, shift=0, original_sign=False):
    funcs = matcher(wide)
    mt = prc.MVT_TIME_UTC_mvt + pd.Timedelta(seconds=shift)
    matched, _ = funcs['choose_matches'](prc, od, match_time=mt, key_date_time=prc.MVT_TIME_UTC_mvt)
    out = prc.merge(matched, on='_row_ix', how='left', validate='1:1')
    leads = pd.DataFrame(index=out.index)
    for typ, col in CLOCKS.items():
        t = pd.to_datetime(out[col], utc=True, errors='coerce')
        lead = (out.MVT_TIME_UTC_mvt-t).dt.total_seconds()
        if original_sign:
            lead = -lead
        ok = lead.between(60, 14400)
        if typ == 'first_seen':
            ok &= funcs['first_seen_ground_ok'](out)
        leads[typ] = lead.where(ok)
    out['has_event'] = leads.notna().any(axis=1)
    out['has_surface'] = leads[['exit-parking_position', 'exit-apron', 'entry-taxiway']].notna().any(axis=1)
    out['new_clock'] = out.has_event & ~out.obs_has_clock
    return out


def run(args):
    work = Path(args.work)
    airports = pd.read_csv(work/'airports.csv').set_index('ident')
    rk = pd.read_parquet(args.raw+'/ranking.parquet')
    rk = rk[rk.PHASE_mvt.eq('DEP')].copy()
    classes = pd.concat([pd.read_parquet(Path(args.class_root)/f'2026-{m:02d}-v25.parquet',
                                       columns=['MVT_ID_mvt', 'obs_has_clock']) for m in (1, 7)])
    rk = rk.merge(classes, on='MVT_ID_mvt', validate='1:1')
    rows = []
    inputs = []
    legacy_keys = None
    if args.legacy_lists:
        parts = []
        source_meta = json.loads((EXP/'source/exp125_opdi_sources.json').read_text())
        source_rows = [r for r in source_meta['inputs'] if r.get('kind') == 'flight_list']
        for r in source_rows:
            name = f'flight_list_{r["month"].replace("-", "")}.parquet'
            path = Path(args.legacy_lists)/name
            assert sha(path) == r['sha256'], f'Original source hash differs: {name}'
            f = pd.read_parquet(path, columns=['id', 'dof', 'adep'])
            f = f[f.adep.isin(source_meta['airport_codes'])].copy()
            t = pd.to_datetime(f.dof, utc=True, errors='coerce')
            f = f[t.dt.strftime('%Y-%m').eq(r['month'])].drop_duplicates('id')
            f['_month'] = r['month']
            parts.append(f)
        joined = pd.concat(parts, ignore_index=True)
        assert joined.id.dtype == np.dtype('float64'), str(joined.id.dtype)
        def canonical(value):
            value = int(value) & ((1 << 64)-1)
            return value-(1 << 64) if value >= (1 << 63) else value
        legacy_keys = set(joined.id.map(canonical))
        lost = sum(int((~f.id.map(canonical).isin(legacy_keys)).sum()) for f in parts)
        (Path(args.out)/'id-loss.json').write_text(json.dumps(dict(
            source_hashes_match=True, original_concat_dtype=str(joined.id.dtype),
            flight_rows=len(joined), rounded_away_ids=lost,
            lost_share=lost/len(joined), mechanism='uint64 and int64 concat to float64 before canonical_flight_id'), indent=2)+'\n')
    for win in ('20260110_20260120', '20260709_20260719'):
        paths = [work/f'opdi_dep_flights_{win}.parquet', work/f'opdi_dep_events_{win}.parquet']
        fl, ev = [pd.read_parquet(p) for p in paths]
        if legacy_keys is not None:
            keys = ev.flight_id.astype('uint64').to_numpy().view('int64')
            ev = ev[pd.Series(keys, index=ev.index).isin(legacy_keys)]
        od = compact(fl, ev, airports)
        p = rk[(rk.MVT_TIME_UTC_mvt >= pd.Timestamp(win[:8], tz='UTC')) &
               (rk.MVT_TIME_UTC_mvt < pd.Timestamp(win[9:], tz='UTC'))].reset_index(drop=True)
        p['_row_ix'] = np.arange(len(p))
        for gate in ('old', 'wide'):
            for shift in (0, -7200, 7200):
                out = replay(p, od, gate == 'wide', shift)
                for a, g in out.groupby('ADEP_mvt'):
                    rows.append(dict(window=win, month=win[:6], airport=a, gate=gate, shift_s=shift,
                                     departures=len(g), matched=int(g.flight_id.notna().sum()),
                                     surface=int(g.has_surface.sum()), any_clock=int(g.has_event.sum()),
                                     no_adsb=int((~g.obs_has_clock).sum()), new_clock=int(g.new_clock.sum())))
                if shift == 0:
                    out[['MVT_ID_mvt', 'ADEP_mvt', 'flight_id', '_first_seen_delta_s', 'obs_has_clock',
                         'has_surface', 'has_event', 'new_clock']].to_parquet(work/f'gate-{gate}-{win}.parquet', index=False)
        inputs += [dict(path=str(p), sha256=sha(p)) for p in paths]
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True, parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(outdir/'gate-counts.csv', index=False)
    meta = dict(inputs=inputs, matcher_sha256=sha(EXP/'source/exp125_analyze.py'),
                event_rules='exp125: 5km or named ADEP, first event, 60s to 4h, first_seen altitude and agreement',
                gate_only_change='first_seen delta [-5400,300]s; FL50 delta [0,720]s or absent',
                total=table.groupby(['gate', 'shift_s'])[['departures', 'matched', 'surface', 'any_clock', 'new_clock']].sum().reset_index().to_dict('records'))
    (outdir/'gate-check.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta['total'], indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--class-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--legacy-lists', help='Replay the original 14-month mixed-integer concat before event lookup')
    run(ap.parse_args())
