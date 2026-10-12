"""Match the corrected full extract and retain per-type times and distances.

exp-185 (finaldata): exp-155 code/match_full.py with its fixed inputs as arguments -- the month list
(--months), the flights file (--flights), the event folders (--events-dir, repeatable; every
events_*.parquet in them, as the original reads full/), the 2026 movement file (--ranking:
ranking.parquet or final_ranking.parquet) and the class file per 2026 month
(<class-root>/<month>-v25.parquet). Matching, gates and outputs are the original lines.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gate_check import matcher, sha

TYPES = ['exit-parking_position', 'exit-apron', 'entry-taxiway', 'entry-runway',
         'exit-deicing_pad', 'entry-deicing_pad', 'first_seen']


def month_path(raw, month):
    y, m = map(int, month.split('-'))
    end = pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)
    return raw/f'training_{month}-01_{end:%Y-%m-%d}.parquet'


def run(args):
    work, raw, out = Path(args.work), Path(args.raw), Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    MONTHS = args.months
    flights_path = Path(args.flights)
    flights = pd.read_parquet(flights_path)
    airports = pd.read_csv(work/'work/airports.csv').set_index('ident')
    for c in ('dof', 'first_seen', 'last_seen'):
        flights[c] = pd.to_datetime(flights[c], utc=True, errors='coerce')
    flights.id = flights.id.astype('uint64').to_numpy().view('int64')
    event_paths = sorted((p for d in args.events_dir for p in Path(d).glob('events_*.parquet')), key=lambda p: (p.name, str(p)))
    events = pd.concat([pd.read_parquet(p) for p in event_paths], ignore_index=True)
    events.event_time = pd.to_datetime(events.event_time, utc=True)
    events.flight_id = events.flight_id.astype('uint64').to_numpy().view('int64')
    events = events.drop_duplicates(['event_id', 'flight_id'])
    print('loaded', len(flights), 'flights', len(events), 'events', len(event_paths), 'windows', flush=True)
    x50 = events[events.type.eq('first-xing-fl50')].groupby('flight_id').event_time.min()
    od = flights.rename(columns={'id': 'flight_id', 'dof': 'dof_utc', 'first_seen': 'first_seen_utc', 'last_seen': 'last_seen_utc'})
    od['first_climb_utc'] = od.flight_id.map(x50)
    for c in ('exit_parking_position_utc', 'exit_apron_utc', 'entry_taxiway_utc', 'takeoff_utc',
              'entry_runway_utc', 'first_seen_event_utc'):
        od[c] = pd.NaT
    od['first_seen_ground_altitude_ft'] = np.nan
    od['first_seen_ground_distance_km'] = np.nan
    funcs = matcher(True)
    output_meta = []
    count_rows, event_rows, all_event_rows = [], [], []
    classes = pd.concat([pd.read_parquet(Path(args.class_root)/f'{m}-v25.parquet',
                                       columns=['MVT_ID_mvt', 'obs_has_clock']) for m in MONTHS if m.startswith('2026')])
    for month in MONTHS:
        print('start month', month, flush=True)
        path = Path(args.ranking) if month.startswith('2026') else month_path(raw, month)
        cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'CALLSIGN_flt', 'FLIGHT_mvt',
                'AIRCRAFT_TYPE_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt']
        p = pd.read_parquet(path, columns=cols)
        p = p[p.PHASE_mvt.eq('DEP') & p.ADEP_mvt.isin(od.adep.unique()) &
              p.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').eq(month)].copy().reset_index(drop=True)
        p['_row_ix'] = np.arange(len(p))
        odmonth = od[od.dof_utc.dt.strftime('%Y-%m').eq(month)]
        matches, counts = funcs['choose_matches'](p, odmonth)
        matches['flight_id'] = pd.array(matches.flight_id, dtype='Int64')
        p = p.merge(matches[['_row_ix', 'flight_id', '_first_seen_delta_s']], on='_row_ix', how='left', validate='1:1')
        # A single OPDI identity can map to multiple supplied movements. Keep it visible.
        multi = p.flight_id.notna() & p.flight_id.duplicated(keep=False)
        p['duplicate_opdi_identity'] = multi
        ev = p[['MVT_ID_mvt', 'flight_id', 'MVT_TIME_UTC_mvt', 'ADEP_mvt']].dropna(subset=['flight_id'])
        # Describe every retained event type, including later identity events.
        all_ev = ev.merge(events, on='flight_id')
        all_ev['lead_s'] = (all_ev.MVT_TIME_UTC_mvt-all_ev.event_time).dt.total_seconds()
        all_first = all_ev.sort_values('event_time').drop_duplicates(['MVT_ID_mvt', 'type'])
        for (a, typ), g in all_first.groupby(['ADEP_mvt', 'type']):
            all_event_rows.append(dict(month=month, airport=a, event=typ, n=len(g),
                                       offset_to_takeoff_median_s=float(g.lead_s.median()),
                                       offset_p05_s=float(g.lead_s.quantile(.05)), offset_p95_s=float(g.lead_s.quantile(.95)),
                                       distance_median_m=float(g.dist_m.median()), distance_p95_m=float(g.dist_m.quantile(.95))))
        del all_ev, all_first
        # Preserve signed 64-bit IDs: a nullable integer avoids float rounding at >2**53.
        # Flight IDs in a left merge retain int64 only through a nullable dtype.
        ev = ev.merge(events[events.type.isin(TYPES) & events.dist_m.le(6000)], on='flight_id')
        ev['lead_s'] = (ev.MVT_TIME_UTC_mvt-ev.event_time).dt.total_seconds()
        ev = ev[ev.lead_s.between(30, 10800)].copy()
        # first_seen needs a low-altitude event; the three surface types do not.
        elevation = ev.ADEP_mvt.map(airports.elevation_ft)
        ev = ev[~ev.type.eq('first_seen') | (ev.altitude-elevation).abs().le(500)]
        first = ev.sort_values('event_time').drop_duplicates(['MVT_ID_mvt', 'type'])
        p['airport'] = p.ADEP_mvt
        p['month'] = month
        p['utc_day'] = p.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d')
        p['aobt_taxi_s'] = (p.MVT_TIME_UTC_mvt-p.AOBT_3_flt).dt.total_seconds()
        for typ in TYPES:
            s = first[first.type.eq(typ)].set_index('MVT_ID_mvt')
            key = typ.replace('-', '_')
            for col, name in [('lead_s', 'lead_s'), ('dist_m', 'dist_m'), ('event_time', 'time'), ('altitude', 'altitude_ft')]:
                p[f'{key}_{name}'] = p.MVT_ID_mvt.map(s[col])
        clock_cols = [f'{typ.replace("-", "_")}_lead_s' for typ in TYPES[:3]]
        p['has_surface'] = p[clock_cols].notna().any(axis=1)
        if month.startswith('2026'):
            p = p.merge(classes, on='MVT_ID_mvt', how='left', validate='1:1')
            assert p.obs_has_clock.notna().all()
        else:
            # These are extraction counts. Exact route-class clocks are joined by the screen.
            p['obs_has_clock'] = pd.NA
        for a, g in p.groupby('airport'):
            count_rows.append(dict(month=month, airport=a, departures=len(g), matched=int(g.flight_id.notna().sum()),
                                   surface=int(g.has_surface.sum()), duplicate_identity_rows=int(g.duplicate_opdi_identity.sum()),
                                   new_clock=int((g.has_surface & ~g.obs_has_clock).sum()) if month.startswith('2026') else None))
        for (a, typ), g in first.groupby(['ADEP_mvt', 'type']):
            event_rows.append(dict(month=month, airport=a, event=typ, n=len(g),
                                   offset_to_takeoff_median_s=float(g.lead_s.median()),
                                   offset_p05_s=float(g.lead_s.quantile(.05)), offset_p95_s=float(g.lead_s.quantile(.95)),
                                   distance_median_m=float(g.dist_m.median()), distance_p95_m=float(g.dist_m.quantile(.95))))
        dest = out/f'events-{month}.parquet'
        p.drop(columns=['_row_ix', 'CALLSIGN_flt', 'FLIGHT_mvt', 'PHASE_mvt']).to_parquet(dest, index=False, compression='zstd')
        output_meta.append(dict(file=str(dest), rows=len(p), bytes=dest.stat().st_size, sha256=sha(dest), raw=str(path), raw_sha256=sha(path)))
        print(month, len(p), 'matched', p.flight_id.notna().sum(), 'surface', p.has_surface.sum(), flush=True)
    pd.DataFrame(count_rows).to_csv(out/'coverage.csv', index=False)
    pd.DataFrame(event_rows).to_csv(out/'event-offset-distance.csv', index=False)
    pd.DataFrame(all_event_rows).to_csv(out/'all-event-offset-distance.csv', index=False)
    manifest = dict(outputs=output_meta,
                    flights=dict(path=str(flights_path), sha256=sha(flights_path)),
                    event_inputs=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in event_paths],
                    coverage_sha256=sha(out/'coverage.csv'), offsets_sha256=sha(out/'event-offset-distance.csv'),
                    all_type_offsets_sha256=sha(out/'all-event-offset-distance.csv'),
                    event_gate='First event per type within 6 km of ADEP, 30s to 3h before takeoff; first_seen within 500ft of airport elevation',
                    identity_gate='Pinned exp125 matcher with only wide first_seen/FL50 gate changed; same UTC DOF')
    (out/'extract-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--class-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--months', nargs='+', required=True)
    ap.add_argument('--flights', required=True)
    ap.add_argument('--events-dir', action='append', required=True)
    ap.add_argument('--ranking', required=True)
    run(ap.parse_args())
