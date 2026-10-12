"""Build label-free first or last OPDI event inputs from the checked unit-1 data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TYPES = ['exit-parking_position', 'exit-apron', 'entry-taxiway']
ID = 'MVT_ID_mvt'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--episode', choices=['first', 'last'], required=True)
    a.add_argument('--member-input-root', type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out/f'opdi-{args.episode}.parquet'
    if dest.exists() or dest.with_suffix('.json').exists():
        raise RuntimeError('input exists; use a new output')
    manifest = json.loads((args.root/'results/extract/extract-manifest.json').read_text())
    source = []
    parts = []
    if args.episode == 'last':
        event_parts = []
        for r in manifest['event_inputs']:
            path = args.root/r['path']
            assert sha(path) == r['sha256']
            d = pd.read_parquet(path, columns=['flight_id', 'type', 'event_time', 'dist_m'])
            event_parts.append(d[d.type.isin(TYPES) & d.dist_m.le(6000)])
            source.append(dict(path='remote:'+str(path), sha256=r['sha256']))
        events = pd.concat(event_parts, ignore_index=True)
        assert str(events.flight_id.dtype) == 'int64'
    for r in manifest['outputs']:
        path = args.root/r['file']
        assert sha(path) == r['sha256']
        common = [ID, 'airport', 'utc_day', 'aobt_taxi_s', 'duplicate_opdi_identity']
        event_cols = [t.replace('-', '_')+'_'+s for t in TYPES for s in ['lead_s', 'dist_m']]
        cols = common+event_cols if args.episode == 'first' else common+['flight_id', 'MVT_TIME_UTC_mvt']
        d = pd.read_parquet(path, columns=cols)
        supplied_ids = d[ID].to_numpy()
        d[ID] = d[ID].astype('int64')
        assert np.array_equal(supplied_ids, d[ID].to_numpy()), 'movement ID is not an exact integer'
        if args.episode == 'last':
            matched = d[[ID, 'flight_id', 'MVT_TIME_UTC_mvt']].dropna(subset=['flight_id'])
            joined = matched.merge(events, on='flight_id')
            joined['lead_s'] = (joined.MVT_TIME_UTC_mvt-joined.event_time).dt.total_seconds()
            joined = joined[joined.lead_s.between(30, 10800)]
            last = joined.sort_values('event_time').drop_duplicates([ID, 'type'], keep='last')
            for t in TYPES:
                src = last[last.type.eq(t)].set_index(ID)
                for s in ['lead_s', 'dist_m']:
                    d[t.replace('-', '_')+'_'+s] = d[ID].map(src[s])
            d = d[common+event_cols]
        assert d[ID].is_unique and str(d[ID].dtype) == 'int64'
        parts.append(d)
        source.append(dict(path='remote:'+str(path), sha256=r['sha256']))
        print(args.episode, r['file'], len(d), flush=True)
    result = pd.concat(parts, ignore_index=True)
    assert result[ID].is_unique
    # Two source-month movements have a UTC date in the preceding month. The
    # unit-1 calendar filter omitted them. Keep all member rows and represent
    # those uncovered identities as no event, without reading a taxi label.
    calendar_edges = []
    for month in [f'2025-{m:02d}' for m in range(1, 13)]+['2026-01', '2026-07']:
        paths = list(args.member_input_root.rglob(f'month_{month}.parquet'))
        assert len(paths) == 1
        p = paths[0]
        d = pd.read_parquet(p, columns=[ID, 'airport', 'utc_day'])
        ids = d[ID].astype('int64')
        assert np.array_equal(d[ID].to_numpy(), ids.to_numpy())
        d[ID] = ids
        missing = d[~d[ID].isin(result[ID])].copy()
        if len(missing):
            assert not missing.utc_day.astype(str).str.startswith(month).any(), 'uncovered member row within its source month'
            missing['duplicate_opdi_identity'] = False
            calendar_edges.extend(missing.to_dict('records'))
            result = pd.concat([result, missing.reindex(columns=result.columns)], ignore_index=True)
        source.append(dict(path='remote:'+str(p), sha256=sha(p), projection=[ID, 'airport', 'utc_day']))
    result[ID] = result[ID].astype('int64')
    result['duplicate_opdi_identity'] = result.duplicate_opdi_identity.astype(bool)
    assert result[ID].is_unique
    assert not any('TAXITIME' in c or c == 'y' for c in result)
    result.to_parquet(dest, index=False, compression='zstd')
    record = dict(episode=args.episode, rows=len(result), columns=list(result.columns),
                  label_free=True, label_columns_read=False, source=source,
                  output_sha256=sha(dest), code_sha256=sha(Path(__file__)),
                  storage='remote:'+str(dest), types=TYPES, no_event_calendar_edges=calendar_edges)
    dest.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', rows=len(result), sha256=record['output_sha256'])), flush=True)


if __name__ == '__main__':
    main()
