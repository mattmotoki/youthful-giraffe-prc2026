"""Derive the two R2 boundary exceptions from caller-owned challenge files.

    python scripts/exp148_missing_features.py --data-dir DIR --out NEW.json

These are departures at 23:59:57 UTC on the last day of February and August
2025, stored in the following month's raw file. Only the caller's output holds
IDs. The public fitting source reads it through PRC_R2_NO_FEATURES.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ID = 'MVT_ID_mvt'


def derive(data_dir):
    found, sources, by_month = [], {}, {}
    for month in ('2025-03', '2025-09'):
        start = pd.Timestamp(month + '-01', tz='UTC')
        stop = start + pd.offsets.MonthBegin(1)
        path = Path(data_dir) / f'training_{start:%Y-%m-%d}_{stop:%Y-%m-%d}.parquet'
        frame = pd.read_parquet(path, columns=[ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
        time = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
        ids = frame.loc[(frame.PHASE_mvt == 'DEP') & (time == start - pd.Timedelta(seconds=3)), ID]
        if len(ids) != 1:
            raise ValueError(f'{month}: expected one recorded boundary departure, found {len(ids)}')
        value = float(ids.iloc[0])
        if not value.is_integer():
            raise ValueError('A boundary ID is not an integer')
        found.append(int(value))
        by_month[month] = [int(value)]
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'missing_feature_ids': sorted(found), 'sources_sha256': sources, 'missing_feature_ids_by_file_month': by_month}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Output exists; use a new caller-owned path')
    data = derive(a.data_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=2) + '\n')
    print('Wrote two caller-owned boundary exceptions')


if __name__ == '__main__':
    main()
