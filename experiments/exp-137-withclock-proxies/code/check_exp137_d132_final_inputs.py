#!/usr/bin/env python3
"""Check exp-137 positive-point rule final-kernel reads, columns, support, and ID order."""
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def kaggle_user() -> str:
    """Read the account only when a Kaggle route is needed."""
    user = os.environ.get("PRC_KAGGLE_USER", "").strip()
    if not user:
        raise RuntimeError("Set PRC_KAGGLE_USER to your Kaggle account before using a Kaggle route")
    return user

ROOT = Path(__file__).resolve().parents[1]
EXP130 = Path('/tmp/prc-exp130-clean-input')
EXP136 = Path('/tmp/prc-exp136-input')
EXP137 = Path('/tmp/prc-exp137-input')
MAIN = Path('/path/to/competitions/prc-data-challenge-2026')
RUN = ROOT / 'kaggle/run-v4'
ID = 'MVT_ID_mvt'
SUPPORT = {
    '2026-01': ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH'],
    '2026-07': ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH', 'LEMD', 'LFPG', 'LIRF'],
}
EXPECTED = {'2026-01': 68716, '2026-07': 104823}


def fields(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def require(path: Path, columns: set[str]) -> None:
    missing = columns - fields(path)
    if missing:
        raise RuntimeError(f'{path.name}: missing fields {sorted(missing)}')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def exact(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f'{root}: expected one {name}, found {len(matches)}')
    return matches[0]


def main() -> None:
    kernel = RUN / 'final/kernel.py'
    metadata_path = RUN / 'final/kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    frozen = json.loads((ROOT / 'kaggle/run-v3/manifest.json').read_text())
    if frozen['d107_pass'] is not False:
        raise RuntimeError('the saved exp-137 verdict is not D107 FAIL')
    if metadata['is_private'] is not True or metadata['enable_gpu'] or metadata['enable_tpu'] or metadata['enable_internet']:
        raise RuntimeError('final kernel must be private, CPU-only, and offline')
    source_inventory = json.loads((EXP136 / 'source_inventory.json').read_text())
    if metadata['kernel_sources'] != source_inventory['point_source_kernels']:
        raise RuntimeError('final kernel point-source attachments differ from the checked list')
    if len(metadata['kernel_sources']) != 17:
        raise RuntimeError('final kernel must attach all 17 point-source kernels')
    if metadata['dataset_sources'] != [
        f'{kaggle_user()}/prc-exp130-clean-input', f'{kaggle_user()}/prc-exp136-input', f'{kaggle_user()}/prc-exp137-input'
    ]:
        raise RuntimeError('final kernel dataset sources changed')

    remote_dataset_files = {
        'exp130': EXP130,
        'exp136': EXP136,
        'exp137': EXP137,
    }
    remote_file_counts = {}
    remote_file_list_hashes = {}
    for name, dataset_root in remote_dataset_files.items():
        remote_list = RUN / f'remote-dataset-files/{name}.json'
        remote_records = json.loads(remote_list.read_text())
        remote = {record['name']: int(record['size']) for record in remote_records}
        local = {
            path.relative_to(dataset_root).as_posix(): path.stat().st_size
            for path in dataset_root.rglob('*')
            if path.is_file() and path.name != 'dataset-metadata.json'
            and '__pycache__' not in path.parts and path.suffix != '.pyc'
        }
        if local != remote:
            missing = sorted(set(local) - set(remote))
            extra = sorted(set(remote) - set(local))
            changed = sorted(name for name in set(local) & set(remote) if local[name] != remote[name])
            raise RuntimeError(f'{name}: remote/local files differ: missing={missing}, extra={extra}, size_changed={changed}')
        remote_file_counts[name] = len(remote)
        remote_file_list_hashes[name] = sha256(remote_list)

    exp130 = json.loads((EXP130 / 'input-manifest.json').read_text())
    exp137_inventory = json.loads((EXP137 / 'exp137_source_inventory.json').read_text())
    if exp130.get('ranking_target_read') is not False or exp137_inventory.get('ranking_target_read') is not False:
        raise RuntimeError('source manifests do not preserve ranking label isolation')
    if exp130.get('weather_used_or_downloaded') is not False or exp137_inventory.get('weather_used_or_downloaded') is not False:
        raise RuntimeError('source manifests do not preserve the weather exclusion')

    fit = EXP130 / 'fit.py'
    proxy = EXP137 / 'code/proxy_features_exp137.py'
    if not fit.is_file() or not proxy.is_file():
        raise RuntimeError('frozen fit or proxy source is missing')
    training_files = [EXP130 / f'month_2025-{m:02d}.parquet' for m in range(1, 13)]
    for path in training_files:
        require(path, {ID, 'month', 'y', 'ghost'})
    ranking_files = [EXP130 / f'month_{month}.parquet' for month in SUPPORT]
    for path in ranking_files:
        require(path, {ID, 'month', 'ghost'})
        if {'y', 'y_true'} & fields(path):
            raise RuntimeError(f'{path.name} contains a target label')
    support_rows = {}
    unsupported_january = {}
    for month, airports in SUPPORT.items():
        meta = EXP130 / f'rank_meta_{month}.parquet'
        require(meta, {ID, 'airport', 'rank_class', 'noac'})
        frame = pd.read_parquet(meta)
        if frame[ID].duplicated().any():
            raise RuntimeError(f'{month}: repeated ranking metadata IDs')
        with_clock = frame.rank_class.astype(str).eq('with-clock') & ~frame.noac.astype(bool)
        supported = with_clock & frame.airport.astype(str).isin(airports)
        count = int(supported.sum())
        if count != EXPECTED[month]:
            raise RuntimeError(f'{month}: support row count is {count}, expected {EXPECTED[month]}')
        support_rows[month] = count
        if month == '2026-01':
            unsupported_january = {
                str(airport): int(n)
                for airport, n in frame.loc[with_clock & ~frame.airport.astype(str).isin(airports)]
                    .groupby(frame.loc[with_clock & ~frame.airport.astype(str).isin(airports), 'airport'].astype(str), observed=True)
                    .size().items()
            }
    if unsupported_january != {'LEMD': 15881, 'LFPG': 1136, 'LIRF': 5700}:
        raise RuntimeError(f'unsupported January counts changed: {unsupported_january}')

    submitting_ids = pd.read_parquet(EXP130 / 'submitting_ids.parquet').sort_values('row_order', kind='stable')
    require(EXP137 / 'ranking_base_v24.parquet', {ID, 'row_order', 'p24'})
    ranking_base = pd.read_parquet(EXP137 / 'ranking_base_v24.parquet').sort_values('row_order', kind='stable')
    actual = pd.read_parquet(MAIN / 'data/raw/submitting.parquet', columns=[ID])
    id_order = submitting_ids[ID].to_numpy(dtype='int64')
    base_order = ranking_base[ID].to_numpy(dtype='int64')
    actual_order = actual[ID].to_numpy(dtype='int64')
    if not (len(set(id_order)) == len(id_order) == len(base_order) == len(actual_order)):
        raise RuntimeError('submission ID counts or uniqueness differ')
    if not ((id_order == base_order).all() and (id_order == actual_order).all()):
        raise RuntimeError('final kernel IDs are not in submitting.parquet order')
    if {'y', 'y_true'} & fields(EXP137 / 'ranking_base_v24.parquet'):
        raise RuntimeError('saved v24 ranking base contains a target label')

    require(EXP136 / 'proxy_departures.parquet', {ID, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'EOBT_1_flt', 'FLIGHT_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'month'})
    if 'TAXITIME_SEC_mvt' in fields(EXP136 / 'proxy_departures.parquet'):
        raise RuntimeError('departure source includes taxi-time target')
    require(EXP136 / 'proxy_arrivals.parquet', {ID, 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'RUNWAY_mvt', 'TAXITIME_SEC_mvt', 'month'})
    if not (EXP136 / 'stand_geometry.csv').is_file() or not (EXP136 / 'points_expected.json').is_file():
        raise RuntimeError('proxy geometry or point manifest is missing')

    code = kernel.read_text()
    if 'to_parquet' in code or 'write_parquet' in code:
        raise RuntimeError('final kernel writes a parquet')
    if 'read_parquet' not in code or 'np.savez_compressed' not in code:
        raise RuntimeError('final kernel input/output code changed')
    result = {
        'state': 'PASS',
        'kernel_slug': metadata['id'],
        'dataset_versions': {'prc-exp130-clean-input': 1, 'prc-exp136-input': 4, 'prc-exp137-input': 3},
        'dataset_sources': metadata['dataset_sources'],
        'point_source_kernel_count': len(metadata['kernel_sources']),
        'remote_dataset_files_match_local_names_and_sizes': True,
        'remote_dataset_file_counts': remote_file_counts,
        'remote_file_list_sha256': remote_file_list_hashes,
        'training_month_files': len(training_files),
        'ranking_month_files': len(ranking_files),
        'supported_rows_by_month': support_rows,
        'unsupported_january_rows_by_airport': unsupported_january,
        'submitting_rows': len(actual_order),
        'submitting_id_order_matches_data_and_v24': True,
        'ranking_base_columns': sorted(fields(EXP137 / 'ranking_base_v24.parquet')),
        '2026_target_label_read': False,
        'weather_used_or_downloaded': False,
        'candidate_parquet_created': False,
        'kernel_code_sha256': sha256(kernel),
        'kernel_metadata_sha256': sha256(metadata_path),
        'submitting_ids_sha256': sha256(EXP130 / 'submitting_ids.parquet'),
        'ranking_base_v24_sha256': sha256(EXP137 / 'ranking_base_v24.parquet'),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
