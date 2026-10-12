#!/usr/bin/env python3
"""Build the label-free exp-130 ranking-exposure census from saved fold predictions."""
from __future__ import annotations

import csv
import hashlib
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
MAIN = Path('/path/to/competitions/prc-data-challenge-2026')
ROOT_CODE = MAIN
sys.path.insert(0, str(MAIN / 'experiments/exp-047-queue/code'))
sys.path.insert(0, str(MAIN))
import base as BS  # noqa: E402
import prc.data as D  # noqa: E402

ID = 'MVT_ID_mvt'
CLASS_ORDER = ('with-clock', 'no-clock_ordinary', 'noac')
MONTH_TO_FOLD = {'2026-01': 'january', '2026-07': 'july_loo'}
TAIL_S = 3600.0
CLASS_DIR = MAIN / 'artifacts/exp-046/class'
BLOCK_DIR = MAIN / 'artifacts/exp-047/block'
V22_VECTOR = MAIN / 'experiments/exp-046-surfacefit/work/d-seasonal-bag10-255-free.npz'
CENSUS_REF = MAIN / 'experiments/exp-047-queue/analysis/census-v22.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def mse(values: np.ndarray) -> float | None:
    return float(np.mean(values * values)) if len(values) else None


def rmse(values: np.ndarray) -> float | None:
    v = mse(values)
    return math.sqrt(v) if v is not None else None


def stat_cols(prefix: str, rows: np.ndarray, out: dict[str, object]) -> None:
    out[f'{prefix}_rows'] = int(len(rows))
    out[f'{prefix}_mse_s2'] = mse(rows)
    out[f'{prefix}_rmse_s'] = rmse(rows)


def read_fold_stats() -> tuple[dict[tuple[str, str, str], dict], dict, list[Path]]:
    folds = BS.load(with_v12=True)
    v22 = np.load(V22_VECTOR)
    ref = __import__('json').loads(CENSUS_REF.read_text())
    cell_stats: dict[tuple[str, str, str], dict] = {}
    fold_checks: dict[str, dict] = {}
    source_paths: list[Path] = [V22_VECTOR, CENSUS_REF,
                                MAIN / 'experiments/exp-047-queue/code/base.py',
                                MAIN / 'prc/data.py',
                                MAIN / 'prc/_private_clean_score_rows.py']

    for fold in ('january', 'july_loo'):
        frame = folds[fold]
        month = BS.MONTH_OF[fold]
        ids = frame[ID].to_numpy()
        block_path = BLOCK_DIR / f'{month}.parquet'
        block = pd.read_parquet(block_path, columns=[ID, 'obs_has_clock'])
        if block[ID].duplicated().any():
            raise SystemExit(f'{fold}: duplicate block IDs')
        clock_map = block.set_index(ID)['obs_has_clock']
        clock_series = pd.Series(ids).map(clock_map)
        if clock_series.isna().any():
            raise SystemExit(f'{fold}: block misses {int(clock_series.isna().sum())} fold IDs')
        clock = clock_series.to_numpy(bool)

        movement = D.load_month(month, columns=[ID, 'PHASE_mvt', 'AIRCRAFT_TYPE_mvt', 'AOBT_3_flt'])
        movement = movement[movement['PHASE_mvt'].eq('DEP')]
        if movement[ID].duplicated().any():
            raise SystemExit(f'{fold}: duplicate movement IDs')
        if not np.isin(ids, movement[ID].to_numpy()).all():
            raise SystemExit(f'{fold}: movement rows miss fold IDs')
        noac_ids = movement.loc[movement['AIRCRAFT_TYPE_mvt'].isna(), ID].to_numpy()
        noac = np.isin(ids, noac_ids)
        aobt_present = pd.Series(ids).map(
            movement.set_index(ID)['AOBT_3_flt']).notna().to_numpy()
        if len(frame) != len(ids) or len(v22[fold]) != len(ids):
            raise SystemExit(f'{fold}: fold/vector length mismatch')

        y = frame['y_true'].to_numpy('float64')
        r12 = frame['r12'].to_numpy('float64')
        d22 = np.asarray(v22[fold], dtype='float64')
        apt = frame['apt'].astype(str).to_numpy()
        with_clock = clock & ~noac
        residual_v22 = r12 - np.where(with_clock, d22, 0.0)
        excluded = np.isin(ids, list(D.excluded_ids(fold)))
        clean = ~excluded
        classes = np.where(noac, 'noac', np.where(clock, 'with-clock', 'no-clock_ordinary'))
        if not (np.isfinite(y).all() and np.isfinite(r12).all()
                and np.isfinite(d22).all() and np.isfinite(residual_v22).all()):
            raise SystemExit(f'{fold}: non-finite fold labels, predictions or residuals')

        want = float(ref['fold_totals'][fold]['rmse_v22'])
        got = rmse(residual_v22)
        if got is None or abs(got - want) > 1e-6:
            raise SystemExit(f'{fold}: v22 RMSE {got} does not match census {want}')
        clean_got = D.rmse_clean(y, y - residual_v22, ids, fold)
        counts = {name: int((classes == name).sum()) for name in CLASS_ORDER}
        ref_cells = ref['the_two_halves'][fold]
        expected_counts = {
            'with-clock': int(ref_cells['with_clock']['rows']),
            'no-clock_ordinary': int(ref_cells['no_clock_only']['rows']),
            'noac': int(ref_cells['noac_any']['rows'])}
        if counts != expected_counts:
            raise SystemExit(f'{fold}: class counts {counts} differ from census {expected_counts}')
        fold_checks[fold] = {
            'valid_month': month,
            'rows': int(len(frame)),
            'v22_rmse_rebuilt_s': got,
            'v22_rmse_census_s': want,
            'v22_rmse_abs_error_s': abs(got - want),
            'v22_rmse_clean_s': clean_got,
            'excluded_rows': int(excluded.sum()),
            'rows_by_class': counts,
            'v22_method': 'r12 - d-seasonal on exp-047 with-clock ordinary rows; r12 on the complement'}

        for airport in sorted(np.unique(apt)):
            airport_mask = apt == airport
            for class_name in CLASS_ORDER:
                mask = airport_mask & (classes == class_name)
                residual = residual_v22[mask]
                cell_stats[(fold, airport, class_name)] = {
                    'fold_rows_for_class': int(mask.sum()),
                    'fold_clean_rows_for_class': int((mask & clean).sum()),
                    'fold_excluded_rows_for_class': int((mask & excluded).sum()),
                    'fold_v22_sse_s2': float(np.sum(residual * residual)) if len(residual) else None,
                    'fold_v22_mse_s2_per_row': mse(residual),
                    'fold_v22_rmse_s': rmse(residual),
                    'fold_v22_clean_sse_s2': (float(np.sum(residual_v22[mask & clean] ** 2))
                                              if (mask & clean).any() else None),
                    'fold_v22_clean_mse_s2_per_row': mse(residual_v22[mask & clean]),
                    'fold_v22_clean_rmse_s': rmse(residual_v22[mask & clean])}

            # p12 is a prediction for every ordinary row. Keep the clock/no-clock
            # subsets as context, while the required all-ordinary metric includes both.
            ordinary = airport_mask & ~noac
            body = ordinary & (y <= TAIL_S)
            tail = ordinary & (y > TAIL_S)
            no_clock_ordinary = ordinary & ~clock
            with_clock_ordinary = ordinary & clock
            p12_all = r12[ordinary]
            p12_body = r12[body]
            p12_tail = r12[tail]
            p12_nc = r12[no_clock_ordinary]
            p12_wc = r12[with_clock_ordinary]
            p12_aobt_body = r12[body & aobt_present]
            clean_ordinary = ordinary & clean
            clean_body = body & clean
            clean_tail = tail & clean
            clean_nc = no_clock_ordinary & clean
            clean_wc = with_clock_ordinary & clean
            clean_aobt_body = body & aobt_present & clean
            diag = {
                'fold_p12_all_ordinary_rows': int(ordinary.sum()),
                'fold_p12_no_clock_ordinary_rows': int(no_clock_ordinary.sum()),
                'fold_p12_with_clock_ordinary_rows': int(with_clock_ordinary.sum()),
                'fold_p12_mse_all_ordinary_s2': mse(p12_all),
                'fold_p12_rmse_all_ordinary_s': rmse(p12_all),
                'fold_p12_body_rows': int(body.sum()),
                'fold_p12_body_mse_s2': mse(p12_body),
                'fold_p12_body_rmse_s': rmse(p12_body),
                'fold_p12_tail_rows': int(tail.sum()),
                'fold_p12_tail_mse_s2': mse(p12_tail),
                'fold_p12_tail_rmse_s': rmse(p12_tail),
                'fold_p12_mse_no_clock_ordinary_s2': mse(p12_nc),
                'fold_p12_rmse_no_clock_ordinary_s': rmse(p12_nc),
                'fold_p12_mse_with_clock_ordinary_s2': mse(p12_wc),
                'fold_p12_rmse_with_clock_ordinary_s': rmse(p12_wc),
                'fold_p12_aobt_present_body_rows': int((body & aobt_present).sum()),
                'fold_p12_aobt_present_body_mse_s2': mse(p12_aobt_body),
                'fold_p12_aobt_present_body_rmse_s': rmse(p12_aobt_body),
                'fold_clean_p12_all_ordinary_rows': int(clean_ordinary.sum()),
                'fold_clean_p12_no_clock_ordinary_rows': int(clean_nc.sum()),
                'fold_clean_p12_with_clock_ordinary_rows': int(clean_wc.sum()),
                'fold_clean_p12_mse_all_ordinary_s2': mse(r12[clean_ordinary]),
                'fold_clean_p12_rmse_all_ordinary_s': rmse(r12[clean_ordinary]),
                'fold_clean_p12_body_rows': int(clean_body.sum()),
                'fold_clean_p12_body_mse_s2': mse(r12[clean_body]),
                'fold_clean_p12_body_rmse_s': rmse(r12[clean_body]),
                'fold_clean_p12_tail_rows': int(clean_tail.sum()),
                'fold_clean_p12_tail_mse_s2': mse(r12[clean_tail]),
                'fold_clean_p12_tail_rmse_s': rmse(r12[clean_tail]),
                'fold_clean_p12_mse_no_clock_ordinary_s2': mse(r12[clean_nc]),
                'fold_clean_p12_rmse_no_clock_ordinary_s': rmse(r12[clean_nc]),
                'fold_clean_p12_mse_with_clock_ordinary_s2': mse(r12[clean_wc]),
                'fold_clean_p12_rmse_with_clock_ordinary_s': rmse(r12[clean_wc]),
                'fold_clean_p12_aobt_present_body_rows': int(clean_aobt_body.sum()),
                'fold_clean_p12_aobt_present_body_mse_s2': mse(r12[clean_aobt_body]),
                'fold_clean_p12_aobt_present_body_rmse_s': rmse(r12[clean_aobt_body])}
            # Only the no-clock ordinary output cell needs this p12 diagnostic.
            cell_stats[(fold, airport, 'no-clock_ordinary')].update(diag)

        source_paths.append(block_path)
        source_paths.append(D.month_file(month))
    return cell_stats, fold_checks, source_paths


def read_rank_cells() -> tuple[dict[tuple[str, str, str], int], list[str], dict[str, dict], list[Path]]:
    rank_path = D.data_dir() / 'ranking.parquet'
    submitting_path = D.data_dir() / 'submitting.parquet'
    # Do not load TAXITIME_SEC_mvt on the ranking side. The submission file supplies IDs only.
    submitted = pd.read_parquet(submitting_path, columns=[ID])
    if submitted[ID].duplicated().any():
        raise SystemExit('submitting IDs are not unique')
    submitted_ids = submitted[ID]
    rank = D.load_ranking(columns=[ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt'])
    rank = rank[rank[ID].isin(submitted_ids)].copy()
    if rank[ID].duplicated().any() or len(rank) != len(submitted_ids):
        raise SystemExit(f'ranking/submitting ID mismatch: {len(rank)} vs {len(submitted_ids)}')
    if not rank['PHASE_mvt'].eq('DEP').all():
        raise SystemExit('submitting IDs include a non-departure row')
    if set(rank['month'].unique()) != set(MONTH_TO_FOLD):
        raise SystemExit(f"unexpected ranking months: {sorted(rank['month'].unique())}")
    if rank['ADEP_mvt'].isna().any():
        raise SystemExit('ranking departure has no airport')

    for month in MONTH_TO_FOLD:
        class_path = CLASS_DIR / f'{month}-v25.parquet'
        class_frame = pd.read_parquet(
            class_path, columns=[ID, 'obs_has_clock', 'noac', 'occupied_by_exp_046'])
        if class_frame[ID].duplicated().any():
            raise SystemExit(f'{month}: duplicate rows in v25 class vector')
        rmonth = rank.loc[rank['month'].eq(month), [ID]]
        joined = rmonth.merge(class_frame, on=ID, how='left', validate='one_to_one', indicator=True)
        if (len(joined) != len(rmonth) or not joined['_merge'].eq('both').all()
                or not np.array_equal(joined[ID].to_numpy(), rmonth[ID].to_numpy())):
            raise SystemExit(f'{month}: rank rows and v25 class vector do not match exactly')
        clock = joined['obs_has_clock'].fillna(False).to_numpy(bool)
        noac = joined['noac'].fillna(False).to_numpy(bool)
        occupied = joined['occupied_by_exp_046'].fillna(False).to_numpy(bool)
        if not np.array_equal(occupied, clock & ~noac):
            raise SystemExit(f'{month}: v25 served class differs from obs_has_clock & ~noac')
        rank_mask = rank['month'].eq(month).to_numpy()
        type_noac = rank.loc[rank_mask, 'AIRCRAFT_TYPE_mvt'].isna().to_numpy()
        if not np.array_equal(noac, type_noac):
            raise SystemExit(f'{month}: v25 noac differs from missing ranking aircraft type')
        rank.loc[rank_mask, 'rank_class'] = np.where(
            noac, 'noac', np.where(occupied, 'with-clock', 'no-clock_ordinary'))

    rank['airport'] = rank['ADEP_mvt'].astype(str)
    group = rank.groupby(['month', 'airport', 'rank_class'], observed=True).size()
    airports = sorted(rank['airport'].unique())
    counts = {key: int(value) for key, value in group.items()}
    month_counts = {
        month: {'rows': int((rank['month'] == month).sum()),
                'by_class': {name: int(((rank['month'] == month)
                                        & (rank['rank_class'] == name)).sum())
                             for name in CLASS_ORDER}}
        for month in MONTH_TO_FOLD}
    return counts, airports, month_counts, [rank_path, submitting_path] + [
        CLASS_DIR / f'{month}-v25.parquet' for month in MONTH_TO_FOLD]


def main() -> None:
    fold_stats, fold_checks, fold_sources = read_fold_stats()
    rank_counts, airports, rank_months, rank_sources = read_rank_cells()
    total_rank_rows = sum(v['rows'] for v in rank_months.values())
    if total_rank_rows != 344_841:
        raise SystemExit(f'expected 344841 submission rows, found {total_rank_rows}')
    total_clock = sum(v['by_class']['with-clock'] for v in rank_months.values())
    total_noac = sum(v['by_class']['noac'] for v in rank_months.values())
    total_no_clock = sum(v['by_class']['no-clock_ordinary'] for v in rank_months.values())
    if (total_clock, total_no_clock + total_noac) != (196_256, 148_585):
        raise SystemExit(f'rank classes differ from v22 freeze: {total_clock}, {total_no_clock}, {total_noac}')

    rows: list[dict[str, object]] = []
    # Include all airport x month x class cells, including zero-exposure cells.
    for month, airport, class_name in itertools.product(
            sorted(MONTH_TO_FOLD), airports, CLASS_ORDER):
        fold = MONTH_TO_FOLD[month]
        rank_n = rank_counts.get((month, airport, class_name), 0)
        stat = fold_stats.get((fold, airport, class_name), {
            'fold_rows_for_class': 0, 'fold_clean_rows_for_class': 0,
            'fold_excluded_rows_for_class': 0, 'fold_v22_sse_s2': None,
            'fold_v22_mse_s2_per_row': None, 'fold_v22_rmse_s': None,
            'fold_v22_clean_sse_s2': None,
            'fold_v22_clean_mse_s2_per_row': None, 'fold_v22_clean_rmse_s': None})
        fold_mse = stat['fold_v22_mse_s2_per_row']
        fold_clean_mse = stat['fold_v22_clean_mse_s2_per_row']
        support_rows = int(stat['fold_rows_for_class'])
        clean_support_rows = int(stat['fold_clean_rows_for_class'])
        cell_mass_all = (float(rank_n * fold_mse) if fold_mse is not None
                         else (None if rank_n else 0.0))
        cell_mass_clean = (float(rank_n * fold_clean_mse) if fold_clean_mse is not None
                           else (None if rank_n else 0.0))
        rec = {
            'ranking_month': month,
            'matched_fold': fold,
            'airport': airport,
            'class': class_name,
            'ranking_rows': rank_n,
            'share_of_ranking_rows': rank_n / total_rank_rows,
            'share_of_no_clock_ordinary_ranking_rows': (
                rank_n / total_no_clock if class_name == 'no-clock_ordinary' else 0.0),
            **stat,
            'fold_support_status': 'present' if support_rows else 'none',
            'fold_clean_support_status': 'present' if clean_support_rows else 'none',
            'estimated_ranking_sse_all_s2': cell_mass_all,
            'estimated_ranking_sse_clean_s2': cell_mass_clean,
        }
        rows.append(rec)

    estimated_mass_all = sum(float(r['estimated_ranking_sse_all_s2']) for r in rows
                              if r['estimated_ranking_sse_all_s2'] is not None)
    estimated_mass_clean = sum(float(r['estimated_ranking_sse_clean_s2']) for r in rows
                               if r['estimated_ranking_sse_clean_s2'] is not None)
    if not estimated_mass_all > 0 or not estimated_mass_clean > 0:
        raise SystemExit('estimated ranking squared error is not positive')
    supported_rank_rows = sum(int(r['ranking_rows']) for r in rows
                              if r['fold_v22_mse_s2_per_row'] is not None)
    clean_supported_rank_rows = sum(int(r['ranking_rows']) for r in rows
                                    if r['fold_v22_clean_mse_s2_per_row'] is not None)
    unsupported_rank_rows = total_rank_rows - supported_rank_rows
    clean_unsupported_rank_rows = total_rank_rows - clean_supported_rank_rows
    for rec in rows:
        mass_all = rec['estimated_ranking_sse_all_s2']
        mass_clean = rec['estimated_ranking_sse_clean_s2']
        rec['share_of_estimated_ranking_sse_all'] = (
            float(mass_all) / estimated_mass_all if mass_all is not None else None)
        rec['share_of_estimated_ranking_sse_clean'] = (
            float(mass_clean) / estimated_mass_clean if mass_clean is not None else None)
        rec['estimated_mse_contribution_all_s2_per_ranking_row'] = (
            float(mass_all) / total_rank_rows if mass_all is not None else None)
        rec['estimated_mse_contribution_clean_s2_per_ranking_row'] = (
            float(mass_clean) / total_rank_rows if mass_clean is not None else None)
    out_path = EXP / 'census.csv'
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else (f'{v:.10f}' if isinstance(v, float) else v))
                             for k, v in row.items()})

    # Roll up the estimated error by airport and class using the ranking's own cell counts.
    summary: list[dict[str, object]] = []
    for airport, class_name in itertools.product(airports, CLASS_ORDER):
        grp = [r for r in rows if r['airport'] == airport and r['class'] == class_name]
        n_rank = sum(int(r['ranking_rows']) for r in grp)
        supported_all = [r for r in grp if r['fold_v22_mse_s2_per_row'] is not None]
        supported_clean = [r for r in grp if r['fold_v22_clean_mse_s2_per_row'] is not None]
        supported_n_all = sum(int(r['ranking_rows']) for r in supported_all)
        supported_n_clean = sum(int(r['ranking_rows']) for r in supported_clean)
        mass_all = sum(float(r['estimated_ranking_sse_all_s2']) for r in supported_all)
        mass_clean = sum(float(r['estimated_ranking_sse_clean_s2']) for r in supported_clean)
        summary.append({
            'airport': airport,
            'class': class_name,
            'ranking_rows': n_rank,
            'supported_ranking_rows_all': supported_n_all,
            'unsupported_ranking_rows_all': n_rank - supported_n_all,
            'estimated_mse_all_s2_per_supported_row': mass_all / supported_n_all if supported_n_all else None,
            'share_of_estimated_sse_all': mass_all / estimated_mass_all if supported_n_all else None,
            'supported_ranking_rows_clean': supported_n_clean,
            'unsupported_ranking_rows_clean': n_rank - supported_n_clean,
            'estimated_mse_clean_s2_per_supported_row': mass_clean / supported_n_clean if supported_n_clean else None,
            'share_of_estimated_sse_clean': mass_clean / estimated_mass_clean if supported_n_clean else None})
    summary.sort(key=lambda r: (CLASS_ORDER.index(r['class']), str(r['airport'])))
    top_all = sorted((r for r in rows if r['ranking_rows'] and r['estimated_ranking_sse_all_s2'] is not None),
                     key=lambda r: (-float(r['share_of_estimated_ranking_sse_all']),
                                    str(r['ranking_month']), str(r['airport']), str(r['class'])))[:10]
    top_clean = sorted((r for r in rows if r['ranking_rows'] and r['estimated_ranking_sse_clean_s2'] is not None),
                       key=lambda r: (-float(r['share_of_estimated_ranking_sse_clean']),
                                      str(r['ranking_month']), str(r['airport']), str(r['class'])))[:10]
    overall_mse_all = estimated_mass_all / supported_rank_rows
    overall_mse_clean = estimated_mass_clean / clean_supported_rank_rows
    class_summary = []
    for class_name in CLASS_ORDER:
        grp = [r for r in rows if r['class'] == class_name]
        n_rank = sum(int(r['ranking_rows']) for r in grp)
        supported_all = [r for r in grp if r['fold_v22_mse_s2_per_row'] is not None]
        supported_clean = [r for r in grp if r['fold_v22_clean_mse_s2_per_row'] is not None]
        supported_n_all = sum(int(r['ranking_rows']) for r in supported_all)
        supported_n_clean = sum(int(r['ranking_rows']) for r in supported_clean)
        mass_all = sum(float(r['estimated_ranking_sse_all_s2']) for r in supported_all)
        mass_clean = sum(float(r['estimated_ranking_sse_clean_s2']) for r in supported_clean)
        class_summary.append({'class': class_name, 'ranking_rows': n_rank,
                              'supported_rows_all': supported_n_all,
                              'unsupported_rows_all': n_rank - supported_n_all,
                              'estimated_mse_all_s2_per_supported_row': mass_all / supported_n_all if supported_n_all else None,
                              'share_estimated_sse_all': mass_all / estimated_mass_all if supported_n_all else None,
                              'supported_rows_clean': supported_n_clean,
                              'unsupported_rows_clean': n_rank - supported_n_clean,
                              'estimated_mse_clean_s2_per_supported_row': mass_clean / supported_n_clean if supported_n_clean else None,
                              'share_estimated_sse_clean': mass_clean / estimated_mass_clean if supported_n_clean else None})

    source_paths = sorted(set(rank_sources + fold_sources + [Path(__file__)]))
    branch_root = EXP.parents[1]
    source_hashes = {str(path.relative_to(MAIN)) if path.is_relative_to(MAIN)
                     else str(path.relative_to(branch_root)) if path.is_relative_to(branch_root)
                     else str(path): sha256(path) for path in source_paths}
    p12_lirf_jul = next(r for r in rows if r['ranking_month'] == '2026-07'
                        and r['airport'] == 'LIRF' and r['class'] == 'no-clock_ordinary')
    lfpg_jan_nc = next(r for r in rows if r['ranking_month'] == '2026-01'
                       and r['airport'] == 'LFPG' and r['class'] == 'no-clock_ordinary')
    unsupported_cells = [r for r in rows if r['ranking_rows'] and
                         r['fold_v22_mse_s2_per_row'] is None]
    unsupported_cell_text = ', '.join(
        f"{r['ranking_month']} {r['airport']} {r['class']} ({r['ranking_rows']:,} rows)"
        for r in unsupported_cells)

    md = [
        '# exp-130 step 1 — ranking-exposure census',
        '',
        '**Status:** step 1 census, with the desk-requested clean view. No model was fitted for this census.',
        '',
        '## Question and source rows',
        '',
        'The exp-128b ledger row records a negative full-data A/B: arm A was far weaker than p12, and stand turn and geometry worsened the update. The H3 planning row in `docs/LEDGER.md` points to changed airport and month exposure as the next question. This census uses the ranking row counts for each cell instead of the validation fold mix.',
        '',
        'The row counts come from the 2026 ranking rows whose IDs appear in `submitting.parquet`. I read only ranking input columns and submission IDs; I did not read the ranking target. The shared v25 class vectors define the current served class: `with-clock = occupied_by_exp_046`, `no-clock_ordinary = not occupied and not noac`, and `noac = missing aircraft type`. The served class matched `obs_has_clock & ~noac` for every row, and the `noac` mask matched missing `AIRCRAFT_TYPE_mvt` for every row.',
        '',
        'January 2026 maps to the January 2025 fold. July 2026 maps to the July leave-one-out fold. On each 2025 fold, the code rebuilds v22 residuals as the exp-047 census does: `r12 - d-seasonal` on ordinary with-clock rows, and `r12` on the complement. Both rebuilt full-fold RMSE values match `census-v22.json` within 1e-6 seconds. No model was trained or refit.',
        '',
        'The full cell table is in `census.csv`. It has one row for every airport × ranking month × class cell and reports both all-row and clean-row fold errors. The clean view drops only the IDs in `prc.data.excluded_ids(fold)` from fold scoring; it does not change the 2026 ranking row counts because the exclusions are 2025 validation rows. The clean view is the step-2 scoring baseline. For no-clock ordinary cells, the CSV reports p12 error on **all ordinary rows** in the matching fold and on the clean subset, with `y ≤ 3600 s` and `y > 3600 s` split apart; separate p12 counts and errors for fold no-clock and fold with-clock ordinary rows are included too.',
        '',
        '| Matching fold | All-row v22 RMSE (s) | exp-047 census RMSE (s) | Clean v22 RMSE (s) | Excluded rows |',
        '|---|---:|---:|---:|---:|',
        *[f"| {fold} | {check['v22_rmse_rebuilt_s']:.6f} | {check['v22_rmse_census_s']:.6f} | {check['v22_rmse_clean_s']:.6f} | {check['excluded_rows']:,} |" for fold, check in fold_checks.items()],
        '',
        '## Coverage and weighted result',
        '',
        f"The ranking has **{total_rank_rows:,} rows**: January {rank_months['2026-01']['rows']:,} and July {rank_months['2026-07']['rows']:,}. Of these, {total_clock:,} are with-clock, {total_no_clock:,} are ordinary no-clock, and {total_noac:,} are noac. Matching fold cells support {supported_rank_rows:,} rows in the all-row view and {clean_supported_rank_rows:,} in the clean view; {unsupported_rank_rows:,} rows have no same-airport, same-class reference and are left unestimated. The unsupported cells are: {unsupported_cell_text}. The all-row ranking-weighted MSE is **{overall_mse_all:,.1f} s²/row** (RMSE {math.sqrt(overall_mse_all):.1f} s); the clean-view MSE is **{overall_mse_clean:,.1f} s²/row** (RMSE {math.sqrt(overall_mse_clean):.1f} s). These are supported-cell estimates, not observed 2026 scores or full-board estimates. The 2026 ranking exposures remain the same in both views; step-2 fit weights and scores use these ranking counts with clean training and validation rows.",
        '',
        '| Class | Ranking rows | All-row MSE | Clean MSE | All-row share | Clean share |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for rec in class_summary:
        all_mse = f"{rec['estimated_mse_all_s2_per_supported_row']:,.1f}" if rec['estimated_mse_all_s2_per_supported_row'] is not None else '—'
        clean_mse = f"{rec['estimated_mse_clean_s2_per_supported_row']:,.1f}" if rec['estimated_mse_clean_s2_per_supported_row'] is not None else '—'
        all_share = f"{rec['share_estimated_sse_all']:.1%}" if rec['share_estimated_sse_all'] is not None else '—'
        clean_share = f"{rec['share_estimated_sse_clean']:.1%}" if rec['share_estimated_sse_clean'] is not None else '—'
        md.append(f"| {rec['class']} | {rec['ranking_rows']:,} | {all_mse} | {clean_mse} | {all_share} | {clean_share} |")
    md += [
        '',
        '### Weighted MSE by airport and class',
        '',
        'Each MSE below weights the matching fold cells by their 2026 ranking row counts. The MSE columns report the all-row and clean views; the shares use each view’s supported estimated squared error.',
        '',
        '| Airport | Class | Ranking rows | All-row MSE | Clean MSE | All-row share | Clean share |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for rec in summary:
        all_mse = f"{rec['estimated_mse_all_s2_per_supported_row']:,.1f}" if rec['estimated_mse_all_s2_per_supported_row'] is not None else '—'
        clean_mse = f"{rec['estimated_mse_clean_s2_per_supported_row']:,.1f}" if rec['estimated_mse_clean_s2_per_supported_row'] is not None else '—'
        all_share = f"{rec['share_of_estimated_sse_all']:.1%}" if rec['share_of_estimated_sse_all'] is not None else '—'
        clean_share = f"{rec['share_of_estimated_sse_clean']:.1%}" if rec['share_of_estimated_sse_clean'] is not None else '—'
        md.append(f"| {rec['airport']} | {rec['class']} | {rec['ranking_rows']:,} | {all_mse} | {clean_mse} | {all_share} | {clean_share} |")
    md += [
        '',
        '### Ten cells with the largest all-row estimated error share',
        '',
        '| Ranking month | Airport | Class | Ranking rows | Matching-fold MSE (s²/row) | Share of all-row estimated SSE |',
        '|---|---|---|---:|---:|---:|',
    ]
    for rec in top_all:
        md.append(f"| {rec['ranking_month']} | {rec['airport']} | {rec['class']} | {rec['ranking_rows']:,} | {rec['fold_v22_mse_s2_per_row']:,.1f} | {rec['share_of_estimated_ranking_sse_all']:.1%} |")
    md += [
        '',
        '### Ten cells with the largest clean-view estimated error share',
        '',
        '| Ranking month | Airport | Class | Ranking rows | Clean matching-fold MSE (s²/row) | Share of clean estimated SSE |',
        '|---|---|---|---:|---:|---:|',
    ]
    for rec in top_clean:
        md.append(f"| {rec['ranking_month']} | {rec['airport']} | {rec['class']} | {rec['ranking_rows']:,} | {rec['fold_v22_clean_mse_s2_per_row']:,.1f} | {rec['share_of_estimated_ranking_sse_clean']:.1%} |")
    md += [
        '',
        '## p12 check on ordinary no-clock cells',
        '',
        f"The LIRF July no-clock ranking cell has {p12_lirf_jul['ranking_rows']:,} ranking rows. On the matching 2025 fold, p12 was measured on {int(p12_lirf_jul['fold_p12_all_ordinary_rows']):,} ordinary rows: RMSE {float(p12_lirf_jul['fold_p12_rmse_all_ordinary_s']):.1f} s overall, {float(p12_lirf_jul['fold_p12_body_rmse_s']):.1f} s on {int(p12_lirf_jul['fold_p12_body_rows']):,} body rows, and {float(p12_lirf_jul['fold_p12_tail_rmse_s']):.1f} s on {int(p12_lirf_jul['fold_p12_tail_rows']):,} tail rows. On AOBT-present body rows it has RMSE {float(p12_lirf_jul['fold_p12_aobt_present_body_rmse_s']):.1f} s across {int(p12_lirf_jul['fold_p12_aobt_present_body_rows']):,} fold rows. Its clean values are identical because the July fold has no excluded rows. LFPG January ordinary no-clock has {lfpg_jan_nc['fold_excluded_rows_for_class']} excluded rows: all-row v22 MSE is {float(lfpg_jan_nc['fold_v22_mse_s2_per_row']):,.1f} s² and clean v22 MSE is {float(lfpg_jan_nc['fold_v22_clean_mse_s2_per_row']):,.1f} s². The CSV gives p12 metrics for every no-clock airport-month cell in both views.",
        '',
        '## Limits',
        '',
        f"This is a label-free ranking exposure census and a 2025 fold error projection. It does not measure 2026 labels or prove that the exposure-weighted MSE will occur on the hidden ranking target. The all-row view retains every fold row; the clean view drops exactly the IDs returned by `prc.data.excluded_ids(fold)`. For January this removes the two exp-020 LFPG `UNREACHED_ROLLOVER` rows; July has no excluded rows. The saved exp-047 fold census has no row-level vector for its served arm-N adjustment in three no-clock cells; both views follow its published fold construction and use p12 on the complement. The separate arm-N fold effect is not included in the cell MSEs. {unsupported_rank_rows:,} rows have no same-airport, same-class reference fold and are not imputed in either view. Weather inputs are untested; this census did not use them.",
        '',
        '## Input hashes',
        '',
        'The census script is `code/census.py`. The direct input hashes below pin the ranking files, class vectors, fold blocks, fold labels, v22 vector, and fold-census reference.',
        '',
    ]
    for rel, digest in source_hashes.items():
        md.append(f'- `{rel}` — `{digest}`')
    (EXP / 'CENSUS.md').write_text('\n'.join(md) + '\n')

    print(f'Wrote {out_path} ({len(rows)} airport-month-class cells)')
    print(f'Ranking rows: {total_rank_rows}; all-row MSE {overall_mse_all:.3f} s^2/row; '
          f'clean MSE {overall_mse_clean:.3f} s^2/row')
    print('Month totals:', rank_months)
    print('Fold checks:', fold_checks)
    print('All-row top 10 cells:')
    for rec in top_all:
        print(rec['ranking_month'], rec['airport'], rec['class'], rec['ranking_rows'],
              f"MSE={rec['fold_v22_mse_s2_per_row']:.1f}",
              f"share={rec['share_of_estimated_ranking_sse_all']:.1%}")
    print('Clean top 10 cells:')
    for rec in top_clean:
        print(rec['ranking_month'], rec['airport'], rec['class'], rec['ranking_rows'],
              f"MSE={rec['fold_v22_clean_mse_s2_per_row']:.1f}",
              f"share={rec['share_of_estimated_ranking_sse_clean']:.1%}")
    print('LIRF July p12 diagnostic:', {k: p12_lirf_jul[k] for k in p12_lirf_jul if k.startswith('fold_p12_')})
    print(f'Wrote {EXP / "CENSUS.md"}')


if __name__ == '__main__':
    main()
