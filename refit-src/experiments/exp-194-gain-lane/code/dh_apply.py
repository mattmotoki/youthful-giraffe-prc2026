"""exp-194 lever (b): the DH part on the 670,790 final departures (template order of final_submitting.parquet).

DH = the mean of the saved `jjrank` boosters of dh_fit.py (2025 clock rows without January and July, as HB3u's 2026
set), on the final-mode feature table of finalroute (features-r4c, 2026 rows) and the raw columns of
final_ranking.parquet, with the code tables of the fit record. Every row gets a value where its inputs exist; the
consumer applies it only where the part's `w_pair` is above 0 (rows with no head value in the made-dark cells; the
weights come from the read, `--weights`).

    OMP_NUM_THREADS=2 $PY -I dh_apply.py --fit-dir <dir with DH-jjrank.models/, DH-jjrefit.json> \
        --features <finalroute features-r4c.parquet> --features-sha <sha> --raw <dir with final_*.parquet> \
        --head <finalclock head.parquet> --head-sha <sha> [--weights <weights.json>] --out <parts/dh.parquet>

Writes <out> (MVT_ID_mvt, pred_DH, has_head, w_pair) and <out>.json (counts, hashes, label-free checks: DH minus
the head value on 2026 clock rows by airport and month).
Consumer step: final = max(60, V + w_pair * (pred_DH - V)) where w_pair > 0; V elsewhere. With a weights "flag"
(dark_a or dark_b), w_pair is set only on the rows of the cells that carry the flag (dh_dark_rows.py, label-free).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import dh_fit as DF  # noqa: E402

ID = 'MVT_ID_mvt'
RAW_COLS = [ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'ADEP_mvt', 'RUNWAY_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt',
            'AIRCRAFT_TYPE_mvt']
N_FINAL = 670790


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fit-dir', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--features-sha', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--head-sha', required=True)
    p.add_argument('--stage', default='DHo', choices=['DH', 'DHo'])
    p.add_argument('--opdi', type=Path, help="finaldata's final opdi-first.parquet (stage DHo)")
    p.add_argument('--opdi-sha')
    p.add_argument('--weights', type=Path, help='JSON {"cells": [{"month": m, "airport": a, "w": w}, ...]} and an '
                   'optional "flag" (cell: every typed no-head row of a cell; dark_a or dark_b: only rows with that '
                   'flag of dh_dark_rows.py, --dark-rows)')
    p.add_argument('--dark-rows', type=Path, help='dh_dark_rows.py parquet (MVT_ID_mvt, dark_a, dark_b)')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import lightgbm as lgb
    for q in (a.out, a.out.with_suffix('.json')):
        if q.exists():
            raise SystemExit(f'{q} exists; write-once')
    assert DF.sha(a.features) == a.features_sha, 'features SHA-256'
    assert DF.sha(a.head) == a.head_sha, 'head part SHA-256'
    rec = json.loads((a.fit_dir / f'{a.stage}-jjrefit.json').read_text())
    assert rec['split'] == 'jjrank' and not rec['fit']['smoke']
    feats, tables = rec['features'], rec['code_tables']
    models = sorted((a.fit_dir / f'{a.stage}-jjrank.models').glob('seed-*.txt'))
    assert [int(m.stem[5:]) for m in models] == rec['fit']['seeds'], 'boosters and the record differ'

    tpl = pd.read_parquet(a.raw / 'final_submitting.parquet', columns=[ID])
    assert len(tpl) == N_FINAL and tpl[ID].is_unique
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=RAW_COLS)
    raw = raw[raw.PHASE_mvt.eq('DEP')].drop(columns='PHASE_mvt')
    assert raw[ID].is_unique and len(raw) == N_FINAL
    import pyarrow.parquet as pq
    names = pq.ParquetFile(a.features).schema_arrow.names
    cols = [ID, 'ctx_year', 'ctx_month', 'ctx_cls'] + [c for c in names if c.startswith('shp_')] + DF.Q_KEEP
    ft = pd.read_parquet(a.features, columns=cols)
    ft = ft[ft.ctx_year.astype(int) == 2026]
    assert ft[ID].is_unique
    f = tpl.merge(raw, on=ID, how='left', validate='one_to_one').merge(ft, on=ID, how='left', validate='one_to_one')
    assert len(f) == N_FINAL and (f[ID].to_numpy() == tpl[ID].to_numpy()).all()
    has_feat = f.ctx_year.notna().to_numpy()
    f['airport'] = f.ADEP_mvt.astype(str)
    opdi = [c for c in feats if c.startswith('opdi_')]
    if opdi:
        assert opdi == DF.OPDI_COLS and DF.sha(a.opdi) == a.opdi_sha, 'stage DHo needs the final OPDI table'
        f = DF.join_opdi(f, a.opdi)          # no thinning on the served rows: 2026 coverage as it is
        assert (f[ID].to_numpy() == tpl[ID].to_numpy()).all()
    t = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True)
    f['month'] = t.dt.month.to_numpy()
    X = pd.concat([f[[c for c in feats if c.startswith('shp_')] + DF.Q_KEEP + opdi], DF.encode(f, tables)], axis=1)[feats]
    Xn = X.to_numpy('float32')
    pred = np.full(N_FINAL, np.nan)
    z = np.zeros((len(models), int(has_feat.sum())))
    for k, m in enumerate(models):
        z[k] = lgb.Booster(model_file=str(m)).predict(Xn[has_feat], num_threads=a.threads)
    pred[has_feat] = z.mean(0)
    head = pd.read_parquet(a.head)
    assert (head[ID].to_numpy() == tpl[ID].to_numpy()).all()
    hv = head.pred_HB3u.to_numpy(float)
    has_head = np.isfinite(hv)
    w = np.zeros(N_FINAL)
    cells = []
    flag = 'cell'
    if a.weights:
        W = json.loads(a.weights.read_text())
        flag = W.get('flag', 'cell')
        rowsel = np.ones(N_FINAL, dtype=bool)
        if flag != 'cell':
            assert flag in ('dark_a', 'dark_b') and a.dark_rows is not None, flag
            dr = pd.read_parquet(a.dark_rows, columns=[ID, flag]).set_index(ID)[flag]
            rowsel = dr.reindex(tpl[ID].to_numpy()).fillna(False).to_numpy(bool)
        for c in W['cells']:
            sel = (f.month.to_numpy() == int(c['month'])) & (f.airport.to_numpy() == c['airport']) & ~has_head & \
                np.isfinite(pred) & f.AIRCRAFT_TYPE_mvt.notna().to_numpy() & rowsel
            w[sel] = float(c['w'])
            cells.append({**c, 'rows': int(sel.sum())})
    out = pd.DataFrame({ID: tpl[ID].to_numpy(), 'pred_DH': pred, 'has_head': has_head, 'w_pair': w})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    # label-free transfer check: DH minus head value on 2026 clock rows (the same read on 2025 is in dh_read)
    g = pd.DataFrame({'m': f.month, 'a': f.airport, 'd': pred - hv})[has_head & np.isfinite(pred)]
    chk = g.groupby(['m', 'a']).d.agg(['size', 'mean', lambda s: float(np.sqrt(np.mean(s ** 2)))])
    chk.columns = ['rows', 'mean_dh_minus_head', 'rms_dh_minus_head']
    info = {'part': str(a.out), 'sha256': DF.sha(a.out), 'rows': N_FINAL,
            'rows_with_value': int(np.isfinite(pred).sum()), 'rows_without_features': int((~has_feat).sum()),
            'rows_with_head': int(has_head.sum()), 'applied_rows': int((w > 0).sum()), 'cells': cells,
            'row_flag': flag, 'dark_rows': DF.sha(a.dark_rows) if flag != 'cell' else None,
            'stage': a.stage, 'opdi_rows_with_event': {e: int(np.isfinite(f[f'opdi_{e}_lead_s'].to_numpy(float)).sum())
                                                       for e in DF.EVENTS} if opdi else None,
            'inputs': {'features': a.features_sha, 'head': a.head_sha, 'opdi': a.opdi_sha if opdi else None,
                       'fit_record': DF.sha(a.fit_dir / f'{a.stage}-jjrefit.json'),
                       'boosters': {m.name: DF.sha(m) for m in models},
                       'final_ranking': DF.sha(a.raw / 'final_ranking.parquet'),
                       'final_submitting': DF.sha(a.raw / 'final_submitting.parquet'),
                       'weights': DF.sha(a.weights) if a.weights else None},
            'code': {HERE.name: DF.sha(HERE), 'dh_fit.py': DF.sha(HERE.parent / 'dh_fit.py')},
            'check_2026_clock_rows': {f'{m}-{ap}': {k: round(float(v), 2) for k, v in r.items()}
                                      for (m, ap), r in chk.iterrows()},
            'pred_mean_by_month_airport': {f'{m}-{ap}': round(float(v), 1) for (m, ap), v in
                                           pd.Series(pred).groupby([f.month, f.airport]).mean().items()}}
    a.out.with_suffix('.json').write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k not in ('pred_mean_by_month_airport',)}, indent=1)[:4000])


if __name__ == '__main__':
    main()
