"""exp-194 lever (b): do the DH inputs exist as often on 2026 rows with no head value as on the 2025 made-dark rows?

Label-free. For each month-airport cell: the finite share of every shp_* and queue input of DH on the 2025 clock rows
(the rows the read makes dark) and on the 2026 typed DEP rows with no head value (the rows the part serves). Reads
finalroute's final features-r4c (2025 rows kept, 2026 rows added), finalclock's head part and final_ranking.parquet.
An input with a large gap would need degrading in the read, as OPDI is.

    $PY -I dh_input_cov.py --features <features-r4c.parquet> --features-sha <sha> --head <head.parquet> \
        --head-sha <sha> --raw <dir with final_ranking.parquet> --out <input_cov.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import dh_fit as DF  # noqa: E402

ID = 'MVT_ID_mvt'
CELLS = ('1-EDDF', '2-EDDF', '6-EDDF', '7-EDDF', '6-LEMD', '6-LIRF', '7-LIRF')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--features-sha', required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--head-sha', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; write-once')
    assert DF.sha(a.features) == a.features_sha and DF.sha(a.head) == a.head_sha
    import pyarrow.parquet as pq
    names = pq.ParquetFile(a.features).schema_arrow.names
    inputs = [c for c in names if c.startswith('shp_')] + DF.Q_KEEP
    assert len(inputs) == 76 + len(DF.Q_KEEP), len(inputs)
    ft = pd.read_parquet(a.features, columns=[ID, 'ctx_year', 'ctx_month', 'ctx_airport', 'ctx_cls'] + inputs)
    ft['cell'] = ft.ctx_month.astype(int).astype(str) + '-' + ft.ctx_airport.astype(str)
    y25 = ft[(ft.ctx_year.astype(int) == 2025) & ft.ctx_cls.astype(str).eq('clock') & ft.cell.isin(CELLS)]
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=[ID, 'PHASE_mvt', 'AIRCRAFT_TYPE_mvt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')]
    head = pd.read_parquet(a.head, columns=[ID, 'pred_HB3u'])
    s26 = raw.merge(head, on=ID, how='left', validate='one_to_one')
    s26 = s26[s26.AIRCRAFT_TYPE_mvt.notna() & ~np.isfinite(s26.pred_HB3u.to_numpy(float))]
    y26 = ft[(ft.ctx_year.astype(int) == 2026) & ft[ID].isin(s26[ID]) & ft.cell.isin(CELLS)]
    out = {}
    for c in CELLS:
        a25, a26 = y25[y25.cell.eq(c)], y26[y26.cell.eq(c)]
        if len(a25) == 0 or len(a26) == 0:
            out[c] = {'rows_2025_clock': int(len(a25)), 'rows_2026_no_head': int(len(a26))}
            continue
        f25 = np.isfinite(a25[inputs].to_numpy(float)).mean(0)
        f26 = np.isfinite(a26[inputs].to_numpy(float)).mean(0)
        gap = f26 - f25
        order = np.argsort(-np.abs(gap))
        out[c] = {'rows_2025_clock': int(len(a25)), 'rows_2026_no_head': int(len(a26)),
                  'mean_finite_2025': round(float(f25.mean()), 4), 'mean_finite_2026': round(float(f26.mean()), 4),
                  'largest_gaps': {inputs[i]: [round(float(f25[i]), 4), round(float(f26[i]), 4)] for i in order[:8]},
                  'inputs_gap_over_0.05': int((np.abs(gap) > 0.05).sum())}
    rec = {'what': 'finite share of each DH shp/queue input: 2025 clock rows against 2026 typed no-head DEP rows',
           'inputs': {'features': a.features_sha, 'head': a.head_sha,
                      'final_ranking': DF.sha(a.raw / 'final_ranking.parquet')},
           'code': {HERE.name: DF.sha(HERE)}, 'cells': out}
    a.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=1)[:6000])


if __name__ == '__main__':
    main()
