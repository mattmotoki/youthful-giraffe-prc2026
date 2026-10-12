"""exp-194 lever (b): the diff of the DH part against V (desk 12:50 UTC 10-09: rows moved per airport and month, rms).

new = max(60, v + w_pair (pred_DH - v)) where w_pair > 0, v elsewhere (the consumer step of dh_apply.py), with v the
`v` column of finalv's part `v` (the values of the V file). Label-free. Writes <out> (JSON): per month x airport with
a moved row, the rows, rows moved, rms of new - v over all rows and over the moved rows, the mean and largest move,
and the rows where the 60 s floor binds; the same over all 670,790 rows.

    $PY -I part_diff.py --part <parts/dh-<tag>.parquet> --v <finalv/parts/v.parquet> --v-sha <sha> \
        --raw <dir with final_ranking.parquet> --out <parts/dh-<tag>-diff-v.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
N_FINAL = 670790


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stats(d, floor):
    mv = d != 0
    return {'rows': int(len(d)), 'moved': int(mv.sum()),
            'rms_all_s': round(float(np.sqrt(np.mean(d ** 2))), 3),
            'rms_moved_s': round(float(np.sqrt(np.mean(d[mv] ** 2))), 3) if mv.any() else 0.,
            'mean_moved_s': round(float(np.mean(d[mv])), 3) if mv.any() else 0.,
            'max_abs_s': round(float(np.max(np.abs(d))), 3), 'floor_60_rows': int(floor.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', type=Path, required=True)
    p.add_argument('--v', type=Path, required=True)
    p.add_argument('--v-sha', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; write-once')
    assert sha(a.v) == a.v_sha, 'V part SHA-256'
    part = pd.read_parquet(a.part, columns=[ID, 'pred_DH', 'w_pair'])
    v = pd.read_parquet(a.v, columns=[ID, 'v'])
    assert len(part) == N_FINAL and (part[ID].to_numpy() == v[ID].to_numpy()).all()
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=[ID, 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'ADEP_mvt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')].set_index(ID).reindex(part[ID].to_numpy())
    assert raw.ADEP_mvt.notna().all()
    vv, w, dh = v.v.to_numpy(float), part.w_pair.to_numpy(float), part.pred_DH.to_numpy(float)
    on = w > 0
    assert np.isfinite(vv).all() and np.isfinite(dh[on]).all() and (w <= 1).all()
    raw_new = vv[on] + w[on] * (dh[on] - vv[on])
    new = vv.copy()
    new[on] = np.maximum(60., raw_new)
    floor = np.zeros(N_FINAL, dtype=bool)
    floor[np.flatnonzero(on)[raw_new < 60.]] = True
    d = new - vv
    month = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    ap = raw.ADEP_mvt.astype(str).to_numpy()
    cells = {}
    for (m, ap_), idx in pd.DataFrame({'m': month, 'a': ap}).groupby(['m', 'a']).indices.items():
        if (d[idx] != 0).any() or on[idx].any():
            cells[f'{m}-{ap_}'] = stats(d[idx], floor[idx])
    info = {'what': 'DH part applied to V (max(60, v + w_pair (pred_DH - v)) where w_pair > 0) minus V',
            'part': {'path': str(a.part), 'sha256': sha(a.part)}, 'v': {'path': str(a.v), 'sha256': a.v_sha},
            'final_ranking': sha(a.raw / 'final_ranking.parquet'), 'rows_w_pair_above_0': int(on.sum()),
            'all_rows': stats(d, floor), 'by_month_airport': cells,
            'code': {Path(__file__).name: sha(__file__)}}
    a.out.write_text(json.dumps(info, indent=1) + '\n')
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
