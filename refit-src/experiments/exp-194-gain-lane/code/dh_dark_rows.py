"""exp-194 lever (b): label-free dark-row flags for 2025 and 2026 departures (lead 07:45 UTC 10-09, part 2 (b)).

A row is dark when the adsb.lol inputs that the members read are absent for the row and for its neighbours. From
finalroute's final features-r4c (2025 rows kept, 2026 rows added), per departure:
  own_absent   no observed taxi for the row (obs_obs_taxi missing) and no ground points (obs_n_ground_points
               missing or 0)
  nf10, no10   other departures in the airport x 10-minute cell from the flight list (q_Nf_n_n_cell) and with an
               observed taxi (q_No_n_n_cell); nf30r, no30r the same on the runway over 30 minutes
  day_share    share of the airport-day's departures with an observed taxi
Definitions (set before any error read, SPEC addendum 8):
  dark_a = own_absent and nf10 + nf30r > 0 and no10 + no30r == 0   (neighbours exist, none of them is seen)
  dark_b = own_absent and day_share < 0.2                          (the airport-day is mostly unseen)
Writes <out>.parquet (MVT_ID_mvt, year, month, airport, cls, the inputs above, dark_a, dark_b; the 2025 months
1, 2, 6, 7 and every 2026 row) and <out>.json (counts per year-month-airport and class; 2026 split by head value).

    $PY -I dh_dark_rows.py --features <features-r4c.parquet> --features-sha <sha> --head <head.parquet> \
        --head-sha <sha> --raw <dir with final_ranking.parquet> --out <dir>/dark_rows-<tag>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
COLS = [ID, 'ctx_year', 'ctx_month', 'ctx_day', 'ctx_airport', 'ctx_cls', 'obs_obs_taxi', 'obs_n_ground_points',
        'q_Nf_n_n_cell', 'q_No_n_n_cell', 'q_Nf_n_n_rwy_30m', 'q_No_n_n_rwy_30m']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--features-sha', required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--head-sha', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    pq_out, js_out = a.out.with_suffix('.parquet'), a.out.with_suffix('.json')
    for q in (pq_out, js_out):
        if q.exists():
            raise SystemExit(f'{q} exists; write-once')
    assert sha(a.features) == a.features_sha and sha(a.head) == a.head_sha
    f = pd.read_parquet(a.features, columns=COLS)
    assert f[ID].is_unique
    yr = f.ctx_year.astype(int).to_numpy()
    mo = f.ctx_month.astype(int).to_numpy()
    f = f[(yr == 2026) | np.isin(mo, [1, 2, 6, 7])].reset_index(drop=True)
    d = pd.DataFrame({ID: f[ID].to_numpy('int64'), 'year': f.ctx_year.astype(int).to_numpy(),
                      'month': f.ctx_month.astype(int).to_numpy(), 'day': f.ctx_day.astype(str).to_numpy(),
                      'airport': f.ctx_airport.astype(str).to_numpy(), 'cls': f.ctx_cls.astype(str).to_numpy()})
    gp = f.obs_n_ground_points.to_numpy(float)
    d['own_absent'] = ~np.isfinite(f.obs_obs_taxi.to_numpy(float)) & ~(np.isfinite(gp) & (gp > 0))
    for c, k in (('q_Nf_n_n_cell', 'nf10'), ('q_No_n_n_cell', 'no10'), ('q_Nf_n_n_rwy_30m', 'nf30r'),
                 ('q_No_n_n_rwy_30m', 'no30r')):
        d[k] = np.nan_to_num(f[c].to_numpy(float), nan=0.)
    seen = np.isfinite(f.obs_obs_taxi.to_numpy(float))
    d['day_share'] = pd.Series(seen.astype(float)).groupby([d.year, d.month, d.day, d.airport]).transform('mean').to_numpy()
    d['dark_a'] = d.own_absent.to_numpy() & ((d.nf10 + d.nf30r).to_numpy() > 0) & ((d.no10 + d.no30r).to_numpy() == 0)
    d['dark_b'] = d.own_absent.to_numpy() & (d.day_share.to_numpy() < 0.2)
    # 2026: head value and type, for the served rows
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=[ID, 'PHASE_mvt', 'AIRCRAFT_TYPE_mvt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')]
    head = pd.read_parquet(a.head, columns=[ID, 'pred_HB3u']).set_index(ID).pred_HB3u
    d['has_head'] = np.isfinite(head.reindex(d[ID].to_numpy()).to_numpy(float))
    d['typed26'] = d[ID].isin(raw.loc[raw.AIRCRAFT_TYPE_mvt.notna(), ID]).to_numpy()
    d.to_parquet(pq_out, index=False)
    out = {}
    for (y, m, ap), g in d.groupby(['year', 'month', 'airport']):
        parts = {'2025': [('clock', g.cls.eq('clock')), ('ordinary', g.cls.eq('ordinary'))],
                 '2026': [('no_head', g.typed26 & ~g.has_head), ('head', g.typed26 & g.has_head)]}[str(y)]
        for nm, s in parts:
            s = s.to_numpy()
            if s.sum() == 0:
                continue
            out[f'{y}-{m}-{ap}-{nm}'] = {'rows': int(s.sum()), 'own_absent': round(float(g.own_absent[s].mean()), 4),
                                         'dark_a': round(float(g.dark_a[s].mean()), 4),
                                         'dark_b': round(float(g.dark_b[s].mean()), 4),
                                         'day_share_median': round(float(g.day_share[s].median()), 4)}
    rec = {'what': 'label-free dark-row flags (own adsb.lol inputs absent and neighbours unseen)',
           'inputs': {'features': a.features_sha, 'head': a.head_sha,
                      'final_ranking': sha(a.raw / 'final_ranking.parquet')},
           'code': {Path(__file__).name: sha(__file__)}, 'rows': int(len(d)), 'parquet_sha256': sha(pq_out),
           'cells': out}
    js_out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if 'EDDF' in k or 'LIRF' in k or 'LEMD' in k}, indent=0)[:5000])


if __name__ == '__main__':
    main()
