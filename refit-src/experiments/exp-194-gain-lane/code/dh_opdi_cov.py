"""exp-194 lever (b): 2026 OPDI coverage on the rows with no head value, per (month, airport) and OPDI event.

Label-free. Reads finaldata's final OPDI table (opdi-first, duplicate identities blanked as exp-159 frame12 does),
finalclock's head part and the DEP rows of final_ranking.parquet. A row counts for an event when the event's lead is
finite. Only typed departures with no head value count (the rows the DH part serves).

    $PY -I dh_opdi_cov.py --opdi <opdi-first.parquet> --opdi-sha <sha> --head <head.parquet> --head-sha <sha> \
        --raw <dir with final_ranking.parquet> --out <opdi_cov_2026.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'
EVENTS = ('exit_parking_position', 'exit_apron', 'entry_taxiway')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--opdi', type=Path, required=True)
    p.add_argument('--opdi-sha', required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--head-sha', required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert sha(a.opdi) == a.opdi_sha and sha(a.head) == a.head_sha
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt',
                                                                      'MVT_TIME_UTC_mvt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')]
    assert raw[ID].is_unique and len(raw) == 670790
    op = pd.read_parquet(a.opdi)
    assert op[ID].is_unique
    lead = [c for c in op.columns if c.endswith(('_lead_s', '_dist_m'))]
    op.loc[op.duplicate_opdi_identity.astype(bool), lead] = np.nan
    head = pd.read_parquet(a.head, columns=[ID, 'pred_HB3u'])
    f = raw.merge(op[[ID] + lead], on=ID, how='left', validate='one_to_one') \
        .merge(head, on=ID, how='left', validate='one_to_one')
    f['month'] = pd.to_datetime(f.MVT_TIME_UTC_mvt, utc=True).dt.month
    s = f[f.AIRCRAFT_TYPE_mvt.notna() & ~np.isfinite(f.pred_HB3u.to_numpy(float))]
    out = {}
    for (m, ap), g in s.groupby(['month', 'ADEP_mvt']):
        out[f'{m}-{ap}'] = {'rows_no_head': int(len(g)),
                            **{e: round(float(np.isfinite(g[f'{e}_lead_s'].to_numpy(float)).mean()), 5) for e in EVENTS}}
    rec = {'what': '2026 OPDI coverage (share of typed no-head DEP rows with a finite event lead), by month-airport',
           'inputs': {'opdi_first_final': a.opdi_sha, 'head_part': a.head_sha,
                      'final_ranking': sha(a.raw / 'final_ranking.parquet')},
           'code': {Path(__file__).name: sha(__file__)}, 'cells': out}
    a.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(out, indent=0)[:3000])


if __name__ == '__main__':
    main()
