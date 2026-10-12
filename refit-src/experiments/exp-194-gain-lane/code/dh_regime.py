"""exp-194 lever (b): are V's members dark on the 2026 rows with no head value? A label-free regime check.

For a set of rows, the spread between members: RMS of (dark18 - sequence039), (dark18 - g), (sequence039 - g),
(sequence021 - sequence039) and (mix12 - g), and the member means. exp-170's forced-dark July values show what the
spread is when no aircraft of the airport-day is seen; the saved held values show it on lit rows. The same numbers on
the 2026 rows with no head value say which of the two the final members are in.

2026 side (remote, finalcore's member parts on the 670,790 final rows, finalclock's head part, final_ranking):
    $PY -I dh_regime.py final --parts <finalcore/parts> --head <head.parquet> --head-sha <sha> \
        --raw <dir with final_ranking.parquet> --out <regime_2026.json>
2025 side (Mac, release export fold frame and exp-170 forced-dark values):
    .venv/bin/python -I dh_regime.py held --export <release export> --x170 work/x170 --out <regime_2025.json>
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
ID = 'MVT_ID_mvt'
PARTS = {'g': ('8758a9716eba08b33fef5f2ce9753594cb68d55755a29e8a5b4cfb2c345ab6ef', 'pred_Gmixed'),
         'sequence021': ('c21db6dca6f8ecabd37abe973c27078368e8c3947977b4cc23b1ae7a46d448fd', 'pred_seq170'),
         'dark18': ('86c049d0795bdaf741fca6f2ee546b427287b070fd4fb7b8f1c6bf360435467f', 'pred_seq170'),
         'sequence039': ('636ab22a0b44d191fd6e0e35dcb7e8fdb52cdc9485490d0661c74c17f7f4203e', 'pred_seq170'),
         'mlp': ('e48a88c6bc1c0a3868b4472aae8ab5de032e4ee435374e6c58a7b038891145e2', 'mix12')}
PAIRS = (('dark18', 'sequence039'), ('dark18', 'g'), ('sequence039', 'g'), ('sequence021', 'sequence039'),
         ('mix12', 'g'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def spread(d, s):
    out = {'rows': int(s.sum())}
    if s.sum() < 50:
        return out
    for a, b in PAIRS:
        x = (d[a].to_numpy(float) - d[b].to_numpy(float))[s]
        x = x[np.isfinite(x)]
        out[f'rms_{a}-{b}'] = round(float(np.sqrt(np.mean(x ** 2))), 1) if len(x) else None
    for c in ('g', 'sequence021', 'sequence039', 'dark18', 'mix12'):
        out[f'mean_{c}'] = round(float(np.nanmean(d[c].to_numpy(float)[s])), 1)
    return out


def final(a):
    assert sha(a.head) == a.head_sha
    raw = pd.read_parquet(a.raw / 'final_ranking.parquet', columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt',
                                                                      'MVT_TIME_UTC_mvt'])
    d = raw[raw.PHASE_mvt.eq('DEP')].reset_index(drop=True)
    assert len(d) == 670790 and d[ID].is_unique
    hashes = {}
    for k, (want, col) in PARTS.items():
        f = a.parts / f'{k}.parquet'
        hashes[k] = sha(f)
        assert hashes[k] == want, k
        q = pd.read_parquet(f, columns=[ID, col]).set_index(ID)[col]
        d['mix12' if k == 'mlp' else k] = q.reindex(d[ID].to_numpy()).to_numpy(float)
    hv = pd.read_parquet(a.head, columns=[ID, 'pred_HB3u']).set_index(ID).pred_HB3u
    d['has_head'] = np.isfinite(hv.reindex(d[ID].to_numpy()).to_numpy(float))
    d['month'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    typed = d.AIRCRAFT_TYPE_mvt.notna().to_numpy()
    out = {}
    for (m, ap), g in d.groupby(['month', 'ADEP_mvt']):
        s_ = d.index.isin(g.index)
        out[f'{m}-{ap}'] = {'no_head': spread(d, s_ & typed & ~d.has_head.to_numpy()),
                            'head': spread(d, s_ & typed & d.has_head.to_numpy())}
    return {'what': '2026 member spread on typed rows with and without a head value', 'inputs': hashes,
            'head': a.head_sha, 'final_ranking': sha(a.raw / 'final_ranking.parquet'), 'cells': out}


def held(a):
    import read_a as RA
    fold, _ = RA.fold_frame(a.export.resolve())
    lit = fold.copy()
    dk = fold.copy()
    for col, dname in (('dark18', 'dark18-Z-063-mask-joint'), ('fx18', 'fixed18-Z-063'),
                       ('fx18_m', 'fixed18-Z-063-mask-joint')):
        q = pd.read_parquet(a.x170 / dname / 'jjrefit.parquet').set_index(ID).pred_seq170
        dk[col] = q.reindex(fold[ID].to_numpy('int64')).to_numpy(float)
    # forced dark: dark18 exact; sequence039 and sequence021 moved by the fixed18 step (stand-in); g and mix12 as saved
    step = dk.fx18_m.to_numpy(float) - dk.fx18.to_numpy(float)
    dk['sequence039'] = fold.sequence039.to_numpy(float) + step
    dk['sequence021'] = fold.sequence021.to_numpy(float) + step
    out = {}
    for m in (1, 7):
        for ap in sorted(fold.airport.unique()):
            for z in ('clock', 'ordinary'):
                s = (fold.typed & fold.month.eq(m) & fold.airport.eq(ap) & fold.cls.eq(z)).to_numpy()
                r = {'lit': spread(lit, s)}
                if m == 7 and ap in ('EDDF', 'LIRF'):
                    r['forced_dark'] = spread(dk, s)
                    r['forced_dark_fixed18_as_sequence039'] = spread(dk.assign(sequence039=dk.fx18_m), s)
                out[f'{m}-{ap}-{z}'] = r
    return {'what': '2025 held member spread, lit and forced dark (exp-170 joint mask, EDDF/LIRF July)',
            'mix12_is_lit': True, 'cells': out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['final', 'held'])
    p.add_argument('--parts', type=Path)
    p.add_argument('--head', type=Path)
    p.add_argument('--head-sha')
    p.add_argument('--raw', type=Path)
    p.add_argument('--export', type=Path)
    p.add_argument('--x170', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; write-once')
    rec = final(a) if a.mode == 'final' else held(a)
    rec['code'] = {HERE.name: sha(HERE)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec, indent=0)[:3000])


if __name__ == '__main__':
    main()
