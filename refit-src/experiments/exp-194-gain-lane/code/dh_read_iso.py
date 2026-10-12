"""exp-194 review repair: the graded reads with the scored month out of every fitted stage (Astra review of dh-final1,
VERDICT FAIL 13:50 UTC 10-09; desk 13:55 UTC: also independent day draws per month and the mix12 count).

Scored month m (January or July), other month mo. Each fitted stage and the labels it reads:
  DHo             2025 clock rows of 02-06 and 08-12 (no January or July row; dh_fit.py jjrank)
  members         g, mix12, sequence021, sequence039, dark18, row6: fits without January and July (LINEAGE-ZABD)
  R4mk on m       the saved held level 2 of m (q01 or q07: the jjx months and mo; no label of m)
  R4mk on mo      r4mk_iso.py owner x0m: the seven jjx months, the fixed rounds of q0m (no label of m)
                  [dh_read_nostand.py used q0mo here: it trains on m's rows]
  V's weights     A6 on mo's ordinary rows (lit) or mo's EDDF and LIRF rows (dark_set), with those values; the same
                  weights give V on mo (where w is fitted) and on m (where it is scored)
                  [dh_read_nostand.py gave V on mo the weights fitted on m]
  pair w          per airport on mo's typed rows of the frame with V on mo
No fitted stage reads a label of m. Forced dark (--r4 litL or darkL): EDDF and LIRF rows of both months with dark
member values, R4mk dark from the saved owner on m and from x0m on mo. Natural (--natural-out): typed rows with a
dh_dark_rows.py flag, lit members, all airports, w per flag on mo's flagged rows (dh_read_natural.py's read with the
same V). Day draws: 2,000 per month from independent generators, numpy default_rng((601, m)) [dh_read_nostand.py used
601 + days, the same draws for January and July]. The part's w (dh_read_nostand.py pair_w '<airport>-both') is
recomputed with the old definition as a parity line (it scores no row; the repair changes no stage of it).

    .venv/bin/python -I experiments/exp-194-gain-lane/code/dh_read_iso.py --export <release export> --dh work/dh \
        --dark work/dark-o2/fetch --r4 darkL --iso work/iso-<tag> --flags work/dark-o2/fetch/dark_rows.parquet \
        --out experiments/exp-194-gain-lane/out/<name>.json [--natural-out experiments/exp-194-gain-lane/out/<n>.json]
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
import dh_read as DR  # noqa: E402
import dh_read_nostand as NS  # noqa: E402
import read_a as RA  # noqa: E402

ID = NS.ID
AIRPORTS, N_FINAL, SEASON, DRAWS, SEED = NS.AIRPORTS, NS.N_FINAL, NS.SEASON, NS.DRAWS, NS.SEED
FRAMES = ('lit', 'dark', 'dark_set')
FLAGS = ('dark_b', 'dark_a')
ISO = {1: 'x01', 7: 'x07'}       # scored month: r4mk_iso owner whose values serve the other month


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def draws_for(day, m):
    """Day blocks of month m; the generator depends on the month only, so the months draw independently."""
    days = np.unique(day)
    rng = np.random.default_rng((SEED, m))
    return days, rng.integers(0, len(days), size=(DRAWS, len(days)))


def iso_frames(lit, dk, m, mo, X, r4):
    """lit and dark member frames of direction m: R4mk on mo's rows from the r4mk_iso owner (no label of m)."""
    litD, dkD = lit.copy(), dk.copy()
    ids = lit[ID].to_numpy('int64')
    on = lit.month.eq(mo).to_numpy()
    x = X.set_index(ID)
    xl = x.pred_Sb_lit.reindex(ids).to_numpy(float)
    xd = x[f'pred_Sb_dark_{r4}'].reindex(ids).to_numpy(float) if r4 else xl
    had = on & np.isfinite(lit.R4mk.to_numpy(float))
    assert (np.isfinite(xl) == had).all(), 'the r4mk_iso rows differ from the saved R4mk rows of the other month'
    dark_rows = on & lit.airport.isin(AIRPORTS).to_numpy() & had
    assert np.isfinite(xd[dark_rows]).all()
    litD['R4mk'] = np.where(on, xl, lit.R4mk.to_numpy(float))
    dkD['R4mk'] = np.where(dark_rows, xd, np.where(on, xl, dk.R4mk.to_numpy(float)))
    d = xl[had] - lit.R4mk.to_numpy(float)[had]
    return litD, dkD, {'rows': int(had.sum()), 'iso_minus_saved_mean_s': round(float(d.mean()), 2),
                       'iso_minus_saved_rms_s': round(float(np.sqrt(np.mean(d ** 2))), 2)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--export', type=Path, required=True)
    ap_.add_argument('--dh', type=Path, required=True)
    ap_.add_argument('--dark', type=Path, required=True, help='fetched outputs of run o2 (dh_read_nostand.load_dark)')
    ap_.add_argument('--r4', choices=['litL', 'darkL'], required=True)
    ap_.add_argument('--iso', type=Path, required=True, help='folder of R4mksX-x01.parquet and R4mksX-x07.parquet')
    ap_.add_argument('--flags', type=Path, required=True)
    ap_.add_argument('--out', type=Path, required=True)
    ap_.add_argument('--natural-out', type=Path)
    a = ap_.parse_args()
    for o in (a.out, a.natural_out):
        if o is not None and o.exists():
            raise SystemExit(f'{o} exists; write-once')
    DR.STAGE = 'DHo'
    fold, _ = RA.fold_frame(a.export.resolve())
    fold, h_jj = DR.attach_dh(fold, a.dh, 'jjrefit')
    lit, dk, parity = NS.load_dark(fold, a.dark, a.r4)
    y, dh, day = fold.y.to_numpy(float), fold.DH.to_numpy(float), fold.day.to_numpy()
    month, airport, cls = fold.month.to_numpy(), fold.airport.to_numpy(), fold.cls.to_numpy()
    typed = fold.typed.to_numpy() & np.isfinite(dh)
    iso_in = {}
    X = {}
    for m, own in ISO.items():
        p = a.iso / f'R4mksX-{own}.parquet'
        rj = json.loads(p.with_suffix('.json').read_text())
        assert rj['parquet_sha256'] == sha(p) and rj['keeps_out_month'] == m and not rj['smoke'], p
        assert all(int(k) not in (1, 7) for k in rj['training']['rows_by_month']), rj['training']['rows_by_month']
        X[m] = pd.read_parquet(p)
        iso_in[own] = {'parquet': str(p), 'sha256': rj['parquet_sha256'], 'json_sha256': sha(p.with_suffix('.json')),
                       'serves_month': rj['serves_month'], 'keeps_out_month': m,
                       'rounds_from': {k: rj['rounds_from'][k] for k in ('owner', 'rounds', 'rounds_cat', 'cv_months')},
                       'training_rows_by_month': rj['training']['rows_by_month']}
    days_pick = {m: draws_for(day[month == m], m) for m in (1, 7)}

    # the part's w with dh_read_nostand.py's definition (parity with the earlier read; it scores no row)
    Vold = {k: np.full(len(fold), np.nan) for k in FRAMES}
    for m, mo in ((1, 7), (7, 1)):
        Vm, _ = NS.frames_for(lit, dk, m, mo)
        for k in FRAMES:
            Vold[k][month == m] = Vm[k][month == m]
    part_w = {}
    for fr in FRAMES:
        ok = typed & np.isfinite(Vold[fr])
        for ap in AIRPORTS:
            both = ok & np.isin(month, (1, 7)) & (airport == ap)
            part_w[f'{fr}|{ap}-both'] = DR.pair_w(y[both], Vold[fr][both], dh[both])

    # each direction: every fitted stage on mo
    V = {k: np.full(len(fold), np.nan) for k in FRAMES}       # V on the scored month m (weights from mo)
    Vmo = {}                                                 # V on mo per direction (the same weights)
    W, iso_rec, lits = {}, {}, {}
    for m, mo in ((1, 7), (7, 1)):
        litD, dkD, iso_rec[m] = iso_frames(lit, dk, m, mo, X[m], a.r4)
        Vm, W[m] = NS.frames_for(litD, dkD, m, mo)
        Vmo[m] = Vm
        lits[m] = litD
        for k in FRAMES:
            V[k][month == m] = Vm[k][month == m]
    draws, cells, pw = {}, {}, {}
    for fr in FRAMES:
        pw[fr] = {}
        for m, mo in ((1, 7), (7, 1)):
            vm = Vmo[m][fr]
            ok = typed & np.isfinite(vm)
            days, pick = days_pick[m]
            for ap in AIRPORTS:
                fit_rows = ok & (month == mo) & (airport == ap)
                w = DR.pair_w(y[fit_rows], vm[fit_rows], dh[fit_rows])
                pw[fr][f'{ap}-{m}'] = {'w': w, 'fit_rows': int(fit_rows.sum())}
                for z in ('clock', 'ordinary'):
                    s = ok & (month == m) & (airport == ap) & (cls == z)
                    if s.sum() < 50:
                        continue
                    p = np.maximum(60., vm[s] + w * (dh[s] - vm[s]))
                    delta = (y[s] - vm[s]) ** 2 - (y[s] - p) ** 2
                    pt, dr = NS.cell_draws(day[s], delta, days, pick)
                    key = f'{fr}|{m}-{ap}-{z}'
                    draws[key] = (pt, dr)
                    cells[key] = {'rows': int(s.sum()), 'w_from_other_month': round(w, 3),
                                  'mse_V': round(float(np.mean((y[s] - vm[s]) ** 2)), 1),
                                  'mse_DH': round(float(np.mean((y[s] - dh[s]) ** 2)), 1),
                                  'mse_pair': round(float(np.mean((y[s] - p) ** 2)), 1), **NS.iv(pt, dr),
                                  'days_with_gain': round(float(np.mean(
                                      [delta[day[s] == k].sum() > 0 for k in np.unique(day[s])])), 3)}
    directions = {}
    for fr in FRAMES:
        for ap in AIRPORTS:
            for z in ('clock', 'ordinary'):
                k1, k7 = f'{fr}|1-{ap}-{z}', f'{fr}|7-{ap}-{z}'
                if k1 in draws and k7 in draws:
                    directions[f'{fr}|{ap}-{z}'] = {
                        'jan_w_from_jul': {k: cells[k1][k] for k in ('gain', 'lo95', 'hi95', 'w_from_other_month')},
                        'jul_w_from_jan': {k: cells[k7][k] for k in ('gain', 'lo95', 'hi95', 'w_from_other_month')},
                        'average': NS.iv(0.5 * (draws[k1][0] + draws[k7][0]), 0.5 * (draws[k1][1] + draws[k7][1]))}
    # 2026 size on the dark_b no-head rows, split by the 2025 clock share (as dh_read_nostand.py)
    cells26 = [c for c in json.loads((HERE.parents[1] / 'out/dh_read-o1.json').read_text())['made_dark_cells_2026']
               if c['airport'] in AIRPORTS]
    share = {(c['month'], c['airport']): c['share_2025'] for c in cells26}
    flags26 = json.loads(a.flags.with_suffix('.json').read_text())['cells']
    Gs = {}
    for fr in FRAMES:
        for ap in AIRPORTS:
            pt, dr, used, by_month = 0., np.zeros(DRAWS), [], {}
            for m in (1, 2, 6, 7):
                c = flags26.get(f'2026-{m}-{ap}-no_head')
                if c is None or (m, ap) not in share:
                    continue
                n = c['rows'] * c['dark_b']
                pt_m, dr_m = 0., np.zeros(DRAWS)
                for z, nz in (('clock', n * share[(m, ap)]), ('ordinary', n * (1 - share[(m, ap)]))):
                    key = f'{fr}|{SEASON[m]}-{ap}-{z}'
                    if key not in draws:
                        continue
                    pt_m += nz * draws[key][0] / N_FINAL
                    dr_m += nz * draws[key][1] / N_FINAL
                    used.append((m, z, round(nz)))
                pt += pt_m
                dr += dr_m
                lo, hi = np.quantile(dr_m, [0.025, 0.975])
                by_month[m] = {'gain': round(pt_m, 2), 'lo95': round(float(lo), 2), 'hi95': round(float(hi), 2),
                               'dark_b_rows_2026': round(n)}
            if used:
                Gs[f'{fr}|{ap}'] = {**NS.iv(pt, dr), 'dark_b_rows_2026': used, 'by_month': by_month}
    sp = {}
    for m in (1, 7):
        for ap in AIRPORTS:
            s = typed & (month == m) & (airport == ap) & (cls == 'clock')
            sp[f'{m}-{ap}-clock-lit'] = NS.spread(lit, s)
            sp[f'{m}-{ap}-clock-forced_dark'] = NS.spread(dk, s)
    rec = {'what': 'DH pair against V with the scored month out of every fitted stage (review repair); forced dark',
           'r4': a.r4, 'isolation': {
               'DHo': 'jjrank fit: 2025 clock rows of 02-06 and 08-12', 'members': 'fits without January and July',
               'R4mk_on_scored_month': 'saved q01/q07 (jjx months and the other month)',
               'R4mk_on_other_month': 'r4mk_iso.py x01/x07 (jjx months; rounds of the owner that keeps the scored month out)',
               'V_weights': 'fitted on the other month', 'pair_w': 'fitted on the other month with V on the other month'},
           'iso': iso_in, 'iso_vs_saved_R4mk_on_other_month': iso_rec, 'parity': parity, 'weights': W,
           'pair_w': pw, 'cells': cells, 'directions': directions, 'G_per_final_row_dark_b_rows_split': Gs,
           'member_spread': sp, 'flags_2026': flags26, 'part_w_old_definition': part_w,
           'inputs': {'DH_jjrefit': h_jj, 'dark_dir': str(a.dark), 'flags': {'path': str(a.flags), 'sha256': sha(a.flags)},
                      'export': str(a.export)},
           'draws': DRAWS, 'seed': f'default_rng(({SEED}, month))', 'code_sha256': {'dh_read_iso.py': sha(HERE)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    for k, v in cells.items():
        print(k, v['rows'], v['w_from_other_month'], v['mse_V'], v['mse_DH'], (v['gain'], v['lo95'], v['hi95']))
    for k, v in directions.items():
        print(k, v['jan_w_from_jul']['gain'], v['jul_w_from_jan']['gain'], v['average'])
    print('G', json.dumps({f: {x: g[x] for x in ('gain', 'lo95', 'hi95')} for f, g in Gs.items()}))
    print('part_w_old_definition', json.dumps(part_w))
    print('iso_vs_saved', json.dumps(iso_rec))

    if a.natural_out is None:
        return
    # natural: rows dark in fact, lit members, all airports; V and w of each direction from mo only
    ids = fold[ID].to_numpy('int64')
    f = pd.read_parquet(a.flags).set_index(ID)
    fl = {c: f[c].reindex(ids).fillna(False).to_numpy(bool) for c in FLAGS}
    fl['seen'] = ~f['own_absent'].reindex(ids).fillna(True).to_numpy(bool)
    nat = {'what': 'DH pair on 2025 held rows dark in fact, the scored month out of every fitted stage (review repair)',
           'isolation': rec['isolation'], 'iso': iso_in, 'weights_V': {m: {k: v for k, v in w.items()
                                                                           if not k.startswith('dark_set')}
                                                                       for m, w in W.items()},
           'inputs': rec['inputs'], 'draws': DRAWS, 'seed': rec['seed'], 'flags': {},
           'code_sha256': rec['code_sha256']}
    for name in (*FLAGS, 'seen'):
        g = fl[name]
        r = {'w_fit': {}, 'months': {}, 'airports': {}}
        dd = {}
        for m, mo in ((1, 7), (7, 1)):
            vm = Vmo[m]['lit']
            ok = typed & np.isfinite(vm)
            fit = ok & g & (month == mo)
            w = DR.pair_w(y[fit], vm[fit], dh[fit]) if fit.sum() >= 50 else 0.
            r['w_fit'][m] = {'w_from_other_month': round(w, 3), 'fit_rows': int(fit.sum())}
            s = ok & g & (month == m)
            if s.sum() < 50:
                r['months'][m] = {'rows': int(s.sum())}
                continue
            days, pick = days_pick[m]
            p = np.maximum(60., vm[s] + w * (dh[s] - vm[s]))
            delta = (y[s] - vm[s]) ** 2 - (y[s] - p) ** 2
            pt, dr = NS.cell_draws(day[s], delta, days, pick)
            dd[m] = (pt, dr)
            r['months'][m] = {'rows': int(s.sum()), 'mse_V': round(float(np.mean((y[s] - vm[s]) ** 2)), 1),
                              'mse_DH': round(float(np.mean((y[s] - dh[s]) ** 2)), 1),
                              'mse_pair': round(float(np.mean((y[s] - p) ** 2)), 1), **NS.iv(pt, dr),
                              'by_class': {z: int((cls[s] == z).sum()) for z in ('clock', 'ordinary', 'noac')}}
            for ap in pd.unique(airport[s]):
                t = s & (airport == ap)
                if t.sum() < 200:
                    continue
                pa = np.maximum(60., vm[t] + w * (dh[t] - vm[t]))
                da = (y[t] - vm[t]) ** 2 - (y[t] - pa) ** 2
                pt_a, dr_a = NS.cell_draws(day[t], da, days, pick)
                r['airports'][f'{m}-{ap}'] = {'rows': int(t.sum()), 'mse_V': round(float(np.mean((y[t] - vm[t]) ** 2)), 1),
                                              'mse_DH': round(float(np.mean((y[t] - dh[t]) ** 2)), 1),
                                              **NS.iv(pt_a, dr_a), 'spread': NS.spread(lits[m], t)}
        if 1 in dd and 7 in dd:
            r['average'] = NS.iv(0.5 * (dd[1][0] + dd[7][0]), 0.5 * (dd[1][1] + dd[7][1]))
        nat['flags'][name] = r
    a.natural_out.write_text(json.dumps(nat, indent=1, default=str) + '\n')
    for name, r in nat['flags'].items():
        print(name, json.dumps(r['w_fit']), json.dumps(r.get('average')))
        for m, d in r['months'].items():
            print(' ', m, json.dumps(d))


if __name__ == '__main__':
    main()
