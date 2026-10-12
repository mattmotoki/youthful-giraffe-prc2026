"""exp-194 lever (b): the read of PLAN-DARK, V forced dark with no stand-in (lead 07:45 UTC 10-09).

Rows: 2025 January and July, typed member-scope rows of the release fold frame (read_a.fold_frame). Members of A6
(base R4mk; g, mlp in the mix12 slot, sequence021, sequence039, dark18, row6):
  lit    R4mk = R4mks pred_Sb of the saved held level 2 (r4mk_dark.py, lit), the others the saved held vectors
  dark   on every row of EDDF and LIRF of the month (the airport-month has no surface reception): R4mk pred_Sb
         dark (r4mk_dark.py; L and Lp lit or dark as --r4 says), mix12 + (dark - lit) (mlp_dark.py), the exp-170
         bags' forced-dark means (x170_dark.py); g reads official columns only and keeps its value
Frames on the scored month m (weights always from the other month m'):
  lit       V = A6 with the released form's weights (ordinary rows of m', lit members), lit members
  dark      V = the same weights, dark members (V in 2026 if its members are dark)
  dark_set  A6 re-weighted on the dark members of m' (EDDF and LIRF rows, unit weights), dark members
Pair = max(60, V + w (DH - V)); w per airport from the m' typed rows of the frame (both classes: the airport-month
is dark and the part serves both; LIRF January has no clock rows), clipped to [0, 1] (dh_read pair_w). Gain = MSE(V) - MSE(pair) per row, per month x airport x class (clock = made dark, ordinary =
natural), 2,000 joint draws of UTC days within each month (seed 601). G at an airport: sum over its 2026 made-dark
cells (dh_read made_dark_cells_2026; January and February from the January gain, June and July from July) of made-dark
rows x clock gain + natural rows x ordinary gain, over 670,790 final rows. With --flags (dh_dark_rows.py): the
member spread and errors of 2025 rows whose own inputs are absent, against the forced-dark rows (lead's (b)), and
the 2026 no-head rows that carry each flag.

    .venv/bin/python -I experiments/exp-194-gain-lane/code/dh_read_nostand.py --export <release export> \
        --dh work/dh --dark work/dark-<tag> --r4 litL --out experiments/exp-194-gain-lane/out/<name>.json \
        [--flags work/dark-<tag>/dark_rows.parquet]
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
import dh_read as DR  # noqa: E402
import read_a as RA  # noqa: E402

ID = 'MVT_ID_mvt'
N_FINAL = 670790
AIRPORTS = ['EDDF', 'LIRF']
BAGS = ['sequence021', 'sequence039', 'dark18', 'row6']
SIX = ['g', 'mlp', 'sequence021', 'sequence039', 'dark18', 'row6']
SEASON = {1: 1, 2: 1, 6: 7, 7: 7}
DRAWS, SEED = 2000, 601
TOL = 0.05
R4_SAVED = str(HERE.parents[2] / 'exp-169-stack-plan/inputs/stack10-025/R4mksFjjrefit-month-2025-{m}.parquet')


def load_dark(fold, dark_dir: Path, r4: str):
    """Lit and dark member columns on the fold frame. `dark_dir` holds the EDDF/LIRF rows of each output
    (pack_fetch.py names); the replay of each lit value must equal the saved held vector."""
    ids = fold[ID].to_numpy('int64')
    on_ap = fold.airport.isin(AIRPORTS).to_numpy()
    lit, dk, rec = fold.copy(), fold.copy(), {}
    S = pd.concat([pd.read_parquet(R4_SAVED.format(m=m)) for m in ('01', '07')]).set_index(ID)
    sb = S.pred_Sb.reindex(ids).to_numpy(float)
    r = pd.read_parquet(dark_dir / f'r4mk-{r4}.parquet').set_index(ID)
    rl = r.pred_Sb_lit.reindex(ids).to_numpy(float)
    rd = r.pred_Sb_dark.reindex(ids).to_numpy(float)
    got = np.isfinite(rl)
    assert (got == (on_ap & np.isfinite(sb))).all(), 'the R4mk rows differ from the saved fold output'
    rec['R4mk'] = {'source': f'r4mk-{r4}.parquet and the saved fold output', 'rows_dark_view': int(got.sum()),
                   'replay_max_abs_s': float(np.abs(rl[got] - sb[got]).max()),
                   'pred_Sb_minus_pack_R4mk_rms_s': float(np.sqrt(np.nanmean((sb - fold.R4mk.to_numpy(float)) ** 2)))}
    assert rec['R4mk']['replay_max_abs_s'] <= 1e-6, rec['R4mk']
    lit['R4mk'] = sb
    dk['R4mk'] = np.where(got, rd, sb)
    q = pd.read_parquet(dark_dir / 'mix12-dark.parquet').set_index(ID)
    ql = q.pred_lit.reindex(ids).to_numpy(float)
    qd = q.pred_dark.reindex(ids).to_numpy(float)
    have = np.isfinite(ql) & np.isfinite(fold.mix12.to_numpy(float))
    assert (have == (on_ap & np.isfinite(fold.mix12.to_numpy(float)))).all(), 'mix12 rows differ'
    d = ql[have] - fold.mix12.to_numpy(float)[have]
    rec['mix12'] = {'rows': int(have.sum()), 'replay_max_abs_s': float(np.abs(d).max())}
    assert rec['mix12']['replay_max_abs_s'] <= TOL, rec['mix12']
    dk['mix12'] = np.where(have, fold.mix12.to_numpy(float) + qd - ql, fold.mix12.to_numpy(float))
    for b in BAGS:
        x = pd.read_parquet(dark_dir / f'x170-{b}.parquet').set_index(ID)
        xl = x.pred_lit.reindex(ids).to_numpy(float)
        xd = x.pred_dark.reindex(ids).to_numpy(float)
        on = np.isfinite(xl)
        assert (on == (on_ap & np.isfinite(fold[b].to_numpy(float)))).all(), f'{b} rows differ'
        d = xl[on] - fold[b].to_numpy(float)[on]
        rec[b] = {'rows': int(on.sum()), 'replay_max_abs_s': float(np.nanmax(np.abs(d)))}
        assert rec[b]['replay_max_abs_s'] <= TOL, (b, rec[b])
        dk[b] = np.where(on, xd, fold[b].to_numpy(float))
    return lit, dk, rec


def a6(d):
    base, members = RA.FORMS['A6']
    return d[base].to_numpy(float), RA.matrix(d, members, 'mix12')


def frames_for(lit, dk, m, mo_other):
    """V of each frame on all rows (NaN off scope); weights from the other month."""
    base, members = RA.FORMS['A6']
    w = lit.w.to_numpy(float)
    ok_l = RA.scope(lit, base) & RA.ready(lit, base, members, 'mix12')
    ok_d = RA.scope(lit, base) & RA.ready(dk, base, members, 'mix12')
    other = lit.month.eq(mo_other).to_numpy()
    tr = ok_l & lit.cls.eq('ordinary').to_numpy() & (w > 0) & other
    bl, Ml = a6(lit)
    bd, Md = a6(dk)
    beta = RA.fit(bl[tr], Ml[tr], lit.y.to_numpy(float)[tr], w[tr])
    trd = ok_d & other & lit.airport.isin(AIRPORTS).to_numpy()
    beta_d = RA.fit(bd[trd], Md[trd], lit.y.to_numpy(float)[trd], np.ones(int(trd.sum())))
    V = {k: np.full(len(lit), np.nan) for k in ('lit', 'dark', 'dark_set')}
    V['lit'][ok_l] = RA.apply(bl[ok_l], Ml[ok_l], beta)
    V['dark'][ok_d] = RA.apply(bd[ok_d], Md[ok_d], beta)
    V['dark_set'][ok_d] = RA.apply(bd[ok_d], Md[ok_d], beta_d)
    names = [base] + members
    return V, {'released_form_weights': dict(zip(members, np.round(beta, 4).tolist())),
               'dark_set_weights': dict(zip(members, np.round(beta_d, 4).tolist())), 'base': base,
               'fit_rows': int(tr.sum()), 'dark_set_fit_rows': int(trd.sum()), 'members': names}


def draws_for(day):
    days = np.unique(day)
    rng = np.random.default_rng(SEED + int(days.size))
    return days, rng.integers(0, len(days), size=(DRAWS, len(days)))


def cell_draws(day, delta, days, pick):
    """Mean of delta per draw (day blocks): (point, draws)."""
    idx = {k: np.flatnonzero(day == k) for k in days}
    s = np.array([delta[idx[k]].sum() for k in days])
    n = np.array([len(idx[k]) for k in days])
    return float(delta.mean()), s[pick].sum(1) / np.maximum(n[pick].sum(1), 1)


def iv(point, dr):
    lo, hi = np.quantile(dr, [0.025, 0.975])
    return {'gain': round(point, 1), 'lo95': round(float(lo), 1), 'hi95': round(float(hi), 1)}


def spread(d, s):
    out = {}
    for a, b in (('dark18', 'sequence039'), ('dark18', 'g'), ('sequence039', 'g')):
        x = d[a].to_numpy(float)[s] - d[b].to_numpy(float)[s]
        out[f'{a}-{b}'] = round(float(np.sqrt(np.nanmean(x ** 2))), 1)
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--export', type=Path, required=True)
    ap_.add_argument('--dh', type=Path, required=True)
    ap_.add_argument('--dark', type=Path, required=True, help='fetched outputs of run_dark_cpu.sh and the GPU bags')
    ap_.add_argument('--r4', choices=['litL', 'darkL'], required=True)
    ap_.add_argument('--flags', type=Path)
    ap_.add_argument('--out', type=Path, required=True)
    a = ap_.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; write-once')
    DR.STAGE = 'DHo'
    fold, _ = RA.fold_frame(a.export.resolve())
    fold, h_jj = DR.attach_dh(fold, a.dh, 'jjrefit')
    lit, dk, parity = load_dark(fold, a.dark, a.r4)
    y, dh, day = fold.y.to_numpy(float), fold.DH.to_numpy(float), fold.day.to_numpy()
    month, airport, cls = fold.month.to_numpy(), fold.airport.to_numpy(), fold.cls.to_numpy()
    typed = fold.typed.to_numpy() & np.isfinite(dh)
    V, W, gains, draws, pw = {}, {}, {}, {}, {}
    for m, mo in ((1, 7), (7, 1)):
        Vm, W[m] = frames_for(lit, dk, m, mo)
        for k, v in Vm.items():
            V.setdefault(k, np.full(len(fold), np.nan))
            V[k][month == m] = v[month == m]
    out_cells = {}
    for fr in ('lit', 'dark', 'dark_set'):
        v = V[fr]
        ok = typed & np.isfinite(v)
        pw[fr] = {}
        for ap in AIRPORTS:
            for m, mo in ((1, 7), (7, 1)):
                fit_rows = ok & (month == mo) & (airport == ap)   # every typed row: the part serves both classes
                pw[fr][f'{ap}-{m}'] = DR.pair_w(y[fit_rows], v[fit_rows], dh[fit_rows])
            # the part's parameter: w on the January and July rows together (2026 months are neither)
            both = ok & np.isin(month, (1, 7)) & (airport == ap)
            pw[fr][f'{ap}-both'] = DR.pair_w(y[both], v[both], dh[both])
        for m in (1, 7):
            days, pick = draws_for(day[month == m])
            for ap in AIRPORTS:
                w = pw[fr][f'{ap}-{m}']
                for z in ('clock', 'ordinary'):
                    s = ok & (month == m) & (airport == ap) & (cls == z)
                    if s.sum() < 50:
                        continue
                    p = np.maximum(60., v[s] + w * (dh[s] - v[s]))
                    delta = (y[s] - v[s]) ** 2 - (y[s] - p) ** 2
                    pt, dr = cell_draws(day[s], delta, days, pick)
                    key = f'{fr}|{m}-{ap}-{z}'
                    draws[key] = (pt, dr)
                    out_cells[key] = {'rows': int(s.sum()), 'w_from_other_month': round(w, 3),
                                      'mse_V': round(float(np.mean((y[s] - v[s]) ** 2)), 1),
                                      'mse_DH': round(float(np.mean((y[s] - dh[s]) ** 2)), 1),
                                      'mse_pair': round(float(np.mean((y[s] - p) ** 2)), 1), **iv(pt, dr),
                                      'days_with_gain': round(float(np.mean(
                                          [delta[day[s] == k].sum() > 0 for k in np.unique(day[s])])), 3)}
                    # w 0.25 (NOTE-READ-B) for comparison
                    p25 = np.maximum(60., v[s] + 0.25 * (dh[s] - v[s]))
                    out_cells[key]['gain_w0.25'] = round(float(np.mean((y[s] - v[s]) ** 2 - (y[s] - p25) ** 2)), 1)
    # G at each airport from its 2026 made-dark cells (joint day draws within each month)
    cells26 = [c for c in json.loads((HERE.parents[1] / 'out/dh_read-o1.json').read_text())['made_dark_cells_2026']
               if c['airport'] in AIRPORTS]
    G = {}
    for fr in ('lit', 'dark', 'dark_set'):
        for ap in AIRPORTS:
            pt, dr, used = 0., np.zeros(DRAWS), []
            for c in (c for c in cells26 if c['airport'] == ap):
                s = SEASON[c['month']]
                for z, n in (('clock', c['made_dark']), ('ordinary', c['natural'])):
                    key = f'{fr}|{s}-{ap}-{z}'
                    if key not in draws:
                        continue
                    pt += n * draws[key][0] / N_FINAL
                    dr += n * draws[key][1] / N_FINAL
                    used.append((c['month'], z, n))
            if used:
                G[f'{fr}|{ap}'] = {**iv(pt, dr), 'rows_2026': int(sum(n for *_, n in used))}
    # lead's (a): each direction (January scored with w fitted on July, July with w fitted on January) and their
    # equal-weight average per row; the two months draw days independently, so the average's draws are the mean
    directions = {}
    for fr in ('lit', 'dark', 'dark_set'):
        for ap in AIRPORTS:
            for z in ('clock', 'ordinary'):
                k1, k7 = f'{fr}|1-{ap}-{z}', f'{fr}|7-{ap}-{z}'
                if k1 in draws and k7 in draws:
                    directions[f'{fr}|{ap}-{z}'] = {
                        'jan_w_from_jul': {k: out_cells[k1][k] for k in ('gain', 'lo95', 'hi95', 'w_from_other_month')},
                        'jul_w_from_jan': {k: out_cells[k7][k] for k in ('gain', 'lo95', 'hi95', 'w_from_other_month')},
                        'average': iv(0.5 * (draws[k1][0] + draws[k7][0]), 0.5 * (draws[k1][1] + draws[k7][1]))}
    rec = {'what': 'DH pair against V forced dark with no stand-in (PLAN-DARK read); w from the other month',
           'r4': a.r4, 'parity': parity, 'weights': W, 'pair_w': pw, 'cells': out_cells, 'directions': directions,
           'G_per_final_row': G,
           'inputs': {'DH_jjrefit': h_jj, 'dark_dir': str(a.dark)}, 'draws': DRAWS, 'seed': SEED}
    # spreads: lit 2025 clock rows, forced-dark rows, natural rows by dark flag (lead's (b))
    sp = {}
    for m in (1, 7):
        for ap in AIRPORTS:
            s = typed & (month == m) & (airport == ap) & (cls == 'clock')
            sp[f'{m}-{ap}-clock-lit'] = spread(lit, s)
            sp[f'{m}-{ap}-clock-forced_dark'] = spread(dk, s)
    rec['member_spread'] = sp
    if a.flags is not None:
        f = pd.read_parquet(a.flags).set_index(ID)
        ids = fold[ID].to_numpy('int64')
        fa = f.dark_a.reindex(ids).fillna(False).to_numpy(bool)
        fb = f.dark_b.reindex(ids).fillna(False).to_numpy(bool)
        own = f.own_absent.reindex(ids).fillna(False).to_numpy(bool)
        fl = {}
        for nm, g in (('dark_a', fa), ('dark_b', fb), ('own_absent_only', own & ~fa & ~fb), ('seen', ~own)):
            s = typed & g & np.isfinite(V['lit'])
            if s.sum() < 30:
                fl[nm] = {'rows': int(s.sum())}
                continue
            r = {'rows': int(s.sum()), 'by_airport': pd.Series(airport[s]).value_counts().head(8).to_dict(),
                 'spread': spread(lit, s), 'mse_V_lit': round(float(np.mean((y[s] - V['lit'][s]) ** 2)), 1),
                 'mse_DH': round(float(np.mean((y[s] - dh[s]) ** 2)), 1)}
            for w in (0.1, 0.25):
                p = np.maximum(60., V['lit'][s] + w * (dh[s] - V['lit'][s]))
                r[f'gain_w{w}'] = round(float(np.mean((y[s] - V['lit'][s]) ** 2 - (y[s] - p) ** 2)), 1)
            fl[nm] = r
        rec['flags_2025'] = fl
        rec['flags_2026'] = json.loads(a.flags.with_suffix('.json').read_text())['cells']
        # G on the 2026 rows the forced-dark frame models: no-head rows that are dark_b (whole airport-day dark),
        # each at the clock gain of its season (dark_b no-head rows of every 2026 month 1/2/6/7)
        Gb = {}
        for fr in ('lit', 'dark', 'dark_set'):
            for ap in AIRPORTS:
                pt, dr, used = 0., np.zeros(DRAWS), []
                for m in (1, 2, 6, 7):
                    c = rec['flags_2026'].get(f'2026-{m}-{ap}-no_head')
                    key = f'{fr}|{SEASON[m]}-{ap}-clock'
                    if c is None or key not in draws:
                        continue
                    n = c['rows'] * c['dark_b']
                    pt += n * draws[key][0] / N_FINAL
                    dr += n * draws[key][1] / N_FINAL
                    used.append((m, round(n)))
                if used:
                    Gb[f'{fr}|{ap}'] = {**iv(pt, dr), 'dark_b_rows_2026': used}
        rec['G_per_final_row_dark_b_rows'] = Gb
        # the same rows split by the month's 2025 clock share (made_dark_cells_2026 share_2025): on a lit day that
        # share of them has a clock (clock gain), the rest is ordinary (ordinary gain); cells with no share are left out
        share = {(c['month'], c['airport']): c['share_2025'] for c in cells26}
        Gs = {}
        for fr in ('lit', 'dark', 'dark_set'):
            for ap in AIRPORTS:
                pt, dr, used, by_month = 0., np.zeros(DRAWS), [], {}
                for m in (1, 2, 6, 7):
                    c = rec['flags_2026'].get(f'2026-{m}-{ap}-no_head')
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
                    Gs[f'{fr}|{ap}'] = {**iv(pt, dr), 'dark_b_rows_2026': used, 'by_month': by_month}
        rec['G_per_final_row_dark_b_rows_split'] = Gs
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    for k, v in out_cells.items():
        if 'EDDF' in k or 'LIRF' in k:
            print(k, v['rows'], v['w_from_other_month'], v['mse_V'], v['mse_DH'], (v['gain'], v['lo95'], v['hi95']),
                  v['gain_w0.25'])
    print(json.dumps(G, indent=0))
    for k, v in directions.items():
        print(k, v['jan_w_from_jul']['gain'], v['jul_w_from_jan']['gain'], v['average'])
    print(json.dumps(parity))
    print('w', json.dumps(pw))
    for k in ('G_per_final_row_dark_b_rows', 'G_per_final_row_dark_b_rows_split'):
        if k in rec:
            print(k, json.dumps({f: {x: g[x] for x in ('gain', 'lo95', 'hi95')} for f, g in rec[k].items()}))


if __name__ == '__main__':
    main()
