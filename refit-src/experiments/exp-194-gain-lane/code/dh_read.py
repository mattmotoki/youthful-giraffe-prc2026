"""exp-194 lever (b) read: DH (code/dh_fit.py) on 2025 clock rows made dark and on natural ordinary rows (SPEC step 3).

    OMP_NUM_THREADS=2 .venv/bin/python -I experiments/exp-194-gain-lane/code/dh_read.py --export <release export> \
        --dh work/dh --out experiments/exp-194-gain-lane/out/dh_read.json

Rows: the 2025 typed member-scope rows of January/July (fold frame of the release pack) and February/June (jjx sets).
"Made dark": a clock row read through the ordinary path, as if it had no head value. Ordinary path ("ord"): the C3
form of read_a (route L + Seq170 + Row170 + mix12 in the mlp slot, WLS on L), ordinary-row weights fitted on the other
month(s): January from July, July from January, February/June from January+July (pooled airports, final-mode exposure
weights). The members and L saw the lit ADS-B inputs on these rows, so ord is optimistic there. January/July also
carry the A6 form (R4mk + the six released members) as a second comparator.
Pair: ord + w (DH - ord) with w per airport from the made-dark rows of the other month(s), clipped to [0, 1].
G: sum over the made-dark cells of 2026 (2025 clock share minus 2026 head share >= 0.10) of the 2026 rows without a
head value times the mean gain, split into made-dark and natural rows by the 2025 shares; over 670,790 final rows;
airport-day draws (read_a.gstat, seed 601).
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
import read_a as RA  # noqa: E402

ID = 'MVT_ID_mvt'
N_FINAL = 670790
MIN_FIT = 300           # rows of an airport for its own pair weight; else the pooled weight
EXCESS = 0.10           # made-dark cell: 2025 clock share minus 2026 head share


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


STAGE = 'DHo'


def attach_dh(d, dh_dir, name):
    q = pd.read_parquet(dh_dir / f'{STAGE}-{name}.parquet')
    assert q[ID].is_unique
    s = q.set_index(ID)
    d = d.copy()
    d['DH'] = s.pred_DH.reindex(d[ID].to_numpy('int64')).to_numpy(float)
    d['dh_cls'] = s.cls.reindex(d[ID].to_numpy('int64')).to_numpy(object)
    return d, sha(dh_dir / f'{STAGE}-{name}.parquet')


def ord_weights(fit_rows, form, wcol):
    base, members = RA.FORMS[form]
    tr = RA.scope(fit_rows, base) & fit_rows.cls.eq('ordinary').to_numpy() & \
        RA.ready(fit_rows, base, members, 'mix12') & (wcol > 0)
    return RA.fit(fit_rows[base].to_numpy(float)[tr], RA.matrix(fit_rows, members, 'mix12')[tr],
                  fit_rows.y.to_numpy(float)[tr], wcol[tr])


def ord_path(d, form, beta):
    base, members = RA.FORMS[form]
    ok = RA.scope(d, base) & RA.ready(d, base, members, 'mix12')
    v = np.full(len(d), np.nan)
    v[ok] = RA.apply(d[base].to_numpy(float)[ok], RA.matrix(d, members, 'mix12')[ok], beta)
    return v


def pair_w(y, o, h):
    u = h - o
    den = float((u * u).sum())
    return float(np.clip(((y - o) * u).sum() / den, 0., 1.)) if den > 0 else 0.


def weights_by_airport(d, sel):
    out = {'_pooled': pair_w(d.y.to_numpy(float)[sel], d.ord.to_numpy(float)[sel], d.DH.to_numpy(float)[sel])}
    for a in sorted(d.airport[sel].unique()):
        m = sel & d.airport.eq(a).to_numpy()
        if m.sum() >= MIN_FIT:
            out[a] = pair_w(d.y.to_numpy(float)[m], d.ord.to_numpy(float)[m], d.DH.to_numpy(float)[m])
    return out


def cell_table(d, months):
    rows = []
    for m in months:
        for a in sorted(d.airport.unique()):
            for z in ('clock', 'ordinary'):
                s = d.month.eq(m).to_numpy() & d.airport.eq(a).to_numpy() & d.cls.eq(z).to_numpy() & d.ok.to_numpy()
                if s.sum() < 50:
                    continue
                y = d.y.to_numpy(float)[s]
                r = {'month': int(m), 'airport': a, 'rows_as': 'made_dark' if z == 'clock' else 'natural',
                     'rows': int(s.sum()), 'w': round(float(d.w_pair.to_numpy(float)[s][0]), 3)}
                for k in ('ord', 'DH', 'pair') + (('head_path', 'A6ord') if 'A6ord' in d else ('head_path',)):
                    if k in d and np.isfinite(d[k].to_numpy(float)[s]).all():
                        r[f'mse_{k}'] = round(float(np.mean((y - d[k].to_numpy(float)[s]) ** 2)), 1)
                r['gain_pair'] = round(r['mse_ord'] - r['mse_pair'], 1)
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--dh', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', default='DHo', choices=['DH', 'DHo'])
    p.add_argument('--chain', type=Path, help='values.npz of assemble_r9.py --variant base (fold_ids, fold_final): '
                   'the released zabd values for the natural-row check')
    a = p.parse_args()
    global STAGE
    STAGE = a.stage
    fold, pack = RA.fold_frame(a.export.resolve())
    cells = RA.cells_2026()
    wfin = RA.final_exposure(fold, cells)
    fold, h_jj = attach_dh(fold, a.dh, 'jjrefit')
    mo = pd.concat([RA.month_frame(2), RA.month_frame(6)], ignore_index=True)
    mo02, h02 = attach_dh(mo[mo.month.eq(2)], a.dh, 'jjx-02')
    mo06, h06 = attach_dh(mo[mo.month.eq(6)], a.dh, 'jjx-06')
    mo = pd.concat([mo02, mo06], ignore_index=True)
    # checks: DH's class is the frames' class on every row it scores
    for d, nm in ((fold, 'fold'), (mo, 'febjun')):
        on = d.typed.to_numpy() & np.isfinite(d.DH.to_numpy(float))
        assert (d.cls[on].astype(str).to_numpy() == d.dh_cls[on].astype(str).to_numpy()).mean() > 0.999, nm

    # ordinary path: January from July, July from January (other-month exposure weights), Feb/Jun from Jan+Jul
    fold['ord'] = np.nan
    fold['A6ord'] = np.nan
    w_fold = fold.w.to_numpy(float)
    for m, other in ((1, 7), (7, 1)):
        fr = fold[fold.month.eq(other)]
        for form, col in (('C3', 'ord'), ('A6', 'A6ord')):
            beta = ord_weights(fr, form, w_fold[fold.month.eq(other).to_numpy()])
            sel = fold.month.eq(m).to_numpy()
            fold.loc[sel, col] = ord_path(fold[sel], form, beta)
    beta_fm = ord_weights(fold, 'C3', wfin)
    mo['ord'] = ord_path(mo, 'C3', beta_fm)
    # head path on clock rows (lit; reference only): head 0.5 base, C3 clock weights (pooled Jan+Jul for Feb/Jun)
    hb = lambda d: np.maximum(60., 0.5 * d.L.to_numpy(float) + 0.5 * d.HB3u.to_numpy(float))  # noqa: E731
    base, members = RA.FORMS['C3']
    tr = RA.scope(fold, base) & fold.cls.eq('clock').to_numpy() & RA.ready(fold, base, members, 'mix12') & \
        np.isfinite(fold.HB3u.to_numpy(float)) & (wfin > 0)
    beta_clk = RA.fit(hb(fold)[tr], RA.matrix(fold, members, 'mix12')[tr], fold.y.to_numpy(float)[tr], wfin[tr])
    for d in (fold, mo):
        ok = RA.scope(d, base) & RA.ready(d, base, members, 'mix12') & np.isfinite(d.HB3u.to_numpy(float))
        v = np.full(len(d), np.nan)
        v[ok] = RA.apply(hb(d)[ok], RA.matrix(d, members, 'mix12')[ok], beta_clk)
        d['head_path'] = v
    for d in (fold, mo):
        d['ok'] = d.typed.to_numpy() & np.isfinite(d.ord.to_numpy(float)) & np.isfinite(d.DH.to_numpy(float)) & \
            d.cls.isin(['clock', 'ordinary']).to_numpy()

    # pair weights per airport from made-dark rows of the other month(s)
    W = {}
    md_f = fold.ok.to_numpy() & fold.cls.eq('clock').to_numpy()
    W[1] = weights_by_airport(fold, md_f & fold.month.eq(7).to_numpy())
    W[7] = weights_by_airport(fold, md_f & fold.month.eq(1).to_numpy())
    W[2] = W[6] = weights_by_airport(fold, md_f)
    for d in (fold, mo):
        w = np.array([W[int(m)].get(ap, W[int(m)]['_pooled']) for m, ap in zip(d.month, d.airport)])
        d['w_pair'] = w
        d['pair'] = np.maximum(60., d.ord.to_numpy(float) + w * (d.DH.to_numpy(float) - d.ord.to_numpy(float)))

    both = pd.concat([fold.assign(src='fold'), mo.assign(src='febjun')], ignore_index=True)
    table = cell_table(both, (1, 2, 6, 7))

    # 2026 made-dark cells and G on 2026 exposure
    tp = both[both.typed]
    share25 = tp.groupby(['month', 'airport']).cls.apply(lambda s: float((s == 'clock').mean()))
    exp26, md_cells = {}, []
    for (m, ap), s25 in share25.items():
        clk = cells.get((m, ap, 'clock'), 0); ordn = cells.get((m, ap, 'ordinary'), 0)
        typed26 = clk + ordn
        if typed26 == 0:
            continue
        s26 = clk / typed26
        if s25 - s26 < EXCESS:
            continue
        nat = min(ordn, round((1 - s25) * typed26))
        exp26[(int(m), ap, 'clock')] = ordn - nat     # made-dark rows: read on 2025 clock rows
        exp26[(int(m), ap, 'ordinary')] = nat           # natural rows: read on 2025 ordinary rows
        md_cells.append({'month': int(m), 'airport': ap, 'share_2025': round(s25, 3), 'share_2026': round(s26, 3),
                         'rows_no_head_2026': int(ordn), 'made_dark': int(ordn - nat), 'natural': int(nat)})
    g = both[both.ok]
    d = (g.y - g.ord) ** 2 - (g.y - g.pair) ** 2
    G = RA.gstat(g, d.to_numpy(float), exp26, N_FINAL)
    d_dh = (g.y - g.ord) ** 2 - (g.y - g.DH) ** 2
    G_dh_alone = RA.gstat(g, d_dh.to_numpy(float), exp26, N_FINAL)
    # EDDF February made-dark rows (the lead's read)
    e = both.ok & both.airport.eq('EDDF') & both.month.eq(2) & both.cls.eq('clock')
    eddf = {k: round(float(np.mean((both.y[e] - both[k][e]) ** 2)), 1) for k in ('ord', 'DH', 'pair', 'head_path')}
    eddf['rows'] = int(e.sum()); eddf['w'] = W[2].get('EDDF')
    # January/July with the A6 form (R4mk + the six released members; the form of V's primary step)
    f6 = fold.assign(ord=fold.A6ord)
    f6['ok'] = f6.ok.to_numpy() & np.isfinite(f6.ord.to_numpy(float))
    md6 = f6.ok.to_numpy() & f6.cls.eq('clock').to_numpy()
    W6 = {1: weights_by_airport(f6, md6 & f6.month.eq(7).to_numpy()),
          7: weights_by_airport(f6, md6 & f6.month.eq(1).to_numpy())}
    w6 = np.array([W6[int(m)].get(ap, W6[int(m)]['_pooled']) for m, ap in zip(f6.month, f6.airport)])
    f6['w_pair'] = w6
    f6['pair'] = np.maximum(60., f6.ord.to_numpy(float) + w6 * (f6.DH.to_numpy(float) - f6.ord.to_numpy(float)))
    table6 = cell_table(f6.drop(columns='A6ord'), (1, 7))
    g6 = f6[f6.ok]
    exp26_jj = {k: v for k, v in exp26.items() if k[0] in (1, 7)}
    G6 = RA.gstat(g6, ((g6.y - g6.ord) ** 2 - (g6.y - g6.pair) ** 2).to_numpy(float), exp26_jj, N_FINAL)
    G3jj = RA.gstat(g, d.to_numpy(float), exp26_jj, N_FINAL)
    W6_serve = weights_by_airport(f6, md6)
    # serve weights for 2026 (SPEC step 3: the part uses the A6 weights, January+July made-dark rows per airport);
    # the C3 weights from all four months are written beside them
    md_all = both.ok.to_numpy() & both.cls.eq('clock').to_numpy()
    W_serve = weights_by_airport(both, md_all)
    serve = {'cells': [{'month': c['month'], 'airport': c['airport'],
                        'w': round(W6_serve.get(c['airport'], W6_serve['_pooled']), 4)} for c in md_cells],
             'weights_by_airport': {k: round(v, 4) for k, v in W6_serve.items()},
             'weights_by_airport_c3_four_months': {k: round(v, 4) for k, v in W_serve.items()},
             'source': 'exp-194 dh_read.py: A6 form, made-dark 2025 rows of January and July'}
    natural = None
    if a.chain:
        v = np.load(a.chain)
        z = pd.Series(v['fold_final'], index=v['fold_ids']).reindex(fold[ID].to_numpy('int64')).to_numpy(float)
        natural = {'chain_sha256': sha(a.chain)}
        for ap in sorted({c['airport'] for c in md_cells}):
            for m in (1, 7):
                s_ = fold.ok.to_numpy() & fold.cls.eq('ordinary').to_numpy() & fold.airport.eq(ap).to_numpy() & \
                    fold.month.eq(m).to_numpy() & np.isfinite(z)
                if s_.sum() < 50:
                    continue
                wv = W6_serve.get(ap, W6_serve['_pooled'])
                y = fold.y.to_numpy(float)[s_]; zz = z[s_]; dh = fold.DH.to_numpy(float)[s_]
                natural[f'{m}-{ap}'] = {'rows': int(s_.sum()), 'w': round(wv, 4),
                                        'mse_zabd': round(float(np.mean((y - zz) ** 2)), 1),
                                        'mse_pair_on_zabd': round(float(np.mean((y - np.maximum(60., zz + wv * (dh - zz))) ** 2)), 1),
                                        'mse_DH': round(float(np.mean((y - dh) ** 2)), 1)}
    a.out.with_name('dh_weights.json').write_text(json.dumps(serve, indent=1) + '\n')
    # label-free transfer check on 2025 clock rows: DH minus the head value (compare dh_apply on 2026 clock rows)
    c = both[both.ok & both.cls.eq('clock') & np.isfinite(both.HB3u)]
    dd = (c.DH - c.HB3u)
    chk = {f'{m}-{ap}': {'rows': int(len(v)), 'mean_dh_minus_head': round(float(v.mean()), 2),
                         'rms_dh_minus_head': round(float(np.sqrt((v ** 2).mean())), 2)}
           for (m, ap), v in dd.groupby([c.month, c.airport])}
    rec = {'code': {HERE.name: sha(HERE), 'read_a.py': sha(HERE.parent / 'read_a.py')},
           'check_2025_clock_rows': chk, 'serve_weights': serve,
           'stage': STAGE, 'inputs': {'jjrefit': h_jj, 'jjx-02': h02, 'jjx-06': h06,
                      'export_pack_sha256': sha(a.export / 'release/ZABD-PACK.json')},
           'eddf_feb_made_dark': eddf, 'pair_weights': W, 'made_dark_cells_2026': md_cells,
           'G_pair_2026_exposure': G, 'G_dh_alone_2026_exposure': G_dh_alone, 'cells': table,
           'janjul_A6': {'pair_weights': W6, 'G_pair_janjul_exposure': G6, 'G_C3_pair_janjul_exposure': G3jj,
                         'cells': table6}, 'natural_rows_on_zabd': natural,
           'comparator': 'C3 ordinary path, members and L with lit ADS-B inputs (optimistic); January/July also A6',
           'rows_outside': {'fold_typed_no_ord': int((fold.typed & ~np.isfinite(fold.ord)).sum()),
                            'febjun_typed_no_ord': int((mo.typed & ~np.isfinite(mo.ord)).sum())}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(RA.rnd(rec), indent=1, default=str) + '\n')
    print(json.dumps(RA.rnd({'G_A6_janjul': {k: G6[k] for k in ('G', 'lo95', 'hi95', 'wo5')},
                             'G_C3_janjul': {k: G3jj[k] for k in ('G', 'lo95', 'hi95', 'wo5')},
                             'serve': serve['weights_by_airport'], 'natural_rows_on_zabd': natural}), indent=1))
    print(json.dumps(RA.rnd({'eddf_feb_made_dark': eddf, 'G': {k: G[k] for k in ('G', 'lo95', 'hi95', 'wo5')},
                             'G_views': G['views'], 'G_dh_alone': G_dh_alone['G'], 'cells_2026': md_cells}), indent=1))
    for r in table:
        if r['rows_as'] == 'made_dark':
            print(r)


if __name__ == '__main__':
    main()
