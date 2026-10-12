"""exp-194 read (a): R9 and the audit checks on February and June 2025, in a reduced chain from saved held sets.

Held sets (nine-month jjx fits: each excludes January, July and the scored month): L, HB3u, Seq170, Row170 (Mac),
mlp008 and mix12 (remote exp-164 vectors, copied to work/remote with their recorded SHA-256). January/July use the
ten-month janjul sets and the released pack (route R4mk, the six primary members, mlp008, exposure weights).

Reduced forms (ordinary member-scope rows; WLS of y - base on member - base, no intercept, exposure weights; floor 60):
  A6  base R4mk, members g, mlp, sequence021, sequence039, dark18, row6 (the released six; January/July only)
  B3  base R4mk, members Seq170, Row170, mlp (January/July only)
  C3  base L, members Seq170, Row170, mlp
  C1  base L, member mlp
R9 step of a form: G of the form with mix12 in the mlp slot over the form with mlp008, each with its own weights.
January/July: weights fitted on the other month (released fold form). February/June: weights fitted on January and
July together (pooled, the released 2026 form), and as a variant on the season month (Feb <- Jan, Jun <- Jul). The
fit weights for February/June are the final-mode exposure weights (2026 four-month cells through the season map).

G on February/June: cells month x airport x class of the 2025 rows, W_h = 2026 rows of the cell / 2026 rows of
February and June (class counts: exp-185 class-final files; the no-type rows: final_ranking.parquet); 2,000 draws of
airport-days within airport x month, seed 601. On January/July the same function is checked against prc/gscore.py.

Checks in the same run: the head weight on clock rows (head = (1 - a) base + a HB3u, a in 0.3..0.7, against 0.5),
and EDDF February 2025 clock rows through the ordinary path (C3 ordinary weights) against the head path (head at 0.5
plus the C3 clock-row weights).

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I experiments/exp-194-gain-lane/code/read_a.py \
        --export <release export> --out experiments/exp-194-gain-lane/out/read_a.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
X = ROOT / 'experiments/exp-194-gain-lane'
TW = ROOT / 'experiments/exp-159-clock-head/artifacts/travelw'
P15 = ROOT / 'experiments/exp-170-diverse/results/native-015/out/pack-level2-015'
MIX_FOLD = (ROOT / 'experiments/exp-164-double-down/results/stack-pack-012/mlp-fold.parquet',
            'fe4558cc807f4128552ac2d54e0aa27e769be3a6b84cadcee6124c6bb188494a')
REM = X / 'work/remote'
REMOTE = {  # month -> slot -> (local copy, SHA-256 recorded in the exp-164 pack-level2 JSON, remote source)
    2: {'mlp008': ('mlp008-jjx-02.parquet', '11e7b3f435b40af4ef408f647809300c55ccbe06639c3a416769470d985a2374',
                   'prc-exp164-work/out/mlp-008/jjx-02/mean-held.parquet'),
        'mix12': ('mix12-jjx-02.parquet', 'e1e9bdd7dd47526796a39ac7873e77ddd5142bff033a75856eaf8173521d5b49',
                  'prc-exp164-work/out/mlp-012/vectors/jjx-02-mixb12.parquet')},
    6: {'mlp008': ('mlp008-jjx-06.parquet', 'cd54544be547e34f8daa5ab3b32d803b3b2ff4cb690fd632ab51959bc65f696b',
                   'prc-exp164-work/out/mlp-008/jjx-06/mean-held.parquet'),
        'mix12': ('mix12-jjx-06.parquet', '149c5cb959f379001d0d657dcceeeb3e0ebe934f7e7ebfd5cef4f7a0ca461b93',
                  'prc-exp164-work/out/mlp-012/vectors/jjx-06-mixb12.parquet')}}
CLASS26 = ROOT / 'experiments/exp-185-final-data/checks/class-final-2026-{m:02d}-ad7cce0608c723b08f6a13d8d8733f6e6eda60b5.json'
FINAL_RANKING = ROOT / 'data/raw/final/final_ranking.parquet'
RAWF = {2: 'training_2025-02-01_2025-03-01.parquet', 6: 'training_2025-06-01_2025-07-01.parquet'}
ID = 'MVT_ID_mvt'
SEASON = {1: 1, 2: 1, 6: 7, 7: 7}
F_MAX = 2400.
SIX = ['g', 'mlp', 'sequence021', 'sequence039', 'dark18', 'row6']
FORMS = {'A6': ('R4mk', SIX), 'B3': ('R4mk', ['Seq170', 'Row170', 'mlp']), 'C3': ('L', ['Seq170', 'Row170', 'mlp']),
         'C1': ('L', ['mlp'])}
#: January/July only: which members the R9 refit gain needs (each form with mlp008 or mix12 in the mlp slot)
CALIB = {'A6-L': ('L', SIX), 'A5-g': ('R4mk', [m for m in SIX if m != 'g']),
         'A5-sequence021': ('R4mk', [m for m in SIX if m != 'sequence021']),
         'A5-sequence039': ('R4mk', [m for m in SIX if m != 'sequence039']),
         'A5-dark18': ('R4mk', [m for m in SIX if m != 'dark18']), 'A5-row6': ('R4mk', [m for m in SIX if m != 'row6']),
         'A6-Seq170-Row170': ('R4mk', ['g', 'mlp', 'Seq170', 'sequence039', 'dark18', 'Row170']),
         'R4mk-g-sequence039': ('R4mk', ['g', 'sequence039', 'mlp']), 'R4mk-g': ('R4mk', ['g', 'mlp']),
         'R4mk-sequence039': ('R4mk', ['sequence039', 'mlp']), 'L-g': ('L', ['g', 'mlp']),
         'L-g-Seq170': ('L', ['g', 'Seq170', 'mlp']), 'L-g-Seq170-Row170': ('L', ['g', 'Seq170', 'Row170', 'mlp']),
         'L-g-sequence039': ('L', ['g', 'sequence039', 'mlp']), 'L-sequence039': ('L', ['sequence039', 'mlp']),
         'L-Seq170-Row170-dark18': ('L', ['Seq170', 'Row170', 'dark18', 'mlp'])}
ALPHAS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
N_DRAWS, SEED = 2000, 601


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vec(path, want=None):
    if want:
        assert sha(path) == want, path
    q = pd.read_parquet(path)
    cols = [c for c in q.columns if c.startswith('pred')]
    assert len(cols) == 1 and q[ID].is_unique, (path, list(q.columns))
    return q.set_index(ID)[cols[0]].astype(float)


# ------------------------------------------------------------------ frames
def fold_frame(export):
    """January/July held rows in fold order with labels, classes, weights, route, members and the jjx-free sets."""
    sys.path.append(str(export / 'release'))
    import assembly_replay as R
    pack = json.loads((export / 'release/ZABD-PACK.json').read_text())
    off = R.official_tables(ROOT, pack['official_sha256'])
    fo = off['fold']; n = len(fo)

    def fr(key):
        spec = pack['frames'][key]
        f = R.rebuild_frame(spec, fo, n)
        assert R.frame_digest(f) == spec['digest'], key
        return f.set_index(ID)
    ids = fo[ID].to_numpy('int64')
    b, pf, ex, rf = fr('fold/bundle'), fr('fold/primary_frame'), fr('fold/exposure'), fr('fold/rule_frame')
    d = pd.DataFrame({ID: ids})
    d['month'] = b.month.reindex(ids).to_numpy()
    d['airport'] = b.airport.reindex(ids).to_numpy()
    d['cls'] = b.cls.reindex(ids).to_numpy()
    d['day'] = fo.MVT_TIME_UTC_mvt.dt.day.to_numpy()
    d['y'] = rf.y.reindex(ids).to_numpy(float)
    d['w'] = ex.weight.reindex(ids).to_numpy(float)
    d['scope_pack'] = pf.member_scope.reindex(ids).to_numpy(bool)
    d['R4mk'] = b.R4mk.reindex(ids).to_numpy(float)
    d['F'] = b.F.reindex(ids).to_numpy(float)
    d['typed'] = fo.AIRCRAFT_TYPE_mvt.notna().to_numpy()
    d['aobt3'] = fo.AOBT_3_flt.notna().to_numpy()
    d['plan'] = fo.LOBT_flt.notna().to_numpy()
    mlp = pf.mlp.reindex(ids).to_numpy(float)
    d['mlp008'] = np.where(d.cls.eq('clock'), np.nan, mlp)  # the pack column holds mlp008 on every non-clock row
    d['mix12'] = vec(*MIX_FOLD).reindex(ids).to_numpy()
    for m, col in (('g', 'AB'), ('sequence021', 'pred_seq170'), ('sequence039', 'pred_seq170'), ('row6', 'pred_seq170'),
                   ('dark18', 'pred_seq170'), ('head', 'pred_HB3u')):
        d[m] = fr(f'fold/vector/{m}')[col].reindex(ids).to_numpy(float)
    d['L'] = vec(TW / 'L-janjul.parquet').reindex(ids).to_numpy()
    d['HB3u'] = vec(TW / 'HB3u-jjrefit.parquet').reindex(ids).to_numpy()
    d['Seq170'] = vec(P15 / 'Seq170/Seq170-janjul.parquet').reindex(ids).to_numpy()
    d['Row170'] = vec(P15 / 'Row170/Row170-janjul.parquet').reindex(ids).to_numpy()
    rule = rf.reindex(ids)
    for c in ('t', 'rw', 'g', 'scope', 'fit'):
        d['rf_' + c] = rule[c].to_numpy()
    return d, pack


def month_frame(m):
    """February or June 2025 DEP rows with labels and the jjx sets; class from the sets (clock = HB3u defined)."""
    raw = pd.read_parquet(ROOT / 'data/raw' / RAWF[m], columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt',
                                                               'AOBT_3_flt', 'LOBT_flt', 'TAXITIME_SEC_mvt',
                                                               'MVT_TIME_UTC_mvt', 'RUNWAY_mvt'])
    raw = raw[raw.PHASE_mvt.eq('DEP')]
    tag = f'jjx-{m:02d}'
    L = vec(TW / f'L-{tag}.parquet')
    meta = pd.read_parquet(TW / f'L-{tag}.parquet', columns=[ID, 'month', 'utc_day', 'airport']).set_index(ID)
    assert set(meta.month.unique()) == {f'2025-{m:02d}'}
    # rows: the L set (typed departures of the month by UTC day) plus the month's untyped departures
    raw = raw.set_index(ID)
    untyped = raw[raw.AIRCRAFT_TYPE_mvt.isna() & (raw.MVT_TIME_UTC_mvt.dt.month == m)]
    ids = np.r_[meta.index.to_numpy('int64'), untyped.index.to_numpy('int64')]
    assert len(np.unique(ids)) == len(ids) and pd.Index(ids).isin(raw.index).all()
    r = raw.reindex(ids)
    d = pd.DataFrame({ID: ids, 'month': m})
    d['airport'] = r.ADEP_mvt.to_numpy()
    d['day'] = r.MVT_TIME_UTC_mvt.dt.day.to_numpy()
    d.loc[:len(meta) - 1, 'day'] = pd.to_datetime(meta.utc_day).dt.day.to_numpy()  # the sets' UTC day (typed rows)
    d['y'] = r.TAXITIME_SEC_mvt.to_numpy(float)
    d['typed'] = r.AIRCRAFT_TYPE_mvt.notna().to_numpy()
    d['aobt3'] = r.AOBT_3_flt.notna().to_numpy()
    d['plan'] = r.LOBT_flt.notna().to_numpy()
    d['t'] = r.MVT_TIME_UTC_mvt.to_numpy()
    d['rw'] = r.RUNWAY_mvt.to_numpy()
    d['L'] = L.reindex(ids).to_numpy()
    d['HB3u'] = vec(TW / f'HB3u-{tag}.parquet').reindex(ids).to_numpy()
    d['Seq170'] = vec(P15 / f'Seq170/Seq170-{tag}.parquet').reindex(ids).to_numpy()
    d['Row170'] = vec(P15 / f'Row170/Row170-{tag}.parquet').reindex(ids).to_numpy()
    for slot, (f, want, _) in REMOTE[m].items():  # absent before the remote copy: the R9 parts then do not run
        d[slot] = vec(REM / f, want).reindex(ids).to_numpy() if (REM / f).exists() else np.nan
    assert d.typed.iloc[:len(meta)].all() and not d.typed.iloc[len(meta):].any()
    d['cls'] = np.where(~d.typed, 'noac', np.where(np.isfinite(d.HB3u), 'clock', 'ordinary'))
    assert np.isfinite(d.y).all()
    return d


def cells_2026():
    """2026 rows per (month, airport, class) for the four months: clock = served by exp-046 (class-final files),
    noac = untyped departures (final_ranking.parquet), ordinary = the rest."""
    f = pd.read_parquet(FINAL_RANKING, columns=[ID, 'PHASE_mvt', 'ADEP_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt'])
    f = f[f.PHASE_mvt.eq('DEP')]
    f = f.assign(m=f.MVT_TIME_UTC_mvt.dt.month, noac=f.AIRCRAFT_TYPE_mvt.isna())
    noac = f.groupby(['m', 'ADEP_mvt']).noac.sum()
    rows = f.groupby(['m', 'ADEP_mvt']).size()
    out = {}
    for m in (1, 2, 6, 7):
        js = json.loads(Path(str(CLASS26).format(m=m)).read_text())['files'][f'2026-{m:02d}']
        for a, v in js['by_airport'].items():
            assert v['rows'] == rows[m, a], (m, a)
            nc = int(noac.get((m, a), 0))
            out[(m, a, 'clock')] = v['served']
            out[(m, a, 'noac')] = nc
            out[(m, a, 'ordinary')] = v['rows'] - v['served'] - nc
    return out


def final_exposure(fold, cells):
    """Final-mode fold weights (frames192.exposure): 2026 rows of the four months mapped to the held cells by the
    season map, over all 2026 rows, divided by the held rows of the cell (>= 2)."""
    held = fold.groupby(['month', 'airport', 'cls']).size()
    mapped = {}
    for (m, a, z), n in cells.items():
        mapped[(SEASON[m], a, z)] = mapped.get((SEASON[m], a, z), 0) + n
    total = sum(cells.values())
    return np.array([mapped.get((m, a, z), 0) / total / held[m, a, z] if held[m, a, z] >= 2 else 0.
                     for m, a, z in zip(fold.month, fold.airport, fold.cls)])


# ------------------------------------------------------------------ fits
def fit(base, M, y, w):
    A = (M - base[:, None]) * np.sqrt(w)[:, None]
    r = (y - base) * np.sqrt(w)
    beta, _, rank, _ = np.linalg.lstsq(A, r, rcond=None)
    assert rank == M.shape[1]
    return beta


def apply(base, M, beta):
    return np.maximum(60., base + (M - base[:, None]) @ beta)


def scope(d, base):
    return d.typed.to_numpy() & d.aobt3.to_numpy() & d.plan.to_numpy() & (d[base].to_numpy(float) < F_MAX)


def ready(d, base, members, slot):
    cols = [base] + [slot if k == 'mlp' else k for k in members]
    return np.isfinite(d[cols].to_numpy(float)).all(1)


def matrix(d, members, slot):
    return d[[slot if k == 'mlp' else k for k in members]].to_numpy(float)


# ------------------------------------------------------------------ G
def gstat(d, delta, W, total, rng_seed=SEED, draws=N_DRAWS, top=5):
    """G = sum_h W_h mean_h(delta) over cells (month, airport, cls) of d; W: {(m, a, z): 2026 rows}; airport-day draws."""
    q = pd.DataFrame({'m': d.month.to_numpy(), 'a': d.airport.to_numpy(), 'z': d.cls.to_numpy(),
                      't': d.day.to_numpy().astype(int), 'v': np.asarray(delta, float)})
    cells = sorted(set(zip(q.m, q.a, q.z)))
    hi = {c: i for i, c in enumerate(cells)}
    H = len(cells)
    h = np.array([hi[c] for c in zip(q.m, q.a, q.z)])
    S = np.zeros((H, 32)); n = np.zeros((H, 32))
    np.add.at(S, (h, q.t.to_numpy()), q.v.to_numpy()); np.add.at(n, (h, q.t.to_numpy()), 1.)
    Wh = np.array([W.get(c, 0) / total for c in cells])
    nh = n.sum(1); sup = nh >= 2
    G = float((Wh * np.where(sup, S.sum(1) / np.maximum(nh, 1), 0.)).sum())
    contrib = Wh * np.where(sup, S.sum(1) / np.maximum(nh, 1), 0.)
    views = {'month': {}, 'airport': {}, 'class': {}}
    for c, v in zip(cells, contrib):
        for name, key in (('month', str(c[0])), ('airport', c[1]), ('class', c[2])):
            views[name][key] = views[name].get(key, 0.) + float(v)
    # airport-days and G without the largest five
    per_day = (Wh / np.maximum(nh, 1) * sup)[:, None] * S
    ad = {}
    for i, c in enumerate(cells):
        for t in np.flatnonzero(n[i]):
            ad[(c[1], c[0], int(t))] = ad.get((c[1], c[0], int(t)), 0.) + per_day[i, t]
    tops = sorted(ad.items(), key=lambda kv: -kv[1])[:top]
    S2, n2 = S.copy(), n.copy()
    for (a, m, t), _ in tops:
        for i, c in enumerate(cells):
            if c[0] == m and c[1] == a:
                S2[i, t] = 0.; n2[i, t] = 0.
    wo = float((Wh * np.where(sup, S2.sum(1) / np.maximum(n2.sum(1), 1), 0.)).sum())
    # draws: days with replacement within each airport x month stratum, every class of a drawn day together
    rng = np.random.default_rng(rng_seed)
    Gb = np.zeros(draws)
    strata = sorted(set((c[0], c[1]) for c in cells))
    Sb = np.zeros((draws, H)); nb = np.zeros((draws, H))
    for (m, a) in strata:
        hs = [i for i, c in enumerate(cells) if c[0] == m and c[1] == a]
        days = np.flatnonzero(n[hs].sum(0) > 0)
        K = len(days)
        idx = rng.integers(0, K, size=(draws, K))
        C = np.zeros((draws, K)); np.add.at(C, (np.arange(draws)[:, None], idx), 1.)
        Sb[:, hs] = C @ S[np.ix_(hs, days)].T
        nb[:, hs] = C @ n[np.ix_(hs, days)].T
    Gb = (Wh[None, :] * sup[None, :] * Sb / np.maximum(nb, 1)).sum(1)
    lo, hi_ = np.percentile(Gb, [2.5, 97.5])
    return {'G': G, 'lo95': float(lo), 'hi95': float(hi_), 'p_G_le_0': float((Gb <= 0).mean()), 'wo5': wo,
            'top_days': [{'airport': k[0], 'month': k[1], 'day': k[2], 'contribution': float(v)} for k, v in tops],
            'views': {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in views.items()},
            'changed_rows': int((np.asarray(delta) != 0).sum())}


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    if isinstance(o, float):
        return round(o, 3)
    return o


# ------------------------------------------------------------------ reads
def r9_janjul(fold, form, W, total, GS):
    """Reduced form on January/July: weights from the other month, R9 = mix12 over mlp008 in the mlp slot."""
    base, members = FORMS[form]
    on = scope(fold, base) & fold.cls.eq('ordinary').to_numpy() & ready(fold, base, members, 'mlp008') \
        & ready(fold, base, members, 'mix12') & (fold.w.to_numpy() > 0)
    y, w, b = fold.y.to_numpy(float), fold.w.to_numpy(float), fold[base].to_numpy(float)
    preds, coefs = {}, {}
    for slot in ('mlp008', 'mix12'):
        M = matrix(fold, members, slot)
        p = np.full(len(fold), np.nan)
        coefs[slot] = {}
        for test, train in ((1, 7), (7, 1)):
            tr = on & fold.month.eq(train).to_numpy(); te = on & fold.month.eq(test).to_numpy()
            beta = fit(b[tr], M[tr], y[tr], w[tr]); coefs[slot][f'serve_{test}'] = dict(zip(members, beta.round(4).tolist()))
            p[te] = apply(b[te], M[te], beta)
        preds[slot] = p
    delta = np.where(on, (y - preds['mlp008']) ** 2 - (y - preds['mix12']) ** 2, 0.)
    out = gstat(fold, delta, W, total)
    if GS is not None:  # same G as the registered scorer: base and candidate equal off the changed rows
        x = np.where(on, preds['mlp008'], 1000.); z = np.where(on, preds['mix12'], 1000.)
        s = GS.score(fold[ID].to_numpy(), x, z, y=y, resample=False)
        out['gscore_G'] = s['G']
        assert abs(s['G'] - out['G']) < 1e-6, (s['G'], out['G'])
    out.update(rows=int(on.sum()), coefficients=coefs,
               member_mse={s: float(np.mean((y[on] - fold[s].to_numpy(float)[on]) ** 2)) for s in ('mlp008', 'mix12')},
               rows_in_pack_scope=int((on & fold.scope_pack.to_numpy()).sum()))
    return out


def r9_month(fold, wfit, mo, form, W, total, season):
    base, members = FORMS[form]
    ftr = scope(fold, base) & fold.cls.eq('ordinary').to_numpy() & ready(fold, base, members, 'mlp008') \
        & ready(fold, base, members, 'mix12') & (wfit > 0)
    on = scope(mo, base) & mo.cls.eq('ordinary').to_numpy() & ready(mo, base, members, 'mlp008') \
        & ready(mo, base, members, 'mix12')
    y, b = mo.y.to_numpy(float), mo[base].to_numpy(float)
    yf, bf = fold.y.to_numpy(float), fold[base].to_numpy(float)
    preds, coefs = {}, {}
    for slot in ('mlp008', 'mix12'):
        Mf, M = matrix(fold, members, slot), matrix(mo, members, slot)
        p = np.full(len(mo), np.nan); coefs[slot] = {}
        for m in sorted(mo.month.unique()):
            tr = ftr & (fold.month.eq(SEASON[m]).to_numpy() if season else True)
            te = on & mo.month.eq(m).to_numpy()
            beta = fit(bf[tr], Mf[tr], yf[tr], wfit[tr]); coefs[slot][f'serve_{m}'] = dict(zip(members, beta.round(4).tolist()))
            p[te] = apply(b[te], M[te], beta)
        preds[slot] = p
    delta = np.where(on, (y - preds['mlp008']) ** 2 - (y - preds['mix12']) ** 2, 0.)
    out = gstat(mo, delta, W, total)
    out.update(rows=int(on.sum()), coefficients=coefs,
               member_mse={s: float(np.mean((y[on] - mo[s].to_numpy(float)[on]) ** 2)) for s in ('mlp008', 'mix12')})
    return out, preds, on


def head_weight(d, base, W, total, wcol=None):
    """Clock rows with a head value: G of head(a) over head(0.5); head(a) = max(60, (1 - a) base + a HB3u)."""
    on = d.cls.eq('clock').to_numpy() & np.isfinite(d[base].to_numpy(float)) & np.isfinite(d.HB3u.to_numpy(float))
    y, b, hb = d.y.to_numpy(float), d[base].to_numpy(float), d.HB3u.to_numpy(float)
    h = lambda a: np.maximum(60., (1 - a) * b + a * hb)
    ref = h(0.5); out = {}
    for a in ALPHAS:
        delta = np.where(on, (y - ref) ** 2 - (y - h(a)) ** 2, 0.)
        g = gstat(d, delta, W, total, draws=400)
        out[str(a)] = {'G': g['G'], 'lo95': g['lo95'], 'hi95': g['hi95'], 'month': g['views']['month']}
    best = max(out, key=lambda k: out[k]['G'])
    return {'rows': int(on.sum()), 'G_over_0.5': out, 'best': best}


def eddf_dark(fold, wfit, mo):
    """EDDF February 2025 clock rows: ordinary path (C3 ordinary weights, pooled) against head path (head 0.5 plus C3
    clock-row weights on the head base, pooled). MSE in s^2 per row."""
    base, members = FORMS['C3']
    yf, wf = fold.y.to_numpy(float), wfit
    res = {}
    for cls_fit in ('ordinary', 'clock'):
        tr = scope(fold, base) & fold.cls.eq(cls_fit).to_numpy() & ready(fold, base, members, 'mix12') & (wf > 0)
        if cls_fit == 'clock':
            tr &= np.isfinite(fold.HB3u.to_numpy(float))
            bf = np.maximum(60., 0.5 * fold.L.to_numpy(float) + 0.5 * fold.HB3u.to_numpy(float))
        else:
            bf = fold.L.to_numpy(float)
        res[cls_fit] = fit(bf[tr], matrix(fold, members, 'mix12')[tr], yf[tr], wf[tr])
    sel = mo.airport.eq('EDDF').to_numpy() & mo.month.eq(2).to_numpy() & mo.cls.eq('clock').to_numpy() \
        & scope(mo, base) & ready(mo, base, members, 'mix12')
    y = mo.y.to_numpy(float)[sel]; M = matrix(mo, members, 'mix12')[sel]
    L = mo.L.to_numpy(float)[sel]; hb = mo.HB3u.to_numpy(float)[sel]
    head = np.maximum(60., 0.5 * L + 0.5 * hb)
    paths = {'ordinary_path': apply(L, M, res['ordinary']), 'head_path': apply(head, M, res['clock']),
             'route_only': np.maximum(60., L), 'head_only': head}
    mse = {k: float(np.mean((y - v) ** 2)) for k, v in paths.items()}
    return {'rows': int(sel.sum()), 'mse': mse, 'gap_ordinary_minus_head': mse['ordinary_path'] - mse['head_path'],
            'weights': {k: dict(zip(members, v.round(4).tolist())) for k, v in res.items()},
            'caveat': 'members and L on these rows saw the lit ADS-B inputs; the ordinary path is optimistic'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--janjul-only', action='store_true', help='January/July reads only (forms, calibration, head)')
    a = p.parse_args()
    sys.path.insert(0, str(ROOT))
    from prc import gscore as GS
    fold, pack = fold_frame(a.export.resolve())
    if a.janjul_only:
        frozen = GS.load_cells()
        W_jj = {(int(r.month), r.airport, r.cls): int(r.N) for r in frozen.itertuples() if r.month in (1, 7)}
        FORMS.update(CALIB)
        rec = {'inputs': {'export_pack_sha256': sha(a.export / 'release/ZABD-PACK.json'), 'rows_fold': len(fold)},
               'code': {Path(__file__).name: sha(__file__)},
               'janjul': {f: r9_janjul(fold, f, W_jj, GS.N_RANKING, GS) for f in FORMS},
               'forms': {f: {'base': b, 'members': m} for f, (b, m) in FORMS.items()}}
        rec['head_weight'] = {'janjul_L': head_weight(fold, 'L', W_jj, GS.N_RANKING),
                              'janjul_R4mk': head_weight(fold, 'R4mk', W_jj, GS.N_RANKING)}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rnd(rec), indent=1) + '\n')
        for f, v in rec['janjul'].items():
            print(f'{f:22s} {v["G"]:7.2f} [{v["lo95"]:7.2f}, {v["hi95"]:7.2f}] Jan {v["views"]["month"]["1"]:7.2f} '
                  f'Jul {v["views"]["month"]["7"]:7.2f}')
        for k, v in rec['head_weight'].items():
            print('head', k, v['best'], {a_: round(x['G'], 1) for a_, x in v['G_over_0.5'].items()})
        return
    mo = pd.concat([month_frame(2), month_frame(6)], ignore_index=True)
    cells = cells_2026()
    W_fm = {k: v for k, v in cells.items() if k[0] in (2, 6)}
    total_fm = sum(W_fm.values())
    frozen = GS.load_cells()
    W_jj = {(int(r.month), r.airport, r.cls): int(r.N) for r in frozen.itertuples() if r.month in (1, 7)}
    wfin = final_exposure(fold, cells)
    rec = {'inputs': {'export_pack_sha256': sha(a.export / 'release/ZABD-PACK.json'),
                      'remote_copies': {f: want for m in REMOTE.values() for f, want, _ in m.values()},
                      'rows': {'fold': len(fold), 'feb': int(mo.month.eq(2).sum()), 'jun': int(mo.month.eq(6).sum())},
                      'classes_2025_febjun': mo.groupby(['month', 'cls']).size().rename(lambda x: str(x)).to_dict(),
                      'cells_2026_febjun_rows': total_fm,
                      'fit_weights_final_sum': float(wfin.sum())},
           'code': {Path(__file__).name: sha(__file__)}}
    rec['inputs']['classes_2025_febjun'] = {f'{k[0]}-{k[1]}': int(v) for k, v in mo.groupby(['month', 'cls']).size().items()}
    # 1. R9 in each reduced form on January/July (other-month weights), against the registered scorer
    rec['janjul'] = {f: r9_janjul(fold, f, W_jj, GS.N_RANKING, GS) for f in FORMS}
    # 2. R9 on February/June: pooled January+July weights (final-mode exposure) and season-month weights
    rec['febjun'] = {}
    for f in ('C3', 'C1'):
        for season in (False, True):
            out, _, _ = r9_month(fold, wfin, mo, f, W_fm, total_fm, season)
            rec['febjun'][f + ('-season' if season else '-pooled')] = out
    # 3. head weight on clock rows (base L on all months; base R4mk on January/July as a check of the proxy)
    rec['head_weight'] = {'febjun_L': head_weight(mo, 'L', W_fm, total_fm),
                          'janjul_L': head_weight(fold, 'L', W_jj, GS.N_RANKING),
                          'janjul_R4mk': head_weight(fold, 'R4mk', W_jj, GS.N_RANKING)}
    # 4. EDDF February 2025 clock rows through the ordinary path
    rec['eddf_feb_dark'] = eddf_dark(fold, wfin, mo)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rnd(rec), indent=1) + '\n')
    for f, v in rec['janjul'].items():
        print('janjul', f, round(v['G'], 2), [round(v['lo95'], 2), round(v['hi95'], 2)], v['views']['month'], v['rows'])
    for f, v in rec['febjun'].items():
        print('febjun', f, round(v['G'], 2), [round(v['lo95'], 2), round(v['hi95'], 2)], v['views']['month'], v['rows'])
    for k, v in rec['head_weight'].items():
        print('head', k, v['best'], {a_: round(x['G'], 1) for a_, x in v['G_over_0.5'].items()})
    print('eddf', json.dumps(rnd(rec['eddf_feb_dark']['mse'])), rec['eddf_feb_dark']['rows'])


if __name__ == '__main__':
    main()
