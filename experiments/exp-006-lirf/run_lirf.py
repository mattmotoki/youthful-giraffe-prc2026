"""exp-006 lirf: LIRF-only models on the cached exp-001 v3 feature frames.

  python run_lirf.py --name a0 --folds july,december,january

Each run appends climb.log lines, writes oof/<name>-<fold>.parquet and
result-<name>.json. Predictions for the other nine airports always come from
the retro-v1 c6-noac out-of-fold file, so the whole-fold number is the honest
"what happens to the score if only LIRF changes".
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/exp-001-retro-v1'))
import lightgbm as lgb
import prc.data as D
from features import CAT_FEATURES
from prc.lirf_features import LIRF_CAT_FEATURES
from run import CatEncoder

ALL_CATS = CAT_FEATURES + LIRF_CAT_FEATURES

FEAT = ROOT/'artifacts/exp-006-lirf/feat_v3'
SIG = ROOT/'artifacts/exp-006-lirf/lirf_sig'
PHI_ROOT = ROOT/'artifacts/exp-006-lirf'
#: every phi construction emits the same columns, so `--phi-mode` changes the
#: construction and nothing else (gpt-5.6-sol round 1, blocking finding 1)
PHI_COLS = ('phi', 'phi_support', 'taxi_hat', 'gap', 'block_hat',
            'phi_airport', 'phi_airport_operator',
            'phi_support_airport', 'phi_support_airport_operator',
            'taxi_hat_airport', 'taxi_hat_airport_operator')
BASE_OOF = ROOT/'experiments/exp-001-retro-v1/oof'
#: the board top on 2026-09-05, for the "fraction of the remaining gap closed"
#: column COMMON.md asks climb.log to carry
BOARD_TOP = 277.3
TAIL_CUT = 3600.0
DROP = ('y','takeoff_epoch','month','MVT_ID_mvt')


SIG_COLS = frozenset()   # filled on first use from the signal cache's columns


def load_months(months, airport='LIRF', signals=False, phi_fold=None, gate_signals=''):
    xs = []
    for m in months:
        X = pd.read_parquet(FEAT/f'{m}.parquet')
        if phi_fold:
            fold_name, mode = phi_fold.split('!')
            P = pd.read_parquet(PHI_ROOT/f'phi_{mode}'/f'{fold_name}-{m}.parquet')
            missing = [c for c in PHI_COLS if c not in P.columns]
            assert not missing, f'phi cache {mode} is missing {missing}'
            P = P[['MVT_ID_mvt'] + list(PHI_COLS)]
            assert len(P) == len(X) and (P.MVT_ID_mvt.to_numpy() == X.MVT_ID_mvt.to_numpy()).all()
            X = pd.concat([X.reset_index(drop=True),
                           P.drop(columns=['MVT_ID_mvt']).reset_index(drop=True)], axis=1)
        if signals:
            S = pd.read_parquet(SIG/f'{m}.parquet')
            assert len(S) == len(X) and (S.MVT_ID_mvt.to_numpy() == X.MVT_ID_mvt.to_numpy()).all()
            S = S.drop(columns=[c for c in S.columns if c == 'MVT_ID_mvt' or c in X.columns])
            globals()['SIG_COLS'] = frozenset(S.columns)
            X = pd.concat([X.reset_index(drop=True), S.reset_index(drop=True)], axis=1)
        if signals and gate_signals:
            # prc.lirf_features is a Rome-motivated block: applied to all ten
            # airports it costs 3.9 s of the july whole core while helping the LIRF
            # core (RESULTS finding 8). Gating it means every other airport sees
            # NaN there, so the model can only use it inside a LIRF branch.
            sig_cols = [c for c in X.columns if c in SIG_COLS]
            off = ~X.airport.eq(gate_signals).to_numpy()
            for c in sig_cols:
                if X[c].dtype == object:
                    X.loc[off, c] = None
                else:
                    X.loc[off, c] = np.nan
        if airport:
            X = X[X.airport.eq(airport)]
        xs.append(X)
    return pd.concat(xs, ignore_index=True)


def feature_cols(X, drop=()):
    return [c for c in X.columns if c not in DROP and c not in drop]


def train_one(params, Xtr, ytr, base_tr, Xva, yva, base_va, max_rounds, es_rounds, es_frac,
              tr_time, cats):
    """Two passes: early stop on the last es_frac of training time, then refit all."""
    cut = np.quantile(tr_time, 1 - es_frac)
    tr, es = tr_time < cut, tr_time >= cut
    enc = CatEncoder([c for c in cats if c in Xtr.columns]).fit(Xtr)
    T = enc.transform(Xtr)
    # the target passed in is already the residual r = base - y, so no init_score
    # is used: squared error on r equals squared error on y exactly.
    d1 = lgb.Dataset(T[tr], label=ytr[tr],
                     categorical_feature=[c for c in cats if c in T.columns])
    d2 = lgb.Dataset(T[es], label=ytr[es],
                     reference=d1, categorical_feature=[c for c in cats if c in T.columns])
    m1 = lgb.train(params, d1, num_boost_round=max_rounds, valid_sets=[d2],
                   callbacks=[lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(0)])
    n = max(20, int(m1.best_iteration / (1 - es_frac)))
    dfull = lgb.Dataset(T, label=ytr,
                        categorical_feature=[c for c in cats if c in T.columns])
    m2 = lgb.train(params, dfull, num_boost_round=n)
    Tva = enc.transform(Xva)
    p = m2.predict(Tva)
    if base_va is not None:
        p = base_va - p          # y = (MVT - AOBT) - r_hat
    return m2, enc, p, n


def base_and_target(X, y, fallback):
    """Residual parameterisation: base = MVT - AOBT, target r = base - y (fitted directly)."""
    b = X['d_takeoff_aobt'].to_numpy('float64').copy()
    miss = ~np.isfinite(b)
    if miss.any():
        b[miss] = fallback
    if y is None:
        return b, None
    return b, b - y


def decompose(y, p, X):
    y = np.asarray(y, float); p = np.asarray(p, float)
    se = (y - p) ** 2
    nonm = X['nm_missing'].to_numpy().astype(bool)
    noac = X['noac'].to_numpy().astype(bool) if 'noac' in X.columns else np.zeros(len(X), bool)
    tail = y > TAIL_CUT
    core = (~nonm) & (~noac) & (~tail)
    out = dict(n=int(len(y)), rmse=float(np.sqrt(se.mean())),
               rmse_le3600=float(np.sqrt(se[~tail].mean())) if (~tail).any() else np.nan,
               rmse_gt3600=float(np.sqrt(se[tail].mean())) if tail.any() else np.nan,
               n_gt3600=int(tail.sum()),
               rmse_nonm=float(np.sqrt(se[nonm].mean())) if nonm.any() else np.nan,
               n_nonm=int(nonm.sum()),
               rmse_noac=float(np.sqrt(se[noac].mean())) if noac.any() else np.nan,
               n_noac=int(noac.sum()),
               rmse_core_input=float(np.sqrt(se[(~nonm) & (~noac)].mean())),
               n_core_input=int(((~nonm) & (~noac)).sum()),
               sse_share_noac=float(se[noac].sum()/se.sum()) if noac.any() else 0.0,
               sse_share_gt3600=float(se[tail].sum()/se.sum()) if tail.any() else 0.0,
               bias=float((p - y).mean()))
    for k in (1, 2, 5, 20):
        idx = np.argsort(-se)[k:]
        out[f'rmse_drop{k}'] = float(np.sqrt(se[idx].mean()))
    return out


def whole_fold(fold, ids, pred):
    """Total RMSE of the fold when only the LIRF rows are replaced."""
    b = pd.read_parquet(BASE_OOF/f'c6-noac-{fold}.parquet')
    sub = pd.Series(pred, index=pd.Index(ids, name='MVT_ID_mvt'))
    before = D.rmse(b.y_true, b.y_pred)
    p = b.set_index('MVT_ID_mvt').y_pred.copy()
    hit = p.index.intersection(sub.index)
    assert len(hit) == len(sub), (len(hit), len(sub))
    p.loc[hit] = sub.loc[hit]
    after = D.rmse(b.set_index('MVT_ID_mvt').y_true, p)
    return before, after


def run_fold(cfg, fold_name):
    fold = D.FOLDS[fold_name]
    t0 = time.time()
    pf = f"{fold_name}!{cfg['phi_mode']}" if cfg['phi_mode'] else None
    Xtr = load_months(fold.train_months, cfg['airport'], cfg['lirf_features'], pf,
                      cfg['gate_signals'])
    Xva = load_months([fold.valid_month], cfg['airport'], cfg['lirf_features'], pf,
                      cfg['gate_signals'])
    ytr = Xtr.y.to_numpy('float64'); yva = Xva.y.to_numpy('float64')
    keep = np.isfinite(ytr)
    if cfg['drop_retired']:
        # LIRF stands 501-507 carry 10,015 rows in 2025 and stop after October;
        # the ranking months have none. Dropping them asks whether a stand group
        # that cannot recur is teaching the model something it should not.
        sn = pd.to_numeric(Xtr.airport_stand.astype('string').str.extract(r'\|(\d+)')[0],
                           errors='coerce')
        ret = (Xtr.airport.eq('LIRF') & sn.between(501, 507).fillna(False)).to_numpy()
        print(f'[retired] dropping {int((ret & keep).sum())} LIRF 501-507 training rows', flush=True)
        keep = keep & ~ret
    Xtr, ytr = Xtr[keep].reset_index(drop=True), ytr[keep]
    fallback = float(np.median(ytr))
    btr, ttr = base_and_target(Xtr, ytr, fallback)
    bva, _ = base_and_target(Xva, None, fallback)
    cols = feature_cols(Xtr, cfg['drop_cols'])
    cats = [c for c in ALL_CATS if c in cols]
    params = dict(objective='l2', metric='l2', learning_rate=cfg['lr'],
                  num_leaves=cfg['leaves'], min_data_in_leaf=cfg['min_data'],
                  lambda_l2=cfg['l2'], feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, num_threads=cfg['threads'], seed=cfg['seed'],
                  linear_tree=cfg['linear_tree'], linear_lambda=cfg['linear_lambda'],
                  max_cat_threshold=32, cat_smooth=10.0, min_data_per_group=20)
    model, enc, p, rounds = train_one(params, Xtr[cols], ttr, btr, Xva[cols], yva, bva,
                                      cfg['max_rounds'], cfg['es_rounds'], cfg['es_frac'],
                                      Xtr.takeoff_epoch.to_numpy(), cats)
    if cfg['noac_head']:
        p = apply_noac_head(cfg, Xtr, ytr, cols, cats, Xva, p)
    if cfg['clip_floor'] is not None:
        p = np.maximum(p, cfg['clip_floor'])
    res = decompose(yva, p, Xva)
    if not cfg['airport']:
        m = Xva.airport.eq('LIRF').to_numpy()
        res['lirf'] = decompose(yva[m], p[m], Xva[m])
        res['other9'] = float(np.sqrt(np.mean((yva[~m] - p[~m]) ** 2)))
    before, after = whole_fold(fold_name, Xva.MVT_ID_mvt.to_numpy(), p)
    base = pd.read_parquet(BASE_OOF/f'c6-noac-{fold_name}.parquet')
    base = base[base.airport.eq('LIRF')].set_index('MVT_ID_mvt')
    res.update(fold=fold_name, rounds=int(rounds), secs=round(time.time()-t0, 1),
               baseline_lirf=float(D.rmse(base.y_true, base.y_pred)),
               fold_rmse_before=float(before), fold_rmse_after=float(after))
    oof = pd.DataFrame(dict(MVT_ID_mvt=Xva.MVT_ID_mvt.to_numpy(), fold=fold_name,
                            y_true=yva, y_pred=p, airport=Xva.airport.to_numpy()))
    (HERE/'oof').mkdir(exist_ok=True)
    oof.to_parquet(HERE/'oof'/f"{cfg['name']}-{fold_name}.parquet", index=False)
    imp = pd.DataFrame(dict(feature=model.feature_name(),
                            gain=model.feature_importance('gain'))).sort_values('gain', ascending=False)
    imp.to_csv(HERE/'work'/f"importance-{cfg['name']}-{fold_name}.csv", index=False)
    return res


#: A noac row has every `_flt` column null, so anchor deltas, phi and the gap are
#: all NaN there and only waste splits. On the july fold the head has 530 training
#: rows against ~190 columns, and the wide set costs july noac RMSE. This is the
#: subset that survives on a stripped movement record.
NOAC_COLS = (
    'd_takeoff_sched', 'noac', 'flight_len', 'nm_missing', 'aobt_missing',
    'noac_day_airport', 'noac_share_day', 'day_noac_n', 'day_noac_frac',
    'day_nonm_frac', 'day_n_dep', 'hour_noac_n', 'noac_prev_gap', 'noac_next_gap',
    'noac_neigh_gap', 'al_day_med', 'al_day_p90',
    'stand_in_block_gap', 'stand_in_taxiin', 'stand_in_turnaround',
    'stand_in_land_gap', 'stand_in_noac', 'stand_next_block_gap',
    'stand_prev_dep_gap', 'stand_next_dep_gap',
    'stand_prev_arr_block_gap', 'stand_prev_arr_land_gap', 'stand_prev_arr_taxiin',
    'stand_next_arr_land_gap', 'stand_group', 'stand_num', 'stand_retired',
    'ades_is_adep', 'flight_prefix', 'airport', 'airport_stand', 'airport_runway',
    'destination', 'flight_rule_mvt', 'local_hour', 'local_dow', 'minute_of_day',
    'doy_sin', 'doy_cos', 'prev_apt_gap', 'prev_rwy_gap',
    'arr_taxiin_mean_10m_before', 'arr_taxiin_mean_30m_before',
    'arr_taxiin_mean_30m_after', 'arr_landings_60m_after',
    'dep_takeoffs_60m_before', 'arr_landings_60m_before',
)


def apply_noac_head(cfg, Xtr, ytr, cols, cats, Xva, p):
    """Separate head for the noac rows, anchored on take-off minus schedule."""
    if cfg['noac_narrow']:
        cols = [c for c in cols if c in NOAC_COLS]
        cats = [c for c in cats if c in cols]
    m = Xtr['noac'].to_numpy().astype(bool)
    if m.sum() < cfg['noac_min_rows']:
        print(f'[noac] only {m.sum()} training rows, head skipped', flush=True)
        return p
    b_tr = Xtr.loc[m, 'd_takeoff_sched'].to_numpy('float64')
    t_tr = b_tr - ytr[m]
    params = dict(objective='l2', metric='l2', learning_rate=cfg['noac_lr'],
                  num_leaves=cfg['noac_leaves'], min_data_in_leaf=cfg['noac_min_data'],
                  feature_fraction=0.8, verbose=-1, num_threads=cfg['threads'],
                  seed=cfg['seed'], linear_tree=False, max_cat_threshold=16,
                  cat_smooth=5.0, min_data_per_group=5)
    enc = CatEncoder([c for c in cats]).fit(Xtr.loc[m, cols])
    Ttr = enc.transform(Xtr.loc[m, cols])
    mv = Xva['noac'].to_numpy().astype(bool)
    if mv.any():
        Tva = enc.transform(Xva.loc[mv, cols])
        acc = np.zeros(int(mv.sum()))
        for s_ in range(cfg['noac_seeds']):
            pr = dict(params, seed=cfg['seed'] + s_, bagging_seed=cfg['seed'] + 100 + s_,
                      feature_fraction_seed=cfg['seed'] + 200 + s_,
                      bagging_fraction=0.9, bagging_freq=1)
            mod = lgb.train(pr, lgb.Dataset(Ttr, label=t_tr, categorical_feature=[c for c in cats]),
                            num_boost_round=cfg['noac_rounds'])
            acc += mod.predict(Tva)
        bv = Xva.loc[mv, 'd_takeoff_sched'].to_numpy('float64')
        p = p.copy()
        p[mv] = bv - acc / cfg['noac_seeds']
    print(f"[noac] head on {m.sum()} training rows, {cfg['noac_rounds']} rounds", flush=True)
    return p


def _gap_closed(r):
    """Fraction of the distance from the baseline fold RMSE to the board top that
    this line closes. NA when the baseline is already past the board top, which is
    true of the december fold — the denominator is not meaningful there."""
    denom = r['fold_rmse_before'] - BOARD_TOP
    if denom <= 1.0:
        return 'NA'
    return f"{(r['fold_rmse_before'] - r['fold_rmse_after']) / denom:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='a0')
    ap.add_argument('--folds', default='july,december,january')
    ap.add_argument('--airport', default='LIRF')
    ap.add_argument('--lirf-features', action='store_true')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--leaves', type=int, default=127)
    ap.add_argument('--min-data', type=int, default=40)
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--linear-tree', action='store_true')
    ap.add_argument('--linear-lambda', type=float, default=1.0)
    ap.add_argument('--max-rounds', type=int, default=4000)
    ap.add_argument('--es-rounds', type=int, default=100)
    ap.add_argument('--es-frac', type=float, default=0.15)
    ap.add_argument('--noac-head', action='store_true')
    ap.add_argument('--noac-rounds', type=int, default=40)
    ap.add_argument('--noac-leaves', type=int, default=63)
    ap.add_argument('--noac-min-data', type=int, default=15)
    ap.add_argument('--noac-lr', type=float, default=0.05)
    ap.add_argument('--noac-min-rows', type=int, default=50)
    ap.add_argument('--noac-seeds', type=int, default=5)
    ap.add_argument('--noac-narrow', action='store_true',
                    help='restrict the noac head to columns that survive a stripped record')
    ap.add_argument('--clip-floor', type=float, default=60.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--phi-mode', default='',
                    choices=['', 'full', 'cv', 'cvfull', 'lomo', 'loo'],
                    help='which phi construction to read from artifacts/exp-006-lirf/phi_<mode>')
    ap.add_argument('--drop-retired', action='store_true')
    ap.add_argument('--gate-signals', default='',
                    help='airport code: null the lirf_signals columns everywhere else')
    ap.add_argument('--drop-cols', default='')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    cfg = vars(a)
    cfg['drop_cols'] = tuple(c for c in a.drop_cols.split(',') if c)
    out = {}
    for f in a.folds.split(','):
        r = run_fold(cfg, f)
        out[f] = r
        print(json.dumps(r), flush=True)
        with open(HERE/'climb.log', 'a') as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{a.name}\t{f}\t"
                     f"lirf={r.get('lirf', r)['rmse']:.1f}\tbase_lirf={r['baseline_lirf']:.1f}\t"
                     f"fold={r['fold_rmse_after']:.2f} (from {r['fold_rmse_before']:.2f})\t"
                     f"core={r.get('lirf', r)['rmse_core_input']:.1f}\tnoac={r.get('lirf', r)['rmse_noac']:.1f}\t"
                     f"gt3600={r.get('lirf', r)['rmse_gt3600']:.1f}\t"
                     f"gap_closed={_gap_closed(r)}\t"
                     f"rounds={r['rounds']}\t{a.note}\t"
                     f"{' '.join(sys.argv[1:])}\n")
    with open(HERE/f'result-{a.name}.json', 'w') as fh:
        json.dump(dict(cfg=cfg, folds=out), fh, indent=1)


if __name__ == '__main__':
    main()
