"""exp-194 review repair (Astra review of dh-final1, VERDICT FAIL 13:50 UTC 10-09; desk 13:55 UTC): R4mk values on one
held month that hold no label of the other held month.

The saved held level 2 of R4mks trains each owner on the jjx months and the other held month (stacker.splits_for,
R4_OWNERS: q07 scores July and trains on the jjx months and January; q01 the reverse). A read that scores January and
fits V's stack weights and the pair w on July rows reads January labels through q07's values on July. This script
fits the level 2 again on the seven jjx months only (consumer jjrefit: no row of January or July or of their buffer
days), with the fixed rounds of the saved owner whose CV months leave the scored month out:
  x01  rounds of q01 (its CV months: the jjx months and July; no January label) -> values on the July rows; used when
       January is the scored month
  x07  rounds of q07 (its CV months: the jjx months and January; no July label) -> values on the January rows; used
       when July is the scored month
Features, parameters, seed 601, loss mask (eligible) and the two learners (LightGBM + CatBoost, pred_Sb the mean) are
stacker.py's for inputs r4mks; only the training months and the fixed rounds differ. Then on the month it serves:
  lit           every row, jjrefit member vectors (as fold_serve.py --split jjrefit and r4mk_dark.py)
  dark_litL     EDDF and LIRF rows forced dark by r4mk_dark.py's rule (dark_values JSON, SHARE_MIN), L and Lp lit
  dark_darkL    the same with L and Lp dark (--member-dark, l_dark.py outputs)
No label is read in serving. Write-once <out>/R4mksX-<owner>.{model.txt,cat.cbm,parquet,json}.

    PRC_SOURCE_ROOT=<snapshot> PRC_DATA_DIR=<raw> python r4mk_iso.py --owner x01 --work <exp-148 work> \
        --dark-values <dark_values.json> --member-dark L=<parquet> --member-dark Lp=<parquet> --out <dir> \
        [--threads 2] [--smoke <training rows>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4mk_dark as RD               # noqa: E402  (stacker as RD.ST, the dark rule and predict)

ST = RD.ST
P, WP = ST.P, ST.WP
ID = ST.ID
STAGE, INPUTS = RD.STAGE, RD.INPUTS
#: owner: (saved owner whose rounds it takes, month it serves, month it keeps out)
OWNERS = {'x01': ('q01', 7, 1), 'x07': ('q07', 1, 7)}
SMOKE_ROUNDS = 20


def dark_frames(mm, f_m, p_m, dark_value, kept, repl_feat, proxy_cols, over):
    """r4mk_dark.py's forced-dark inputs of the EDDF and LIRF rows of one month (the same rule, as a function)."""
    dark_f = f_m.ctx_airport.astype(str).isin(RD.AIRPORTS).to_numpy()
    f_d, p_d, m_d = f_m.copy(), p_m.copy(), mm.copy()
    for c in repl_feat:
        v = dark_value[c][0]
        if c == 'ctx_cls':
            f_d[c] = f_d[c].astype(object)
        f_d.loc[dark_f, c] = np.nan if v is None else v
    dark_ids = set(f_m.loc[dark_f, ID].astype('int64'))
    dark_sorted = np.array(sorted(dark_ids), dtype='int64')
    own = p_m.drop_duplicates(ID).set_index(ID).reindex(dark_sorted)
    new = pd.DataFrame({ID: dark_sorted,
                        **{c: (own[c].to_numpy() if c in kept else
                               (np.nan if dark_value[c][0] is None else dark_value[c][0])) for c in proxy_cols}})
    p_d = pd.concat([p_d[~p_d[ID].isin(dark_ids)], new], ignore_index=True)
    dark_m = m_d[ID].isin(dark_ids).to_numpy()
    for st, (f, _) in over.items():
        have = dark_m & np.isfinite(m_d[st].to_numpy(float))
        v = f.reindex(m_d.loc[have, ID].to_numpy()).to_numpy(float)
        assert np.isfinite(v).all(), f'{st}: dark values missing on {int((~np.isfinite(v)).sum())} rows'
        m_d.loc[have, st] = v
    return ST.frame(m_d, f_d, p_d), dark_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--owner', choices=sorted(OWNERS), required=True)
    ap.add_argument('--work', type=Path, required=True, help='exp-148 work root (stack/, inputs/, out, out-jj, kaggle)')
    ap.add_argument('--dark-values', type=Path, required=True)
    ap.add_argument('--member-dark', action='append', default=[], help='L=<parquet> and Lp=<parquet>')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--smoke', type=int, help=f'path test: this many training rows and {SMOKE_ROUNDS} rounds')
    a = ap.parse_args()
    import catboost as cb
    import lightgbm as lgb
    qown, serve_month, out_month = OWNERS[a.owner]
    stem = a.out / f'{STAGE}X-{a.owner}'
    paths = {k: Path(f'{stem}.{k}') for k in ('model.txt', 'cat.cbm', 'parquet', 'json')}
    for q in paths.values():
        if q.exists():
            raise SystemExit(f'{q} exists; write-once')
    t0 = time.monotonic()
    W = a.work
    ST.MEMBERS.update(ST.EXTRA_MEMBERS.get(INPUTS, {}))
    ST.MEMBER_SOURCE.update(ST.SOURCE_OF.get(INPUTS, {}))
    ST.ZEROED.update(ST.ZERO_OF.get(INPUTS, ()))
    feats = pd.read_parquet(W / 'inputs/features-r4c.parquet')
    proxy = pd.read_parquet(W / 'inputs/proxy-features-r4c.parquet')
    proxy[ID] = proxy[ID].astype('int64')
    fx = ST.feature_list(feats, proxy, INPUTS)
    fl_sha = hashlib.sha256(json.dumps(fx).encode()).hexdigest()

    # the saved owner's record: rounds, and the months its CV read (the scored month must not be one)
    ref_path = W / 'stack' / f'{STAGE}-{P.consumer(qown).name}.json'
    ref = json.loads(ref_path.read_text())
    assert (ref['owner'], ref['inputs'], ref['feature_list_sha256']) == (qown, INPUTS, fl_sha), \
        (ref['owner'], ref['inputs'], ref['feature_list_sha256'])
    ref_months = sorted(int(m) for m in ref['cv'])
    assert ref.get('rounds_from') is None, 'the saved rounds come from another fit'
    assert out_month not in ref_months and str(out_month) not in ref['train_rows_by_month'], ref_months
    assert {k: v for k, v in ref['params'].items() if k != 'num_threads'} == ST.PARAMS, ref['params']
    assert {k: v for k, v in ref['cat_params'].items() if k != 'thread_count'} == ST.CAT_PARAMS, ref['cat_params']
    rounds, rounds_c = int(ref['rounds']), int(ref['rounds_cat'])

    # training rows: the seven jjx fits' rows of their own months, labels of those months only
    consumer = P.consumer('jjrefit')
    fits = P.FITS_R2 + P.FITS_R4
    for f in fits:
        consumer.assert_admits(P.split(f), what=f'{f} for jjrefit')
    drop = frozenset(ST.r2_no_features())
    arts_tr = [W / 'out'] + sorted((W / 'kaggle').glob('*/out'))
    mem_tr, rec_tr = ST.members_for(arts_tr, [(f, f) for f in fits], 'training', drop)
    mem_tr, no_feat_tr = ST.drop_no_features(mem_tr, feats)
    tr = ST.frame(mem_tr, feats, proxy)
    keep = consumer.label_mask(ST.time_frame(tr))
    buffer_dropped = int((~keep).sum())
    tr = tr.loc[keep].reset_index(drop=True)
    if a.smoke:
        tr = tr.sample(n=min(a.smoke, len(tr)), random_state=601).reset_index(drop=True)
        rounds, rounds_c = min(rounds, SMOKE_ROUNDS), min(rounds_c, SMOKE_ROUNDS)
    months_tr = tr['ctx_month'].astype('int64')
    assert not months_tr.isin([1, 7]).any(), 'a January or July row is a training row'
    y = ST.labels_2025(tr[ID].to_numpy('int64'))
    consumer.assert_training_frame(ST.time_frame(tr), target=y, what=f'S {a.owner}')
    assert np.isfinite(y).all()
    z = y - tr['anchor'].to_numpy('float64')
    lm = ~ST.pass_through_2025(tr) if ST.LOSS_MASK.get(INPUTS) else np.ones(len(tr), bool)
    params = {**ST.PARAMS, 'num_threads': a.threads}
    cparams = {k: v for k, v in {**ST.CAT_PARAMS, 'thread_count': a.threads}.items() if not k.startswith('od_')}
    print(json.dumps({'owner': (a.owner), 'train_rows': int(len(tr)), 'loss_rows': int(lm.sum()), 'rounds': rounds,
                      'rounds_cat': rounds_c, 'months': months_tr.value_counts().sort_index().to_dict()}), flush=True)
    t1 = time.monotonic()
    bst = lgb.train(params, lgb.Dataset(tr.loc[lm, fx], z[lm], categorical_feature=ST.CATEGORICAL), rounds)
    assert bst.num_trees() == rounds, (bst.num_trees(), rounds)
    bst.save_model(str(paths['model.txt']))
    t2 = time.monotonic()
    print(json.dumps({'lgb_seconds': round(t2 - t1, 1)}), flush=True)
    xc_tr = ST.cat_frame(tr, fx)
    cm = cb.CatBoostRegressor(iterations=rounds_c, **cparams)
    cm.fit(cb.Pool(xc_tr[lm], z[lm], cat_features=ST.CATEGORICAL))
    assert cm.tree_count_ == rounds_c, (cm.tree_count_, rounds_c)
    cm.save_model(str(paths['cat.cbm']))
    t3 = time.monotonic()
    print(json.dumps({'cat_seconds': round(t3 - t2, 1)}), flush=True)
    train_rec = {'rows': int(len(tr)), 'rows_by_month': {int(k): int(v) for k, v in months_tr.value_counts().sort_index().items()},
                 'loss_rows': int(lm.sum()), 'buffer_rows_dropped': buffer_dropped,
                 'rows_without_features_dropped': no_feat_tr, 'member_artifacts': rec_tr,
                 'consumer': consumer.manifest(), 'fits': list(fits)}
    del tr, xc_tr, mem_tr

    # serving on the month it serves: lit, then forced dark with L and Lp lit and dark
    proxy_cols = [c for c in proxy.columns if c not in (ID, 'month')]
    dv = json.loads(a.dark_values.read_text())
    surf_feat = sorted(dv['values']['features'])
    assert sorted(dv['values']['proxy']) == sorted(proxy_cols)
    dark_value = {c: (v['value'], v['share']) for t in dv['values'].values() for c, v in t.items()}
    kept = sorted(c for c, (_, s) in dark_value.items() if s < RD.SHARE_MIN)
    repl_feat = [c for c in surf_feat if c not in kept]
    over = {}
    for item in a.member_dark:
        st, path = item.split('=', 1)
        assert st in RD.MEMBER_COL, st
        f = pd.read_parquet(path, columns=[ID, RD.MEMBER_COL[st]]).set_index(ID)[RD.MEMBER_COL[st]]
        over[st] = (f, {'path': path, 'sha256': ST.sha(Path(path))})
    assert sorted(over) == ['L', 'Lp'], 'give --member-dark for L and Lp'
    arts_sc = [W / 'out-jj', W / 'kaggle/fj-L/out'] + [d for d in sorted((W / 'kaggle').glob('*/out'))
                                                        if d != W / 'kaggle/fj-L/out']
    mem, rec_sc = ST.members_for(arts_sc, [('jjrefit', 'jjrefit')], 'janjul', drop, {})
    mem, no_feat_sc = ST.drop_no_features(mem, feats)
    fi = feats[[ID, 'ctx_month', 'ctx_airport', 'ctx_cls']].set_index(ID)
    mon = fi.ctx_month.reindex(mem[ID].to_numpy()).to_numpy()
    mm = mem[mon == serve_month].reset_index(drop=True)
    ids = mm[ID].to_numpy('int64')
    f_m = feats[feats[ID].isin(ids)].reset_index(drop=True)
    p_m = proxy[proxy[ID].isin(ids)].reset_index(drop=True)
    lit = RD.predict(bst, cm, ST.frame(mm, f_m, p_m), fx, a.threads).set_index(ID).reindex(ids)
    res = {'litL': None, 'darkL': None}
    other = {}
    for tag, ov in (('litL', {}), ('darkL', over)):
        sc, dark_ids = dark_frames(mm, f_m, p_m, dark_value, kept, repl_feat, proxy_cols, ov)
        d = RD.predict(bst, cm, sc, fx, a.threads).set_index(ID).reindex(ids)
        res[tag] = d
        nd = ~np.isin(ids, list(dark_ids))
        dd = d.pred_Sb.to_numpy()[nd] - lit.pred_Sb.to_numpy()[nd]
        other[tag] = {'rows': int(nd.sum()), 'changed': int((dd != 0).sum()),
                      'max_abs_s': float(np.abs(dd).max()) if len(dd) else 0.0}
    is_dark = np.isin(ids, list(dark_ids))
    out = pd.DataFrame({ID: ids, 'month': serve_month, 'airport': fi.ctx_airport.reindex(ids).astype(str).to_numpy(),
                        'cls_lit': fi.ctx_cls.reindex(ids).astype(str).to_numpy(), 'anchor': lit.anchor.to_numpy(),
                        'pred_Sb_lit': lit.pred_Sb.to_numpy(),
                        'pred_Sb_dark_litL': np.where(is_dark, res['litL'].pred_Sb.to_numpy(), np.nan),
                        'pred_Sb_dark_darkL': np.where(is_dark, res['darkL'].pred_Sb.to_numpy(), np.nan),
                        'dark_row': is_dark})
    assert np.isfinite(out.pred_Sb_lit).all() and np.isfinite(out.loc[is_dark, ['pred_Sb_dark_litL', 'pred_Sb_dark_darkL']]).all().all()
    out.to_parquet(paths['parquet'], index=False)
    # label-free comparison with the saved owner that serves this month (trained with the other held month)
    saved_path = W / 'fold/jjrefit' / f'{STAGE}Fjjrefit-month-2025-{serve_month:02d}.parquet'
    saved = pd.read_parquet(saved_path).set_index(ID).pred_Sb.reindex(ids).to_numpy(float)
    dv_ = out.pred_Sb_lit.to_numpy() - saved
    cells = {}
    for ap_ in sorted(out.airport.unique()):
        s = out.airport.eq(ap_).to_numpy()
        cells[ap_] = {'rows': int(s.sum()), 'minus_saved_mean_s': round(float(np.nanmean(dv_[s])), 2),
                      'minus_saved_rms_s': round(float(np.sqrt(np.nanmean(dv_[s] ** 2))), 2)}
        if ap_ in RD.AIRPORTS:
            for tag in ('litL', 'darkL'):
                x = out[f'pred_Sb_dark_{tag}'].to_numpy()[s] - out.pred_Sb_lit.to_numpy()[s]
                cells[ap_][f'dark_{tag}_minus_lit_mean_s'] = round(float(np.mean(x)), 2)
                cells[ap_][f'dark_{tag}_minus_lit_rms_s'] = round(float(np.sqrt(np.mean(x ** 2))), 2)
    rec = {'what': f'R4mks level 2 on the seven jjx months with the rounds of {qown}; serves 2025-{serve_month:02d} '
                   f'with no label of 2025-{out_month:02d} (exp-194 review repair)',
           'owner': (a.owner), 'serves_month': serve_month, 'keeps_out_month': out_month, 'smoke': a.smoke,
           'rounds_from': {'path': str(ref_path), 'sha256': ST.sha(ref_path), 'owner': (qown), 'rounds': rounds,
                           'rounds_cat': rounds_c, 'cv_months': ref_months,
                           'train_rows_by_month': ref['train_rows_by_month']},
           'training': train_rec, 'params': params, 'cat_params': cparams, 'feature_list_sha256': fl_sha,
           'models': {k: ST.sha(paths[k]) for k in ('model.txt', 'cat.cbm')},
           'serving': {'rows': int(len(out)), 'dark_rows': int(is_dark.sum()), 'member_artifacts': rec_sc,
                       'rows_without_features_dropped': no_feat_sc, 'other_airports_moved': other,
                       'saved_owner_output': {'path': str(saved_path), 'sha256': ST.sha(saved_path)},
                       'cells': cells},
           'dark_rule': {'share_min': RD.SHARE_MIN, 'kept_columns': kept, 'replaced_features': repl_feat,
                         'dark_values_json': {'path': str(a.dark_values), 'sha256': ST.sha(a.dark_values)},
                         'member_dark': {k: v[1] for k, v in over.items()}},
           'tables': {'features': ST.sha(W / 'inputs/features-r4c.parquet'),
                      'proxy': ST.sha(W / 'inputs/proxy-features-r4c.parquet')},
           'code_sha256': {'r4mk_iso.py': ST.sha(Path(__file__)), 'r4mk_dark.py': ST.sha(HERE / 'r4mk_dark.py'),
                           'stacker.py': ST.sha(Path(ST.__file__)), 'pilot.py': ST.sha(Path(P.__file__))},
           'versions': {'lightgbm': lgb.__version__, 'catboost': cb.__version__, 'pandas': pd.__version__,
                        'numpy': np.__version__},
           'threads': a.threads, 'parquet_sha256': ST.sha(paths['parquet']),
           'seconds': {'total': round(time.monotonic() - t0, 1), 'lgb': round(t2 - t1, 1), 'cat': round(t3 - t2, 1)}}
    paths['json'].write_text(json.dumps(rec, indent=1, default=str) + '\n')
    print(json.dumps({k: rec[k] for k in ('owner', 'serves_month', 'seconds')}), flush=True)
    print(json.dumps(cells), flush=True)


if __name__ == '__main__':
    main()
