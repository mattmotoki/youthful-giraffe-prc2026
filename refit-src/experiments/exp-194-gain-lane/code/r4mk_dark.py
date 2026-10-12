"""exp-194 PLAN-DARK job 2: forced-dark held values of R4mk (R4mks pred_Sb, the A6 base), January and July 2025.

The saved held level 2 of R4mks (exp-148 stack/R4mks-month-2025-0M: q01 scores January, q07 July) serves the
jjrefit member vectors as `fold_serve.py --split jjrefit` does (stacker.members_for, frame, feature_list, cat_frame,
fold_serve.model_of and predict), twice:
  lit   the saved features and proxy tables; pred_Sb must equal the saved fold output
        <saved>/R4mksFjjrefit-month-2025-0M.parquet (--tol)
  dark  on every row of EDDF and LIRF in the scored month (the airport-month has no surface reception), before
        frame(): each surface column gets the value of a 2026 row with no reception (dark_values.py JSON:
        features obs_*, q_taxi_o and ctx_cls, and every column of the proxy table, the harvest proxies and the
        row clock inputs) when one value holds SHARE_MIN or more of those 2026 rows (an input that reception
        removes); a column that varies by row there (airborne and flight-list proxies: arrival_*, filed_delay_*,
        departure_count_q95_*, spacing_short_share_*) keeps the row's own value; frame() then derives the queue
        residuals (qr_No_*) and the x_ columns from them.
        Members: the six official-data members keep their values;
        L and Lp (the two that read clock inputs) keep their lit values unless --member-dark gives dark ones
        (STAGE=MVT_ID_mvt parquet with that member's column).
Rows outside the dark airports move only by rounding in exp-047 window_sums (one cumulative sum over every airport):
recorded per owner (rows changed, largest move, airports). No label is read. Write-once
<out>.parquet (MVT_ID_mvt, month, airport, cls_lit, anchor, pred_Sb_lit, pred_Sb_dark, pred_S_dark, pred_Scat_dark)
and <out>.json.

    PRC_SOURCE_ROOT=<snapshot> PRC_DATA_DIR=<raw> python r4mk_dark.py --work <exp-148 work> --saved <fold dir> \
        --dark-values <dark_values.py JSON> --out <path stem> [--member-dark L=<parquet> --member-dark Lp=<parquet>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SNAP = Path(os.environ['PRC_SOURCE_ROOT'])
sys.path.insert(0, str(SNAP / 'experiments/exp-148-short-route/code'))
import fold_serve as FS              # noqa: E402  (stacker as FS.ST, pilot as FS.P)

ST = FS.ST
ID = ST.ID
STAGE, INPUTS = 'R4mks', 'r4mks'
SHARE_MIN = 0.9    # a column is replaced on dark rows when one value holds this share of the 2026 dark rows
OWNERS = (('q01', 1), ('q07', 7))
AIRPORTS = ['EDDF', 'LIRF']
MEMBER_COL = {'L': 'pred_L', 'Lp': 'pred_Lp'}


def predict(bst, cm, sc: pd.DataFrame, fx: list[str], threads: int) -> pd.DataFrame:
    """fold_serve.predict with thread caps."""
    f_l = bst.predict(sc[fx], num_threads=threads)
    f_c = cm.predict(ST.cat_frame(sc, fx), thread_count=threads)
    anc = sc['anchor'].to_numpy('float64')
    return pd.DataFrame({ID: sc[ID].to_numpy('int64'), 'pred_S': np.maximum(anc + f_l, ST.FLOOR),
                         'pred_Scat': np.maximum(anc + f_c, ST.FLOOR),
                         'pred_Sb': np.maximum(anc + (f_l + f_c) / 2, ST.FLOOR), 'anchor': anc})


def mode(s: pd.Series):
    vc = s.value_counts(dropna=False)
    v = vc.index[0]
    return (None if pd.isna(v) else (v.item() if hasattr(v, 'item') else v)), float(vc.iloc[0] / len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True, help='exp-148 work root (stack/, inputs/, out-jj, kaggle/*/out)')
    ap.add_argument('--saved', type=Path, required=True, help='folder of the saved R4mksFjjrefit-month-2025-0M.parquet')
    ap.add_argument('--dark-values', type=Path, required=True)
    ap.add_argument('--member-dark', action='append', default=[], help='L=<parquet> or Lp=<parquet>')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args()
    pq, js = a.out.with_suffix('.parquet'), a.out.with_suffix('.json')
    for q in (pq, js):
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
    models = {o: FS.model_of(W / 'stack', STAGE, o, INPUTS, fx) for o, _ in OWNERS}
    proxy_cols = [c for c in proxy.columns if c not in (ID, 'month')]
    dv = json.loads(a.dark_values.read_text())
    surf_feat = sorted(dv['values']['features'])
    assert sorted(dv['values']['proxy']) == sorted(proxy_cols), 'the held and final proxy tables differ in columns'
    assert all(c in feats.columns for c in surf_feat) and all(c in fx for c in proxy_cols)
    dark_value = {c: (v['value'], v['share']) for t in dv['values'].values() for c, v in t.items()}
    kept = sorted(c for c, (_, s) in dark_value.items() if s < SHARE_MIN)     # present on 2026 dark rows
    repl_feat = [c for c in surf_feat if c not in kept]
    # members
    drop = frozenset(ST.r2_no_features())
    arts = [W / 'out-jj', W / 'kaggle/fj-L/out'] + [d for d in sorted((W / 'kaggle').glob('*/out'))
                                                     if d != W / 'kaggle/fj-L/out']
    mem, mrec = ST.members_for(arts, [('jjrefit', 'jjrefit')], 'janjul', drop, {})
    mem, no_features = ST.drop_no_features(mem, feats)
    over = {}
    for item in a.member_dark:
        st, path = item.split('=', 1)
        assert st in MEMBER_COL, st
        f = pd.read_parquet(path, columns=[ID, MEMBER_COL[st]]).set_index(ID)[MEMBER_COL[st]]
        over[st] = (f, {'path': path, 'sha256': ST.sha(Path(path))})
    fi = feats[[ID, 'ctx_month', 'ctx_airport', 'ctx_cls']].set_index(ID)
    mon = fi.ctx_month.reindex(mem[ID].to_numpy()).to_numpy()
    outs, recs = [], {}
    for o, m in OWNERS:
        bst, cm, mr = models[o]
        mm = mem[mon == m].reset_index(drop=True)
        ids = mm[ID].to_numpy('int64')
        f_m = feats[feats[ID].isin(ids)].reset_index(drop=True)
        p_m = proxy[proxy[ID].isin(ids)].reset_index(drop=True)
        lit = predict(bst, cm, ST.frame(mm, f_m, p_m), fx, a.threads).set_index(ID).reindex(ids)
        saved_path = a.saved / f'{STAGE}Fjjrefit-month-2025-{m:02d}.parquet'
        saved = pd.read_parquet(saved_path).set_index(ID).reindex(ids)
        d_lit = float(np.nanmax(np.abs(lit.pred_Sb.to_numpy() - saved.pred_Sb.to_numpy())))
        assert saved.pred_Sb.notna().all() and d_lit <= a.tol, (o, d_lit)
        dark_f = f_m.ctx_airport.astype(str).isin(AIRPORTS).to_numpy()
        f_d, p_d, m_d = f_m.copy(), p_m.copy(), mm.copy()
        for c in repl_feat:
            v = dark_value[c][0]
            if c == 'ctx_cls':
                f_d[c] = f_d[c].astype(object)
            f_d.loc[dark_f, c] = np.nan if v is None else v
        dark_ids = set(f_m.loc[dark_f, ID].astype('int64'))
        # every dark row gets a proxy row: its own values of the kept columns (NaN when the row is absent from the
        # proxy table, as in the lit merge) and the dark values of the replaced columns
        dark_sorted = np.array(sorted(dark_ids), dtype='int64')
        own = p_m.drop_duplicates(ID).set_index(ID).reindex(dark_sorted)
        new = pd.DataFrame({ID: dark_sorted,
                            **{c: (own[c].to_numpy() if c in kept else
                                   (np.nan if dark_value[c][0] is None else dark_value[c][0])) for c in proxy_cols}})
        p_d = pd.concat([p_d[~p_d[ID].isin(dark_ids)], new], ignore_index=True)
        dark_m = m_d[ID].isin(dark_ids).to_numpy()
        for st, (f, _) in over.items():   # the member's rows (L and Lp have no value on ghost rows)
            have = dark_m & np.isfinite(m_d[st].to_numpy(float))
            v = f.reindex(m_d.loc[have, ID].to_numpy()).to_numpy(float)
            assert np.isfinite(v).all(), f'{st}: dark values missing on {int((~np.isfinite(v)).sum())} rows'
            m_d.loc[have, st] = v
        dark = predict(bst, cm, ST.frame(m_d, f_d, p_d), fx, a.threads).set_index(ID).reindex(ids)
        is_dark = np.isin(ids, list(dark_ids))
        # rows of other airports move only by rounding: exp-047 state.window_sums takes window sums as differences
        # of one cumulative sum over every airport, so the dark rows' values change the last bits of the other
        # airports' queue residuals (stacker.queue_residuals), and a row at a split threshold can move
        nd = ~is_dark
        dd = dark.pred_Sb.to_numpy()[nd] - lit.pred_Sb.to_numpy()[nd]
        ch = dd != 0
        fq = f_m.drop_duplicates(ID).set_index(ID).reindex(ids[nd])
        other = {'rows': int(nd.sum()), 'changed': int(ch.sum()),
                 'max_abs_s': float(np.abs(dd).max()) if len(dd) else 0.0,
                 'rms_changed_s': float(np.sqrt(np.mean(dd[ch] ** 2))) if ch.any() else 0.0,
                 'changed_by_airport': fq.loc[ch, 'ctx_airport'].astype(str).value_counts().head(10).to_dict()}
        print(json.dumps({o: {'other_airports': other}}), flush=True)
        out = pd.DataFrame({ID: ids, 'month': m, 'airport': fi.ctx_airport.reindex(ids).astype(str).to_numpy(),
                            'cls_lit': fi.ctx_cls.reindex(ids).astype(str).to_numpy(),
                            'anchor': lit.anchor.to_numpy(), 'pred_Sb_lit': lit.pred_Sb.to_numpy(),
                            'pred_Sb_dark': dark.pred_Sb.to_numpy(), 'pred_S_dark': dark.pred_S.to_numpy(),
                            'pred_Scat_dark': dark.pred_Scat.to_numpy(), 'dark_row': is_dark})
        outs.append(out)
        cells = {}
        for ap_ in AIRPORTS:
            s = out.airport.eq(ap_).to_numpy()
            dd = out.pred_Sb_dark.to_numpy()[s] - out.pred_Sb_lit.to_numpy()[s]
            cells[ap_] = {'rows': int(s.sum()), 'dark_minus_lit_mean_s': float(dd.mean()),
                          'dark_minus_lit_rms_s': float(np.sqrt(np.mean(dd ** 2)))}
        recs[o] = {'month': m, 'model': mr, 'rows': int(len(ids)), 'dark_rows': int(is_dark.sum()),
                   'lit_vs_saved_max_abs_s': d_lit, 'saved': {'path': str(saved_path), 'sha256': ST.sha(saved_path)},
                   'other_airports': other,
                   'cells': cells}
        print(json.dumps({o: recs[o]['cells'], 'lit_max': d_lit}), flush=True)
    res = pd.concat(outs, ignore_index=True)
    res.to_parquet(pq, index=False)
    rec = {'what': 'forced-dark held values of R4mks pred_Sb (exp-194 PLAN-DARK job 2); label-free',
           'stage': STAGE, 'inputs': INPUTS, 'owners': recs, 'dark_airports': AIRPORTS,
           'zeroed_columns': sorted(ST.ZEROED),
           'dark_values': {c: {'value': v, 'share': s} for c, (v, s) in dark_value.items()},
           'share_min': SHARE_MIN, 'kept_columns': kept,
           'dark_values_json': {'path': str(a.dark_values), 'sha256': ST.sha(a.dark_values)},
           'member_dark': {k: v[1] for k, v in over.items()},
           'members_lit': sorted(set(ST.MEMBERS) - set(over)),
           'member_artifacts': mrec, 'rows_without_features_dropped': no_features,
           'tables': {'features': {'path': str(W / 'inputs/features-r4c.parquet'),
                                   'sha256': ST.sha(W / 'inputs/features-r4c.parquet')},
                      'proxy': {'path': str(W / 'inputs/proxy-features-r4c.parquet'),
                                'sha256': ST.sha(W / 'inputs/proxy-features-r4c.parquet')},
                      },
           'code_sha256': {'r4mk_dark.py': ST.sha(Path(__file__)),
                           'fold_serve.py': ST.sha(SNAP / 'experiments/exp-148-short-route/code/fold_serve.py'),
                           'stacker.py': ST.sha(SNAP / 'experiments/exp-148-short-route/code/stacker.py')},
           'threads': a.threads, 'rows': int(len(res)), 'parquet_sha256': ST.sha(pq),
           'seconds': round(time.monotonic() - t0, 1)}
    js.write_text(json.dumps(rec, indent=1, default=str) + '\n')
    print(json.dumps({'rows': rec['rows'], 'dark_values': {c: rec['dark_values'][c] for c in list(rec['dark_values'])[:60]},
                      'seconds': rec['seconds']}, default=str), flush=True)


if __name__ == '__main__':
    main()
