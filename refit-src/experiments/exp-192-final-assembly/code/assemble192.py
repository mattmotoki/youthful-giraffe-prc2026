"""The zabd assembly on rebuilt frames (exp-192, finalplan, 2026-10-08).

Flow: the numerical lines of `release/assembly_replay.assemble` (build035 order) with the exp-169 code of the
release export, patched by finalcore's `pins/E-final-months.patch` (copy in ../export, PROVENANCE.md).

Frames (SPLIT.md): every pack frame is built column by column in the pack's column order and dtype.
- My rule columns (both modes): fold exposure weight, fold and 2026 member scope, pair scope, control scope,
  primary-frame cls (copy of the bundle cls). Rules: frames192.py.
- Ranking mode: every other column as the pack builds it (id, official, label, saved array).
  Check: each of the 28 frame digests equals the pack's, and the parent, C65 and candidate bytes equal the
  pack's hashes.
- Final mode: official columns from the final files (prc.data final mode), fold frames as saved (2025 inputs),
  and every saved column of a 2026 frame from a part named in the sources file (no saved ranking-file vector).

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I code/assemble192.py --mode ranking
       --repo <repo root> --export <release export> --out <record json> [--sources <json>] [--parts-dir <dir>]
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPORT_COPY = HERE.parent / 'export'
CODE = EXPORT_COPY / 'experiments/exp-169-stack-plan/code'
ID = 'MVT_ID_mvt'
#: R-only build (--r-only): the member columns that the R chain does not read. Without a part they get a placeholder.
MEMBER_COLUMNS = {(f'ranking/vector/{m}', c) for m, c in (('g', 'pred_Gmixed'), ('sequence021', 'pred_seq170'),
                  ('sequence039', 'pred_seq170'), ('row6', 'pred_seq170'), ('sequence011', 'pred_seq170'),
                  ('row011', 'pred_seq170'), ('dark18', 'pred_seq170'))} | {('ranking/primary_frame', 'mlp')}
R_ONLY_PLACEHOLDER = 1000.0

RULE_COLUMNS = {
    'fold/exposure': ['weight'],
    'fold/primary_frame': ['member_scope'],
    'fold/pair_frame': ['member_scope', 'control_scope'],
    'ranking/primary_frame': ['member_scope', 'cls'],
    'ranking/pair_frame': ['member_scope', 'control_scope'],
}


LAST = {}  # arrays of the last run (fold final, 2026 candidate, v64, ids, bodies) for in-process reads


def plain(o):
    """JSON-ready copy of nested weight records (numpy arrays and scalars to lists and floats)."""
    if isinstance(o, dict): return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [plain(v) for v in o]
    if isinstance(o, np.ndarray): return plain(o.tolist())
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)): return int(o)
    return o if isinstance(o, (str, type(None))) else str(o)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def setup(args):
    os.environ['PRC_RANKING_MODE'] = args.mode
    for p in (str(CODE), str(HERE)):
        if p not in sys.path: sys.path.insert(0, p)
    sys.path.append(str(args.export / 'release'))
    sys.path.append(str(args.repo))
    import assembly_replay as R
    import assemble_final_032 as A
    import dark_twin035 as T
    import final_months169 as M
    import frames192 as K
    from prc import data as D
    from prc import gscore as GS
    assert Path(A.__file__).resolve().parent == CODE and Path(T.__file__).resolve().parent == CODE
    assert Path(R.__file__).resolve().parent == (args.export / 'release').resolve()
    assert Path(D.__file__).resolve().parent == (args.repo / 'prc').resolve(), D.__file__
    assert M.final() == (args.mode == 'final')
    return R, A, T, M, K, D, GS


def official(args, R, D, pack):
    """Official rows in split order: fold = Jan then Jul 2025 DEP rows; 2026 = DEP rows in template order."""
    if args.mode == 'ranking':
        return R.official_tables(args.repo, pack['official_sha256'])
    data = D.data_dir()
    parts = []
    for key in ('training_2025_01', 'training_2025_07'):
        path = data / Path(R.OFFICIAL_FILES[key]).name
        assert sha256_file(path) == pack['official_sha256'][key], path
        month = pd.read_parquet(path)
        parts.append(month[month['PHASE_mvt'].eq('DEP')])
    fold = pd.concat(parts, ignore_index=True)
    assert fold[ID].is_unique
    template_path = D.submitting_file('final')
    template = pd.read_parquet(template_path)
    ranking = pd.read_parquet(D.ranking_file('final'))
    ranking = ranking[ranking['PHASE_mvt'].eq('DEP')]
    assert ranking[ID].is_unique and template[ID].is_unique and len(template) == D.n_ranking('final')
    ranking = ranking.set_index(ID).loc[template[ID].to_numpy()].reset_index()
    return {'fold': fold, 'ranking': ranking, 'template_path': template_path, 'ranking_path': D.ranking_file('final')}


def read_part(src, ids):
    """One column of a part: a parquet in template order, checked by SHA-256 and by its ID column."""
    path = Path(src['path'])
    assert sha256_file(path) == src['sha256'], f'part hash mismatch: {path}'
    q = pd.read_parquet(path, columns=[ID, src['column']])
    assert len(q) == len(ids) and np.array_equal(q[ID].to_numpy('int64'), np.asarray(ids).astype('int64')), path
    return q[src['column']].to_numpy()


def build_frames(args, R, K, GS, pack, off, sources, injected=None):
    rows = {s: len(off[s]) for s in ('fold', 'ranking')}; injected = injected or {}
    raw, provenance = {}, {}
    for key, spec in pack['frames'].items():
        split = spec['split']; cols = {}
        for col in spec['columns']:
            name, kind = col['name'], col['kind']
            if name in RULE_COLUMNS.get(key, []) or (key, name) == ('ranking/primary_frame', 'mlp') and 'mlp_part' in sources:
                cols[name] = None; provenance[f'{key}:{name}'] = 'rule'; continue
            src = sources.get(key, {}).get(name) if key in sources and isinstance(sources[key], dict) else None
            if name in injected.get(key, {}):
                assert src is None, f'two sources for {key}:{name}'
                cols[name] = injected[key][name]; provenance[f'{key}:{name}'] = 'DRY RUN stand-in'
            elif src is not None:
                cols[name] = read_part(src, off[split][ID].to_numpy())
                provenance[f'{key}:{name}'] = f"part {src['path']} {src['sha256'][:12]}"
            elif getattr(args, 'r_only', False) and (key, name) in MEMBER_COLUMNS:
                cols[name] = np.full(rows[split], R_ONLY_PLACEHOLDER)
                provenance[f'{key}:{name}'] = 'R-ONLY placeholder (no part; the R chain does not read it)'
            elif kind == 'saved':
                assert args.mode == 'ranking' or split == 'fold', f'final mode needs a part for {key}:{name}'
                cols[name] = R.decode_array(args.export / col['file'], col['sha256'], rows[split])
                provenance[f'{key}:{name}'] = 'pack saved'
            elif kind == 'id':
                cols[name] = off[split][ID].to_numpy()
            elif kind == 'official':
                s = off[split][col['source']]
                cols[name] = s.dt.month.to_numpy() if col.get('transform') == 'month' else s.to_numpy()
            elif kind == 'label':
                cols[name] = off[split]['TAXITIME_SEC_mvt'].to_numpy()
            else:
                raise SystemExit(f'unknown column kind {kind}')
        raw[key] = cols

    # Rule columns (frames192.py), from the bundle, the rule frame F and the official rows.
    fb, rb = raw['fold/bundle'], raw['ranking/bundle']
    held = GS.load_rows().set_index(ID).loc[off['fold'][ID].to_numpy('int64')].reset_index()
    assert np.array_equal(held.cls.to_numpy(), np.asarray(fb['cls'], object))
    assert np.array_equal(held.month.to_numpy(), np.asarray(fb['month']))
    n26 = len(off['ranking'])
    cells = K.cell_rows(rb['month'], rb['airport'], rb['cls'])
    if args.mode == 'ranking':
        frozen = GS.load_cells().set_index(['month', 'airport', 'cls']).N
        assert cells.reindex(frozen.index).fillna(0).astype(int).eq(frozen).all() and int(cells.sum()) == GS.N_RANKING
    w = K.exposure(held, cells, n26)
    fscope = K.fold_scope(fb['F'], held.aobt, fb['cls'], w, fb['noplan'])
    raw['fold/exposure']['weight'] = w
    raw['fold/primary_frame']['member_scope'] = fscope
    raw['fold/pair_frame']['member_scope'] = fscope
    raw['fold/pair_frame']['control_scope'] = K.control_scope(fscope, fb['cls'])
    rk = off['ranking']; F = np.asarray(raw['ranking/rule_frame']['F'], float)
    aobt3 = rk.AOBT_3_flt.notna().to_numpy(); typed = rk.AIRCRAFT_TYPE_mvt.notna().to_numpy()
    sup = K.support(K.held_counts(held), rb['month'], rb['airport'], rb['cls'])
    scope = K.ranking_scope(F, aobt3, typed, sup)
    pair = K.pair_scope(scope, rb['extension'], F, aobt3)
    raw['ranking/primary_frame']['member_scope'] = scope
    raw['ranking/primary_frame']['cls'] = np.asarray(rb['cls'], object)
    raw['ranking/pair_frame']['member_scope'] = pair
    raw['ranking/pair_frame']['control_scope'] = K.control_scope(pair, rb['cls'])
    # mlp from finalcore's part (columns mlp008, mix12): mix12 on member-scope clock rows, mlp008 on every other
    # row (the released column; checked byte-equal on 344,841 rows, REPORT section 1).
    mp = sources.get('mlp_part')
    if mp is None and getattr(args, 'r_only', False):
        raw['ranking/primary_frame']['mlp'] = np.full(n26, R_ONLY_PLACEHOLDER)
        provenance['ranking/primary_frame:mlp'] = 'R-ONLY placeholder (no part; the R chain does not read it)'
    elif mp is not None:
        m8 = read_part({**mp, 'column': 'mlp008'}, rk[ID].to_numpy()).astype(float)
        m12 = read_part({**mp, 'column': 'mix12'}, rk[ID].to_numpy()).astype(float)
        mlp = K.mlp_choice(m8, m12, scope, rb['cls'])
        assert np.isfinite(mlp[scope]).all(), 'mlp missing on a member-scope row'
        raw['ranking/primary_frame']['mlp'] = mlp
        provenance['ranking/primary_frame:mlp'] = f"rule from part {mp['path']} {mp['sha256'][:12]}"
    counts = {'fold_scope': int(fscope.sum()), 'fold_exposure_positive': int((w > 0).sum()),
              'ranking_scope': int(scope.sum()), 'ranking_pair_scope': int(pair.sum()),
              'ranking_control_scope': int(raw['ranking/pair_frame']['control_scope'].sum()),
              'ranking_rows': n26, 'cells': {f'{m}-{a}-{z}': int(n) for (m, a, z), n in cells.items()}}

    frames, digests = {}, {}
    for key, spec in pack['frames'].items():
        q = pd.DataFrame({c['name']: raw[key][c['name']] for c in spec['columns']})
        for c in spec['columns']:
            if str(q[c['name']].dtype) != c['dtype']: q[c['name']] = q[c['name']].astype(c['dtype'])
        assert len(q) == rows[spec['split']] and q.columns.tolist() == [c['name'] for c in spec['columns']]
        assert q[ID].is_unique and np.isfinite(q[ID].to_numpy(float)).all()
        digests[key] = R.frame_digest(q); q.attrs['digest'] = digests[key]; frames[key] = q
    return frames, digests, provenance, counts


#: Target v64 of release/ENTRIES.json: the registered facts of T11b-L in the ranking mode.
V64 = {'sha256': '9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b', 'p2_rows': 4902,
       'long_taxi_days': {'EGLL': 1, 'EHAM': 8, 'LFPG': 3, 'LTFM': 3}, 'changed_rows': 4078}


def t11b(args, D, off, frames, base):
    """T11b-L (release/t11b_stage.py functions) on the 2026 split; the registered counts become a record."""
    import t11b_stage as S
    rk = off['ranking']; b = frames['ranking/bundle']; ef = frames['ranking/exploit_frame']
    ids = rk[ID].to_numpy('int64')
    blo, bhi = b.lo.to_numpy(float), b.hi.to_numpy(float)
    p2 = S.p2_rows(rk, b.cls.to_numpy(), blo, bhi)
    assert not (p2 & frames['ranking/primary_frame'].member_scope.to_numpy(bool)).any(), 'P2 row in member scope'
    d = frames['ranking/vector/dark18']
    member = pd.Series(d.pred_seq170.to_numpy(float), index=d[ID].to_numpy('int64')).reindex(ids).to_numpy()
    lo, hi = ef.lower.to_numpy(float), ef.upper.to_numpy(float)
    path = args.repo / 'data/raw/ranking.parquet' if args.mode == 'ranking' else D.ranking_file('final')
    raw = pd.read_parquet(path, columns=S.COLS)
    _, g = S.flags(raw)
    days = g[g.snow]
    counts = {str(a): int(n) for a, n in days.groupby('ap').size().items()}
    long_rows = S.row_keys(ids, raw).isin(set(days.ap + '|' + days.dstr)).to_numpy()
    base = np.asarray(base, float)
    out = S.apply(base, member, lo, hi, p2, long_rows)
    fin = np.isfinite(blo) & np.isfinite(bhi); changed = out != base
    checks = {'finite': bool(np.isfinite(out).all()), 'min_60': bool(out.min() >= 60),
              'bounds_029_hold': bool(np.array_equal(np.clip(out, lo, hi), out)),
              'plan_brackets_hold': bool(((out[fin] >= blo[fin]) & (out[fin] <= bhi[fin])).all()),
              'rows_outside_P2_unchanged': bool(np.array_equal(out[~p2], base[~p2])),
              'long_day_rows_unchanged': bool(np.array_equal(out[long_rows], base[long_rows]))}
    assert all(checks.values()), checks
    month = rk.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    facts = {'p2_rows': int(p2.sum()), 'long_taxi_days': counts, 'changed_rows': int(changed.sum()),
             'p2_by_month': {int(m): int((p2 & (month == m)).sum()) for m in sorted(set(month.tolist()))},
             'long_day_keys': sorted(days.ap + '|' + days.dstr), 'rows_on_long_days': int(long_rows.sum()),
             'checks': checks}
    return out, facts


def t11b_held(args, R, D, off, frames, base):
    """T11b-L on the held fold (January and July 2025 departures), with the release functions in the form of finalv's
    iso/dump_chain.py. Only the held reads use it: NOAC and ROLL serve LIRF no-type rows, and those are never P2."""
    import t11b_stage as S
    fd = off['fold']; b = frames['fold/bundle']; ef = frames['fold/exploit_frame']
    ids = fd[ID].to_numpy('int64')
    for key in ('fold/bundle', 'fold/exploit_frame', 'fold/vector/dark18', 'fold/primary_frame'):
        assert np.array_equal(frames[key][ID].to_numpy().astype('int64'), ids), key
    blo, bhi = b.lo.to_numpy(float), b.hi.to_numpy(float)
    p2 = S.p2_rows(fd, b.cls.to_numpy(), blo, bhi)
    assert not (p2 & frames['fold/primary_frame'].member_scope.to_numpy(bool)).any(), 'held P2 row in member scope'
    member = frames['fold/vector/dark18'].pred_seq170.to_numpy(float)
    lo, hi = ef.lower.to_numpy(float), ef.upper.to_numpy(float)
    root = args.repo if args.mode == 'ranking' else D.data_dir()
    raw = pd.concat([pd.read_parquet(root / (R.OFFICIAL_FILES[k] if args.mode == 'ranking' else
                                             Path(R.OFFICIAL_FILES[k]).name), columns=S.COLS)
                     for k in ('training_2025_01', 'training_2025_07')], ignore_index=True)
    _, g = S.flags(raw)
    days = g[g.snow]
    long_rows = S.row_keys(ids, raw).isin(set(days.ap + '|' + days.dstr)).to_numpy()
    base = np.asarray(base, float)
    out = S.apply(base, member, lo, hi, p2, long_rows)
    assert np.isfinite(out).all() and np.array_equal(out[~p2], base[~p2]) and np.array_equal(out[long_rows], base[long_rows])
    assert not (p2 & (b.cls.to_numpy() == 'noac')).any()
    return out, {'p2_rows': int(p2.sum()), 'long_taxi_days': len(days), 'changed_rows': int((out != base).sum())}


def held_g(y, month, weights, chains):
    """G = sum w ((y - old)^2 - (y - new)^2) of each step on the held fold (finalv's fold_shift_read.py form), under
    each weight set {name: w}; chains {name: [(label, values)]} in step order, the first entry being the start."""
    assert np.isfinite(y).all()
    def g(w, old, new, m):
        d = w * ((y - old) ** 2 - (y - new) ** 2)
        return round(float(d[m].sum()), 1)
    out = {}
    for cname, steps in chains.items():
        rows = []
        for (_, old), (label, new) in zip(steps, steps[1:]):
            old, new = np.asarray(old, float), np.asarray(new, float)
            row = {'step': label, 'rows_changed': int((old != new).sum())}
            for wname, w in weights.items():
                row[wname] = g(w, old, new, np.ones(len(y), bool))
                row[wname + '_by_month'] = {str(k): g(w, old, new, month == k) for k in (1, 7)}
            rows.append(row)
        out[cname] = rows
    return out


def manifest_for(pack, digests, template_sha):
    """The pack manifest with each frame record pointing at its new digest and the template at its file hash."""
    m = copy.deepcopy(pack['manifest'])
    def fix(o):
        if isinstance(o, dict):
            if 'path' in o and 'sha256' in o:
                if o['path'] in digests: o['sha256'] = digests[o['path']]
                elif o['path'] == 'official:submitting': o['sha256'] = template_sha
            else:
                for v in o.values(): fix(v)
    fix(m)
    return m


def run(args):
    R, A, T, M, K, D, GS = setup(args)
    pack = json.loads((args.export / 'release/ZABD-PACK.json').read_text())
    sources = json.loads(Path(args.sources).read_text()) if args.sources else {}
    args.dry_run = bool(args.standin) or bool(sources.get('dry_run'))  # stand-in values: no output is a part
    t0 = time.time()
    off = official(args, R, D, pack)
    injected = None
    if args.standin:
        assert args.mode == 'final' and not args.parts_dir, 'a stand-in run is a final-mode dry run and writes no part'
        import standin192 as SI
        old = Path(R.OFFICIAL_FILES['submitting']).name
        old = D.data_dir() / old
        assert sha256_file(old) == pack['official_sha256']['submitting'], old
        off['pack_ranking_ids'] = pd.read_parquet(old)[ID].to_numpy('int64')
        injected = SI.standins(R, args.export, pack, off, args.standin)
        if getattr(args, 'r_only', False):  # test of the R-only path: the members get placeholders instead of stand-ins
            injected = {k: {c: v for c, v in cols.items() if (k, c) not in MEMBER_COLUMNS} for k, cols in injected.items()}
    frames, digests, provenance, counts = build_frames(args, R, K, GS, pack, off, sources, injected)
    frame_check = {k: digests[k] == s['digest'] for k, s in pack['frames'].items()}
    if args.mode == 'ranking' and not getattr(args, 'side_read', None):
        bad = [k for k, ok in frame_check.items() if not ok]
        assert not bad, f'rebuilt frames differ from the pack: {bad}'
    template_sha = sha256_file(off['template_path'])
    manifest = manifest_for(pack, digests, template_sha)
    rd = R.make_reader(A, frames, pack['records'], off['template_path'])
    parent, extra = manifest['parent'], manifest['extra']
    # assembly_replay.assemble, numerical lines (build035 order).
    contexts = A.load(parent, rd)
    pw = A.fit_weights(parent, contexts, rd, {'refit': True})
    old = {split: A.pre_stack(c, pw, split)[0] for split, c in contexts.items()}
    A.rules(parent, contexts, old, rd)
    A.tail(parent, contexts, old, rd)
    _, parent_body = A.serialize(contexts['ranking'], old['ranking']['final'], rd, parent['common']['template'])
    extras = {split: rd.aligned(extra[split], c['b'], c['scope'], year=2025 if split == 'fold' else 2026)
              for split, c in contexts.items()}
    fw, pp = T.fit(contexts['fold'], extras['fold'], rd.frame(parent['common']['rule_frame']['fold']).y.to_numpy(),
                   rd.frame(parent['common']['exposure']).weight.to_numpy())
    stages = {}
    for split, c in contexts.items():
        assert T.zero_identity(c, extras[split], pw, split)
        stages[split], _ = T.pre(c, extras[split], pw, fw, pp, split)
    statistics, zrules = A.rules(parent, contexts, stages, rd)
    A.tail(parent, contexts, stages, rd)
    _, with_c65_body = A.serialize(contexts['ranking'], stages['ranking']['final'], rd, parent['common']['template'])
    with_c65 = {s: np.asarray(stages[s]['final'], float).copy() for s in stages}
    rd.zero_order_delta = frozenset(parent['common']['exploit_frame'][s]['path'] for s in ('fold', 'ranking'))
    A.tail(parent, contexts, stages, rd)
    table, body = A.serialize(contexts['ranking'], stages['ranking']['final'], rd, parent['common']['template'])

    h = {'parent': hashlib.sha256(parent_body).hexdigest(), 'with_c65': hashlib.sha256(with_c65_body).hexdigest(),
         'candidate': hashlib.sha256(body).hexdigest()}
    want = {'parent': pack['parent_sha256'], 'with_c65': pack['with_c65_sha256'], 'candidate': pack['candidate_sha256']}
    record = {'mode': args.mode + (' DRY RUN with stand-ins (standin192.py); not a part' if args.standin else '')
              + (' DRY RUN with stand-in parts (standin_parts.py); not a part' if sources.get('dry_run') else '')
              + (' R-ONLY: member placeholders; only the r part, the R file and their records are outputs'
                 if getattr(args, 'r_only', False) else ''),
              'export': str(args.export), 'template_sha256': template_sha,
              'code_copy': {p.name: sha256_file(p) for p in sorted(CODE.glob('*.py'))},
              'frames_equal_pack': frame_check, 'frame_digests': digests, 'provenance': provenance,
              'counts': counts, 'hashes': h, 'pack_hashes': want,
              'equal_pack': {k: h[k] == want[k] for k in h},
              'rows': {s: len(off[s]) for s in ('fold', 'ranking')}, 'seconds': round(time.time() - t0, 1),
              'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
    if args.mode == 'ranking' and not getattr(args, 'side_read', None):
        if getattr(args, 'machine_ref', None):
            # The pack bytes are Mac bytes (Accelerate); another machine is checked against the released, unchanged
            # assembly_replay on that machine (code/machine_read.py release; REPORT.md section 4).
            ref = json.loads(Path(args.machine_ref).read_text())
            record['machine_ref'] = {'path': str(args.machine_ref), 'machine': ref['machine'], 'blas': ref['blas'],
                                     'hashes': ref['hashes']}
            record['equal_machine_ref'] = {k: h[k] == ref['hashes'][k] for k in h}
            assert ref['machine'] == platform.machine(), (ref['machine'], platform.machine())
            assert all(record['equal_machine_ref'].values()), record['equal_machine_ref']
        else:
            assert all(record['equal_pack'].values()), record['equal_pack']
    record['weights'] = plain({'pair_and_primary': pw, 'dark18': {'primary_fold': fw, 'primary_ranking': pp}})
    record['rule_statistics'] = plain(statistics)
    if getattr(args, 'side_read', None): record['mode'] += f' SIDE READ: {args.side_read}'
    LAST.clear(); LAST.update(fold_final=np.asarray(stages['fold']['final'], float).copy())
    v64, facts = t11b(args, D, off, frames, stages['ranking']['final'])
    q = pd.DataFrame({ID: off['ranking'][ID].to_numpy('float64'), 'TAXITIME_SEC_mvt': v64})
    import io
    buffer = io.BytesIO(); q.to_parquet(buffer, index=False)
    record['hashes']['v64'] = hashlib.sha256(buffer.getvalue()).hexdigest(); record['t11b'] = facts
    LAST.update(candidate=np.asarray(stages['ranking']['final'], float).copy(), v64=v64.copy(),
                ids=off['ranking'][ID].to_numpy('int64').copy(),
                bodies={'parent': parent_body, 'with_c65': with_c65_body, 'candidate': body})
    if args.mode == 'ranking' and not getattr(args, 'side_read', None):
        assert all(facts[k] == V64[k] for k in ('p2_rows', 'long_taxi_days', 'changed_rows')), facts
        if not getattr(args, 'machine_ref', None):
            assert record['hashes']['v64'] == V64['sha256'], record['hashes']['v64']
            record['equal_pack']['v64'] = True
    stages['ranking']['v64'] = v64
    # The held fold gets T11b-L too (finalv's held read of the v64 step); no 2026 value reads it.
    stages['fold']['v64'], record['t11b_held'] = t11b_held(args, R, D, off, frames, stages['fold']['final'])
    # R chain (PLAN.md path R): the route and head blend, the rules and the candidate's tail (C65 out); no member and
    # no T11b-L. A.rules fits its C95/C98 statistics on the R chain's own held fold values.
    rst = {}
    for split, c in contexts.items():
        pre = A.independent_tail(c['b'], c['base'])
        np.testing.assert_array_equal(pre, A.clip(pre, c['b'].lo.to_numpy(), c['b'].hi.to_numpy()))
        rst[split] = {'route': c['b'].R4mk.to_numpy(), 'head': c['base'], 'pre_exploit': pre}
    r_statistics, rrules = A.rules(parent, contexts, rst, rd)
    A.tail(parent, contexts, rst, rd)
    record['r_rule_statistics'] = plain(r_statistics)
    # The Z and R files: NOAC and ROLL of c5 on top of v64 and of the R chain (post192.py).
    post, record['post'] = post_steps(args, D, off, contexts, rd, parent,
                                      {'Z': {s: stages[s]['v64'] for s in stages}, 'R': {s: rst[s]['final'] for s in rst}})
    for name, st in (('Z', stages), ('R', rst)):
        for split in st:
            st[split]['NOAC'], st[split]['ROLL'] = post[name][split]['NOAC'], post[name][split]['ROLL']
    y = rd.frame(parent['common']['rule_frame']['fold']).y.to_numpy(float)
    month = contexts['fold']['b'].month.to_numpy()
    record['held_rmse'] = {name: {key: held_rmse(y, month, st['fold'][key]) for key in ('final', 'NOAC', 'ROLL')}
                           for name, st in (('Z', stages), ('R', rst))}
    # Held G of each step under the released exposure weights (the pack's saved column) and under the weights of
    # this run (design row 3 in the final mode; the same column in the ranking mode).
    wcol = next(c for c in pack['frames']['fold/exposure']['columns'] if c['name'] == 'weight')
    assert wcol['kind'] == 'saved'
    w_rel = np.asarray(R.decode_array(args.export / wcol['file'], wcol['sha256'], len(y)), float)
    ex = rd.frame(parent['common']['exposure'])
    assert np.array_equal(ex[ID].to_numpy().astype('int64'), contexts['fold']['b'][ID].to_numpy('int64'))
    w_run = ex.weight.to_numpy(float)
    assert args.mode == 'final' or np.array_equal(w_run, w_rel)
    record['held_g'] = {
        'weights': {k: {'sum': float(w.sum()), **{str(m): float(w[month == m].sum()) for m in (1, 7)},
                        'rows_zero': int((w == 0).sum())} for k, w in (('released', w_rel), ('run', w_run))},
        'rows_weight_differs': int((w_rel != w_run).sum()),
        'steps': held_g(y, month, {'released': w_rel, 'run': w_run},
                        {'Z': [('route', stages['fold']['route'])] + [(n, stages['fold'][k]) for n, k in CHAIN + POST],
                         'R': [('route', rst['fold']['route'])] + [(n, rst['fold'][k]) for n, k in R_CHAIN + POST],
                         'Z file against R file': [('R file', rst['fold']['ROLL']), ('Z file', stages['fold']['ROLL'])]})}
    b = contexts['ranking']['b']
    eligible = {name: post[name]['ranking']['eligible'] for name in post}
    record['bounds_planted'] = {'Z': planted_bounds(A, zrules['ranking'], stages['ranking'], b),
                                'R': planted_bounds(A, rrules['ranking'], rst['ranking'], b)}
    planted_path = str(Path(args.parts_dir) / 'bounds-planted.json') if args.parts_dir else f'{args.out} bounds_planted'
    bz, br = ({'masks': bound_rows(rules['ranking'], st['ranking']['C98']), 'planted': record['bounds_planted'][k],
               'path': planted_path, 'key': k} for k, rules, st in (('Z', zrules, stages), ('R', rrules, rst)))
    record['stage_changes'] = stages_record(b, stages['ranking'], bounds=bz)
    record['stage_records'] = {'zabd': record['stage_changes'],
                               'z_file': stages_record(b, stages['ranking'], CHAIN + POST, eligible=eligible['Z'],
                                                       bounds=bz),
                               'r': stages_record(b, rst['ranking'], R_CHAIN, bounds=br),
                               'r_file': stages_record(b, rst['ranking'], R_CHAIN + POST, eligible=eligible['R'],
                                                       bounds=br)}
    files = {'r': rst['ranking']['final'], 'z_file': stages['ranking']['ROLL'], 'r_file': rst['ranking']['ROLL']}
    bodies = {k: A.serialize(contexts['ranking'], v, rd, parent['common']['template'])[1] for k, v in files.items()}
    record['hashes'].update({k: hashlib.sha256(v).hexdigest() for k, v in bodies.items()})
    LAST.update(rst=rst, file_bodies=bodies, zst={k: np.asarray(v, float).copy() for k, v in stages['ranking'].items()})
    if args.parts_dir:
        write_parts(args, contexts, stages, with_c65, old, body, record, rst, bodies)
    if args.pack_out:
        assert args.mode == 'final' and not getattr(args, 'r_only', False)
        write_pack(args, R, pack, off, frames, digests, manifest, record)
    Path(args.out).write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('mode', 'hashes', 'equal_pack', 'rows', 'seconds')}, indent=1))
    return record


#: The candidate chain of the 2026 split: (stage name in the --stages record, stage key after the step).
CHAIN = [('head', 'head'), ('pair', 'pair'), ('primary', 'primary'),
         ('pre_exploit (no-plan v3cal, C39b, C73, C33)', 'pre_exploit'), ('C71', 'C71'), ('C95', 'C95'),
         ('C98', 'C98'), ('bounds (stack12)', 'stack12'), ('C65', 'C65'), ('C66', 'C66'), ('C58', 'C58'),
         ('bounds (final)', 'final'), ('T11b-L', 'v64')]
#: The R chain (no member stage, no T11b-L) and the two c5 steps of the Z and R files.
R_CHAIN = [('head', 'head'), ('pre_exploit (no-plan v3cal, C39b, C73, C33)', 'pre_exploit'), ('C71', 'C71'),
           ('C95', 'C95'), ('C98', 'C98'), ('bounds (stack12)', 'stack12'), ('C65', 'C65'), ('C66', 'C66'),
           ('C58', 'C58'), ('bounds (final)', 'final')]
POST = [('NOAC (c5 post_noac; post192.noac)', 'NOAC'), ('ROLL (c5 post_roll; post192.roll)', 'ROLL')]


def post_steps(args, D, off, contexts, rd, parent, chains):
    """NOAC then ROLL (post192.py) on each chain {name: {split: value}}; returns the values and the record."""
    import post192 as P
    iso = str(args.export / 'iso')
    if iso not in sys.path: sys.path.append(iso)
    import roll_step as RS
    assert Path(RS.__file__).resolve().parent == (args.export / 'iso').resolve()
    RS.RAW = D.data_dir()  # the twelve 2025 training files (roll_step's own default is <export>/input-root)
    common = parent['common']
    ms, bounds, cross = {}, {}, {}
    rank_path = off.get('ranking_path') or args.repo / 'data/raw/ranking.parquet'
    paths = {'fold': [RS.RAW / f'{h}.parquet' for h in RS.HELD_FILES], 'ranking': [rank_path]}
    for split, c in contexts.items():
        ids = c['b'][ID].to_numpy('int64')
        assert np.array_equal(off[split][ID].to_numpy('int64'), ids), split
        ms[split] = P.ms_of(off[split])
        f = rd.frame(common['exploit_frame'][split])
        assert np.array_equal(f[ID].to_numpy('int64'), ids)
        bounds[split] = (f.lower.to_numpy(float), f.upper.to_numpy(float))
        cross[split] = P.cross(RS, paths[split], ids)
    y = rd.frame(common['rule_frame']['fold']).y.to_numpy(float)
    out, info = {}, {'roll_raw': str(RS.RAW), 'ranking_file': str(rank_path)}
    for name, ch in chains.items():
        fold = {'v': np.asarray(ch['fold'], float), 'y': y, 'b': contexts['fold']['b'], 'ms': ms['fold']}
        out[name], info[name] = {}, {}
        for split, c in contexts.items():
            lo, hi = bounds[split]
            v1, i1, e1 = P.noac(split, ch[split], c['b'], ms[split], lo, hi, fold)
            v2, i2, e2 = P.roll(split, v1, c['b'], ms[split], lo, hi, RS, cross[split])
            assert np.isfinite(v2).all() and (v2 >= 60).all()
            out[name][split] = {'NOAC': v1, 'ROLL': v2, 'eligible': {'NOAC': e1, 'ROLL': e2}}
            info[name][split] = {'NOAC': i1, 'ROLL': i2}
    return out, info


def held_rmse(y, month, v):
    """Development read: plain RMSE of a held fold value on the rows with a label, all and by month."""
    ok = np.isfinite(y)
    r = lambda m: float(np.sqrt(np.mean((np.asarray(v, float)[m] - y[m]) ** 2)))
    return {'all': r(ok), **{str(k): r(ok & (month == k)) for k in (1, 7)}}


def bound_rows(rule, c98):
    """Rows of the stack12 clip (exploit_rules.Exploit: lower, upper over all rows): rows with lower above 60 s, rows
    under an active bound (lower above 60 s or a finite upper), and rows whose C98 value lies within 30 s inside an
    active bound."""
    v = np.asarray(c98, float); lower, upper = np.asarray(rule.lower, float), np.asarray(rule.upper, float)
    inside = (v >= lower) & (v <= upper)
    margin = np.fmin(np.where(lower > 60, v - lower, np.inf), upper - v)
    return {'lower_above_60': lower > 60, 'active': (lower > 60) | np.isfinite(upper),
            'near_30s': inside & (margin < 30)}


def planted_bounds(A, rule, st, b, n=3):
    """Planted read of the stack12 clip on each new month (desk 07:35 UTC, the month-loop trap): on a copy of the
    stage's parent (pre_exploit), n rows of the month in the C95/C98 scope (no C71 row) with lower above 360 s get
    60 s, so their C98 input lies below their lower bound. The released Exploit.apply runs on the copy (a copy of the
    rule object with the copy's digest). Every other row must equal the build's stack12 value; the read counts the
    planted rows that the clip moves. The build values do not read it."""
    ids = b[ID].to_numpy(); month = b.month.to_numpy(); parent = np.asarray(st['pre_exploit'], float)
    lower = np.asarray(rule.lower, float); out = {}
    for m in [m for m in (2, 6) if (month == m).any()]:  # the ranking file has January/July only
        rows = np.flatnonzero((month == m) & (rule.scope95 | rule.scope98) & ~rule.c71 & (lower > 360))[:n]
        if len(rows) == 0:  # no such row (the stand-in frames have no February/June rule scope)
            out[f'2026-{m:02d}'] = {'rows': [], 'planted': 0, 'moved': 0}
            continue
        planted = parent.copy(); planted[rows] = 60.
        steps = replace(rule, parent_digest=A.vector_digest(ids, planted)).apply(ids, planted)
        others = np.ones(len(ids), bool); others[rows] = False
        assert np.array_equal(steps['stack9'][others], np.asarray(st['stack12'], float)[others])
        moved = steps['stack9'][rows] != steps['C98'][rows]
        out[f'2026-{m:02d}'] = {
            'rows': [int(i) for i in ids[rows]], 'lower': lower[rows].tolist(), 'planted_parent': 60.0,
            'C98': steps['C98'][rows].tolist(), 'below_lower': int((steps['C98'][rows] < lower[rows]).sum()),
            'stack12': steps['stack9'][rows].tolist(), 'moved': int(moved.sum()), 'planted': len(rows),
            'other_rows_equal_build': True}
    return out


def stages_record(b, st, chain=CHAIN, board_only=None, eligible=None, bounds=None):
    """check_final_candidate.py --stages form: per stage, the 2026 rows it changes, by month.

    `eligible` {stage key: row mask} gives a stage's served rows by month. A new month (February, June) where such a
    stage serves no row gets `board_only` with that count as the reason (the stage runs on every month).
    `bounds` {'masks': bound_rows, 'planted': planted_bounds, 'path': its file} is for the stack12 clip, which
    crosses 1 to 5 rows a month. A new month with no crossing gets `board_only` (desk 07:35 UTC, REPORT section 11)
    when the clip has rows with lower above 60 s there and the planted read moved all its rows of that month. The
    reason gives those counts, the expected crossings at the other months' rate, and the planted read's path."""
    month = b.month.to_numpy(); prev = np.asarray(st['route'], float); out = []
    for name, key in chain:
        cur = np.asarray(st[key], float); ch = cur != prev
        row = {'name': name, 'changed_by_month': {f'2026-{m:02d}': int((ch & (month == m)).sum())
                                                  for m in sorted(set(month.tolist()))}}
        if eligible and key in eligible:
            row['eligible_by_month'] = {f'2026-{m:02d}': int((eligible[key] & (month == m)).sum())
                                        for m in sorted(set(month.tolist()))}
            empty = [m for m in ('2026-02', '2026-06') if m in row['eligible_by_month']
                     and row['eligible_by_month'][m] == 0]
            if empty:
                row['board_only'] = (f"no served row in {', '.join(empty)} (eligible_by_month); "
                                     'the stage runs on every month')
        if bounds and key == 'stack12':
            for k, mask in bounds['masks'].items():
                row[f'{k}_by_month'] = {f'2026-{m:02d}': int((mask & (month == m)).sum())
                                        for m in sorted(set(month.tolist()))}
            row['planted'] = {m: {k: v[k] for k in ('moved', 'planted')} for m, v in bounds['planted'].items()}
            ch, n = row['changed_by_month'], {f'2026-{m:02d}': int((month == m).sum()) for m in set(month.tolist())}
            empty = [m for m in ('2026-02', '2026-06') if ch.get(m) == 0 and row['lower_above_60_by_month'][m] > 0
                     and bounds['planted'][m]['planted'] >= 3
                     and bounds['planted'][m]['moved'] == bounds['planted'][m]['planted']]
            if empty:
                rest = [m for m in ch if m not in empty]
                rate = sum(ch[m] for m in rest) / sum(n[m] for m in rest)
                row['board_only'] = (
                    'the stage runs on all rows: one np.clip (exploit_rules.Exploit.apply); '
                    + '; '.join(f"{m}: lower > 60 s on {row['lower_above_60_by_month'][m]:,} rows, "
                                f"{row['near_30s_by_month'][m]} rows within 30 s of a bound, expected crossings "
                                f"{rate * n[m]:.1f} at the other months' rate ({sum(ch[r] for r in rest)} of "
                                f"{sum(n[r] for r in rest):,} rows), 0 seen; planted read: "
                                f"{bounds['planted'][m]['moved']} of {bounds['planted'][m]['planted']} rows set below "
                                f"their lower bound move" for m in empty)
                    + f" ({bounds['path']}, key {bounds['key']})")
        if board_only and name in board_only: row['board_only'] = board_only[name]
        out.append(row); prev = cur
    return {'stages': out}


def write_pack(args, R, pack, off, frames, digests, manifest, record):
    """The final pack in ZABD-PACK form at <pack-out>/release/ZABD-PACK.json (an export copy of export_copy.py).

    A saved column whose values equal the source pack's keeps its saved file; any other saved column is encoded
    (assembly_replay.encode_array) to release/saved-vectors/final/<frame>/<column>.json. Each frame is rebuilt
    from the written files and must give its digest."""
    dest = Path(args.pack_out).resolve(); rel = dest / 'release'
    assert (dest / 'FINAL-COPY.json').is_file() and not (rel / 'ZABD-PACK.json').exists(), dest
    rows = {s: len(off[s]) for s in ('fold', 'ranking')}
    out = {k: v for k, v in pack.items() if k not in ('frames', 'manifest')}
    out['frames'] = {}; reused = written = 0
    for key, spec in pack['frames'].items():
        cols = []
        for col in spec['columns']:
            if col['kind'] != 'saved':
                cols.append(dict(col)); continue
            values = frames[key][col['name']]
            if rows[spec['split']] == pack['rows'][spec['split']]:
                old = R.decode_array(args.export / col['file'], col['sha256'], rows[spec['split']])
                same = pd.Series(old).astype(col['dtype']) if col['dtype'] != 'object' else pd.Series(old, dtype=object)
                if R.frame_digest(pd.DataFrame({col['name']: same})) == R.frame_digest(values.to_frame()):
                    assert (dest / col['file']).is_file() and sha256_file(dest / col['file']) == col['sha256']
                    cols.append(dict(col)); reused += 1; continue
            body = R.encode_array(values)
            f = Path('release/saved-vectors/final') / key / f"{col['name']}.json"
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            assert not (dest / f).exists(), f
            (dest / f).write_text(json.dumps(body) + '\n')
            cols.append({**col, 'file': str(f), 'sha256': sha256_file(dest / f)}); written += 1
        out['frames'][key] = {**spec, 'columns': cols, 'digest': digests[key]}
    out['manifest'] = manifest
    out['target'] = 'zabd-final' + (' DRY RUN stand-in' if args.dry_run else '')
    out['rows'] = rows
    out['official_sha256'] = {**pack['official_sha256'], 'submitting': record['template_sha256'],
                              'ranking': sha256_file(off['ranking_path'])}
    out['parent_sha256'] = record['hashes']['parent']; out['with_c65_sha256'] = record['hashes']['with_c65']
    out['candidate_sha256'] = record['hashes']['candidate']
    out['final_build'] = {'mode': record['mode'], 'v64_sha256': record['hashes']['v64'], 't11b': record['t11b'],
                          'counts': record['counts'], 'provenance': record['provenance']}
    # Rebuild every frame from the written pack (rebuild_frame with the copy as root).
    for key, spec in out['frames'].items():
        q = {}
        for col in spec['columns']:
            if col['kind'] == 'saved':
                v = R.decode_array(dest / col['file'], col['sha256'], rows[spec['split']])
                q[col['name']] = v.astype(col['dtype']) if col['dtype'] != 'object' else v
            else:
                q[col['name']] = frames[key][col['name']].to_numpy()
        q = pd.DataFrame(q)
        for col in spec['columns']:
            if str(q[col['name']].dtype) != col['dtype']: q[col['name']] = q[col['name']].astype(col['dtype'])
        assert R.frame_digest(q) == spec['digest'], key
    (rel / 'ZABD-PACK.json').write_text(json.dumps(out, indent=1) + '\n')
    record['pack'] = {'path': str(rel / 'ZABD-PACK.json'), 'sha256': sha256_file(rel / 'ZABD-PACK.json'),
                      'saved_columns_reused': reused, 'saved_columns_written': written}


def write_parts(args, contexts, stages, with_c65, old, body, record, rst, bodies):
    """Parts `zabd` (zabd, v64) and `r` (r), their stage tables, the Z and R files (NOAC and ROLL on top, submission
    form) and the four --stages records, all in template order under <parts-dir>."""
    out = Path(args.parts_dir); out.mkdir(parents=True, exist_ok=True)
    b = contexts['ranking']['b']; ids = b[ID].to_numpy('int64')
    st, rs = stages['ranking'], rst['ranking']
    f64 = lambda v: np.asarray(v, float)
    keys = ['route', 'head', 'pair', 'primary', 'pre_exploit', 'C71', 'C95', 'C98', 'stack12', 'C65', 'C66', 'C58']
    rkeys = ['route', 'head', 'pre_exploit', 'C71', 'C95', 'C98', 'stack12', 'C65', 'C66', 'C58', 'final', 'NOAC',
             'ROLL']
    tables = {
        'zabd': pd.DataFrame({ID: ids, 'zabd': f64(st['final']), 'v64': f64(st['v64'])}),
        'r': pd.DataFrame({ID: ids, 'r': f64(rs['final'])}),
        'zabd-stages': pd.DataFrame({ID: ids, **{k: f64(st[k]) for k in keys}, 'with_c65': with_c65['ranking'],
                                     'parent': f64(old['ranking']['final']), 'final': f64(st['final']),
                                     'v64': f64(st['v64']), 'NOAC': f64(st['NOAC']), 'ROLL': f64(st['ROLL']),
                                     'member_scope': contexts['ranking']['scope'], 'cls': b.cls.to_numpy(object),
                                     'month': b.month.to_numpy()}),
        'r-stages': pd.DataFrame({ID: ids, **{k: f64(rs[k]) for k in rkeys}, 'cls': b.cls.to_numpy(object),
                                  'month': b.month.to_numpy()})}
    if getattr(args, 'r_only', False):  # the Z outputs read the member placeholders
        tables = {k: v for k, v in tables.items() if k in ('r', 'r-stages')}
    parts = {}
    for name, q in tables.items():
        path = out / f'{name}.parquet'
        assert not path.exists(), f'part exists: {path}'
        q.to_parquet(path, index=False)
        parts[name] = {'path': str(path), 'sha256': sha256_file(path), 'rows': len(q), 'columns': q.columns.tolist()}
    files = (('r-file', 'r_file'),) if getattr(args, 'r_only', False) else (('z-file', 'z_file'), ('r-file', 'r_file'))
    for name, key in files:
        path = out / f'{name}.parquet'
        assert not path.exists(), f'file exists: {path}'
        path.write_bytes(bodies[key])
        parts[name] = {'path': str(path), 'sha256': sha256_file(path), 'rows': len(ids),
                       'columns': [ID, 'TAXITIME_SEC_mvt']}
        assert parts[name]['sha256'] == record['hashes'][key]
    path = out / 'bounds-planted.json'
    assert not path.exists(), f'record exists: {path}'
    planted = {k: v for k, v in record['bounds_planted'].items() if k == 'R' or not getattr(args, 'r_only', False)}
    path.write_text(json.dumps(plain(planted), indent=1) + '\n')
    parts['bounds-planted'] = {'path': str(path), 'sha256': sha256_file(path)}
    for name, rec in record['stage_records'].items():
        if getattr(args, 'r_only', False) and name not in ('r', 'r_file'):
            continue
        path = out / f"{name.replace('_', '-')}-stages-record.json"
        assert not path.exists(), f'record exists: {path}'
        path.write_text(json.dumps(rec, indent=1) + '\n')
        parts[f"{name.replace('_', '-')}-stages-record"] = {'path': str(path), 'sha256': sha256_file(path)}
    record['parts'] = parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['ranking', 'final'], required=True)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--export', type=Path, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sources')
    ap.add_argument('--parts-dir')
    ap.add_argument('--pack-out', help='final export copy (export_copy.py) that gets release/ZABD-PACK.json')
    ap.add_argument('--standin', help='DRY RUN: GBDT parquet for the stand-in values (standin192.py)')
    ap.add_argument('--r-only', action='store_true',
                    help='final mode: build the R part and the R file before the member parts land (placeholders)')
    ap.add_argument('--machine-ref', help='ranking mode off the Mac: machine_read.py release record of this machine')
    args = ap.parse_args()
    args.repo, args.export = args.repo.resolve(), args.export.resolve()
    assert not args.r_only or args.mode == 'final'
    run(args)


if __name__ == '__main__':
    main()
