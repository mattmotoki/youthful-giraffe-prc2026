"""R9 on finalplan's assembly (exp-192 assemble192.py, imported unchanged): mix12 in the mlp column of every
member-scope row of the primary frame (fold and 2026); clock rows hold mix12 already, so only ordinary rows move.

Held side: exp-164 results/stack-pack-012/mlp-fold.parquet (pred_mlp164; the vector of the earlier R9 read).
2026 side, ranking mode: stack-pack-012/mlp-ranking.parquet (pred_mlp164); final mode: finalplan's mlp rule with
finalcore's part (mlp_choice gives mix12 on every member-scope row).
Every later stage refits as assemble192 does. The run asserts that the new vector equals the old column on the
member-scope clock rows, so the change is on ordinary rows only.

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I experiments/exp-194-gain-lane/code/assemble_r9.py \
        --variant r9|base --mode ranking --repo . --export <release export> --out <dir>

Writes <dir>/record.json (assemble192's record) and <dir>/values.parquet (fold: zabd final and v64 per stage; 2026:
zabd final and v64), plus <dir>/r9.json (counts and checks of the swap).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'experiments/exp-192-final-assembly/code'))
import assemble192 as P  # noqa: E402

ID = 'MVT_ID_mvt'
PACK12 = REPO / 'experiments/exp-164-double-down/results/stack-pack-012'
VEC = {'fold': (PACK12 / 'mlp-fold.parquet', 'fe4558cc807f4128552ac2d54e0aa27e769be3a6b84cadcee6124c6bb188494a'),
       'ranking': (PACK12 / 'mlp-ranking.parquet', '275d7fb36a798d43caaa525ba58e2825b7a9bdbea5624f18495e442d35479ad6')}
SWAP = {}
HELD = {}
FR = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mix12(split, ids):
    path, want = VEC[split]
    assert sha(path) == want, path
    q = pd.read_parquet(path, columns=[ID, 'pred_mlp164'])
    assert q[ID].is_unique
    return q.set_index(ID).pred_mlp164.reindex(pd.Index(ids, name=ID)).to_numpy(float)


def swap(frames, digests, provenance, R, split, mode):
    key = f'{split}/primary_frame'
    q = frames[key].copy()
    ids = q[ID].to_numpy('int64')
    scope = q.member_scope.to_numpy(bool)
    if split == 'ranking':
        cls = q.cls.to_numpy(object)
    else:
        b = frames['fold/bundle']
        assert np.array_equal(b[ID].to_numpy('int64'), ids)
        cls = b.cls.to_numpy(object)
    old = q.mlp.to_numpy(float)
    if split == 'ranking' and mode == 'final':
        new = old  # finalplan's mlp rule with the patched mlp_choice already gives mix12 on scope rows
    else:
        new = np.where(scope, mix12(split, ids), old)
    clk = scope & (cls == 'clock')
    assert np.isfinite(new[scope]).all(), f'{split}: mix12 missing on a member-scope row'
    assert np.array_equal(new[clk], old[clk]), f'{split}: mix12 differs from the column on clock rows'
    q['mlp'] = new.astype(q.mlp.dtype)
    digests[key] = R.frame_digest(q)
    q.attrs['digest'] = digests[key]
    frames[key] = q
    provenance[f'{key}:mlp'] = 'R9: mix12 on member-scope rows (exp-194 assemble_r9.py)'
    ch = scope & (new != old)
    SWAP[split] = {'scope_rows': int(scope.sum()), 'clock_scope_rows': int(clk.sum()),
                   'changed_rows': int(ch.sum()), 'changed_outside_ordinary_scope': int((ch & (cls == 'clock')).sum()),
                   'mean_abs_move_s': float(np.abs(new - old)[ch].mean()) if ch.any() else 0.0,
                   'digest': digests[key]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['base', 'r9'], required=True)
    p.add_argument('--mode', choices=['ranking', 'final'], default='ranking')
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sources')
    p.add_argument('--pack-overlay', type=Path, help='r9, ranking mode: write the R9 ZABD-PACK.json, its two saved '
                   'mlp vectors and the chain reference (ID, zabd, v64 of 2026) under this new folder')
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(mode=a.mode, repo=a.repo.resolve(), export=a.export.resolve(), out=out / 'record.json',
                              sources=a.sources, parts_dir=None, pack_out=None, standin=None, r_only=False,
                              machine_ref=None)
    if a.variant == 'r9':
        args.side_read = 'R9 (exp-194 assemble_r9.py): mix12 on member-scope rows of the primary frame'
        orig = P.build_frames

        def build_frames_r9(args_, R, K, GS, pack, off, sources, injected=None):
            if a.mode == 'final':
                K.mlp_choice = lambda m8, m12, scope, cls: np.where(np.asarray(scope, bool), m12, m8)
            frames, digests, provenance, counts = orig(args_, R, K, GS, pack, off, sources, injected)
            for split in ('fold', 'ranking'):
                swap(frames, digests, provenance, R, split, a.mode)
            counts['r9'] = SWAP
            FR.update(R=R, frames=frames, digests=digests)
            return frames, digests, provenance, counts
        P.build_frames = build_frames_r9
    orig_held = P.t11b_held

    def t11b_held_keep(*x, **k):
        v, facts = orig_held(*x, **k)
        HELD['v64'] = np.asarray(v, float).copy()
        return v, facts
    P.t11b_held = t11b_held_keep
    orig_post = P.post_steps

    def post_steps_keep(args_, D, off, contexts, *x, **k):
        b = contexts['fold']['b']
        HELD['ids'] = b[ID].to_numpy('int64').copy()
        return orig_post(args_, D, off, contexts, *x, **k)
    P.post_steps = post_steps_keep
    rec = P.run(args)
    L = P.LAST
    vals = {'fold_ids': HELD['ids'], 'fold_final': L['fold_final'], 'fold_v64': HELD['v64'],
            'ranking_final': L['candidate'], 'ranking_v64': L['v64'], 'ranking_ids': L['ids']}
    np.savez_compressed(out / 'values.npz', **vals)
    (out / 'r9.json').write_text(json.dumps({'variant': a.variant, 'mode': a.mode, 'swap': SWAP,
                                             'hashes': rec['hashes'], 'code': {HERE.name: sha(HERE)}}, indent=1) + '\n')
    if a.pack_overlay:
        assert a.variant == 'r9' and a.mode == 'ranking'
        overlay(a.pack_overlay.resolve(), args.export, rec, L)
    print(json.dumps({'variant': a.variant, 'swap': SWAP, 'hashes': rec['hashes']}, indent=1))


def overlay(dest, export, rec, L):
    """The R9 pack in ZABD-PACK form: the two primary frames' mlp columns point at new saved files under
    release/saved-vectors/r9/; frame digests, manifest records and the three body hashes are the R9 run's. Each changed
    frame is rebuilt from the written file and must give its digest. Copy <dest>/release/* over an export copy."""
    R, frames, digests = FR['R'], FR['frames'], FR['digests']
    pack = json.loads((export / 'release/ZABD-PACK.json').read_text())
    dest.mkdir(parents=True, exist_ok=False)
    out = copy.deepcopy(pack)
    for split in ('fold', 'ranking'):
        key = f'{split}/primary_frame'
        f = Path('release/saved-vectors/r9') / key / 'mlp.json'
        (dest / f).parent.mkdir(parents=True)
        (dest / f).write_text(json.dumps(R.encode_array(frames[key]['mlp'])) + '\n')
        cols = out['frames'][key]['columns']
        i = next(j for j, c in enumerate(cols) if c['name'] == 'mlp')
        cols[i] = {**cols[i], 'file': str(f), 'sha256': sha(dest / f)}
        out['frames'][key]['digest'] = digests[key]
        v = R.decode_array(dest / f, cols[i]['sha256'], len(frames[key]))
        q = frames[key].copy(); q['mlp'] = v.astype(cols[i]['dtype'])
        assert R.frame_digest(q) == digests[key], key
    for key, spec in pack['frames'].items():
        if not key.endswith('primary_frame'):
            assert digests[key] == spec['digest'], key
    out['manifest'] = P.manifest_for(pack, digests, rec['template_sha256'])
    out['target'] = 'zabd-r9 (exp-194 development read; R9 = mix12 on member-scope rows)'
    out['parent_sha256'], out['with_c65_sha256'] = rec['hashes']['parent'], rec['hashes']['with_c65']
    out['candidate_sha256'] = rec['hashes']['candidate']
    out['r9'] = {'v64_sha256': rec['hashes']['v64'], 'swap': SWAP, 'code': {HERE.name: sha(HERE)}}
    (dest / 'release/ZABD-PACK.json').write_text(json.dumps(out, indent=1) + '\n')
    ref = pd.DataFrame({ID: L['ids'].astype('float64'), 'zabd': L['candidate'], 'v64': L['v64']})
    ref.to_parquet(dest / 'chain_ref_r9.parquet', index=False)
    files = sorted(p for p in dest.rglob('*') if p.is_file())
    (dest / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(dest)}\n' for p in files))


if __name__ == '__main__':
    main()
