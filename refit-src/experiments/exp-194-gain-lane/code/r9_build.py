"""R9 on the final base (SPEC-R9.md): finalplan's final build (exp-192 assemble192.py, imported unchanged)
with mix12 in the primary-frame mlp column of every member-scope row, fold and 2026. Variant `base` runs the
released rule through the same wrapper (code check: final-2's parts, files and pack, byte-equal).

R9 rule: 2026 rows through finalplan's mlp rule with mix12 on every member-scope row (finalcore's part; clock scope
rows hold mix12 already); fold rows: the held mix12 (exp-164 stack-pack-012/mlp-fold.parquet, read by assemble_r9.py's
`mix12` with its SHA-256) on the non-clock fold member-scope rows, the pack's saved column elsewhere. So only
non-clock rows change in both splits. Every later stage refits as in final-2
(assemble_final_032.fit_weights with refit).

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I r9_build.py --variant base|r9 --repo <snapshot>
        --export <finalplan's source export> --sources <final-sources-2.json> --parts-dir <new dir>
        --pack-out <export copy of export_copy.py> --out <record json> --fold-out <fold-stages.npz>

Writes assemble192's parts, pack and record, plus <fold-out>: the fold IDs (assembly order) and every Z fold stage
of the held_g chain (route ... ROLL), for the stage reads (r9_read.py). No label is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / 'experiments/exp-192-final-assembly/code'))
import assemble192 as P  # noqa: E402
import assemble_r9 as R9  # noqa: E402

ID = 'MVT_ID_mvt'
CAP = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fold_swap(frames, digests, provenance, R):
    """Fold primary frame: the held mix12 on the non-clock member-scope rows; every other row keeps the pack's saved
    column. On the clock scope rows the count of rows where the saved column differs from the held mix12 is recorded
    (the final fold scope uses the design row 3 weights; 0 expected)."""
    key = 'fold/primary_frame'
    q = frames[key].copy()
    ids = q[ID].to_numpy('int64')
    b = frames['fold/bundle']
    assert np.array_equal(b[ID].to_numpy('int64'), ids)
    cls = b.cls.to_numpy(object)
    scope = q.member_scope.to_numpy(bool)
    old = q.mlp.to_numpy(float)
    m12 = R9.mix12('fold', ids)
    on = scope & (cls != 'clock')
    assert np.isfinite(m12[on]).all(), 'held mix12 missing on a fold member-scope row'
    new = np.where(on, m12, old)
    clk = scope & (cls == 'clock')
    q['mlp'] = new.astype(q.mlp.dtype)
    digests[key] = R.frame_digest(q)
    q.attrs['digest'] = digests[key]
    frames[key] = q
    provenance[f'{key}:mlp'] = 'R9: held mix12 on non-clock member-scope rows (exp-194 r9_build.py)'
    ch = new != old
    return {'scope_rows': int(scope.sum()), 'clock_scope_rows': int(clk.sum()), 'changed_rows': int(ch.sum()),
            'changed_by_class': {str(c): int((ch & (cls == c)).sum()) for c in sorted(set(cls[ch].tolist()))},
            'rms_move_s': float(np.sqrt(np.mean((new - old)[ch] ** 2))) if ch.any() else 0.0,
            'clock_scope_rows_saved_differs_from_mix12': int((clk & (np.abs(old - m12) > 1e-6)).sum()),
            'digest': digests[key]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['base', 'r9'], required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--parts-dir', required=True)
    p.add_argument('--pack-out', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fold-out', type=Path, required=True)
    a = p.parse_args()
    assert not a.fold_out.exists() and not Path(a.out).exists(), 'write-once'
    args = argparse.Namespace(mode='final', repo=a.repo.resolve(), export=a.export.resolve(), out=a.out,
                              sources=a.sources, parts_dir=a.parts_dir, pack_out=a.pack_out, standin=None,
                              r_only=False, machine_ref=None)
    if a.variant == 'r9':
        args.side_read = 'R9 (exp-194 r9_build.py, D169): mix12 on member-scope rows of the primary frame'
        orig = P.build_frames

        def build_frames_r9(args_, R, K, GS, pack, off, sources, injected=None):
            def mlp_choice_r9(m8, m12, scope, cls):
                scope = np.asarray(scope, bool)
                released, out = np.where(scope & (np.asarray(cls) == 'clock'), m12, m8), np.where(scope, m12, m8)
                ch = out != released
                CAP['ranking_rule'] = {'scope_rows': int(scope.sum()), 'changed_rows': int(ch.sum()),
                                       'changed_clock_rows': int((ch & (np.asarray(cls) == 'clock')).sum()),
                                       'rms_move_s': float(np.sqrt(np.mean((out - released)[ch] ** 2))) if ch.any() else 0.0}
                return out
            K.mlp_choice = mlp_choice_r9
            frames, digests, provenance, counts = orig(args_, R, K, GS, pack, off, sources, injected)
            CAP['fold_rule'] = fold_swap(frames, digests, provenance, R)
            counts['r9'] = {'fold': CAP['fold_rule'], 'ranking': CAP['ranking_rule']}
            return frames, digests, provenance, counts
        P.build_frames = build_frames_r9
    orig_held_g = P.held_g

    def held_g_keep(y, month, weights, chains):
        CAP['z'] = [(name, np.asarray(v, float).copy()) for name, v in chains['Z']]
        return orig_held_g(y, month, weights, chains)
    P.held_g = held_g_keep
    orig_post = P.post_steps

    def post_steps_keep(args_, D, off, contexts, *x, **k):
        CAP['ids'] = contexts['fold']['b'][ID].to_numpy('int64').copy()
        return orig_post(args_, D, off, contexts, *x, **k)
    P.post_steps = post_steps_keep
    rec = P.run(args)
    names = [n for n, _ in CAP['z']]
    assert len(set(names)) == len(names) and all(len(v) == len(CAP['ids']) for _, v in CAP['z'])
    a.fold_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.fold_out, ids=CAP['ids'], names=np.array(names), **{f's{i:02d}': v for i, (_, v) in
                                                                               enumerate(CAP['z'])})
    info = {'variant': a.variant, 'fold_rule': CAP.get('fold_rule'), 'ranking_rule': CAP.get('ranking_rule'),
            'hashes': rec['hashes'],
            'pack': rec.get('pack'), 'fold_stages': names, 'fold_out': str(a.fold_out), 'fold_out_sha256': sha(a.fold_out),
            'code': {HERE.name: sha(HERE), 'assemble_r9.py': sha(HERE.parent / 'assemble_r9.py'),
                     'assemble192.py': sha(P.__file__)}}
    print('R9BUILD', json.dumps(info, default=str))


if __name__ == '__main__':
    main()
