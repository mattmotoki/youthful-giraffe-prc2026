"""Load an admitted lane A-C member and weights selected from 2025 folds only."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ID = 'MVT_ID_mvt'


def vector_digest(ids, values):
    ids = np.asarray(ids)
    assert np.isfinite(ids).all() and np.equal(ids, ids.astype('int64')).all()
    h = hashlib.sha256(b'exp169-base-v1\0')
    h.update(ids.astype('<i8').tobytes())
    h.update(np.asarray(values, dtype='<f8').tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class MemberBlend:
    values: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    base_digest: str
    member: str
    ids: np.ndarray

    def apply(self, ids, base):
        assert np.array_equal(np.asarray(ids), self.ids), 'Member arrays belong to a different ID order.'
        assert np.isfinite(base).all() and (base >= 60).all(), 'Blend base is not a served vector above the floor.'
        assert vector_digest(ids, base) == self.base_digest, 'Blend weights belong to a different base.'
        assert self.values.shape == self.mask.shape == self.weights.shape == base.shape
        assert self.mask.dtype == np.dtype(bool)
        assert np.isfinite(self.weights).all() and ((0 <= self.weights) & (self.weights <= 1)).all()
        assert np.isfinite(self.values[self.mask]).all() and (self.values[self.mask] >= 60).all()
        active = self.mask & (self.weights > 0)
        out = base.copy()
        out[active] = (1-self.weights[active])*base[active] + self.weights[active]*self.values[active]
        assert np.array_equal(out[~active], base[~active])
        return out


def load_member(path, expected_sha256, bundle, split):
    """Manifest contract is in BLEND-CONTRACT.md; this function does not select weights."""
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, 'Blend manifest changed.'
    spec = json.loads(path.read_text())
    if spec['schema'] == 3:
        from pair_blend import load_pair
        return load_pair(path,spec,bundle,split)
    if spec['schema'] in (4,5):
        from joint_blend import load_joint
        return load_joint(path,spec,bundle,split)
    assert spec['schema'] in (1,2) and spec['lane'] in ['A','B','C']
    assert split in ('fold','ranking')
    assert spec['weight_basis'] == 'two_fold_2025' and spec['serving_point'] == 'after_head'
    assert spec['member_training_months']['fold'] == [2,3,4,5,6,8,9,10,11,12]
    assert spec['member_training_months']['ranking'] == list(range(1,13))
    if spec['schema'] == 2:
        # Class and apply-month are fixed input fields, never inferred from a score.
        assert bundle.month.isin([1,7]).all()
        assert bundle.cls.isin(['ordinary','clock','noac']).all()
        assert set(spec['fold_weights']) == {'ordinary','clock','noac'}
        weights = np.zeros(len(bundle),dtype=float)
        for cls,folds in spec['fold_weights'].items():
            for m,other in [(1,7),(7,1)]:
                r = folds[str(m)]
                assert r['trained_on_months'] == [other], 'A held weight used its own fold.'
                assert np.isfinite(r['weight']) and 0 <= r['weight'] <= 1
                if cls == 'noac':
                    assert r['weight'] == 0, 'C49 keeps no-aircraft rows.'
                weights[bundle.cls.eq(cls).to_numpy() & bundle.month.eq(m).to_numpy()] = r['weight']
        if split == 'ranking':
            r = spec['ranking_weight']
            assert r['method'] == 'same_as_held_month'
            assert r['trained_on_months'] == [1,7]
    else:
        folds = spec['fold_weights']
        for m, other in [(1,7),(7,1)]:
            r = folds[str(m)]
            assert r['trained_on_months'] == [other], 'A held weight used its own fold.'
            assert np.isfinite(r['weight']) and 0 <= r['weight'] <= 1
        if split == 'fold':
            assert bundle.month.isin([1,7]).all()
            weights = bundle.month.map({m:folds[str(m)]['weight'] for m in [1,7]}).to_numpy(float)
        else:
            r = spec['ranking_weight']
            assert r['trained_on_months'] == [1,7]
            if r['method'] == 'mean_other_fold':
                expected = (folds['1']['weight'] + folds['7']['weight'])/2
            else:
                assert r['method'] == 'pooled_folds'
                # Sufficient statistics must come from the admitted fold-weight record.
                assert np.isfinite(r['numerator']) and np.isfinite(r['denominator']) and r['denominator'] > 0
                expected = np.clip(r['numerator']/r['denominator'],0,1)
            assert np.isclose(r['weight'],expected,rtol=0,atol=1e-12), 'Ranking weight differs from its fold rule.'
            weights = np.full(len(bundle),r['weight'],dtype=float)
    rec = spec['vectors'][split]
    source = Path(rec['path'])
    if not source.is_absolute():
        source = path.parent/source
    assert hashlib.sha256(source.read_bytes()).hexdigest() == rec['sha256'], 'Member vector changed.'
    frame = pd.read_parquet(source)
    assert frame[ID].is_unique and len(frame) == len(bundle)
    assert np.isfinite(frame[ID]).all() and (frame[ID] == frame[ID].astype('int64')).all()
    assert set(frame[ID]) == set(bundle[ID]), 'Member IDs differ from the bundle.'
    frame = frame.set_index(ID).reindex(bundle[ID])
    assert frame[rec['mask_column']].dtype == np.dtype(bool)
    if spec['schema'] == 2:
        assert np.array_equal(frame[rec['class_column']],bundle.cls), 'Member classes differ from the bundle.'
        assert np.array_equal(frame[rec['month_column']],bundle.month), 'Member months differ from the bundle.'
    return MemberBlend(frame[rec['value_column']].to_numpy(float),
                       frame[rec['mask_column']].to_numpy(bool), weights,
                       rec['base_digest'], spec['member'], bundle[ID].to_numpy('int64').copy())
