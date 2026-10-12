"""SPEC-021: explicit, hash-bound HB1 -> HB3u rebase with fixed blend weights."""
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diverse_blend import load_member, vector_digest
from joint_blend import JointBlend

ID = 'MVT_ID_mvt'
ROOT = Path(__file__).resolve().parents[3]
PARENT = '184dd8ecd3a87d06767dde85b7524112eb61b1b43ac2e7a35ed3e6d3b73263ee'
PACK = '72ca9e4e90a83428b2cd3221019d65e4b383b552beb569d4f362157c63a641a8'
SPEC = '5293163f2ed05ef15b8680b079616a6f6c6022b063fc8c3f81641efb45578b56'
JOINT = 'b2b1996b7299870fd7c3c69a22433b87dd1be0d0a7f72d73b91435986229f28d'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(rec):
    p = Path(rec['path'])
    if not p.is_absolute(): p = ROOT / p
    assert sha(p) == rec['sha256'], f'Head-swap input changed: {p}'
    return p


def fingerprint(obj):
    """Include every saved member value, coefficient, scope and old base receipt."""
    h = hashlib.sha256()
    def add(x):
        if is_dataclass(x):
            h.update(type(x).__name__.encode())
            for f in fields(x): h.update(f.name.encode()); add(getattr(x, f.name))
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype.str, x.shape)).encode()); h.update(x.tobytes())
        else: h.update(repr(x).encode())
    add(obj)
    return h.hexdigest()


def _rebase(parent, ids, old, new):
    """Internal kernel; caller must prove the exact, authorized head substitution."""
    parent.apply(ids, old, enable_extension=True)  # all original checks, before rebasing
    digest = vector_digest(ids, new)
    fallback = replace(parent.fallback, base_digest=digest)
    return replace(parent, base_digest=digest, fallback=fallback)


def aligned_head(pack_path, receipt_path, raw_path, raw_receipt_path, bundle, split):
    pack = json.loads(pack_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    rawrec = json.loads(raw_receipt_path.read_text())
    name = {'fold':'jjrefit', 'ranking':'jjrank'}[split]
    assert sha(pack_path) == PACK and pack['stage'] == 'HB3u'
    assert receipt['stage'] == rawrec['stage'] == 'HB3u'
    assert receipt['owner'] == rawrec['owner'] == name
    assert receipt['train_rows'] == rawrec['train_rows'] == 934128
    assert receipt['split']['token'] == pack['splits'][name]['token']
    assert receipt['train_split']['name'] == 'jjrank'
    assert set(int(m[-2:]) for m in receipt['train_rows_by_month']) == {2,3,4,5,6,8,9,10,11,12}
    assert set(receipt['train_split']['excluded']) == set(rawrec['split']['excluded'])
    assert rawrec['fit']['seeds'] == list(range(601,611))
    assert set(rawrec['merged_from']) == {'HU18','HU30s2'}
    assert [rawrec['fit']['params_by_part'][p]['min_child_samples'] for p in ['HU18','HU30s2']] == [100,30]
    p = receipt_path.with_suffix('.parquet')
    assert sha(p) == receipt['output_sha256'] == pack['files'][name]['parquet']
    assert sha(receipt_path) == pack['files'][name]['json']
    assert sha(raw_path) == rawrec['output_sha256']
    table = pd.read_parquet(p).set_index(ID); raw = pd.read_parquet(raw_path).set_index(ID)
    assert table.index.is_unique and raw.index.is_unique
    assert len(table) == pack['splits'][name]['rows']
    assert set(table.index) <= set(bundle[ID]) and set(raw.index) <= set(table.index)
    np.testing.assert_array_equal(table.pred_HB3u.reindex(raw.index), raw.pred_HB3u)
    assert table.pred_HB3u.notna().sum() == len(raw) == pack['splits'][name]['class_rows']
    head = table.pred_HB3u.reindex(bundle[ID]).to_numpy(float)
    on = (bundle.rule | bundle.extension).to_numpy(bool)
    missing = bundle.loc[on & ~np.isfinite(head), ID].tolist()
    assert not missing, f'Active HB3u values missing: {missing}'
    assert not np.isfinite(head[bundle.cls.ne('clock').to_numpy()]).any()
    absent = ~bundle[ID].isin(table.index)
    assert bundle.cls[absent].eq('noac').all()
    return head


@dataclass(frozen=True)
class HeadSwap:
    ids: np.ndarray
    head: np.ndarray
    old_base: np.ndarray
    new_base: np.ndarray
    parent_fingerprint: str
    scope: np.ndarray
    bundle: pd.DataFrame

    def bases(self, bundle, route):
        from assembly import head_blend
        pd.testing.assert_frame_equal(bundle, self.bundle, check_exact=True)
        np.testing.assert_array_equal(route, bundle.R4mk.to_numpy(float))
        old = head_blend(route, bundle.HB1.to_numpy(float), self.scope, .5)
        new = head_blend(route, self.head, self.scope, .5)
        np.testing.assert_array_equal(old, self.old_base)
        np.testing.assert_array_equal(new, self.new_base)
        np.testing.assert_array_equal(new[~self.scope], old[~self.scope])
        return new

    def blend(self, parent, ids, base, enable_extension):
        assert isinstance(parent, JointBlend)
        assert fingerprint(parent) == self.parent_fingerprint, 'Head swap changed a member or weight.'
        np.testing.assert_array_equal(ids, self.ids)
        np.testing.assert_array_equal(base, self.new_base)
        rebound = _rebase(parent, ids, self.old_base, base)
        return rebound.apply(ids, base, enable_extension=enable_extension)


def load_head_swap(path, expected_sha256, bundle, parent, split):
    from assembly import head_blend
    assert split in ('fold','ranking')
    path = Path(path); assert sha(path) == expected_sha256, 'Head-swap manifest changed.'
    rec = json.loads(path.read_text())
    assert rec['schema'] == 1 and rec['change'] == 'HB1_to_HB3u'
    assert rec['weight_policy'] == 'inherit_stack5_exact' and rec['head_weight'] == .5
    assert rec['route'] == 'R4mk' and rec['extension'] is True
    assert rec['parent']['sha256'] == PARENT; checked(rec['parent'])
    assert rec['spec']['sha256'] == SPEC; checked(rec['spec'])
    spec_review = checked(rec['plan_review'])
    assert 'VERDICT: PASS' in spec_review.read_text()
    review = checked(rec['source_review']); assert 'VERDICT: PASS' in review.read_text()
    approval = json.loads(checked(rec['desk_approval']).read_text())
    assert approval['approved'] is True and approval['spec_sha256'] == SPEC
    assert approval['source_review_sha256'] == rec['source_review']['sha256']
    assert rec['joint']['sha256'] == JOINT
    joint = checked(rec['joint'])
    want_bundle = pd.read_parquet(checked(rec['splits'][split]['bundle']))
    pd.testing.assert_frame_equal(bundle, want_bundle, check_exact=True)
    original = load_member(joint, JOINT, bundle, split)
    assert isinstance(parent, JointBlend) and fingerprint(parent) == fingerprint(original)
    r = rec['splits'][split]
    assert r['source_split'] == {'fold':'jjrefit','ranking':'jjrank'}[split]
    head = aligned_head(checked(rec['pack']), checked(r['receipt']), checked(r['raw_vector']), checked(r['raw_receipt']), bundle, split)
    checked(r['vector'])
    ids = bundle[ID].to_numpy('int64'); mask = (bundle.rule | bundle.extension).to_numpy(bool)
    assert vector_digest(ids, mask) == r['mask_digest']
    old = head_blend(bundle.R4mk.to_numpy(float), bundle.HB1.to_numpy(float), mask, .5)
    new = head_blend(bundle.R4mk.to_numpy(float), head, mask, .5)
    assert vector_digest(ids, old) == r['old_base_digest'] == parent.base_digest
    assert vector_digest(ids, new) == r['new_base_digest']
    assert fingerprint(parent) == r['member_fingerprint']
    # Original normal equations and member-base hashes remain the old fit's evidence.
    parent.apply(ids, old, enable_extension=True)
    return HeadSwap(ids, head, old, new, fingerprint(parent), mask, want_bundle)
