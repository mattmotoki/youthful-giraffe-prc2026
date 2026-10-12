"""Unit032: manifest-driven development replay and final-member refit harness."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from assembly import ID, COL, head_blend, compose, clip
from diverse_blend import vector_digest
from pair_blend import PairedBlend
from route_swap import fit_pair, rebase_pair
from lane_c_swap import NAMES, make_transition, matrix_apply
from stack7 import fit as fit_primary, independent_tail
from exploit_rules import prepare_rule
from exploit029_rules import apply as exploit_tail
from final_months169 import final as final_mode, serve_months, n_rows

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parents[1]
INPUTS = E / 'inputs/final-harness-032'
OUTPUTS = E / 'results/final-harness-032'
REFERENCE = INPUTS / 'reference.json'
ROLES = {'g': 'g', 'mlp': 'mlp', 'sequence021': 'sequence021',
         'sequence039': 'sequence_new', 'row6': 'control'}
MEMBERS = [*ROLES, 'sequence011', 'row011']
EXPECTED = {
    'fold': '22b71023e6729b117c2508d6d7146e8873a664632da49f164f1f42e12a1be54a',
    'final': 'e6dfedd5f7ae33081decd1f4c7314480f27d57ec9de1ab4ffc417e0fd22c3262',
    'shift': '8703a7d578b4aeae1c9656db7d2957ed5f03a098d1178b1d7fe436c508f006f5',
}


def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p): return json.loads(Path(p).read_text())
def record(p):
    p = Path(p).resolve()
    return {'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), 'sha256': sha(p)}

def write(p, obj):
    p = Path(p)
    assert not p.exists(), f'Output already exists: {p}'
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def table(p, q):
    p = Path(p)
    assert not p.exists(), f'Output already exists: {p}'
    q.to_parquet(p, index=False)
    return record(p)


class Reader:
    def __init__(self): self.inputs = {}; self.frames = {}

    def path(self, rec):
        p = Path(rec['path']); p = p if p.is_absolute() else ROOT / p
        p = p.resolve()
        assert p.is_file() and sha(p) == rec['sha256'], f'Input hash mismatch: {p}'
        key = str(p)
        if key in self.inputs: assert self.inputs[key] == rec['sha256']
        self.inputs[key] = rec['sha256']
        return p

    def frame(self, rec):
        p = self.path(rec)
        if str(p) not in self.frames: self.frames[str(p)] = pd.read_parquet(p)
        return self.frames[str(p)]

    def aligned(self, rec, b, required, *, boolean=False, year=None, floor=True):
        q = self.frame(rec); idcol = rec.get('id_column', ID); col = rec['column']
        ids = q[idcol].to_numpy()
        assert len(np.unique(ids)) == len(ids) and np.isfinite(ids).all()
        assert np.equal(ids, ids.astype('int64')).all(), 'Non-integral source ID.'
        assert col in q and col not in ['y', 'target'], 'Invalid prediction column.'
        indexed = q.set_index(idcol); v = indexed[col].reindex(b[ID])
        for source, receiver in [('month', 'month'), ('mon', 'month'), ('airport', 'airport'),
                                 ('ADEP_mvt', 'airport'), ('cls', 'cls')]:
            if source in indexed:
                common = b[ID].isin(indexed.index).to_numpy()
                metadata = indexed[source].reindex(b[ID]).to_numpy()[common]
                if receiver == 'month' and metadata.dtype.kind in 'OU':
                    text = pd.Series(metadata).astype(str)
                    if text.str.fullmatch(r'\d{4}-\d{2}').all():
                        if year is not None: assert text.str[:4].astype(int).eq(year).all(), 'Wrong source year.'
                        metadata = text.str[-2:].astype(int).to_numpy()
                np.testing.assert_array_equal(metadata, b[receiver].to_numpy()[common])
        if boolean:
            assert v.notna().all() and indexed[col].dtype == np.dtype(bool)
            return v.to_numpy(bool)
        values = v.to_numpy(float)
        assert values.shape == required.shape and np.isfinite(values[required]).all(), 'Missing active prediction.'
        if floor: assert (values[required] >= 60).all(), f'Prediction below60: {rec["path"]}:{col}'
        return values

    def unchanged(self):
        for p, h in self.inputs.items(): assert sha(p) == h, f'Input changed during assembly: {p}'


class Timer:
    def __init__(self): self.stages = []
    def run(self, name, fn):
        started = now(); t = time.perf_counter(); value = fn()
        self.stages.append({'stage': name, 'started_at_utc': started,
                            'ended_at_utc': now(), 'seconds': time.perf_counter() - t})
        print('STAGE', name, f'{self.stages[-1]["seconds"]:.3f}s', flush=True)
        return value


def dependencies(manifest):
    return {role: manifest['vectors']['fold'][role] for role in [*MEMBERS, 'route', 'head']}


def weight_action(manifest, ref, mode):
    assert mode in ['auto', 'refit']
    old = dependencies(ref); new = dependencies(manifest)
    changed = [key for key in old if new[key] != old[key]]
    return {'refit': bool(changed) or mode == 'refit', 'mode': mode,
            'changed_held_dependencies': changed,
            'reason': 'held input descriptor changed' if changed else 'explicit unchanged-input refit' if mode == 'refit' else 'held inputs exact; recorded weights reused'}


def validate_manifest(path, rd):
    m = js(rd.path(record(path))); ref = js(rd.path(record(REFERENCE)))
    assert m['schema'] == ref['schema'] == 1 and m['reference'] == record(REFERENCE)
    assert set(m['vectors']) == {'fold', 'ranking'}
    assert m['common'] == ref['common'], 'Frozen recipes, scopes or rule records changed.'
    assert m['baseline_inputs'] == ref['baseline_inputs']
    assert m['recipe'] == ref['recipe'] == 'stack12_other_month_then_pooled_029_no_bets'
    for split in ['fold', 'ranking']:
        assert set(m['vectors'][split]) == set(MEMBERS) | {'route', 'head'}
        for rec in m['vectors'][split].values(): rd.path(rec)
    def pins(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value: rd.path(value)
            else:
                for item in value.values(): pins(item)
        elif isinstance(value, list):
            for item in value: pins(item)
    pins(m['common']); pins(m['baseline_inputs']); pins(m.get('evidence', []))
    for p, h in ref['protected'].items(): rd.path({'path': p, 'sha256': h})
    return m, ref


def load(manifest, rd):
    common = manifest['common']; contexts = {}
    for split in ['fold', 'ranking']:
        b = rd.frame(common['bundle'][split]).copy()
        months = serve_months(split)
        assert b[ID].is_unique and b.month.isin(months).all()
        assert months == [1, 7] or sorted(set(b.month.tolist())) == months, 'Final mode: every rank month needs rows.'
        assert b.cls.isin(['clock', 'ordinary', 'noac']).all()
        f = rd.frame(common['primary_frame'][split]).copy()
        pair_source = rd.frame(common['pair_frame'][split])
        np.testing.assert_array_equal(f[ID], b[ID]); np.testing.assert_array_equal(pair_source[ID], b[ID])
        scope = f.member_scope.to_numpy(bool); ordinary = b.cls.eq('ordinary').to_numpy()
        pair_scope = pair_source.member_scope.to_numpy(bool); control_scope = pair_source.control_scope.to_numpy(bool)
        np.testing.assert_array_equal(scope, pair_scope & ~b.extension.to_numpy(bool))
        np.testing.assert_array_equal(control_scope, pair_scope & ordinary)
        assert not (pair_scope & b.cls.eq('noac').to_numpy()).any()
        masks = {role: scope & ordinary if role == 'row6' else scope for role in ROLES}
        masks.update(sequence011=pair_scope, row011=control_scope,
                     route=np.ones(len(b), bool), head=(b.rule | b.extension).to_numpy(bool))
        vals = {role: rd.aligned(rec, b, masks[role], year=2025 if split == 'fold' else 2026, floor=role != 'head') for role, rec in manifest['vectors'][split].items()}
        b['R4mk'] = vals['route']; base = head_blend(vals['route'], vals['head'], masks['head'], .5)
        for role, column in ROLES.items(): f.loc[masks[role], column] = vals[role][masks[role]]
        proxy = js(rd.path(common['extension_proxy']))
        gains = []
        for keys in common['extension_gain_paths']:
            value = proxy
            for key in keys: value = value[key]
            assert np.isfinite(value) and value > 0
            gains.append(float(value))
        pair = PairedBlend(b[ID].to_numpy('int64'), vals['sequence011'], vals['row011'],
                           pair_scope, control_scope, np.zeros(len(b)), np.zeros(len(b)),
                           vector_digest(b[ID], base), b.extension.to_numpy(bool), min(gains), split)
        # The receiver fixes scopes; new vectors cannot supply masks or metadata used in fitting.
        contexts[split] = {'b': b, 'frame': f, 'pair': pair, 'base': base,
                           'scope': scope, 'ordinary': ordinary, 'masks': masks}
    return contexts


def base_tail(c):
    # Equivalent to the frozen no-member fit target after the HB3u base.
    b = c['b'].copy(); b['R4mk'] = c['base']
    return compose(b, route='R4mk', form='F', extension=False, noplan='v3cal',
                   practice='selected', c33=True, c55=False)


def fit_weights(m, contexts, rd, action):
    common = m['common']; pair_record = js(rd.path(common['pair_weights']))
    primary_record = js(rd.path(common['primary_weights']))
    weights = {'pair_fold': pair_record['pair_fold_weights'], 'pair_ranking': pair_record['pair_ranking_weights'],
               'primary_fold': primary_record['fold_weights'], 'primary_ranking': primary_record['ranking_weights']}
    if action['refit']:
        c = contexts['fold']; frame = rd.frame(common['rule_frame']['fold'])
        np.testing.assert_array_equal(frame[ID], c['b'][ID]); y = frame.y.to_numpy(float)
        exposure = rd.frame(common['exposure']); np.testing.assert_array_equal(exposure[ID], c['b'][ID])
        w = exposure.weight.to_numpy(float); assert np.isfinite(y).all() and np.isfinite(w).all() and (w >= 0).all()
        B = base_tail(c)
        pf, pp = fit_pair(c['b'], c['pair'], c['base'], B, y, w)
        fw, pw = fit_primary(c['b'], c['frame'], c['base'], B, y, w)
        weights = {'pair_fold': pf, 'pair_ranking': pp, 'primary_fold': fw, 'primary_ranking': pw}
    return weights


def pre_stack(c, weights, split):
    b = c['b']; pair = rebase_pair(c['pair'], b, c['base'], weights['pair_fold'], weights['pair_ranking'], split)
    # make_transition uses only ids/scope/ordinary from its parent and validates normal equations.
    parent = type('Scope', (), {'ids': b[ID].to_numpy('int64'), 'scope': c['scope'], 'ordinary': c['ordinary']})()
    transition = make_transition(None, parent, c['frame'], weights['primary_fold'], weights['primary_ranking'], split)
    prior = pair.apply(b[ID].to_numpy(), c['base'], enable_extension=True)
    blended = matrix_apply(c['base'], prior, transition.values, transition.weights, c['scope'], c['ordinary'])
    pre = independent_tail(b, blended)
    np.testing.assert_array_equal(pre, clip(pre, b.lo.to_numpy(), b.hi.to_numpy()))
    return {'route': b.R4mk.to_numpy(), 'head': c['base'], 'pair': prior,
            'primary': blended, 'pre_exploit': pre}, pair, transition


def rules(m, contexts, stages, rd):
    common = m['common']; train = rd.frame(common['rule_frame']['fold']); result = {}; statistics = {}
    np.testing.assert_array_equal(train[ID], contexts['fold']['b'][ID])
    for split in ['fold', 'ranking']:
        c = contexts[split]; test = rd.frame(common['rule_frame'][split])
        np.testing.assert_array_equal(test[ID], c['b'][ID])
        obj, stats, _, _ = prepare_rule(train, stages['fold']['pre_exploit'], test,
                                        stages[split]['pre_exploit'], c['b'], pooled=split == 'ranking')
        steps = obj.apply(c['b'][ID].to_numpy(), stages[split]['pre_exploit'])
        stages[split].update({key: steps[key] for key in ['C71', 'C95', 'C98']})
        stages[split]['stack12'] = steps['stack9']
        # Independent fixed sequence, including C71 lower projection.
        z = np.clip(np.maximum(60., np.maximum(60., np.where(obj.c71, np.maximum(stages[split]['pre_exploit'], obj.g71), stages[split]['pre_exploit']) + obj.step95) + obj.step98), obj.lower, obj.upper)
        np.testing.assert_array_equal(z, stages[split]['stack12'])
        statistics[split] = stats; result[split] = obj
    return statistics, result


def tail(m, contexts, stages, rd):
    for split in ['fold', 'ranking']:
        c = contexts[split]; f = rd.frame(m['common']['exploit_frame'][split])
        np.testing.assert_array_equal(f[ID], c['b'][ID]); out = exploit_tail(f, stages[split]['stack12'], include67=False)
        stages[split].update({k: out[k] for k in ['C65', 'C66', 'C58']}); stages[split]['final'] = out['final']
        z = stages[split]['stack12'] + f.order_delta.to_numpy()
        mask = f.C66.to_numpy(bool); z[mask] = f.point66.to_numpy()[mask]
        mask = f.C58.to_numpy(bool); z[mask] += .5 * (f.point58.to_numpy()[mask] - z[mask])
        np.testing.assert_array_equal(np.clip(z, f.lower, f.upper), out['final'])
        np.testing.assert_array_equal(out['final'], np.clip(out['final'], f.lower, f.upper))
        np.testing.assert_array_equal(out['final'], clip(out['final'], c['b'].lo.to_numpy(), c['b'].hi.to_numpy()))
        assert np.isfinite(out['final']).all() and (out['final'] >= 60).all()


def distance(b, a, z):
    delta = np.asarray(z) - np.asarray(a); n = len(b)
    def view(mask):
        d = delta[mask]
        return {'rows': int(mask.sum()), 'changed': int((d != 0).sum()),
                'D2': float(np.mean(d*d)), 'RMS': float(np.sqrt(np.mean(d*d))),
                'mean': float(d.mean()), 'max_abs': float(np.max(np.abs(d))),
                'contribution_to_full_D2': float(np.sum(d*d)/n)}
    result = {'all': view(np.ones(n, bool))}
    for key in ['month', 'cls', 'airport']:
        result[key] = {str(v): view(b[key].eq(v).to_numpy()) for v in sorted(b[key].unique())}
    result['airport_month'] = {f'{ap}-{mon:02}': view(b.airport.eq(ap).to_numpy() & b.month.eq(mon).to_numpy())
                               for ap in sorted(b.airport.unique())
                               for mon in (sorted(set(b.month.tolist())) if final_mode() else [1, 7])}
    return result


def serialize(c, final, rd, template_rec):
    ids = rd.frame(template_rec)[ID]
    np.testing.assert_array_equal(c['b'][ID], ids)
    q = pd.DataFrame({ID: ids.to_numpy('float64'), COL: np.asarray(final, 'float64')})
    assert len(q) == n_rows() and q[ID].is_unique and np.isfinite(q.to_numpy()).all() and q[COL].min() >= 60
    buffer = io.BytesIO(); q.to_parquet(buffer, index=False)
    return q, buffer.getvalue()


def run(path, out, mode='auto', dry_run=False):
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
        assert int(os.environ.get(key, '1')) <= 2, 'Mac numerical thread cap is2.'
    out = Path(out).resolve(); assert out.is_relative_to(E) and not out.exists(), 'Output needs a new exp-169 directory.'
    timer = Timer(); rd = Reader(); started = now()
    m, ref = timer.run('validate_inputs', lambda: validate_manifest(path, rd))
    action = weight_action(m, ref, mode)
    contexts = timer.run('load_vectors', lambda: load(m, rd))
    weights = timer.run('fit_weights' if action['refit'] else 'load_recorded_weights', lambda: fit_weights(m, contexts, rd, action))
    stages = {}; pairs = {}; transitions = {}
    for split in ['fold', 'ranking']:
        stages[split], pairs[split], transitions[split] = timer.run('member_chain_' + split, lambda split=split: pre_stack(contexts[split], weights, split))
    statistics, rule_objects = timer.run('refit_C71_C95_C98', lambda: rules(m, contexts, stages, rd))
    timer.run('exploit029_one_clip', lambda: tail(m, contexts, stages, rd))
    q, body = timer.run('byte_checks_and_serialization', lambda: serialize(contexts['ranking'], stages['ranking']['final'], rd, m['common']['template']))
    comparisons = {}
    for name, rec in m['common']['comparisons'].items():
        old = rd.frame(rec); np.testing.assert_array_equal(old[ID], q[ID])
        comparisons[name] = distance(contexts['ranking']['b'], old[COL], q[COL])
    stage_distances = {}
    for split in stages:
        stage_distances[split] = {}; previous = stages[split]['route']
        for key, values in stages[split].items():
            stage_distances[split][key] = distance(contexts[split]['b'], previous, values); previous = values
    timer.run('final_input_hash_check', rd.unchanged)
    out.mkdir(parents=True)
    outputs = {'weights': table(out/'weights-held.parquet', pd.DataFrame({ID: contexts['fold']['b'][ID],
                'pair_sequence': pairs['fold'].sequence_weight, 'pair_control': pairs['fold'].control_weight,
                **{'primary_' + k: transitions['fold'].weights[:, j] for j, k in enumerate(NAMES)}}))}
    write(out/'weights.json', weights); outputs['weight_records'] = record(out/'weights.json')
    write(out/'statistics.json', statistics); outputs['statistics'] = record(out/'statistics.json')
    if not dry_run:
        (out/'candidate.parquet').write_bytes(body); outputs['candidate'] = record(out/'candidate.parquet')
        assert pd.read_parquet(out/'candidate.parquet').equals(q)
    receipt = {'schema': 1, 'development_only': True, 'upload_allowed': False, 'D123': 'desk for a future final candidate',
               'manifest': record(path), 'reference': record(REFERENCE), 'input_sha256': rd.inputs,
               'outputs': outputs, 'candidate_written': not dry_run, 'candidate_sha256': hashlib.sha256(body).hexdigest(),
               'held_prediction_digest': vector_digest(contexts['fold']['b'][ID], stages['fold']['final']),
               'ranking_prediction_digest': vector_digest(q[ID], q[COL]), 'weight_action': action,
               'comparisons': comparisons, 'stage_distances': stage_distances, 'stages': timer.stages,
               'started_at_utc': started, 'ended_at_utc': now(), 'label_source': m['common']['rule_frame']['fold'],
               'annual_label_used': False, 'selection': m.get('description', ''),
               'HEAD_at_receipt': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'harness_code': record(__file__)}
    write(out/'RECEIPT.json', receipt)
    print('DONE', receipt['candidate_sha256'], 'refit', action['refit'], 'dry_run', dry_run, flush=True)
    return receipt, weights


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'dry-run'])
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--weights', choices=['auto', 'refit'], default='auto'); args = p.parse_args()
    run(args.manifest, args.out, args.weights, args.command == 'dry-run')
