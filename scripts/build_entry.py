"""Replay a registered entry from the export and caller-supplied official inputs.

    python scripts/build_entry.py --target v40 --data-root DIR --out-dir NEW --record NEW.json

The default target is release/ENTRIES.json's default_target. Saved values contain
no organizer IDs or labels. A new entry needs a registered replay recipe.
Exact bytes need pandas 2.3.3 and pyarrow 24.0.0. No member fit runs; the
"assembly" recipe (target zabd) refits only the level-2 weights and rule
statistics from saved member values and the caller's 2025 labels. The
"composition" recipe (target v64) replays its base entry (zabd) and then applies
a fixed stage with no fit (release/t11b_stage.py). The "steps" recipe (target v73)
runs the code of iso/ and w2/ on the v64 chain (release/v73_stage.py): it needs the
14 official files, refits the step models from the 2025 labels and runs no member
fit; its bytes are recorded on x86-64 Linux.
An entry with "mode": "final" reads the organizers' final files (final_ranking.parquet,
final_submitting.parquet); every entry in its chain has the same mode, this script sets
PRC_RANKING_MODE from it before any release module is imported, and no entry of a
final-mode chain (the target and each base entry) can use the published-mean-shift recipe.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import build_v32 as legacy

ROOT = Path(__file__).resolve().parents[1]


def registry():
    return json.loads((ROOT / 'release/ENTRIES.json').read_text())


def target_record(target=None):
    book = registry()
    name = target or book['default_target']
    if name.isdigit():
        name = 'v' + name
    if name not in book['entries']:
        raise ValueError(f'Unknown target {name}; registered entries: {sorted(book["entries"])}')
    return name, book['entries'][name]


def chain_mode(name, book, seen=()):
    """The mode of an entry ('ranking' or 'final'); every entry it replays (its base chain) has the same mode."""
    spec = book['entries'][name]
    mode = spec.get('mode', 'ranking')
    if mode not in ('ranking', 'final'):
        raise ValueError(f'{name}: unknown mode {mode}')
    base = spec.get('base')
    if base is not None:
        if base in seen or base == name:
            raise ValueError("Cyclic entry composition")
        if chain_mode(base, book, (*seen, name)) != mode:
            raise SystemExit(f'{name} ({mode} mode) has a base entry of another mode')
    return mode


def replay(target, data_root, out_dir, record_path=None, overwrite=False, _ancestors=()):
    name, spec = target_record(target)
    if name in _ancestors:
        raise ValueError("Cyclic entry composition")
    mode = chain_mode(name, registry())
    if (os.environ.get('PRC_RANKING_MODE') or 'ranking') != mode:
        raise SystemExit(f'{name} is a {mode}-mode entry; PRC_RANKING_MODE must be {mode} (main() sets it)')
    if mode == 'final':
        # The steps recipe reads its base chain from saved values and never replays the base entries,
        # so the refusal covers every entry of the chain here.
        book, link, chain = registry(), name, []
        while link is not None:
            chain.append(link)
            link = book['entries'][link].get('base')
        shifted = [n for n in chain if book['entries'][n]['recipe'] == 'published-mean-shift']
        if shifted:
            raise SystemExit(f'{name}: a final-mode chain cannot use the published-mean-shift recipe (exp-151): '
                             f'{", ".join(shifted)}')
    out_dir, data_root = Path(out_dir), Path(data_root)
    record_path = Path(record_path) if record_path else None
    output = out_dir / (name + '-replay.parquet')
    legacy.refuse_existing([output, *([record_path] if record_path else [])], overwrite)
    if spec['recipe'] == 'v32-chain':
        # Keep the earlier checks and chain as the explicit historical target.
        import subprocess, sys
        args = [sys.executable, str(ROOT / 'scripts/build_v32.py'), '--data-root', str(data_root),
                '--out-dir', str(out_dir)]
        if record_path:
            args += ['--record', str(record_path)]
        if overwrite:
            args += ['--overwrite']
        subprocess.run(args, check=True)
        return output
    if spec['recipe'] == 'assembly':
        return assembly_replay(name, spec, data_root, output, record_path)
    if spec['recipe'] == 'composition':
        return composition_replay(name, spec, data_root, out_dir, output, record_path, overwrite,
                                  (*_ancestors, name))
    if spec['recipe'] == 'steps':
        return steps_replay(name, spec, data_root, out_dir, output, record_path)
    if spec['recipe'] not in ('saved-values', 'fixed-blend', 'published-mean-shift'):
        raise ValueError('Unsupported replay recipe: ' + spec['recipe'])
    book = registry()
    for filename, key in [('submitting', 'input_template_sha256'), ('ranking', 'ranking_sha256')]:
        path = data_root / 'data/raw' / (filename + '.parquet')
        if legacy.sha256(path) != book[key]:
            raise SystemExit('wrong ' + filename + '.parquet input hash')
    template = pd.read_parquet(data_root / 'data/raw/submitting.parquet')
    ids = template[legacy.ID].to_numpy(dtype=spec['id_dtype'])
    if len(ids) != spec['rows'] or not np.isfinite(ids).all() or not np.all(ids == np.round(ids)) or pd.Index(ids).has_duplicates:
        raise SystemExit('Template IDs are not the recorded unique integer population')
    components = None
    if spec['recipe'] == 'saved-values':
        values = legacy.saved_array(spec)
    elif spec['recipe'] == 'fixed-blend':
        if spec['a'] == name or spec['b'] == name or not 0.0 < spec['weight_on_a'] < 1.0:
            raise ValueError('Invalid fixed-blend target or weight')
        paths = {side: replay(spec[side], data_root, out_dir / (name + '-components') / spec[side],
                             overwrite=overwrite, _ancestors=(*_ancestors, name)) for side in ('a', 'b')}
        parts = {}
        for side, path in paths.items():
            f = pd.read_parquet(path)
            if not np.array_equal(f[legacy.ID].to_numpy('float64'), ids):
                raise SystemExit('Component IDs differ from the official template order')
            parts[side] = f[legacy.Y].to_numpy('float64')
        w = spec['weight_on_a']
        values = w * parts['a'] + (1 - w) * parts['b']
        components = {side: {'target': spec[side], 'sha256': legacy.sha256(path)}
                      for side, path in paths.items()}
    else:
        from published_shift import apply
        base_path = replay(spec['base'], data_root, out_dir / (name + '-components') / spec['base'],
                           overwrite=overwrite, _ancestors=(*_ancestors, name))
        base = pd.read_parquet(base_path)
        if not np.array_equal(base[legacy.ID].to_numpy('float64'), ids):
            raise SystemExit('Base IDs differ from the official template order')
        param_path = ROOT / spec['parameters']
        if legacy.sha256(param_path) != spec['parameters_sha256']:
            raise SystemExit('frozen parameter hash mismatch')
        params = json.loads(param_path.read_text())['parameters']
        support = legacy.saved_array(spec['support_vector'])
        ranking = pd.read_parquet(data_root / 'data/raw/ranking.parquet')
        meta = ranking.set_index(legacy.ID).loc[ids]
        values, cells = apply(meta['ADEP_mvt'].to_numpy(),
                              pd.to_datetime(meta['MVT_TIME_UTC_mvt'], utc=True).dt.month.to_numpy(),
                              base[legacy.Y].to_numpy('float64'), support, params)
        components = {'base': {'target': spec['base'], 'sha256': legacy.sha256(base_path)}}
    if len(values) != len(ids) or not np.isfinite(values).all() or values.min() < legacy.FLOOR_S:
        raise SystemExit('Saved entry has the wrong population or values below the serving floor')
    frame = pd.DataFrame({legacy.ID: ids, legacy.Y: values})
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    sha, vsha = legacy.sha256(output), legacy.value_sha256(pd.read_parquet(output))
    result = {'target': name, 'recipe': spec['recipe'], 'rows': len(frame), 'bytes': output.stat().st_size,
              'sha256': sha, 'value_sha256': vsha, 'expected_sha256': spec['sha256'],
              'expected_value_sha256': spec['value_sha256'], 'saved_vector_sha256': spec.get('public_vector_sha256'),
              'runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'pyarrow': __import__('pyarrow').__version__},
              'inputs': {key: book[key] for key in ('input_template_sha256', 'ranking_sha256')},
              'fit_ran': False}
    if components is not None:
        result['components'] = components
        if spec['recipe'] == 'fixed-blend':
            result['weight_on_a'] = spec['weight_on_a']
        else:
            result['parameters_sha256'] = spec['parameters_sha256']
            result['support_vector_sha256'] = spec['support_vector']['public_vector_sha256']
            result['cells'] = cells
            result['unsupported_rows_held'] = int(np.sum(support == 0))
    result['verdict'] = 'PASS' if (sha == spec['sha256'] and vsha == spec['value_sha256'] and result['bytes'] == spec['bytes']) else 'FAIL'
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['verdict'] != 'PASS':
        raise SystemExit('Entry bytes or values differ; use the exact recorded writer versions')
    return output


def assembly_replay(name, spec, data_root, output, record_path):
    """Rerun the exp-169 assembly: level-2 fits, rules and tail from saved member values."""
    import hashlib
    import sys
    sys.path.insert(0, str(ROOT / 'release'))
    import assembly_replay as replay_module
    pack_path = ROOT / spec['pack']
    if legacy.sha256(pack_path) != spec['pack_sha256']:
        raise SystemExit('assembly pack hash mismatch')
    pack = json.loads(pack_path.read_text())
    if (pack['candidate_sha256'] != spec['sha256'] or pack['parent_sha256'] != spec['parent_sha256']
            or pack.get('with_c65_sha256') != spec.get('with_c65_sha256')):
        raise SystemExit('assembly pack names another candidate')
    result = replay_module.assemble(pack, data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result['body'])
    frame = pd.read_parquet(output)
    if not frame.equals(result['table']):
        raise SystemExit('written entry differs from the assembled table')
    sha, vsha = legacy.sha256(output), legacy.value_sha256(frame)
    expected = legacy.saved_array(spec['comparison_vector'], spec['rows'])
    difference = float(np.max(np.abs(frame[legacy.Y].to_numpy('float64') - expected)))
    parent_sha = hashlib.sha256(result['parent_body']).hexdigest()
    with_c65 = result['with_c65_body']
    with_c65_sha = hashlib.sha256(with_c65).hexdigest() if with_c65 is not None else None
    record = {'target': name, 'recipe': 'assembly', 'rows': len(frame), 'bytes': output.stat().st_size,
              'sha256': sha, 'value_sha256': vsha, 'expected_sha256': spec['sha256'],
              'expected_value_sha256': spec['value_sha256'],
              'parent_sha256': parent_sha, 'expected_parent_sha256': spec['parent_sha256'],
              'with_c65_sha256': with_c65_sha, 'expected_with_c65_sha256': spec.get('with_c65_sha256'),
              'max_abs_difference_to_saved_entry_s': difference,
              'pack_sha256': spec['pack_sha256'], 'inputs': pack['official_sha256'],
              'frames_checked': len(pack['frames']), 'fitted': ['pair weights', 'primary weights',
              'dark18 primary weights', 'C95/C98 statistics'], 'member_fit_ran': False,
              'weights': result['weights'], 'rule_statistics': result['statistics'],
              'runtime': replay_module.runtime()}
    exact = (sha == spec['sha256'] and vsha == spec['value_sha256'] and record['bytes'] == spec['bytes']
             and parent_sha == spec['parent_sha256'] and with_c65_sha == spec.get('with_c65_sha256'))
    record['verdict'] = 'PASS' if exact else 'FAIL'
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('weights', 'rule_statistics')}, indent=2))
    if not exact:
        raise SystemExit('Entry bytes differ; max_abs_difference_to_saved_entry_s shows the size of the '
                         'difference (another BLAS can change the last bits of the level-2 fit)')
    return output


def composition_replay(name, spec, data_root, out_dir, output, record_path, overwrite, ancestors):
    """Replay the base entry, then apply the registered stage (T11b-L for v64); no fit runs."""
    import sys
    sys.path.insert(0, str(ROOT / 'release'))
    import t11b_stage as stage
    if spec['stage'] != 'T11b-L':
        raise ValueError('Unsupported composition stage: ' + spec['stage'])
    if target_record(spec['base'])[1]['sha256'] != spec['base_sha256']:
        raise ValueError('Composition base is not the registered base entry')
    components = out_dir / (name + '-components')
    base_record = components / (spec['base'] + '-record.json')
    base_path = replay(spec['base'], data_root, components / spec['base'], base_record, overwrite, ancestors)
    result = stage.compose(spec, base_path, data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result['body'])
    frame = pd.read_parquet(output)
    if not frame.equals(result['table']):
        raise SystemExit('written entry differs from the composed table')
    sha, vsha = legacy.sha256(output), legacy.value_sha256(frame)
    record = {'target': name, 'recipe': 'composition', 'stage': spec['stage'], 'rows': len(frame),
              'bytes': output.stat().st_size, 'sha256': sha, 'value_sha256': vsha,
              'expected_sha256': spec['sha256'], 'expected_value_sha256': spec['value_sha256'],
              'base': json.loads(base_record.read_text()), **result['facts'],
              'stage_fit_ran': False, 'stage_reads_label': False, 'runtime': stage.R.runtime()}
    exact = sha == spec['sha256'] and vsha == spec['value_sha256'] and record['bytes'] == spec['bytes']
    record['verdict'] = 'PASS' if exact and record['base']['verdict'] == 'PASS' else 'FAIL'
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'base'}, indent=2))
    if record['verdict'] != 'PASS':
        raise SystemExit('Entry bytes differ after the composition stage')
    return output


#: the stage module of each steps stage: v73-steps (release/v73_stage.py, the v73 steps on the v64 chain) and
#: dh-step (release/dh_stage.py, the DH step on the final V file)
STEP_STAGES = {'v73-steps': 'v73_stage', 'dh-step': 'dh_stage'}


def steps_replay(name, spec, data_root, out_dir, output, record_path):
    """Run a steps stage on its base entry; member values are saved."""
    import importlib
    import sys
    sys.path.insert(0, str(ROOT / 'release'))
    if spec['stage'] not in STEP_STAGES:
        raise ValueError('Unsupported steps stage: ' + spec['stage'])
    stage = importlib.import_module(STEP_STAGES[spec['stage']])
    result = stage.compose(spec, registry(), data_root, out_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result['body'])
    frame = pd.read_parquet(output)
    if not frame.equals(result['table']):
        raise SystemExit('written entry differs from the stage table')
    sha, vsha = legacy.sha256(output), legacy.value_sha256(frame)
    record = {'target': name, 'recipe': 'steps', 'stage': spec['stage'], 'rows': len(frame),
              'bytes': output.stat().st_size, 'sha256': sha, 'value_sha256': vsha,
              'expected_sha256': spec['sha256'], 'expected_value_sha256': spec['value_sha256'],
              **result['facts'], 'runtime': stage.R.runtime()}
    exact = sha == spec['sha256'] and vsha == spec['value_sha256'] and record['bytes'] == spec['bytes']
    record['verdict'] = 'PASS' if exact else 'FAIL'
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'steps'}, indent=2))
    if not exact:
        if spec['stage'] == 'dh-step':
            raise SystemExit('Entry bytes differ after the DH step; the record gives the platform and the facts')
        raise SystemExit('Entry bytes differ after the v73 steps; the record gives the platform and the '
                         'difference to the saved v73 values (the L2C trees depend on the platform)')
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--target', help='registered entry, e.g. v40 or 40; default from ENTRIES.json')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--record', type=Path)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    name, _ = target_record(args.target)
    mode = chain_mode(name, registry())
    if os.environ.get('PRC_RANKING_MODE') not in (None, '', mode):
        raise SystemExit(f'PRC_RANKING_MODE={os.environ["PRC_RANKING_MODE"]} differs from the mode of {name} ({mode})')
    os.environ['PRC_RANKING_MODE'] = mode  # before any release module is imported (they read it at import)
    replay(args.target, args.data_root, args.out_dir, args.record, args.overwrite)


if __name__ == '__main__':
    main()
