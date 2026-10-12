"""Hash-bound C50/C51 affine pair; signed control coefficient is intentional."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diverse_blend import vector_digest

ID = 'MVT_ID_mvt'


@dataclass(frozen=True)
class PairedBlend:
    ids: np.ndarray
    sequence: np.ndarray
    control: np.ndarray
    scope: np.ndarray
    control_scope: np.ndarray
    sequence_weight: np.ndarray
    control_weight: np.ndarray
    base_digest: str
    extension: np.ndarray | None = None
    extension_proxy_gain: float | None = None
    split: str = 'ranking'

    def apply(self, ids, base, *, enable_extension=False):
        assert isinstance(enable_extension, bool)
        assert np.array_equal(ids,self.ids), 'Pair arrays belong to another ID order.'
        assert np.isfinite(base).all() and (base >= 60).all()
        assert vector_digest(ids,base) == self.base_digest, 'Pair weights belong to another base.'
        arrays = [self.sequence,self.control,self.scope,self.control_scope,self.sequence_weight,self.control_weight]
        assert all(a.shape == base.shape for a in arrays)
        assert self.scope.dtype == self.control_scope.dtype == np.dtype(bool)
        assert not (self.control_scope & ~self.scope).any()
        assert np.isfinite(self.sequence[self.scope]).all() and (self.sequence[self.scope] >= 60).all()
        assert np.isfinite(self.control[self.control_scope]).all() and (self.control[self.control_scope] >= 60).all()
        assert np.isfinite(self.sequence_weight).all() and np.isfinite(self.control_weight).all()
        assert (self.control_weight[~self.control_scope] == 0).all()
        ext = np.zeros(base.shape,dtype=bool) if self.extension is None else self.extension
        assert ext.shape == base.shape and ext.dtype == np.dtype(bool)
        if enable_extension:
            assert self.extension_proxy_gain is not None and np.isfinite(self.extension_proxy_gain) and self.extension_proxy_gain > 0, 'Pair extension needs a positive pinned valid proxy read 2.'
            assert self.split in ('fold','ranking')
            if self.split == 'ranking':
                assert (self.scope & ext).any(), 'Pair extension needs an admitted annual vector with extension scope.'
        out = base.copy()
        on = self.scope.copy()
        if not enable_extension:
            on &= ~ext
        delta = self.sequence_weight[on]*(self.sequence[on]-base[on])
        ctrl = self.control_scope[on]
        delta[ctrl] += self.control_weight[on][ctrl]*(self.control[on][ctrl]-base[on][ctrl])
        out[on] = np.maximum(60.,base[on]+delta)
        assert np.array_equal(out[~on],base[~on])
        assert np.isfinite(out).all() and (out >= 60).all()
        return out


def load_pair(path,spec,bundle,split):
    """Called by the common loader after it checks the manifest hash."""
    assert spec['schema'] == 3 and spec['lane'] == 'C'
    assert spec['weight_basis'] == 'two_fold_2025' and spec['serving_point'] == 'after_head'
    assert spec['member_training_months']['fold'] == [2,3,4,5,6,8,9,10,11,12]
    assert spec['member_training_months']['ranking'] == list(range(1,13))
    assert split in ('fold','ranking') and bundle.month.isin([1,7]).all()
    assert bundle.cls.isin(['ordinary','clock','noac']).all()
    assert bundle.extension.dtype == np.dtype(bool)
    extension = bundle.extension.to_numpy(bool).copy()
    allowed = bundle.month.eq(1) & bundle.airport.isin(['LEMD','LFPG']) & bundle.cls.eq('clock')
    assert not (extension & ~allowed.to_numpy()).any(), 'Pair extension cells differ from LEMD/LFPG January clock.'
    proxy_gain = None
    if 'extension_proxy' in spec:
        proxy = spec['extension_proxy']
        assert proxy['owner'] == 'valid' and proxy['read'] == 'proxy_2'
        source = Path(proxy['path'])
        if not source.is_absolute():source = Path(path).parent/source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == proxy['sha256'], 'Extension proxy read changed.'
        result = json.loads(source.read_text())
        extra_paths = proxy.get('required_positive_paths',[])
        assert isinstance(extra_paths,list)
        gains = []
        for keys in [proxy['gain_path']]+extra_paths:
            assert isinstance(keys,list) and keys and all(type(k) in (str,int) for k in keys)
            value = result
            for key in keys:value = value[key]
            assert type(value) in (int,float) and np.isfinite(value) and value > 0, 'Extension proxy read 2 is not positive.'
            gains.append(float(value))
        proxy_gain = min(gains)
    assert set(spec['fold_weights']) == {'ordinary','clock'}
    for cls,table in spec['fold_weights'].items():
        assert set(table) == {'1','7'}
        for m,other in [(1,7),(7,1)]:
            r = table[str(m)]
            assert r['trained_on_months'] == [other]
            assert np.isfinite([r['sequence'],r['control']]).all()
            if cls == 'clock':
                assert r['control'] == 0 and 0 <= r['sequence'] <= 1
    if split == 'ranking':
        r = spec['ranking_weights']
        assert r['trained_on_months'] == [1,7]
        ordinary,clock = r['ordinary'],r['clock']
        assert ordinary['method'] == 'pooled_practice_scaled_wls'
        xtwx = np.asarray(ordinary['xtwx'],float)
        xtwr = np.asarray(ordinary['xtwr'],float)
        assert xtwx.shape == (2,2) and xtwr.shape == (2,)
        assert np.isfinite(xtwx).all() and np.isfinite(xtwr).all()
        assert np.allclose(xtwx,xtwx.T,rtol=0,atol=1e-10) and np.linalg.eigvalsh(xtwx).min() > 0
        coefficients = np.linalg.solve(xtwx,xtwr)
        assert np.allclose(coefficients,[ordinary['sequence'],ordinary['control']],rtol=0,atol=1e-10)
        assert clock['method'] == 'pooled_served_grid' and clock['control'] == 0
        grid,losses = np.asarray(clock['grid'],float),np.asarray(clock['losses'],float)
        assert np.array_equal(grid,np.arange(51)*.02)
        assert losses.shape == grid.shape and np.isfinite(losses).all() and (losses >= 0).all()
        assert clock['sequence'] == grid[int(np.argmin(losses))]
    rec = spec['vectors'][split]
    source = Path(rec['path'])
    if not source.is_absolute():source = Path(path).parent/source
    assert hashlib.sha256(source.read_bytes()).hexdigest() == rec['sha256'], 'Pair vector changed.'
    frame = pd.read_parquet(source)
    assert frame[ID].is_unique and len(frame) == len(bundle) and set(frame[ID]) == set(bundle[ID])
    assert np.isfinite(frame[ID]).all() and np.equal(frame[ID],frame[ID].astype('int64')).all()
    frame = frame.set_index(ID).reindex(bundle[ID])
    assert np.array_equal(frame['month'],bundle.month) and np.array_equal(frame['cls'],bundle.cls)
    for name in ['member_scope','control_scope']:
        assert frame[name].dtype == np.dtype(bool)
    on = frame.member_scope.to_numpy(bool)
    control_on = frame.control_scope.to_numpy(bool)
    assert np.array_equal(control_on,on & bundle.cls.eq('ordinary').to_numpy())
    assert not (on & bundle.cls.eq('noac').to_numpy()).any()
    ws,wc = np.zeros(len(bundle)),np.zeros(len(bundle))
    for cls in ['ordinary','clock']:
        for m in [1,7]:
            row = spec['fold_weights'][cls][str(m)] if split == 'fold' else spec['ranking_weights'][cls]
            use = on & bundle.cls.eq(cls).to_numpy() & bundle.month.eq(m).to_numpy()
            ws[use],wc[use] = row['sequence'],row['control']
    return PairedBlend(bundle[ID].to_numpy('int64').copy(),frame.pred_seq170.to_numpy(float),
                       frame.pred_row170.to_numpy(float),on,control_on,ws,wc,rec['base_digest'],extension,proxy_gain,split)
