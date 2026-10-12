from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
sys.path.insert(0, str(CODE))


def test_imports_and_local_sham_export():
    for name in ("e60local", "e54", "e65", "prepare65", "mask65", "fit65", "price65", "bound65"):
        importlib.import_module(name)
    import e65
    assert callable(e65.sham_table)


def test_sham_preserves_group_multisets_and_moves_values():
    import e65
    rows = []
    for fam in e65.PRIOR_FAMS:
        for apt in ("AAAA", "BBBB"):
            for i in range(5):
                rows.append({"key": fam, "value": f"{apt}|{i}", "true": float(i + 1)})
    tab = pd.DataFrame(rows)
    for seed in range(1, 11):
        got = e65.sham_table(tab, seed).to_numpy("float64")
        assert not np.array_equal(got, tab.true.to_numpy("float64"))
        for (_, apt), idx in pd.Series(np.arange(len(tab))).groupby(
                [tab.key, tab.value.str.split("|").str[0]]).indices.items():
            idx = np.sort(idx)
            assert np.array_equal(np.sort(got[idx]), np.sort(tab.true.to_numpy()[idx]))


def test_one_fold_and_registered_draws_only():
    import e65
    assert e65.ORDER == ("may",)
    assert e65.BOARD == {"may": 1.0}
    assert [e65.draw_seed("may", n) for n in (1, 2, 3)] == [650052027, 650052028, 650052029]
