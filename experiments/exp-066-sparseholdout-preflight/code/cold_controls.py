"""Label-free synthetic contract for the future A-only cold-control producer."""
from __future__ import annotations

import numpy as np


def sham_table(values, groups, permutation):
    """Permute values inside groups and refuse identity or multiset changes."""
    values = np.asarray(values, dtype="float64")
    groups = np.asarray(groups)
    permutation = np.asarray(permutation, dtype="int64")
    if sorted(permutation.tolist()) != list(range(len(values))):
        raise ValueError("permutation is not bijective")
    out = values[permutation]
    for group in np.unique(groups):
        rows = groups == group
        if not np.array_equal(np.sort(out[rows]), np.sort(values[rows])):
            raise ValueError("sham changed a within-group multiset")
        if not np.array_equal(np.sort(permutation[rows]), np.flatnonzero(rows)):
            raise ValueError("sham crossed groups")
        if np.array_equal(out[rows], values[rows]):
            raise ValueError(f"required group {group!r} is unchanged")
    return out


def zero_singletons(values, groups):
    values = np.asarray(values, dtype="float64").copy()
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    values[counts[inverse] == 1] = 0.0
    return values
