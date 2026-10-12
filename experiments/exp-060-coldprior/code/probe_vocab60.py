"""Label-free probe: is the sparse ridge solve byte-identical when the vocabulary loses
columns that no fit row and no scored row uses?  Reads ONE month's key columns only."""
import sys, hashlib
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, "/path/to/competitions/prc-data-challenge-2026")
sys.path.insert(0, "/path/to/competitions/prc-data-challenge-2026/experiments/exp-056-q12advocate/code")
import e56 as E

d = E.dep_frame("2025-08")
fams = ("stand", "rwy", "op")
vals = {f: sorted(set(x for x in d[f] if x is not None)) for f in fams}
rng = np.random.default_rng(0)
# extra arrival-only values that no departure row uses (what a warm table adds and a cold one drops)
extra = {f: [f"ZZ{f}|{i}" for i in range(max(1, len(vals[f]) // 6))] for f in fams}
full = {f: sorted(vals[f] + extra[f]) for f in fams}
red = {f: sorted(vals[f]) for f in fams}

def design(voc):
    nlev = {f: len(voc[f]) for f in fams}
    codes = {}
    for f in fams:
        pos = {v: i for i, v in enumerate(voc[f])}
        codes[f] = np.array([pos[x] if x is not None else -1 for x in d[f]], dtype="int64")
    return E.design(codes, nlev), nlev

r = rng.normal(0, 300, len(d))
out = {}
for name, voc in (("full", full), ("reduced", red)):
    X, nlev = design(voc)
    for lam in (3.0, 300.0, 3000.0):
        lu = splu((X.T @ X + lam * sp.identity(X.shape[1], format="csr")).tocsc())
        th = lu.solve(np.asarray(X.T @ r, dtype="float64"))
        h = X @ th
        out[(name, lam)] = h
for lam in (3.0, 300.0, 3000.0):
    a, b = out[("full", lam)], out[("reduced", lam)]
    same = np.array_equal(a, b)
    print(f"lam={lam:>7}: byte-identical={same}  max|diff|={np.max(np.abs(a-b)):.3e}  "
          f"sha_full={hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]} "
          f"sha_red={hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()[:12]}")
print("cols full/reduced:", {f: (len(full[f]), len(red[f])) for f in fams})
