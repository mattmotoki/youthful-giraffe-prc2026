"""exp-065-sparseholdout-retry: the registered constants, the source boundary, the audited label reader
and the hard-threshold family (SPEC sections 1-3, 6).

Every job imports this module, so the projection, the masks, the selection, the pricing and the
bound cannot disagree on a rotation, a population, a key, a grid, a threshold, a seed, a tie rule
or a statistic.

`experiments/exp-060-coldprior/code/e60.py` is imported READ-ONLY after a SHA-256 pin and supplies
the correction, the exact clipped-SSE machinery, the tie rules and the statistics unchanged
(SPEC 1.3).  Nothing here writes through e60's path constants: `out_path()` asserts that every
output lives under this experiment or `artifacts/exp-065`.

THE SOURCE BOUNDARY (assignment, "Registration and frozen sources"):
  * selection, correction, pricing and bound code open only `work/inputs65-<r>.npz`,
    `work/vocab65-<r>.json`, `work/masks65-<r>.npz` and this experiment's own records;
  * `REFUSED_SOURCES` names every exp-056 archive and every exp-060 result, row-loss, prediction,
    gate and bound artifact; `guard_open()` refuses any of them and is called by every reader;
  * S, I and V departure labels are read ONLY by `dep_labels()`, which checks the raw month file
    against the SHA-256 the projection recorded before it reads, and appends to `LABEL_READS`.

`code/audit61.py` is the separate audit process and is the ONLY job allowed to open exp-060's
results; it is not imported by any producing job and writes nothing a producing job reads.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E60_DIR = HERE


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


E60_PIN = {"code/e60local.py": _sha_file(HERE / "e60local.py")}


for _rel, _want in E60_PIN.items():
    _got = _sha_file(EXP / _rel)
    if _got != _want:
        raise SystemExit(f"{_rel} is {_got[:16]}, not the pinned {_want[:16]}: refusing to import")

if str(E60_DIR) not in sys.path:
    sys.path.insert(0, str(E60_DIR))
import e60local as E60                                                     # noqa: E402
import prc.data as D                                                  # noqa: E402

# ---- re-exported, unchanged, from the pinned exp-060 module ---------------------------------
sha_file = E60.sha_file
sha_arr = E60.sha_arr
sha_text = E60.sha_text
utc = E60.utc
load_json = E60.load_json
dep_frame = E60.dep_frame
month_key_counts = E60.month_key_counts
design = E60.design
fam_offsets = E60.fam_offsets
prior_vector = E60.prior_vector
support_weights = E60.support_weights
demean_by_airport = E60.demean_by_airport
final_pred = E60.final_pred
monomials = E60.monomials
grid_A = E60.grid_A
grid_B = E60.grid_B
cell_stats = E60.cell_stats
sse_coeffs = E60.sse_coeffs
res_coeffs = E60.res_coeffs
clip_split = E60.clip_split
mixed_correction = E60.mixed_correction
prepare_cells = E60.prepare_cells
grid_curves = E60.grid_curves
pick_grid_row = E60.pick_grid_row
day_index = E60.day_index
block_draws = E60.block_draws
draw_seed = E60.draw_seed
ci = E60.ci
sham_table = E60.sham_table

# ---- the registered design (SPEC 1.1, 1.5, 2) ------------------------------------------------
ROT = {"may": ("2025-05", "2025-06", "2025-04", None)}
ORDER = ("may",)
BOARD = {"may": 1.0}
FAMS = E60.FAMS
PRIOR_FAMS = E60.PRIOR_FAMS
LAMBDAS = E60.LAMBDAS
BETAS = E60.BETAS
C_GRID = E60.C_GRID
CLIP = E60.CLIP
MIN_PRED = E60.MIN_PRED
BOOT = E60.BOOT
BLOCK_LENGTHS = E60.BLOCK_LENGTHS
DEP_COLS = E60.DEP_COLS
LABEL_COLS = E60.LABEL_COLS
KEY_COL = E60.KEY_COL
LOAD_LIMIT = 10.0

#: the one new axis: the hard support thresholds, strict `n_A(k) < n*`
THRESHOLDS = (25, 50, 100, 200)
#: the ten primary matched shams, built here after hard masking and support weighting (recorded rule)
SHAM_NAMES = tuple(range(1, 11))
SHAM_SEEDS = SHAM_NAMES
SHAM_ARMS = tuple(f"sham{s:02d}" for s in SHAM_NAMES)
TABLE_ARMS = ("true", "const", "gmean", "donor", "shuffle")
#: the seventeen primary arms, in exp-060's order
ARMS = ("zero", "literal_sham") + TABLE_ARMS + SHAM_ARMS
#: the arms with a prior that the hard mask and the singleton rule act on (SPEC reading (d))
PRIOR_ARMS = ("true", "gmean", "donor", "shuffle") + SHAM_ARMS
DEGENERATE_ARMS = ("zero", "const", "literal_sham")
GATE_CONTROLS = ("zero", "const", "gmean", "donor", "shuffle") + SHAM_ARMS
#: the sham permutation seed base; the spawn key is [SHAM_SEED_BASE, seed, i(n*), i(family), apt]
SHAM_SEED_BASE = 65000
SHAM_SEED_CANDIDATES = tuple(range(1, 100001))
DRAW_SEED = 650052026

WORK = EXP / "work"
ANALYSIS = EXP / "analysis"
LOGS = EXP / "logs"
ART = ROOT / "artifacts/exp-065"
E60EXP = EXP


def cold_months(r: str) -> list[str]:
    V, S, I, _ = ROT[r]
    out = [m for m in D.MONTHS if m not in (V, S, I)]
    assert out == ["2025-01", "2025-02", "2025-03", "2025-07", "2025-08",
                   "2025-09", "2025-10", "2025-11", "2025-12"]
    return out

# ---- the source boundary (SPEC 1.3) ----------------------------------------------------------
#: patterns of exp-056 and exp-060 artifacts no producing job may open
REFUSED_SOURCES = (
    r"experiments/exp-056-q12advocate/",
    r"experiments/exp-060-coldprior/analysis/(fit60|oos|bound|check-|grid60|identity60|answers60"
    r"|banddraws60|ceiling60|align60|verifyinputs60|report|board-tables)",
    r"experiments/exp-060-coldprior/work/pred-",
    r"experiments/exp-060-coldprior/oof/",
    r"artifacts/exp-060/",
)
#: array names no projection may request or write (SPEC 1.2)
FORBID_ARRAY = (r"^h_", r"^hI_", r"^member_", r"^thin_", r"^p_S(expo|ref)$",
                r"loss", r"label", r"expo", r"sham", r"^y_", r"pred", r"bound", r"gate",
                r"oos", r"select", r"result")

OPENED: list = []


def guard_open(path, why: str = "") -> Path:
    """Refuse an exp-056 archive or an exp-060 result, row-loss, prediction, gate or bound
    artifact, and record every path a producing job opens."""
    p = Path(path)
    rel = str(p.resolve().relative_to(ROOT)) if p.is_absolute() else str(p)
    for pat in REFUSED_SOURCES:
        if re.search(pat, rel):
            raise SystemExit(f"the source boundary refuses {rel} ({why})")
    OPENED.append({"path": rel, "why": why})
    return p


def forbidden_array(name: str) -> bool:
    return any(re.search(pat, name) for pat in FORBID_ARRAY)


def out_path(p) -> Path:
    """Every output of this experiment lives under it or under artifacts/exp-065."""
    q = Path(p).resolve()
    ok = str(q).startswith(str(EXP.resolve())) or str(q).startswith(str(ART.resolve()))
    if not ok:
        raise SystemExit(f"refusing to write outside exp-065: {q}")
    return q


def code_shas() -> dict:
    d = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(HERE.glob("*.py"))}
    d |= {f"experiments/exp-065-sparseholdout-retry/{k}": v for k, v in E60_PIN.items()}
    return d


def hold_load(what: str) -> dict:
    return E60.hold_load(what)


def draw_seed(r: str, length: int) -> int:
    if r != "may" or length not in BLOCK_LENGTHS:
        raise SystemExit("exp-065 permits only the registered May fold and block lengths")
    return DRAW_SEED + int(length)


# ---- THE label reader (SPEC 1.3) -------------------------------------------------------------
LABEL_READS: list = []
#: rotation -> {month: sha256}, filled by `pin_months()` from the projection record
MONTH_PINS: dict = {}


def pin_months(rec: dict) -> None:
    MONTH_PINS.update(rec["month_files_sha256"])


def dep_labels(month: str, ids: np.ndarray, why: str) -> np.ndarray:
    """Departure labels of `month` in the order of `ids`.  The ONLY label read of exp-065.
    The raw month file is checked against the SHA-256 the projection recorded before it is read."""
    import pandas as pd
    f = D.month_file(month)
    if month not in MONTH_PINS:
        raise SystemExit(f"{month} is not pinned: call pin_months() with the projection record")
    got = sha_file(f)
    if got != MONTH_PINS[month]:
        raise SystemExit(f"{f.name} is {got[:16]}, not the pinned {MONTH_PINS[month][:16]}")
    t = time.time()
    tb = pq.read_table(f, columns=["MVT_ID_mvt", "PHASE_mvt", D.TARGET]).to_pandas()
    tb = tb[tb["PHASE_mvt"].to_numpy() == "DEP"]
    s = pd.Series(tb[D.TARGET].to_numpy("float64"), index=tb["MVT_ID_mvt"].to_numpy("int64"))
    if not s.index.is_unique:
        raise SystemExit(f"{month}: departure ids are not unique")
    y = s.reindex(np.asarray(ids, dtype="int64")).to_numpy("float64")
    LABEL_READS.append({"month": month, "why": why, "unix": t, "utc": utc(),
                        "rows": int(len(y)), "file_sha256": got})
    return y


# ---- the extended tie rule (SPEC 3.1) --------------------------------------------------------
def cell_order() -> list:
    """The registered stage-2 preference order over (n*, lambda, c): the smaller n* first, then
    the larger lambda, then the larger c.  Only a strictly smaller total may displace an earlier
    cell, so an exact tie keeps the more preferred one."""
    return sorted(((n, lam, c) for n in THRESHOLDS for lam in LAMBDAS for c in C_GRID),
                  key=lambda k: (k[0], -k[1], -k[2]))


def pick_cell3(totals: dict) -> tuple:
    """Stage 2 over (n*, lambda, c) with the extended tie order."""
    best = None
    for key in cell_order():
        if key not in totals:
            continue
        if best is None or totals[key] < totals[best]:
            best = key
    return best


def cell_name(key: tuple) -> str:
    n, lam, c = key
    ls = "inf" if np.isinf(lam) else f"{lam:g}"
    cs = "inf" if np.isinf(c) else f"{c:g}"
    return f"n={n},lam={ls},c={cs}"


def jsonable(x):
    if isinstance(x, float) and np.isinf(x):
        return "inf"
    return x


# ---- the hard mask and the effective prior (SPEC 1.5) ----------------------------------------
def column_airport(vocab: dict) -> np.ndarray:
    """The airport code of every correction column (the key string's `APT|` prefix)."""
    apt_pos = {v: i for i, v in enumerate(vocab["apt"])}
    return np.concatenate([np.array([apt_pos.get(str(v).split("|")[0], -1) for v in vocab[f]],
                                    dtype="int64") for f in PRIOR_FAMS])


def column_family(nlev: dict) -> np.ndarray:
    return np.concatenate([np.full(nlev[f], i, dtype="int64")
                           for i, f in enumerate(PRIOR_FAMS)])


def group_ids(col_fam: np.ndarray, col_apt: np.ndarray, nA: np.ndarray, n_star: int):
    """(group id per column, kept mask).  A group is (family, airport) restricted to the columns
    with n_A < n*; a group with fewer than two columns is EXCLUDED and its columns are zeroed in
    every arm that carries a prior (SPEC reading (d))."""
    below = nA < n_star
    gid = np.where(below & (col_apt >= 0), col_fam * 10_000 + col_apt, -1)
    keep = np.zeros(len(gid), dtype=bool)
    for g in np.unique(gid[gid >= 0]):
        idx = np.flatnonzero(gid == g)
        if len(idx) >= 2:
            keep[idx] = True
    return gid, keep


def effective(table: np.ndarray, keep: np.ndarray, nA: np.ndarray, c: float) -> np.ndarray:
    """table(k) * 1[k in a kept group below n*] * c/(c + n_A(k)); the mask is applied BEFORE the
    smooth weight, and c = inf is the literal weight 1.0."""
    w = np.ones(len(table)) if np.isinf(c) else c / (c + nA.astype("float64"))
    return np.where(keep, table * w, 0.0)


def sham_permutation(seed: int, i_n: int, fam: int, apt: int, size: int) -> np.ndarray:
    """A deterministic NON-IDENTITY permutation of a kept group's columns, in ascending column
    order (SPEC 1.5).  `size >= 2`, so a non-identity permutation exists."""
    if size < 2:
        raise SystemExit("a kept group has fewer than two columns")
    ss = np.random.SeedSequence([SHAM_SEED_BASE, int(seed), int(i_n), int(fam), int(apt)])
    rng = np.random.default_rng(ss)
    ident = np.arange(size)
    for _ in range(1000):
        p = rng.permutation(size)
        if not np.array_equal(p, ident):
            return p
    raise SystemExit("no non-identity permutation was drawn in 1000 tries")
