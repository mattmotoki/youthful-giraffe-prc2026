"""exp-060-coldprior: the registered constants, the label-free readers, the correction, the exact
clipped grid machinery and the statistics (SPEC sections 1, 3, 4, 6).

Every job imports this module, so the projection, the cold tables, the freeze, the selection, the
pricing and the bound cannot disagree on a rotation, a population, a key, a grid, a seed, a tie
rule or a statistic.

Label reads happen in ONE function, `dep_labels()`, which reads three columns
(`MVT_ID_mvt`, `PHASE_mvt`, `TAXITIME_SEC_mvt`) of one month and returns the departure labels in
a given id order.  `fit60.py` calls it for S and I, `price60.py` for V, `bound60.py` for S and V.
No other function here reads a label column, and `dep_frame()` asserts that neither label column
is in its read.

This module never opens an exp-056 `pred-*` archive or row-loss artifact (assignment, "Cold factor
tables" and "Support-aware ..." sections).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prc.data as D                                                  # noqa: E402
from prc.blend import regimes_from_raw                                # noqa: E402

E56 = ROOT / "experiments/exp-056-q12advocate"
E54 = ROOT / "experiments/exp-054-arrivalrepr"
WORK = EXP / "work"
ANALYSIS = EXP / "analysis"
ART = ROOT / "artifacts/exp-060"

# ---- the design (SPEC 1.1, 3.1-3.4) ---------------------------------------------------------
ROT = D.ISO_ROTATIONS                     # r -> (V, S, I, M); M is unused here
ORDER = ("jul", "jan", "dec")
BOARD = {"jul": 0.5571321275602379, "jan": 0.4428678724397621}    # ranking-month shares
FAMS = ("apt", "stand", "rwy", "op")
PRIOR_FAMS = ("stand", "rwy", "op")
LAMBDAS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, float("inf"))
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
#: c = inf is weight 1: the unweighted prior, so this family contains exp-056's registered form
C_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, float("inf"))
CLIP = 1800.0                                       # fitting residual clip, seconds
MIN_PRED = 60.0                                     # served floor
SHAM_SEEDS = tuple(range(1, 11))
TABLE_ARMS = ("true", "const", "gmean", "donor", "shuffle")
#: every arm of the correction, in the registered order
ARMS = ("zero", "literal_sham") + TABLE_ARMS + tuple(f"sham{s:02d}" for s in SHAM_SEEDS)
SHAM_ARMS = tuple(f"sham{s:02d}" for s in SHAM_SEEDS)
#: the named controls the gate compares `true` against, one contrast each
GATE_CONTROLS = ("zero", "const", "gmean", "donor", "shuffle") + SHAM_ARMS
BOOT = 2000                                         # exp-056's draw count
BOOT_SEED = 56                                      # exp-056's draw seed base (reused: same draws)
BLOCK_LENGTHS = (1, 2, 3)
THREADS = 3
LOAD_LIMIT = 10.0

DEP_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
            "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt"]
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", D.TARGET)
KEY_COL = {"stand": "STAND_mvt", "rwy": "RUNWAY_mvt", "op": "AIRCRAFT_OPERATOR_flt"}


# ---- hashing and bookkeeping ----------------------------------------------------------------
def sha_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes())
    return h.hexdigest()


def sha_text(values) -> str:
    return hashlib.sha256("\x1f".join("" if v is None else str(v) for v in values).encode()).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hold_load(what: str) -> dict:
    """Assignment: hold a launch while the 1-minute load average is above 10."""
    while True:
        l1 = os.getloadavg()[0]
        if l1 <= LOAD_LIMIT:
            break
        print(f"  {utc()} load {l1:.2f} > {LOAD_LIMIT:g}: holding {what}", flush=True)
        time.sleep(30)
    return {"launch_utc": utc(), "launch_load_1m": round(float(l1), 2), "unix": time.time()}


def code_shas() -> dict:
    return {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(HERE.glob("*.py"))}


def load_json(p) -> dict:
    return json.loads(Path(p).read_text())


def cold_months(r: str) -> list:
    """A = the nine 2025 months that are neither V nor S nor I of this rotation."""
    V, S, I, _ = ROT[r]
    out = [m for m in D.MONTHS if m not in (V, S, I)]
    assert len(out) == 9 and not ({V, S, I} & set(out))
    return out


# ---- label-free departure frame, keys, population, calendar ---------------------------------
def _txt(s: pd.Series) -> np.ndarray:
    """exp-054 `e54._txt` / exp-056 `e56._txt`: missing and empty strings are missing (None)."""
    v = s.astype("string")
    v = v.where(v.str.len() > 0)
    return np.array([None if pd.isna(x) else str(x) for x in v], dtype=object)


def _join(apt: np.ndarray, val: np.ndarray) -> np.ndarray:
    """exp-054 `e54._join`: `APT|value`, missing when either part is missing."""
    return np.array([None if (v is None or a is None) else f"{a}|{v}" for a, v in zip(apt, val)],
                    dtype=object)


def dep_frame(month: str) -> pd.DataFrame:
    """The month's departure rows with keys, regime and UTC date, rebuilt from the raw file with
    exp-056's explicit label-free column list.  Neither label column is in the read."""
    assert not set(LABEL_COLS) & set(DEP_COLS)
    raw = D.load_month(month, columns=DEP_COLS)
    raw = raw[raw["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
    apt = _txt(raw["ADEP_mvt"])
    out = pd.DataFrame({
        "MVT_ID_mvt": raw["MVT_ID_mvt"].to_numpy("int64"),
        "apt": apt,
        "stand": _join(apt, _txt(raw["STAND_mvt"])),
        "rwy": _join(apt, _txt(raw["RUNWAY_mvt"])),
        "op": _join(apt, _txt(raw["AIRCRAFT_OPERATOR_flt"])),
        "regime": regimes_from_raw(raw),
        "day": raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
    })
    out["pop"] = out["regime"].astype(str).to_numpy() == "nm_bulk"
    return out


def month_key_counts(month: str) -> dict:
    """Departure count per key value for all three prior families of one month, from the raw file
    with phase and the key columns only (label-free).  One read per month."""
    cols = ["PHASE_mvt", "ADEP_mvt"] + [KEY_COL[f] for f in PRIOR_FAMS]
    assert not set(LABEL_COLS) & set(cols)
    raw = pd.read_parquet(D.month_file(month), columns=cols)
    raw = raw[raw["PHASE_mvt"].to_numpy() == "DEP"]
    apt = _txt(raw["ADEP_mvt"])
    out = {}
    for f in PRIOR_FAMS:
        k = _join(apt, _txt(raw[KEY_COL[f]]))
        out[f] = {str(key): int(n) for key, n in pd.Series(k).dropna().value_counts().items()}
    return out


# ---- THE label reader -----------------------------------------------------------------------
LABEL_READS: list = []


def dep_labels(month: str, ids: np.ndarray, why: str) -> np.ndarray:
    """Departure labels of `month` in the order of `ids`.  The only label read of exp-060."""
    t = time.time()
    tb = pq.read_table(D.month_file(month), columns=["MVT_ID_mvt", "PHASE_mvt", D.TARGET]).to_pandas()
    tb = tb[tb["PHASE_mvt"].to_numpy() == "DEP"]
    s = pd.Series(tb[D.TARGET].to_numpy("float64"), index=tb["MVT_ID_mvt"].to_numpy("int64"))
    if not s.index.is_unique:
        raise SystemExit(f"{month}: departure ids are not unique")
    y = s.reindex(np.asarray(ids, dtype="int64")).to_numpy("float64")
    LABEL_READS.append({"month": month, "why": why, "unix": t, "utc": utc(), "rows": int(len(y))})
    return y


# ---- the cold table and its shams ------------------------------------------------------------
def sham_table(tab: pd.DataFrame, seed: int) -> pd.Series:
    """exp-056's permuted-prior sham, on the COLD table: the `true` factors permuted among the
    supported keys of the SAME family and airport (seeded).  Keeps each family-airport value
    multiset exactly and breaks the key identity."""
    rng = np.random.default_rng(56000 + seed)
    out = tab["true"].to_numpy("float64").copy()
    apt = tab["value"].astype(str).str.split("|").str[0].to_numpy()
    fam = tab["key"].astype(str).to_numpy()
    grp = pd.Series(np.arange(len(tab))).groupby([fam, apt]).indices
    for key in sorted(grp):
        idx = np.sort(grp[key])
        out[idx] = out[idx][rng.permutation(len(idx))]
    return pd.Series(out, index=tab.index)


# ---- the correction (SPEC 3.1): additive key effects shrunk toward beta * prior --------------
def design(codes: dict, nlev: dict, rows: np.ndarray | None = None) -> sp.csr_matrix:
    """One-hot design over the three correction families; a missing key (code -1) has no column."""
    cols, rr = [], []
    off = 0
    n = len(codes["stand"])
    sel = np.arange(n) if rows is None else np.flatnonzero(rows)
    for f in PRIOR_FAMS:
        c = codes[f][sel]
        ok = c >= 0
        rr.append(np.flatnonzero(ok))
        cols.append(c[ok] + off)
        off += nlev[f]
    r_ = np.concatenate(rr)
    c_ = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(sel), off))


def fam_offsets(nlev: dict) -> np.ndarray:
    return np.cumsum([0] + [nlev[f] for f in PRIOR_FAMS])


def prior_vector(prior: dict) -> np.ndarray:
    return np.concatenate([prior[f] for f in PRIOR_FAMS])


def support_weights(nA: dict, c: float) -> np.ndarray:
    """c / (c + n_A(k)) per correction column; c = inf is weight 1 (the unweighted prior)."""
    if np.isinf(c):
        return np.ones(sum(len(nA[f]) for f in PRIOR_FAMS))
    return np.concatenate([c / (c + nA[f].astype("float64")) for f in PRIOR_FAMS])


def demean_by_airport(r: np.ndarray, apt_code: np.ndarray) -> np.ndarray:
    """The fit month's residual minus its airport mean over the fit rows: the correction learns
    within-airport key contrasts, never the fit month's airport-level offset."""
    a = np.maximum(apt_code, 0)
    s = np.bincount(a, weights=r)
    n = np.bincount(a).astype("float64")
    return r - (s / np.maximum(n, 1.0))[a]


def final_pred(p: np.ndarray, h: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """max(60, p + h) on the population; p (already floored) elsewhere."""
    out = p.copy()
    out[pop] = np.maximum(MIN_PRED, p[pop] + h[pop])
    return out


# ---- the exact clipped grid machinery (SPEC 3.3) ---------------------------------------------
#: the 10 monomials of a quadratic form in (b0, b1, b2), in this fixed order
MONO = ("1", "b0", "b1", "b2", "b0b0", "b1b1", "b2b2", "b0b1", "b0b2", "b1b2")


def monomials(B: np.ndarray) -> np.ndarray:
    """(10, K) matrix of the monomials of every grid row of B (K, 3)."""
    b0, b1, b2 = B[:, 0], B[:, 1], B[:, 2]
    return np.stack([np.ones(len(B)), b0, b1, b2, b0 * b0, b1 * b1, b2 * b2,
                     b0 * b1, b0 * b2, b1 * b2])


def grid_A(betas=BETAS) -> np.ndarray:
    """Level A: one beta for all three families (the diagonal of the grid)."""
    b = np.asarray(betas, dtype="float64")
    return np.stack([b, b, b], 1)


def grid_B(betas=BETAS) -> np.ndarray:
    """Level B: the full 7^3 product grid, in stand-major order so that lexicographic order on
    (beta_stand, beta_runway, beta_operator) is the row order (the registered tie rule)."""
    b = np.asarray(betas, dtype="float64")
    g = np.array(np.meshgrid(b, b, b, indexing="ij")).reshape(3, -1).T
    return np.ascontiguousarray(g)


def cell_stats(e: np.ndarray, S: np.ndarray, cell: np.ndarray, ncell: int) -> np.ndarray:
    """The 15 exact sufficient statistics of every cell (an airport, or an airport-date pair):
    n, sum e, sum e^2, sum S_f (3), sum S_f e (3), sum S_f S_g (6, order 00 11 22 01 02 12).
    Returns (ncell, 15)."""
    out = np.zeros((ncell, 15))
    out[:, 0] = np.bincount(cell, minlength=ncell)
    out[:, 1] = np.bincount(cell, weights=e, minlength=ncell)
    out[:, 2] = np.bincount(cell, weights=e * e, minlength=ncell)
    for f in range(3):
        out[:, 3 + f] = np.bincount(cell, weights=S[:, f], minlength=ncell)
        out[:, 6 + f] = np.bincount(cell, weights=S[:, f] * e, minlength=ncell)
    for j, (f, g) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        out[:, 9 + j] = np.bincount(cell, weights=S[:, f] * S[:, g], minlength=ncell)
    return out


def sse_coeffs(st: np.ndarray) -> np.ndarray:
    """(ncell, 10): the monomial coefficients of sum (e - S.b)^2 per cell."""
    m = np.zeros((len(st), 10))
    m[:, 0] = st[:, 2]                               # sum e^2
    m[:, 1:4] = -2.0 * st[:, 6:9]                    # -2 sum S_f e
    m[:, 4:7] = st[:, 9:12]                          # sum S_f^2
    m[:, 7:10] = 2.0 * st[:, 12:15]                  # 2 sum S_f S_g
    return m


def res_coeffs(st: np.ndarray) -> np.ndarray:
    """(ncell, 10): the monomial coefficients of sum (e - S.b) per cell (only the linear part)."""
    m = np.zeros((len(st), 10))
    m[:, 0] = st[:, 1]                               # sum e
    m[:, 1:4] = -st[:, 3:6]                          # -sum S_f
    return m


def clip_split(u: np.ndarray, S: np.ndarray, B: np.ndarray):
    """Classify rows by the served floor over the grid's bounding box (exact for a product grid,
    conservative otherwise: a row is `never` only if the box minimum is at or above the floor, and
    `always` only if the box maximum is below it).  Returns (never, always, mixed) boolean masks."""
    lo_b, hi_b = B.min(0), B.max(0)
    lo = u + np.minimum(S * lo_b, S * hi_b).sum(1)
    hi = u + np.maximum(S * lo_b, S * hi_b).sum(1)
    never = lo >= MIN_PRED
    always = hi < MIN_PRED
    return never, always, ~(never | always)


def mixed_correction(y: np.ndarray, u: np.ndarray, S: np.ndarray, B: np.ndarray,
                     cell: np.ndarray, ncell: int, chunk: int = 4000):
    """Exact per-cell corrections of the mixed rows, for every grid row of B:
    d_sse = (y - max(60, lin))^2 - (y - lin)^2 and d_res = min(0, lin - 60), lin = u + S.b.
    Returns (ncell, K) and (ncell, K); both zero when there is no mixed row."""
    K = len(B)
    dsse = np.zeros((ncell, K))
    dres = np.zeros((ncell, K))
    for i0 in range(0, len(y), chunk):
        sl = slice(i0, i0 + chunk)
        lin = u[sl, None] + S[sl] @ B.T
        cl = np.maximum(MIN_PRED, lin)
        a = (y[sl, None] - cl) ** 2 - (y[sl, None] - lin) ** 2
        b = np.minimum(0.0, lin - MIN_PRED)
        np.add.at(dsse, cell[sl], a)
        np.add.at(dres, cell[sl], b)
    return dsse, dres


def prepare_cells(y, u, S, cell, ncell, B):
    """The exact SSE and residual-sum ingredients of one (arm, c, lambda) cell over a grid B.

    Rows whose served prediction is floored at every grid point enter the statistics with
    e = y - 60 and S = 0 (exact and constant in b); mixed rows are corrected exactly.
    Returns (stats(ncell,15), dsse(ncell,K), dres(ncell,K), counts, mixed-row mask)."""
    never, always, mixed = clip_split(u, S, B)
    e = np.where(never | mixed, y - u, y - MIN_PRED)
    Sz = np.where((never | mixed)[:, None], S, 0.0)
    st = cell_stats(e, Sz, cell, ncell)
    if mixed.any():
        dsse, dres = mixed_correction(y[mixed], u[mixed], S[mixed], B, cell[mixed], ncell)
    else:
        dsse = dres = None
    return st, dsse, dres, {"rows": int(len(y)), "never": int(never.sum()),
                            "always_clipped": int(always.sum()), "mixed": int(mixed.sum())}, mixed


def grid_curves(st, dsse, dres, PHI, st_out=None):
    """Exact per-cell curves over the grid: (sse(ncell,K), rsum(ncell,K), n(ncell,)).

    `st_out` (ncell, 3: n, sum e, sum e^2) holds rows that never move with the grid (outside the
    corrected population); when given, its constants enter both the SSE and the residual sum, so
    the free-intercept mean is taken over every row of the cell."""
    sse = sse_coeffs(st) @ PHI
    rsum = res_coeffs(st) @ PHI
    n = st[:, 0].copy()
    if dsse is not None:
        sse = sse + dsse
    if dres is not None:
        rsum = rsum + dres
    if st_out is not None:
        sse = sse + st_out[:, 2:3]
        rsum = rsum + st_out[:, 1:2]
        n = n + st_out[:, 0]
    return sse, rsum, n


# ---- the registered tie rules (SPEC 3.3) -----------------------------------------------------
def pick_grid_row(sse_row: np.ndarray, B: np.ndarray) -> int:
    """Stage 1: the grid point with the smallest SSE; a tie takes the smaller beta in stand,
    runway, operator order.  `B` is built in that lexicographic row order, so the first argmin of
    the exact float SSE is the registered pick."""
    return int(np.argmin(sse_row))


def pick_cell(totals: dict) -> tuple:
    """Stage 2: the (lambda, c) with the smallest total; a tie takes the larger lambda, then the
    larger c.  The keys are visited in the preferred order (-lambda, -c ascending puts inf, the
    largest, first) and only a strictly smaller total displaces the incumbent, so an exact tie
    keeps the more preferred key."""
    best = None
    for key in sorted(totals, key=lambda k: (-k[0], -k[1])):
        if best is None or totals[key] < totals[best]:
            best = key
    return best


# ---- statistics (SPEC 4) ---------------------------------------------------------------------
def day_index(days: np.ndarray) -> tuple:
    uniq = sorted(set(days.tolist()))
    pos = {d: i for i, d in enumerate(uniq)}
    return np.array([pos[d] for d in days]), uniq


def block_draws(k: int, L: int, seed: int) -> np.ndarray:
    """exp-056's moving blocks of L consecutive dates, BOOT draws, k dates per draw (trimmed)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(k / L))
    st = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
    return np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1), k - 1)[:, :k]


def draw_seed(r: str, L: int) -> int:
    """exp-056's seeds, reused unchanged so the draws are the same dates."""
    return BOOT_SEED * 1000 + ORDER.index(r) * 10 + L


def ci(x: np.ndarray) -> list:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]
