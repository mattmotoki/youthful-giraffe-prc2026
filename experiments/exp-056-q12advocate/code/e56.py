"""exp-056-q12advocate: the registered constants, the label-free readers, the correction and the
statistics (SPEC section 3).

Every job imports this module, so the freeze, fit and pricing jobs cannot disagree on a
rotation, a population, a key, a grid, a seed or a statistic.

Label reads happen in one function only, `dep_labels()`. It reads three columns
(`MVT_ID_mvt`, `PHASE_mvt`, `TAXITIME_SEC_mvt`) of one month and returns the departure
labels in a given id order. `fit.py` calls it for S and I; `price.py` calls it for V.
No other function reads a label column. The departure frame reader asserts that neither
label column is in its read.
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

# ---- the design (SPEC 3.1-3.4) -------------------------------------------------------------
ROT = D.ISO_ROTATIONS                     # r -> (V, S, I, M), committed in 80786e6
ORDER = ("jul", "jan", "dec")
BOARD = {"jul": 0.5571321275602379, "jan": 0.4428678724397621}    # ranking-month shares
MEMBERS = ("F1-fresh", "T-tailv4", "L2-hub8725")
KAGGLE = ROOT / "artifacts/exp-056/kaggle"          # pulled kernel outputs (git-ignored)
WORK = EXP / "work"
ANALYSIS = EXP / "analysis"
ART = ROOT / "artifacts/exp-056"

#: exp-054's frozen factor tables, fitted on the eleven months other than V (arrivals only)
E54 = ROOT / "experiments/exp-054-arrivalrepr"
REP_FOLD = {"jul": "july_loo", "jan": "january", "dec": "december"}
REP_SHA = {"july_loo": "a8f440aea6bee83058c368bab617c50c9c866e7ba5fa33527e3f059e56999b14",
           "december": "0650b6456f4cb97a8422aa511491591f76fa7dca86628201d00a49795ee602f5",
           "january": "826c1c6e1ddd51ae1110f5020c5a28012f22a79e08f60de63461de59ac169de4"}
TABLE_ARMS = ("true", "const", "gmean", "donor", "shuffle")

FAMS = ("apt", "stand", "rwy", "op")               # vocabulary families
PRIOR_FAMS = ("stand", "rwy", "op")                # correction families (no airport term)
LAMBDAS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, float("inf"))
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
CLIP = 1800.0                                       # fitting residual clip, seconds
MIN_PRED = 60.0
SHAM_SEEDS = tuple(range(1, 11))
ORIGIN_SALT = "exp056-origin"
#: matched-size exposure pair per rotation (SPEC 3.5): (exposed fold, isolated fold) on S. The
#: larger side is thinned by SIZE_SALT (`kernels/iso/prc-iso-thin.py`): `xm` is iso_<r>_x thinned
#: to |A|, `sm` is iso_<r>_s thinned to |A - M + S|; `s` and `x` are the unthinned folds.
EXPOSURE_PAIR = {"jul": ("xm", "s"), "jan": ("x", "sm"), "dec": ("x", "sm")}
SIZE_SALT = "exp056-size"
BOOT = 2000
BOOT_SEED = 56
BLOCK_LENGTHS = (1, 2, 3)
THREADS = 3
#: dry-run version: 1 had an airport term (superseded, SPEC 3.10); 2 had the whole-month exposure
#: design; 3 is the registered design (matched-size exposure pair)
DRY_VERSION = 3
LOAD_LIMIT = 10.0
N_A_BINS = (50, 500)                                # n_A(stand) strata: <50, 50-499, >=500
N_S_BINS = (1, 10, 50)                              # n_S(stand) strata: 0, 1-9, 10-49, >=50

DEP_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
            "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
            "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt"]
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", D.TARGET)


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
    while True:
        l1 = os.getloadavg()[0]
        if l1 <= LOAD_LIMIT:
            break
        print(f"  {utc()} load {l1:.2f} > {LOAD_LIMIT:g}: holding {what}", flush=True)
        time.sleep(30)
    return {"launch_utc": utc(), "launch_load_1m": round(float(l1), 2)}


def code_shas() -> dict:
    return {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(HERE.glob("*.py"))}


# ---- label-free departure frame, keys, population, calendar ---------------------------------
def _txt(s: pd.Series) -> np.ndarray:
    """exp-054 `e54._txt`: missing and empty strings are missing (None); nothing else changes."""
    v = s.astype("string")
    v = v.where(v.str.len() > 0)
    return np.array([None if pd.isna(x) else str(x) for x in v], dtype=object)


def _join(apt: np.ndarray, val: np.ndarray) -> np.ndarray:
    """exp-054 `e54._join`: `APT|value`, missing when either part is missing."""
    return np.array([None if (v is None or a is None) else f"{a}|{v}" for a, v in zip(apt, val)],
                    dtype=object)


def dep_frame(month: str) -> pd.DataFrame:
    """The month's departure rows with keys, regime and UTC date. Neither label column is read."""
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


_SIZE_CACHE: dict = {}


def size_keep(r: str) -> dict:
    """The registered size selection of rotation r (label-free: ids and phase only): the thinned
    fold, its training departures, and the sha of the kept ids (sorted, float64 bytes), computed
    exactly as the thin kernel does."""
    if r not in _SIZE_CACHE:
        thin = [w for w in EXPOSURE_PAIR[r] if w.endswith("m")][0]
        base = f"iso_{r}_{thin[0]}"
        other = f"iso_{r}_{'s' if thin[0] == 'x' else 'x'}"

        def dep_ids(fold):
            out = []
            for m in D.FOLDS[fold].train_months:
                t = pd.read_parquet(D.month_file(m), columns=["MVT_ID_mvt", "PHASE_mvt"])
                out.append(t.loc[t["PHASE_mvt"].to_numpy() == "DEP", "MVT_ID_mvt"].to_numpy("int64"))
            return np.concatenate(out)
        ids, n_keep = dep_ids(base), len(dep_ids(other))
        hv = np.array([int(hashlib.sha256(f"{SIZE_SALT}|{int(i)}".encode()).hexdigest()[:15], 16)
                       for i in ids], dtype="int64")
        keep = np.sort(ids[np.argsort(hv, kind="stable")[:n_keep]])
        _SIZE_CACHE[r] = {"alias": f"iso_{r}_{thin}", "base_fold": base, "salt": SIZE_SALT,
                          "train_departures": int(len(ids)), "n_keep": int(n_keep),
                          "matched_to": other,
                          "kept_ids_sorted_sha256": hashlib.sha256(
                              np.ascontiguousarray(keep.astype("float64")).tobytes()).hexdigest()}
    return _SIZE_CACHE[r]


def key_counts(months, fam: str) -> dict:
    """Departure count per key over `months` (label-free: key columns and phase only)."""
    cnt: dict = {}
    for m in months:
        raw = pd.read_parquet(D.month_file(m), columns=["PHASE_mvt", "ADEP_mvt",
                                                         {"stand": "STAND_mvt", "rwy": "RUNWAY_mvt",
                                                          "op": "AIRCRAFT_OPERATOR_flt"}[fam]])
        raw = raw[raw["PHASE_mvt"].to_numpy() == "DEP"]
        apt = _txt(raw["ADEP_mvt"])
        k = _join(apt, _txt(raw.iloc[:, 2]))
        for key, n in pd.Series(k).dropna().value_counts().items():
            cnt[key] = cnt.get(key, 0) + int(n)
    return cnt


# ---- THE label reader -----------------------------------------------------------------------
LABEL_READS: list = []


def dep_labels(month: str, ids: np.ndarray, why: str) -> np.ndarray:
    """Departure labels of `month` in the order of `ids`. The only label read in the experiment."""
    t = time.time()
    tb = pq.read_table(D.month_file(month), columns=["MVT_ID_mvt", "PHASE_mvt", D.TARGET]).to_pandas()
    tb = tb[tb["PHASE_mvt"].to_numpy() == "DEP"]
    s = pd.Series(tb[D.TARGET].to_numpy("float64"), index=tb["MVT_ID_mvt"].to_numpy("int64"))
    if not s.index.is_unique:
        raise SystemExit(f"{month}: departure ids are not unique")
    y = s.reindex(np.asarray(ids, dtype="int64")).to_numpy("float64")
    LABEL_READS.append({"month": month, "why": why, "unix": t, "utc": utc(), "rows": int(len(y))})
    return y


# ---- factor tables --------------------------------------------------------------------------
def factor_table(r: str) -> pd.DataFrame:
    fold = REP_FOLD[r]
    p = E54 / f"work/rep-{fold}.parquet"
    got = sha_file(p)
    if got != REP_SHA[fold]:
        raise SystemExit(f"{p.name} is {got[:16]}, not exp-054's recorded {REP_SHA[fold][:16]}")
    return pd.read_parquet(p)


def sham_table(tab: pd.DataFrame, seed: int) -> pd.Series:
    """The permuted-prior sham: the `true` factors permuted among the supported keys of the SAME
    family and airport (seeded). It keeps each airport's factor distribution and breaks the
    key identity."""
    rng = np.random.default_rng(56000 + seed)
    out = tab["true"].to_numpy("float64").copy()
    apt = tab["value"].astype(str).str.split("|").str[0].to_numpy()
    fam = tab["key"].astype(str).to_numpy()
    grp = pd.Series(np.arange(len(tab))).groupby([fam, apt]).indices
    for key in sorted(grp):
        idx = np.sort(grp[key])
        out[idx] = out[idx][rng.permutation(len(idx))]
    return pd.Series(out, index=tab.index)


# ---- the correction (SPEC 3.3): additive key effects shrunk toward beta * prior -------------
def design(codes: dict, nlev: dict, rows: np.ndarray | None = None) -> sp.csr_matrix:
    """One-hot design over the correction families; a missing key (code -1) has no column."""
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


def prior_vector(prior: dict, nlev: dict) -> np.ndarray:
    return np.concatenate([prior[f] for f in PRIOR_FAMS])


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


def select(parts: dict, y: dict, p: dict, pop: dict) -> tuple[dict, list]:
    """Pick (lam, beta) on the two inner directions (fit S -> score I, fit I -> score S) by the
    pooled raw squared error on the scored month's population rows with a label.

    `parts[d][lam] = (h0, h1)`: the scored month's h at beta 0 and the beta slope, direction d.
    Ties go to the smaller beta, then the larger lam."""
    best, table = None, []
    for beta in BETAS:
        for lam in sorted(LAMBDAS, reverse=True):
            sse = 0.0
            for d in ("S>I", "I>S"):
                h0, h1 = parts[d][lam]
                sc = "I" if d == "S>I" else "S"
                m = pop[sc] & np.isfinite(y[sc])
                pr = np.maximum(MIN_PRED, p[sc][m] + h0[m] + beta * h1[m])
                sse += float(np.sum((y[sc][m] - pr) ** 2))
            table.append({"lam": lam, "beta": beta, "sse": sse})
            if best is None or sse < best["sse"]:
                best = {"lam": lam, "beta": beta, "sse": sse}
    return best, table


#: route A, iteration 2 (SPEC 3.9): support-aware prior f_k * c / (c + n_A(k))
C_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0)


def support_weights(nA: dict, c: float) -> np.ndarray:
    """c / (c + n_A(k)) per correction column, each family on its own key's A-month count."""
    return np.concatenate([c / (c + nA[f].astype("float64")) for f in PRIOR_FAMS])


def select_c(parts: dict, y: dict, p: dict, pop: dict) -> tuple[dict, list]:
    """Route A: `select` with the support constant c as a third grid axis. `parts[c][d][lam]`
    as in `select`. Ties go to the smaller beta, then the larger lam, then the larger c."""
    best, table = None, []
    for beta in BETAS:
        for lam in sorted(LAMBDAS, reverse=True):
            for c in sorted(parts, reverse=True):
                sse = 0.0
                for d in ("S>I", "I>S"):
                    h0, h1 = parts[c][d][lam]
                    sc = "I" if d == "S>I" else "S"
                    m = pop[sc] & np.isfinite(y[sc])
                    pr = np.maximum(MIN_PRED, p[sc][m] + h0[m] + beta * h1[m])
                    sse += float(np.sum((y[sc][m] - pr) ** 2))
                table.append({"c": c, "lam": lam, "beta": beta, "sse": sse})
                if best is None or sse < best["sse"]:
                    best = {"c": c, "lam": lam, "beta": beta, "sse": sse}
    return best, table


def thin_mask(ids: np.ndarray, pop: np.ndarray, n_keep: int) -> np.ndarray:
    """The n_keep population rows with the smallest salted id hash (label-free, fixed)."""
    idx = np.flatnonzero(pop)
    hv = np.array([int(hashlib.sha256(f"{ORIGIN_SALT}|{int(i)}".encode()).hexdigest()[:15], 16)
                   for i in ids[idx]], dtype="int64")
    keep = idx[np.argsort(hv, kind="stable")[:n_keep]]
    m = np.zeros(len(ids), bool)
    m[keep] = True
    return m


# ---- statistics (SPEC 3.7) ------------------------------------------------------------------
def day_index(days: np.ndarray) -> tuple[np.ndarray, list]:
    uniq = sorted(set(days.tolist()))
    pos = {d: i for i, d in enumerate(uniq)}
    return np.array([pos[d] for d in days]), uniq


def block_draws(k: int, L: int, seed: int) -> np.ndarray:
    """Moving blocks of L consecutive dates, BOOT draws, k dates per draw (trimmed)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(k / L))
    st = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
    return np.minimum((st[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT, -1), k - 1)[:, :k]


def draw_seed(r: str, L: int) -> int:
    return BOOT_SEED * 1000 + ORDER.index(r) * 10 + L


def ci(x: np.ndarray) -> list:
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def load_json(p) -> dict:
    return json.loads(Path(p).read_text())
