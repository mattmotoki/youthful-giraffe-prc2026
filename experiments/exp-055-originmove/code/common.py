"""exp-055 shared loaders (SPEC sections 2-4).  Every source is sha-checked against
`analysis/prehash.json`, the byte census committed with the registration.

Label discipline for iteration 1: every fold-panel read names its columns and never includes
`y_true` or `clean`; the ranking read names input columns only; exp-051's loss records are not
opened.  Iteration 2's label reads live in its own script.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E46 = ROOT / "experiments/exp-046-surfacefit"
E50 = ROOT / "experiments/exp-050-nmblind"
E51 = ROOT / "experiments/exp-051-transferadv"
E53 = ROOT / "experiments/exp-053-transport"
W51 = ROOT / "artifacts/exp-051/work"
ANALYSIS = EXP / "analysis"
RECEIPTS = ANALYSIS / "receipts"
sys.path.insert(0, str(E46 / "code"))
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import nooverwrite as NO                                              # noqa: E402

FOLDS = ("july_loo", "december", "january")
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
RANK_MONTH = {"july_loo": "2026-07", "january": "2026-01"}             # Q15's month mapping
RMONTHS = ("2026-01", "2026-07")
REGIMES = ("H", "G", "A", "C")
NM_REGIMES = ("nm_bulk", "nm_late", "nonm")
STEP = {1: "v20", 2: "v21", 3: "v22"}
LEVELS10 = ("v19", "v20", "v21", "v22")                                # K = 10 heads = L0..L3
SEEDS = (1, 2, 3)
#: SPEC section 3.1, Q15 pin.json (read back and compared in pin.py)
H_ROWS = {"july_loo": 88428, "december": 102186, "january": 69840}
H_ROWS_RANK = {"2026-07": 104609, "2026-01": 90074}

PREHASH = json.loads((ANALYSIS / "prehash.json").read_text())
PATH_SHA = {v["path"]: v["sha256"] for v in PREHASH["files"].values()}

PANEL = {f: E50 / f"work/panel-v22-{f}.parquet" for f in FOLDS}
#: label-free panel columns (SPEC: no y_true, no clean in iteration 1)
PANEL_COLS = ["MVT_ID_mvt", "airport", "t", "day", "regime", "noac", "usable", "guard_held",
              "pbase", "d_v22", "p_v22", "obs_has_clock"]
HEADS = {"v19": E46 / "work/d-bag10-255-free.npz", "v20": E46 / "work/d-onemonth-bag10-255-free.npz",
         "v21": E46 / "work/d-threemonth-bag10-255-free.npz",
         "v22": E46 / "work/d-seasonal-bag10-255-free.npz"}
RANKING = ROOT / "data/raw/ranking.parquet"
RANK_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]    # inputs only
CLASS = {v: {m: ROOT / f"artifacts/exp-046/class/{m}-{v}.parquet" for m in RMONTHS}
         for v in ("v22", "v23", "v24", "v25")}
UPLOAD = {v: ROOT / f"submissions/uploaded/youthful-giraffe_{v}.parquet" for v in LEVELS10}
OOF51 = {(it, f): E51 / f"oof/{it}-{f}.parquet" for it in ("i1", "i2") for f in FOLDS}
ONLY0306 = E46 / "work/d-onemonth-only202506-255-free.npz"


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(*arrays) -> str:
    """exp-053 `common.sha_arr`: SHA-256 of the arrays' bytes in order."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def check(p: Path) -> str:
    """The file's bytes must carry the SHA-256 that prehash.json registered for its path."""
    want = PATH_SHA.get(rel(p))
    if want is None:
        raise SystemExit(f"{rel(p)} is not in the registered census")
    got = sha_file(p)
    if got != want:
        raise SystemExit(f"{rel(p)} is {got[:16]}, not the registered {want[:16]}")
    return got


def member_path(L: int, s: int) -> Path:
    """exp-051 `memberspread.member_path` (prehash.py carries the same copy)."""
    if L == 0:
        return E46 / f"work/d-bundle-placebo{s}-255-free.npz"
    if s == 1 and L in (1, 2):
        return {1: E46 / "work/d-onemonth-screen-255-free.npz",
                2: E46 / "work/d-onemonth-twomonth-plus03-255-free.npz"}[L]
    return E46 / "work" / {1: f"d-onemonth-m{s}-255-free.npz", 2: f"d-threemonth-m{s}-255-free.npz",
                           3: f"d-seasonal-m{s}-255-free.npz"}[L]


def carrier_path(L: int, s: int) -> Path:
    """June carrier member s at level L; seed 1 level 0 is the recorded integrity refit."""
    if s == 1 and L == 0:
        return W51 / "i2-integrity-C0-s1.npz"
    return W51 / f"i2-C-s{s}-L{L}.npz"


def bl_path(f: str, half: str, L: int) -> Path:
    return W51 / f"bl-{f}-{half}-L{L}.npz"


def sham_path(L: int) -> Path:
    return W51 / f"i2-S-s1-L{L}.npz"


def npz_fold(p: Path, f: str) -> np.ndarray:
    check(p)
    return np.load(p)[f].astype("float64")


def load_panel(f: str) -> pd.DataFrame:
    check(PANEL[f])
    return pd.read_parquet(PANEL[f], columns=PANEL_COLS)


def fold_regime(p: pd.DataFrame) -> np.ndarray:
    """Q15 `fold_regime`: H = usable & ~guard_held, G = guard_held, A = noac, else C."""
    usable = p.usable.to_numpy(bool)
    guard = p.guard_held.to_numpy(bool)
    noac = p.noac.to_numpy(bool)
    if np.any(guard & ~usable) or np.any(noac & usable):
        raise SystemExit("the fold ownership masks overlap")
    r = np.full(len(p), "C", dtype=object)
    r[usable & ~guard] = "H"
    r[guard] = "G"
    r[noac] = "A"
    return r


def regime_index(r: np.ndarray) -> np.ndarray:
    return np.array([REGIMES.index(x) for x in r], "int8")


def load_ranking() -> pd.DataFrame:
    """Q15 `load_ranking` (class vector v25 order, 2026-01 then 2026-07), with the take-off
    day added from `MVT_TIME_UTC_mvt`.  Inputs only."""
    check(RANKING)
    rk = pd.read_parquet(RANKING, columns=RANK_COLS)
    rk = rk[rk.PHASE_mvt == "DEP"].copy()
    rk["MVT_ID_mvt"] = rk.MVT_ID_mvt.astype("float64")
    if not rk.MVT_ID_mvt.is_unique:
        raise SystemExit("ranking departure ids are not unique")
    parts = []
    for m in RMONTHS:
        check(CLASS["v25"][m])
        c = pd.read_parquet(CLASS["v25"][m])
        c["month"] = m
        parts.append(c)
    cv = pd.concat(parts, ignore_index=True)
    cv["MVT_ID_mvt"] = cv.MVT_ID_mvt.astype("float64")
    if not cv.MVT_ID_mvt.is_unique or len(cv) != len(rk):
        raise SystemExit(f"class vector v25 has {len(cv)} rows, the ranking {len(rk)} departures")
    rki = rk.set_index("MVT_ID_mvt")
    ap = rki.ADEP_mvt.reindex(cv.MVT_ID_mvt.to_numpy())
    tk = rki.MVT_TIME_UTC_mvt.reindex(cv.MVT_ID_mvt.to_numpy())
    if ap.isna().any() or tk.isna().any():
        raise SystemExit("a class-vector row is not a ranking departure with a take-off time")
    cv["airport"] = ap.astype(str).to_numpy()
    cv["day"] = pd.to_datetime(tk.to_numpy(), utc=True).strftime("%Y-%m-%d")
    occ = cv.occupied_by_exp_046.to_numpy(bool)
    guard = cv.guard_held_by_exp_046.to_numpy(bool)
    noac = cv.noac.to_numpy(bool)
    if np.any(guard & ~occ) or np.any(noac & occ):
        raise SystemExit("the ranking ownership masks overlap")
    r = np.full(len(cv), "C", dtype=object)
    r[occ & ~guard] = "H"
    r[guard] = "G"
    r[noac] = "A"
    cv["regime"] = r
    return cv


def load_uploads(rid: np.ndarray) -> dict:
    up = {}
    for lv in LEVELS10:
        check(UPLOAD[lv])
        u = pd.read_parquet(UPLOAD[lv], columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
        u["MVT_ID_mvt"] = u.MVT_ID_mvt.astype("float64")
        s = u.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(rid)
        if s.isna().any():
            raise SystemExit(f"upload {lv} lacks {int(s.isna().sum())} ranking departures")
        up[lv] = s.to_numpy("float64")
    return up


def bag(members: dict, seeds, pbase: np.ndarray, usable: np.ndarray, guard: np.ndarray) -> np.ndarray:
    """exp-046 `bagarm.build_bag` for one fold, on the exp-050 panel's masks: the row mean of
    the members in the order given, 0 outside `usable`, the serving floor, 0 on guard rows."""
    mean = np.vstack([members[s] for s in seeds]).mean(0)
    mean = np.where(usable, mean, 0.0)
    mean = A.apply_floor(pbase, mean)
    mean[guard] = 0.0
    return mean


def mbb_starts(rng: np.random.Generator, n_days: int, L: int, B: int) -> np.ndarray:
    """SPEC 4.4: B draws of ceil(n/L) block starts, uniform on 0..n-L."""
    b = -(-n_days // L)
    return rng.integers(0, n_days - L + 1, size=(B, b))


def mbb_weights(starts: np.ndarray, n_days: int, L: int) -> np.ndarray:
    """Day multiplicities (B x n_days): concatenate the blocks, keep the first n days."""
    B, b = starts.shape
    seq = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, b * L)[:, :n_days]
    w = np.zeros((B, n_days))
    np.add.at(w, (np.repeat(np.arange(B), n_days), seq.ravel()), 1.0)
    return w


def kish(x) -> float:
    x = np.asarray(x, dtype="float64")
    return float(x.sum() ** 2 / (x ** 2).sum()) if (x ** 2).sum() > 0 else 0.0


def write_json(p: Path, obj: dict) -> None:
    NO.refuse_existing([p], overwrite=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
