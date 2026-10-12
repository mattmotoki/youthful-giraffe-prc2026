"""exp-052-nmv22: shared inputs, checked by hash, and the registered constants (SPEC section 1).

Everything that fixes the instrument lives here, so that the fit job (`rx_fit.py`) and the
pricing job (`rx_price.py`) cannot disagree on the population, the blocks, the columns, the
recipe or a seed.  exp-050's code is imported READ-ONLY and its files are sha-checked first:
the population, the date blocks, the feature builder, the encoder and the statistics are
Q08's own, not re-derived.
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E50 = ROOT / "experiments/exp-050-nmblind"
for p in (HERE, ROOT, E50 / "code", ROOT / "experiments/exp-046-surfacefit/code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def sha_file(p: Path) -> str:
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


# ---- exp-050's files, pinned (SPEC "Inputs") --------------------------------------------
PINNED = {
    E50 / "code/nb_features.py": "4fb3ae467c90862c8119d4feeaf71f64e1d5b279787e32f41bea43def058bec2",
    E50 / "code/nb_router.py": "73a4486bd23f2ca7949e4c476e9abc461f2dd48711ed1b38ef2268e96a2c9be8",
    E50 / "code/nb_experts.py": "4b482b862793e3497511f90fe545fbacadbd97d211aaf84996f9de6891759324",
    E50 / "code/nb_panel.py": "4f8da0a13bd2d94fe83b814d0f2029007a3d3a53900673135fc9656e40c14b03",
    E50 / "analysis/panel.json": "c091246f2fadac4feff277e5f76a7f1f71940cf38a27063687be1fd8a86e2f0f",
    E50 / "analysis/features.json": "7cc15f7387b3c69166871e1b8af8880405c0d15c08435cd62d78ab92be3a211e",
    E50 / "analysis/router-july_loo-december-january.json":
        "571f0510578ef104f5be067d42ce0363b309676ebfeaf8ad4a4a73d86ee89454",
    E50 / "work/vocab.json": "f748bcc93b254622d262b6d46f52ed046b79e1cde9cc4e1fb792ddafbe8a482d",
}


def check_pinned() -> dict:
    out = {}
    for p, want in PINNED.items():
        got = sha_file(p)
        if got != want:
            raise SystemExit(f"{p.relative_to(ROOT)} is {got[:16]}, not the pinned {want[:16]}")
        out[str(p.relative_to(ROOT))] = got
    return out


check_pinned()
import nb_experts as NE                                               # noqa: E402
import nb_features as NF                                              # noqa: E402
import nb_panel as NP                                                 # noqa: E402
import nb_router as NR                                                # noqa: E402
import nooverwrite as NO                                              # noqa: E402
import prc.data as D                                                  # noqa: E402

FOLDS = ["july_loo", "december", "january"]
K = 5                                   # Q08's date blocks
PAIRS = [(b, j) for b in range(K) for j in range(b + 1, K)]
TYPES = ["blind", "aware", "sham"]
COLS = {"blind": NF.BLIND_FEATURES, "aware": NF.AWARE_FEATURES, "sham": NF.AWARE_FEATURES}

# ---- the recipe (SPEC 1.4) ----------------------------------------------------------------
THREADS = 3
SEED = 52
ROUNDS = 400
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31,
              min_child_samples=300, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
              reg_lambda=10.0, max_bin=255, cat_smooth=10.0, min_data_per_group=200,
              max_cat_threshold=32, deterministic=True, force_row_wise=True, verbose=-1)
LOAD_LIMIT = 10.0

# ---- seeds (SPEC 1.2, 1.6, 1.8, 1.9) --------------------------------------------------------
SEED_SHAM = 5201
SEED_SHUFFLE = 5202
N_PERM = 50
SEEDS_PLACEBO = list(range(5300, 5300 + N_PERM))
SEEDS_ORACLE_D22 = list(range(5400, 5400 + N_PERM))
SEEDS_ORACLE_SHAM = list(range(5500, 5500 + N_PERM))


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hold_load(what: str) -> dict:
    """The assignment: before every fit launch, hold while the 1-minute load is above 10."""
    while True:
        l1 = os.getloadavg()[0]
        if l1 <= LOAD_LIMIT:
            break
        print(f"  {utc()} load {l1:.2f} > {LOAD_LIMIT:g}: holding {what}", flush=True)
        time.sleep(30)
    return {"launch_utc": utc(), "launch_load_1m": round(float(l1), 2)}


def population_sha_q08() -> dict:
    rj = json.loads((E50 / "analysis/router-july_loo-december-january.json").read_text())
    return {f: rj["folds"][f]["population_ids_sha256"] for f in FOLDS}


def panel(fold: str, columns: list[str]) -> pd.DataFrame:
    """exp-050's frozen v22 panel, sha-checked against its record, only `columns` read."""
    rec = json.loads((E50 / "analysis/panel.json").read_text())["folds"][fold]
    p = NP.panel_path(fold)
    if sha_file(p) != rec["parquet_sha256"]:
        raise SystemExit(f"{p.name} moved since exp-050 froze it")
    return pd.read_parquet(p, columns=columns)


LABEL_FREE = ["MVT_ID_mvt", "airport", "day", "regime", "guard_held", "clean", "p_v22"]


def population(pan: pd.DataFrame, fold: str) -> np.ndarray:
    """P = nm_bulk and not guard-held (Q08 1.1); its sorted-id sha must equal Q08's."""
    pop = (pan.regime.astype(str).to_numpy() == "nm_bulk") & ~pan.guard_held.to_numpy(bool)
    got = sha_arr(np.sort(pan.MVT_ID_mvt.to_numpy("float64")[pop]))
    want = population_sha_q08()[fold]
    if got != want:
        raise SystemExit(f"{fold}: the population sha {got[:16]} is not Q08's {want[:16]}")
    return pop


def blocks(pan: pd.DataFrame) -> np.ndarray:
    return NR.dayblocks(pan.day.to_numpy())


def features(fold: str, cols: list[str]) -> pd.DataFrame:
    """The frozen feature parquet of the fold month (sha-checked by exp-050's loader)."""
    F = NE.load_month(D.FOLDS[fold].valid_month)
    return F[["MVT_ID_mvt"] + cols]


def airday_perm(key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of positions that moves each row only inside its own key group.

    Positions sorted by group (original order) receive the values of the same group's
    positions sorted by (group, random draw): out[base] = src[shuf], so new[i] = old[perm[i]].
    """
    codes, _ = pd.factorize(pd.Series(key), sort=True)
    base = np.argsort(codes, kind="stable")
    shuf = np.lexsort((rng.random(len(codes)), codes))
    perm = np.empty(len(codes), dtype=np.int64)
    perm[base] = shuf
    return perm


SEED_NULL = 5801                        # SPEC 1.15 (a)


def null_residual(r: np.ndarray, fitmask: np.ndarray, key: np.ndarray) -> np.ndarray:
    """SPEC 1.15 (a): r22 permuted among the fitting rows of P inside airport x day; the
    rows of P outside the fitting rows keep their own value."""
    idx = np.flatnonzero(fitmask)
    perm = airday_perm(key[idx], np.random.default_rng(SEED_NULL))
    out = r.copy()
    out[idx] = r[idx][perm]
    return out


def tagged(name: str, null: bool) -> str:
    return f"{name}-null" if null else name


def resid_path(fold: str, out_dir: Path | None = None, null: bool = False) -> Path:
    return (out_dir or EXP / "work") / f"{tagged('resid', null)}-{fold}.parquet"


def experts_json(fold: str, out_dir: Path | None = None, null: bool = False) -> Path:
    return (out_dir or EXP / "analysis") / f"{tagged('experts', null)}-{fold}.json"
