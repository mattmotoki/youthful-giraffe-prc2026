"""exp-053 shared loaders: the pinned fold panels and head vectors, the ranking-side regime,
the COMMON.md no-NM mask, and write-once output.  Every source is sha-checked against SPEC §2.

Nothing here computes a loss.  `pin.py` reconciles the all-row faces; `instrument.py` reads the
stratum losses.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E46 = ROOT / "experiments/exp-046-surfacefit"
E50 = ROOT / "experiments/exp-050-nmblind"
ANALYSIS = EXP / "analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E46 / "code"))

import prc.data as D                                                  # noqa: E402
import nooverwrite as NO                                              # noqa: E402

FOLDS = ("july_loo", "december", "january")
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
#: SPEC §1: July 2025 maps only to ranking 2026-07, January 2025 only to ranking 2026-01.
RANK_MONTH = {"july_loo": "2026-07", "january": "2026-01"}
BOARD_FOLDS = ("july_loo", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}   # exp-046 base.SHARES
CUT = 3600.0
N_MIN, D_MIN = 300, 10                                                      # SPEC §3
REGIMES = ("H", "G", "A", "C")

PANEL = {f: E50 / f"work/panel-v22-{f}.parquet" for f in FOLDS}
PANEL_SHA = {"july_loo": "fc8700e9910f03e7f353624cc2d22bf252b81105032e905e6339bb5930b3630c",
             "december": "8b3d855ca727f88e441063bf18eaa93213e5d081fda1bc7606c4ba06889e4846",
             "january": "4672b7b3784b57f3903b5e58a6dd84ffa8ce67c86989874974d6fc59c8be3d62"}
#: level -> (upload, fold head vector, sha256); SPEC §2, = exp-046 serve.BASE_SETS v22-v25
HEADS = {
    "v19": (E46 / "work/d-bag10-255-free.npz",
            "d32a993b48d18fa428cb6c0127de34a5ccb0ce5ad09d2dd0a3e89db7a35cd26e"),
    "v20": (E46 / "work/d-onemonth-bag10-255-free.npz",
            "e1af652c0e5b07a8786532005e1854129470138a07f0999c38d002d6559a871e"),
    "v21": (E46 / "work/d-threemonth-bag10-255-free.npz",
            "1e0d0085b6ecea726e8da603554bfff95eef85e1afc3bc17a703de59d7b627c6"),
    "v22": (E46 / "work/d-seasonal-bag10-255-free.npz",
            "3edfa35fee55e1d03ec2df72192eeb596732e62f8164b162e9ef42597ecea6f5"),
}
LEVELS = ("v19", "v20", "v21", "v22")
STEPS = (("v20", "v19", "v20"), ("v21", "v20", "v21"), ("v22", "v21", "v22"))
CUMULATIVE = ("v22|v19", "v19", "v22")
#: full-precision recorded faces (exp-046 analysis/*-bag10.json, board_all_row.gain_s)
FACE_RECORD = {
    "v20": (E46 / "analysis/onemonth-bag10.json", "in_fold_over_v8_head"),
    "v21": (E46 / "analysis/threemonth-bag10.json", "in_fold_over_v9_head"),
    "v22": (E46 / "analysis/seasonal-bag10.json", "in_fold_over_v10_head"),
}
#: read-only board comparators (docs/SUBMISSIONS.md rows 19-22).  score-feedback rule: nothing is fitted,
#: selected, scaled or changed from these.
BOARD_GAIN = {"v20": 245.7752 - 245.2901, "v21": 245.2901 - 245.094, "v22": 245.094 - 245.0207}

RANKING = ROOT / "data/raw/ranking.parquet"
RANKING_SHA = "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86"
CLASS = {v: {m: ROOT / f"artifacts/exp-046/class/{m}-{v}.parquet" for m in ("2026-01", "2026-07")}
         for v in ("v22", "v23", "v24", "v25")}
CLASS_SHA = {
    "v22": {"2026-01": "4c9a91b6cd760277445ab78d80dc131287feec9ffd6e0a72c7a71c22d556f97f",
            "2026-07": "a3a4aa3be298240d7b93bc0d723f83e5d902f25f146340d1acdc799fd28869be"},
    "v23": {"2026-01": "8cef2f2d8175be9c846650dc494439b1cda2f2c94d154d5bcf269addd6e23cb2",
            "2026-07": "e69faf3a46e8ea61ad71dc6cb3b74e74ab3add1720b84524015eb45f8a2bca3f"},
    "v24": {"2026-01": "e6be6c0fdedeb5807f7ea99c57a3037e36a2ba66b4fbaeb1db191f14df801465",
            "2026-07": "0db0de12d33cb39b9d40351a094fe8a6838a78341902a2abc9a9696292809b15"},
    "v25": {"2026-01": "e69f78b4f7d17fa3c0c95151fc171f2c7cd9022efd79489add1bbce119494697",
            "2026-07": "4eee477d219d32fa2ebb31346dbb105af0736e37ceaf786ec0eb0d0906be13fe"},
}
#: the upload that carried each class vector
CLASS_OF = {"v19": "v22", "v20": "v23", "v21": "v24", "v22": "v25"}
UPLOAD = {v: ROOT / f"submissions/uploaded/youthful-giraffe_{v}.parquet" for v in LEVELS}
UPLOAD_SHA = {"v19": "d03d132a5ff5d3fc70a72b9f60a1b418a169e911ac32ef08b14378695932aa14",
              "v20": "8fd4b34166b61756a20746c77f97fc1d06d06af28ee959ef7deac1de0a365d78",
              "v21": "7b0557a497f8c09e4f56a11352783b9dda37fe1d0579345d4303e4f74d6e7aca",
              "v22": "f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c"}
OWNERSHIP_COLS = ("occupied_by_exp_046", "guard_held_by_exp_046", "noac", "served_by_exp_046",
                  "obs_has_clock", "chain_mode_4")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def check(p: Path, want: str) -> str:
    got = sha_file(p)
    if got != want:
        raise SystemExit(f"{p.relative_to(ROOT)} is {got[:16]}, not the registered {want[:16]}")
    return got


def load_panel(f: str) -> pd.DataFrame:
    check(PANEL[f], PANEL_SHA[f])
    return pd.read_parquet(PANEL[f])


def load_heads() -> dict:
    """{level: {fold: float64 vector}}, sha-checked."""
    out = {}
    for lv, (p, want) in HEADS.items():
        check(p, want)
        z = np.load(p)
        out[lv] = {f: z[f].astype("float64") for f in FOLDS}
    return out


def fold_regime(p: pd.DataFrame) -> np.ndarray:
    """SPEC §3 on the fold side: H = usable & ~guard_held, G = guard_held, A = noac, else C."""
    usable = p.usable.to_numpy(bool)
    guard = p.guard_held.to_numpy(bool)
    noac = p.noac.to_numpy(bool)
    if np.any(guard & ~usable):
        raise SystemExit("a guard-held fold row is outside the served class")
    if np.any(noac & usable):
        raise SystemExit("a noac fold row is in the served class")
    r = np.full(len(p), "C", dtype=object)
    r[usable & ~guard] = "H"
    r[guard] = "G"
    r[noac] = "A"
    return r


def load_ranking() -> pd.DataFrame:
    """Ranking departures: MVT_ID_mvt, month (class-vector file), airport, regime, mode-4,
    clock flag.  Ownership from the v22 manifest (class vector v25)."""
    check(RANKING, RANKING_SHA)
    rk = pd.read_parquet(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"].copy()
    rk["MVT_ID_mvt"] = rk.MVT_ID_mvt.astype("float64")
    if not rk.MVT_ID_mvt.is_unique:
        raise SystemExit("ranking departure ids are not unique")
    parts = []
    for m in ("2026-01", "2026-07"):
        check(CLASS["v25"][m], CLASS_SHA["v25"][m])
        c = pd.read_parquet(CLASS["v25"][m])
        c["month"] = m
        parts.append(c)
    cv = pd.concat(parts, ignore_index=True)
    cv["MVT_ID_mvt"] = cv.MVT_ID_mvt.astype("float64")
    if not cv.MVT_ID_mvt.is_unique or len(cv) != len(rk):
        raise SystemExit(f"class vector v25 has {len(cv)} rows, the ranking {len(rk)} departures")
    ap = rk.set_index("MVT_ID_mvt").ADEP_mvt.reindex(cv.MVT_ID_mvt.to_numpy())
    if ap.isna().any():
        raise SystemExit("a class-vector row is not a ranking departure")
    cv["airport"] = ap.astype(str).to_numpy()
    occ = cv.occupied_by_exp_046.to_numpy(bool)
    guard = cv.guard_held_by_exp_046.to_numpy(bool)
    noac = cv.noac.to_numpy(bool)
    if np.any(guard & ~occ):
        raise SystemExit("a guard-held ranking row is not occupied")
    if np.any(noac & occ):
        raise SystemExit("a noac ranking row is occupied")
    r = np.full(len(cv), "C", dtype=object)
    r[occ & ~guard] = "H"
    r[guard] = "G"
    r[noac] = "A"
    cv["regime"] = r
    return cv


def no_nm_mask(ids: np.ndarray, f: str) -> np.ndarray:
    """COMMON.md's no-NM rows: every `_flt` column of the raw month null (exp-051
    `populations.no_nm_mask`, reproduced)."""
    month = FOLD_MONTH[f]
    flt = [c for c in pq.read_schema(D.month_file(month)).names if c.endswith("_flt")]
    raw = D.load_month(month, ["MVT_ID_mvt", "PHASE_mvt"] + flt)
    dep = raw[raw.PHASE_mvt == "DEP"]
    direct = pd.Series(dep[flt].isna().all(axis=1).to_numpy(),
                       index=dep.MVT_ID_mvt.to_numpy("float64"))
    if not direct.index.is_unique or not np.isin(ids, direct.index.to_numpy()).all():
        raise SystemExit(f"{f}: panel ids do not map one to one onto {month}'s departures")
    return direct.reindex(ids).to_numpy(bool)


def day_block(days: pd.Series, L: int) -> np.ndarray:
    """SPEC §4.7: block = floor((day-of-month - 1) / L) of the UTC take-off day."""
    dom = pd.to_datetime(days).dt.day.to_numpy()
    return (dom - 1) // L


def kish(counts) -> float:
    c = np.asarray(counts, dtype="float64")
    return float(c.sum() ** 2 / (c ** 2).sum()) if c.sum() > 0 else 0.0


def write_json(p: Path, obj: dict) -> None:
    NO.refuse_existing([p], overwrite=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
