"""exp-018 fresh lineage: cached per-month feature tables and fold assembly."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc import data  # noqa: E402
import features as F  # noqa: E402

CACHE = data.ROOT / "artifacts" / "exp-018-fresh" / "cache"
VOCAB = CACHE / "vocab.json"


def _vocab_encode(X: pd.DataFrame) -> pd.DataFrame:
    """Replace the string category columns by int32 codes from a persistent,
    append-only vocabulary, so no fold has to re-hash two million strings."""
    import json
    voc = json.loads(VOCAB.read_text()) if VOCAB.exists() else {}
    for c in F.CAT_COLS:
        vals = voc.get(c, [])
        idx = pd.Index(vals)
        col = X[c].astype(str)
        codes = idx.get_indexer(col)
        if (codes < 0).any():
            new = pd.unique(col[codes < 0])
            vals = list(vals) + list(new)
            voc[c] = vals
            codes = pd.Index(vals).get_indexer(col)
        else:
            voc[c] = list(vals)
        X[c] = codes.astype("int32")
    VOCAB.write_text(json.dumps(voc))
    return X
KEEP = ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt"]


def month_table(month: str, rebuild: bool = False) -> pd.DataFrame:
    """Features + label for one 2025 month, or for a 2026 ranking month."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # The cache key MUST carry the feature-set flag. Keyed on the month alone, a
    # PRC_WINDOW=1 run silently reuses a flag-off table and reports the base
    # member's score as the arm's -- a failure that costs GPU and reads as a
    # clean negative. See experiments/exp-027-step/SPEC.md.
    suffix = ""
    if os.environ.get("PRC_WINDOW") == "1":
        # The scale set is part of the feature set, so it is part of the key.
        # Unset leaves the suffix at exactly "-w", the shipped K2 block's key.
        import window_features as _wf
        suffix = "-w"
        if _wf.SCALES != (600.0, 3600.0):
            suffix += "-" + "-".join(str(int(x)) for x in _wf.SCALES)
    p = CACHE / f"{month}{suffix}.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    if month.startswith("2025"):
        frame = data.load_month(month)
        masked, labels = data.mask_like_ranking(frame)
        X = F.build(masked)
        X["y"] = labels.reindex(X.index).to_numpy(dtype="float64")
    else:
        rk = data.load_ranking()
        frame = rk[rk["month"] == month]
        X = F.build(frame)
        X["y"] = np.nan
    dep = frame.loc[X.index]
    X["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy()
    X["month"] = month
    X = _vocab_encode(X)
    X = X.reset_index(drop=True)
    X.to_parquet(p, index=False)
    return X


def load(months) -> pd.DataFrame:
    return pd.concat([month_table(m) for m in months], ignore_index=True)


# ------------------------------------------------------------ target encoding
def _key(df: pd.DataFrame, cols) -> pd.Series:
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "\x1f" + df[c].astype(str)
    return k


ENC_SPECS = [
    ("apt", ["c_apt"]),
    ("rwy", ["c_rwy"]),
    ("stand", ["c_stand"]),
    ("standrwy", ["c_stand", "c_rwy"]),
    ("standpfxrwy", ["c_standpfx", "c_rwy"]),
    ("actype", ["c_apt", "c_actype"]),
    ("oper", ["c_apt", "c_oper"]),
    ("rwyhour", ["c_rwy", "hourbin"]),
    ("ades", ["c_apt", "c_adescty"]),
]


def add_encodings(train: pd.DataFrame, targets: list[pd.DataFrame], smooth: float = 30.0):
    """Fit smoothed group means on `train` and apply them to every frame in
    `targets`. The training rows get a leave-one-month-out version of the same
    statistic, so a training row never sees its own month's mean."""
    tr = train
    tr["hourbin"] = (tr["hour"] // 3).astype("int8")
    for t in targets:
        t["hourbin"] = (t["hour"] // 3).astype("int8")
    y = tr["y"].to_numpy(dtype="float64")
    off = y - tr["r_aobt"].to_numpy(dtype="float64")
    prior = {"y": float(np.nanmedian(y)), "off": float(np.nanmedian(off))}
    mcodes, months = pd.factorize(tr["month"].to_numpy())
    nm = len(months)
    for name, cols in ENC_SPECS:
        codes, uniq = pd.factorize(_key(tr, cols).to_numpy(), sort=False)
        K = len(uniq)
        tcodes = [pd.Index(uniq).get_indexer(_key(t, cols).to_numpy()) for t in targets]
        cnt_all = np.bincount(codes, minlength=K).astype("float64")
        tr[f"n_{name}"] = cnt_all[codes]
        for t, tc in zip(targets, tcodes):
            t[f"n_{name}"] = np.where(tc >= 0, cnt_all[np.maximum(tc, 0)], 0.0)
        cm = codes * nm + mcodes
        for tag, v in (("y", y), ("off", off)):
            ok = np.isfinite(v)
            pr = prior[tag]
            s_all = np.bincount(codes[ok], weights=v[ok], minlength=K)
            n_all = np.bincount(codes[ok], minlength=K).astype("float64")
            s_m = np.bincount(cm[ok], weights=v[ok], minlength=K * nm)
            n_m = np.bincount(cm[ok], minlength=K * nm).astype("float64")
            loo = (s_all[codes] - s_m[cm] + smooth * pr) / (n_all[codes] - n_m[cm] + smooth)
            tr[f"te_{tag}_{name}"] = loo
            enc = (s_all + smooth * pr) / (n_all + smooth)
            for t, tc in zip(targets, tcodes):
                t[f"te_{tag}_{name}"] = np.where(tc >= 0, enc[np.maximum(tc, 0)], pr)
    return tr, targets


ENC_COLS = ([f"te_y_{n}" for n, _ in ENC_SPECS] + [f"te_off_{n}" for n, _ in ENC_SPECS]
            + [f"n_{n}" for n, _ in ENC_SPECS])
