"""The chain columns this experiment puts inside `L2-hub8725`, and nothing else.

Two arms of new inputs, both computed from a RANKING-MASKED month frame:

  `chain`    `ts_gap_arr1`, `ts_gap_arr2`, `ts_gap_arr3` -- the three stand-chain
             gaps exactly as `prc/tail_features.py::_stand_signals` computes
             them, imported the way the retro **v4** frame imports them
             (`experiments/exp-001-retro-v1/features.py`, `TAIL_IMPORT`).  These
             are the columns the assignment calls "the chain", and exp-037 §0
             measured `g_any == gap_arr1` and `g_prev2 == gap_arr2` at 0 s on the
             noac rows.  `equality()` below extends that check to every
             departure, which is the population Q09 is measured on.

  `derived`  the same three, plus the five columns exp-037 ADDED to a head that
             already had the chain: `slack`, `slack2`, `log_g`, `n_dep_between`,
             `arr_stand_per_day` (`experiments/exp-037-tail/code/frames.py`,
             `CHAIN_FEATS`).  exp-037's `g_any` and `g_prev2` are NOT added on
             top: they are the same numbers as `ts_gap_arr1`/`ts_gap_arr2` (see
             `analysis/equality.csv`), so adding them would put two identical
             float columns in front of a tree that samples 80 % of its features.
             The seven-column list in the assignment is therefore served as
             three shared columns plus five genuinely new ones.

`n_dep_between` and `arr_stand_per_day` come from exp-037's own `chain.py`,
imported rather than re-derived, so the second arm is exp-037's instrument and
not a look-alike.  `slack`, `slack2` and `log_g` are its `add_chain_feats`
arithmetic written out on the columns this frame has.

Label-free by construction: the only entry point takes the masked frame, and
`stand_chain` raises if a departure label is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

from prc.tail_features import tail_signals  # noqa: E402


def _load(name: str, path: Path):
    """Import one file by path, without putting its directory on `sys.path`.

    `experiments/exp-037-tail/code/` holds a `frames.py` of its own, and this
    experiment swaps a module called `frames` under `exp-008`'s `run.py`.  Two
    files of that name on one path is exactly the accident that would make the
    member fit on the wrong frame and still finish, so exp-037's chain is loaded
    by address instead.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


stand_chain = _load("_exp037_chain", ROOT / "experiments/exp-037-tail/code/chain.py").stand_chain

#: The three chain gaps, named as the v4 frame names them.
CHAIN = ("ts_gap_arr1", "ts_gap_arr2", "ts_gap_arr3")
#: What exp-037 added on top of a head that already had the chain.
DERIVED = ("slack", "slack2", "log_g", "n_dep_between", "arr_stand_per_day")
ARMS = {"none": (), "chain": CHAIN, "derived": CHAIN + DERIVED}


def columns(masked: pd.DataFrame, arm: str, month: str) -> pd.DataFrame:
    """The arm's new columns for one masked month frame, indexed like its departures.

    `month` is `YYYY-MM`: `arr_stand_per_day` divides a whole-month arrival count
    by the month's length, which is exp-037's own normalisation and the reason
    the column is comparable between a 28-day and a 31-day month.
    """
    want = ARMS[arm]
    dep_idx = masked.index[(masked["PHASE_mvt"] == "DEP").to_numpy()]
    out = pd.DataFrame(index=dep_idx)
    if not want:
        return out

    sig = tail_signals(masked, airports=None)
    for c in CHAIN:
        out[c] = sig[c.removeprefix("ts_")].reindex(dep_idx).to_numpy("float64")
    if not any(c in want for c in DERIVED):
        return out

    ch = stand_chain(masked).reindex(dep_idx)
    mvt_sched = sig["mvt_sched"].reindex(dep_idx).to_numpy("float64")
    g_any = ch["g_any"].to_numpy("float64")
    g_prev2 = ch["g_prev2"].to_numpy("float64")
    out["slack"] = g_any - mvt_sched
    out["slack2"] = g_prev2 - mvt_sched
    out["log_g"] = np.log1p(np.clip(g_any, 0, None))
    out["n_dep_between"] = ch["n_dep_between"].to_numpy("float64")
    dim = float(pd.Timestamp(month + "-01").days_in_month)
    out["arr_stand_per_day"] = ch["n_arr_stand"].to_numpy("float64") / dim
    return out.loc[:, list(want)]


def equality(masked: pd.DataFrame) -> dict:
    """Is `tail_features`'s chain the same object as exp-037's, on EVERY departure?

    exp-037 §0 measured this on the 1,485 chained 2025 noac rows and the 386
    chained ranking rows.  Q09 is scored on the ordinary rows, so the claim it
    rests on is checked on the whole month.  Returns counts and the maximum
    absolute difference where both are defined, plus the disagreement in WHERE
    each one is defined -- `stand_chain` drops a null stand, `_stand_signals`
    keys on the string `"nan"` and does not.
    """
    dep_idx = masked.index[(masked["PHASE_mvt"] == "DEP").to_numpy()]
    sig = tail_signals(masked, airports=None).reindex(dep_idx)
    ch = stand_chain(masked).reindex(dep_idx)
    rec = {"n_dep": int(len(dep_idx))}
    for a, b in (("gap_arr1", "g_any"), ("gap_arr2", "g_prev2")):
        u = sig[a].to_numpy("float64")
        v = ch[b].to_numpy("float64")
        du, dv = np.isfinite(u), np.isfinite(v)
        both = du & dv
        d = np.abs(u[both] - v[both])
        rec[f"{a}_n_ts"] = int(du.sum())
        rec[f"{a}_n_chain"] = int(dv.sum())
        rec[f"{a}_n_both"] = int(both.sum())
        rec[f"{a}_max_abs_diff"] = float(d.max()) if both.any() else float("nan")
        rec[f"{a}_n_diff"] = int((d > 0).sum())
        rec[f"{a}_n_ts_only"] = int((du & ~dv).sum())
        rec[f"{a}_n_chain_only"] = int((dv & ~du).sum())
    null_stand = masked.loc[dep_idx, "STAND_mvt"].isna().to_numpy()
    rec["n_null_stand"] = int(null_stand.sum())
    return rec
