"""The base this experiment corrects, and the canonical panel every feature reads.

The base is **stack-5 keep-noac**, read back from `experiments/exp-026-rebuild/oof/`
rather than refitted, and gated against the numbers exp-026 published
(`analysis/stack5.json`).  Reading it back is what makes it impossible for this
experiment and the blend lane to disagree about what stack-5 is; the gate is what
makes it impossible to correct a base that has silently moved.

The canonical panel is the one frame `features.window_features` accepts, from
either side of the fit/serve line:

    MVT_ID_mvt  airport  t  d_takeoff_aobt  d_aobt_lobt  regime  nm_missing  noac
    y_true  base

`t` is the take-off time in epoch seconds (`MVT_TIME_UTC_mvt`), frozen into
`exp-016`'s out-of-fold frame; `d_*`, `regime`, `nm_missing`, `noac` and `airport`
come from `exp-027-step`'s head frame, which took them from the members' own OOF.
On the serving side `features.panel_from_movements` builds the same eight columns
from the raw ranking file, and `tests/shape_check.py` checks the two agree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
#: exp-016 `members.SHARES`: the 2026 row counts of the two ranking months.
#: `december` is scored and gated, but it is not a board month and carries no share.
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

STACK5 = "experiments/exp-026-rebuild/oof/stack5-{fold}.parquet"
STACK5_JSON = "experiments/exp-026-rebuild/analysis/stack5.json"
HEAD = "experiments/exp-027-step/oof/blend5-{fold}.parquet"
X16 = "experiments/exp-016-stack/oof/exp-016-oof-{fold}.parquet"

CANON = ["MVT_ID_mvt", "airport", "t", "d_takeoff_aobt", "d_aobt_lobt",
         "regime", "nm_missing", "noac", "y_true", "base"]


def load_fold(fold: str) -> pd.DataFrame:
    """The canonical panel for one fold, every row of the month, in OOF order."""
    s = pd.read_parquet(ROOT / STACK5.format(fold=fold))
    h = pd.read_parquet(ROOT / HEAD.format(fold=fold))
    x = pd.read_parquet(ROOT / X16.format(fold=fold), columns=["MVT_ID_mvt", "t"])
    for d in (s, h, x):
        d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
    m = s.merge(h[["MVT_ID_mvt", "regime", "airport", "noac", "nm_missing",
                   "d_aobt_lobt", "d_takeoff_aobt"]], on="MVT_ID_mvt", how="inner")
    m = m.merge(x, on="MVT_ID_mvt", how="inner")
    if len(m) != len(s):
        raise ValueError(f"{fold}: the head/`t` join covers {len(m)} of {len(s)} stack-5 rows")
    if not np.array_equal(m["MVT_ID_mvt"].to_numpy(), s["MVT_ID_mvt"].to_numpy()):
        raise ValueError(f"{fold}: the join reordered stack-5's rows")
    if m["t"].isna().any():
        raise ValueError(f"{fold}: {int(m['t'].isna().sum())} rows have no take-off time")
    # `y_pred_anchor` is blend-5, which `exp-027-step` also saved as its `y_pred`;
    # agreement is the evidence that the two OOF families describe the same rows.
    off = float(np.abs(h.set_index("MVT_ID_mvt")["y_pred"]
                       .reindex(m["MVT_ID_mvt"]).to_numpy()
                       - m["y_pred_anchor"].to_numpy()).max())
    if off > 1e-9:
        raise ValueError(f"{fold}: the head frame and stack-5's anchor differ by {off:g}")
    m["base"] = m["y_pred"].to_numpy("float64")          # stack-5 keep-noac
    m["anchor"] = m["y_pred_anchor"].to_numpy("float64")  # blend-5, for reference only
    m["fold"] = fold
    return m[CANON + ["anchor", "fold"]].reset_index(drop=True)


def load(gate: bool = True) -> dict:
    """Every fold's panel, with exp-026's published rotation as the gate."""
    data = {f: load_fold(f) for f in FOLDS}
    if gate:
        pub = json.loads((ROOT / STACK5_JSON).read_text())
        want = pub["rotated_clean"]["stack-5 keep-noac"]
        for f in FOLDS:
            got = D.rmse_clean(data[f]["y_true"], data[f]["base"],
                               data[f]["MVT_ID_mvt"], f)
            if abs(got - want[f]) > 1e-3:
                raise SystemExit(f"base does not reproduce on {f}: {got:.4f} against "
                                 f"exp-026's published {want[f]:.4f}")
        got_b = board({f: D.rmse_clean(data[f]["y_true"], data[f]["base"],
                                       data[f]["MVT_ID_mvt"], f) for f in FOLDS})
        if abs(got_b - pub["clean_board"]["stack-5 keep-noac"]) > 1e-3:
            raise SystemExit(f"base board {got_b:.4f} against published "
                             f"{pub['clean_board']['stack-5 keep-noac']:.4f}")
    return data


def board(per_fold: dict) -> float:
    """The two ranking months in quadrature, weighted by their row counts."""
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


def clusters(p: pd.DataFrame) -> np.ndarray:
    """Calendar-day x airport clusters, exp-026 `stack5.clusters`."""
    day = np.floor(p["t"].to_numpy("float64") / 86400.0).astype("int64")
    return pd.factorize(p["airport"].astype(str) + "|" + pd.Series(day).astype(str))[0]


def boot_delta(y, p_new, p_ref, cl, draws=2000, seed=7):
    """(delta, 2.5 %, 97.5 %) of RMSE(new) - RMSE(ref), clustered bootstrap.

    Byte-for-byte `experiments/exp-026-rebuild/code/stack5.boot_delta`, so the
    intervals in this file and in exp-026 §4 are the same instrument.
    """
    y = np.asarray(y, "float64")
    e_new = (np.asarray(p_new, "float64") - y) ** 2
    e_ref = (np.asarray(p_ref, "float64") - y) ** 2
    k = cl.max() + 1
    order = np.argsort(cl, kind="stable")
    starts = np.searchsorted(cl[order], np.arange(k))
    sn = np.add.reduceat(e_new[order], starts)
    sr = np.add.reduceat(e_ref[order], starts)
    cnt = (np.searchsorted(cl[order], np.arange(k), side="right") - starts).astype("float64")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype("float64")
    n = W @ cnt
    out = np.sqrt((W @ sn) / n) - np.sqrt((W @ sr) / n)
    return (float(np.sqrt(e_new.mean()) - np.sqrt(e_ref.mean())),
            float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))


def decomp(y, p, panel: pd.DataFrame) -> dict:
    """COMMON.md's required decomposition: bulk, tail, no-NM, noac."""
    e = (np.asarray(p, "float64") - np.asarray(y, "float64")) ** 2
    r = lambda m: float(np.sqrt(e[m].mean())) if m.any() else float("nan")
    bulk = np.asarray(y, "float64") <= 3600.0
    nn = panel["regime"].to_numpy() != "noac"
    return dict(rmse=float(np.sqrt(e.mean())), le3600=r(bulk), gt3600=r(~bulk),
                no_nm=r(panel["nm_missing"].to_numpy().astype(bool)),
                noac=r(panel["noac"].to_numpy().astype(bool)),
                no_noac=r(nn))


if __name__ == "__main__":
    data = load()
    for f in FOLDS:
        p = data[f]
        r = D.rmse_both(p["y_true"], p["base"], p["MVT_ID_mvt"], f)
        print(f"{f:9s} n={len(p):7d} clean {r['rmse_clean']:9.4f}  all {r['rmse']:9.4f}  "
              f"excluded {r['n_excluded']}  regimes "
              f"{dict(p['regime'].value_counts())}")
    print("clean board %.4f (published 258.0372)"
          % board({f: D.rmse_clean(data[f]['y_true'], data[f]['base'],
                                   data[f]['MVT_ID_mvt'], f) for f in FOLDS}))
