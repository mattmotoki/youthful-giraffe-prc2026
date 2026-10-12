"""Round-1 review repairs: the three findings that change a number.

Sol (`review/sol-round1.log`, 2026-09-07) found no defect that moves a staged
candidate's score, and the leakage contract passes for the shipping block.  It
did find three measurements computed on the wrong population or with an
estimator that contradicted its own docstring.  This file recomputes all three
off the saved rotation vectors -- nothing is refitted -- and writes
`analysis/review1.json` next to the numbers it replaces.

1. **The oracle** (Standards 1).  `diagnose.oracle` kept the two excluded
   january rows and, for a cell of one row, added `n*m^2 - 0` to the *unbiased*
   reduction, so a singleton contributed its whole squared error as signal --
   the opposite of what its docstring promised.  Here: clean rows, and a cell of
   one row contributes nothing to `unbiased`.

2. **The intervals** (Standards 2).  `stack5w.py` and `restrict.py` quote a
   clean point estimate beside a bootstrap over all rows.  Here both
   populations the assignment asks for: clean, and clean with the `noac` rows
   dropped (the served block never touches them).

3. **The no-NM slice** (Standards 4).  `COMMON.md` defines it as *all* `_flt`
   columns null; `features.py` used `FLIGHT_ID_mvt` alone, which is an `_mvt`
   column and misses a few rows per fold.  Here the mask is read from the raw
   month files, and both definitions are reported so the difference is visible.

    python review1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel as P  # noqa: E402
import where as W  # noqa: E402
import arm as A  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

FOLD_FILE = {"july_loo": "training_2025-07-01_2025-08-01.parquet",
             "december": "training_2025-12-01_2026-01-01.parquet",
             "january": "training_2025-01-01_2025-02-01.parquet"}
SETS = {"all-but-noac": ("nm_bulk", "nm_late", "nonm"), "bulk": ("nm_bulk",)}


# --------------------------------------------------------------- 1. oracle ---
def oracle(data: dict, win: float) -> dict:
    """`unbiased`, done as the docstring always said: clean rows, singletons out.

    A cell of `n` rows with mean error `m` and sample variance `v` can remove at
    most `n*m^2` of squared error, and `E[n*mean^2] = n*mu^2 + sigma^2`, so one
    within-cell variance per cell is what fitting the cell mean costs.  A cell of
    one row has no `mu` to estimate: it contributes nothing.
    """
    per, base = {}, {}
    for f in P.FOLDS:
        d = data[f]
        keep = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        e = (d["base"].to_numpy("float64") - d["y_true"].to_numpy("float64"))[keep]
        b = np.floor(d["t"].to_numpy("float64")[keep] / win).astype("int64")
        key = pd.Series(d["airport"].astype(str).to_numpy()[keep]).str.cat(
            pd.Series(b).astype(str), sep="|")
        g = pd.DataFrame({"e": e, "k": key.to_numpy()}).groupby("k")["e"]
        n = g.size().to_numpy("float64")
        term = n * g.mean().to_numpy() ** 2 - np.nan_to_num(g.var(ddof=1).to_numpy())
        term[n == 1] = 0.0                      # <- the fix
        per[f] = float(np.sqrt(max(float((e ** 2).sum()) - float(term.sum()), 0.0) / len(e)))
        base[f] = float(np.sqrt(np.mean(e ** 2)))
        per[f + "_singletons"] = int((n == 1).sum())
    board_after = P.board({f: per[f] for f in P.SHARES})
    return dict(win=win, board_before=P.board({f: base[f] for f in P.SHARES}),
                board_after=board_after,
                gain=P.board({f: base[f] for f in P.SHARES}) - board_after,
                per_fold={f: round(per[f], 4) for f in P.FOLDS},
                singletons={f: per[f + "_singletons"] for f in P.FOLDS})


# ------------------------------------------------------------ 2. intervals ---
def intervals(tag: str) -> list:
    data = {}
    W.TAG = tag
    for f in P.FOLDS:
        data[f] = W.load(f)
    rows = []
    for name, regs in SETS.items():
        for f in P.FOLDS:
            d = data[f]
            a = d["y_pred_stack5"].to_numpy("float64")
            b = d["y_pred_depx"].to_numpy("float64")
            p = np.where(np.isin(d["regime"].to_numpy(), regs), b, a)
            y = d["y_true"].to_numpy("float64")
            cl = P.clusters(d)
            ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
            nn = ok & (d["regime"].to_numpy() != "noac")
            r = dict(tag=tag, variant=name, fold=f)
            for pop, m in (("all", np.ones(len(d), bool)), ("clean", ok), ("clean_no_noac", nn)):
                dl, lo, hi = P.boot_delta(y[m], p[m], a[m], pd.factorize(cl[m])[0])
                r[f"{pop}_delta"] = round(dl, 4)
                r[f"{pop}_ci"] = f"[{lo:+.3f}, {hi:+.3f}]"
                r[f"{pop}_n"] = int(m.sum())
            rows.append(r)
    return rows


# --------------------------------------------------------------- 3. no-NM ---
def no_nm_mask(fold: str, ids: np.ndarray) -> np.ndarray:
    """COMMON.md's definition: every `_flt` column null, read from the raw month."""
    raw = pd.read_parquet(P.ROOT / "data" / "raw" / FOLD_FILE[fold])
    flt = [c for c in raw.columns if c.endswith("_flt")]
    m = pd.Series(raw[flt].isna().all(axis=1).to_numpy(), index=raw["MVT_ID_mvt"].to_numpy())
    m = m[~m.index.duplicated()]
    out = m.reindex(pd.Index(ids)).to_numpy()
    if pd.isna(out).any():
        raise SystemExit(f"{fold}: {int(pd.isna(out).sum())} ids are absent from the raw month")
    return out.astype(bool)


def no_nm(tag: str) -> list:
    W.TAG = tag
    rows = []
    for f in P.FOLDS:
        d = W.load(f)
        ids = d["MVT_ID_mvt"].to_numpy()
        strict = no_nm_mask(f, ids)
        loose = d["nm_missing"].to_numpy().astype(bool)
        y = d["y_true"].to_numpy("float64")
        for var, p in (("stack-5", d["y_pred_stack5"].to_numpy("float64")),
                       ("stack-5+depx", d["y_pred_depx"].to_numpy("float64"))):
            e = (p - y) ** 2
            rows.append(dict(tag=tag, fold=f, variant=var,
                             common_no_nm=round(float(np.sqrt(e[strict].mean())), 4),
                             flight_id_only=round(float(np.sqrt(e[loose].mean())), 4),
                             n_common=int(strict.sum()), n_flight_id=int(loose.sum()),
                             n_extra=int((strict & ~loose).sum())))
    return rows


def main() -> None:
    data = P.load(gate=True)
    out = {"oracle": {}, "intervals": [], "no_nm": []}
    for base in ("blend5", "stack5"):
        d = A.rebase(data, base)
        for win in (300.0, 600.0, 1800.0):
            out["oracle"][f"{base}-{int(win)}"] = oracle(d, win)
    for tag in ("w300", "w600"):
        out["intervals"] += intervals(tag)
        out["no_nm"] += no_nm(tag)
    (P.EXP / "analysis" / "review1.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["oracle"], indent=1))
    print(pd.DataFrame(out["intervals"]).to_string(index=False))
    print(pd.DataFrame(out["no_nm"]).to_string(index=False))


if __name__ == "__main__":
    main()
