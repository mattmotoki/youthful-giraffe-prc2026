"""Why the window arm pays on blend-5 and not on stack-5.

Three questions, in the order the answer needs them:

1. **Is it the base or the code?**  The same arm, the same rotation, run on both
   bases — blend-5's residual (K1's base) and stack-5's.  If the blend-5 column
   reproduces `experiments/exp-027-step/` §8d, the difference is the base.
2. **Did the component shrink?**  The oracle that knows each airport x window
   cell's mean error, priced on both bases.  This is `docs/STRUCTURAL-STEP.md`
   §8c's ceiling, recomputed on the residual the arm actually faces.
3. **Did the sharing shrink?**  The correlation of the residual between two
   departures at the same airport minutes apart, on both bases — §8a's statistic.

    python diagnose.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm as A  # noqa: E402
import features as Fe  # noqa: E402
import panel as P  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.data as D  # noqa: E402

BASES = ("blend5", "stack5")
ARMS = ("window", "win_only", "row", "shuffle")
CLIP = 1200.0


# ---------------------------------------------------------------- 2. oracle --
def oracle(data: dict, win: float, clip: float | None = None) -> dict:
    """Board gain from knowing every airport x window cell's mean error.

    `raw` is `sum n*m^2`; `unbiased` subtracts one within-cell variance per cell,
    which is what `E[n * mean^2] = n*mu^2 + sigma^2` costs.  Neither is clipped
    at zero.  Cells of one row contribute their whole squared error to `raw` and
    nothing to `unbiased`, which is the point of reporting both.

    Corrected 2026-09-07 after Sol round 1 (`code/review1.py`): the excluded
    january rows were kept, and a cell of one row reached `unbiased` with its
    whole squared error because `var(ddof=1)` is NaN there and NaN became zero.
    Both are fixed below, so this function now does what the paragraph above
    always said.  The published numbers it replaces are in `RESULTS.md` SS 3.
    """
    out = {}
    for kind in ("raw", "unbiased"):
        per, before = {}, {}
        for f in P.FOLDS:
            d = data[f]
            keep = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
            d = d.loc[keep].reset_index(drop=True)
            e = d["base"].to_numpy("float64") - d["y_true"].to_numpy("float64")
            if clip:
                e = np.clip(e, -clip, clip)
            b = np.floor(d["t"].to_numpy("float64") / win).astype("int64")
            key = pd.Series(d["airport"].astype(str).to_numpy()).str.cat(pd.Series(b).astype(str),
                                                                        sep="|")
            g = pd.DataFrame({"e": e, "k": key.to_numpy()}).groupby("k")["e"]
            n, m, v = g.size().to_numpy("float64"), g.mean().to_numpy(), g.var(ddof=1).to_numpy()
            if kind == "raw":
                red = float(np.nansum(n * m ** 2))
            else:
                term = n * m ** 2 - np.nan_to_num(v)
                term[n == 1] = 0.0          # a cell of one row has no mean to estimate
                red = float(np.nansum(term))
            sse = float((e ** 2).sum())
            per[f] = float(np.sqrt(max(sse - red, 0.0) / len(e)))
            before[f] = float(np.sqrt(np.mean(e ** 2)))
        # the same rows on both sides of the difference: the two excluded january
        # rows carry enough squared error to move that fold by 119 s on their own
        base = P.board({f: before[f] for f in P.SHARES})
        out[kind] = dict(board_after=P.board(per), gain=base - P.board(per),
                         per_fold={f: per[f] for f in P.FOLDS})
    return out


# ------------------------------------------------------------ 3. the sharing --
GAPS = ((0, 120), (120, 300), (300, 900), (900, 1800), (1800, 3600))


def sharing(data: dict, fold: str, clip: float = CLIP, max_pairs: int = 400_000,
            seed: int = 5) -> pd.DataFrame:
    """Correlation of the residual between same-airport departures, by take-off gap.

    `nm_bulk` only and winsorised, as `docs/STRUCTURAL-STEP.md` §8a: 0.3 % of rows
    must not set the answer.
    """
    d = data[fold]
    m = (d["regime"].to_numpy() == "nm_bulk")
    e = np.clip(d["base"].to_numpy("float64")[m] - d["y_true"].to_numpy("float64")[m],
                -clip, clip)
    t = d["t"].to_numpy("float64")[m]
    apt = pd.factorize(d["airport"].astype(str).to_numpy()[m])[0]
    order = np.lexsort((t, apt))
    e, t, apt = e[order], t[order], apt[order]
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in GAPS:
        a, b = [], []
        # pairs within the gap band: walk forward from each row until the gap
        # exceeds `hi`, keeping at most a few partners per row so one busy
        # airport-hour cannot dominate
        j = np.arange(len(t))
        for step in range(1, 40):
            k = j + step
            ok = (k < len(t)) & (apt[np.minimum(k, len(t) - 1)] == apt)
            gap = np.where(ok, t[np.minimum(k, len(t) - 1)] - t, np.inf)
            sel = ok & (gap >= lo) & (gap < hi)
            if not sel.any():
                continue
            a.append(e[j[sel]]); b.append(e[k[sel]])
        if not a:
            rows.append(dict(fold=fold, lo=lo, hi=hi, n=0, corr=np.nan)); continue
        a, b = np.concatenate(a), np.concatenate(b)
        if len(a) > max_pairs:
            i = rng.choice(len(a), max_pairs, replace=False)
            a, b = a[i], b[i]
        rows.append(dict(fold=fold, lo=lo, hi=hi, n=len(a),
                         corr=float(np.corrcoef(a, b)[0, 1]),
                         shared=float(np.corrcoef(a, b)[0, 1] * a.std() * b.std())))
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    raw = P.load()
    data = {b: A.rebase(raw, b) for b in BASES}
    X = A.design(raw, A.CANDIDATE["win"])     # features do not depend on the base
    rows = []
    for b in BASES:
        for arm in ARMS:
            T, _ = A.run(data[b], X, arm=arm, **A.CANDIDATE)
            r = A.board_row(T); r["base"] = b
            rows.append(r)
            print(f"[{b:7s} {arm:9s}] july_loo {r['july_loo']:+7.3f}  "
                  f"december {r['december']:+7.3f}  january {r['january']:+7.3f}  |  "
                  f"board {r['gain_board']:+7.3f}  no-noac {r['gain_nonoac']:+7.3f}  "
                  f"3f {r['three_folds']}", flush=True)
    B = pd.DataFrame(rows)[["base", "arm", "july_loo", "december", "january",
                            "gain_board", "gain_nonoac", "clean_board", "three_folds"]]
    B.round(4).to_csv(P.EXP / "analysis" / "diagnose_arms.csv", index=False)

    print("\n=== the airport x window oracle on each base ===", flush=True)
    orc = {}
    for b in BASES:
        for w in (300.0, 600.0, 1800.0):
            o = oracle(data[b], w)
            orc[f"{b}|{w:.0f}"] = o
            print(f"  {b:7s} {w:6.0f}s   raw {o['raw']['gain']:7.3f} s   "
                  f"unbiased {o['unbiased']['gain']:7.3f} s", flush=True)
    (P.EXP / "analysis" / "diagnose_oracle.json").write_text(json.dumps(orc, indent=2))

    print("\n=== residual sharing between same-airport departures (nm_bulk) ===", flush=True)
    sh = []
    for b in BASES:
        for f in ("july_loo", "january"):
            s = sharing(data[b], f); s["base"] = b
            sh.append(s)
    S = pd.concat(sh, ignore_index=True)
    S.round(4).to_csv(P.EXP / "analysis" / "diagnose_sharing.csv", index=False)
    piv = S.pivot_table(index=["lo", "hi"], columns=["base", "fold"], values="corr")
    print(piv.round(4).to_string())
    print(f"\n[diagnose] {time.time() - t0:.0f}s wall", flush=True)


if __name__ == "__main__":
    main()
