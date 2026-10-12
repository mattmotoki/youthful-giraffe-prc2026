"""Where the remaining squared error of the head sits, and what draining a cell
would move on the clean board.

Currency. The board is a mixture of the two ranking months at their row shares
(`exp-001 board.py`): 0.5571 July 2026, 0.4429 January 2026. RMSE combines in
quadrature, so

    board_clean^2 = 0.5571 * MSE_clean(july_loo) + 0.4429 * MSE_clean(january)

and a cell's *board mass* is the same weighted average of its share of each
fold's clean squared-error sum. Draining a cell to zero moves the board from
`board` to `sqrt(board^2 - mass)`; that difference is the column
`d_perfect`, and it is the only honest way to say "this cell is worth N seconds".
The december fold is not in the board mixture (it is a robustness read), so it
is reported in its own column and never mixed in.

Two ceilings per cell, because they answer different questions:
  * `d_perfect`   -- an oracle that knows y on the cell. Never reachable; it is
    the bound a candidate insight must fit inside.
  * `d_debias`    -- an oracle that knows only the cell's MEAN error and shifts
    the whole cell by it. This is what a per-cell calibration constant buys, and
    it is reachable from the ranking file whenever the cell is input-readable.
    The gap between the two is the part that needs a real signal, not a level.

`--cells` chooses the partition. Label bands are a diagnostic (the label is not
an input); `takeoff` bands are input-readable and therefore actionable.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
OOF = ROOT / "experiments/exp-027-step/oof"

YBANDS = [(-np.inf, 600), (600, 1200), (1200, 1800), (1800, 3600),
          (3600, 14400), (14400, np.inf)]
YLAB = ["y<600", "600-1200", "1200-1800", "1800-3600", "3600-14400", "y>14400"]
TBANDS = [(-np.inf, 0), (0, 600), (600, 1200), (1200, 1800), (1800, 3600),
          (3600, 14400), (14400, np.inf)]
TLAB = ["t<0", "t 0-600", "t 600-1200", "t 1200-1800", "t 1800-3600",
        "t 3600-14400", "t>14400"]


def band(v, edges, labels, nullname):
    out = pd.Series(nullname, index=range(len(v)), dtype=object)
    v = np.asarray(v, dtype="float64")
    for (lo, hi), lab in zip(edges, labels):
        out[(v >= lo) & (v < hi)] = lab
    out[~np.isfinite(v)] = nullname
    return out.to_numpy()


#: Round-1 review, blocking finding 1: the desk ships `stack5`, so the budget is
#: stated on both heads. The covariates (`regime`, `airport`, the two deltas) are
#: properties of the row and come from the blend-5 frame either way.
HEADS = {
    "blend5": "experiments/exp-027-step/oof/blend5-{fold}.parquet",
    "stack5": "experiments/exp-026-rebuild/oof/stack5-{fold}.parquet",
}


def load(fold: str, drop_noac: bool = False, head: str = "blend5") -> pd.DataFrame:
    d = pd.read_parquet(OOF / f"blend5-{fold}.parquet")
    if head != "blend5":
        h = pd.read_parquet(ROOT / HEADS[head].format(fold=fold))[["MVT_ID_mvt", "y_pred"]]
        d = d.drop(columns=["y_pred"]).merge(h, on="MVT_ID_mvt", how="inner")
    t = pd.read_parquet(ROOT / f"experiments/exp-016-stack/oof/exp-016-oof-{fold}.parquet",
                        columns=["MVT_ID_mvt", "t"])
    d = d.merge(t, on="MVT_ID_mvt", how="left")
    # clean population: the fold's unreached-rollover rows never enter a budget,
    # because the board estimate they feed is `rmse_clean`.
    d = d[~d["MVT_ID_mvt"].isin(D.excluded_ids(fold))].reset_index(drop=True)
    d["err"] = d["y_pred"] - d["y_true"]
    d["se"] = d["err"] ** 2
    d["yband"] = band(d["y_true"], YBANDS, YLAB, "y?")
    d["tband"] = band(d["d_takeoff_aobt"], TBANDS, TLAB, "t missing")
    ts = pd.to_datetime(d["t"], unit="s", utc=True)
    d["hour"] = ts.dt.hour
    d["date"] = ts.dt.strftime("%Y-%m-%d")
    d["dow"] = ts.dt.dayofweek
    if drop_noac:
        # The desk's rule (2026-09-07): 402 fold rows dominate the between-candidate
        # variance, so every ceiling is stated twice -- on the whole population and
        # on the population with them removed. They are their own problem.
        d = d[d["regime"] != "noac"].reset_index(drop=True)
    return d


def budget(frames: dict[str, pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """One row per cell, with the board mass and both ceilings."""
    mse = {f: frames[f]["se"].mean() for f in FOLDS}
    b2 = sum(SHARES[f] * mse[f] for f in SHARES)
    board = float(np.sqrt(b2))
    rows = {}
    for f in FOLDS:
        d, n = frames[f], len(frames[f])
        g = d.groupby(keys, dropna=False)
        agg = g.agg(n=("se", "size"), S=("se", "sum"), bias=("err", "mean"))
        # a per-cell constant removes exactly n*bias^2 of the cell's mass
        agg["S_debias"] = agg["S"] - agg["n"] * agg["bias"] ** 2
        for k, r in agg.iterrows():
            k = k if isinstance(k, tuple) else (k,)
            e = rows.setdefault(k, {})
            e[f"n_{f}"] = r["n"]
            e[f"rmse_{f}"] = np.sqrt(r["S"] / r["n"])
            e[f"bias_{f}"] = r["bias"]
            e[f"mass_{f}"] = r["S"] / n          # this cell's share of the fold MSE
            e[f"massdb_{f}"] = r["S_debias"] / n
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=keys)
    out = out.fillna(0.0)
    out["mass"] = sum(SHARES[f] * out[f"mass_{f}"] for f in SHARES)
    out["mass_db"] = sum(SHARES[f] * out[f"massdb_{f}"] for f in SHARES)
    drop = lambda m: board - np.sqrt(np.maximum(b2 - m, 0.0))
    out["d_perfect"] = drop(out["mass"])
    out["d_debias"] = drop(out["mass"] - out["mass_db"])
    # the same two ceilings restricted to one ranking month's half of the cell:
    # what an insight that only works in January (or only in July) can move.
    for f in SHARES:
        tag = "jul" if f == "july_loo" else "jan"
        out[f"d_perfect_{tag}"] = drop(SHARES[f] * out[f"mass_{f}"])
        out[f"d_debias_{tag}"] = drop(SHARES[f] * (out[f"mass_{f}"] - out[f"massdb_{f}"]))
    out["n_board"] = out["n_july_loo"] + out["n_january"]
    out["share_pct"] = 100 * out["mass"] / b2
    return out.sort_values("mass", ascending=False), board, mse


def show(t: pd.DataFrame, board: float, n=25, cols=None):
    cols = cols or ["n_board", "n_july_loo", "n_january", "rmse_july_loo", "rmse_january",
                    "bias_july_loo", "bias_january", "share_pct", "d_perfect", "d_debias"]
    cols = [c for c in cols if c in t.columns]
    v = t[cols].head(n).copy()
    fmt = {"n_board": "{:,.0f}", "n_july_loo": "{:,.0f}", "n_january": "{:,.0f}",
           "n_december": "{:,.0f}",
           "rmse_july_loo": "{:,.0f}", "rmse_january": "{:,.0f}", "rmse_december": "{:,.0f}",
           "bias_july_loo": "{:+.1f}", "bias_january": "{:+.1f}", "bias_december": "{:+.1f}",
           "share_pct": "{:.2f}", "d_perfect": "{:.3f}", "d_debias": "{:.3f}",
           "d_perfect_jul": "{:.3f}", "d_debias_jul": "{:.3f}",
           "d_perfect_jan": "{:.3f}", "d_debias_jan": "{:.3f}"}
    for c in v.columns:
        v[c] = v[c].map(lambda x, c=c: fmt.get(c, "{:.3f}").format(x))
    print(v.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="regime")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--cols", default=None)
    ap.add_argument("--drop-noac", action="store_true")
    ap.add_argument("--head", default="blend5", choices=sorted(HEADS))
    a = ap.parse_args()
    frames = {f: load(f, a.drop_noac, a.head) for f in FOLDS}
    keys = a.cells.split(",")
    t, board, mse = budget(frames, keys)
    print(f"head = {a.head}, clean population"
          + ("  [noac rows REMOVED]" if a.drop_noac else ""))
    print(f"board_clean = {board:.4f}   "
          + "   ".join(f"{f} {np.sqrt(mse[f]):.3f}" for f in FOLDS))
    for step in (2, 5, 10, 14):
        print(f"  a {step:>2d} s clean step needs {board**2 - (board-step)**2:>8,.0f} s^2 "
              f"of weighted MSE ({100*(board**2-(board-step)**2)/board**2:.1f} % of the budget)")
    print(f"\ncells = {keys}   ({len(t)} cells)")
    show(t, board, a.n, a.cols.split(",") if a.cols else None)
    if a.csv:
        t.to_csv(ROOT / a.csv)
        print(f"\nwritten {a.csv}")


if __name__ == "__main__":
    main()
