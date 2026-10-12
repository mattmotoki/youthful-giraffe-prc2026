"""How much of the head's error is shared inside an airport-window, and what an
oracle on that component would move.

Model: `e_i = c_{g(i)} + u_i`, cell `g` = airport x time bucket. For a cell with
`n` rows, sample mean `m` and sample variance `s²` (ddof 1),
`E[n·m²] = n·mu² + sigma²`, so `n·m² − s²` is unbiased for `n·mu²` and the sum over
cells is the shared mass. `d_perfect = board − sqrt(board² − mass)`.

**The singleton correction, and why it is not optional.** `var(ddof=1)` is
undefined for a cell of one row, and treating it as zero makes that row's whole
squared error count as "shared" -- the estimator then reports the noise as
structure. At a 5-minute bucket 8.4 % of rows sit in singleton cells and the
uncorrected figure reads **27.6 s** against **16.1 s** corrected; at 10 minutes,
2.1 % of rows and **18.8 s** against **15.1 s**. Cells below `minn` here
contribute **nothing** to the shared mass while their rows stay in the
denominator. That direction understates a sparse population, but the figure is a
SENSITIVITY and not an identified estimate: the singleton cells' true contribution
is unidentified rather than zero, and `board - sqrt(board^2 - mass)` is convex, so
estimation noise in `mass` biases the SECOND number upward. Read every ceiling
here as an upper bound that is itself uncertain. `minn=4` is printed beside
`minn=2` as the sensitivity on the first point.

Everything is a CEILING. No feature recovers a cell mean exactly, and the
cross-fitted figures in `RESULTS.md` §7 -- which estimate the cell mean on half a
fold and score it on the other half -- are what an estimator of it is actually
worth (+4.9 s at one hour against this file's 9.0 s oracle there).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}


sys.path.insert(0, str(HERE))
from window_probe import load  # noqa: E402  (one loader, one head switch)


def shared(fr, win, sub, minn=2, wins=None, denom=None):
    """`denom` states the population the ceiling is quoted ON. By default it is
    the same population the cells are built from, so `sub=nm_bulk` returns
    nm_bulk's ceiling on nm_bulk's own board. Round-1 review, blocking finding 4:
    a candidate that drains one regime must also be priced against the board the
    desk actually reads, so `denom=all rows` divides the same shared mass by the
    full clean population and converts it with the full board."""
    b2 = tot = 0.0
    drop = []
    for f in SHARES:
        d = fr[f][sub(fr[f])]
        nd = int((denom(fr[f]) if denom is not None else sub(fr[f])).sum())
        ed = (fr[f]["err"].to_numpy()[denom(fr[f])] if denom is not None else None)
        e = d["err"].to_numpy()
        if wins:
            e = np.clip(e, -wins, wins)
        n = len(d)  # noqa: F841 (kept for the same-population default below)
        c = (d["airport"] + "|" + (d["t"] // win).astype("int64").astype(str)).to_numpy()
        g = pd.DataFrame({"c": c, "e": e}).groupby("c")["e"]
        m, k, v = g.mean(), g.size(), g.var(ddof=1)
        ok = (k >= minn) & v.notna()
        if denom is not None:
            b2 += SHARES[f] * ((np.clip(ed, -wins, wins) if wins else ed) ** 2).mean()
        else:
            b2 += SHARES[f] * (e ** 2).mean()
        tot += SHARES[f] * ((k[ok] * m[ok] ** 2 - v[ok]).sum()) / (nd if denom is not None else n)
        drop.append(1 - k[ok].sum() / k.sum())
    board = b2 ** 0.5
    return {"board": board, "shared_pct": 100 * tot / b2,
            "ceiling": board - np.sqrt(max(b2 - tot, 0.0)),
            "rows_dropped_pct": 100 * float(np.mean(drop))}


def main(head: str = "blend5"):
    fr = {f: load(f, head) for f in SHARES}
    print(f"head = {head}")
    allr = lambda d: np.ones(len(d), bool)          # noqa: E731
    nonoac = lambda d: (d["regime"] != "noac").to_numpy()  # noqa: E731
    print("oracle on the airport x window cell mean, singleton-corrected (minn=2)")
    print(f"  {'cell':26s} {'board':>8s} {'shared':>8s} {'ceiling':>9s} {'dropped':>9s}")
    for sub, tag in ((allr, "all rows"), (nonoac, "no noac")):
        for w, l in ((300, "5 min"), (600, "10 min"), (900, "15 min"),
                     (1800, "30 min"), (3600, "1 h")):
            r = shared(fr, w, sub, 2)
            print(f"  {l + ', ' + tag:26s} {r['board']:8.3f} {r['shared_pct']:7.2f}% "
                  f"{r['ceiling']:9.2f} {r['rows_dropped_pct']:8.1f}%")
    print("\n  sensitivity, minn=4 (no noac):")
    for w, l in ((600, "10 min"), (1800, "30 min"), (3600, "1 h")):
        r = shared(fr, w, nonoac, 4)
        print(f"    {l:8s} ceiling {r['ceiling']:6.2f} s   rows dropped {r['rows_dropped_pct']:.1f} %")
    print("\n  by regime, 10 min, minn=2 (a regime with most rows in singleton cells")
    print("  is under-measured, not over-measured -- the dropped column says how far):")
    for reg in ("nm_bulk", "nm_late", "nonm"):
        r = shared(fr, 600, lambda d, reg=reg: (d["regime"] == reg).to_numpy(), 2)
        print(f"    {reg:8s} board {r['board']:8.3f}  shared {r['shared_pct']:5.2f} %  "
              f"ceiling {r['ceiling']:7.2f} s  rows dropped {r['rows_dropped_pct']:.1f} %")
    print("\n  winsorised to +/-1200 s (no noac), the bulk-only reading:")
    for w, l in ((600, "10 min"), (1800, "30 min")):
        r = shared(fr, w, nonoac, 2, 1200)
        print(f"    {l:8s} board {r['board']:8.3f}  ceiling {r['ceiling']:6.2f} s")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
