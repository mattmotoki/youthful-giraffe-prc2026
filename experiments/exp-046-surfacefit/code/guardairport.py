"""Order 7: is the tail guard a rule, or is it the LIRF july tail wearing a global coat?

Desk pick, 2026-09-09:

    then (7) the guard on held-out airports (arithmetic + one re-price) -- it decides
    +0.1369 and unblocks (4) LFPG.

The guard is a MAYBE arm worth +0.1369 s whose interval touches zero on two of three folds
(section 13.3), and section 16.1 forbids serving any guard variant on the current fold
evidence.  The reason to doubt it is not the interval: it is that `analysis/lirf-july.json`
FOUND the tail at LIRF july and the guard was then priced on folds that contain LIRF july.
An arm discovered and priced on the same rows carries the maximum of that search, exactly as
section 16.2's arms did -- but on the AIRPORT axis, which no fold cross-fit can see.

**The instrument.**  The guard's action is "hold the row at the price base", so the guarded
vector is the unguarded one with the held rows zeroed.  That means it can be re-derived for
ANY cut without refitting anything:

    g(c) = u, except g(c)[gap > c] = 0

The file checks that against the shipped vector first -- re-deriving with the shipped cut
must reproduce `guardarm.py`'s bytes exactly -- and only then re-cuts.  For each airport A
the quantile is taken over the fitting folds with **A's own rows removed**, and the resulting
guard is priced on A's rows alone.  A guard that is a rule pays on airports whose rows never
informed its threshold; a guard that is the LIRF tail pays at LIRF and nowhere else.

Nothing is refitted, no vector is rebuilt, and nothing is served.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/guardairport.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import guard as G                                                     # noqa: E402

UNGUARDED = "leaves_255x300_lr050"
GUARDED = f"c_lgb_shipped@{UNGUARDED}+guard995"
HOLD = ("LFPG",)
OUT = EXP / "analysis/guard-airport.json"


def held(pan: dict, v: np.ndarray, f: str) -> np.ndarray:
    out = v.astype("float64").copy()
    out[np.isin(pan[f].airport.to_numpy().astype(str), list(HOLD))] = 0.0
    return out


def cut_excluding(pan: dict, folds, airport: str | None) -> float:
    """`guard.cut`, over the fitting folds, with one airport's rows removed."""
    vs = []
    for f in folds:
        p = pan[f]
        keep = p.usable.to_numpy(bool)
        if airport is not None:
            keep = keep & (p.airport.to_numpy().astype(str) != airport)
        vs.append(G.gap(p.pbase.to_numpy("float64"), p.obs_taxi)[keep])
    return float(np.nanpercentile(np.concatenate(vs), G.Q))


def main() -> None:
    pan = A.panel_with_block(BA.load())
    un = dict(np.load(S.npz_path(f"c_lgb_shipped@{UNGUARDED}")))
    gu = dict(np.load(S.npz_path(GUARDED)))

    # ---- the self-check: re-deriving with the shipped cut must give the shipped bytes ---
    check = {}
    for f in A.FOLDS:
        c = cut_excluding(pan, [g for g in A.FOLDS if g != f], None)
        u, g = held(pan, un[f], f), held(pan, gu[f], f)
        re = u.copy()
        re[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, c)] = 0.0
        check[f] = {"cut_s": c, "rows_differing_from_the_shipped_vector":
                    int((np.abs(re - g) > 1e-9).sum())}
    bad = {f: v for f, v in check.items() if v["rows_differing_from_the_shipped_vector"]}
    if bad:
        raise SystemExit(f"re-derivation does not reproduce guardarm.py's vector: {bad}")
    print("self-check: the guard re-derives from the unguarded vector exactly, all folds",
          flush=True)

    out = {"what": ("the tail guard priced on airports whose rows never informed its "
                    "threshold, so the airport-level selection premium becomes visible"),
           "ordered_by": "the desk, 2026-09-09, pick (7)",
           "no_refit": "the guarded vector is re-derived from the unguarded one by its cut",
           "self_check": check, "matcher_sha256": A.MATCHER_SHA,
           "found_at": "LIRF july (analysis/lirf-july.json)", "by_airport": {}}

    airports = sorted({str(a) for f in A.FOLDS for a in pan[f].airport.unique()}
                      - set(HOLD))
    for a in airports:
        inc, base = {}, {}
        rows = 0
        for f in A.FOLDS:
            sel = [g for g in A.FOLDS if g != f]
            c = cut_excluding(pan, sel, a)              # A's rows never inform its own cut
            u = held(pan, un[f], f)
            g = u.copy()
            g[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, c)] = 0.0
            on = (pan[f].airport.to_numpy().astype(str) == a)
            d = (g - u)
            d[~on] = 0.0                                 # priced on A's rows alone
            inc[f] = d
            rows += int((d != 0).sum())
            q = pan[f].copy()
            q["pbase"] = q["pbase"].to_numpy("float64") + u
            q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
            base[f] = q
        rec = BA.price(base, inc, np.random.default_rng(A.SEED))
        ci = rec["board_all_row"]["interval"]["ci95"]
        out["by_airport"][a] = {
            "rows_held_across_the_three_folds": rows,
            "board_gain_s": rec["board_all_row"]["gain_s"], "ci95": ci,
            "clear_above_zero": bool(ci[0] > 0),
            "by_fold": {f: {"mse_gain_s2": rec["by_fold"][f]["all_row"]["mse_gain_s2"],
                            "ci95": rec["by_fold"][f]["all_row"]["ci95"]}
                        for f in A.FOLDS}}
        v = out["by_airport"][a]
        print(f"  {a}  held {rows:5d}  {v['board_gain_s']:+8.4f} "
              f"[{ci[0]:+7.4f},{ci[1]:+7.4f}]  clear {v['clear_above_zero']}", flush=True)

    # ---- and the pooled reading with the airport it was found at removed ---------------
    for drop in (None, "LIRF"):
        inc, base = {}, {}
        for f in A.FOLDS:
            sel = [g for g in A.FOLDS if g != f]
            c = cut_excluding(pan, sel, drop)
            u = held(pan, un[f], f)
            g = u.copy()
            g[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, c)] = 0.0
            d = g - u
            if drop:
                d[pan[f].airport.to_numpy().astype(str) == drop] = 0.0
            inc[f] = d
            q = pan[f].copy()
            q["pbase"] = q["pbase"].to_numpy("float64") + u
            q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
            base[f] = q
        rec = BA.price(base, inc, np.random.default_rng(A.SEED))
        ci = rec["board_all_row"]["interval"]["ci95"]
        key = "pooled_as_shipped" if drop is None else f"pooled_without_{drop}"
        out[key] = {"board_gain_s": rec["board_all_row"]["gain_s"], "ci95": ci,
                    "clear_above_zero": bool(ci[0] > 0)}
        print(f"{key:28s} {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{ci[0]:+7.4f},{ci[1]:+7.4f}]  clear {ci[0] > 0}", flush=True)

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written", OUT)


if __name__ == "__main__":
    main()
