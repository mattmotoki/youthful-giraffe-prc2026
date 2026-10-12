"""Step 1: where january's tail worsens by 7.2 s, and whether it is the anchor or the member.

The desk, 13:15 UTC: "a gate that fails produces a repair, not a stop."  §20d(5)
records that the `mod` member through exp-027's **refit-bulk** arm — the arm v11's
own head is built on — fails that arm's registered second kill clause on january:
`y > 3600` against v7, **2088.43 -> 2095.65, −7.22 s** against a 1 s allowance,
while july_loo (+11.44) and december (+29.69) pass.  This file says where.

Four arm heads, the 2x2 of (member x anchor) that §20d(3) prices on the board,
read here on the january TAIL only:

    K2 / hold        exp-027-k2-k2-hold-{fold}.parquet          (v9's own recipe)
    K2 / refit-bulk  exp-027-k2-refit-bulk-{fold}.parquet       (v10's, and so v11's)
    mod / hold       cand-k2c-mod-k2-hold-{fold}.parquet        (the served candidate)
    mod / refit-bulk cand-k2c-mod-refit-bulk-{fold}.parquet     (form 2', the FAIL)

All four are `y_pred`, i.e. after v7's EHAM mask, which is the vector exp-030's
`score.slices` reads, so the numbers here reproduce the clause exactly.  The
population is the clause's own: CLEAN rows (`prc.data.excluded_ids` dropped) with
`y > 3600`.

Squared error is additive and RMSE is not, so every attribution below is a sum of
`e_cand^2 - e_v7^2` over a group, reported three ways: the group's share of the
total change, and the tail RMSE the fold would have if ONLY that group moved.
Nothing here is fitted and no label is used to select a row -- the `y > 3600`
split IS the clause, and the label-free selector is step 2's, not this file's.

    PRC_THREADS=3 python tail_diag_v11.py
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
P38 = ROOT / "experiments/exp-038-packet"
sys.path.insert(0, str(P38 / "code"))
import align as A  # noqa: E402
import vectors as V  # noqa: E402

FOLDS = V.FOLDS
TAIL = 3600.0
#: exp-027's clause and its allowance, as `refit_bulk.py` reads it.
ALLOWANCE = 1.0
HEADS = {
    "K2 / hold": ("experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet",
                  {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582}),
    "K2 / refit-bulk": ("experiments/exp-027-step/oof/exp-027-k2-refit-bulk-{fold}.parquet",
                        {"july_loo": 284.1774, "december": 204.9052, "january": 218.2115}),
    "mod / hold": ("experiments/exp-035-k2climb/oof/cand-k2c-mod-k2-hold-{fold}.parquet",
                   {"july_loo": 283.9958, "december": 204.8691, "january": 217.9169}),
    "mod / refit-bulk": ("experiments/exp-035-k2climb/oof/cand-k2c-mod-refit-bulk-{fold}.parquet",
                         {"july_loo": 283.6627, "december": 204.2511, "january": 217.6985}),
}
#: exp-030 `analysis/arm-eham-out.json`, the tail v7 leaves -- the clause's base.
V7_TAIL = {"july_loo": 2406.961, "december": 1798.904, "january": 2088.426}
BANDS = [(3600, 5400), (5400, 7200), (7200, 10800), (10800, 21600), (21600, np.inf)]


def load(ref: dict, tpl: str, published: dict, what: str) -> dict:
    out = {}
    for f in FOLDS:
        d = pd.read_parquet(ROOT / tpl.format(fold=f))
        V._order(d, ref[f], what, f)
        out[f] = d["y_pred"].to_numpy("float64")
    got = V.clean_of(out, ref)
    off = {f: round(got[f] - published[f], 4) for f in FOLDS}
    if max(abs(x) for x in off.values()) > 1e-3:
        raise SystemExit(f"{what} does not reproduce its published rotation: {off}")
    return out


def tail_rows(ref: dict, f: str) -> np.ndarray:
    ok = ~ref[f]["MVT_ID_mvt"].isin(V.D.excluded_ids(f)).to_numpy()
    return ok & (ref[f]["y_true"].to_numpy("float64") > TAIL)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def attribute(y: np.ndarray, base: np.ndarray, cand: np.ndarray, key: np.ndarray,
              names=None) -> list:
    """Per-group `sum(e_cand^2 - e_base^2)` and the tail RMSE if only it moved."""
    eb, ec = (y - base) ** 2, (y - cand) ** 2
    n, sse_b = len(y), float(eb.sum())
    tot = float(ec.sum()) - sse_b
    r0 = np.sqrt(sse_b / n)
    out = []
    for g in (names if names is not None else sorted(set(key))):
        m = key == g
        d = float(ec[m].sum() - eb[m].sum())
        out.append({"group": str(g), "n": int(m.sum()),
                    "rmse_base": round(float(np.sqrt(eb[m].mean())), 2) if m.any() else None,
                    "rmse_cand": round(float(np.sqrt(ec[m].mean())), 2) if m.any() else None,
                    "dsse": d, "share_of_change": round(d / tot, 4) if tot else None,
                    "seconds_if_only_this_moved": round(float(np.sqrt((sse_b + d) / n) - r0), 4)})
    out.sort(key=lambda r: -abs(r["dsse"]))
    return out


def band_of(y: np.ndarray) -> np.ndarray:
    lab = np.empty(len(y), dtype=object)
    for lo, hi in BANDS:
        lab[(y > lo) & (y <= hi)] = f"{lo}-{'inf' if hi == np.inf else hi}"
    return lab


def main() -> None:
    ref, vec, prov, files = V.load(verbose=False)
    v7 = vec["v7"]
    heads = {k: load(ref, tpl, pub, k) for k, (tpl, pub) in HEADS.items()}

    rec = {"clause": "y > 3600 clean rows, RMSE against v7, allowance 1.0 s "
                     "(exp-027 SPEC's second kill clause, read by code/refit_bulk.py)",
           "v7_tail_published": V7_TAIL, "folds": {}}

    print("=== the clause on all four heads, all three folds ===")
    print(f"  {'head':17s} " + "".join(f"{f:>26s}" for f in FOLDS))
    tails = {}
    for k, h in heads.items():
        row = []
        for f in FOLDS:
            m = tail_rows(ref, f)
            y = ref[f]["y_true"].to_numpy("float64")[m]
            r = rmse(y, h[f][m])
            tails[(k, f)] = r
            row.append(f"{r:10.2f} ({V7_TAIL[f] - r:+7.2f})")
        print(f"  {k:17s} " + "".join(f"{x:>26s}" for x in row))
    v7t = {}
    for f in FOLDS:
        m = tail_rows(ref, f)
        y = ref[f]["y_true"].to_numpy("float64")[m]
        v7t[f] = rmse(y, v7[f][m])
        off = round(v7t[f] - V7_TAIL[f], 3)
        if abs(off) > 0.01:
            raise SystemExit(f"v7's own tail does not reproduce exp-030's number on {f}: {off}")
    print(f"  {'v7 (the base)':17s} " + "".join(f"{v7t[f]:>26.2f}" for f in FOLDS))
    rec["tails"] = {f: {k: round(tails[(k, f)], 4) for k in heads} | {"v7": round(v7t[f], 4)}
                    for f in FOLDS}
    rec["clause_vs_v7"] = {f: {k: round(v7t[f] - tails[(k, f)], 4) for k in heads}
                           for f in FOLDS}

    # ---- the 2x2 on the tail: is the january loss the anchor or the member? ----
    print("\n=== january's tail: member and anchor, held one at a time ===")
    cmp = [("member, inside the refit-bulk anchor", "K2 / refit-bulk", "mod / refit-bulk"),
           ("member, inside the hold anchor", "K2 / hold", "mod / hold"),
           ("anchor, with K2 in the slot", "K2 / hold", "K2 / refit-bulk"),
           ("anchor, with the mod member in the slot", "mod / hold", "mod / refit-bulk")]
    rec["tail_2x2"] = {}
    for f in FOLDS:
        m = tail_rows(ref, f)
        y = ref[f]["y_true"].to_numpy("float64")[m]
        rec["tail_2x2"][f] = {}
        for note, a, b in cmp:
            ra, rb = rmse(y, heads[a][f][m]), rmse(y, heads[b][f][m])
            rec["tail_2x2"][f][note] = {"from": a, "to": b, "rmse_from": round(ra, 4),
                                        "rmse_to": round(rb, 4), "gain": round(ra - rb, 4)}
            if f == "january":
                print(f"  {note:42s} {ra:9.2f} -> {rb:9.2f}   {ra - rb:+8.2f} s")

    # ---- where the january loss sits ----
    f = "january"
    m = tail_rows(ref, f)
    y = ref[f]["y_true"].to_numpy("float64")[m]
    base, cand = v7[f][m], heads["mod / refit-bulk"][f][m]
    apt = ref[f]["airport"].astype(str).to_numpy()[m]
    reg = ref[f]["regime"].astype(str).to_numpy()[m]
    keep = json.loads((ROOT / V.MASK_JSON).read_text())["keep"]
    served = np.isin(apt, keep)
    print(f"\n=== january tail, mod/refit-bulk against v7: {int(m.sum())} rows, "
          f"{v7t[f]:.2f} -> {tails[('mod / refit-bulk', f)]:.2f} "
          f"({v7t[f] - tails[('mod / refit-bulk', f)]:+.2f} s) ===")
    for what, key, names in (("airport", apt, None), ("regime", reg, None),
                             ("label band", band_of(y), [f"{lo}-{'inf' if hi == np.inf else hi}"
                                                         for lo, hi in BANDS]),
                             ("served by exp-036's mask", served, [True, False])):
        rows = attribute(y, base, cand, key, names)
        rec.setdefault("january_attribution", {})[what] = rows
        print(f"\n  by {what}:")
        print(f"    {'group':14s} {'n':>6s} {'v7':>9s} {'arm':>9s} {'share':>8s} {'s if only this':>15s}")
        for r in rows[:10]:
            print(f"    {r['group']:14s} {r['n']:6d} {r['rmse_base'] or 0:9.1f} "
                  f"{r['rmse_cand'] or 0:9.1f} {(r['share_of_change'] or 0) * 100:7.1f}% "
                  f"{-r['seconds_if_only_this_moved']:+15.3f}")

    (EXP / "analysis" / "tail-january.json").write_text(json.dumps(rec, indent=1, default=str))
    print("\nwrote analysis/tail-january.json")


if __name__ == "__main__":
    main()
