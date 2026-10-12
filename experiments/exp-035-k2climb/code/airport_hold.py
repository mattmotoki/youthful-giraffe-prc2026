"""Section 25: exp-036's own rule applied to this member, airport by airport.

The desk, 13:30 UTC: serve form 2' per airport only where it **gains on all three
folds** AND **meets the tail clause on that airport's own tail rows**, hold v11
everywhere else.  exp-036 chose its eight-airport mask by the first of those two
tests on the K2 change (`experiments/exp-036-mask/code/mask.py`,
`analysis/airport_change.csv`, rule "serve the K2 change only where it gains on
all three folds"); this file asks the same question of the climbed member and
adds the clause that stopped form 2' in §24.

**This is a POST-HOC selection on the folds it is scored on**, exactly as
exp-036's mask was, and it is recorded as such: the airports are chosen by
reading three fold numbers and the arm is then scored on those same three folds.
Under `docs/QUEUE-2026-09-07.md`'s line on selection channels that makes it
**discovery evidence**, not a passed gate, and the leave-one-fold-out column is
printed beside every gain so a reader can see how much of each depends on the
fold it is read on.

Form 2' already differs from v11 **only** at exp-036's eight kept airports (it is
the same mask and the same 0.217 blend over a different head), so EHAM, EDDF,
`noac` and every unkept airport are bit-identical to v11 in every arm here, by
construction and not by a further rule.  That is asserted, not assumed.

Separability, as exp-036 argued it: clean SSE is additive over rows, so the SSE
an airport contributes is the same whether the member is served at that airport
alone or the airport is held inside the full arm.  RMSE is monotone in that sum,
so the SIGN of an airport's contribution is the same either way; the magnitudes
differ slightly because the square root is concave, and both are printed.

    PRC_THREADS=3 python airport_hold.py
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
sys.path.insert(0, str(HERE))
import price_vs_v11 as PV  # noqa: E402

FOLDS = V.FOLDS
TAIL = PV.TAIL
ALLOWANCE = 1.0
DRAWS = 2000


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> None:
    ref, vec, prov, files = V.load(verbose=False)
    keep = json.loads((ROOT / V.MASK_JSON).read_text())["keep"]
    rec38 = json.loads(PV.V11_RECORD.read_text())
    S = float(rec38["s"])
    p10, s5, v7 = vec[PV.BASE], vec["stack-5"], vec["v7"]
    p11 = {f: p10[f] + S * (s5[f] - p10[f]) for f in FOLDS}
    head_rb = PV.load_head(ref, PV.CAND_RB_OOF, PV.CAND_RB_PUBLISHED,
                           "the mod head (refit-bulk anchor)")
    apt = {f: ref[f]["airport"].astype(str).to_numpy() for f in FOLDS}
    m = {f: np.where(np.isin(apt[f], keep), head_rb[f], v7[f]) for f in FOLDS}
    form2p = {f: m[f] + S * (s5[f] - m[f]) for f in FOLDS}

    # the construction claim, asserted
    for f in FOLDS:
        off = form2p[f] - p11[f]
        bad = float(np.abs(off[~np.isin(apt[f], keep)]).max())
        if bad > 1e-9:
            raise SystemExit(f"{f}: form 2' differs from v11 off the kept airports by {bad:g}")

    clean = {f: ~ref[f]["MVT_ID_mvt"].isin(V.D.excluded_ids(f)).to_numpy() for f in FOLDS}
    y = {f: ref[f]["y_true"].to_numpy("float64") for f in FOLDS}
    day = {f: A.day_of(ref[f]["t"].to_numpy("float64")) for f in FOLDS}

    rows, table = [], {}
    print("=== form 2' against v11, airport by airport (clean rows) ===")
    for a in sorted(keep):
        r = {"airport": a, "folds": {}}
        pos, clause_ok = 0, 0
        for f in FOLDS:
            sel = clean[f] & (apt[f] == a)
            yy, pv, pc = y[f][sel], p11[f][sel], form2p[f][sel]
            g = rmse(yy, pv) - rmse(yy, pc)
            blk = pd.factorize(day[f][sel])[0]
            ci = A.boot_fold(blk, (yy - pv) ** 2, (yy - pc) ** 2, DRAWS)
            # the airport's own tail rows, and the registered clause on them
            t = sel & (y[f] > TAIL)
            yt = y[f][t]
            if t.sum() == 0:
                # LEBL and LEMD have NO rows above the threshold on any fold, so
                # the clause is vacuously met there: it cannot be violated on a
                # row set that is empty.  Recorded as such, not scored as a pass.
                tv7 = tv11 = tc = None
                ok, vacuous = True, True
            else:
                tv7, tv11 = rmse(yt, v7[f][t]), rmse(yt, p11[f][t])
                tc = rmse(yt, form2p[f][t])
                ok, vacuous = bool(tv7 - tc > -ALLOWANCE), False
            # the whole-fold board if this airport alone were served
            sse_v11 = float(np.sum((y[f][clean[f]] - p11[f][clean[f]]) ** 2))
            d_sse = float(np.sum((yy - pc) ** 2) - np.sum((yy - pv) ** 2))
            n_cl = int(clean[f].sum())
            alone = float(np.sqrt(sse_v11 / n_cl) - np.sqrt((sse_v11 + d_sse) / n_cl))
            r["folds"][f] = {"n": int(sel.sum()), "gain": round(g, 4),
                             "ci": [round(x, 4) for x in ci],
                             "gain_alone_whole_fold": round(alone, 4),
                             "tail_n": int(t.sum()),
                             "tail_rmse_v7": None if vacuous else round(tv7, 2),
                             "tail_rmse_v11": None if vacuous else round(tv11, 2),
                             "tail_rmse_arm": None if vacuous else round(tc, 2),
                             "tail_vs_v7": None if vacuous else round(tv7 - tc, 4),
                             "tail_vs_v11": None if vacuous else round(tv11 - tc, 4),
                             "clause_pass": ok, "clause_vacuous": vacuous}
            pos += int(g > 0)
            clause_ok += int(ok)
        r["folds_positive"] = pos
        r["folds_clause_pass"] = clause_ok
        r["selected"] = bool(pos == 3 and clause_ok == 3)
        # The clause read POOLED over the three folds' tail rows instead of fold
        # by fold.  It is not the rule applied above and selects nothing here; it
        # is reported because several of these airports carry 1 to 11 tail rows on
        # a fold, where a per-fold reading is mostly noise, and because the desk
        # said it would decide on the clause itself.
        yy = np.concatenate([y[f][clean[f] & (apt[f] == a) & (y[f] > TAIL)] for f in FOLDS])
        pv7 = np.concatenate([v7[f][clean[f] & (apt[f] == a) & (y[f] > TAIL)] for f in FOLDS])
        pca = np.concatenate([form2p[f][clean[f] & (apt[f] == a) & (y[f] > TAIL)] for f in FOLDS])
        if len(yy):
            r["tail_pooled"] = {"n": int(len(yy)), "rmse_v7": round(rmse(yy, pv7), 2),
                                "rmse_arm": round(rmse(yy, pca), 2),
                                "vs_v7": round(rmse(yy, pv7) - rmse(yy, pca), 4)}
            r["tail_pooled"]["clause_pass"] = bool(r["tail_pooled"]["vs_v7"] > -ALLOWANCE)
        else:
            r["tail_pooled"] = {"n": 0, "clause_pass": True, "vacuous": True}
        r["selected_pooled_clause"] = bool(pos == 3 and r["tail_pooled"]["clause_pass"])
        r["selected_gain_only"] = bool(pos == 3)
        rows.append(r)
        table[a] = r
        print(f"  {a}  gains " + " ".join(f"{r['folds'][f]['gain']:+7.4f}" for f in FOLDS)
              + f"  ({pos}/3 positive)   tail vs v7 "
              + " ".join((f"{r['folds'][f]['tail_vs_v7']:+8.2f}"
                        if r['folds'][f]['tail_vs_v7'] is not None else "   no tail")
                       for f in FOLDS)
              + f"  ({clause_ok}/3 pass)   -> "
              + ("SERVE" if r["selected"] else "HOLD"))

    sel = sorted(a for a in table if table[a]["selected"])
    sel_pool = sorted(a for a in table if table[a]["selected_pooled_clause"])
    sel_gain = sorted(a for a in table if table[a]["selected_gain_only"])
    print(f"\n  selected (the rule as ordered, clause per fold): {sel}")
    print(f"  held at v11: {sorted(set(keep) - set(sel))}")
    print(f"\n  for the desk's decision on the clause itself, from the same table:")
    print(f"    clause read POOLED over the three folds: {sel_pool}")
    print(f"    no clause at all (exp-036's original rule, gain on 3 folds): {sel_gain}")
    for a in sorted(table):
        tp = table[a]["tail_pooled"]
        print(f"      {a}  pooled tail n={tp['n']:4d}  "
              + (f"{tp['rmse_v7']:9.1f} -> {tp['rmse_arm']:9.1f}  {tp['vs_v7']:+8.2f}"
                 if tp["n"] else "        no tail rows on any fold"))

    out = {"rule": "serve form 2' at an airport only where it gains on all three folds "
                   "AND its own tail rows meet the registered clause "
                   "(y > 3600 clean, RMSE against v7, 1 s allowance); v11 elsewhere",
           "selection": "POST-HOC on the three folds it is scored on, as exp-036's mask was; "
                        "discovery evidence, not a passed gate",
           "source": "experiments/exp-036-mask/code/mask.py rule, plus §24's clause",
           "candidate_airports": sorted(keep), "selected": sel,
           "selected_pooled_clause": sel_pool, "selected_gain_only": sel_gain,
           "held_at_v11": sorted(set(keep) - set(sel)),
           "allowance": ALLOWANCE, "tail_threshold": TAIL, "draws": DRAWS,
           "table": table}
    (EXP / "analysis" / "airport-hold.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nwrote analysis/airport-hold.json")


if __name__ == "__main__":
    main()
