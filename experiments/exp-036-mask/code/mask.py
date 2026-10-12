"""Arms 1 and 2: the K2-minus-v7 change served per airport, and the mask it names.

v7 = v6 with EHAM served from stack-5 gained 0.230 s on the board for 0.163 s
clean (exp-030 arm 3, BOARD READING 7): a per-airport regime mask on a *served*
change transfers.  K2 (v9) changes 306,253 ranking rows against v7 and gained
0.380 s on the board for 0.2945 clean (BOARD READING 8).  This file asks the same
question of that change: at which of the ten airports does serving K2 instead of
v7 help, and what does holding the rest at v7 buy?

**Nothing is refitted.**  Every arm is `np.where` over three committed rotations
(`code/bases.py`), so a fold number here costs seconds and the arm's served file
is the same arithmetic on two verified candidate files.

The per-airport contribution is separable.  The clean sum of squared errors is
additive over rows, so the SSE the airport contributes is the same number whether
K2 is served at `a` alone or `a` is held at v7 inside the full K2 mask; RMSE is
monotone in that sum, so both RMSE movements carry the same SIGN.  Their RMSE
magnitudes are not equal -- the square root is concave, and the largest
disagreement in this table is 1.2e-4 s (Sol round 1, blocker 3).  The gate below
therefore checks the signs, which is the property the mask rule uses, and both
columns are printed so the reader does not have to take the algebra on trust.

    python mask.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import bases as Bs  # noqa: E402

Sc = Bs.Sc
FOLDS = Bs.FOLDS
ROOT = Bs.ROOT
#: `assignments/mask.md`: a shipping arm needs a clean gain of at least 0.10 s
#: over K2 with every fold positive.
GAIN_MIN = 0.10


def save(rec: dict) -> Path:
    """The arm's record, and COMMON.md's climb line, in this experiment."""
    p = EXP / "analysis" / f"arm-{rec['arm']}.json"
    p.write_text(json.dumps(rec, indent=2))
    log = EXP / "climb.log"
    n = sum(1 for ln in log.read_text().splitlines()
            if ln and not ln.startswith("#")) + 1 if log.exists() else 1
    with open(log, "a") as fh:
        fh.write(f"{n:03d}  {pd.Timestamp.utcnow():%Y-%m-%dT%H:%M}  {rec['arm']:22s} "
                 + "  ".join(f"{f} {rec['clean'][f]:9.4f}" for f in FOLDS)
                 + f"  board {rec['clean_board']:9.4f}"
                 + "  vs K2 " + " ".join(f"{rec['vs_K2'][f]:+.3f}" for f in FOLDS)
                 + f" board {rec['gain_board_vs_K2']:+.3f}"
                 + f"  gap-closed {rec['gain_board_vs_K2'] / Sc.ORACLE:+.4f} of the "
                   f"{Sc.ORACLE} s window oracle"
                 + f"  predict {rec.get('predicted_board_vs_K2', float('nan')):.2f}"
                 + f"  gate {'PASS' if rec['gate_mask'] else 'FAIL'}\n")
    return p


def arm(name: str, preds: dict, ref: dict, vec: dict, note: str,
        extra_bases=("v7",)) -> dict:
    """One arm's full record: exp-030's scoring, then the gain over K2."""
    rec = Sc.score(name, preds, ref, note=note)
    Sc.show(rec)
    Bs.against(rec, preds, ref, vec["K2"], "K2", Bs.PUBLISHED["K2"]["measured_board"])
    Bs.show(rec, "K2")
    for b in extra_bases:
        Bs.against(rec, preds, ref, vec[b], b, Bs.PUBLISHED[b]["measured_board"])
        Bs.show(rec, b)
    # `score.score` sets `gate`, `gate_gain` and `gate_folds` against **v6**,
    # which is exp-030's comparison point and not this experiment's.  The exp-032
    # correction row in `docs/LEDGER.md` was written because an inherited `gate`
    # field read `true` on an arm that failed the assignment's real gate, so the
    # inherited fields are renamed here and the only field called `gate_*_K2` is
    # the one `assignments/mask.md` actually sets.
    for k in ("gate", "gate_gain", "gate_folds"):
        rec[f"{k}_v6_inherited"] = rec.pop(k)
    rec["gate_gain_K2"] = bool(rec["gain_board_vs_K2"] >= GAIN_MIN)
    rec["gate_folds_K2"] = bool(rec["worst_fold_vs_K2"] > 0.0)
    rec["gate_mask"] = bool(rec["gate_gain_K2"] and rec["gate_folds_K2"])
    print(f"  GATE vs K2: gain >= {GAIN_MIN} {rec['gate_gain_K2']}, "
          f"every fold positive {rec['gate_folds_K2']}  ->  "
          f"{'PASS' if rec['gate_mask'] else 'FAIL'}")
    return rec


def per_airport(ref: dict, vec: dict, src: str = "K2") -> pd.DataFrame:
    """For each airport: serving `src`'s change there alone, and holding it out."""
    apts = sorted(set(ref["july_loo"]["airport"].astype(str)))
    rows = []
    for a in apts:
        r = {"airport": a}
        alone, out = {}, {}
        for f in FOLDS:
            d = ref[f]
            m = d["airport"].astype(str).to_numpy() == a
            y = d["y_true"].to_numpy("float64")
            ids = d["MVT_ID_mvt"].to_numpy("float64")
            ok = ~d["MVT_ID_mvt"].isin(Bs.D.excluded_ids(f)).to_numpy()
            alone[f] = np.where(m, vec[src][f], vec["v7"][f])
            out[f] = np.where(m, vec["v7"][f], vec[src][f])
            r[f"n_{f}"] = int((m & ok).sum())
            r[f"gain_{f}"] = round(
                Bs.D.rmse_clean(y, vec["v7"][f], ids, f)
                - Bs.D.rmse_clean(y, alone[f], ids, f), 4)
            r[f"heldout_{f}"] = round(
                Bs.D.rmse_clean(y, out[f], ids, f)
                - Bs.D.rmse_clean(y, vec[src][f], ids, f), 4)
            cl = Sc.P.clusters(d)
            dl, lo, hi = Sc.P.boot_delta(y[ok], alone[f][ok], vec["v7"][f][ok],
                                         pd.factorize(cl[ok])[0])
            r[f"lo_{f}"], r[f"hi_{f}"] = round(-hi, 4), round(-lo, 4)
        r["folds_positive"] = int(sum(r[f"gain_{f}"] > 0 for f in FOLDS))
        r["board_alone"] = round(
            Sc.P.board({f: Bs.D.rmse_clean(ref[f]["y_true"], vec["v7"][f],
                                           ref[f]["MVT_ID_mvt"], f) for f in FOLDS})
            - Sc.P.board({f: Bs.D.rmse_clean(ref[f]["y_true"], alone[f],
                                             ref[f]["MVT_ID_mvt"], f) for f in FOLDS}), 4)
        # Serving the change at `a` alone and holding `a` out of the full mask
        # contribute the same clean SSE, so the two RMSE columns must agree in
        # SIGN on every fold; a disagreement means the mask is not separable and
        # the table below it would be meaningless.  Their magnitudes are not
        # equal and are not checked (Sol round 1, blocker 3).
        r["signs_agree"] = bool(all(
            np.sign(r[f"gain_{f}"]) == np.sign(r[f"heldout_{f}"]) for f in FOLDS))
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ref, vec, clean = Bs.load()
    print("[mask] bases gated: "
          + "; ".join(f"{n} board {Sc.P.board(c):.4f}" for n, c in clean.items()),
          flush=True)

    T = per_airport(ref, vec, "K2")
    T = T.sort_values("board_alone", ascending=False)
    T.to_csv(EXP / "analysis" / "airport_change.csv", index=False)
    print("\n=== serving the K2 change at one airport, against v7 (clean, + is better) ===")
    print(T[["airport"] + [f"gain_{f}" for f in FOLDS] + ["board_alone",
              "folds_positive", "signs_agree"] + [f"heldout_{f}" for f in FOLDS]
              + [f"n_{f}" for f in FOLDS]]
          .to_string(index=False))
    if not T["signs_agree"].all():
        raise SystemExit("the served-alone and held-out forms disagree in sign")

    keep = sorted(T.loc[T["folds_positive"] == 3, "airport"])
    drop = sorted(T.loc[T["folds_positive"] != 3, "airport"])
    print(f"\n[mask] kept on K2 ({len(keep)}): {', '.join(keep)}")
    print(f"[mask] held at v7 ({len(drop)}): {', '.join(drop)}")
    (EXP / "analysis" / "mask.json").write_text(json.dumps(
        {"rule": "serve the K2 change only where it gains on all three folds",
         "keep": keep, "hold_at_v7": drop,
         "note": "EHAM is held from stack-5 in v7 and in K2 alike, so the change "
                 "is exactly zero there and it cannot be kept by this rule.",
         "table": "analysis/airport_change.csv"}, indent=2))

    # Diagnostic, not a selection: the same table for refit-bulk, which is what
    # says whether arm 2's outcome is the mask or the anchor repair.  Arm 2 is
    # `assignments/mask.md`'s words -- **the same mask** -- so nothing here
    # chooses a second one.
    TB = per_airport(ref, vec, "refit-bulk").sort_values("board_alone", ascending=False)
    TB.to_csv(EXP / "analysis" / "airport_change_refitbulk.csv", index=False)
    print("\n=== the same table for refit-bulk (diagnostic; the mask is K2's) ===")
    print(TB[["airport"] + [f"gain_{f}" for f in FOLDS]
              + ["board_alone", "folds_positive"]].to_string(index=False))

    recs = []
    for label, src, extra in (("k2-mask", "K2", ("v7",)),
                              ("refitbulk-mask", "refit-bulk", ("v7", "refit-bulk"))):
        preds = {}
        for f in FOLDS:
            m = np.isin(ref[f]["airport"].astype(str).to_numpy(), keep)
            preds[f] = np.where(m, vec[src][f], vec["v7"][f])
        rec = arm(label, preds, ref, vec,
                  note=f"{src} served at {', '.join(keep)}; the other "
                       f"{10 - len(keep)} airports held at v7. No refit: "
                       f"np.where over committed rotations.",
                  extra_bases=extra)
        rec.update({"source": src, "keep": keep, "hold_at_v7": drop,
                    "mask_rule": "gains on all three folds against v7"})
        recs.append(rec)
        print(f"\n  written: {save(rec)}\n", flush=True)

    # The control arm 2 needs: refit-bulk with no mask, against K2.  Without it
    # the masked refit-bulk's gain cannot be split into what the anchor repair
    # buys and what the mask buys.
    rec = arm("refitbulk-nomask", vec["refit-bulk"], ref, vec,
              note="the refit-bulk MAYBE exactly as exp-027 measured it, priced "
                   "against K2 rather than against v7. Arm 2's control.",
              extra_bases=("v7",))
    # `keep` is the field `serve.py` masks on, so it is a list of airport codes in
    # every record or it is not one.  Sol round 1, blocker 1: this line used to
    # store the string "all ten", which `serve.py` turned into seven one-character
    # "airports".  The control is not a candidate, so its mask is empty and
    # `servable` says so; `serve.py` refuses an empty or non-list mask.
    rec.update({"source": "refit-bulk", "keep": [], "hold_at_v7": [],
                "servable": False,
                "keep_note": "every airport; this arm is arm 2's control, not a "
                             "candidate, and it is not served"})
    recs.append(rec)
    print(f"\n  written: {save(rec)}", flush=True)


if __name__ == "__main__":
    main()
