"""L3, the free re-fits: weights only, no model, on committed OOF (desk nested validation rule).

`docs/QUEUE-2026-09-07.md` L3 lists four re-fits.  Three cost nothing but
arithmetic and are run here; the fourth needs a model fit and only its free half
is run.

1. **Fit the weights on the clean population.**  NNLS currently minimises
   squared error over ALL rows, including the two unreached-rollover rows the
   january fold contains and `rmse_clean` excludes -- rows carrying 116 s of
   january RMSE that the ranking months are believed not to contain at all
   (`exp-020 poisson`, P(k=0) = 0.516, confirmed by v1's board read).  They pull
   the fitted vector toward a population we are not scored on.  This is
   exp-017's A1 finding one layer up, and it has never been tested.

2. **Weight the fit by ranking-month share** rather than by row count, so
   december -- which has no counterpart in the board mix -- stops contributing
   32 % of the fit rows.  The literal reading gives december weight 0; because
   that degenerates a two-fold fit to one fold, the whole curve in december's
   weight is swept rather than one point taken.

3. **Drop `D1-winter`.**  Already measured, in two places that agree:
   `docs/queue-subsets-2026-09-07.csv` and `analysis/ladder_loo.csv` both put
   the six-member rotation at clean 259.178 against blend-5's 259.134, a cost of
   0.044 s.  The queue says this item "does not need running; it needs
   deciding", so it is reported from the record and not recomputed.

4. **Swap `V3-allin` -> `V3-roll171`.**  Not free: it has two folds and no
   ranking fit.  What IS free is the two-fold rotation on the OOF that exists
   (`published/V3-roll171-v1-{july_loo,january}.parquet`), read against the
   **0.1687 s** two-fold half-width rather than the three-fold 0.1025 s.  That
   is run; the december fold and the twelve-month fit are not, and the arm
   cannot be served without them.

**The kill for all of them** is the queue's: a change that does not improve the
rotated clean board estimate by more than the half-width is not adopted.  Both
currencies are reported -- face value, and with blend-5's `noac`-concentrated
gain given back (`code/noac_discount.py`, RESULTS section 5f), because §6b
showed the discounted column is the one that predicted the board.

    .venv/bin/python experiments/exp-026-rebuild/code/refit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import rule as R  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = F.EXP / "analysis"
B5 = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin", "N1-nmtail")
#: The queue's kill thresholds: the blend lane's measured rotation half-width.
HALF_WIDTH_3FOLD = 0.1025
HALF_WIDTH_2FOLD = 0.1687
#: Ranking-month share by fold. December has no counterpart in the board mix.
SHARE = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621, "december": 0.0}


# ------------------------------------------------------------ weighted fit ---
def fit_weighted(fit: pd.DataFrame, names, w: np.ndarray) -> dict:
    """`rule.fit_hybrid` with a per-row weight on the objective.

    Weighted least squares is ordinary least squares on `sqrt(w) * P` and
    `sqrt(w) * y`, so the same NNLS solves it and the non-negativity and the
    ridge are untouched. `w = 1` reproduces `fit_hybrid` exactly, which is
    asserted in `main` rather than assumed.
    """
    names = list(names)
    P, sel = R._split(fit, names)
    gen = [n for n in names if n != R.SPECIALIST]
    y = fit["y_true"].to_numpy("float64")
    s = np.sqrt(np.asarray(w, "float64"))
    w_late = B.fit_weights(P[sel] * s[sel, None], y[sel] * s[sel], "nnls")
    Pg = np.stack([fit[n].to_numpy("float64") for n in gen], axis=1)
    w_out = B.fit_weights(Pg[~sel] * s[~sel, None], y[~sel] * s[~sel], "nnls")
    late = dict(zip(names, map(float, w_late)))
    out = {n: 0.0 for n in names}
    out.update(dict(zip(gen, map(float, w_out))))
    return {"nm_late": late, "nm_bulk": out, "nonm": out, "noac": out, "_all": out}


def row_weights(fit: pd.DataFrame, clean: bool, dec: float | None) -> np.ndarray:
    """Per-row objective weight. `clean` drops the excluded rows; `dec` is
    december's share relative to the two ranking months' own shares."""
    w = np.ones(len(fit), "float64")
    if clean:
        ids = fit["MVT_ID_mvt"].to_numpy("float64")
        for f in F.FOLDS:
            ex = D.excluded_ids(f)
            if ex:
                w[np.isin(ids, list(ex))] = 0.0
    if dec is not None:
        fold = fit["fold"].to_numpy()
        for f in F.FOLDS:
            share = dec if f == "december" else SHARE[f]
            m = fold == f
            if m.any():
                # normalise by fold size so a share is a share of the objective,
                # not an accident of how many rows the month happens to have
                w[m] *= share / m.sum() * len(fit)
    return w


def rotate(data: dict, names, clean: bool, dec: float | None,
           folds=F.FOLDS) -> dict:
    res = {}
    for f in folds:
        ap = data[f]
        fit = pd.concat([data[g] for g in folds if g != f], ignore_index=True)
        w = row_weights(fit, clean, dec)
        if w.sum() <= 0:
            raise ValueError(f"scoring {f}: every fitting row has weight 0")
        W = fit_weighted(fit, names, w)
        p = R.apply_hybrid(ap, names, W)
        y = ap["y_true"].to_numpy("float64")
        res[f] = dict(D.rmse_both(y, p, ap["MVT_ID_mvt"].to_numpy("float64"), f),
                      pred=p, weights=W)
    keys = [f for f in F.SHARES if f in folds]
    if len(keys) == 2:
        res["board_clean"] = F.board_estimate({f: res[f]["rmse_clean"] for f in keys})
    else:
        res["board_clean"] = float("nan")
    return res


# ---------------------------------------------------------------- discount ---
def noac_discount(data: dict, res: dict, names) -> float:
    """The clean board with blend-4's noac head substituted back in.

    RESULTS section 5f's currency: blend-5's gain over blend-4 is 85 % / 67 % a
    `noac` gain on 402 fold rows, and the board said that gain did not arrive.
    Giving it back is the pessimistic reading every candidate is also quoted at.
    """
    import ladder as L
    base = L.run(L.B4, {})
    per = {}
    for f in F.SHARES:
        ap = data[f]
        p = res[f]["pred"].copy()
        noac = ap["regime"].to_numpy() == "noac"
        p[noac] = base[f]["pred"][noac]
        per[f] = D.rmse_both(ap["y_true"].to_numpy("float64"), p,
                             ap["MVT_ID_mvt"].to_numpy("float64"), f)["rmse_clean"]
    return F.board_estimate(per)


def no_noac_board(data: dict, res: dict) -> float:
    """The clean board over the rows that are NOT `noac`.

    The discount of `noac_discount` substitutes the SAME blend-4 head on those
    rows for every arm, so a difference between two arms' discounted boards is
    already a difference on the non-noac rows only. This states that directly,
    in exp-027's `no-noac` currency (blend-5 = 231.2698), so the two readings
    can be checked against each other.
    """
    per = {}
    for f in F.SHARES:
        ap = data[f]
        keep = ap["regime"].to_numpy() != "noac"
        per[f] = D.rmse_both(ap["y_true"].to_numpy("float64")[keep],
                             res[f]["pred"][keep],
                             ap["MVT_ID_mvt"].to_numpy("float64")[keep], f)["rmse_clean"]
    return F.board_estimate(per)


# ------------------------------------------------------------------- main ---
#: `V3-roll171`'s OOF, which `frames.py` does not carry because the member has
#: no ranking file and cannot be served. Registered here, locally, so the free
#: half of L3 item 4 can run without changing the addresses the candidates use.
V3ROLL = "experiments/exp-012-blend2/published/V3-roll171-v1-{fold}.parquet"


def v3roll(data: dict) -> pd.DataFrame:
    """L3 item 4, the free half: the two folds `V3-roll171` actually has."""
    folds = ("july_loo", "january")
    swapped = tuple("V3-roll171" if n == "V3-allin" else n for n in B5)
    missing = [f for f in folds if not (F.ROOT / V3ROLL.format(fold=f)).exists()]
    if missing:
        return pd.DataFrame([{"arm": "V3-roll171 swap",
                              "note": f"OOF absent for {missing}"}])
    d2 = {}
    for f in folds:
        base = data[f].copy()
        r = pd.read_parquet(F.ROOT / V3ROLL.format(fold=f))
        r = r[r["fold"] == f] if "fold" in r.columns else r
        s = pd.Series(r["y_pred"].to_numpy("float64"),
                      index=r["MVT_ID_mvt"].to_numpy("float64"))
        v = s.reindex(base["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
        if not np.isfinite(v).all():
            return pd.DataFrame([{"arm": "V3-roll171 swap",
                                  "note": f"{int((~np.isfinite(v)).sum())} of "
                                          f"{len(v)} rows unmatched on {f}"}])
        # prepared as every other member is: floor, then the residual guard
        a = base["d_takeoff_aobt"].to_numpy("float64")
        v = np.maximum(v, B.FLOOR)
        base["V3-roll171"] = B.apply_residual_guard(v, a, *F.GUARD, B.FLOOR)
        d2[f] = base
    rows = []
    for lab, nm, dd in (("blend-5 (2 folds)", B5, {f: data[f] for f in folds}),
                        ("V3-allin -> V3-roll171", swapped, d2)):
        r = rotate(dd, nm, clean=False, dec=None, folds=folds)
        rows.append({"arm": lab, **{f: round(r[f]["rmse_clean"], 4) for f in folds},
                     "clean board (2 folds)": round(r["board_clean"], 4)})
    g = rows[0]["clean board (2 folds)"] - rows[1]["clean board (2 folds)"]
    rows[1]["vs blend-5"] = round(-g, 4)
    rows[1]["verdict"] = "CLEARS" if g > HALF_WIDTH_2FOLD else "killed"
    rows[1]["servable?"] = "no: needs a december fold and a 12-month ranking fit"
    return pd.DataFrame(rows)


def table(rows) -> str:
    df = pd.DataFrame(rows)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def main() -> None:
    names = B5
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}

    # --- the gate on the instrument: w = 1 must reproduce the shipped rule ----
    base = rotate(data, names, clean=False, dec=None)
    ref = R.rotate(data, names)
    d = max(abs(base[f]["rmse_clean"] - ref[f]["rmse_clean"]) for f in F.FOLDS)
    print(f"[refit] gate: weighted fit at w=1 reproduces the shipped rule, "
          f"max delta {d:.3e} s")
    if d > 1e-9:
        raise SystemExit(f"the weighted fit does not reduce to fit_hybrid ({d:.3g} s)")
    b0 = base["board_clean"] = F.board_estimate(
        {f: base[f]["rmse_clean"] for f in F.SHARES})
    disc0 = noac_discount(data, base, names)
    nn0 = no_noac_board(data, base)
    print(f"[refit] baseline clean board {b0:.4f}, discounted {disc0:.4f}, "
          f"no-noac {nn0:.4f}\n")

    rows = [{"arm": "blend-5 as shipped (control)", "july_loo": round(base["july_loo"]["rmse_clean"], 3),
             "december": round(base["december"]["rmse_clean"], 3),
             "january": round(base["january"]["rmse_clean"], 3),
             "clean board": round(b0, 4), "vs control": 0.0,
             "predict": round(b0 + 4, 2),
             "discounted": round(disc0, 4), "no-noac": round(nn0, 4),
             "verdict face": "—", "verdict disc.": "—"}]

    arms = [("1. clean-objective weights", True, None),
            ("2. month-share, december 0.00", False, 0.00),
            ("2. month-share, december 0.25", False, 0.25),
            ("2. month-share, december 0.50", False, 0.50),
            ("2. month-share, december 0.75", False, 0.75),
            ("1+2. clean + december 0.00", True, 0.00),
            ("1+2. clean + december 0.50", True, 0.50)]
    for lab, clean, dec in arms:
        r = rotate(data, names, clean=clean, dec=dec)
        bb = F.board_estimate({f: r[f]["rmse_clean"] for f in F.SHARES})
        dd = noac_discount(data, r, names)
        nn = no_noac_board(data, r)
        gain = b0 - bb
        rows.append({"arm": lab, "july_loo": round(r["july_loo"]["rmse_clean"], 3),
                     "december": round(r["december"]["rmse_clean"], 3),
                     "january": round(r["january"]["rmse_clean"], 3),
                     "clean board": round(bb, 4), "vs control": round(-gain, 4),
                     "predict": round(bb + 4, 2),
                     "discounted": round(dd, 4), "no-noac": round(nn, 4),
                     "verdict face": "SHIPS" if gain > HALF_WIDTH_3FOLD else "killed",
                     "verdict disc.": ("SHIPS" if (disc0 - dd) > HALF_WIDTH_3FOLD
                                       else "killed")})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "refit_l3.csv", index=False)
    print("=== L3 items 1 and 2: free re-fits, three-fold rotation ===")
    print(f"(kill: a gain of {HALF_WIDTH_3FOLD} s or less is not adopted)")
    print(table(rows))

    print("\n=== L3 item 4, the free half: V3-roll171 on the two folds it has ===")
    print(f"(kill: a gain of {HALF_WIDTH_2FOLD} s or less is not adopted)")
    V = v3roll(data)
    V.to_csv(OUT / "refit_v3roll.csv", index=False)
    print(table(V.to_dict("records")))

    best = T.iloc[1:].sort_values("clean board").iloc[0]
    best_d = T.iloc[1:].sort_values("discounted").iloc[0]
    rec = {"baseline_clean_board": round(b0, 4), "baseline_discounted": round(disc0, 4),
           "half_width_3fold": HALF_WIDTH_3FOLD, "half_width_2fold": HALF_WIDTH_2FOLD,
           "best_arm": best["arm"], "best_clean_board": float(best["clean board"]),
           "best_gain": round(b0 - float(best["clean board"]), 4),
           "best_verdict_face": best["verdict face"],
           "best_arm_discounted": best_d["arm"],
           "best_discounted": float(best_d["discounted"]),
           "best_gain_discounted": round(disc0 - float(best_d["discounted"]), 4),
           "best_verdict_discounted": best_d["verdict disc."],
           "item3_drop_D1_winter": {"clean_board": 259.178, "cost": 0.044,
                                    "source": "analysis/ladder_loo.csv and "
                                              "docs/queue-subsets-2026-09-07.csv",
                                    "note": "already measured; a decision, not a run"}}
    # A point estimate against a half-width is the queue's kill rule; an
    # interval says whether the winner is distinguishable from the control at
    # all. Same day-clustered paired bootstrap the stacker rotation uses.
    import stack5 as S
    lab = best_d["arm"]
    clean_f, dec_f = dict((a[0], (a[1], a[2])) for a in arms)[lab]
    rw = rotate(data, names, clean=clean_f, dec=dec_f)
    iv = []
    for f in F.FOLDS:
        ap = data[f]
        keep = ap["regime"].to_numpy() != "noac"
        d_, lo_, hi_ = S.boot_delta(ap["y_true"].to_numpy("float64")[keep],
                                    rw[f]["pred"][keep], base[f]["pred"][keep],
                                    S.clusters(ap)[keep])
        iv.append({"arm": lab, "fold": f, "delta_no_noac": round(d_, 4),
                   "lo": round(lo_, 4), "hi": round(hi_, 4),
                   "excludes_zero": bool(hi_ < 0 or lo_ > 0)})
    pd.DataFrame(iv).to_csv(OUT / "refit_interval.csv", index=False)
    print(f"\n=== {lab}: day-clustered 95 % against the control, noac rows out ===")
    print(table(iv))
    rec["interval_no_noac"] = iv
    (OUT / "refit_l3.json").write_text(json.dumps(rec, indent=2))
    # The sweep's shape, because one point of five chosen by the metric quoted
    # is selection, and the curve says whether the winner is a mechanism or a
    # wobble.
    sw = [(0.0, None), (0.25, None), (0.5, None), (0.75, None), (1.0, nn0)]
    for i, (w, _) in enumerate(sw[:-1]):
        m = T[T.arm == f"2. month-share, december {w:.2f}"]
        sw[i] = (w, float(m["no-noac"].iloc[0]))
    print("\n=== december's weight against the no-noac board (1.00 is the control) ===")
    print("  " + "  ".join(f"{w:.2f}:{v:.3f}" for w, v in sw))
    drop = sw[0][1] - sw[-1][1]
    print(f"  removing december entirely costs {drop:+.3f} s of no-noac board; the "
          f"curve falls to {min(sw, key=lambda z: z[1])[0]:.2f} then turns")

    print(f"\n[refit] best on FACE       {best['arm']!r}: {rec['best_gain']:+.4f} s "
          f"-> {best['verdict face']}")
    print(f"[refit] best on DISCOUNTED {best_d['arm']!r}: "
          f"{rec['best_gain_discounted']:+.4f} s -> {best_d['verdict disc.']}")


if __name__ == "__main__":
    main()
