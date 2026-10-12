"""The control that decides whether any of this screen family can serve.

Steps 2 and 3 both put a matched base `h0` -- v10 plus the CURRENT clock,
runway and window summaries, no new information at all -- through the screen
family's standard fit: five contiguous day blocks inside the scoring month, four
fitted, the fifth scored.  That base reads **+0.38 s of all-row fold board**
(+0.82 s on july_loo alone).  A base carrying nothing new cannot be worth that,
so either v10 leaves a large learnable residual, or the within-month cross-fit is
reading month-specific structure that no served model can have: the 2026 ranking
months come with no labels at all.

This file separates the two.  Same features, same learner, same capacity, same
rows, same contract.  Only the fitting data changes:

    within       the screen convention: day blocks inside the scoring month
    cross        fitted on the OTHER TWO fold months' eligible rows; no row and
                 no day of the scored month enters the fit
    cross_amp    the same prediction at one scalar amplitude fitted on the other
                 two folds -- Q03's cross-month device, never on the scored rows

and each of those for the base alone and for the base plus Q10's two-factor
block, so the step-3 MAYBE is priced under the same separation.

    python crossmonth.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B      # noqa: E402
import order as O     # noqa: E402
import twofactor as T  # noqa: E402

sys.path.insert(0, str(B.S39))
import rowfeat as R   # noqa: E402
import anatomy as AN  # noqa: E402

EXP = B.EXP
FOLDS = B.FOLDS
D = B.D
PK = B.PK
CAP = O.HEADLINE_CAP
BLOCKS_OF_FAMILIES = ("base", "two")


def build(fold: str, pan: pd.DataFrame) -> dict:
    """One fold's rows, residual, days and the two feature blocks."""
    m = B.panel(fold, pan)
    month = R.month_of(fold)
    ac = AN.arrival_cells(month)
    others = [x for x in D.MONTHS if x != month]
    nu = T.Nuisance().fit(others)
    df = T._delay_frame(month)
    u = pd.Series(df["u"].to_numpy() - nu.predict(df),
                  index=df["mid"].to_numpy()).reindex(
                      m["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    have = np.isfinite(u)
    apt = m["apt"].to_numpy(); bins = m["bin"].to_numpy()
    rwy = m["rwy"].to_numpy(); ope = m["op"].to_numpy()
    F = T.routed(*B.codes_of(apt, bins), *B.codes_of(apt, bins, rwy),
                 *B.codes_of(apt, bins, ope), *B.codes_of(apt, bins, rwy, ope),
                 u, have)
    return {"m": m, "ac": ac, "two_block": np.column_stack([
                F["f_same_rwy_diff_op"], F["n_same_rwy_diff_op"],
                F["f_diff_rwy_same_op"], F["n_diff_rwy_same_op"]]),
            "ei": np.flatnonzero(m["elig"].to_numpy()),
            "days": m["day"].to_numpy()}


def vocabulary(S: dict) -> dict:
    """One ordinal encoding shared by every fold.  Unused: the shipped block's
    own codes already are one (Sol round 1, blocking finding 7).  Kept so a
    caller that wants hand-built ordinals can still build them."""
    v = {}
    for c in ("apt",) + O.BASE_CATS:
        vals = sorted(set().union(*(set(S[f]["m"][c].to_numpy()) for f in FOLDS)))
        v[c] = {x: i for i, x in enumerate(vals)}
    return v


def rotation_leak() -> dict:
    """The confound this control cannot remove on the v10 base, stated plainly.

    Sol round 1 finding 7, Sol round 2 finding 4.  The cross fit trains on the
    other folds' residuals `r_g = y_g - p10_g`, and `p10_g` is out of fold for
    `g` only: the members behind it were trained on month sets that CONTAIN the
    scored month `f` (`prc/data.py`, the NESTED_FOLDS comment).  So the training
    TARGET knows `f`'s labels one step removed.

    **The direction argument round 1 accepted is withdrawn.**  It claimed the
    leak could only inflate the cross reading, so a negative was conservative.
    That is not established: contaminated training targets change what the model
    learns, and the change has no proven sign -- it can help, hurt, or rotate the
    fitted correction.  Nothing in the v10-based `cross` arms of this file can
    settle the portability question by itself.

    The repair is the `clean` family in this same file: a base trained INSIDE
    this experiment on one other month, with `p10` removed from the columns, so
    no v10 vector and no rotation enters any arm.  Every base a clean arm uses
    excludes the scored fold, so the leak has no path.  The v10-based arms stay
    in the record as the contaminated reading, next to the clean one.

    The other clean instrument, still not on disk, is the six nested folds of
    `prc.data.NESTED_FOLDS`; this function looks for their vectors.
    """
    found = sorted(str(p.relative_to(B.ROOT))
                   for p in B.ROOT.glob("experiments/*/oof/*_nx_*.parquet"))
    return {"training_target_sees_the_scored_month": True,
            "why": "each fold's p10 is out-of-fold for that fold only; its "
                   "members were trained on month sets containing the other "
                   "folds' months (prc/data.py, NESTED_FOLDS)",
            "clean_instrument": "the six nested folds of prc.data.NESTED_FOLDS, "
                                "or -- what this file now runs -- the `clean` "
                                "family, whose base is fitted here on one other "
                                "month with p10 dropped from the columns",
            "nested_vectors_on_disk": found,
            "direction": "WITHDRAWN (Sol round 2, finding 4).  A contaminated "
                         "training target has no proven sign: it can help, hurt "
                         "or rotate the fitted correction, so the v10-based "
                         "cross arms cannot settle portability on their own",
            "also_uncontrolled": "within -> cross changes the month AND the "
                                 "rotation that produced p10 together"}


def clean_family(S: dict, pan: dict, rng) -> dict:
    """Portability with NO v10 vector anywhere -- the leak-free instrument.

    Sol round 2, blocking finding 4: on the v10 base, every cross arm trains on a
    residual whose `p10` was produced by members that saw the scored month, and
    the direction of that contamination is not established.  This family removes
    v10 from the question entirely.

    For a scored fold `f`, a base predictor is fitted HERE, on ONE other month,
    from the same columns minus `p10`, and the same construction is used for the
    two training folds -- each of their bases is fitted on a month that is
    neither its own nor `f`.  So no model behind any number in this family has
    seen a row of `f`.  Then the same two arms as above:

        clean_within   day blocks inside `f`, on `f`'s own clean residual
        clean_cross    fitted on the other two folds' clean residuals

    and the endpoint is the same ranking-weighted all-row board, with the clean
    base in the place v10 holds for the other arms.
    """
    names = S[FOLDS[0]]["names"]
    assert names[0] == "p10", names[:1]
    base_fits: dict[str, HistGradientBoostingRegressor] = {}

    def base_of(train: str) -> HistGradientBoostingRegressor:
        if train not in base_fits:
            Xt = S[train]["X"]["base"][:, 1:]
            yt = S[train]["m"]["y_true"].to_numpy("float64")
            g = HistGradientBoostingRegressor(random_state=B.SEED, **O.CAPS[CAP])
            g.fit(Xt, yt)
            base_fits[train] = g
            print(f"  clean base fitted on {train} ({len(yt):,} rows)", flush=True)
        return base_fits[train]

    def train_for(x: str, f: str) -> str:
        return sorted(g for g in FOLDS if g != x and g != f)[0]

    out, preds, refs = {}, {}, {}
    for f in FOLDS:
        rc, src = {}, {}
        for x in FOLDS:
            tr = train_for(x, f)
            src[x] = tr
            yx = S[x]["m"]["y_true"].to_numpy("float64")
            rc[x] = yx - base_of(tr).predict(S[x]["X"]["base"][:, 1:])
        s = S[f]
        e = s["ei"]
        Xf = s["X"]["base"][:, 1:]
        h_within = O.fit_arm(Xf, rc[f], e, s["blk"], CAP)
        Xg = np.vstack([S[g]["X"]["base"][:, 1:][S[g]["ei"]] for g in FOLDS if g != f])
        yg = np.concatenate([rc[g][S[g]["ei"]] for g in FOLDS if g != f])
        gm = HistGradientBoostingRegressor(random_state=B.SEED, **O.CAPS[CAP])
        gm.fit(Xg, yg)
        h_cross = np.zeros(len(rc[f]))
        h_cross[e] = gm.predict(Xf[e])
        y = s["m"]["y_true"].to_numpy("float64")
        base_pred = y - rc[f]
        refs[f] = (s["days"], (y - base_pred) ** 2)
        for arm, h in (("clean_within", h_within), ("clean_cross", h_cross)):
            A_ = float((rc[f][e] * h[e]).mean()); Bq = float((h[e] ** 2).mean())
            v = 2 * rc[f][e] * h[e] - h[e] ** 2
            pt, lo, hi = B.boot_ratio(v, np.ones(len(e)), s["days"][e], rng)
            out.setdefault(arm, {})[f] = {
                "A_mean_r_h": round(A_, 3), "B_mean_h2": round(Bq, 3),
                "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                "unrestricted_scale": round(A_ / Bq, 4) if Bq > 0 else None,
                "multiday": PK.boot_blocks(v, s["days"][e], rng),
                "base_trained_on": src[f],
                "rmse_all_base": round(D.rmse(y, base_pred), 4),
                "rmse_all_new": round(D.rmse(y, base_pred + h), 4),
                "gain_all": round(D.rmse(y, base_pred)
                                  - D.rmse(y, base_pred + h), 4)}
            preds.setdefault(arm, {})[f] = base_pred + h
    boards = {}
    for arm, pr in preds.items():
        parts = {f: (refs[f][0], refs[f][1],
                     (S[f]["m"]["y_true"].to_numpy("float64")
                      - pr[f]) ** 2) for f in FOLDS}
        boards[arm] = B.board_boot(parts, rng)
    return {"note": "no v10 vector and no rotation enters this family; the base "
                    "is fitted here on one other month, with p10 dropped from "
                    "the columns, and every base a scored fold uses excludes "
                    "that fold.  `board_v10` in the interval block is the CLEAN "
                    "BASE's board, not v10's.",
            "arms": out, "boards": boards}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(B.SEED)
    pan = B.V.load()
    S = {f: build(f, pan[f]) for f in FOLDS}
    for f in FOLDS:
        #  Sol round 1, blocking finding 7: the first version dropped the shipped
        #  block's 13 categorical codes and replaced them with six hand-built
        #  ordinals, on the belief that the codes were per-month.  They are not --
        #  one persistent vocabulary encodes every month -- so this control now
        #  reads EXACTLY the columns steps 2 and 3 read, and the only thing that
        #  changes between `within` and `cross` is the fitting data.
        base, base_names = O.base_features(S[f]["m"], S[f]["ac"], R.month_of(f))
        S[f]["X"] = {"base": base,
                     "two": np.column_stack([base, S[f]["two_block"]])}
        S[f]["names"] = base_names
        S[f]["r"] = S[f]["m"]["r"].to_numpy("float64")
        S[f]["blk"] = B.blocks_of(S[f]["days"])

    H, DIAG = {}, {}
    for fam in BLOCKS_OF_FAMILIES:
        for f in FOLDS:
            s = S[f]
            H[(fam, "within", f)] = O.fit_arm(s["X"][fam], s["r"], s["ei"],
                                              s["blk"], CAP)
            Xf = np.vstack([S[g]["X"][fam][S[g]["ei"]] for g in FOLDS if g != f])
            yf = np.concatenate([S[g]["r"][S[g]["ei"]] for g in FOLDS if g != f])
            g_ = HistGradientBoostingRegressor(random_state=B.SEED, **O.CAPS[CAP])
            g_.fit(Xf, yf)
            h = np.zeros(len(s["r"]))
            h[s["ei"]] = g_.predict(s["X"][fam][s["ei"]])
            H[(fam, "cross", f)] = h
            # Is the cross fit broken, or is the residual simply not portable?
            # Two diagnostics separate those: the model must fit its OWN training
            # rows (in-sample 2A - B strongly positive), and its prediction must
            # look like the within-month model's prediction (a correlation well
            # above zero).  A broken fit fails the first; a non-portable residual
            # fails neither and still loses out of month.
            hin = g_.predict(Xf)
            A_ = float((yf * hin).mean()); Bq = float((hin ** 2).mean())
            hw = H[(fam, "within", f)][s["ei"]]
            hc = h[s["ei"]]
            DIAG[(fam, f)] = {
                "insample_2A_B_s2": round(2 * A_ - Bq, 2),
                "insample_scale": round(A_ / Bq, 4) if Bq > 0 else None,
                "sd_h_within": round(float(hw.std()), 2),
                "sd_h_cross": round(float(hc.std()), 2),
                "corr_within_cross": round(float(np.corrcoef(hw, hc)[0, 1]), 4),
                "corr_r_cross": round(float(np.corrcoef(S[f]["r"][s["ei"]], hc)[0, 1]), 4),
                "corr_r_within": round(float(np.corrcoef(S[f]["r"][s["ei"]], hw)[0, 1]), 4)}
            print(f"  fitted {fam}/{f}: cross on {len(yf):,} rows from "
                  f"{[g for g in FOLDS if g != f]}; in-sample 2A-B "
                  f"{DIAG[(fam, f)]['insample_2A_B_s2']:+.0f} s^2, "
                  f"corr(h_within, h_cross) {DIAG[(fam, f)]['corr_within_cross']}, "
                  f"corr(r, h_cross) {DIAG[(fam, f)]['corr_r_cross']}, "
                  f"corr(r, h_within) {DIAG[(fam, f)]['corr_r_within']}", flush=True)

    # the cross-month amplitude: for fold f, a scalar fitted on the OTHER folds'
    # cross predictions.
    #
    #  Sol round 2, blocking finding 5: the first version calibrated on
    #  `H[(fam, "cross", g)]`, and that prediction was fitted on every fold except
    #  `g` -- which INCLUDES the target fold `f`.  The scalar therefore saw `f`'s
    #  rows through the model that produced the calibration predictions.  Each
    #  calibration prediction is now fitted with BOTH `f` and `g` excluded, which
    #  with three folds leaves exactly one training month; the prediction it
    #  calibrates is still the two-month `cross` model, and that mismatch is
    #  recorded rather than hidden.
    amps, cal_fits = {}, {}
    for fam in BLOCKS_OF_FAMILIES:
        for f in FOLDS:
            num = den = 0.0
            for g in FOLDS:
                if g == f:
                    continue
                train = tuple(x for x in FOLDS if x not in (f, g))
                key = (fam, train)
                if key not in cal_fits:
                    Xt = np.vstack([S[x]["X"][fam][S[x]["ei"]] for x in train])
                    yt = np.concatenate([S[x]["r"][S[x]["ei"]] for x in train])
                    gm = HistGradientBoostingRegressor(random_state=B.SEED,
                                                       **O.CAPS[CAP])
                    gm.fit(Xt, yt)
                    cal_fits[key] = gm
                e = S[g]["ei"]
                d = cal_fits[key].predict(S[g]["X"][fam][e])
                num += float((S[g]["r"][e] * d).sum()); den += float((d * d).sum())
            amps[(fam, f)] = num / den if den > 0 else 0.0
            H[(fam, "cross_amp", f)] = amps[(fam, f)] * H[(fam, "cross", f)]
            print(f"  amplitude {fam}/{f}: {amps[(fam, f)]:+.4f} "
                  f"(calibration models exclude both folds)", flush=True)

    recs, preds = {}, {}
    for fam in BLOCKS_OF_FAMILIES:
        for arm in ("within", "cross", "cross_amp"):
            name = f"{fam}_{arm}"
            pr = {}
            for f in FOLDS:
                s = S[f]; e = s["ei"]; h = H[(fam, arm, f)]
                y = s["m"]["y_true"].to_numpy("float64")
                p10 = s["m"]["p10"].to_numpy("float64")
                ids = s["m"]["MVT_ID_mvt"].to_numpy("float64")
                p_new = p10 + h
                pr[f] = p_new
                A_ = float((s["r"][e] * h[e]).mean()); Bq = float((h[e] ** 2).mean())
                v = 2 * s["r"][e] * h[e] - h[e] ** 2
                pt, lo, hi = B.boot_ratio(v, np.ones(len(e)), s["days"][e], rng)
                recs.setdefault(name, {})[f] = {
                    "A_mean_r_h": round(A_, 3), "B_mean_h2": round(Bq, 3),
                    "mse_gain_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                    "unrestricted_scale": round(A_ / Bq, 4) if Bq > 0 else None,
                    "amplitude": round(amps.get((fam, f), 1.0), 4)
                    if arm == "cross_amp" else 1.0,
                    "multiday": PK.boot_blocks(v, s["days"][e], rng),
                    "rmse_all_v10": round(D.rmse(y, p10), 4),
                    "rmse_all_new": round(D.rmse(y, p_new), 4),
                    "gain_all": round(D.rmse(y, p10) - D.rmse(y, p_new), 4),
                    "gain_clean": round(D.rmse_clean(y, p10, ids, f)
                                        - D.rmse_clean(y, p_new, ids, f), 4)}
                PK.dates("crossmonth", name, f, s["days"], y, p10, p_new)
            preds[name] = pr

    boards = {n: PK.primary(pan, preds[n], f"arm {n} at capacity {CAP}")
              for n in preds}
    for n in preds:
        parts = {f: (B.V.day(pan[f]["t"].to_numpy("float64")),
                     (pan[f]["y_true"].to_numpy("float64")
                      - pan[f]["p10"].to_numpy("float64")) ** 2,
                     (pan[f]["y_true"].to_numpy("float64")
                      - np.asarray(preds[n][f], "float64")) ** 2)
                 for f in FOLDS}
        boards[n]["board_interval"] = B.board_boot(parts, rng)
    clean = clean_family(S, pan, rng)
    out = {"screen": "crossmonth",
           "n_base_features": int(S[FOLDS[0]]["X"]["base"].shape[1]),
           "base_features": base_names,
           "predeclared": {"cap": CAP, "caps": O.CAPS, "blocks": B.BLOCKS,
                           "seed": B.SEED, "eligible_regime": B.ELIGIBLE_REGIME,
                           "hold_airports": list(B.HOLD_AIRPORTS),
                           "question": "how much of the within-month screen's "
                                       "gain survives when no row of the scored "
                                       "month enters the fit"},
           "rotation_leak": rotation_leak(),
           "clean_family": clean,
           "arms": recs, "boards": boards,
           "diagnostics": {f"{k[0]}:{k[1]}": v for k, v in DIAG.items()},
           "amplitudes":
           {f"{k[0]}:{k[1]}": round(v, 4) for k, v in amps.items()},
           "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "crossmonth.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    for n in preds:
        print(f"  oof: {B.write_oof(f'crossmonth-{n}', pan, preds[n])}")
    B.write_packet("crossmonth", out, pan)
    print("\n  2A - B per eligible row (s^2) and the all-row board:")
    for n, v in recs.items():
        row = "  ".join(f"{f[:4]} {v[f]['mse_gain_s2']:+8.2f} "
                        f"[{v[f]['ci'][0]:+.1f}, {v[f]['ci'][1]:+.1f}]"
                        for f in FOLDS)
        b = boards[n]["board_all_row"]
        print(f"    {n:16s} {row}   board {b['arm']:.4f} ({b['gain']:+.4f} s)"
              f"  clean {boards[n]['board_clean']['gain']:+.4f}")
    print(f"  amplitudes: {out['amplitudes']}")
    print("  leak-free clean family (base fitted here, p10 dropped):")
    for arm, v in clean["arms"].items():
        row = "  ".join(f"{f[:4]} {v[f]['mse_gain_s2']:+8.2f}" for f in FOLDS)
        print(f"    {arm:14s} {row}   board {clean['boards'][arm]['gain']:+.4f} s "
              f"3d {clean['boards'][arm]['ci_block_3d']}")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
