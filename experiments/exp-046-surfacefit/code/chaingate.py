"""Do the three independent checks corroborate any of the ungated chain pairs?

Reads the frames `code/chaincheck.py` wrote -- the real pairs and the two placebos -- and
runs, in the order the desk's addendum set:

  (0) the WHY split of the ungated pool: a poorly observed wheels-off (the clause failed for
      an instrument reason) against a WELL observed one that still disagrees (evidence
      AGAINST the placement, which leaves the pool);
  (i) the off-block clock, segment P against the departure's AOBT A;
  (ii) runway agreement from the wheels-off position;
  (iii) stand-position agreement.

Every threshold is a quantile of the SHIPPED 3,179 -- never of the pool being judged and
never of a placebo -- so a check cannot be tuned to the rows it is deciding.  Each check is
reported as a true-pass rate `t` (on the shipped), a placebo-pass rate `f` (on pairs that
are wrong by construction), and the precision arithmetic.

The pool's own true fraction is ESTIMATED rather than assumed: if a pool of prior `pi`
passes at rate `q`, then `q = pi*t + (1-pi)*f`, so `pi_hat = (q - f) / (t - f)`.  That is
the number the desk needs, and it is what decides whether any of these rows can ship.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/chaingate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

ARM = "c_lgb_shipped@leaves_255x300_lr050+guard995"
LEVEL = 246.6571
N_RANKING = 344841
#: the quantile of the shipped rows every threshold is taken at
Q = 95
OUT = EXP / "analysis/chain-gate.json"


def q95(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.percentile(v, Q)) if len(v) else float("nan")


def rate(mask: np.ndarray, have: np.ndarray) -> dict:
    n = int(have.sum())
    k = int((mask & have).sum())
    return {"rows": int(len(have)), "with_the_field": n, "passes": k,
            "rate": (k / n if n else None),
            # Wilson 95 % interval, so a rate on 27 placebo rows is not read as a point
            "ci95": _wilson(k, n)}


def _wilson(k: int, n: int) -> list | None:
    if not n:
        return None
    z, p = 1.959963985, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def main() -> None:
    P = pd.read_parquet(EXP / "work/chain-pairs.parquet")
    PL = {k: pd.read_parquet(EXP / f"work/chain-pairs-placebo-{k}.parquet")
          for k in ("day", "stand")}
    ship = P[P.gated].copy()
    pool0 = P[~P.gated].copy()
    out = {"what": __doc__.strip().splitlines()[0], "threshold_quantile_on_the_shipped": Q,
           "shipped_rows": int(len(ship)), "ungated_rows_before_step_0": int(len(pool0)),
           "stale_reference_in_the_record": {
               "value": 8478, "now": int(len(pool0)),
               "why": ("8,478 = 11,579 - 3,101, both counted BEFORE the claimed-set repair "
                       "of section 21.5. On the shipped matcher the pass makes 11,432 unique "
                       "pairs and gates 3,179, so the ungated pool is 8,253. The probe "
                       "reproduces analysis/chain-pass.json exactly (2026-01: 4,785 pairs, "
                       "3,403 ungated; 2026-07: 6,647 and 4,850)")}}

    # ---------------------------------------------------- (0) why the clause failed
    # `wheels_gap_s` = t_airborne - obs_wheels_off: how tightly the take-off instant is
    # pinned by the segment itself.  On the shipped rows it is a median of 29 s; a pair
    # whose airborne confirmation is hours away from its own wheels-off has not observed a
    # take-off at all, and its failure of the +-60 s clause says nothing about the pairing.
    cut_gap = q95(ship.wheels_gap_s.to_numpy("float64"))
    well = pool0.wheels_gap_s.to_numpy("float64") <= cut_gap
    out["step_0_why_the_take_off_check_failed"] = {
        "field": "wheels_gap_s = t_airborne - obs_wheels_off",
        "threshold_s": cut_gap,
        "shipped_median_s": float(np.nanmedian(ship.wheels_gap_s)),
        "pool_median_s": float(np.nanmedian(pool0.wheels_gap_s)),
        "well_observed_and_still_disagrees": int(well.sum()),
        "poorly_observed": int((~well).sum()),
        "reading": ("the well-observed class is EVIDENCE AGAINST the placement and leaves "
                    "the pool; only the poorly-observed class can be rescued by an "
                    "independent check")}
    pool = pool0[~well].copy()
    out["pool_after_step_0"] = int(len(pool))
    print(f"step 0: of {len(pool0)} ungated, {int(well.sum())} have a well-observed "
          f"wheels-off that still disagrees (dropped), {len(pool)} remain", flush=True)

    # ------------------------------------------------------------- the three checks
    cut_off = q95(ship.abs_dt_offblock.to_numpy("float64"))
    cut_std = q95(ship.stand_km.to_numpy("float64"))
    checks = {
        "i_offblock_clock": ("abs_dt_offblock", "le", cut_off,
                             "|segment off-block - departure AOBT_3_flt| <= the shipped p95"),
        "ii_runway_agreement": ("runway_agrees", "eq", None,
                                "the runway the wheels-off position implies IS RUNWAY_mvt"),
        "iii_stand_position": ("stand_km", "le", cut_std,
                               "the segment's start position is within the shipped p95 of "
                               "the placed departure's learned stand centroid"),
    }

    def passes(df: pd.DataFrame, col: str, how: str, thr) -> tuple:
        if how == "eq":
            v = df[col].to_numpy(bool)
            return v, np.ones(len(df), bool)
        x = df[col].to_numpy("float64")
        have = np.isfinite(x)
        return (x <= thr) if how == "le" else (x >= thr), have

    res = {}
    for name, (col, how, thr, what) in checks.items():
        t_m, t_h = passes(ship, col, how, thr)
        t = rate(t_m, t_h)
        rec = {"what": what, "threshold": thr, "true_on_the_shipped": t, "placebo": {}}
        for lab, q in PL.items():
            f_m, f_h = passes(q, col, how, thr)
            rec["placebo"][lab] = rate(f_m, f_h)
            gg = q[q.gated]
            gm, gh = passes(gg, col, how, thr)
            rec["placebo"][lab + "_gated_placements_only"] = rate(gm, gh)
        p_m, p_h = passes(pool, col, how, thr)
        rec["pool_after_step_0"] = rate(p_m, p_h)
        p0m, p0h = passes(pool0, col, how, thr)
        rec["pool_before_step_0"] = rate(p0m, p0h)

        # the mixture estimate and the precision, against the worse of the two placebos
        tt = t["rate"]
        ff = max(rec["placebo"]["day"]["rate"], rec["placebo"]["stand"]["rate"])
        qq = rec["pool_after_step_0"]["rate"]
        pi = ((qq - ff) / (tt - ff)) if (tt is not None and ff is not None
                                         and qq is not None and tt > ff) else None
        rec["false_rate_used"] = ff
        rec["pool_true_fraction_estimate"] = pi
        rec["precision_if_that_estimate_holds"] = (
            (pi * tt) / (pi * tt + (1 - pi) * ff)
            if pi is not None and 0 <= pi <= 1 and (pi * tt + (1 - pi) * ff) > 0 else None)
        rec["precision_at_a_grid_of_priors"] = {
            str(p): ((p * tt) / (p * tt + (1 - p) * ff))
            for p in (0.1, 0.25, 0.5, 0.75, 0.9)
            if tt is not None and ff is not None and (p * tt + (1 - p) * ff) > 0}
        res[name] = rec
        print(f"{name:22s} true {tt:.4f}  placebo day "
              f"{rec['placebo']['day']['rate']:.4f} stand {rec['placebo']['stand']['rate']:.4f}"
              f"  pool {qq:.4f}  pi_hat "
              f"{'--' if pi is None else format(pi, '.4f')}", flush=True)
    out["checks"] = res
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------------- the conjunction, and what it keeps
    def all_pass(df):
        m = np.ones(len(df), bool)
        for name, (col, how, thr, _) in checks.items():
            pm, ph = passes(df, col, how, thr)
            m &= pm & ph
        return m
    t_all = all_pass(ship); p_all = all_pass(pool)
    f_all = {lab: all_pass(q) for lab, q in PL.items()}
    tt, ff = t_all.mean(), max(f_all["day"].mean(), f_all["stand"].mean())
    qq = p_all.mean()
    pi = ((qq - ff) / (tt - ff)) if tt > ff else None
    out["all_three_checks"] = {
        "true_rate_on_the_shipped": float(tt),
        "placebo_rate": {k: float(v.mean()) for k, v in f_all.items()},
        "pool_pass_rate": float(qq), "kept_rows": int(p_all.sum()),
        "pool_true_fraction_estimate": pi,
        "precision": ((pi * tt) / (pi * tt + (1 - pi) * ff)
                      if pi is not None and 0 <= pi <= 1 else None)}
    print(f"\nall three: true {tt:.4f} placebo {ff:.4f} pool {qq:.4f} "
          f"keeps {int(p_all.sum())} rows, pi_hat "
          f"{'--' if pi is None else format(pi,'.4f')}", flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------------------------------------- the face
    # measured, not borrowed: the served arm's own squared-error gain per mode-3 row
    pan = A.panel_with_block(BA.load())
    z = np.load(S.npz_path(ARM))
    num = den = 0.0
    for f in A.FOLDS:
        p = pan[f]
        m3 = (p.match_mode.fillna(-1).to_numpy("float64") == 3.0) & p.usable.to_numpy(bool)
        d = z[f].astype("float64")[m3]
        r = p.rbase.to_numpy("float64")[m3]
        num += float((2 * r * d - d * d).sum()); den += int(m3.sum())
    g = num / den if den else 0.0
    out["face"] = {"s2_per_chain_row_measured_on_the_folds": g,
                   "fold_mode3_rows": int(den), "level_s": LEVEL}
    for label, n in (("all three checks", int(p_all.sum())),
                     ("runway only", int(((pool.runway_agrees.to_numpy(bool))).sum())),
                     ("the whole pool after step 0", int(len(pool)))):
        gg = g * n / N_RANKING
        lvl = LEVEL - (LEVEL ** 2 - gg) ** 0.5
        out["face"][label] = {"rows": n, "board_s_at_x1_2": lvl,
                              "board_s_at_x1_5": lvl * 1.492}
        print(f"face  {label:28s} {n:>5} rows -> {lvl:+.4f} board s at x1.2")
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
