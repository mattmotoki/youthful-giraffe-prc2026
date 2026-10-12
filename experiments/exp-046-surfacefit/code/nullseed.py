"""The NULL: what does the comparator read when NOTHING changes? (desk, 2026-09-09)

Reads A and B both put july_loo clear below zero -- A the 511 rung at -62.58 s2/row, B the
19-row plane-metric correction at -85.43 -- and B's sign pattern showed the number could not
be the correction: july's class count did not change at all, while december (+23 rows) and
january (-4) both read POSITIVE.  The desk's ruling:

> "increment over v18's bytes" prices a NEW fit against ONE fit, and july's price fit is
> trained on december + january, whose classes moved by +23 / -4 rows.  PAUSE C, D and the
> repair.  Run the NULL first, pre-registered: the shipped recipe (255 leaves, shipped class,
> same params) refitted (i) with the same seed -- if it reproduces v19's fold vector bit for
> bit, the comparator is deterministic and B's july is a real training-set sensitivity;
> (ii) with two other seeds -- the refit-noise envelope per fold over v18's bytes, all-row
> and clean.

Three refits of the SHIPPED recipe on the SHIPPED class, changing one thing only: the
LightGBM seed, which drives `bagging_fraction` 0.8 and `feature_fraction` 0.8 and therefore
which rows and columns each tree sees.  Everything else is held -- the same panel, the same
rotation, the same five day blocks, the same per-airport amplitude, the same floor, the frozen
p99.5 guard, LFPG held, mode-3 pins asserted before the build -- and the bootstrap rng stays
at `arms.SEED` so the intervals are the intervals every other read on this lane carries.

Row 1 is the determinism check: seed 46 is the shipped recipe exactly, so it must reproduce
`work/d-bundle-ungated1299-guard995.npz` bit for bit.  Rows 2 and 3 are the envelope.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/nullseed.py
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
EXP = HERE.parent
ROOT = EXP.parents[1]

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import guard as G                                                     # noqa: E402
import rung511 as R                                                   # noqa: E402

#: seed 46 is `arms.SEED`, the shipped recipe; 2 and 3 are the envelope the desk named
SEEDS = [A.SEED, 2, 3]
LEAVES = 255
OUT = EXP / "analysis/null-seed.json"


def fit_at_seed(pan: dict, seed: int, leaves: int = LEAVES) -> dict:
    """`chainfold.fit_arm` with the LightGBM seed exposed; the airport hold is NOT applied
    here, so the caller applies it exactly where `fit_arm` does -- last."""
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in CF.BU.FEATURES if c not in ("stand_lat", "stand_lon")]
    d_by_fold = {}
    for f in A.FOLDS:
        others = [g for g in A.FOLDS if g != f]
        fit = pd.concat([pan[g][pan[g].usable] for g in others], ignore_index=True)
        sco = pan[f]
        fit = fit.join(pd.concat([ship[g] for g in others])
                       .reindex(fit.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
        sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                       .reset_index(drop=True))
        d, _ = A.arm_lgb(fit, sco, list(obs_cols) + scols, seed=seed,
                         params={"num_leaves": leaves})
        d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
        assert not np.any(d[~pan[f].usable.to_numpy(bool)]), f"{f}: a row outside the class moved"
        d_by_fold[f] = d
    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    out = {}
    for f in A.FOLDS:
        v = d_by_fold[f].astype("float64").copy()
        v[G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, cuts[f])] = 0.0
        out[f] = v
    return out


def main() -> None:
    t0 = time.time()
    if R.sha256(R.HEAD) != R.HEAD_SHA:
        raise SystemExit(f"{R.HEAD.name} is not the vector v18's bytes ship")
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b
    print("mode-3 asserted on the shipped builds", flush=True)
    pan = CF.panel_for(blk, R.THR, {m: CF.GAP_CUT_S for m in blk})
    H = {f: v.astype("float64") for f, v in np.load(R.HEAD).items()}

    out = {"what": __doc__.strip().splitlines()[0],
           "head": {"file": str(R.HEAD.relative_to(ROOT)), "sha256": R.HEAD_SHA},
           "seeds": SEEDS, "leaves": LEAVES,
           "shipped_seed": A.SEED,
           "held_constant": ("the panel, the rotation, the five day blocks, the per-airport "
                             "amplitude, the floor, the frozen p99.5 guard, the LFPG hold, "
                             "the bootstrap rng; only the LightGBM seed moves"),
           "draws": {}}
    for seed in SEEDS:
        t1 = time.time()
        cache = EXP / f"work/d-nullseed{seed}-255-free.npz"
        if cache.exists():
            d_free = {f: v.astype("float64") for f, v in np.load(cache).items()}
            refit = False
        else:
            d_free = fit_at_seed(pan, seed)
            np.savez_compressed(cache, **d_free)
            refit = True
        d_held = R.hold_lfpg(pan, d_free)
        same = {f: {"rows": int(len(H[f])),
                    "rows_differing_from_the_shipped_vector": int((d_held[f] != H[f]).sum()),
                    "max_abs_difference_s": float(np.abs(d_held[f] - H[f]).max()),
                    "bit_for_bit": bool(np.array_equal(d_held[f], H[f]))}
                for f in A.FOLDS}
        rec = R.price(pan, d_held, H, f"null: the shipped recipe refitted at seed {seed}")
        rec["reproduces_the_shipped_vector"] = same
        rec["refitted_this_run"] = refit
        rec["seconds"] = round(time.time() - t1, 1)
        rec["is_the_shipped_seed"] = bool(seed == A.SEED)
        out["draws"][str(seed)] = rec
        print(f"\n[null seed {seed}{'  (the shipped seed)' if seed == A.SEED else ''}]  "
              f"({rec['seconds']:.0f}s, refit {refit})", flush=True)
        print("    bit-for-bit vs the shipped vector: "
              + " ".join(f"{f[:3]} {same[f]['bit_for_bit']}"
                         f"({same[f]['rows_differing_from_the_shipped_vector']} rows, max "
                         f"{same[f]['max_abs_difference_s']:.3g} s)" for f in A.FOLDS),
              flush=True)
        print(R.row(rec), flush=True)
        print(R.fold_rows(rec), flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    env = {f: [out["draws"][str(s)]["by_fold"][f]["mse_gain_s2"] for s in SEEDS if s != A.SEED]
           for f in A.FOLDS}
    env_c = {f: [out["draws"][str(s)]["by_fold_clean"][f]["mse_gain_s2"]
                 for s in SEEDS if s != A.SEED] for f in A.FOLDS}
    board = [out["draws"][str(s)]["board_all_row"]["gain_s"] for s in SEEDS if s != A.SEED]
    out["envelope"] = {
        "what": ("the spread of a refit that changes NOTHING but the LightGBM seed; an "
                 "increment counts only where it lies outside this"),
        "by_fold_s2_per_row_raw": env, "by_fold_s2_per_row_clean": env_c,
        "board_all_row_s": board,
        "widest_abs_by_fold": {f: max(abs(x) for x in v) for f, v in env.items()},
        "widest_abs_board_s": max(abs(x) for x in board) if board else 0.0,
    }
    out["re_reading"] = {}
    for name, g, by in (("A_the_511_rung", -0.06281,
                         {"july_loo": -62.58, "december": 12.54, "january": -6.13}),
                        ("B_the_plane_metric_correction", -0.06087,
                         {"july_loo": -85.43, "december": 23.21, "january": 25.23})):
        out["re_reading"][name] = {
            "all_row_gain_s": g,
            "outside_the_board_envelope": bool(abs(g) > out["envelope"]["widest_abs_board_s"]),
            "by_fold": {f: {"s2_per_row": by[f],
                            "envelope_widest_abs": out["envelope"]["widest_abs_by_fold"][f],
                            "outside_the_envelope": bool(
                                abs(by[f]) > out["envelope"]["widest_abs_by_fold"][f])}
                        for f in A.FOLDS}}
    out["desk_resume_condition"] = {
        "rule": ("desk, 2026-09-09: if the same-seed refit reproduces v19 exactly and the "
                 "other seeds sit within +-10 s2/row on july, resume C and D as ordered"),
        "same_seed_reproduces_exactly": bool(all(
            out["draws"][str(A.SEED)]["reproduces_the_shipped_vector"][f]["bit_for_bit"]
            for f in A.FOLDS)),
        "july_envelope_within_10_s2_per_row": bool(
            max(abs(x) for x in env["july_loo"]) <= 10.0),
    }
    out["desk_resume_condition"]["resume"] = bool(
        out["desk_resume_condition"]["same_seed_reproduces_exactly"]
        and out["desk_resume_condition"]["july_envelope_within_10_s2_per_row"])
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("\nENVELOPE (seeds other than the shipped one), s2/row raw:", flush=True)
    for f in A.FOLDS:
        print(f"  {f:10s} {[round(x, 2) for x in env[f]]}  widest |.| "
              f"{out['envelope']['widest_abs_by_fold'][f]:.2f}", flush=True)
    print("  board all-row", [round(x, 5) for x in board], flush=True)
    print("\nRESUME condition:", json.dumps(out["desk_resume_condition"], default=float),
          flush=True)
    print("written to", OUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
