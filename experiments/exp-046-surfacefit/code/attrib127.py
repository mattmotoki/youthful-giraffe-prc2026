"""The attribution triple at the capacity that SHIPS, not at the one it was proved on.

The control (`c_lgb_shipped_noobs`, -0.4063 s) and the placebo (`c_lgb_shipped_placebo`,
-0.4316 s) were priced at the predeclared LightGBM: 31 leaves, 300 rounds.  The candidate
that ships is the same head at **127 leaves**.  A larger model can extract more from the
shipped block alone, so the attribution argument -- "the whole of the gain is the row's own
aircraft, because the same fit without the observation reads negative" -- is only proved at
the size it was measured at.  This re-prices both nulls at 127 leaves so the claim and the
candidate rest on one model size.

Same rotation, same seed, same cross-fit day blocks, same per-airport amplitude, same
floor, same clauses, same permutation report as `arms.shuffle_obs`.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/attrib127.py
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
ROOT = EXP.parent.parent

import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402

SETTING = "leaves_127x300_lr050"
PARAMS = {"num_leaves": 127}
OUT = EXP / "analysis/attribution-127.json"


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    ship, scols = A.shipped_cols(pan)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    pan_shuf, shuf_report = A.shuffle_obs(pan, obs_cols)
    rule, _ = A.serving_rule()
    out = {
        "what": ("the no-observation control and the both-clocks placebo at the capacity "
                 "that ships (127 leaves), so the attribution and the candidate rest on "
                 "one model size"),
        "setting": SETTING, "lgb_override": PARAMS,
        "rule": {"sha256": rule,
                 "block_cache_keys": {m: A.block_cache_key(m, rule)
                                      for m in A.FOLD_MONTH.values()}},
        "permutation": shuf_report,
        "at_31_leaves": {
            "arm": 8.1410, "control": -0.4063, "placebo": -0.4316,
            "source": "analysis/arms.json"},
        "arms": {},
    }
    plan = {
        "c_lgb_shipped@127": ("real", list(obs_cols)),
        "c_lgb_shipped_noobs@127": ("real", []),          # the shipped block alone
        "c_lgb_shipped_placebo@127": ("shuffled", list(obs_cols)),
    }
    for name, (src_kind, ocols) in plan.items():
        t0 = time.time()
        src = pan_shuf if src_kind == "shuffled" else pan
        d_by_fold, detail = {}, {}
        for f in A.FOLDS:
            others = [g for g in A.FOLDS if g != f]
            fit = pd.concat([src[g][src[g].usable] for g in others], ignore_index=True)
            sco = src[f]
            fit = fit.join(pd.concat([ship[g] for g in others])
                           .reindex(fit.MVT_ID_mvt.to_numpy("float64"))
                           .reset_index(drop=True))
            sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                           .reset_index(drop=True))
            d, info = A.arm_lgb(fit, sco, list(ocols) + scols, params=PARAMS)
            d = A.apply_floor(pan[f].pbase.to_numpy("float64"), d)
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{name} {f}: a noac row moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{name} {f}: a row outside the with-clock class moved"
            d_by_fold[f] = d
            detail[f] = {k: info[k] for k in ("weights", "n_features", "oof_rmse_vs_r11")}
        # a placebo is FITTED on the permuted panel and PRICED against the real one, so
        # `r`, the day and the class are the real ones and only the identity is destroyed
        rec = BA.price(pan, d_by_fold, rng)
        rec["detail"] = detail
        rec["fitted_on"] = src_kind
        rec["n_observation_columns"] = len(ocols)
        rec["tail_clauses"] = {f: A.tail_clauses(pan[f], d_by_fold[f]) for f in A.FOLDS}
        rec["seconds"] = round(time.time() - t0, 1)
        out["arms"][name] = rec
        np.savez_compressed(EXP / f"work/d-attrib-{name.replace('@', '-')}.npz", **d_by_fold)
        print(f"{name:30s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{rec['board_all_row']['interval']['ci95'][0]:+7.3f},"
              f"{rec['board_all_row']['interval']['ci95'][1]:+7.3f}] "
              f"clean {rec['board_clean']['gain_s']:+8.4f}  {rec['seconds']:.0f}s",
              flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    a = out["arms"]["c_lgb_shipped@127"]["board_all_row"]["gain_s"]
    c = out["arms"]["c_lgb_shipped_noobs@127"]["board_all_row"]["gain_s"]
    pl = out["arms"]["c_lgb_shipped_placebo@127"]["board_all_row"]["gain_s"]
    out["reading"] = {
        "arm": a, "control": c, "placebo": pl,
        "control_and_placebo_agree_to_s": abs(c - pl),
        "the_whole_gain_is_the_rows_own_aircraft": bool(c < 0 and pl < 0 and a > 0),
        "note": ("the claim holds at the capacity that ships only if BOTH nulls stay "
                 "negative here; a larger model extracting more from the shipped block "
                 "alone would show up as a control above zero"),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"arm {a:+.4f}  control {c:+.4f}  placebo {pl:+.4f}  "
          f"nulls agree to {abs(c - pl):.4f} s")


if __name__ == "__main__":
    main()
