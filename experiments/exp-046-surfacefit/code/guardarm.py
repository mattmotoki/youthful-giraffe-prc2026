"""Price the guarded arm and register it, so the serve and both gates can read it.

The guarded arm is not a new fit: it is `c_lgb_shipped@leaves_255x300_lr050` with the
frozen guard of `code/guard.py` applied to its saved update vector.  This writes

  work/d-guard-leaves_255x300_lr050-guard995.npz    the guarded fold-side vector
  analysis/guard-arms.json                          the contract table and the two cuts

in the shape `serve.priced_arms()` reads, so the arm enters the pool, `choose_arm()` can
see it, `fold_gate` can re-price it with the hold and `increment_gate` can price it against
the arm v15 ships.  No refit; one fit, one mask.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/guardarm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import guard as G                                                     # noqa: E402

BASE_ARMS = ["c_lgb_shipped@leaves_255x300_lr050",
             "c_lgb_shipped@leaves_127x300_lr050"]
OUT = EXP / "analysis/guard-arms.json"


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    rule, _ = A.serving_rule()
    out = {"what": "the capacity arms with the frozen tail guard applied, no refit",
           "guard": {"quantile": G.Q, "tag": G.TAG,
                     "expression": "hold the update where pbase - obs_taxi > cut",
                     "frozen_by": "code/guard.py",
                     "climb": "95 / 99 / 99.5 / 99.8 / 99.9 / 99.95, stopped at the "
                              "turnover (analysis/tail-guard.json)"},
           "rule": {"sha256": rule}, "settings": {}}

    # the two cuts, both label-free: leave-one-out for a scored fold, all three for serving
    cuts = {f: G.cut(pan, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    out["guard"]["cut_s_by_scored_fold"] = {f: round(c, 2) for f, c in cuts.items()}
    out["guard"]["cut_s_for_the_ranking_serve"] = round(G.cut(pan, list(A.FOLDS)), 2)

    for base_arm in BASE_ARMS:
        z = np.load(S.npz_path(base_arm))
        kept, held = {}, {}
        for f in A.FOLDS:
            d = z[f].astype("float64").copy()
            m = G.mask(pan[f].pbase.to_numpy("float64"), pan[f].obs_taxi, cuts[f])
            held[f] = int((d[m] != 0).sum())
            d[m] = 0.0
            kept[f] = d
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{base_arm} {f}: noac moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{base_arm} {f}: a row outside the with-clock class moved"
        rec = BA.price(pan, kept, rng)
        rec["tail_clauses"] = {f: A.tail_clauses(pan[f], kept[f]) for f in A.FOLDS}
        rec["rows_held_by_the_guard_by_fold"] = held
        rec["guarded_from"] = base_arm
        rec["guard_cut_s_by_scored_fold"] = {f: round(c, 2) for f, c in cuts.items()}
        setting = base_arm.split("@", 1)[1] + G.NAME
        out["settings"][setting] = rec
        np.savez_compressed(EXP / f"work/d-guard-{setting.replace('+', '-')}.npz", **kept)
        tc = {f: round(rec["tail_clauses"][f]["registered_clause_y_over_3600s"]
                       ["rmse_gain_s"], 3) for f in A.FOLDS}
        ok = all(rec["tail_clauses"][f]["registered_clause_y_over_3600s"]["passes_the_clause"]
                 for f in A.FOLDS)
        print(f"{setting:34s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{rec['board_all_row']['interval']['ci95'][0]:+7.3f},"
              f"{rec['board_all_row']['interval']['ci95'][1]:+7.3f}] "
              f"clean {rec['board_clean']['gain_s']:+8.4f} held {held} tail {tc} "
              f"{'ALL PASS' if ok else 'TAIL FAILS'}", flush=True)
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
