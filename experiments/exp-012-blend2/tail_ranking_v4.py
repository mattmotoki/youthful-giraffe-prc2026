"""Ranking predictions from the second lineage at the **v4 noac head**.

A copy of exp-008's `tail_ranking.py` with the head repinned. The v2 builder is
left untouched so the shipped v2 member stays rebuildable.

exp-002 delivered four-fold OOF and *noac-only* ranking predictions; a blend
that gives the tail model weight needs its prediction on every ranking row, so
this fits it on all twelve 2025 months and predicts both ranking months. It
changes nothing in `prc/` — it is a caller.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import prc.data as D  # noqa: E402
from prc.tail_model import TailModel  # noqa: E402

#: The noac-head version this experiment's blend weights were fitted against.
NOAC_PIN = "v4"   # see the note below: the weights have now been refit on v4

OUT = ROOT / "artifacts" / "exp-012" / "pred" / f"T-tail{NOAC_PIN}-ranking.parquet"


def main() -> None:
    t0 = time.time()
    train = D.load_months(D.MONTHS)
    print(f"[tail] train {train.shape} loaded in {time.time()-t0:.0f}s "
          f"(noac head pinned to {NOAC_PIN})", flush=True)
    # PIN THE NOAC HEAD. `prc.tail_model.NOAC_VERSION` moved from "v1" to "v2"
    # (commit 6d4edf0, 09:24 today) after this experiment's blend weights were
    # fitted. The weight this member carries -- 0.475, the largest in the blend --
    # was fitted on `exp-002-tail-oof.parquet`, which is v1, and v1/v2 differ
    # exactly on the noac rows that carry a fifth of the squared error. Serving v2
    # against a v1-fitted weight would mean the candidate is not the estimator
    # that was evaluated, so the version is pinned to what the OOF says.
    # To move to v2, refit the weights on a v2 OOF and re-report; do not just
    # unpin this.
    #
    # 2026-09-06, exp-012: that precondition is now MET, which is the only reason
    # this variant exists. The weights were refit on `exp-009-noac2-oof-v3.parquet`
    # (sha d7e028fa, the v4 noac mixture) and re-reported: the swap is worth
    # -1.187 s of clean board in nnls and -0.987 s flat, with the tail member's
    # own weight rising 0.078 -> 0.201. So serving v4 against v4-fitted weights is
    # the matched pairing, and serving v2 would now be the mismatch.
    #
    # Module drift checked before building rather than assumed. OOF v3 was built
    # under tail_model 2.2.0 and this fit runs under 2.5.0. 2.4.0 ends the band
    # mixture's domain at BAND_TOP = 26 h, which touches ZERO rows on every
    # delivered fold and exactly one ranking row (+2,728.52 s, the
    # reviewer's projected +0.46 s of board). 2.5.0 adds the schedule-placeholder
    # regime, which is off by default and stays off here. So the fold predictions
    # the weights were fitted on are unchanged and this is not a repeat of B2.
    model = TailModel().fit(train, noac_head=NOAC_PIN)
    print(f"[tail] fitted at {time.time()-t0:.0f}s", flush=True)
    rk = D.load_ranking()
    parts = []
    for m, g in rk.groupby("month", sort=True):
        p = model.predict(g)
        p["MVT_ID_mvt"] = g["MVT_ID_mvt"].reindex(p.index)
        parts.append(p[["MVT_ID_mvt", "y_pred"]])
        print(f"[tail] {m}: {len(p)} rows at {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[tail] wrote {OUT.relative_to(ROOT)} rows={len(out)} sha256={D.sha256(OUT)} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
