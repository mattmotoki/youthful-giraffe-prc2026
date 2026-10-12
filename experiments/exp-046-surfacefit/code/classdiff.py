"""The published class vector against v15's, row by row, with the disagreements named.

Round 3 finding 1 said `serve.ranking_block()` had to build ONE BLOCK PER MONTH.  The
vector published for the composition (`artifacts/exp-046/class/<month>.parquet`) was
written before that repair, from a whole-file build; v15's class comes from the per-month
build.  They differ, and a composition that reads one while a candidate was served on the
other puts rows on both sides of the boundary -- exactly the failure the published vector
existed to stop.

This names every disagreement, so RESULTS can carry the rows and not a count:

  new_in_v15      served by v15, called no-clock by the published vector
  gone_in_v15     called with-clock by the published vector, not served by v15, split by
                  WHY: held at LFPG / a clock with no usable `obs_taxi` / out of the
                  per-month class altogether

and it re-states the material property the desk's upload check asserts: of the rows v15
moves, how many did `exp-047-queue` move (must be 0), and how many of queue's 63,862 rows
survive bit for bit.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/classdiff.py
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
ROOT = EXP.parent.parent

import serve as S                                                     # noqa: E402

KEY = "MVT_ID_mvt"
TGT = S.D.TARGET
CLASSDIR = ROOT / "artifacts/exp-046/class"
V12 = ROOT / "submissions/candidates/exp-035-k2climb-v11member.parquet"
V13 = ROOT / "submissions/candidates/exp-046-surface.parquet"
V14 = ROOT / "submissions/candidates/exp-047-queue.parquet"
V15 = ROOT / "submissions/candidates/exp-046-surface-v2.parquet"
OUT = EXP / "analysis/class-diff-v15.json"


def read_class(suffix: str) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(CLASSDIR.glob(f"2026-*{suffix}.parquet"))
             if suffix or "-" not in p.stem[len("2026-01"):]]
    if not parts:
        raise SystemExit(f"no class vector with suffix {suffix!r} under {CLASSDIR}")
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    pub = read_class("")
    v15c = read_class("-v15")
    p12, p13, p14, p15 = (pd.read_parquet(f).set_index(KEY)[TGT] for f in
                          (V12, V13, V14, V15))
    key = p12.index.to_numpy("float64")
    a12, a13, a14, a15 = (s.reindex(key).to_numpy("float64") for s in (p12, p13, p14, p15))

    pub = pub.set_index(KEY).reindex(key)
    v15c = v15c.set_index(KEY).reindex(key)
    dep, _ = S.ranking_block()
    air = dep.set_index(dep[KEY].astype("float64")).reindex(key).airport.to_numpy()

    moved15 = a15 != a14                      # what THIS upload changes
    moved13 = a13 != a12                      # the head's own class, measured
    movedq = a14 != a13                       # queue's no-clock update
    served15 = v15c.served_by_exp_046.to_numpy(bool)
    pubclock = pub.obs_has_clock.to_numpy(bool)

    # round 4 FALLBACK, finding 3: "leaves the class" is a statement about the CLASS
    # (`obs_has_clock`), and comparing the published class against the v15 SERVE mask
    # mixed in the 38 LIRF `noac` rows that carry a clock and are held in both vectors --
    # they never left the class, they were never served by either.  Compare class with
    # class; report the held rows as what they are.
    newclock = v15c.obs_has_clock.to_numpy(bool)
    noac = v15c.noac.to_numpy(bool)
    new_in_v15 = newclock & ~pubclock
    gone = pubclock & ~newclock
    noac_with_a_clock = pubclock & newclock & noac
    held = np.isin(air.astype(str), ["LFPG"])
    reverted = moved13 & ~served15 & ~movedq

    def census(m):
        return {"rows": int(m.sum()),
                "by_airport": {str(a): int(((air == a) & m).sum())
                               for a in sorted(set(air[m]))},
                "MVT_ID_mvt": sorted(float(x) for x in key[m])}

    rep = {
        "what": ("the published whole-file class vector against v15's per-month one, "
                 "with every disagreeing row named"),
        "published": str((CLASSDIR / "2026-01.parquet").relative_to(ROOT)),
        "v15": str((CLASSDIR / "2026-01-v15.parquet").relative_to(ROOT)),
        "rows": int(len(key)),
        "new_in_v15": census(new_in_v15),
        # two different questions, and mixing them is how the 1,660 was first misread:
        # who is in the CLASS, and whom the file MOVES.  LFPG is in v15's class and is
        # held at the base, so it leaves the class count alone and the move count short.
        "gone_from_the_class": {
            "total": int(gone.sum()),
            "out_of_the_per_month_class_and_reverted": census(gone & reverted),
            "out_of_the_per_month_class_and_not_reverted": census(gone & ~reverted),
        },
        "noac_with_a_clock_in_both_vectors": {
            "what": ("in the class in BOTH vectors and served by NEITHER: the exp-033 "
                     "hold outranks the clock.  They are not a class movement and §5.1 "
                     "already names them"),
            **census(noac_with_a_clock),
        },
        "published_with_clock_that_v15_does_not_move": {
            "total": int((pubclock & ~moved15).sum()),
            "held_at_LFPG": int((pubclock & ~moved15 & held).sum()),
            "noac_with_a_clock": int((pubclock & ~moved15 & noac).sum()),
            "other": int((pubclock & ~moved15 & ~held & ~noac).sum()),
        },
        "reverted_to_v12": census(reverted),
        "the_desks_assertions": {
            "v15_differs_from_v14_only_on_served_or_reverted_rows": bool(
                not np.any(moved15 & ~served15 & ~reverted)),
            "served_rows_equal_v12_plus_the_new_head": bool(
                np.any(moved15 & served15)),
            "queue_rows_v15_overwrites": int((movedq & moved15).sum()),
            "queue_rows_that_survive": int((movedq & ~moved15).sum()),
            "queue_rows_total": int(movedq.sum()),
            "reverted_rows_equal_v12": bool(np.array_equal(a15[reverted], a12[reverted])),
            "noac_rows_moved": int((v15c.noac.to_numpy(bool) & moved15).sum()),
        },
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("new_in_v15", "gone_from_the_class", "reverted_to_v12",
                     "noac_with_a_clock_in_both_vectors",
                     "published_with_clock_that_v15_does_not_move")},
                     indent=2, sort_keys=True))
    for name in ("new_in_v15", "reverted_to_v12"):
        print(name, rep[name]["rows"], rep[name]["by_airport"])
    for name in ("gone_from_the_class", "published_with_clock_that_v15_does_not_move",
                 "noac_with_a_clock_in_both_vectors"):
        print(name, {k: (v if isinstance(v, int) else
                         (v["rows"] if isinstance(v, dict) and "rows" in v else v))
                     for k, v in rep[name].items() if k != "what"})


if __name__ == "__main__":
    main()
