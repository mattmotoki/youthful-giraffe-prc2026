"""exp-043-pilot: how far the point-ordering defect actually reaches, measured.

Round 4 blocked on `segments.py` sorting points with pandas' default, which is not
stable, and the repair was made.  Round 5 then blocked on the CLAIM made about that
repair: the pilot de-duplicates `(icao, t)` in `census.load_slices`, so by the time
`segments()` sees the points there are no ties left for an unstable sort to break,
and saying "every measurement in this lineage was a property of the machine" was
wrong.  This file measures the reach instead of arguing about it:

  ties          how many duplicate `(icao, t)` rows the week holds before the
                de-duplication, and how many of those groups disagree about
                `on_ground` -- the ones that could have split a ground run
  equivalence   `segments()` over the de-duplicated points under a stable sort, an
                unstable sort, and a shuffled input order, compared row for row
  harvest       `exp-045-harvest`'s `all_segments()` over the same points, so the
                claim that the two run the same rule is a measurement and not an
                assertion.  The harvest keeps both copies of a tied point and the
                pilot keeps the first, so this is the one place they can differ.

  .venv/bin/python experiments/exp-043-pilot/code/ordering.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
# APPENDED, not inserted: exp-045-harvest has a `census.py` of its own, and this
# file must read the pilot's.  `allsegments` imports `segments` and gets the
# pilot's too, which is the point -- the two share one rule module.
sys.path.append(str(ROOT / "experiments/exp-045-harvest/code"))
import census as C          # noqa: E402
import segments as S        # noqa: E402

WEEK = [f"2025-07-{d:02d}" for d in range(7, 14)]
CMP = ["obs_wheels_off", "t_airborne", "seg_start", "seg_points", "seg_span_s",
       "stand_stop_s", "obs_offblock_strict", "obs_offblock_loose",
       "obs_offblock_disp", "n_ground_points", "callsign_modal"]


def _raw(days: list[str]) -> pd.DataFrame:
    """The week's points as the day files hold them, before de-duplication."""
    return pd.concat([pd.read_parquet(C.ART / f"points_{d}.parquet", columns=C.COLS)
                      for d in sorted(days)], ignore_index=True)


def _frame(pts: pd.DataFrame, stable: bool) -> pd.DataFrame:
    S.STABLE = stable
    g = S.segments(pts)
    S.STABLE = True
    return g.set_index(S.seg_key(g))[CMP].sort_index()


def _same(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    if list(a.index) != list(b.index):
        return {"identical": False, "rows_a": len(a), "rows_b": len(b),
                "keys_only_in_a": len(set(a.index) - set(b.index)),
                "keys_only_in_b": len(set(b.index) - set(a.index))}
    diff = [c for c in CMP if not a[c].equals(b[c])]
    return {"identical": not diff, "rows": len(a), "columns_that_differ": diff}


def main() -> None:
    raw = _raw(WEEK)
    dup = raw.duplicated(["icao", "t"], keep=False)
    grp = raw[dup].groupby(["icao", "t"])["on_ground"].nunique()
    out = {
        "days": WEEK,
        "ties_before_deduplication": {
            "rows_in_a_tied_group": int(dup.sum()),
            "tied_groups": int(len(grp)),
            "groups_that_disagree_about_on_ground": int((grp > 1).sum()),
            "why_it_matters": ("only a group that DISAGREES can split a ground run "
                               "differently, and only before the de-duplication"),
        },
    }

    pts, _ = C.load_slices(WEEK)
    out["ties_after_deduplication"] = int(pts.duplicated(["icao", "t"]).sum())

    stable = _frame(pts, True)
    rng = np.random.default_rng(0)
    shuffled = pts.iloc[rng.permutation(len(pts))].reset_index(drop=True)
    out["equivalence"] = {
        "unstable_sort_same_input": _same(stable, _frame(pts, False)),
        "stable_sort_shuffled_input": _same(stable, _frame(shuffled, True)),
        "unstable_sort_shuffled_input": _same(stable, _frame(shuffled, False)),
        "reading": ("with no ties left, the sort key determines the order, so the "
                    "segment step is input-order independent and the stable sort is "
                    "a guard against a future input that does have ties -- not a "
                    "repair of a number that was wrong"),
    }

    try:
        from allsegments import all_segments      # noqa: E402
        h = all_segments(pts.assign(airport="EHAM"))
        h = h[h.kind.eq("departure")] if "kind" in h else h
        hk = h["icao"].astype(str) + "@" + h["obs_wheels_off"].astype(str)
        h = h.set_index(hk)
        both = stable.index.intersection(h.index)
        cols = [c for c in CMP if c in h.columns]
        diff = [c for c in cols if not stable.loc[both, c].equals(h.loc[both, c])]
        out["against_the_harvest"] = {
            "pilot_departure_segments": int(len(stable)),
            "harvest_departure_segments": int(len(h)),
            "shared": int(len(both)),
            "only_in_pilot": int(len(set(stable.index) - set(h.index))),
            "only_in_harvest": int(len(set(h.index) - set(stable.index))),
            "columns_compared": cols,
            "columns_that_differ_on_shared_rows": diff,
            "note": ("the pilot de-duplicates tied points and the harvest keeps both, "
                     "so this is the one place the two can disagree; run over the "
                     "pilot's own de-duplicated slice, which is what its numbers use"),
        }
    except Exception as e:                                   # pragma: no cover
        out["against_the_harvest"] = {"error": f"{type(e).__name__}: {e}"}

    (EXP / "analysis/ordering.json").write_text(json.dumps(out, indent=2, sort_keys=True,
                                                           default=str))
    print(json.dumps(out, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
