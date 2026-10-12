"""exp-047-queue: how many rows the no-clock route actually reaches, on a RANKING month.

The label-side table is measured on the one harvested week that has labels beside it,
and that week is EHAM at 96 % receipt.  The question the serving decision turns on is a
different one and needs no label at all: **on the ranking file, how many rows with no
clock of their own have a neighbour that does?**  The desk's reading says "at LFPG a
16 % receipt still samples the queue"; this is where that is checked.

Measured on the harvest's delivered ranking days (`exp-045-harvest/manifest.json`), per
airport and per covered day, so the number is the airport's reach and not the harvest's
progress.  No label is read: the ranking file has none for departures.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/census.py --rule <sha>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402


def covered_days(require_rule: str | None) -> set[tuple[str, str]]:
    rules = BK.B46.rule_by_day()
    out = set()
    for (root, day), sha in rules.items():
        if require_rule is not None and sha != require_rule:
            continue
        for a in BK.AIRPORTS:
            p = Path(root) / a / f"{day}.parquet"
            if p.exists():
                out.add((a, day))
    return out


#: the board head, desk 2026-09-08 14:50 UTC (tail rule): v11 with only its K2 member
#: replaced by exp-035 arm 3b's climbed member.  Board 262.7418.
HEAD_FILE = "submissions/candidates/exp-035-k2climb-v11member.parquet"
HEAD_SHA = "48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default=None)
    ap.add_argument("--out", default="census-ranking.json")
    ap.add_argument("--month", default="2026-01",
                    help="the ranking month this census reads; airport-days of other "
                         "months on disk are not counted")
    ap.add_argument("--no-outage", action="store_true",
                    help="price what the desk's 14:45 outage rule costs: keep the "
                         "degraded feeder days' clocks and read the reach again")
    a = ap.parse_args()

    have = covered_days(a.rule)
    if a.month:
        # `rule_by_day()` sees every month on disk, and since 2025-12 landed a census
        # of a 2026-01 ranking file was reporting 558 airport-days.  The census reads
        # ONE month; say which, and count only its days.
        have = {(x, d) for x, d in have if d.startswith(a.month)}
    days = sorted({d for _, d in have})
    apts = sorted({x for x, _ in have})
    print(f"{len(have)} airport-days: {apts} over {days[0]}..{days[-1]}")

    rk = D.load_ranking(columns=BK.FRAME_COLS)
    dep_day = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    keep = pd.Series(list(zip(rk["ADEP_mvt"].astype(str), dep_day))).isin(have).to_numpy()
    # arrivals of the same airport-days are the arrival side of the state block
    arr_day = rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    keep_arr = pd.Series(list(zip(rk["ADES_mvt"].astype(str), arr_day))).isin(have).to_numpy()
    sub = rk[(keep & (rk.PHASE_mvt == "DEP")) | (keep_arr & (rk.PHASE_mvt == "ARR"))]
    sub = sub.reset_index(drop=True)
    print(f"{len(sub):,} ranking rows on the covered days "
          f"({int((sub.PHASE_mvt == 'DEP').sum()):,} departures)")

    # ONE MAP IN BOTH LANES (desk, 2026-09-09), `base.prefix_frame`
    train = BS.prefix_frame()
    # the served BOARD HEAD, so the census sees the same `*_n_res_*` columns the
    # serving path would.  That is v12 from the desk's 14:50 UTC message (tail rule): v11
    # with only its K2 member replaced, sha `48dd572d…`, board 262.7418.
    got = D.sha256(ROOT / HEAD_FILE)
    if got != HEAD_SHA:
        raise SystemExit(f"the board head is not {HEAD_SHA[:12]}…: {got[:12]}…")
    head = pd.read_parquet(ROOT / HEAD_FILE)
    pred = pd.Series(head[D.TARGET].to_numpy("float64"),
                     index=head[D.ID].to_numpy("float64"))
    b, rep = BK.build(sub, train, require_rule=a.rule, sources=("F", "O", "H", "N", "geo"),
                      pred=pred, outage=not a.no_outage)
    dep = sub[sub.PHASE_mvt == "DEP"].reset_index(drop=True)
    b["apt"] = dep["ADEP_mvt"].astype(str).to_numpy()
    b["has_nm"] = dep["AOBT_3_flt"].notna().to_numpy()

    own = b["obs_has_clock"].fillna(False).to_numpy(bool)
    n_cell = b["No_n_n_cell"].fillna(0).to_numpy("float64")
    n_rwy = b["No_n_n_rwy_cell"].fillna(0).to_numpy("float64")
    n_w30 = b["No_n_n_rwy_30m"].fillna(0).to_numpy("float64")
    reached = ~own & (n_cell >= 1)

    def block(sel) -> dict:
        if sel.sum() == 0:
            return {}
        noclk = sel & ~own
        return {
            "rows": int(sel.sum()),
            "own_clock": int((sel & own).sum()),
            "own_clock_share": round(float((sel & own).mean() / sel.mean()), 4),
            "no_clock": int(noclk.sum()),
            "no_clock_reached_by_a_neighbour": int((noclk & (n_cell >= 1)).sum()),
            "no_clock_reached_share": (round(float((noclk & (n_cell >= 1)).sum()
                                                   / max(noclk.sum(), 1)), 4)),
            "no_clock_reached_3plus": int((noclk & (n_cell >= 3)).sum()),
            "median_neighbours_in_cell": float(np.median(n_cell[noclk])) if noclk.any() else None,
            "median_neighbours_in_runway_cell": float(np.median(n_rwy[noclk])) if noclk.any() else None,
            "median_neighbours_rwy_30m": float(np.median(n_w30[noclk])) if noclk.any() else None,
            "no_nm_rows": int((sel & ~b["has_nm"].to_numpy(bool)).sum()),
            "no_nm_reached": int((sel & ~b["has_nm"].to_numpy(bool) & reached).sum())}

    out = {"rule": a.rule, "airport_days": len(have), "days": days,
           "prediction": {"file": HEAD_FILE, "name": "v12 (D20)",
                          "sha256": D.sha256(ROOT / HEAD_FILE)},
           "airports_with_summaries": apts,
           "all": block(np.ones(len(b), bool)),
           "by_airport": {x: block((b["apt"] == x).to_numpy()) for x in
                          sorted(b["apt"].unique())},
           "build": {k: v for k, v in rep.items() if k != "receipt_rate"},
           "receipt_rate": rep.get("receipt_rate")}
    (EXP / "analysis" / a.out).write_text(json.dumps(out, indent=2, sort_keys=True,
                                                     default=str))
    print(json.dumps({x: {"rows": v.get("rows"), "own": v.get("own_clock_share"),
                          "no_clock_reached": v.get("no_clock_reached_share"),
                          "median_nb": v.get("median_neighbours_in_cell")}
                      for x, v in out["by_airport"].items()}, indent=1))
    print("wrote", EXP / "analysis" / a.out)


if __name__ == "__main__":
    main()
