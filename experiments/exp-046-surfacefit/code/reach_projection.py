"""exp-046-surfacefit: the reach of `analysis/reachwhy.json` projected onto the file.

`reachwhy.py` measures a match share per airport on the days whose summaries exist.
This turns those shares into rows of `ranking.parquet`, under the strict rule and under
the desk's wide rule side by side, so the programme's ceiling is a number and not an
impression.  An airport with no summary day yet contributes 0, and is named.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/reach_projection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

CARRIES = 0.80          # the share above which an airport is called "carries"


def main(month: str = "2026-01") -> None:
    rep = json.loads((EXP / "analysis/reachwhy.json").read_text())["months"][month]
    rk = D.load_ranking()
    rows = rk[rk.PHASE_mvt == "DEP"].ADEP_mvt.value_counts().to_dict()
    total = int(sum(rows.values()))

    out = {"basis": f"{month} summaries only, block match share per covered day",
           "covered_days_by_airport": rep.get("days_with_summaries_by_airport", {}),
           "rules_seen": rep.get("rules_seen"),
           "ranking_rows_by_airport": {k: int(v) for k, v in
                                       sorted(rows.items(), key=lambda kv: -kv[1])},
           "ranking_rows": total, "by_rule": {}}
    for rule, key in (("strict", "matched"), ("wide", "matched_wide")):
        proj, carry, unknown = 0.0, [], []
        for a, n in rows.items():
            v = rep["by_airport"].get(a, {})
            if key not in v:
                unknown.append(a)
                continue
            proj += n * v[key]
            if v[key] >= CARRIES:
                carry.append(a)
        out["by_rule"][rule] = {
            "projected_reachable_rows": proj,
            "projected_reach_share": proj / total,
            "airports_that_carry": sorted(carry),
            "carry_rows": int(sum(rows[a] for a in carry)),
            "carry_share_of_file": sum(rows[a] for a in carry) / total,
            "airports_with_no_summary_day": sorted(unknown),
        }
    (EXP / "analysis/reach-projection.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float))
    print(json.dumps(out, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
