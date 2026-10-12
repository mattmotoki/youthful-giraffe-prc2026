"""Frozen round counts (m6 m11) from one or more --select runs.

More than one file may report the same family: the eleven-month family has two
legitimate early-stop months (2025-11 from the `december` fold, 2025-12 from the
`january` fold), and CatBoost's counts differ by a factor of seven between
families, so where two readings exist the frozen count is their mean rather than
whichever one happened to run.  The count is still chosen once and used for
every fold and seed of the arm.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
fam: dict[str, list[int]] = {}
for name in sys.argv[1:]:
    for v in json.loads((HERE / f"select-{name}.json").read_text()).values():
        fam.setdefault(v["family"], []).append(int(v["refit_rounds"]))


def pick(k, default):
    v = fam.get(k)
    return int(round(sum(v) / len(v) / 25.0) * 25) if v else default


print(pick("m6", 300), pick("m11s", 0) or pick("m11", 400))
