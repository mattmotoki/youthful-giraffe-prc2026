"""exp-056-q12advocate: absolute UTC-date loss sums of every arm and design (review round 1, finding 9).

From each iteration's committed row losses (`artifacts/exp-056/iter<n>-rows.parquet`, hash-checked
against iter<n>.json), for every rotation and UTC date of V: the number of departures, the number
in the population P, and the sum of squared errors of every loss column (base; zero, true, const,
gmean, donor, shuffle, literal_sham, each of the ten permuted shams and their mean, in every
design; iteration 2 also iteration 1's true). Any contrast of the records can be recomputed from
these sums; as a check, the stored per-date sums of every priced contrast are recomputed from
them. No label is read (the row file holds losses only).

    .venv/bin/python experiments/exp-056-q12advocate/code/datesums.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def one(n_it: int) -> None:
    out = E.ANALYSIS / f"datesums-iter{n_it}.json"
    NO.refuse_existing([out], False)
    rec = E.load_json(E.ANALYSIS / f"iter{n_it}.json")
    rows_p = E.ROOT / rec["artifacts"]["rows"]["path"]
    if E.sha_file(rows_p) != rec["artifacts"]["rows"]["sha256"]:
        raise SystemExit(f"{rows_p.name} is not the recorded file")
    df = pd.read_parquet(rows_p)
    cols = [c for c in df.columns if c.startswith("loss:")]
    res = {"what": __doc__.strip().splitlines()[0], "iteration": n_it, "rows_file": rec["artifacts"]["rows"],
           "code_sha256": E.code_shas(), "units": "seconds squared, summed over the departures of the date",
           "rotations": {}}
    for r in E.ORDER:
        d = df[df["rotation"] == r]
        dates = E.load_json(E.ANALYSIS / f"inputs-{r}.json")["months_detail"]["V"]["utc_dates"]
        k = len(dates)
        di = d["day_index"].to_numpy()
        pop = d["pop"].to_numpy(bool)
        res["rotations"][r] = {
            "V": E.ROT[r][0], "utc_dates": dates,
            "departures": np.bincount(di, minlength=k).tolist(),
            "population_rows": np.bincount(di[pop], minlength=k).tolist(),
            "sse": {c[5:]: [float(x) for x in np.bincount(di, weights=d[c].to_numpy("float64"), minlength=k)]
                    for c in cols}}
    # check: each priced contrast's per-date sums (gain of B over A) equal A's sums minus B's
    worst, n = 0.0, 0
    for name, x in rec["contrasts"].items():
        for sub in ([x] if "A" in x else [v for v in x.values() if isinstance(v, dict) and "A" in v]):
            for r, f in sub["folds"].items():
                s = res["rotations"][r]["sse"]
                if sub["A"] not in s or sub["B"] not in s:
                    continue
                got = np.array(s[sub["A"]]) - np.array(s[sub["B"]])
                scale = max(1.0, float(np.max(np.abs(s[sub["A"]]))))
                worst = max(worst, float(np.max(np.abs(got - np.array(f["date_sums_s2"])))) / scale)
                n += 1
    res["check"] = {"contrast_folds_recomputed": n, "max_rel_diff": worst}
    if n == 0 or worst > 1e-9:
        raise SystemExit(f"the date sums do not reproduce the priced contrasts ({n}, {worst})")
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"iteration {n_it}: {len(cols)} loss columns x {sum(len(v['utc_dates']) for v in res['rotations'].values())} "
          f"dates; {n} contrast folds recomputed, max relative diff {worst:.1e}; written {out.name}")


if __name__ == "__main__":
    for it in (1, 2):
        one(it)
