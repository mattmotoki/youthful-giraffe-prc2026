"""exp-056-q12advocate: UTC-date loss sums on the secondary read's stratum (review round 1, finding 9).

price.py's secondary read (SPEC 3.9; not the gate) prices true against the gate controls on the
P rows whose stand key has n_A(stand) < 500 and drops those contrasts' per-date sums. This job
saves them: for every rotation and UTC date of V, the stratum's departures and the SSE of every
loss column on it, from the committed row losses (hash-checked). The stratum is rebuilt from the
frozen inputs (no label). Check: each stored secondary fold point (mean gain per stratum row)
is recomputed from these sums.

    .venv/bin/python experiments/exp-056-q12advocate/code/datesums_sparse.py
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
    out = E.ANALYSIS / f"datesums-sparse-iter{n_it}.json"
    NO.refuse_existing([out], False)
    rec = E.load_json(E.ANALYSIS / f"iter{n_it}.json")
    rows_p = E.ROOT / rec["artifacts"]["rows"]["path"]
    if E.sha_file(rows_p) != rec["artifacts"]["rows"]["sha256"]:
        raise SystemExit(f"{rows_p.name} is not the recorded file")
    df = pd.read_parquet(rows_p)
    cols = [c for c in df.columns if c.startswith("loss:")]
    sec = rec["secondary_sparse_stand"]
    res = {"what": __doc__.strip().splitlines()[0], "iteration": n_it, "rows_file": rec["artifacts"]["rows"],
           "stratum": sec["stratum"], "code_sha256": E.code_shas(), "rotations": {}}
    worst, n = 0.0, 0
    for r in E.ORDER:
        inp = E.load_json(E.ANALYSIS / f"inputs-{r}.json")
        z = np.load(E.ROOT / inp["npz"])
        d = df[df["rotation"] == r]
        if not np.array_equal(d["MVT_ID_mvt"].to_numpy("int64"), z["ids_V"]):
            raise SystemExit(f"{r}: the row file is not in the frozen id order")
        cs = z["code_V_stand"]
        nA = np.where(cs >= 0, z["nA_stand"][np.maximum(cs, 0)], -1)
        m = z["pop_V"] & (nA >= 0) & (nA < E.N_A_BINS[1])
        if int(m.sum()) != sec["rows"][r]:
            raise SystemExit(f"{r}: the stratum is not the priced one")
        dates = inp["months_detail"]["V"]["utc_dates"]
        k = len(dates)
        di = d["day_index"].to_numpy()[m]
        res["rotations"][r] = {"V": E.ROT[r][0], "utc_dates": dates, "stratum_rows": int(m.sum()),
                               "departures": np.bincount(di, minlength=k).tolist(),
                               "sse": {c[5:]: [float(x) for x in np.bincount(di, weights=d[c].to_numpy("float64")[m],
                                                                             minlength=k)] for c in cols}}
        for name, x in sec.items():
            if not (isinstance(x, dict) and "A" in x):
                continue
            s = res["rotations"][r]["sse"]
            got = (np.sum(s[x["A"]]) - np.sum(s[x["B"]])) / m.sum()
            want = x["folds"][r]["mse_gain_s2_per_row"]
            worst = max(worst, abs(got - want) / max(1.0, abs(want)))
            n += 1
    res["check"] = {"secondary_fold_points_recomputed": n, "max_rel_diff": worst}
    if n == 0 or worst > 1e-9:
        raise SystemExit(f"the stratum sums do not reproduce the secondary fold points ({n}, {worst})")
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"iteration {n_it}: stratum rows {[res['rotations'][r]['stratum_rows'] for r in E.ORDER]}; {n} secondary "
          f"fold points recomputed, max relative diff {worst:.1e}; written {out.name}")


if __name__ == "__main__":
    for it in (1, 2):
        one(it)
