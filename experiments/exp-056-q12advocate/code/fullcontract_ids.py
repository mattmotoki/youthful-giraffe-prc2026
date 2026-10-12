"""exp-056-q12advocate: the full contract's month sets, pair counts and kept-id hashes (review round 1, finding 8).

RUNBOOK section 2 (Plan A, the seven-family analogue) holds a fourth month J out of the members so
that every upper layer can be refitted on out-of-sample member predictions. This job fixes J and
computes, label-free, what the full-contract controls need:

- J = the first month beyond I, moving away from V, that is not M (jul 10, jan 04, dec 08);
  members train on A8 = the eight months outside {V, S, I, J};
- the matched-training-size pair: exposed E8 = A8 - M + S against isolated A8, the larger side
  thinned to the smaller side's departure count by the smallest sha256("exp056-size|<id>")
  (Plan B's rule); counts and the sorted kept-id hash (float64 bytes, as the thin kernel);
- J's ordered departure-id hash;
- the time-origin sets are Plan B's (the same S, I and population rule; RUNBOOK 1.6).

Check: the same function reproduces Plan B's three kept-id hashes (`e56.size_keep`). Reads only
MVT_ID_mvt and PHASE_mvt of the raw months: no label.

    .venv/bin/python experiments/exp-056-q12advocate/code/fullcontract_ids.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

OUT = E.ANALYSIS / "fullcontract-ids.json"
_IDS: dict = {}


def dep_ids(month: str) -> np.ndarray:
    if month not in _IDS:
        t = pd.read_parquet(E.D.month_file(month), columns=["MVT_ID_mvt", "PHASE_mvt"])
        _IDS[month] = t.loc[t["PHASE_mvt"].to_numpy() == "DEP", "MVT_ID_mvt"].to_numpy("int64")
    return _IDS[month]


def kept_hash(months: list[str], n_keep: int) -> str:
    ids = np.concatenate([dep_ids(m) for m in months])
    hv = np.array([int(hashlib.sha256(f"{E.SIZE_SALT}|{int(i)}".encode()).hexdigest()[:15], 16)
                   for i in ids], dtype="int64")
    keep = np.sort(ids[np.argsort(hv, kind="stable")[:n_keep]])
    return hashlib.sha256(np.ascontiguousarray(keep.astype("float64")).tobytes()).hexdigest()


def j_month(r: str) -> str:
    V, S, I, M = E.ROT[r]
    months = list(E.D.MONTHS)
    step = 1 if months.index(I) > months.index(V) else -1
    i = months.index(I) + step
    while months[i] in (V, S, I, M):
        i += step
    return months[i]


def main() -> None:
    NO.refuse_existing([OUT], False)
    res = {"what": __doc__.strip().splitlines()[0], "code_sha256": E.code_shas(), "salt": E.SIZE_SALT,
           "plan_b_check": {}, "rotations": {}}
    for r in E.ORDER:
        sk = E.size_keep(r)
        months = list(E.D.FOLDS[sk["base_fold"]].train_months)
        res["plan_b_check"][r] = kept_hash(months, sk["n_keep"]) == sk["kept_ids_sorted_sha256"]
    if not all(res["plan_b_check"].values()):
        raise SystemExit(f"the rule does not reproduce Plan B's kept ids: {res['plan_b_check']}")
    for r in E.ORDER:
        V, S, I, M = E.ROT[r]
        J = j_month(r)
        A8 = [m for m in E.D.MONTHS if m not in (V, S, I, J)]
        E8 = sorted([m for m in A8 if m != M] + [S])
        n = {"A8": sum(len(dep_ids(m)) for m in A8), "E8": sum(len(dep_ids(m)) for m in E8)}
        big, small = ("E8", "A8") if n["E8"] > n["A8"] else ("A8", "E8")
        sets = {"A8": A8, "E8": E8}
        res["rotations"][r] = {
            "V": V, "S": S, "I": I, "M": M, "J": J, "A8": A8, "E8_exposed": E8,
            "departures": {k: int(v) for k, v in n.items()},
            "pair": {"thinned_side": big, "thinned_from": int(n[big]), "n_keep": int(n[small]),
                     "matched_to": small, "kept_ids_sorted_sha256": kept_hash(sets[big], n[small])},
            "J_departures": int(len(dep_ids(J))),
            "J_ordered_ids_sha256": hashlib.sha256(np.ascontiguousarray(dep_ids(J).astype("float64")).tobytes()).hexdigest()}
        x = res["rotations"][r]
        print(f"{r}: J {J}; A8 {n['A8']:,} vs E8 {n['E8']:,} departures; thin {big} to {n[small]:,}, kept "
              f"{x['pair']['kept_ids_sorted_sha256'][:16]}; J ids {x['J_ordered_ids_sha256'][:16]}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"Plan B kept ids reproduced: {res['plan_b_check']}; written {OUT.name}")


if __name__ == "__main__":
    main()
