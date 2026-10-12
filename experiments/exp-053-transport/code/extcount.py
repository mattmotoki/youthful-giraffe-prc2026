"""exp-053 iteration 1, step 1a: the writable rows each extension month carries per airport.
INPUT-ONLY, zero fit, no loss.

SPEC §8's support condition SC-U(k) needs, for each step, the airports its added months cover.
A month's writable rows are exactly the rows its fit carries: exp-046 `onemonth.nov_rows(...)`'s
`usable` subset (`arms.served_class`), built here with `with_neighbour=False` and no shipped
columns, which stops before any column the arm uses.  The totals must equal the registered
counts (exp-046 `analysis/seasonal-bag10.json` `month_rows`), or nothing is written.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-053-transport/code/extcount.py
"""
from __future__ import annotations

import time

import common as C

import onemonth as OM                                                 # noqa: E402

OUT = C.ANALYSIS / "extcount.json"
REGISTERED = {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984, "2025-02": 68122,
              "2025-08": 96537}


def main() -> None:
    C.NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    out = {"what": __doc__.strip().splitlines()[0], "months": {}}
    for mo, want in REGISTERED.items():
        fr = OM.nov_rows([], month=mo, with_neighbour=False)
        u = fr[fr.usable.to_numpy(bool)]
        if len(u) != want:
            raise SystemExit(f"{mo}: {len(u)} writable rows, registered {want}")
        by = u.airport.astype(str).value_counts().sort_index()
        out["months"][mo] = {"writable_rows": int(len(u)),
                             "by_airport": {a: int(n) for a, n in by.items()},
                             "ids_sha256": C.sha_arr(sorted(u.MVT_ID_mvt.to_numpy("float64")))}
        print(f"{mo}: {len(u)} writable rows = registered; "
              + ", ".join(f"{a} {n}" for a, n in by.items()), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, out)
    print(f"written {OUT.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
