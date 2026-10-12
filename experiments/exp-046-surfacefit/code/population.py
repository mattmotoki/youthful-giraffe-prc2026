"""Write the pinned fold population: what a fold block is allowed to read.

The desk, 2026-09-09, as a STANDING POLICY for both lanes:

    A fold block is pinned to the airport-day population it was priced on.  Pricing, the
    gate, the dry run and the serve all read the pinned set, and the cache-key guard
    compares the pinned key.  A delivery enters a fold only at a recorded re-price -- a
    new population file, a new `arms.json`, the gate re-read, and a ledger row.  The
    serve never races the harvest.

The occasion was the v16 dry run refusing to write a candidate: the harvest delivered
2025-06-30 for nine airports into the july block between the price and the serve, moving
its cache key from `4521a8fe…` to `6a27ed27…`.  `block.py` and the rule version
`d4c59868…` were both unchanged; only the data moved.

    write            snapshot today's population for the three fold months
    --from-key=KEY   pin a month to a population that reproduces an earlier key, by
                     dropping days delivered since; the key check IS the proof that the
                     reconstruction is exact, and the run fails if it does not reproduce

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/population.py
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/population.py \\
        --month=2025-07 --from-key=4521a8fefabfae37 --drop-day=2025-06-30
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402,F401
import arms as A                                                      # noqa: E402
import block as B                                                     # noqa: E402

OUT = EXP / "analysis/fold-population.json"


def snapshot(month: str, drop_days: tuple[str, ...] = ()) -> list[dict]:
    """Every airport-day the month would read today, minus `drop_days`.

    Recorded with the three things the cache key hashes over -- the file's size and its
    `summaries_rule_sha256`, beside the airport and day -- so the file says what it pins
    and a reader can check the key by hand.
    """
    # UNPINNED on purpose: a snapshot has to see the whole of today's disk to be able to
    # say which days it is dropping.  The triples come from `block.population_triples`,
    # the one definition of a population in this lane.
    with B.pinned(None):
        tri = B.population_triples(B.AIRPORTS, A.month_days(month))
    return [x for x in tri if x["day"] not in drop_days]


def main() -> None:
    args = {a.split("=", 1)[0]: a.split("=", 1)[1]
            for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    rule, _ = A.serving_rule()
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    months = dict(prev.get("months", {}))

    only = args.get("--month")
    drop = tuple(x for x in args.get("--drop-day", "").split(",") if x)
    want_key = args.get("--from-key")

    for m in A.FOLD_MONTH.values():
        if only and m != only:
            continue
        pop = snapshot(m, drop if m == only else ())
        rec = {"airport_days": pop, "airport_days_count": len(pop),
               "dropped_days": list(drop) if m == only else [],
               "rule_sha256": rule}
        months[m] = rec
        # the key under this population, through the same hash the serve and the cache
        # key use, so a pin cannot reproduce a key the block build would not
        key = B.population_key(pop, rule)
        rec["block_cache_key"] = key
        if only and want_key and m == only and key != want_key:
            raise SystemExit(
                f"{m}: the pinned population gives key {key}, not the {want_key} it was "
                "asked to reproduce -- the reconstruction is not exact, so nothing is "
                "written")
        print(f"{m}  {len(pop):4d} airport-days  key {key}"
              + (f"  (dropped {', '.join(drop)})" if (only == m and drop) else ""), flush=True)

    out = {
        "what": ("the airport-day population each FOLD month is pinned to: the set its "
                 "block was priced on. Pricing, the gates, the dry run and the serve all "
                 "read this set, so a delivery landing between the price and the serve "
                 "cannot change what a candidate was gated on"),
        "policy": ("desk ruling 2026-09-09, standing for both lanes: a delivery enters a "
                   "fold only at a recorded re-price -- new population file, new "
                   "arms.json, gate re-read, ledger row. The serve never races the "
                   "harvest."),
        "written_by": "experiments/exp-046-surfacefit/code/population.py",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "read_by": ("block._summary_path (every summary read and every cache key goes "
                    "through it), via arms.block_cache_key and arms.month_block"),
        "ranking_months_are_not_pinned": (
            "the 2026 blocks are the serve's target, not a priced fold table; they are "
            "rebuilt from whatever the harvest has delivered and their key is not gated. "
            "The serve records their population and diffs it against the previous serve's "
            "(analysis/ranking-population.json), so a delivery that changes a served file "
            "is stated rather than silent -- it is never a stop by itself"),
        "the_pin_is_a_WHITELIST": (
            "`airport_days` is the whole of what a pinned month may read. A day that is "
            "not named is OUT whether or not it existed when the pin was written, so a "
            "delivery landing after the pin is held out with no further action. "
            "`dropped_days` is a note of which days were deliberately excluded from the "
            "snapshot that built this file, not the mechanism -- the mechanism is the "
            "whitelist. Round 5 FALLBACK finding 4: 2025-08-01 landed at 00:28:51 UTC, "
            "65 seconds after this file was first written and six minutes before the v16 "
            "serve, and was held out by the whitelist without being named anywhere; it is "
            "named now."),
        "months": months,
    }
    # WHAT THIS RE-PRICE MOVED FROM.  A pin file that only says where a fold is now leaves
    # a reader to diff two git revisions to see that it moved at all; the previous keys and
    # counts are cheap and belong in the file, beside the new ones.
    if prev.get("months"):
        out["superseded"] = {
            "written_at_utc": prev.get("written_at_utc"),
            "months": {m: {"block_cache_key": v.get("block_cache_key"),
                           "airport_days_count": v.get("airport_days_count"),
                           "dropped_days": v.get("dropped_days")}
                       for m, v in prev["months"].items()},
            "days_added": {m: sorted(
                {f"{x['airport']}|{x['day']}" for x in months[m]["airport_days"]}
                - {f"{x['airport']}|{x['day']}"
                   for x in prev["months"].get(m, {}).get("airport_days", [])})[:200]
                for m in months},
            "days_gone": {m: sorted(
                {f"{x['airport']}|{x['day']}"
                 for x in prev["months"].get(m, {}).get("airport_days", [])}
                - {f"{x['airport']}|{x['day']}" for x in months[m]["airport_days"]})[:200]
                for m in months},
        }
        for m, v in out["superseded"]["months"].items():
            a = len(out["superseded"]["days_added"].get(m, []))
            g = len(out["superseded"]["days_gone"].get(m, []))
            print(f"{m}  was {v['block_cache_key']} on {v['airport_days_count']} "
                  f"airport-days; +{a} added, -{g} gone", flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
