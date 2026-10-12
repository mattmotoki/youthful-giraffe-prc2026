"""The fold POPULATION pin (desk rule, 2026-09-09).

A fold block is pinned to the airport-day population it was priced on.  A harvest
delivery that lands between a pricing run and a serve otherwise moves the population
under a gate that was read on a different one: surfacefit's guard caught exactly that
when 2025-06-30 walked into its july fold block, and this lane's july had already gained
`2025-06-29` and `2025-06-30` at nine airports an hour after v14 was priced (RESULTS §7m).

One file per fold month, `analysis/pin/<fold>.json`:

    population_key   the airport-days, each with its summary file's SHA-256, and the
                     summaries rule -- this is the POPULATION and nothing else.  The
                     matcher sha is deliberately NOT in it (desk: a cache or a key that
                     mixes the two returns pre-rebuild rows when only the matching
                     function moved); it is recorded beside the key, not inside it.
    priced_on        the run that set the pin: its record, its matcher, its date.
    history          every earlier population this fold was priced on, so the record can
                     say which side of a delivery a number is from.

`check()` is fail-closed for a build that finds a pin, permissive for one that finds
none, and a RE-PRICE passes `admit=True` to move the pin forward on the record.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PIN_DIR = EXP / "analysis" / "pin"
MONTH_OF = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def population(days_present, summary_path) -> dict:
    """`{airport|day: sha}` for the airport-days a build actually loaded."""
    out = {}
    for a, d in sorted(map(tuple, days_present or [])):
        p = summary_path(a, d)
        out[f"{a}|{d}"] = _sha(p)[:16] if p is not None else None
    return out


def key_of(pop: dict, rule: str | None) -> str:
    """The population key: the airport-days, their bytes, and the summaries rule."""
    payload = json.dumps({"airport_days": pop, "summaries_rule": rule}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def path_of(fold: str) -> Path:
    return PIN_DIR / f"{fold}.json"


def load(fold: str) -> dict | None:
    p = path_of(fold)
    return json.loads(p.read_text()) if p.exists() else None


def write(fold: str, pop: dict, rule: str | None, matcher_sha: str,
          priced_by: str, note: str = "") -> dict:
    """Set the pin for `fold`, keeping what it replaces in `history`."""
    PIN_DIR.mkdir(parents=True, exist_ok=True)
    prev = load(fold)
    rec = {
        "fold": fold, "month": MONTH_OF.get(fold),
        "airport_days": len(pop), "population_key": key_of(pop, rule),
        "summaries_rule": rule, "population": pop,
        "priced_on": {"record": priced_by, "matcher_sha256": matcher_sha,
                      "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "note": note},
        "history": (prev or {}).get("history", []),
    }
    if prev:
        rec["history"] = rec["history"] + [{
            "population_key": prev.get("population_key"),
            "airport_days": prev.get("airport_days"),
            "priced_on": prev.get("priced_on"),
            "replaced_utc": rec["priced_on"]["utc"]}]
    path_of(fold).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def check(fold: str, pop: dict, rule: str | None, admit: bool = False) -> dict:
    """Compare a build's population with the pin.  Fail closed unless `admit`."""
    pin = load(fold)
    got = key_of(pop, rule)
    if pin is None:
        return {"pinned": False, "population_key": got, "airport_days": len(pop)}
    same = pin.get("population_key") == got
    rec = {"pinned": True, "population_key": got, "pin_key": pin.get("population_key"),
           "matches_pin": bool(same), "airport_days": len(pop),
           "pin_airport_days": pin.get("airport_days")}
    if not same:
        have, want = set(pop), set(pin.get("population", {}))
        rec["added"] = sorted(have - want)
        rec["removed"] = sorted(want - have)
        rec["changed_bytes"] = sorted(k for k in (have & want)
                                      if pop[k] != pin["population"][k])
        if not admit:
            raise SystemExit(
                f"{fold}: the block's population is not the pinned one "
                f"({len(pop)} airport-days against {pin.get('airport_days')}; "
                f"+{len(rec['added'])} / -{len(rec['removed'])} / "
                f"{len(rec['changed_bytes'])} changed). A delivery enters a fold only at "
                "a RECORDED re-price: run with admit=True and write the new pin.")
    return rec
