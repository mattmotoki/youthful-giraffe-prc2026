"""`(2A - B) / before` written into records that were fitted before it existed.

FALLBACK review round 1, blocking 1.  `explain.annotate_shares` is arithmetic on
numbers each record already carries -- the population's own `before_s2` and the
contract's `mse_gain_s2` and `ci95` -- so applying it here gives exactly what a re-run
would write, WITHOUT refitting.  That equivalence holds only where the code change of
this round left the feature matrix alone, which is the `F` arms: they read no `_sc`
column, so the receipt repair cannot touch them, and `--sources F` never builds a
matcher at all.  Every record that does read an `_sc` column was re-run instead; this
script refuses to touch one.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from imports import load  # noqa: E402

EX = load("exp-047-queue", "explain")
ANA = Path(__file__).resolve().parents[1] / "analysis"
ALLOWED = ("explain-july_loo.json", "explain-december.json", "explain-january.json",
           "explain-july_loo-placebo.json")


def main() -> None:
    for name in ALLOWED:
        p = ANA / name
        if not p.exists():
            print("missing", name)
            continue
        rec = json.loads(p.read_text())
        for arm, r in rec.get("arms", {}).items():
            # the invariant is the RECORD's source set, not the arm's name (round 2
            # non-blocking F): a record built from sources other than `F` may carry an
            # `_sc` column whatever its arm is called.
            src = (rec.get("block") or {}).get("sources")
            if src != ["F"]:
                raise SystemExit(f"{name}: sources {src} may carry _sc columns; "
                                 "backfill only annotates an F-only record. Re-run it.")
            if any(a.startswith(("Oob", "Hob")) for a in [arm]):
                raise SystemExit(f"{name}: {arm} reads _sc columns; re-run it")
            EX.annotate_shares(r)
        p.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
        got = {a: r.get("all", {}).get("gain_over_shared")
               for a, r in rec.get("arms", {}).items()}
        print(f"{name}: {got}")


if __name__ == "__main__":
    main()
