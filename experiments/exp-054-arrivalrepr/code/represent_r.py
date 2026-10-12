"""exp-054-arrivalrepr: the review-repair re-run of the representation job, ONE fold per process
(SPEC amendment A1; Sol round 1, finding 1).

`represent.py` runs as registered, with one substitution made before it starts: its reader
`e54.arrival_outcomes` is replaced by `e54r.arrival_outcomes_arr`, which gives the same values from
a read of ARR rows only.  So this job never holds a departure row of any month, and so never a
departure label.  The held month's arrival outcomes are read once, after the freeze, only for the
accuracy report.  One fold per process, so no fold's job holds another fold's data or state.

Before the job starts, the fold's panel file is checked against its pin (the registered
`represent.py` did not check it; SPEC section 1, "Inputs").  Outputs go to --out-dir only, never
to the experiment's own `work/` and `analysis/` (write-once).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-054-arrivalrepr/code/represent_r.py july_loo \
        --out-dir experiments/exp-054-arrivalrepr/rerun
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402
import e54r as ER                                                     # noqa: E402
import represent as REP                                               # noqa: E402  (registered code, unchanged)

NO = E.NO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", choices=E.FOLDS)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir == E.EXP.resolve():
        raise SystemExit("the re-run writes to its own --out-dir, never to the experiment's outputs")
    f = args.fold
    paths = [out_dir / f"work/rep-{f}.parquet", out_dir / f"analysis/rep-{f}.json"]
    NO.refuse_existing(paths, False)
    print("guard: no existing artifact at " + ", ".join(str(p) for p in paths), flush=True)
    got = E.sha_file(E.panel_path(f))
    if got != E.PANEL_PINS[f][0]:
        raise SystemExit(f"{E.panel_path(f).name} is {got[:16]}, not the pinned {E.PANEL_PINS[f][0][:16]}")
    (out_dir / "work").mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
    E.arrival_outcomes = ER.arrival_outcomes_arr                      # the one substitution
    rec = REP.run_fold(f, out_dir, False, False)
    rec["code_sha256"] = {n: E.sha_file(E.HERE / n) for n in ("e54.py", "represent.py", "e54r.py", "represent_r.py")}
    rec["rerun"] = {"reader": "e54r.arrival_outcomes_arr (ARR rows only; no departure row in any frame)",
                    "one_fold_per_process": True, "panel_file_sha256": got}
    j = out_dir / f"analysis/rep-{f}.json"
    NO.refuse_existing([j], False)
    j.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {j} ({rec['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
