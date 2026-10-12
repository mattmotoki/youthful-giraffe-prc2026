"""§7bj addendum: the CPU member vectors for 2025-11, at their real budgets.

The desk: run T-tailv2 and D1-winter to completion on `november_loo` and commit
their OOF beside F1-fresh; then L2-hub and C2-cat with the `m11` round-budget
line injected at runtime, each capped at 20 minutes of CPU.  No GPU, no stacker,
no head, no shared-code edit.  Goal: the 2025-11 base is one token away when
Matt answers recorded rule.

Each producer runs at the configuration the record names for that member, not
the smoke's reduced one:

    T-tailv2    exp-009-noac2/deliver.py at OOF v3, one fold (no round argument)
    D1-winter   winter_arm.py --seeds 7,107,207 --ds-columns deice  (THREE seeds,
                the published D1-winter-deice-v1 configuration; the §7bj smoke
                ran one seed and is not this)
    L2-hub      run.py --kind lgb --name L2-hub8725 --huber-delta 1200
                --rounds-m6 3225 --rounds-m11 8725
    C2-cat      run.py --kind cat --name C2-d8p --depth 8
                --rounds-m6 3625 --rounds-m11 5325 --cpu

`frames.FAMILY["november_loo"] = "m11"` is injected in-process for the two
`exp-008-diversity` producers.  `frames.py` is not edited: eleven training months
is the same family as july_loo, december and january, and the entry is set at
runtime exactly as §7bj established.

Every vector is normalised to the COMMON.md contract -- `MVT_ID_mvt, fold,
y_true, y_pred` -- which is what the Kaggle kernel wrote for F1-fresh and what
`prc.blend.stack_members` reads.  The two producers without `--outdir` write
beside their own experiment under a `_` name and are moved here and deleted
there; `exp-008-diversity`'s appends to `guard_bounds.json` and `climb.log` are
reverted with `git checkout` at the end.

    OMP_NUM_THREADS=2 PRC_THREADS=2 .venv/bin/python work/member_folds.py [member ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import lane                                                  # noqa: E402
lane.install()

OOF = EXP / "oof"
SUF = "" if lane.MONTH == "2025-11" else f"-{lane.MONTH}"   # §7bk keeps its names
OUT = EXP / f"analysis/member-folds{SUF}.json"
LOGS = EXP / "logs"
FOLD = lane.FOLD
PY = str(ROOT / ".venv/bin/python")
BLOCK = ROOT / f"artifacts/exp-047/block/{lane.MONTH}.parquet"
CONTRACT = ["MVT_ID_mvt", "fold", "y_true", "y_pred"]
CAP_S = 20 * 60
#: shared files the exp-008 producer appends to; reverted after the run.
TOUCHED = ["experiments/exp-008-diversity/guard_bounds.json",
           "experiments/exp-008-diversity/climb.log",
           # a published result the harness rewrote: --outdir moves only the OOF
           # parquet, so result-<name>.json landed in exp-008's lane and its
           # top-level `argv` was overwritten with OURS (desk, 2026-09-10)
           "experiments/exp-008-diversity/result-C2-d8p.json"]

E8 = "experiments/exp-008-diversity/run.py"
MEMBERS = {
    "T-tailv2": {
        "kind": "inject-deliver", "cap_s": None,
        "produced": ROOT / "experiments/exp-009-noac2/oof" / f"_q-{FOLD}.parquet",
        "config": "deliver.py, OOF v3 head, one fold",
    },
    "D1-winter": {
        # §7br: `winter_arm.py` has no --outdir and wrote BOTH its OOF parquet and
        # `result-_q-D1-winter.json` into exp-012's lane.  The parquet was moved
        # out and deleted; the result json was left behind (untracked, so no
        # published record was damaged -- but it is the same out-of-lane write the
        # desk caught in exp-008).  `HERE` is now repointed the same way.
        "kind": "inject-here", "cap_s": None,
        "cmd": [ROOT / "experiments/exp-012-blend2/winter_arm.py",
                "--name", "_q-D1-winter", "--folds", FOLD,
                "--seeds", "7,107,207", "--ds-columns", "deice"],
        "cwd": "experiments/exp-012-blend2",
        "here": HERE / "d1-results",
        "produced": HERE / "d1-results/oof" / f"_q-D1-winter-{FOLD}.parquet",
        "config": "winter_arm.py --seeds 7,107,207 --ds-columns deice (3 seeds)",
    },
    "L2-hub": {
        "kind": "inject-e8", "cap_s": CAP_S,
        "cmd": [ROOT / E8, "--kind", "lgb", "--name", "L2-hub8725", "--variant", "plain",
                "--huber-delta", "1200", "--folds", FOLD,
                "--rounds-m6", "3225", "--rounds-m11", "8725", "--outdir", str(OOF)],
        "cwd": "experiments/exp-008-diversity",
        "produced": OOF / f"L2-hub8725-{FOLD}.parquet",
        "config": "run.py lgb huber-1200 rounds-m11 8725",
    },
    "C2-cat": {
        "kind": "inject-e8", "cap_s": CAP_S,
        "cmd": [ROOT / E8, "--kind", "cat", "--name", "C2-d8p", "--depth", "8",
                "--folds", FOLD, "--rounds-m6", "3625", "--rounds-m11", "5325",
                "--cpu", "--outdir", str(OOF)],
        "cwd": "experiments/exp-008-diversity",
        "produced": OOF / f"C2-d8p-{FOLD}.parquet",
        "config": "run.py cat depth 8 rounds-m11 5325 --cpu",
    },
}


#: Installed in every SUBPROCESS.  `lane.install()` in this orchestrator only
#: reaches this process; each producer is spawned, so a leave-one-out fold that
#: `prc/data.py` does not ship has to be added again on the other side.  Empty for
#: November, whose fold `prc.data` already carries -- §7bk's commands are byte
#: identical to what it ran.
INJECT = ("" if not lane.ADDED else
          f"import sys; sys.path.insert(0, {str(HERE)!r}); "
          "import lane; lane.install(); ")


def command(name: str, spec: dict) -> list[str]:
    if spec["kind"] == "inject-deliver":
        # exp-009-noac2/threads.py hardcodes THREADS = 6 and deliver.py passes it
        # straight to LightGBM as n_jobs, so the assignment's two-thread rule needs
        # the module global set after import.  `deterministic` and the row-wise
        # histogram are pinned in prc.tail_model, so this changes the thread count
        # and not the fit.
        return [PY, "-c", INJECT +
                "import sys; sys.path.insert(0, 'experiments/exp-009-noac2'); "
                f"import deliver; deliver.THREADS = 2; deliver.FOLDS = ('{FOLD}',); "
                f"deliver.main(out_name='_q-{FOLD}.parquet')"]
    argv = [str(x) for x in spec["cmd"]]
    if spec["kind"] == "inject-e8":
        return [PY, "-c", INJECT +
                "import sys, runpy; "
                "sys.path.insert(0, 'experiments/exp-008-diversity'); "
                f"import frames; frames.FAMILY[{FOLD!r}] = 'm11'; "
                f"sys.argv = {argv!r}; "
                "import importlib.util as _i; "
                "_s=_i.spec_from_file_location('e8','experiments/exp-008-diversity/run.py'); "
                "_m=_i.module_from_spec(_s); _s.loader.exec_module(_m); "
                "_m.HERE=__import__('pathlib').Path('/path/to/competitions/prc-data-challenge-2026/experiments/exp-047-queue/work/e8-results'); "
                "_m.HERE.mkdir(parents=True, exist_ok=True); _m.main()"]
    if spec["kind"] == "inject-here":
        # same repointing as inject-e8: the module's ROOT is derived from HERE at
        # import time, so exec first and move HERE afterwards.
        return [PY, "-c", INJECT +
                f"import sys; sys.argv = {argv!r}; "
                "import importlib.util as _i; "
                f"_s=_i.spec_from_file_location('_prod', {argv[0]!r}); "
                "_m=_i.module_from_spec(_s); _s.loader.exec_module(_m); "
                f"_m.HERE=__import__('pathlib').Path({str(spec['here'])!r}); "
                "_m.HERE.mkdir(parents=True, exist_ok=True); _m.main()"]
    if INJECT:                      # a plain producer still needs the fold
        return [PY, "-c", INJECT +
                f"import sys, runpy; sys.argv = {argv!r}; "
                f"runpy.run_path({argv[0]!r}, run_name='__main__')"]
    return [PY] + argv


def project(log: Path, wall_s: float, spec: dict) -> str:
    """From a capped run's own progress lines, what the full fit would have cost."""
    txt = log.read_text(errors="replace") if log.exists() else ""
    done, total = None, None
    for line in txt.splitlines():
        s = line.strip()
        # lightgbm `[123]` iteration marks and catboost `123:` learn lines
        if s.startswith("[") and "]" in s[:8] and s[1:s.index("]")].isdigit():
            done = int(s[1:s.index("]")])
        elif ":" in s[:6] and s.split(":")[0].strip().isdigit():
            done = int(s.split(":")[0].strip())
    for flag in ("--rounds-m11",):
        c = [str(x) for x in spec.get("cmd", [])]
        if flag in c:
            total = int(c[c.index(flag) + 1])
    if done and total and done > 0:
        return (f"reached iteration {done:,} of {total:,} in {wall_s/60:.1f} min "
                f"-> projected {wall_s/60 * total/done:.0f} min for the full budget")
    return f"no iteration marker in {wall_s/60:.1f} min of output; projection unavailable"


def check(path: Path, member: str) -> dict:
    sys.path.insert(0, str(ROOT))
    import prc.blend as B
    d = pd.read_parquet(path)
    blk = pd.read_parquet(BLOCK, columns=["MVT_ID_mvt"])
    ids_new = set(d["MVT_ID_mvt"].astype("float64"))
    ids_blk = set(blk["MVT_ID_mvt"].astype("float64"))
    try:
        frame, _ = B.stack_members({member: str(path)}, ROOT, fold=FOLD)
        lost = int(len(d) - len(frame))
        ok = True
    except Exception as e:                                   # noqa: BLE001
        frame, lost, ok = None, None, f"{type(e).__name__}: {e}"
    r = d["y_true"] - d["y_pred"]
    return {
        "rows": int(len(d)),
        "columns_are_the_contract": list(d.columns) == CONTRACT,
        "dtypes": [str(t) for t in d.dtypes],
        "ids_identical_to_the_block": ids_new == ids_blk,
        "in_vector_not_in_block": len(ids_new - ids_blk),
        "in_block_not_in_vector": len(ids_blk - ids_new),
        "duplicate_ids": int(len(d) - d["MVT_ID_mvt"].nunique()),
        "y_pred_nulls": int(d["y_pred"].isna().sum()),
        "stack_members_ok": ok,
        "rows_lost_to_the_join": lost,
        "rmse": round(float((r ** 2).mean() ** 0.5), 4) if r.notna().any() else None,
        "n_scored": int(r.notna().sum()),
        "bytes": int(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    OOF.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "2", "PRC_THREADS": "2", "PYTHONPATH": str(ROOT)})
    rec = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, spec in MEMBERS.items():
        if only and name not in only:
            continue
        log = LOGS / f"fold-{name}{SUF}.log"
        cmd = command(name, spec)
        print(f"\n=== {name}: {spec['config']}", flush=True)
        t0 = time.time()
        timed_out = False
        with open(log, "w") as fh:
            p = subprocess.Popen(cmd, cwd=str(ROOT / spec.get("cwd", ".")), env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
            try:
                p.wait(timeout=spec["cap_s"])
            except subprocess.TimeoutExpired:
                timed_out = True
                p.kill()
                p.wait()
        dt = time.time() - t0
        entry = {"config": spec["config"], "wall_s": round(dt, 1),
                 "wall_min": round(dt / 60, 1), "returncode": p.returncode,
                 "capped_at_s": spec["cap_s"], "hit_the_cap": timed_out,
                 "log": str(log.relative_to(ROOT))}
        if timed_out:
            entry["projection"] = project(log, dt, spec)
            entry["kept"] = False
            print(f"    CAP HIT at {dt/60:.1f} min -- {entry['projection']}", flush=True)
        elif p.returncode != 0:
            tail = log.read_text(errors="replace").strip().splitlines()[-3:]
            entry["kept"] = False
            entry["why"] = " | ".join(t.strip()[:200] for t in tail)
            print(f"    rc={p.returncode} -- {entry['why']}", flush=True)
        else:
            src = spec["produced"]
            if not src.exists():
                entry["kept"] = False
                entry["why"] = f"producer returned 0 but {src} is absent"
            else:
                d = pd.read_parquet(src)
                d = d[[c for c in CONTRACT if c in d.columns]].copy()
                d["fold"] = FOLD
                d = d[CONTRACT]
                dst = OOF / f"{name}--{FOLD}.parquet"
                d.to_parquet(dst, index=False)
                if src != dst:
                    src.unlink()
                entry["kept"] = True
                entry["path"] = str(dst.relative_to(ROOT))
                entry["checks"] = check(dst, name)
                c = entry["checks"]
                print(f"    OK {dt/60:.1f} min  rows {c['rows']:,}  "
                      f"ids={c['ids_identical_to_the_block']}  "
                      f"stack_members={c['stack_members_ok']} lost={c['rows_lost_to_the_join']}  "
                      f"rmse {c['rmse']}  sha {c['sha256'][:16]}", flush=True)
        rec[name] = entry
        OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str))
    # the exp-008 producer appends a fold entry and a run line to shared files.
    subprocess.run(["git", "checkout", "--"] + TOUCHED, cwd=str(ROOT))
    print("\nreverted:", ", ".join(TOUCHED))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
