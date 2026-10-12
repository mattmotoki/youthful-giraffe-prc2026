"""exp-062 ownvalue-refine: the one pinned accounting controller (SPEC section 12).

EVERY process of this experiment is launched through this file.  It appends an immutable START
receipt to `analysis/receipts.jsonl` BEFORE it launches the child and a separate immutable END
receipt AFTER the child exits; both carry the same unique attempt id.  Neither is ever edited.

    .venv/bin/python experiments/exp-062-ownvalue-refine/code/vp_ctl.py \
        --kind repro --log logs/repro.log -- .venv/bin/python .../vp_repro.py

    vp_ctl.py --status          prints the charged use, the gate arithmetic and the launch count

Accounting (SPEC 12, recorded rule's round-2 answer):
  * The measured interval starts BEFORE the start receipt is written, so the controller's own
    receipt-writing overhead is inside the charge.  The end receipt adds the registered constants
    RECEIPT_OVERHEAD_WALL_S / _CPU_S as an upper bound on the write and exit that follow the last
    measurement, which cannot measure itself.
  * A paired attempt (start + end) is charged at its END values.  An attempt with a start and no
    end is charged at the start receipt's FULL wall timeout and its projected process CPU.
  * An end receipt whose attempt id has no start receipt forces a stop.
  * Process CPU is the controller's own `time.process_time()` delta plus the
    `getrusage(RUSAGE_CHILDREN)` user+sys delta, so a child's own subprocesses are charged too.
  * Two allocations: FIT (2,460 s wall / 4,800 s process cpu) and NON-FIT (1,500 s / 600 s).  A
    process is refused if its own allocation would be exceeded, or if
        charged_fit + remaining_registered_fits + max(charged_nonfit, NON-FIT allocation)
    would exceed an experiment cap (3,960 s wall / 5,400 s process cpu).
  * `analysis/launches.jsonl` is the separate append-only ledger of FIT ATTEMPTS, one line per
    registered fit the driver enters.  Every stopped, failed or guard-refused attempt is a line.
    The controller refuses a fit process whose registered fits would take the count past 18.
  * The wall timeout of every process lies inside the remaining allocation wall, and the child is
    stopped at STOP_FRACTION of it -- strictly before the timeout -- then killed at the timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ANALYSIS = EXP / "analysis"
RECEIPTS = ANALYSIS / "receipts.jsonl"
LAUNCHES = ANALYSIS / "launches.jsonl"
#: SPEC 2.4, amended: the append-only code-pin ledger, committed before the amended code runs
PINS = ANALYSIS / "code-pins.jsonl"
MANIFEST = ANALYSIS / "manifest.json"
FITS = ANALYSIS / "fits"

#: SPEC 12: the caps and the two allocations, in seconds
CAP_WALL_S = 3960.0
CAP_CPU_S = 5400.0
ALLOC_NONFIT_WALL_S = 1500.0
ALLOC_NONFIT_CPU_S = 600.0
ALLOC_FIT_WALL_S = CAP_WALL_S - ALLOC_NONFIT_WALL_S      # 2460.0
ALLOC_FIT_CPU_S = CAP_CPU_S - ALLOC_NONFIT_CPU_S         # 4800.0
MAX_FIT_LAUNCHES = 18
#: the upper bound on the end-receipt write and the controller exit that follow the measurement
RECEIPT_OVERHEAD_WALL_S = 0.5
RECEIPT_OVERHEAD_CPU_S = 1.5
STOP_FRACTION = 0.95
KILL_GRACE_S = 10.0
LOAD_MAX = 10.0

#: SPEC 3: the registered near-origin grid rows and the fitting-row model
GRID_N = {1: 2_536, 2: 5_068, 3: 7_601}
PANEL_ROWS = 172_783
STEP_EXTRA = {0: 0, 1: 99_709, 2: 290_037, 3: 386_574}
#: SPEC 12: exp-058's least-squares cost law, refitted on its 24 fit records
#: (experiments/exp-058-origincontrast/analysis/costlaw-r1.json), as exp-059 registered it
LAW_WALL = (47.5449478, 1.31849731e-4)
LAW_CPU = (89.5397667, 3.60042481e-4)
#: SPEC 4: the registered fit order.  Stage 1 is arm-major (k = 1, then 2, then 3) then member
#: seed, at L0 only.  Stage 2, conditional on June*, is member seed then level.
STAGE1 = [(f"G{k:02d}", s, 0) for k in (1, 2, 3) for s in (1, 2, 3)]


def stage2(k: int) -> list:
    return [(f"G{k:02d}", s, L) for s in (1, 2, 3) for L in (1, 2, 3)]


#: SPEC 12: the registered wall / process-CPU projection of every non-fit process kind, with its
#: basis.  A kind not named here is refused.
KIND_PROJECTION = {
    "manifest": (40.0, 40.0),      # exp-059's census read 4.34 GB in 3.8 s; plus interpreter start
    "plan": (70.0, 80.0),          # exp-059 plan stage 34.2 s wall / 37.2 s process cpu
    "plan2": (70.0, 80.0),         # the same stage for the conditional June* arm
    "repro": (220.0, 240.0),       # exp-059 repro-only 128.0 / 121.0, plus exp-055's subprocesses
    "value": (120.0, 140.0),       # exp-059's ladder read, charged honestly from process start
    #: the dry run does every non-draw step of the measurement (SPEC 11), so it is projected at the
    #: measurement's own cost less the draw loop, not at a small fraction of it
    "measure-standin": (110.0, 120.0),
    #: the measurement runs no 2,000-draw ten-member loop (SPEC 12.3 cites exp-057 for it), so it is
    #: the dry run plus three block lengths of matrix draws and 2,457 percentile reads
    "measure": (130.0, 140.0),
    "tables": (20.0, 25.0),        # a stdout printer over this experiment's own JSON records
    #: the required fresh Sol review through `codex exec`, read-only: almost all of its cost is
    #: network wall, and at most two rounds fit inside the non-fit wall allocation
    "review": (450.0, 30.0),
}
FIT_KINDS = ("fit-stage1", "fit-stage2")


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_lines(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def append_line(p: Path, obj: dict) -> None:
    """Append one immutable JSON line.  A receipt is never edited and never rewritten."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------ the charged use ------------

def charged(exclude: str | None = None) -> dict:
    """SPEC 12: sum every charged attempt.  A start without an end is charged at its full wall
    timeout and its projected process CPU; an end without a start forces a stop.

    `exclude` drops one attempt id.  A child process passes its OWN attempt id (the controller
    exports it as VP_ATTEMPT_ID), because its start receipt is already on disk with no end receipt
    yet, and charging it at its timeout while it is still running would double count it."""
    rows = read_lines(RECEIPTS)
    starts = {r["attempt_id"]: r for r in rows if r["receipt"] == "start"}
    ends = {}
    for r in rows:
        if r["receipt"] != "end":
            continue
        if r["attempt_id"] not in starts:
            raise SystemExit(f"receipts.jsonl: end receipt {r['attempt_id']} has no start receipt; stop")
        if r["attempt_id"] in ends:
            raise SystemExit(f"receipts.jsonl: two end receipts for {r['attempt_id']}; stop")
        ends[r["attempt_id"]] = r
    out = {"fit_wall": 0.0, "fit_cpu": 0.0, "nonfit_wall": 0.0, "nonfit_cpu": 0.0,
           "attempts": len(starts), "unpaired": [], "by_attempt": []}
    for aid, s in starts.items():
        if exclude is not None and aid == exclude:
            continue
        e = ends.get(aid)
        if e is None:
            w, c, how = s["wall_timeout_s"], s["projected_process_cpu_s"], "unpaired: charged at the timeout"
            out["unpaired"].append(aid)
        else:
            w, c, how = e["measured_wall_s"], e["measured_process_cpu_s"], "paired: charged at the end values"
        key = "fit" if s["fit_launch"] else "nonfit"
        out[f"{key}_wall"] += w
        out[f"{key}_cpu"] += c
        out["by_attempt"].append({"attempt_id": aid, "kind": s["kind"], "fit_launch": s["fit_launch"],
                                  "wall_s": round(w, 1), "process_cpu_s": round(c, 1), "how": how,
                                  "status": None if e is None else e["status"]})
    out["total_wall"] = out["fit_wall"] + out["nonfit_wall"]
    out["total_cpu"] = out["fit_cpu"] + out["nonfit_cpu"]
    return out


def fit_launches() -> int:
    """SPEC 12: every registered fit the driver entered, stopped, failed or guard-refused included."""
    return len([r for r in read_lines(LAUNCHES) if r.get("record") == "attempt"])


def done_fits() -> dict:
    if not FITS.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(FITS.glob("*.json"))}


def june_star_k() -> int | None:
    p = ANALYSIS / "probe.json"
    if not p.exists():
        return None
    js = json.loads(p.read_text()).get("june_star")
    return None if not js else int(js["k"])


def law(rows: int) -> tuple:
    return (LAW_WALL[0] + LAW_WALL[1] * rows, LAW_CPU[0] + LAW_CPU[1] * rows)


def fit_projection(arm: str, level: int, done: dict) -> tuple:
    """SPEC 12: the larger of exp-058's refitted law and this experiment's completed mean at the
    same level."""
    k = int(arm[1:])
    w, c = law(PANEL_ROWS + GRID_N[k] + STEP_EXTRA[level])
    same = [r for r in done.values() if r["level"] == level]
    if same:
        w = max(w, sum(r["seconds_fit_wall"] for r in same) / len(same))
        c = max(c, sum(r["seconds_fit_process_cpu"] for r in same) / len(same))
    return w, c


def remaining_fits(done: dict) -> tuple:
    """Every registered fit with no record yet.  Stage 2's grid row is June* when it is fixed and
    the worst case k = 3 (N = 7,601) until then."""
    k = june_star_k()
    plan = STAGE1 + stage2(k if k is not None else 3)
    w = c = 0.0
    n = 0
    for arm, s, L in plan:
        if f"{arm}-s{s}-L{L}" in done:
            continue
        dw, dc = fit_projection(arm, L, done)
        w, c, n = w + dw, c + dc, n + 1
    return w, c, n


def gate(kind: str, fits_here: list, done: dict) -> dict:
    """SPEC 12: the projection before every process.  Refuses on an allocation or a cap."""
    ch = charged()
    is_fit = kind in FIT_KINDS
    if is_fit:
        pw = sum(fit_projection(a, L, done)[0] for a, _, L in fits_here)
        pc = sum(fit_projection(a, L, done)[1] for a, _, L in fits_here)
        alloc_w, alloc_c = ALLOC_FIT_WALL_S, ALLOC_FIT_CPU_S
        used_w, used_c = ch["fit_wall"], ch["fit_cpu"]
    else:
        if kind not in KIND_PROJECTION:
            raise SystemExit(f"kind {kind!r} has no registered projection (SPEC 12)")
        pw, pc = KIND_PROJECTION[kind]
        alloc_w, alloc_c = ALLOC_NONFIT_WALL_S, ALLOC_NONFIT_CPU_S
        used_w, used_c = ch["nonfit_wall"], ch["nonfit_cpu"]
    rem_w, rem_c, rem_n = remaining_fits(done)
    #: the registered cap projection: charged fit use, every remaining registered fit, and the
    #: unspent non-fit allocation as a lump (so a non-fit process is inside it, never double counted)
    cap_w = ch["fit_wall"] + rem_w + max(ch["nonfit_wall"], ALLOC_NONFIT_WALL_S)
    cap_c = ch["fit_cpu"] + rem_c + max(ch["nonfit_cpu"], ALLOC_NONFIT_CPU_S)
    if not is_fit:
        #: a fit process's own fits are already inside `remaining`; a non-fit process's projection
        #: is inside the unspent non-fit lump, and this keeps the bound explicit either way
        cap_w = max(cap_w, ch["fit_wall"] + rem_w + ch["nonfit_wall"] + pw)
        cap_c = max(cap_c, ch["fit_cpu"] + rem_c + ch["nonfit_cpu"] + pc)
    reasons = []
    if used_w + pw > alloc_w:
        reasons.append(f"{kind}: {used_w:.0f} s wall charged to its allocation plus {pw:.0f} s "
                       f"projected exceeds the {alloc_w:.0f} s allocation")
    if used_c + pc > alloc_c:
        reasons.append(f"{kind}: {used_c:.0f} s process cpu charged to its allocation plus {pc:.0f} s "
                       f"projected exceeds the {alloc_c:.0f} s allocation")
    if cap_w > CAP_WALL_S:
        reasons.append(f"the cap projection {cap_w:.0f} s wall exceeds the {CAP_WALL_S:.0f} s cap")
    if cap_c > CAP_CPU_S:
        reasons.append(f"the cap projection {cap_c:.0f} s process cpu exceeds the {CAP_CPU_S:.0f} s cap")
    launches = fit_launches()
    if is_fit and launches + len(fits_here) > MAX_FIT_LAUNCHES:
        reasons.append(f"{launches} fit attempts recorded plus {len(fits_here)} planned exceeds the "
                       f"{MAX_FIT_LAUNCHES}-launch cap")
    remaining_alloc_wall = alloc_w - used_w
    return {"kind": kind, "fit_launch": is_fit, "charged": ch, "launches_before": launches,
            "projected_wall_s": pw, "projected_process_cpu_s": pc,
            "allocation_wall_s": alloc_w, "allocation_process_cpu_s": alloc_c,
            "remaining_registered_fits": rem_n, "remaining_fit_wall_s": rem_w,
            "remaining_fit_process_cpu_s": rem_c,
            "cap_projection_wall_s": cap_w, "cap_projection_process_cpu_s": cap_c,
            "remaining_allocation_wall_s": remaining_alloc_wall, "refusals": reasons}


# ------------------------------------------------------------------ the code guard -------------

def code_pins() -> dict:
    """The pins every code file of this experiment must carry: `analysis/manifest.json`, amended by
    `analysis/code-pins.jsonl`.

    The ledger is append-only and is COMMITTED BEFORE the amended code runs, so the assignment's
    requirement -- every new code file pinned before that code runs -- holds for a correction too.
    Each line carries the path, the pin it replaces, the new hash, the reason and the UTC time.  It
    never removes a pin and never rewrites a line.  exp-058's `analysis/codepins-r1.json` is the
    precedent for a committed code-pin amendment."""
    pins = json.loads(MANIFEST.read_text())["new_code"]["files"]
    amended = {}
    for row in read_lines(PINS):
        if row.get("record") != "code-pin":
            raise SystemExit(f"{PINS.name}: a line is not a code-pin record")
        rel = row["path"]
        if rel in pins and row.get("replaces") != pins[rel] and rel not in amended:
            raise SystemExit(f"{PINS.name}: the amendment of {rel} does not replace its manifest pin")
        if rel in amended and row.get("replaces") != amended[rel]["sha256"]:
            raise SystemExit(f"{PINS.name}: the amendment of {rel} does not replace its last pin")
        amended[rel] = {"sha256": row["sha256"], "reason": row["reason"], "utc": row["utc"]}
    return {"manifest": pins, "amended": amended,
            "effective": dict(sorted((pins | {k: v["sha256"] for k, v in amended.items()}).items()))}


def code_guard(kind: str) -> dict:
    """Every code file of this experiment must carry its effective pin before anything runs.  The
    manifest run itself is the one ordering exception, disclosed in SPEC 2.4: it produces the pins."""
    files = {str(p.relative_to(ROOT)): sha_file(p) for p in sorted(HERE.glob("vp_*.py"))}
    if not MANIFEST.exists():
        if kind != "manifest":
            raise SystemExit("analysis/manifest.json does not exist; only --kind manifest may run")
        return {"manifest_absent": True, "code_files": files}
    cp = code_pins()
    pins = cp["effective"]
    if set(pins) != set(files):
        raise SystemExit(f"the code files differ from their pins: "
                         f"{sorted(set(pins) ^ set(files))}")
    bad = {k: {"pin": pins[k], "now": v} for k, v in files.items() if pins[k] != v}
    if bad:
        raise SystemExit(f"a code file differs from its effective pin: {json.dumps(bad, indent=1)}")
    return {"manifest_absent": False, "code_files": files,
            "manifest_sha256": sha_file(MANIFEST),
            "amended_pins": cp["amended"],
            "code_pins_ledger_sha256": sha_file(PINS) if PINS.exists() else None}


# ------------------------------------------------------------------ main -----------------------

def hold_for_load(kind: str) -> tuple:
    """COMMON / the assignment: hold a new launch while the 1-minute load is above 10.  The hold
    runs BEFORE the measured interval opens, so idle waiting is not charged as compute; the held
    seconds are recorded in the start receipt."""
    t = time.monotonic()
    la = os.getloadavg()[0]
    while la > LOAD_MAX:
        print(f"{kind}: 1-minute load {la:.2f} > {LOAD_MAX}, holding 60 s", flush=True)
        time.sleep(60)
        la = os.getloadavg()[0]
    return la, round(time.monotonic() - t, 1)


def children_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--kind")
    ps.add_argument("--log", help="the child's log file, relative to the experiment directory")
    ps.add_argument("--timeout", type=float, default=None,
                    help="the wall timeout in seconds; the default is the registered projection "
                         "times 3, capped by the remaining allocation wall")
    ps.add_argument("--note", default="")
    ps.add_argument("--status", action="store_true")
    ps.add_argument("cmd", nargs="*", help="-- then the command")
    a = ps.parse_args()
    if a.status:
        print(json.dumps(gate("tables", [], done_fits()), indent=1, default=float))
        print(f"fit attempts recorded: {fit_launches()} of {MAX_FIT_LAUNCHES}")
        print(f"fit records: {len(done_fits())}")
        return
    if not a.kind or not a.cmd:
        raise SystemExit("--kind and a command are required")
    cmd = list(a.cmd)
    done = done_fits()
    if a.kind in FIT_KINDS:
        k = june_star_k()
        fits_here = [f for f in (STAGE1 if a.kind == "fit-stage1" else
                                 stage2(k if k is not None else 3))
                     if f"{f[0]}-s{f[1]}-L{f[2]}" not in done]
        if a.kind == "fit-stage2" and k is None:
            raise SystemExit("fit-stage2 needs a fixed June* in analysis/probe.json")
    else:
        fits_here = []
    g = gate(a.kind, fits_here, done)
    guard = code_guard(a.kind)
    aid = uuid.uuid4().hex
    log = (EXP / a.log) if a.log else None
    if log is not None and log.exists():
        raise SystemExit(f"{log} exists; logs are write-once")
    load, held = hold_for_load(a.kind)

    # --- the measured interval opens here, before the start receipt is written -----------------
    t0 = time.monotonic()
    cpu_self0, cpu_kid0 = time.process_time(), children_cpu()
    factor = 2.0 if a.kind in FIT_KINDS else 3.0
    default_to = min(factor * max(g["projected_wall_s"], 1.0),
                     max(g["remaining_allocation_wall_s"], 1.0))
    timeout = float(a.timeout) if a.timeout is not None else round(default_to, 1)
    refused = bool(g["refusals"])
    if refused:
        timeout = 0.0
    if timeout > g["remaining_allocation_wall_s"] and not refused:
        raise SystemExit(f"the wall timeout {timeout:.0f} s is outside the remaining "
                         f"{g['remaining_allocation_wall_s']:.0f} s of the {a.kind} allocation")
    start = {"receipt": "start", "attempt_id": aid, "kind": a.kind, "fit_launch": a.kind in FIT_KINDS,
             "command": cmd, "command_sha256": sha_bytes(("\x00".join(cmd)).encode()),
             "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "wall_timeout_s": timeout, "stop_at_s": round(timeout * STOP_FRACTION, 1),
             "projected_wall_s": g["projected_wall_s"],
             "projected_process_cpu_s": g["projected_process_cpu_s"],
             "fits_in_this_process": [f"{x[0]}-s{x[1]}-L{x[2]}" for x in fits_here],
             "launches_before": g["launches_before"],
             "charged_before": {k_: round(v, 1) for k_, v in g["charged"].items()
                                if isinstance(v, float)},
             "cap_projection_wall_s": round(g["cap_projection_wall_s"], 1),
             "cap_projection_process_cpu_s": round(g["cap_projection_process_cpu_s"], 1),
             "gate_refusals": g["refusals"], "load_1min": load, "held_seconds_before_t0": held,
             "log": None if log is None else str(log.relative_to(ROOT)),
             "code_guard": guard, "note": a.note,
             "omp_num_threads": "3"}
    append_line(RECEIPTS, start)
    status, sig, rc = "refused-by-gate", None, None
    if not refused:
        env = dict(os.environ, OMP_NUM_THREADS="3", MKL_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3",
                   VECLIB_MAXIMUM_THREADS="3", NUMEXPR_NUM_THREADS="3", PYTHONUNBUFFERED="1",
                   VP_ATTEMPT_ID=aid, VP_KIND=a.kind)
        fh = open(log, "w") if log is not None else None
        try:
            pr = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh or None,
                                  stderr=subprocess.STDOUT if fh else None)
            print(f"{a.kind} attempt {aid[:8]} pid {pr.pid} timeout {timeout:.0f}s "
                  f"stop at {timeout * STOP_FRACTION:.0f}s log {log}", flush=True)
            stop_at = t0 + timeout * STOP_FRACTION
            kill_at = min(t0 + timeout, stop_at + KILL_GRACE_S)
            while pr.poll() is None:
                now = time.monotonic()
                if now >= kill_at:
                    pr.kill()
                    sig, status = "SIGKILL", "killed-at-timeout"
                    break
                if now >= stop_at and sig is None:
                    pr.send_signal(signal.SIGTERM)
                    sig, status = "SIGTERM", "stopped-before-timeout"
                time.sleep(0.2)
            rc = pr.wait()
            if sig is None:
                status = "ok" if rc == 0 else "failed"
        finally:
            if fh is not None:
                fh.close()
    wall = time.monotonic() - t0
    cpu = (time.process_time() - cpu_self0) + (children_cpu() - cpu_kid0)
    end = {"receipt": "end", "attempt_id": aid, "kind": a.kind, "fit_launch": a.kind in FIT_KINDS,
           "status": status, "returncode": rc, "signal": sig,
           "refusal": g["refusals"] if refused else None,
           "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "measured_wall_s": round(wall + RECEIPT_OVERHEAD_WALL_S, 1),
           "measured_process_cpu_s": round(cpu + RECEIPT_OVERHEAD_CPU_S, 1),
           "measured_wall_before_overhead_s": round(wall, 1),
           "measured_process_cpu_before_overhead_s": round(cpu, 1),
           "receipt_overhead_wall_s": RECEIPT_OVERHEAD_WALL_S,
           "receipt_overhead_process_cpu_s": RECEIPT_OVERHEAD_CPU_S,
           "launches_after": fit_launches(), "fit_records_after": len(done_fits())}
    append_line(RECEIPTS, end)
    ch = charged()
    print(f"{a.kind} {status}: {end['measured_wall_s']}s wall / {end['measured_process_cpu_s']}s "
          f"process cpu; experiment charged {ch['total_wall']:.0f}s wall / {ch['total_cpu']:.0f}s cpu "
          f"(fit {ch['fit_wall']:.0f}/{ch['fit_cpu']:.0f}, non-fit {ch['nonfit_wall']:.0f}/"
          f"{ch['nonfit_cpu']:.0f}); {fit_launches()} of {MAX_FIT_LAUNCHES} fit attempts", flush=True)
    if refused:
        print("REFUSED BY THE GATE:\n  " + "\n  ".join(g["refusals"]), flush=True)
        raise SystemExit(2)
    if status != "ok":
        raise SystemExit(1 if rc is None else (rc or 1))


if __name__ == "__main__":
    main()
