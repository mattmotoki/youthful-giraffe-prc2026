"""exp-098 controller: the only entry point, and the only thing that writes.

Order of access, enforced here and recorded in the receipts:

  1. every source in `source-manifest.json` is verified by SHA-256, byte size, row
     count and schema.  A single mismatch stops the run before any value is read;
  2. the label-free block runs -- joins, populations, the update's own statistics,
     the absorbed-alignment decomposition and the ranking-side structural report.
     None of it touches a label, and `label_free_finished_unix` is written before
     the next step begins;
  3. the labels are opened, every arm and both controls are priced, the registered
     gates are decided and an atomic phase receipt is written;
  4. phase 2 -- ranking values, the uploaded v22 bytes and a candidate -- runs only
     when the durable receipt says a decision arm passed every gate.

A second invocation refuses before it opens anything.  Every write goes through the
exp-046 write-once guard and an fsync plus atomic replace.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent.parent.parent          # the repository this code lives in
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT / "experiments/exp-046-surfacefit/code"))

# `EXP` is where the registration and every receipt live; `ROOT` is the data root the
# manifest's paths are relative to.  Both are overridable ONLY so the registered-path
# integration test can drive this exact controller over synthetic fixtures; an
# unset environment is the real registration and the real repository.
EXP = Path(os.environ.get("EXP098_EXP", str(HERE.parent)))
ROOT = Path(os.environ.get("EXP098_ROOT", str(CODE_ROOT)))

MAN = json.loads((EXP / "source-manifest.json").read_text())
C = MAN["constants"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> str:
    """Atomic, fsynced, NaN-free JSON.  Returns the file's SHA-256."""
    txt = json.dumps(obj, indent=1, allow_nan=False, sort_keys=False, default=str) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(txt)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    d = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(d)
    finally:
        os.close(d)
    return sha_file(path)


def verify_sources(kinds=None) -> dict:
    """Hash, size, row-count and schema gate.  No value is read."""
    import pyarrow.parquet as pq
    seen = {}
    for s in MAN["sources"]:
        if kinds is not None and s["kind"] not in kinds:
            continue
        p = ROOT / s["path"]
        if not p.exists():
            raise SystemExit(f"missing source: {s['path']}")
        got = {"bytes": p.stat().st_size, "sha256": sha_file(p)}
        if got["sha256"] != s["sha256"] or got["bytes"] != s["bytes"]:
            raise SystemExit(f"source changed since registration: {s['path']}")
        if s["path"].endswith(".parquet"):
            f = pq.ParquetFile(p)
            schema = {n: str(t) for n, t in zip(f.schema_arrow.names, f.schema_arrow.types)}
            if f.metadata.num_rows != s["rows"] or schema != s["schema"]:
                raise SystemExit(f"source schema or row count changed: {s['path']}")
            got["rows"] = s["rows"]
        seen[s["path"]] = got
    return seen


def load_column(path: Path, cols: list) -> "object":
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=cols).to_pandas()


# --------------------------------------------------------------------------- phase 1
def run(overwrite: bool) -> int:
    import numpy as np
    import pandas as pd
    import nooverwrite as NO
    import q09v22 as Q

    out_result = EXP / "analysis/result.json"
    out_receipt = EXP / "analysis/phase-receipt.json"
    out_started = EXP / "analysis/started.json"
    NO.refuse_existing([out_result, out_receipt, out_started], overwrite)

    load1 = os.getloadavg()[0]
    if load1 > C["workstation_launch_refuses_above_1min_load"]:
        raise SystemExit(f"refusing to launch: one-minute load {load1}")

    started = {"utc": utc(), "unix": time.time(), "pid": os.getpid(),
               "one_minute_load_at_launch": load1,
               "host": platform.node(), "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "thread_env": {k: os.environ.get(k) for k in
                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
               "source_commit": os.environ.get("EXP098_COMMIT", "unset")}
    write_json(out_started, started)
    print(f"[start] {started['utc']} pid {started['pid']} load {load1}", flush=True)

    res = {"experiment": "exp-098-q09-serving-repair", "started": started,
           "manifest_sha256": sha_file(EXP / "source-manifest.json"),
           "registered_prediction": MAN.get("registered_prediction"),
           "0_sources": {}, "1_label_free": {}, "2_priced": {}, "3_gates": {}}

    res["0_sources"] = verify_sources({"arm_vectors", "v22_equivalent_panel",
                                       "ranking_head_chain", "ranking_head_published",
                                       "ranking_regime_table", "instrument",
                                       "write_once_guard"})
    print(f"[sources] {len(res['0_sources'])} verified", flush=True)

    # ---------------- 1. label-free -------------------------------------------------
    F = list(C["folds"])
    dat, lf = {}, {"per_fold": {}}
    for f in F:
        pk = load_column(ROOT / f"experiments/exp-040-chain/packet/q09-{f}.parquet",
                         ["MVT_ID_mvt", "airport", "regime", "band", "day",
                          "v10 replay", "chain d11-noac", "none d11-noac"])
        pn = load_column(ROOT / f"experiments/exp-050-nmblind/work/panel-v22-{f}.parquet",
                         ["MVT_ID_mvt", "p_v22", "clean", "obs_has_clock", "fold"])
        if len(pk) != len(pn):
            raise SystemExit(f"{f}: packet {len(pk)} rows, panel {len(pn)} rows")
        ids = pk["MVT_ID_mvt"].to_numpy("float64")
        if not np.array_equal(np.sort(ids), np.sort(pn["MVT_ID_mvt"].to_numpy("float64"))):
            raise SystemExit(f"{f}: packet and panel do not hold the same movement ids")
        if len(np.unique(ids)) != len(ids):
            raise SystemExit(f"{f}: packet ids are not unique")
        pn = pn.set_index("MVT_ID_mvt").reindex(ids)
        if pn["p_v22"].isna().any() or not set(pn["fold"].unique()) == {f}:
            raise SystemExit(f"{f}: panel reindex left a hole or names another fold")
        regime = pk["regime"].to_numpy(str)
        d = {"ids": ids, "airport": pk["airport"].to_numpy(str), "regime": regime,
             "band": np.asarray(pk["band"].fillna(False), dtype=bool), "day": pk["day"].to_numpy("int64"),
             "p10": pk["v10 replay"].to_numpy("float64"),
             "p_v22": pn["p_v22"].to_numpy("float64"),
             "clean": pn["clean"].to_numpy(bool),
             "obs_has_clock": pn["obs_has_clock"].fillna(False).to_numpy(bool)}
        d["u"] = Q.raw_update(pk["chain d11-noac"].to_numpy("float64"), d["p10"], regime)
        d["u_refit"] = Q.raw_update(pk["none d11-noac"].to_numpy("float64"), d["p10"], regime)
        d["perm"] = Q.sham_permutation(regime, d["day"], ids)
        d["u_perm"] = d["u"][d["perm"]]
        dat[f] = d

        moved = d["u"] != 0.0
        head_move = d["p_v22"] - d["p10"]
        lf["per_fold"][f] = {
            "rows": int(len(ids)),
            "rows_by_regime": {g: int((regime == g).sum()) for g in Q.ALL_REGIMES},
            "rows_the_update_moves": int(moved.sum()),
            "rows_moved_by_regime": {g: int((moved & (regime == g)).sum()) for g in Q.ALL_REGIMES},
            "rows_moved_in_the_fixed_table_band": int((moved & d["band"]).sum()),
            "rows_moved_without_a_clock": int((moved & ~d["obs_has_clock"]).sum()),
            "update_rms_s": float(np.sqrt(np.mean(d["u"] ** 2))),
            "update_max_abs_s": float(np.abs(d["u"]).max()),
            "B_energy_s2": float(np.mean(d["u"] ** 2)),
            "sham_refit_rows_moved": int((d["u_refit"] != 0.0).sum()),
            "sham_refit_rms_s": float(np.sqrt(np.mean(d["u_refit"] ** 2))),
            "sham_permutation_sha256": Q.sha_values(d["perm"]),
            "head_movement_v10_to_v22_rms_s": float(np.sqrt(np.mean(head_move ** 2))),
            "absorbed_alignment_A_v10_minus_A_v22_s2": float(np.mean(d["u"] * head_move)),
            "absorbed_alignment_by_regime_s2": {
                g: float(np.sum(d["u"][regime == g] * head_move[regime == g]) / len(ids))
                for g in Q.ALL_REGIMES},
            "absorbed_alignment_by_airport_s2": {
                str(a): float(np.sum(d["u"][d["airport"] == a] * head_move[d["airport"] == a])
                              / len(ids))
                for a in sorted(np.unique(d["airport"]))},
        }
        print("[label-free] %-9s rows %6d moved %6d  B %.3f  absorbed A %.3f"
              % (f, len(ids), moved.sum(), lf["per_fold"][f]["B_energy_s2"],
                 lf["per_fold"][f]["absorbed_alignment_A_v10_minus_A_v22_s2"]), flush=True)

    lf["ranking_structural"] = ranking_structural(np, pd, Q)
    lf["what_absorbed_alignment_means"] = (
        "A_v10 - A_v22 = mean(u*(p_v22 - p_v10)) is an inner product of two prediction "
        "vectors, so the fall of this update's alignment between the two heads is "
        "label-free.  B = mean(u*u) is identical on every head.")
    lf["label_free_finished_unix"] = time.time()
    lf["label_free_finished_utc"] = utc()
    res["1_label_free"] = lf

    # ---------------- 2. the labels -------------------------------------------------
    res["2_priced"] = {"first_label_read_unix": time.time(), "first_label_read_utc": utc()}
    for f in F:
        pn_y = load_column(ROOT / f"experiments/exp-050-nmblind/work/panel-v22-{f}.parquet",
                           ["MVT_ID_mvt", "y_true"]).set_index("MVT_ID_mvt") \
            .reindex(dat[f]["ids"])["y_true"].to_numpy("float64")
        pk_y = load_column(ROOT / f"experiments/exp-040-chain/packet/q09-{f}.parquet",
                           ["MVT_ID_mvt", "y_true"]).set_index("MVT_ID_mvt") \
            .reindex(dat[f]["ids"])["y_true"].to_numpy("float64")
        if not np.array_equal(pn_y, pk_y):
            raise SystemExit(f"{f}: the panel and the packet disagree on the label")
        dat[f]["y"] = pn_y
        dat[f]["r"] = pn_y - dat[f]["p_v22"]

    r = {f: dat[f]["r"] for f in F}
    day = {f: dat[f]["day"] for f in F}
    apt = {f: dat[f]["airport"] for f in F}
    directions = {"real": "u", "sham_refit": "u_refit", "sham_permutation": "u_perm"}
    priced = {}
    for dname, key in directions.items():
        u = {f: dat[f][key] for f in F}
        s1 = {f: 1.0 for f in F}
        sA1 = Q.global_loo_scales(r, u)
        sA2, provA2 = Q.airport_loo_scales(r, u, apt, sA1)
        scales = {"A0": s1, "A1": sA1, "A2": sA2}
        priced[dname] = {"scales": {"A0": s1, "A1": sA1,
                                    "A2_provenance": {f: provA2[f] for f in F}},
                         "A3_within_month_cv": Q.cv_scales(r, u, day),
                         "arms": {}}
        for arm in Q.ARMS:
            folds, parts = {}, {}
            for f in F:
                s = scales[arm][f]
                proposed = np.asarray(s, "float64") * u[f]
                d = Q.served_delta(dat[f]["p_v22"], u[f], s)
                folds[f] = Q.price_fold(r[f], d, day[f], dat[f]["regime"], dat[f]["band"],
                                        apt[f], dat[f]["y"], dat[f]["clean"], proposed)
                parts[f] = (Q.A.blocks(day[f], Q.BLOCK_DAYS_PRIMARY),
                            r[f] ** 2, (r[f] - d) ** 2)
            entry = {"folds": folds,
                     "board": Q.board(folds, {f: parts[f] for f in Q.BOARD_FOLDS})}
            priced[dname]["arms"][arm] = entry
            print("[%-16s %s] board %+.5f  jul %+.5f %s  dec %+.5f  jan %+.5f %s"
                  % (dname, arm, entry["board"]["board_gain_s"],
                     folds["july_loo"]["gain_rmse_s"], folds["july_loo"]["ci95_1day"],
                     folds["december"]["gain_rmse_s"],
                     folds["january"]["gain_rmse_s"], folds["january"]["ci95_1day"]),
                  flush=True)
    res["2_priced"].update(priced)

    # ---------------- 3. the registered gates --------------------------------------
    g = {}
    for arm in Q.ARMS:
        g[arm] = Q.gates(priced["real"]["arms"][arm],
                         priced["sham_refit"]["arms"][arm],
                         priced["sham_permutation"]["arms"][arm])
    decision = next((a for a in Q.DECISION_PRECEDENCE if g[a]["all_pass"]), None)
    a0 = priced["real"]["arms"]["A0"]["board"]["board_gain_s"]
    verdict = "ADVANCE" if decision else ("FAIL" if a0 <= 0.0 else "MAYBE")
    res["3_gates"] = {"per_arm": g, "decision_arm": decision, "verdict": verdict,
                      "gate_text": "board gain >= +0.10 s, July and January 95% lower bounds "
                                   "above zero, December point non-negative, and the arm beats "
                                   "both registered controls in board currency",
                      "A0_board_gain_s": a0}
    receipt = {"utc": utc(), "verdict": verdict, "decision_arm": decision,
               "per_arm_gates": g, "phase": "candidate_authorized" if decision else "historical_only",
               "A0_board_gain_s": a0}
    NO.refuse_existing([out_receipt], overwrite)
    res["phase_receipt_sha256"] = write_json(out_receipt, receipt)

    # ---------------- 4. phase 2, only on a durable passing receipt -----------------
    landed = json.loads(out_receipt.read_text())
    if landed["phase"] != "candidate_authorized":
        res["4_candidate"] = {"phase": landed["phase"],
                              "opened": False,
                              "why": "no registered arm passed every gate, so no ranking value, "
                                     "no uploaded-v22 byte and no candidate was opened"}
        print("[phase] historical only; ranking and v22 bytes were not opened", flush=True)
    else:
        res["4_candidate"] = phase2(np, pd, Q, NO, overwrite, landed["decision_arm"],
                                    priced["real"], r, {f: dat[f]["u"] for f in F}, apt)

    res["finished_utc"] = utc()
    res["wall_seconds"] = time.time() - started["unix"]
    NO.refuse_existing([out_result], overwrite)
    sha = write_json(out_result, res)
    print(f"\nverdict: {verdict}  decision arm: {decision}  A0 board {a0:+.5f}", flush=True)
    print(f"[done] result.json sha256 {sha} in {res['wall_seconds']:.1f} s", flush=True)
    return 0


def deployment_scale(np, Q, arm: str, r: dict, u: dict, apt: dict, ranking_airport):
    """The amplitude a candidate would serve: the same estimator fitted on all three folds.

    A0 serves 1.0.  A1 serves one pooled bounded scale.  A2 serves one pooled bounded
    scale per airport, with the A1 scale wherever the airport is below the registered
    support.  No ranking value enters any of these fits; ranking labels do not exist.
    """
    if arm == "A0":
        return 1.0, {"arm": "A0", "s": 1.0}
    num = sum(float(np.sum(r[f] * u[f])) for f in u)
    den = sum(float(np.sum(u[f] * u[f])) for f in u)
    lo, hi = Q.SCALE_BOUNDS
    s_all = float(min(max(num / den if den > 0 else 0.0, lo), hi))
    if arm == "A1":
        return s_all, {"arm": "A1", "s": s_all}
    per, prov = {}, {}
    for a in sorted({x for f in apt for x in np.unique(apt[f])}):
        n_, d_, ch = 0.0, 0.0, 0
        for f in u:
            m = (apt[f] == a) & (u[f] != 0.0)
            ch += int(m.sum())
            n_ += float(np.sum(r[f][m] * u[f][m]))
            d_ += float(np.sum(u[f][m] * u[f][m]))
        own = ch >= Q.AIRPORT_MIN_CHANGED and d_ > 0.0
        s = float(min(max(n_ / d_, lo), hi)) if own else s_all
        per[str(a)] = s
        prov[str(a)] = {"s": s, "own_fit": bool(own), "changed_rows": ch}
    s_row = np.array([per.get(str(a), s_all) for a in ranking_airport], "float64")
    return s_row, {"arm": "A2", "fallback_s": s_all, "per_airport": prov}


def phase2(np, pd, Q, NO, overwrite: bool, arm: str, real: dict,
           r: dict, u: dict, apt: dict) -> dict:
    """Ranking values, the uploaded v22 bytes and one candidate.  Gated by the receipt."""
    out_cand = EXP / "analysis/candidate.json"
    cand_path = EXP / "analysis/exp-098-q09-v22-candidate.parquet" if os.environ.get("EXP098_ROOT") \
        else ROOT / "submissions/candidates/exp-098-q09-v22.parquet"
    NO.refuse_existing([out_cand, cand_path], overwrite)
    verify_sources({"uploaded_v22"})

    hc = pd.read_parquet(ROOT / "experiments/exp-040-chain/candidates/head-chain.parquet")
    hp = pd.read_parquet(ROOT / "experiments/exp-040-chain/candidates/head-published.parquet")
    gt = pd.read_parquet(ROOT / "experiments/exp-026-rebuild/candidates/ranking-gates.parquet",
                         columns=["MVT_ID_mvt", "regime", "airport"])
    v22 = pd.read_parquet(ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet")
    ids = v22["MVT_ID_mvt"].to_numpy("float64")
    for name, frame in (("head-chain", hc), ("head-published", hp)):
        if not np.array_equal(frame["MVT_ID_mvt"].to_numpy("float64"), ids):
            raise SystemExit(f"{name} is not in the uploaded v22 id order")
    gt = gt.set_index("MVT_ID_mvt").reindex(ids)
    regime = gt["regime"].to_numpy(str)
    airport = gt["airport"].astype(str).to_numpy()
    u_rank = Q.raw_update(hc["TAXITIME_SEC_mvt"].to_numpy("float64"),
                          hp["TAXITIME_SEC_mvt"].to_numpy("float64"), regime)
    s, prov = deployment_scale(np, Q, arm, r, u, apt, airport)
    base = v22["TAXITIME_SEC_mvt"].to_numpy("float64")
    d = Q.served_delta(base, u_rank, s)
    new = base + d
    if not np.isfinite(new).all() or new.min() < Q.FLOOR:
        raise SystemExit("the candidate is not finite or breaks the 60 s floor")
    frame = pd.DataFrame({"MVT_ID_mvt": v22["MVT_ID_mvt"].to_numpy(),
                          "TAXITIME_SEC_mvt": new})
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    NO.refuse_existing([cand_path], overwrite)
    frame.to_parquet(cand_path, index=False)
    back = pd.read_parquet(cand_path)
    if not np.array_equal(back["TAXITIME_SEC_mvt"].to_numpy("float64"), new):
        raise SystemExit("the candidate did not read back as it was written")
    moved = d != 0.0
    rep = {"opened": True, "decision_arm": arm, "deployment_scale": prov,
           "path": str(cand_path), "sha256": sha_file(cand_path),
           "rows": int(len(ids)), "rows_changed": int(moved.sum()),
           "rows_identical_to_v22": int((~moved).sum()),
           "rms_change_s": float(np.sqrt(np.mean(d ** 2))),
           "max_abs_change_s": float(np.abs(d).max()),
           "min": float(new.min()), "max": float(new.max()),
           "by_regime": {g: {"rows": int((regime == g).sum()),
                             "rows_changed": int((moved & (regime == g)).sum())}
                         for g in Q.ALL_REGIMES},
           "by_airport": {a: {"rows": int((airport == a).sum()),
                              "rows_changed": int((moved & (airport == a)).sum())}
                          for a in sorted(set(airport))}}
    rep["all_388_ranking_noac_rows"] = rep["by_regime"]["noac"]
    rep["noac_held_bit_identical"] = bool(rep["by_regime"]["noac"]["rows_changed"] == 0)
    write_json(out_cand, rep)
    print(f"[candidate] {rep['path']} sha {rep['sha256'][:16]} moved {rep['rows_changed']}",
          flush=True)
    return rep


def ranking_structural(np, pd, Q) -> dict:
    """Label-free: what the served update would change on the 344,841 ranking rows.

    Q09 asks for changes to all 388 ranking `noac` rows.  Neither head carries a
    label and the regime table carries none either, so this answer is available
    without opening a label, a v22 byte or a candidate.
    """
    hc = pd.read_parquet(ROOT / "experiments/exp-040-chain/candidates/head-chain.parquet")
    hp = pd.read_parquet(ROOT / "experiments/exp-040-chain/candidates/head-published.parquet")
    gt = pd.read_parquet(ROOT / "experiments/exp-026-rebuild/candidates/ranking-gates.parquet",
                         columns=["MVT_ID_mvt", "regime", "airport"])
    ids = hc["MVT_ID_mvt"].to_numpy("float64")
    if not np.array_equal(ids, hp["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the two ranking heads are not in the same id order")
    gt = gt.set_index("MVT_ID_mvt").reindex(ids)
    if gt["regime"].isna().any():
        raise SystemExit("the ranking regime table does not cover every head row")
    regime = gt["regime"].to_numpy(str)
    u = Q.raw_update(hc["TAXITIME_SEC_mvt"].to_numpy("float64"),
                     hp["TAXITIME_SEC_mvt"].to_numpy("float64"), regime)
    moved = u != 0.0
    out = {"rows": int(len(ids)), "rows_the_update_moves": int(moved.sum()),
           "by_regime": {}, "by_airport": {}}
    for g in Q.ALL_REGIMES:
        m = regime == g
        out["by_regime"][g] = {"rows": int(m.sum()), "rows_changed": int((moved & m).sum()),
                               "rms_s": float(np.sqrt(np.mean(u[m] ** 2))) if m.any() else 0.0,
                               "max_abs_s": float(np.abs(u[m]).max()) if m.any() else 0.0}
    for a in sorted(pd.unique(gt["airport"].astype(str))):
        m = gt["airport"].astype(str).to_numpy() == a
        out["by_airport"][a] = {"rows": int(m.sum()), "rows_changed": int((moved & m).sum())}
    out["noac_rows_all_reported"] = out["by_regime"]["noac"]
    return out


def main(argv) -> int:
    overwrite = "--overwrite" in argv
    sys.path.insert(0, str(CODE_ROOT / "experiments/exp-046-surfacefit/code"))
    import nooverwrite as NO
    (EXP / "analysis").mkdir(parents=True, exist_ok=True)
    NO.refuse_existing([EXP / "analysis/result.json", EXP / "analysis/phase-receipt.json",
                        EXP / "analysis/started.json", EXP / "analysis/exit.json",
                        EXP / "analysis/candidate.json"], overwrite)
    claim = EXP / "analysis/launch.claim"
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print("refusing: this registration has already been claimed", file=sys.stderr)
        return 3
    os.write(fd, (utc() + f" pid {os.getpid()}\n").encode())
    os.close(fd)
    exit_path = EXP / "analysis/exit.json"
    t0 = time.time()
    try:
        code = run(overwrite)
        write_json(exit_path, {"utc": utc(), "exit_code": code, "pid": os.getpid(),
                               "wall_seconds": time.time() - t0, "state": "complete"})
        return code
    except BaseException as e:  # a failure is terminal and durable
        write_json(exit_path, {"utc": utc(), "exit_code": 1, "pid": os.getpid(),
                               "wall_seconds": time.time() - t0, "state": "INCOMPLETE",
                               "error": f"{type(e).__name__}: {e}"})
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
