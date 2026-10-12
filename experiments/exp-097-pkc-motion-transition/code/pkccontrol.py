"""exp-097 PK-C motion transition: the one entry point, in three ordered phases.

    phase 1  label-free   support, identity and the physical-bound diagnostic.
                          Reads no label, no residual and no target under any name.
    phase 2  label-bearing the registered out-of-fold fit and its score against the
                          exact v22-equivalent fold vectors.  Runs only on a durable
                          phase-1 receipt that says the support condition passed.
    phase 3  candidate    the ranking write.  Runs only on a durable phase-2 receipt
                          that says every gate passed.  Never uploads.

Access order is enforced by the receipts, not by discipline: each phase refuses to open
its sources until the previous phase's receipt exists on disk and says PASS.

    OMP_NUM_THREADS=8 .venv/bin/python \
        experiments/exp-097-pkc-motion-transition/code/pkccontrol.py --phase=1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pkcmotion as M                                                  # noqa: E402

EXP = M.EXP
ROOT = M.ROOT
ANALYSIS = EXP / "analysis"
CANDIDATE_PATH = str(ROOT / "submissions/candidates/exp-097-motion-transition.parquet")
RECEIPT = str(EXP / "analysis" / "receipt-phase{n}.json")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict) -> None:
    """Atomic, fsynced, NaN-free."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=float)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def verify_manifest() -> dict:
    """Re-hash every bound source before a phase opens any of them.

    `pkcmanifest.verify` existed but nothing called it (Sol round 1, blocking 6).
    """
    import pkcmanifest as MAN                                          # noqa: PLC0415
    man = json.loads((EXP / "source-manifest.json").read_text())
    v = MAN.verify(man)
    if not v["ok"]:
        raise SystemExit("source manifest mismatch: "
                         + json.dumps(v["mismatches"][:10], indent=2))
    return {"checked": v["checked"], "ok": True,
            "matcher_sha256": man["matcher_sha256"],
            "serving_rule_sha256": man["serving_rule_sha256"]}


def claim(phase: int, revision: str) -> Path:
    """An exclusive, permanent claim taken BEFORE a one-shot phase opens a source.

    `O_CREAT | O_EXCL` on a file that is never removed: a second invocation of the same
    phase and revision refuses before it reads anything (Sol round 1, blocking 6).
    """
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    p = ANALYSIS / f"claim-phase{phase}{revision}.json"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(
            f"phase {phase}{revision} was already claimed at {p}; it gets ONE read. "
            "A new revision needs a new registration.") from None
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"phase": phase, "revision": revision,
                             "claimed_utc": utc(), "pid": os.getpid()},
                            indent=2, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    return p


def read_receipt(n: int) -> dict | None:
    p = Path(RECEIPT.format(n=n))
    if not p.exists():
        return None
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- loaders

def panel_label_free(fold: str) -> pd.DataFrame:
    cols = list(M.PANEL_COLS_LABEL_FREE)
    M.refuse_columns(cols, M.PANEL_COLS_LABEL_FREE, f"panel {fold} (label-free)")
    d = pd.read_parquet(ROOT / M.PANEL[fold], columns=cols)
    if list(d.columns) != cols:
        raise SystemExit(f"panel {fold}: returned columns {list(d.columns)} != {cols}")
    d["id64"] = M.check_ids(d["MVT_ID_mvt"].to_numpy(), f"panel {fold}")
    d["own_has_clock"] = d["obs_has_clock"].astype(bool)
    d["noac_bool"] = d["noac"].astype("float64").to_numpy() != 0
    return d


def panel_labels(fold: str) -> pd.DataFrame:
    cols = list(M.PANEL_COLS_LABEL)
    d = pd.read_parquet(ROOT / M.PANEL[fold], columns=cols)
    if list(d.columns) != cols:
        raise SystemExit(f"panel {fold}: returned columns {list(d.columns)} != {cols}")
    d["id64"] = M.check_ids(d["MVT_ID_mvt"].to_numpy(), f"panel {fold} labels")
    return d[["id64", "y_true"]]


BLOCK_KEEP = ["MVT_ID_mvt", "reach_class", "obs_matched", "obs_has_clock",
              "obs_taxi", "obs_taxi_motion", "stand_stop_s", "turnaround_anchored",
              "seg_span_s", "n_ground_points", "dt_takeoff", "stand_dist_km"]


def fold_frame(fold: str) -> pd.DataFrame:
    """The panel's label-free side joined one-to-one to v22's own rebuilt block."""
    month = M.FOLD_MONTH[fold]
    pan = panel_label_free(fold)
    blk = M.month_block(month)[BLOCK_KEEP].rename(
        columns={"obs_has_clock": "obs_has_clock_block"})
    blk["id64"] = M.check_ids(blk["MVT_ID_mvt"].to_numpy(), f"block {month}")
    d = pan.merge(blk.drop(columns=["MVT_ID_mvt"]), on="id64", how="left",
                  validate="one_to_one")
    if len(d) != len(pan):
        raise SystemExit(f"{fold}: join changed the row count")
    return M.add_signal(d)


def fold_nm(fold: str) -> pd.DataFrame:
    """`MVT - AOBT_3_flt` per row: an INPUT-only taxi reference, never a label.

    `AOBT_3_flt` is the Network Manager's actual off-block and is present in the
    ranking file; it is an input under `docs/input-rule.md`, not the target.  It is the
    only reference the label-free phase has for the motion clock, and it is a weak one:
    the pilot measured 309 s of NM error against the same rows the ADS-B clock made
    144 s on.  It is reported as a contrast, never as a gate.
    """
    D = M.prc_data()
    month = M.FOLD_MONTH[fold]
    dep = M._departures(D.month_file(month))
    mvt = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    aobt = (dep["AOBT_3_flt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return pd.DataFrame({
        "id64": M.check_ids(dep["MVT_ID_mvt"].to_numpy(), f"nm {month}"),
        "nm_taxi": mvt.to_numpy("float64") - aobt.to_numpy("float64")})


def ranking_frame(month: str) -> pd.DataFrame:
    """v22's ownership vector for a ranking month joined to its rebuilt block."""
    dep, blk = M.ranking_month_block(month)
    cls = pd.read_parquet(ROOT / M.CLASS_FILE[month], columns=list(M.CLASS_COLS))
    cls["id64"] = M.check_ids(cls["MVT_ID_mvt"].to_numpy(), f"class {month}")
    cls["own_has_clock"] = cls["obs_has_clock"].astype(bool)
    cls["noac_bool"] = cls["noac"].astype(bool)
    b = blk[BLOCK_KEEP].rename(columns={"obs_has_clock": "obs_has_clock_block"}).copy()
    b["id64"] = M.check_ids(b["MVT_ID_mvt"].to_numpy(), f"block {month}")
    d = cls.merge(b.drop(columns=["MVT_ID_mvt"]), on="id64", how="left",
                  validate="one_to_one")
    dd = dep[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].copy()
    dd["id64"] = M.check_ids(dd["MVT_ID_mvt"].to_numpy(), f"ranking dep {month}")
    dd["airport"] = dd["ADEP_mvt"].astype(str)
    dd["day"] = dd["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    dd["mvt_s"] = (dd["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    dd["nm_taxi"] = dd["mvt_s"] - (dd["AOBT_3_flt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    d = d.merge(dd.drop(columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AOBT_3_flt"]),
                on="id64", how="left", validate="one_to_one")
    return M.add_signal(d)


def uploaded_v22() -> pd.DataFrame:
    p = ROOT / M.UPLOADED_V22
    d = pd.read_parquet(p)
    cols = [c for c in d.columns]
    if "MVT_ID_mvt" not in cols or "TAXITIME_SEC_mvt" not in cols:
        raise SystemExit(f"{M.UPLOADED_V22}: unexpected schema {cols}")
    out = pd.DataFrame({"id64": M.check_ids(d["MVT_ID_mvt"].to_numpy(), "uploaded v22"),
                        "prediction": d["TAXITIME_SEC_mvt"].to_numpy("float64")})
    return out


# --------------------------------------------------------------------------- phase 1

def uploaded_v22_frame() -> pd.DataFrame:
    """The uploaded v22 file as it stands on disk.  The candidate is written from it."""
    return pd.read_parquet(ROOT / M.UPLOADED_V22)


def phase1() -> dict:
    t0 = time.time()
    man = verify_manifest()
    rec = {"experiment": "exp-097-pkc-motion-transition", "phase": 1,
           "phase_kind": "label-free support and identity",
           "manifest": man, "started_utc": utc(), "folds": {}, "ranking": {}}
    v22 = uploaded_v22()
    pred = dict(zip(v22["id64"].tolist(), v22["prediction"].tolist()))

    support_total = 0
    for f in M.FOLDS:
        d = fold_frame(f)
        nm = fold_nm(f)
        d = d.merge(nm, on="id64", how="left", validate="one_to_one")
        comp = M.complement_mask(d)
        sup = M.support_mask(d)
        support_total += int(sup.sum())
        s = d[sup]
        e = s["obs_taxi_motion"].to_numpy("float64") - s["nm_taxi"].to_numpy("float64")
        e = e[np.isfinite(e)]
        # the reference the same instrument reaches on the rows v22 DOES own
        own = d[d["own_has_clock"].to_numpy(bool) & ~d["noac_bool"].to_numpy(bool)
                & np.isfinite(d["obs_taxi_motion"].to_numpy("float64"))]
        eo = own["obs_taxi_motion"].to_numpy("float64") - own["nm_taxi"].to_numpy("float64")
        eo = eo[np.isfinite(eo)]
        # the physical lower bound: the aircraft was MOVING on the ground at
        # `first_motion`, so its off-block is no later, so taxi >= obs_taxi_motion.
        p = s["p_v22"].to_numpy("float64")
        L = s["obs_taxi_motion"].to_numpy("float64")
        exc = np.clip(L - p, 0.0, None)
        rec["folds"][f] = {
            "month": M.FOLD_MONTH[f],
            "panel_rows": int(len(d)),
            "complement_rows": int(comp.sum()),
            "own_vs_block_clock_disagreements": int(np.sum(
                d["own_has_clock"].to_numpy(bool) != d["block_has_clock"].to_numpy(bool))),
            "reach_class_census": {str(k): int(v) for k, v in
                                   d["reach_class"].value_counts().items()},
            "support": M.grade_counts(d, sup),
            "support_share_of_month": float(sup.sum() / max(len(d), 1)),
            "obs_taxi_motion_quantiles": M.quantiles(s["obs_taxi_motion"]),
            "p_v22_quantiles_on_support": M.quantiles(s["p_v22"]),
            "nm_cross_check": {
                "rows": int(len(e)),
                "median_motion_minus_nm_s": float(np.median(e)) if len(e) else None,
                "rmse_motion_minus_nm_s": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                "reference_rows_v22_owns": int(len(eo)),
                "reference_median_s": float(np.median(eo)) if len(eo) else None,
                "reference_rmse_s": float(np.sqrt(np.mean(eo ** 2))) if len(eo) else None,
            },
            "lower_bound_clip": {
                "rows_with_p_below_bound": int(np.sum(exc > 0)),
                "sum_sq_excess_s2": float(np.sum(exc ** 2)),
                "per_month_row_s2": float(np.sum(exc ** 2) / max(len(d), 1)),
                "approx_board_s": float(np.sum(exc ** 2) / max(len(d), 1)
                                        / (2 * M.V22_BOARD)),
            },
        }

    for m in M.RANK_MONTHS:
        d = ranking_frame(m)
        d["p_v22"] = d["id64"].map(pred).astype("float64")
        comp = M.complement_mask(d)
        sup = M.support_mask(d)
        s = d[sup]
        e = s["obs_taxi_motion"].to_numpy("float64") - s["nm_taxi"].to_numpy("float64")
        e = e[np.isfinite(e)]
        p = s["p_v22"].to_numpy("float64")
        L = s["obs_taxi_motion"].to_numpy("float64")
        exc = np.clip(L - p, 0.0, None)
        exc = exc[np.isfinite(exc)]
        rec["ranking"][m] = {
            "class_rows": int(len(d)),
            "complement_rows": int(comp.sum()),
            "own_vs_block_clock_disagreements": int(np.sum(
                d["own_has_clock"].to_numpy(bool) != d["block_has_clock"].to_numpy(bool))),
            "reach_class_census": {str(k): int(v) for k, v in
                                   d["reach_class"].value_counts().items()},
            "support": M.grade_counts(d, sup),
            "support_share_of_month": float(sup.sum() / max(len(d), 1)),
            "obs_taxi_motion_quantiles": M.quantiles(s["obs_taxi_motion"]),
            "p_v22_quantiles_on_support": M.quantiles(s["p_v22"]),
            "nm_cross_check": {
                "rows": int(len(e)),
                "median_motion_minus_nm_s": float(np.median(e)) if len(e) else None,
                "rmse_motion_minus_nm_s": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
            },
            "lower_bound_clip": {
                "rows_with_p_below_bound": int(np.sum(exc > 0)),
                "sum_sq_excess_s2": float(np.sum(exc ** 2)),
                "approx_board_s": float(np.sum(exc ** 2) / M.BOARD_TOTAL
                                        / (2 * M.V22_BOARD)),
            },
        }

    per_fold_ok = {f: rec["folds"][f]["support"]["rows"] >= M.MIN_SUPPORT_PER_FOLD
                   for f in M.FOLDS}
    passed = support_total >= M.MIN_SUPPORT_TOTAL and all(per_fold_ok.values())
    rec["support_condition"] = {
        "total_support": int(support_total),
        "min_support_total": M.MIN_SUPPORT_TOTAL,
        "min_support_per_fold": M.MIN_SUPPORT_PER_FOLD,
        "per_fold_pass": {f: bool(v) for f, v in per_fold_ok.items()},
        "verdict": "PASS" if passed else "FAIL",
    }
    rec["seconds"] = round(time.time() - t0, 1)
    rec["finished_utc"] = utc()
    return rec


# --------------------------------------------------------------------------- phase 2

ARMS = {"motion": M.MOTION_COLS, "extent": M.EXTENT_COLS, "placebo": M.MOTION_COLS}


def preflight(phase: int, revision: str, bound_files=None) -> dict:
    """Every barrier a label-bearing phase must pass, BEFORE it claims its one read.

    Nothing here writes, claims or opens a source, so a direct controller call — which has
    no launcher handoff — is refused with the single phase-2 read still unspent
    (assignment item 3).
    """
    import pkcguard as G                                               # noqa: PLC0415
    try:
        chain = G.predecessor_chain()
        repo = G.require_clean_pushed_branch(phase, revision)
        start = G.start_barrier(phase, revision)
    except G.BoundaryError as exc:
        raise SystemExit(str(exc)) from exc
    accepted = desk_acceptance(phase, bound_files=bound_files)
    for field, seen, want in (("registration", accepted["source_commit"], repo["head"]),):
        if seen != want:
            raise SystemExit(f"phase {phase} refused: acceptance names source commit "
                             f"{seen[:12]}, the run is at {want[:12]}")
    return {"phase1_chain": chain, "repo": repo, "start": start,
            "desk_acceptance": accepted}


def desk_acceptance(phase: int, bound_files=None) -> dict:
    """The external desk receipt, or nothing runs.

    Registered acceptance was one fresh Sol xhigh verdict. The desk stopped Sol reviews at
    2026-09-16 05:30 UTC before a verdict existed (`review/round2-stopped.md`), so
    acceptance is a receipt the desk issues instead (`SPEC.md` §12). This is checked first,
    before the manifest, before any claim and before a single source is opened.
    """
    import pkcaccept as ACC                                            # noqa: PLC0415
    try:
        return ACC.verify(phase, bound_files=bound_files)
    except ACC.AcceptanceError as exc:
        raise SystemExit(str(exc)) from exc


def phase2(boundary: dict) -> dict:
    """The one label-bearing read.  `boundary` is `preflight`'s record: the phase-1 r2
    chain with its digests, the repository state, the launcher handoff and the desk
    acceptance.  Phase 2 does NOT consult `read_receipt(1)`: that would read whichever
    revision this run happens to carry, and the screen of record is r2 and only r2."""
    accepted = boundary["desk_acceptance"]
    man = verify_manifest()
    t0 = time.time()
    rec = {"experiment": "exp-097-pkc-motion-transition", "phase": 2,
           "phase_kind": "label-bearing out-of-fold fit and score",
           "manifest": man, "desk_acceptance": accepted,
           "phase1_chain": boundary["phase1_chain"], "repo": boundary["repo"],
           "start": boundary["start"],
           "started_utc": utc(), "constants": {
               "ridge": M.RIDGE, "taxi_lo": M.TAXI_LO, "taxi_hi": M.TAXI_HI,
               "boot": M.BOOT, "seed": M.SEED, "board_share": M.BOARD_SHARE,
               "v22_board": M.V22_BOARD, "upload_gate_s": M.UPLOAD_GATE_S},
           "arms": {}}

    frames = {}
    for f in M.FOLDS:
        d = fold_frame(f)
        lab = panel_labels(f)
        d = d.merge(lab, on="id64", how="left", validate="one_to_one")
        y = d["y_true"].to_numpy("float64")
        keep = np.isfinite(y)
        d = d[keep].reset_index(drop=True)
        d["support"] = M.support_mask(d)
        frames[f] = d
    rec["fold_rows"] = {f: {"scored_rows": int(len(frames[f])),
                            "support_rows": int(frames[f]["support"].sum())}
                        for f in M.FOLDS}

    # Each fold's supported rows are permuted ONCE per arm, with exactly
    # `SEED + ARM_OFFSET + FOLD_OFFSET[that fold]`, and the same permuted fold is used
    # both when the fold trains and when it is scored (Sol round 1, blocking 3: the
    # application side used an unregistered `+7`).  A non-placebo arm uses the rows
    # unchanged.
    shuffled = {}
    for arm, cols in ARMS.items():
        shuffled[arm] = {}
        for f in M.FOLDS:
            sup = frames[f][frames[f]["support"]].reset_index(drop=True)
            shuffled[arm][f] = (M.permute_within(sup, cols,
                                                 M.SEED + M.ARM_OFFSET[arm]
                                                 + M.FOLD_OFFSET[f])
                                if arm == "placebo" else sup)

    for arm, cols in ARMS.items():
        per_fold = {}
        coefs = {}
        for f in M.FOLDS:
            tgt = frames[f]
            others = [g for g in M.FOLDS if g != f]
            tr = pd.concat([shuffled[arm][g] for g in others], ignore_index=True)
            Xtr, med = M.design(tr, cols)
            beta = M.ridge_fit(Xtr, tr["y_true"].to_numpy("float64"))
            coefs[f] = {"beta": [float(b) for b in beta],
                        "columns": ["intercept", "p_v22", *cols],
                        "train_rows": int(len(tr)), "medians": med}
            sel = tgt["support"].to_numpy(bool)
            app = shuffled[arm][f]
            if not np.array_equal(app["id64"].to_numpy(np.int64),
                                  tgt["id64"].to_numpy(np.int64)[sel]):
                raise SystemExit(f"{arm}/{f}: the applied frame is not the support rows"
                                 " in their own order")
            Xap, _ = M.design(app, cols, med)
            p_base = tgt["p_v22"].to_numpy("float64").copy()
            p_new = p_base.copy()
            p_new[sel] = M.apply_fit(Xap, beta)
            y = tgt["y_true"].to_numpy("float64")
            sc = M.fold_scores(y, p_base, p_new)
            sc["bootstrap"] = M.day_bootstrap_gain(
                y, p_base, p_new, tgt["day"].astype(str).to_numpy(),
                M.SEED + M.ARM_OFFSET[arm] + M.FOLD_OFFSET[f])
            sc["support_rows"] = int(sel.sum())
            sc["support_rmse_base"] = float(np.sqrt(np.mean((y[sel] - p_base[sel]) ** 2))) \
                if sel.any() else None
            sc["support_rmse_new"] = float(np.sqrt(np.mean((y[sel] - p_new[sel]) ** 2))) \
                if sel.any() else None
            per_fold[f] = sc
            per_fold[f]["_arrays"] = {"e0": (y - p_base) ** 2, "e1": (y - p_new) ** 2,
                                      "days": tgt["day"].astype(str).to_numpy()}
        mse0 = {f: per_fold[f]["mse_base"] for f in M.BOARD_SHARE}
        mse1 = {f: per_fold[f]["mse_new"] for f in M.BOARD_SHARE}
        board_gain = M.board_rmse(mse0) - M.board_rmse(mse1)
        bb = M.board_bootstrap({f: per_fold[f]["_arrays"] for f in M.BOARD_SHARE},
                               M.SEED + M.ARM_OFFSET[arm] + 500)
        for f in per_fold:
            per_fold[f].pop("_arrays")
        rec["arms"][arm] = {
            "columns": ["intercept", "p_v22", *cols],
            "by_fold": per_fold,
            "coefficients": coefs,
            "board_rmse_base": M.board_rmse(mse0),
            "board_rmse_new": M.board_rmse(mse1),
            "board_share_gain_s": float(board_gain),
            "board_gain_ci95": bb["ci95"],
        }

    # the deployment fit, for a later candidate: all three folds' support rows
    tr_all = pd.concat([frames[g][frames[g]["support"]] for g in M.FOLDS],
                       ignore_index=True)
    Xall, med_all = M.design(tr_all, M.MOTION_COLS)
    beta_all = M.ridge_fit(Xall, tr_all["y_true"].to_numpy("float64"))
    rec["deployment_fit"] = {"beta": [float(b) for b in beta_all],
                             "columns": ["intercept", "p_v22", *M.MOTION_COLS],
                             "medians": med_all, "train_rows": int(len(tr_all))}

    m = rec["arms"]["motion"]
    e = rec["arms"]["extent"]
    pl = rec["arms"]["placebo"]
    gates = {
        "G1_board_gain_ge_0.10": bool(m["board_share_gain_s"] >= M.UPLOAD_GATE_S),
        "G2_nonnegative_every_fold": bool(all(m["by_fold"][f]["rmse_gain"] >= 0
                                              for f in M.FOLDS)),
        "G3_positive_lower_bound_july_january": bool(
            m["by_fold"]["july_loo"]["bootstrap"]["ci95"][0] > 0
            and m["by_fold"]["january"]["bootstrap"]["ci95"][0] > 0),
        "G4_beats_extent_control": bool(m["board_share_gain_s"] > e["board_share_gain_s"]),
        "G5_beats_placebo_control": bool(m["board_share_gain_s"] > pl["board_share_gain_s"]),
    }
    # `gates` holds EXACTLY the five registered gates and nothing else, so the producer and
    # the watcher's validator read one schema.  The aggregate lives beside the map, never
    # inside it: an extra key there would either break the validator or, worse, be counted
    # as a gate by `all(gates.values())`.
    if tuple(gates) != M.GATES:
        raise SystemExit(f"phase 2 built the gate set {tuple(gates)}, registered {M.GATES}")
    rec["gates"] = gates
    rec["all_gates_pass"] = bool(all(gates[g] for g in M.GATES))
    rec["decision"] = "ADVANCE" if rec["all_gates_pass"] else "FAIL"
    rec["board_after_if_transferred"] = float(M.V22_BOARD - m["board_share_gain_s"])
    rec["seconds"] = round(time.time() - t0, 1)
    rec["finished_utc"] = utc()
    return rec


# --------------------------------------------------------------------------- phase 3

def phase3(boundary: dict) -> dict:
    accepted = boundary["desk_acceptance"]
    r2 = read_receipt(2)
    if not r2 or r2.get("decision") != "ADVANCE":
        raise SystemExit("phase 3 refused: no phase-2 receipt with an ADVANCE decision")
    man = verify_manifest()
    t0 = time.time()
    beta = np.asarray(r2["deployment_fit"]["beta"], dtype="float64")
    med = r2["deployment_fit"]["medians"]
    v22 = uploaded_v22()
    pred = dict(zip(v22["id64"].tolist(), v22["prediction"].tolist()))
    changed = {}
    new_by_id = {}
    for m in M.RANK_MONTHS:
        d = ranking_frame(m)
        d["p_v22"] = d["id64"].map(pred).astype("float64")
        sel = M.support_mask(d) & np.isfinite(d["p_v22"].to_numpy("float64"))
        app = d[sel]
        X, _ = M.design(app, M.MOTION_COLS, med)
        vals = M.apply_fit(X, beta)
        for i, v in zip(app["id64"].tolist(), vals.tolist()):
            new_by_id[int(i)] = float(v)
        changed[m] = int(sel.sum())
    out = v22.copy()
    base = out["prediction"].to_numpy("float64").copy()
    newv = base.copy()
    ids = out["id64"].to_numpy(np.int64)
    hit = np.array([i in new_by_id for i in ids])
    newv[hit] = np.array([new_by_id[int(i)] for i in ids[hit]])
    n_changed = int(np.sum(newv != base))
    rec = {"experiment": "exp-097-pkc-motion-transition", "phase": 3,
           "manifest": man, "desk_acceptance": accepted,
           "phase1_chain": boundary["phase1_chain"], "repo": boundary["repo"],
           "start": boundary["start"],
           "started_utc": utc(), "changed_by_month": changed,
           "changed_rows": n_changed,
           "gate_min_changed": M.MIN_RANKING_CHANGED,
           "gate_pass": bool(n_changed >= M.MIN_RANKING_CHANGED)}
    if not rec["gate_pass"]:
        rec["candidate"] = None
        rec["seconds"] = round(time.time() - t0, 1)
        rec["finished_utc"] = utc()
        return rec
    src = uploaded_v22_frame()
    if len(src) != len(out):
        raise SystemExit("candidate: the uploaded file changed length")
    src_ids = M.check_ids(src["MVT_ID_mvt"].to_numpy(), "candidate source")
    if not np.array_equal(src_ids, ids):
        raise SystemExit("candidate: the uploaded file's id order changed")
    src["TAXITIME_SEC_mvt"] = newv
    path = Path(CANDIDATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    src.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    # the read-back facts are ENFORCED, not merely recorded (Sol round 1, blocking 5)
    got = back["TAXITIME_SEC_mvt"].to_numpy("float64")
    back_ids = M.check_ids(back["MVT_ID_mvt"].to_numpy(), "candidate read-back")
    if len(back) != len(src):
        raise SystemExit("candidate read-back: row count changed")
    if not np.array_equal(back_ids, ids):
        raise SystemExit("candidate read-back: movement id order changed")
    if not np.isfinite(got).all():
        raise SystemExit("candidate read-back: a prediction is not finite")
    if not np.array_equal(got[~hit], base[~hit]):
        raise SystemExit("candidate read-back: an unchanged row is not bit-identical")
    if int(np.sum(got != base)) != n_changed:
        raise SystemExit("candidate read-back: the changed-row count does not match")
    if got[hit].size and (got[hit].min() < M.TAXI_LO or got[hit].max() > M.TAXI_HI):
        raise SystemExit("candidate read-back: a served value is outside the taxi window")
    rec["candidate"] = {
        "path": str(path),
        "sha256": M.sha256(path),
        "rows": int(len(back)),
        "finite": True,
        "min": float(got.min()),
        "max": float(got.max()),
        "unchanged_bit_identical": int(np.sum(got[~hit] == base[~hit])),
        "unchanged_rows": int(np.sum(~hit)),
        "changed_rows": n_changed,
        "id_order_exact": True,
        "enforced": ["row count", "id order", "finiteness", "unchanged bit equality",
                     "changed-row count", "served values inside the taxi window"],
    }
    rec["seconds"] = round(time.time() - t0, 1)
    rec["finished_utc"] = utc()
    return rec


# ------------------------------------------------------------------------------- main

PHASES = {1: (phase1, "screen"), 2: (phase2, "score"), 3: (phase3, "candidate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--revision", default="",
                    help="a suffix such as r1: a repaired phase writes a NEW artifact "
                         "beside the old one and never overwrites a measurement")
    a = ap.parse_args()
    rev = ("-" + a.revision.lstrip("-")) if a.revision else ""
    fn, stem = PHASES[a.phase]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    globals()["RECEIPT"] = str(ANALYSIS / ("receipt-phase{n}" + rev + ".json"))
    out = ANALYSIS / f"{stem}{rev}.json"
    receipt = Path(RECEIPT.format(n=a.phase))
    sys.path.insert(0, str(ROOT / "experiments/exp-046-surfacefit/code"))
    import nooverwrite as NO                                           # noqa: PLC0415
    # there is no `--overwrite` and no real-source smoke mode (Sol round 1, blocking 6):
    # a measured artifact is never replaced, and a repaired run takes a new revision.
    NO.refuse_existing([out, receipt], False, flag="--revision")

    # A label-bearing phase passes every barrier BEFORE it claims its one read, so a direct
    # call — which carries no launcher handoff — refuses with the read still unspent.
    boundary = None
    if a.phase in (2, 3):
        import pkcmanifest as MAN                                      # noqa: PLC0415
        bound = [*MAN.CODE,
                 "experiments/exp-097-pkc-motion-transition/source-manifest.json"]
        boundary = preflight(a.phase, a.revision.lstrip("-"), bound_files=bound)
        start_path = ANALYSIS / f"start-phase{a.phase}{rev}.json"
        NO.refuse_existing([start_path], False, flag="--revision")

    claimed = claim(a.phase, rev)
    # From here the claim is spent, so every exit must leave a durable receipt.
    import pkcguard as G                                               # noqa: PLC0415
    try:
        if boundary is not None:
            start = dict(boundary["start"])
            # The claim's PID is read BACK from the file on disk and bound to this
            # process, so the start receipt cannot claim a run that some other process
            # claimed (assignment item 2).
            claim_pid = json.loads(claimed.read_text()).get("pid")
            if claim_pid != os.getpid():
                G.durable_failure(a.phase, rev,
                                  f"the claim names pid {claim_pid!r}, this process is "
                                  f"{os.getpid()}", analysis=ANALYSIS)
                raise SystemExit(f"phase {a.phase}: the claim names another process")
            start.update({"phase": a.phase, "revision": a.revision.lstrip("-") or None,
                          "claim": str(claimed.relative_to(ROOT)),
                          "claim_pid": claim_pid,
                          "phase1_chain": boundary["phase1_chain"],
                          "desk_acceptance": boundary["desk_acceptance"],
                          "repo": boundary["repo"], "started_utc": utc()})
            write_json(start_path, start)
            rec = fn(boundary)
        else:
            rec = fn()
    except BaseException as exc:                                       # noqa: BLE001
        G.durable_failure(a.phase, rev, f"{type(exc).__name__}: {exc}"[:400],
                          extra={"claim": str(claimed.relative_to(ROOT)),
                                 "result_written": False}, analysis=ANALYSIS)
        raise
    rec["claim"] = str(claimed.relative_to(ROOT))
    rec["revision"] = rev.lstrip("-") or None
    if boundary is not None:
        rec["start_receipt"] = {"path": str(start_path.relative_to(ROOT)),
                                "sha256": G.sha256(start_path)}
    try:
        NO.refuse_existing([out, receipt], False, flag="--revision")
        write_json(out, rec)
        write_json(receipt, rec)
    except BaseException as exc:                                       # noqa: BLE001
        G.durable_failure(a.phase, rev, f"result not written: {type(exc).__name__}: {exc}"[:400],
                          extra={"claim": str(claimed.relative_to(ROOT)),
                                 "result_written": False}, analysis=ANALYSIS)
        raise
    print(json.dumps({"phase": a.phase, "revision": rec["revision"], "wrote": str(out),
                      "decision": rec.get("decision"),
                      "support": rec.get("support_condition", {}).get("verdict"),
                      "gate_pass": rec.get("gate_pass"),
                      "seconds": rec.get("seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
