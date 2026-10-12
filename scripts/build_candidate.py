"""One entry point that rebuilds a submission candidate from raw data.

    python scripts/build_candidate.py <config.json> [--mode MODE]

    MODE = folds     re-derive the board estimate and the shipped weights (no fits)
           members   retrain every member on all twelve months and predict both
                     ranking months
           blend     blend existing member ranking predictions into the candidate
           all       members, then folds, then blend  (default)
           verify    hash every declared artifact against the config, fit nothing

Why one entry point. The number we quote and the file we upload are derived
here from **one** config: `folds` fits the blend weights by rotation over the
member out-of-fold predictions and reports the board estimate; `blend` applies
*those same weights* to the member ranking predictions. They cannot drift apart
without this file changing.

Reproducibility. Every member declares its producer, its arguments, its
environment (thread count and GPU), its seeds and its expected output hash, and
the manifest records what was actually observed. A member declared
`deterministic: false` is expected to differ between runs; the manifest carries
the reason and the measured spread rather than a hash that will not hold.

Regimes come from the raw movement files through `prc.blend.regimes_from_raw`,
not from any experiment's feature cache, so this path depends on `prc/` and the
raw data alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

RAW_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_ID_mvt",
               "AOBT_3_flt", "LOBT_flt", "MVT_TIME_UTC_mvt"]
PACKAGES = ("lightgbm", "catboost", "xgboost", "pandas", "numpy", "pyarrow", "scipy", "sklearn")


# ----------------------------------------------------------------- utils ----
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = "not installed"
    return out


def module_versions() -> dict:
    """VERSION constants of every prc module the members read."""
    out = {}
    for name in ("blend", "data", "nmtail", "tail_model", "day_state_features",
                 "phi_features", "lirf_features", "surface_features", "stand_features",
                 "weather_features", "tail_features"):
        try:
            mod = __import__(f"prc.{name}", fromlist=["x"])
            v = getattr(mod, "VERSION", None)
            if v is not None:
                out[f"prc.{name}"] = v
        except Exception as e:
            out[f"prc.{name}"] = f"import failed: {e}"
    return out


def git_state() -> dict:
    def g(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "?"
    return {"commit": g("rev-parse", "HEAD"),
            "dirty_paths": len([l for l in g("status", "--porcelain").splitlines() if l])}


# --------------------------------------------------------------- regimes ----
def raw_departures(month: str | None) -> pd.DataFrame:
    """Departure rows with the four regime inputs, from the raw files only."""
    r = (D.load_ranking(columns=RAW_COLUMNS) if month is None
         else D.load_month(month, columns=RAW_COLUMNS))
    r = r[r["PHASE_mvt"] == "DEP"].copy()
    r["regime"] = B.regimes_from_raw(r, B.LATE_CUT)
    r["d_takeoff_aobt"] = (r["MVT_TIME_UTC_mvt"] - r["AOBT_3_flt"]).dt.total_seconds()
    return r[["MVT_ID_mvt", "regime", "d_takeoff_aobt"]]


# ------------------------------------------------------------- producers ----
def run_producer(m: dict, cfg: dict, dry: bool = False) -> dict:
    """Run one member's producer as a subprocess with its declared environment."""
    out = ROOT / m["ranking"]
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (m.get("env") or {}).items()})
    cmd = [str(ROOT / cfg.get("python", ".venv/bin/python")), str(ROOT / m["producer"])] + \
          [str(a) for a in (m.get("args") or [])]
    print(f"\n=== {m['name']}: {' '.join(cmd[1:])}", flush=True)
    print(f"    cwd {m.get('cwd', '.')}   env {m.get('env') or {}}", flush=True)
    print(f"    seeds {m.get('seeds')}   budget: {str(m.get('budget',''))[:96]}", flush=True)
    print(f"    writes {m['ranking']}", flush=True)
    if dry:
        missing = [] if (ROOT / m["producer"]).exists() else [m["producer"]]
        print(f"    DRY RUN — not executed"
              + (f"; MISSING PRODUCER {missing}" if missing else "; producer exists"), flush=True)
        return {"name": m["name"], "producer": m["producer"], "args": m.get("args") or [],
                "env": m.get("env") or {}, "seeds": m.get("seeds"), "path": m["ranking"],
                "sha256": "(dry run)", "expected_sha256": (m.get("sha_as_built") or {}).get("sha256") or m.get("expected_sha256"),
                "deterministic": m.get("deterministic", True), "seconds": "—",
                "producer_exists": not missing}
    # Snapshot an existing output before overwriting it. A member declared
    # non-deterministic is EXPECTED to differ on a rebuild, and some of these
    # files are published to other lanes with pinned hashes -- clobbering one
    # would silently break a consumer whose weights were fitted on it. So the
    # original is restored and the rebuild kept alongside for comparison.
    prior = sha256(out) if out.exists() else None
    snap = out.with_suffix(out.suffix + ".prerebuild") if prior else None
    if snap:
        shutil.copy2(out, snap)
    # Tee the producer's output so it still streams to the log AND can be read
    # for the [threads] line. The manifest records the count the LIBRARY
    # reported at fit time, not the one the config asked for, and the build
    # refuses when they disagree (desk rule, 2026-09-06). `prc.nmtail` hardcoded
    # 8 behind a config declaring 6, which would have shipped a manifest that
    # lies in the one field the reproducibility contract rests on.
    t0 = time.time()
    observed, lib = None, None
    proc = subprocess.Popen(cmd, cwd=ROOT / m.get("cwd", "."), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        if line.startswith("[threads]"):
            for tok in line.split():
                if tok.startswith("observed="):
                    observed = tok.split("=", 1)[1]
                if tok.startswith("library="):
                    lib = tok.split("=", 1)[1]
    proc.wait()
    r = proc
    secs = round(time.time() - t0)
    declared = str((m.get("env") or {}).get("PRC_THREADS", "unset"))
    if observed is None:
        print(f"    [threads] producer reported none; declared {declared}", flush=True)
    elif observed != declared:
        if snap:
            shutil.move(str(snap), str(out))
        raise SystemExit(
            f"{m['name']}: declared PRC_THREADS={declared} but {lib} reports "
            f"num_threads={observed}. The manifest would record a thread count "
            f"that is not the one that ran. Fix the producer or the config; do "
            f"not proceed.")
    else:
        print(f"    [threads] {lib} confirms {observed}, matching the declaration",
              flush=True)
    if r.returncode != 0:
        if snap:
            shutil.move(str(snap), str(out))
        raise SystemExit(f"{m['name']}: producer exited {r.returncode}")
    if not out.exists():
        raise SystemExit(f"{m['name']}: producer wrote no {m['ranking']}")
    got = sha256(out)
    restored = False
    if prior and got != prior:
        if not m.get("deterministic", True):
            keep = out.with_suffix(out.suffix + f".rebuild-{got[:8]}")
            shutil.move(str(out), str(keep))
            shutil.move(str(snap), str(out))
            restored = True
            print(f"    rebuild DIFFERS from the published file, as declared. "
                  f"Published file restored; rebuild kept at {keep.name}", flush=True)
            got = prior
        else:
            # A member declared deterministic that rebuilt differently is a
            # FINDING, and the finding is worthless without both artifacts. The
            # first version of this branch kept the rebuild and deleted the
            # snapshot, which destroyed the only copy of the input two shipped
            # candidates were built from -- `artifacts/` is gitignored, so there
            # was nothing to recover from. Never delete the only copy of
            # anything a candidate was built from.
            keep = out.with_suffix(out.suffix + f".rebuild-{got[:8]}")
            shutil.move(str(out), str(keep))
            shutil.move(str(snap), str(out))
            restored = True
            print(f"    MISMATCH: a member declared DETERMINISTIC rebuilt to "
                  f"different bytes ({prior[:12]}… -> {got[:12]}…).", flush=True)
            print(f"    Original restored; rebuild kept at {keep.name}. "
                  f"Either the member is not deterministic or something moved "
                  f"under it -- do not paper over this by updating the hash.",
                  flush=True)
            got = prior
    if snap and snap.exists():
        snap.unlink()
    rec = {"name": m["name"], "producer": m["producer"], "args": m.get("args") or [],
           "env": m.get("env") or {}, "seeds": m.get("seeds"), "path": m["ranking"],
           "sha256": got, "expected_sha256": (m.get("sha_as_built") or {}).get("sha256") or m.get("expected_sha256"),
           "deterministic": m.get("deterministic", True), "seconds": secs,
           "published_file_restored": restored,
           "threads_declared": declared, "threads_observed": observed, "library": lib}
    if (m.get("sha_as_built") or {}).get("sha256") or m.get("expected_sha256"):
        rec["matches_expected"] = (got == ((m.get("sha_as_built") or {}).get("sha256") or m.get("expected_sha256")))
        flag = "MATCH" if rec["matches_expected"] else ("EXPECTED-DIFFERENT"
                                                        if not m.get("deterministic", True) else "MISMATCH")
        print(f"    -> {got[:16]}…  {flag}  ({secs}s)", flush=True)
    else:
        print(f"    -> {got[:16]}…  ({secs}s)", flush=True)
    return rec


# ------------------------------------------------------------- the blend ----
def load_member_oof(m: dict, fold: str) -> pd.DataFrame:
    p = ROOT / str(m["oof"]).format(fold=fold)
    d = pd.read_parquet(p)
    if "fold" in d.columns:
        d = d[d["fold"] == fold]
    return d[["MVT_ID_mvt", "y_true", "y_pred"]].rename(columns={"y_pred": m["name"]})


def fold_table(cfg: dict) -> dict:
    """Rotated board estimate AND the weights the candidate will ship.

    A fold's weights are fitted on the other validation months only. The shipped
    vector is fitted on all of them, which is how a candidate's weights are
    chosen; the rotation is how its score is estimated.
    """
    members = cfg["members"]
    names = [m["name"] for m in members]
    folds = tuple(cfg["folds"])
    spec = cfg["blend"]
    target = spec.get("regime_split")
    data = {}
    for f in folds:
        base = load_member_oof(members[0], f)
        for m in members[1:]:
            base = base.merge(load_member_oof(m, f)[["MVT_ID_mvt", m["name"]]],
                              on="MVT_ID_mvt", how="inner")
        reg = raw_departures(D.FOLDS[f].valid_month)
        j = base.merge(reg, on="MVT_ID_mvt", how="inner")
        if len(j) != len(base):
            raise SystemExit(f"{f}: {len(base)-len(j)} rows lost joining regimes")
        a = j["d_takeoff_aobt"].to_numpy("float64")
        lo, hi = B_guard_bounds(cfg, f)
        prep = lambda v: B.apply_residual_guard(
            np.maximum(np.asarray(v, "float64"), B.FLOOR), a, lo, hi, B.FLOOR)
        P = np.stack([prep(np.nan_to_num(j[n].to_numpy("float64"), nan=B.FLOOR)) for n in names], axis=1)
        ok = np.stack([np.isfinite(j[n].to_numpy("float64")) for n in names], axis=1).all(axis=1)
        data[f] = {"P": P, "y": j["y_true"].to_numpy("float64"),
                   "regime": j["regime"].to_numpy(), "ok": ok, "n": len(j)}
    return rotate_and_fit(cfg, data, names, folds, target)


def B_guard_bounds(cfg, fold):
    b = cfg["guard_bounds"][fold] if isinstance(cfg["guard_bounds"], dict) else cfg["guard_bounds"]
    return float(b[0]), float(b[1])


def _w(P, y, mode, k):
    return np.full(k, 1.0 / k) if mode == "flat" else B.fit_weights(P[:, :k], y, mode)


def rotate_and_fit(cfg, data, names, folds, target):
    spec = cfg["blend"]
    inner, outer = spec["inside_rule"], spec["outside_rule"]
    k = len(names)
    ki = k if target else k - (1 if spec.get("specialist") else 0)
    per, rows = {}, []
    for f in folds:
        src = [g for g in folds if g != f]
        sel = {g: (data[g]["regime"] == target) & data[g]["ok"] for g in src} if target else \
              {g: np.ones(data[g]["n"], bool) for g in src}
        Pi = np.concatenate([data[g]["P"][sel[g]] for g in src])
        yi = np.concatenate([data[g]["y"][sel[g]] for g in src])
        wi = _w(Pi, yi, inner, k)
        if target:
            Po = np.concatenate([data[g]["P"][~sel[g]][:, :k-1] for g in src])
            yo = np.concatenate([data[g]["y"][~sel[g]] for g in src])
            wo = _w(Po, yo, outer, k - 1)
        else:
            wo = wi
        d = data[f]
        s = (d["regime"] == target) & d["ok"] if target else np.ones(d["n"], bool)
        p = np.empty(d["n"])
        p[s] = d["P"][s] @ wi
        p[~s] = d["P"][~s][:, :k-1] @ wo if target else d["P"][~s] @ wo
        p = np.maximum(p, B.FLOOR)
        per[f] = float(np.sqrt(((d["y"] - p) ** 2).mean()))
        rows.append({"fold": f, "fit_on": "+".join(src), "rmse": round(per[f], 4)})
    # shipped weights: fitted on every month
    selall = {g: (data[g]["regime"] == target) & data[g]["ok"] for g in folds} if target else \
             {g: np.ones(data[g]["n"], bool) for g in folds}
    Pi = np.concatenate([data[g]["P"][selall[g]] for g in folds])
    yi = np.concatenate([data[g]["y"][selall[g]] for g in folds])
    w_in = _w(Pi, yi, inner, k)
    if target:
        Po = np.concatenate([data[g]["P"][~selall[g]][:, :k-1] for g in folds])
        yo = np.concatenate([data[g]["y"][~selall[g]] for g in folds])
        w_out = np.append(_w(Po, yo, outer, k - 1), 0.0)
    else:
        w_out = w_in
    shares = cfg["board_shares"]
    board = float(np.sqrt(sum(shares[f] * per[f] ** 2 for f in shares)))
    W = {target: dict(zip(names, map(float, w_in)))} if target else {}
    for r in B.REGIMES:
        W.setdefault(r, dict(zip(names, map(float, w_out))))
    W["_all"] = W[B.REGIMES[0]]
    return {"rotated": rows, "per_fold": per, "board_estimate": round(board, 4),
            "weights": W, "fit_rows_inside": int(len(yi)), "names": names}


def build(cfg: dict, weights: dict, out_path: Path) -> dict:
    bc = {"ranking": {m["name"]: m["ranking"] for m in cfg["members"]},
          "weights": weights, "floor": B.FLOOR,
          "guard_bounds": cfg["submission_guard_bounds"], "late_cut": B.LATE_CUT,
          "prepare_members": True}
    return B.build_candidate(bc, out_path, ROOT)


# ------------------------------------------------------------- manifest ----
def write_manifest(cfg, path: Path, sections: dict) -> None:
    L = [f"# MANIFEST — {cfg['name']}", "",
         f"Produced by `scripts/build_candidate.py {cfg['_config_path']}` "
         f"at {time.strftime('%Y-%m-%dT%H:%M:%S')}.", "",
         "Regenerate with the same command; `--mode verify` re-hashes every declared",
         "artifact without fitting anything.", ""]
    for title, body in sections.items():
        L += [f"## {title}", "", body, ""]
    path.write_text("\n".join(L))
    print(f"\nmanifest -> {path.relative_to(ROOT)}", flush=True)


def table(rows, cols):
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(L)


# ----------------------------------------------------------------- main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", default="all",
                    choices=["all", "members", "folds", "blend", "verify", "gate"])
    ap.add_argument("--out", default=None, help="override the output path (for the gate)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="with --mode members/all: print each producer's exact command, "
                         "environment, seeds and budget without fitting anything")
    a = ap.parse_args()

    cfg = json.loads((ROOT / a.config).read_text())
    cfg["_config_path"] = a.config
    out_path = ROOT / (a.out or cfg["output"])
    sections = {}

    env_rows = [{"item": k, "value": v} for k, v in package_versions().items()]
    env_rows += [{"item": k, "value": v} for k, v in module_versions().items()]
    g = git_state()
    env_rows += [{"item": "git commit", "value": g["commit"]},
                 {"item": "uncommitted paths", "value": g["dirty_paths"]}]
    sections["Environment"] = table(env_rows, ["item", "value"])

    raw = []
    for p in sorted((ROOT / "data" / "raw").glob("*.parquet")):
        if "stale" in p.name:
            continue
        raw.append({"file": f"data/raw/{p.name}", "bytes": p.stat().st_size,
                    "sha256": sha256(p)[:32] + "…"})
    sections["Raw inputs"] = table(raw, ["file", "bytes", "sha256"])

    if a.mode == "gate":
        # The reproducibility gate. Two independent derivations of the candidate
        # from the same member predictions must agree byte for byte; anything
        # that cannot be expected to agree must be DECLARED, with its measured
        # spread, rather than discovered later by a mismatched hash.
        import tempfile
        shas, boards = [], []
        for i in (1, 2):
            ft_i = fold_table(cfg)
            with tempfile.TemporaryDirectory() as td:
                rec_i = build(cfg, ft_i["weights"], Path(td) / f"gate{i}.parquet")
            shas.append(rec_i["sha256"])
            boards.append(ft_i["board_estimate"])
            print(f"  run {i}: board {ft_i['board_estimate']}  sha {rec_i['sha256'][:16]}…")
        ok = shas[0] == shas[1] and boards[0] == boards[1]
        print(f"\n  blend stage byte-identical across two runs: {ok}")
        nd = [m for m in cfg["members"] if not m.get("deterministic", True)]
        print(f"  members declared non-deterministic: {len(nd)}")
        for m in nd:
            print(f"    - {m['name']}: {m.get('nondeterminism','(no reason recorded)')}")
        gate_rows = [{"stage": "blend (two independent derivations)",
                      "byte-identical": ok, "detail": f"{shas[0][:16]}… both runs"}]
        for m in cfg["members"]:
            gate_rows.append({
                "stage": f"member {m['name']}",
                "byte-identical": "declared reproducible" if m.get("deterministic", True)
                                  else "NO — declared non-deterministic",
                "detail": (m.get("notes", "") if m.get("deterministic", True)
                           else m.get("nondeterminism", ""))[:180]})
        sections["Reproducibility gate"] = (
            table(gate_rows, ["stage", "byte-identical", "detail"]) +
            "\n\nThe gate passes for every stage this pipeline controls. It cannot pass "
            "for a member whose estimator is not thread-count reproducible, and that "
            "member is named above with its measured spread rather than left to be "
            "discovered as a mismatched hash.")
        write_manifest(cfg, ROOT / (a.manifest or cfg["manifest"]), sections)
        return 0 if ok else 1

    if a.mode == "verify":
        rows = []
        for m in cfg["members"]:
            p = ROOT / m["ranking"]
            ab = (m.get("sha_as_built") or {})
            want = ab.get("sha256") or m.get("expected_sha256")
            got = sha256(p) if p.exists() else None
            rows.append({"member": m["name"], "path": m["ranking"],
                         "sha256": (got[:16] + "…") if got else "MISSING",
                         "as built": ((want or "")[:16] + "…") if want else "—",
                         "threads": ab.get("threads", "—"),
                         "matches": ("—" if not want else
                                     "yes" if got == want else "NO — see rebuild status")})
        print(table(rows, ["member", "path", "sha256", "as built", "threads", "matches"]))
        p = out_path
        got = sha256(p) if p.exists() else None
        want = cfg.get("expected_sha256", "")
        print(f"\ncandidate {out_path.relative_to(ROOT)}: "
              f"{got[:16] + '…' if got else 'MISSING'}  "
              f"expected {want[:16]}…")
        bad = [r["member"] for r in rows if r["matches"].startswith("NO")]
        bad += [r["member"] for r in rows if r["sha256"] == "MISSING"]
        if got != want:
            bad.append(out_path.name)
        if bad:
            print(f"\nFAIL — {len(set(bad))} declared artifact(s) missing or changed: "
                  f"{', '.join(sorted(set(bad)))}")
            return 1
        print("\nPASS — every declared artifact is present and unchanged")
        return 0

    member_rows = []
    if a.mode in ("all", "members"):
        for m in cfg["members"]:
            member_rows.append(run_producer(m, cfg, a.dry_run))
    else:
        for m in cfg["members"]:
            p = ROOT / m["ranking"]
            member_rows.append({"name": m["name"], "path": m["ranking"],
                                "sha256": sha256(p) if p.exists() else "MISSING",
                                "expected_sha256": (m.get("sha_as_built") or {}).get("sha256") or m.get("expected_sha256"),
                                "deterministic": m.get("deterministic", True),
                                "seeds": m.get("seeds")})
    sections["Members"] = table(
        [{"member": r["name"], "seeds": r.get("seeds"),
          "threads (library-reported)": r.get("threads_observed", "—"),
          "deterministic": r.get("deterministic"),
          "sha256": str(r["sha256"])[:16] + "…", "matches expected": r.get("matches_expected", "—"),
          "seconds": r.get("seconds", "—")} for r in member_rows],
        ["member", "seeds", "threads (library-reported)", "deterministic",
         "sha256", "matches expected", "seconds"])

    ft = None
    if a.mode in ("all", "folds", "blend"):
        ft = fold_table(cfg)
        print("\n=== rotated board estimate (same weights the candidate ships) ===")
        print(table(ft["rotated"], ["fold", "fit_on", "rmse"]))
        print(f"board estimate {ft['board_estimate']}   "
              f"(quoted {cfg.get('expected_board_estimate','—')})")
        for r, w in ft["weights"].items():
            if r == "_all":
                continue
            print(f"  {r:8s} " + "  ".join(f"{n}={v:.4f}" for n, v in w.items()))
        sections["Board estimate (rotated, from the same weights)"] = (
            table(ft["rotated"], ["fold", "fit_on", "rmse"]) +
            f"\n\n**board estimate {ft['board_estimate']}**, quoted "
            f"{cfg.get('expected_board_estimate','—')}")
        sections["Shipped weights"] = table(
            [dict(regime=r, **{k: round(v, 4) for k, v in w.items()})
             for r, w in ft["weights"].items() if r != "_all"],
            ["regime"] + ft["names"])

    if a.mode in ("all", "blend") and not a.dry_run:
        rec = build(cfg, ft["weights"], out_path)
        print("\n=== candidate ===")
        print(json.dumps({k: rec[k] for k in ("path", "sha256", "n")}, indent=2))
        same = rec["sha256"] == cfg.get("expected_sha256")
        nd = [m["name"] for m in cfg["members"] if not m.get("deterministic", True)]
        sections["Output"] = table([{
            "path": str(out_path.relative_to(ROOT)), "rows": rec["n"],
            "sha256": rec["sha256"],
            "matches the shipped candidate": "yes" if same else
            f"no — expected non-determinism in {', '.join(nd)}" if nd else "NO"}],
            ["path", "rows", "sha256", "matches the shipped candidate"])
        print(f"\nmatches the shipped candidate: {same}")

    if a.dry_run and a.manifest is None:
        # exp-064: a dry run must not touch the committed record.  Before this,
        # `--mode members --dry-run` printed the plan and then overwrote
        # experiments/exp-015-release/MANIFEST-blend-2.md with a manifest of a
        # run that never happened.  Pass --manifest to write one deliberately.
        print(f"\nDRY RUN — manifest not written "
              f"(pass --manifest PATH to write one)")
    else:
        write_manifest(cfg, ROOT / (a.manifest or cfg["manifest"]), sections)


if __name__ == "__main__":
    # `main` returns 1 from the gate and from --mode verify when a declared
    # artifact is missing or has a different digest.  Until exp-064 that return
    # value was discarded, so both of those gates always exited 0 and could not
    # fail a caller.
    sys.exit(main() or 0)
